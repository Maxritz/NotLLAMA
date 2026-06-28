#include "rdna4_engine.hpp"
#include "rdna4_types.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include "rdna4_kv_cache.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>

namespace rdna4 {

static uint64_t findTensorAddr(const ModelDesc& model, const std::string& name) {
    for (const auto& t : model.tensors) {
        if (t.name == name) return t.gpuAddress;
    }
    return 0;
}

static const TensorDesc* findTensor(const ModelDesc& model, const std::string& name) {
    for (const auto& t : model.tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// Compute the f16 output size for a quantized tensor (0 if already F16/F32)
static size_t getF16OutputSize(const ModelDesc& model, const std::string& name) {
    const TensorDesc* t = findTensor(model, name);
    if (!t || t->format == QuantFormat::F16 || t->format == QuantFormat::F32) return 0;
    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;
    return nElements * sizeof(float);
}

// Dequantize a weight tensor to float32 at a specific offset in the staging buffer.
// Dispatches in chunks to avoid GPU hang from excessive workgroup counts.
static uint64_t dequantWeight(Scheduler* sched, PipelineBuilder* pipes,
                               uint64_t dequantBufAddr, size_t dequantBufSize,
                               const ModelDesc& model, const std::string& name,
                               size_t outOffset = 0, bool sync = true) {
    const TensorDesc* t = findTensor(model, name);
    if (!t) return 0;
    if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) return t->gpuAddress;

    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;

    size_t outBytes = (size_t)nElements * sizeof(float);
    if (outOffset + outBytes > dequantBufSize) return 0;

    const uint32_t MAX_WG_PER_DISPATCH = 256 * 1024;
    uint32_t totalWorkgroups = (nElements + 63) / 64;
    if (totalWorkgroups == 0) totalWorkgroups = 1;

    uint32_t elementsPerChunk = MAX_WG_PER_DISPATCH * 64;
    uint32_t offset = 0;

    while (offset < nElements) {
        uint32_t chunkSize = std::min(elementsPerChunk, nElements - offset);
        uint32_t chunkWg = (chunkSize + 63) / 64;

        DequantizePushConstants pc = {};
        pc.addrQuant = t->gpuAddress;
        pc.addrOut = dequantBufAddr + outOffset + (uint64_t)offset * sizeof(float);
        pc.nElements = chunkSize;
        pc.quantFormat = static_cast<uint32_t>(t->format);
        pc.totalThreads = chunkWg * 64;
        pc.elementOffset = offset;

        sched->dispatch(pipes->getPipeline("dequantize"), pipes->getLayout("dequantize"),
                        &pc, sizeof(pc), chunkWg, 1, 1);

        offset += chunkSize;
    }
    if (sync) sched->syncAllThrottled();

    return dequantBufAddr + outOffset;
}

// ============================================================================
// dequantWeightInBatch: records dequant dispatches into the ACTIVE batch
// command buffer via dispatchInBatch(). NO separate submits. NO syncs.
// Must only be called between beginBatch() and endBatch().
// Returns the GPU address where the dequantized float32 weights will be.
// ============================================================================
static uint64_t dequantWeightInBatch(Scheduler* sched, PipelineBuilder* pipes,
    uint64_t dequantBufAddr, size_t dequantBufSize,
    const ModelDesc& model, const std::string& name,
    size_t outOffset = 0) {
    const TensorDesc* t = findTensor(model, name);
    if (!t) return 0;
    if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) return t->gpuAddress;

    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;

    size_t outBytes = (size_t)nElements * sizeof(float);
    if (outOffset + outBytes > dequantBufSize) return 0;

    const uint32_t MAX_WG_PER_DISPATCH = 256 * 1024;
    uint32_t elementsPerChunk = MAX_WG_PER_DISPATCH * 64;
    uint32_t offset = 0;

    while (offset < nElements) {
        uint32_t chunkSize = std::min(elementsPerChunk, nElements - offset);
        uint32_t chunkWg = (chunkSize + 63) / 64;

        DequantizePushConstants pc = {};
        pc.addrQuant = t->gpuAddress;
        pc.addrOut = dequantBufAddr + outOffset + (uint64_t)offset * sizeof(float);
        pc.nElements = chunkSize;
        pc.quantFormat = static_cast<uint32_t>(t->format);
        pc.totalThreads = chunkWg * 64;
        pc.elementOffset = offset;

        sched->dispatchInBatch(pipes->getPipeline("dequantize"), pipes->getLayout("dequantize"),
            &pc, sizeof(pc), chunkWg, 1, 1);
        offset += chunkSize;
    }
    return dequantBufAddr + outOffset;
}

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

bool InferenceEngine::initDequantBuffer() {
    // The staging buffer must hold every dequantized weight that is live at the
    // same time for a single dispatch. The worst case is the MLP, which needs
    // ffn_up + ffn_gate + ffn_down simultaneously. The embedding table uses a
    // separate persistent cache and is excluded here.
    // The staging buffer must hold ALL dequantized weights for one full layer
    // simultaneously: attn_norm + Q + K + V + O + ffn_norm + up + gate + down.
    // This enables batched dequant (one sync) before any compute dispatches.
    size_t maxAttnSet = 0;
    size_t maxFfnSet = 0;
    for (uint32_t layer = 0; layer < model->blockCount; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);
        size_t attnTotal = 0;
        attnTotal += getF16OutputSize(*model, prefix + ".attn_norm.weight");
        attnTotal += getF16OutputSize(*model, prefix + ".attn_q.weight");
        attnTotal += getF16OutputSize(*model, prefix + ".attn_k.weight");
        attnTotal += getF16OutputSize(*model, prefix + ".attn_v.weight");
        attnTotal += getF16OutputSize(*model, prefix + ".attn_output.weight");
        if (attnTotal > maxAttnSet) maxAttnSet = attnTotal;

        size_t ffnTotal = 0;
        ffnTotal += getF16OutputSize(*model, prefix + ".ffn_norm.weight");
        ffnTotal += getF16OutputSize(*model, prefix + ".ffn_up.weight");
        ffnTotal += getF16OutputSize(*model, prefix + ".ffn_gate.weight");
        ffnTotal += getF16OutputSize(*model, prefix + ".ffn_down.weight");
        if (ffnTotal > maxFfnSet) maxFfnSet = ffnTotal;
    }
    size_t maxLayerSet = std::max(maxAttnSet, maxFfnSet);

    size_t maxSize = maxLayerSet;
    if (maxSize == 0) maxSize = 16 * 1024 * 1024;
    // Also account for weight-tied LM head (output.weight -> token_embd.weight)
    size_t embedSize = getF16OutputSize(*model, "token_embd.weight");
    if (embedSize > maxSize) maxSize = embedSize;

    if (maxSize > 640 * 1024 * 1024) maxSize = 640 * 1024 * 1024;  // Hard cap at 640 MB

    // Cap at 80% of total GPU VRAM — query memProps once
    {
        VkPhysicalDeviceMemoryProperties vramProps;
        vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &vramProps);
        size_t vramBytes = 0;
        for (uint32_t i = 0; i < vramProps.memoryHeapCount; ++i) {
            if (vramProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vramBytes += vramProps.memoryHeaps[i].size;
        }
        size_t vram80 = vramBytes * 8 / 10;
        if (maxSize > vram80) maxSize = vram80;
        fprintf(stderr, "[dequant] VRAM=%zu MB, 80%% cap=%zu MB, buffer=%zu MB\n",
                vramBytes / 1024 / 1024, vram80 / 1024 / 1024, maxSize / 1024 / 1024);
    }

    dequantCapacity = maxSize;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = maxSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(ctx->device, &bufInfo, nullptr, &dequantBuffer);
    if (r != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, dequantBuffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    // Prefer HOST_VISIBLE + DEVICE_LOCAL (resizable BAR) for debug readback
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = &flagsInfo;

    r = vkAllocateMemory(ctx->device, &allocInfo, nullptr, &dequantMemory);
    if (r != VK_SUCCESS) return false;

    vkBindBufferMemory(ctx->device, dequantBuffer, dequantMemory, 0);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = dequantBuffer;
    dequantAddr = vkGetBufferDeviceAddress(ctx->device, &addrInfo);

    dequantBufEnd = dequantAddr + maxSize;
    fprintf(stderr, "[dequant] staging buffer: %zu MB @ 0x%llx..0x%llx memType=%u props=0x%x\n",
            maxSize / 1024 / 1024, (unsigned long long)dequantAddr,
            (unsigned long long)dequantBufEnd,
            memTypeIndex, memProps.memoryTypes[memTypeIndex].propertyFlags);
    return true;
}

void InferenceEngine::cleanupDequantBuffer() {
    if (dequantBuffer) vkDestroyBuffer(ctx->device, dequantBuffer, nullptr);
    if (dequantMemory) vkFreeMemory(ctx->device, dequantMemory, nullptr);
    dequantBuffer = VK_NULL_HANDLE;
    dequantMemory = VK_NULL_HANDLE;
    dequantAddr = 0;
    dequantBufEnd = 0;
}

bool InferenceEngine::initEmbedCache() {
    // TEMP: Skip embed cache to avoid hang on large models during benchmarking
    embedCacheReady = false;
    return true;

    // Find token_embd tensor and allocate persistent cache for dequantized embedding
    const TensorDesc* t = findTensor(*model, "token_embd.weight");
    if (!t) return false;
    if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) {
        // Already float — no cache needed
        embedCacheReady = false;
        return true;
    }

    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;
    embedCacheSize = (size_t)nElements * sizeof(float);

    fprintf(stderr, "[embed-cache] allocating %zu MB for %u elements\n",
            embedCacheSize / 1024 / 1024, nElements);

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = embedCacheSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(ctx->device, &bufInfo, nullptr, &embedCacheBuffer);
    if (r != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, embedCacheBuffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProps);

    // Prefer DEVICE_LOCAL for performance
    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) return false;

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = &flagsInfo;

    r = vkAllocateMemory(ctx->device, &allocInfo, nullptr, &embedCacheMemory);
    if (r != VK_SUCCESS) return false;

    vkBindBufferMemory(ctx->device, embedCacheBuffer, embedCacheMemory, 0);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = embedCacheBuffer;
    embedCacheAddr = vkGetBufferDeviceAddress(ctx->device, &addrInfo);

    // Dequantize embedding table once directly into the persistent embed cache.
    // Batched: all chunks recorded into ONE command buffer, submitted once,
    // synced once with syncAll() (no throttle — this is one-time init).
    fprintf(stderr, "[embed-cache] dequantizing embedding table (%u elements, %u total wg)...\n",
            nElements, (nElements + 31) / 32);

    auto embedStart = std::chrono::steady_clock::now();

    // Reduced chunk size: 64K workgroups × 32 threads = 2M elements/chunk
    // Smaller dispatches are safer against driver TDR and scheduling hiccups.
    const uint32_t MAX_WG = 64 * 1024;
    uint32_t elemPerChunk = MAX_WG * 32;
    uint32_t offset = 0;

    scheduler->beginBatch(0);
    uint32_t chunkCount = 0;
    while (offset < nElements) {
        uint32_t chunk = std::min(elemPerChunk, nElements - offset);
        uint32_t chunkWg = (chunk + 63) / 64;

        DequantizePushConstants pc = {};
        pc.addrQuant = t->gpuAddress;
        pc.addrOut = embedCacheAddr + (uint64_t)offset * sizeof(float);
        pc.nElements = chunk;
        pc.quantFormat = static_cast<uint32_t>(t->format);
        pc.totalThreads = chunkWg * 64;
        pc.elementOffset = offset;

        scheduler->dispatchInBatch(pipelines->getPipeline("dequantize"), pipelines->getLayout("dequantize"),
                                   &pc, sizeof(pc), chunkWg, 1, 1);
        // Barrier between chunks to ensure ordering (non-overlapping writes,
        // but barrier is cheap inside one command buffer).
        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        offset += chunk;
        chunkCount++;
    }
    scheduler->endBatch(VK_NULL_HANDLE);
    // Use syncAll() instead of syncAllThrottled() — no sleep overhead for one-time init.
    scheduler->syncAll();

    embedCacheReady = true;
    auto embedEnd = std::chrono::steady_clock::now();
    auto embedMs = std::chrono::duration_cast<std::chrono::milliseconds>(embedEnd - embedStart).count();
    fprintf(stderr, "[embed-cache] ready @ 0x%llx (%zu MB) in %lld ms (%u chunks, batch-submitted)\n",
            (unsigned long long)embedCacheAddr, embedCacheSize / 1024 / 1024,
            (long long)embedMs, chunkCount);
    return true;
}

void InferenceEngine::cleanupEmbedCache() {
    if (embedCacheBuffer) vkDestroyBuffer(ctx->device, embedCacheBuffer, nullptr);
    if (embedCacheMemory) vkFreeMemory(ctx->device, embedCacheMemory, nullptr);
    embedCacheBuffer = VK_NULL_HANDLE;
    embedCacheMemory = VK_NULL_HANDLE;
    embedCacheAddr = 0;
    embedCacheReady = false;
}

// ============================================================================
// Helper: create a VkBuffer with buffer device address, return its address.
// ============================================================================
static bool createBufferBDA(VkDevice device, VkPhysicalDevice physDev,
                            VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr,
                            size_t size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buf, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    uint32_t memTypeIdx = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIdx = i;
            break;
        }
    }
    if (memTypeIdx == UINT32_MAX) return false;

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIdx;
    allocInfo.pNext = &flagsInfo;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &mem) != VK_SUCCESS) return false;

    vkBindBufferMemory(device, buf, mem, 0);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buf;
    addr = vkGetBufferDeviceAddress(device, &addrInfo);
    return true;
}

// Helper: create a HOST_VISIBLE + HOST_COHERENT buffer with mapped pointer.
static bool createBufferHostVisible(VkDevice device, VkPhysicalDevice physDev,
                                    VkBuffer& buf, VkDeviceMemory& mem,
                                    void** mapped, VkDeviceAddress& addr,
                                    size_t size, VkBufferUsageFlags extraUsage = 0) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extraUsage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buf, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    uint32_t memTypeIdx = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memTypeIdx = i;
            break;
        }
    }
    if (memTypeIdx == UINT32_MAX) return false;

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIdx;
    allocInfo.pNext = &flagsInfo;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &mem) != VK_SUCCESS) return false;

    vkBindBufferMemory(device, buf, mem, 0);
    vkMapMemory(device, mem, 0, size, 0, mapped);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buf;
    addr = vkGetBufferDeviceAddress(device, &addrInfo);
    return true;
}

// ============================================================================
// initWeightBuffer — dequant ALL weights to a persistent float32 buffer.
// Called once at startup. The buffer lives for the lifetime of the engine.
// ============================================================================
bool InferenceEngine::initWeightBuffer() {
    // PHILOSOPHY VIOLATION: This function pre-dequantizes ALL weights to float32.
    // This is WRONG. Weights must stay quantized. GPU dequantizes on-demand per layer.
    // DISABLED pending architecture review (GLM Laguna XS.2).
    fprintf(stderr, "[weight-buffer] SKIPPED: pre-dequantization violates project philosophy.\n");
    return true;
}

void InferenceEngine::cleanupWeightBuffer() {
    if (weightBuffer) vkDestroyBuffer(ctx->device, weightBuffer, nullptr);
    if (weightMemory) vkFreeMemory(ctx->device, weightMemory, nullptr);
    weightBuffer = VK_NULL_HANDLE;
    weightMemory = VK_NULL_HANDLE;
    weightBufferAddr = 0;
    weightBufferSize = 0;
}

// ============================================================================
// initLayerParams — build LayerParams array in GPU-visible buffer.
// ============================================================================
bool InferenceEngine::initLayerParams() {
    size_t bufSize = model->blockCount * sizeof(LayerParams);
    if (!createBufferBDA(ctx->device, ctx->physicalDevice,
                         layerParamsBuffer, layerParamsMemory, layerParamsAddr,
                         bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    // Map and write LayerParams for each layer
    void* mapped = nullptr;
    VkResult r = vkMapMemory(ctx->device, layerParamsMemory, 0, bufSize, 0, &mapped);
    if (r != VK_SUCCESS || !mapped) return false;

    LayerParams* lp = reinterpret_cast<LayerParams*>(mapped);

    // Walk the weight buffer to find offsets for each weight
    // We dequantized in the same order as initWeightBuffer:
    //   per layer: attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_output, ffn_up, ffn_gate, ffn_down
    //   then: output_norm, token_embd
    // We need to track the running offset to build addresses.
    size_t writeOffset = 0;
    auto getWeightAddr = [&](uint32_t layer, const std::string& suffix) -> uint64_t {
        const TensorDesc* t = findTensor(*model, "blk." + std::to_string(layer) + suffix);
        if (!t) return 0;
        uint32_t nElem = 1;
        for (auto d : t->shape) nElem *= d;

        // If F16/F32, point to original buffer (no dequant copy in weight buffer)
        if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) {
            return t->gpuAddress;
        }

        // Otherwise, find the offset in weight buffer by scanning
        // (We dequantized in order, so the offset matches our writeOffset tracking)
        // This is computed during the scan below — we pre-compute a lookup table
        return 0;  // placeholder, computed below
    };

    // Build a lookup table of weight addresses in the weight buffer
    // by re-walking the dequant order
    std::unordered_map<std::string, uint64_t> weightAddrs;
    size_t off = 0;
    for (uint32_t layer = 0; layer < model->blockCount; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);
        for (const std::string& suffix : {".attn_norm.weight", ".ffn_norm.weight",
                                           ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
                                           ".attn_output.weight",
                                           ".ffn_up.weight", ".ffn_gate.weight", ".ffn_down.weight"}) {
            std::string fullName = prefix + suffix;
            const TensorDesc* t = findTensor(*model, fullName);
            if (!t) continue;
            uint32_t nElem = 1;
            for (auto d : t->shape) nElem *= d;
            size_t outBytes = (size_t)nElem * sizeof(float);

            if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) {
                weightAddrs[fullName] = t->gpuAddress;
            } else {
                weightAddrs[fullName] = weightBufferAddr + off;
            }
            off += outBytes;
        }
    }
    // output_norm
    {
        const TensorDesc* t = findTensor(*model, "output_norm.weight");
        if (!t) t = findTensor(*model, "norm.weight");
        if (t) {
            uint32_t nElem = 1;
            for (auto d : t->shape) nElem *= d;
            if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) {
                weightAddrs["output_norm.weight"] = t->gpuAddress;
            } else {
                weightAddrs["output_norm.weight"] = weightBufferAddr + off;
            }
            off += (size_t)nElem * sizeof(float);
        }
    }
    // token_embd
    {
        const TensorDesc* t = findTensor(*model, "token_embd.weight");
        if (t) {
            uint32_t nElem = 1;
            for (auto d : t->shape) nElem *= d;
            if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) {
                weightAddrs["token_embd.weight"] = t->gpuAddress;
            } else {
                weightAddrs["token_embd.weight"] = weightBufferAddr + off;
            }
            off += (size_t)nElem * sizeof(float);
        }
    }

    // Now fill LayerParams
    for (uint32_t layer = 0; layer < model->blockCount; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);
        lp[layer].addrAttnNorm = weightAddrs[prefix + ".attn_norm.weight"];
        lp[layer].addrQ        = weightAddrs[prefix + ".attn_q.weight"];
        lp[layer].addrK        = weightAddrs[prefix + ".attn_k.weight"];
        lp[layer].addrV        = weightAddrs[prefix + ".attn_v.weight"];
        lp[layer].addrAttnOut  = weightAddrs[prefix + ".attn_output.weight"];
        lp[layer].addrFfnNorm  = weightAddrs[prefix + ".ffn_norm.weight"];
        lp[layer].addrFfnUp    = weightAddrs[prefix + ".ffn_up.weight"];
        lp[layer].addrFfnGate  = weightAddrs[prefix + ".ffn_gate.weight"];
        lp[layer].addrFfnDown  = weightAddrs[prefix + ".ffn_down.weight"];
        lp[layer].addrKCache   = kvCache->getKBufferAddress(layer);
        lp[layer].addrVCache   = kvCache->getVBufferAddress(layer);
    }

    vkUnmapMemory(ctx->device, layerParamsMemory);
    fprintf(stderr, "[layer-params] %u layers written @ 0x%llx\n",
            model->blockCount, (unsigned long long)layerParamsAddr);
    return true;
}

// ============================================================================
// initKernelEntryBuffers — mailbox, output token, scratch for kernel_entry.comp
// ============================================================================
bool InferenceEngine::initKernelEntryBuffers() {
    // Mailbox: 6 × uint32 = 24 bytes
    if (!createBufferBDA(ctx->device, ctx->physicalDevice,
                         mailboxBuffer, mailboxMemory, mailboxAddr,
                         64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    // Output token: 1 × uint32 = 4 bytes
    if (!createBufferBDA(ctx->device, ctx->physicalDevice,
                         outputTokenBuffer, outputTokenMemory, outputTokenAddr,
                         16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    // Hidden state: maxContext × dim floats (persistent activation buffer)
    uint32_t dim = model->embeddingLength;
    uint32_t maxCtx = std::min(model->contextLength, 2048u);
    size_t hiddenBytes = (size_t)maxCtx * dim * sizeof(float);
    if (!createBufferBDA(ctx->device, ctx->physicalDevice,
                         hiddenStateBuffer, hiddenStateMemory, hiddenStateAddr,
                         hiddenBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    // Scratch: residual(dim) + Q(dim) + K(kvDim) + V(kvDim) floats
    uint32_t kvDim = model->headCountKv * (dim / model->headCount);
    size_t scratchFloats = dim + dim + kvDim + kvDim;
    size_t scratchBytes = scratchFloats * sizeof(float);

    if (!createBufferBDA(ctx->device, ctx->physicalDevice,
                         scratchBuffer, scratchMemory, scratchAddr,
                         scratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return false;
    }

    fprintf(stderr, "[kernel-entry] mailbox=%llu output=%llu hidden=%llu(%zuMB) scratch=%llu(%zuB)\n",
            (unsigned long long)mailboxAddr, (unsigned long long)outputTokenAddr,
            (unsigned long long)hiddenStateAddr, hiddenBytes / 1024 / 1024,
            (unsigned long long)scratchAddr, scratchBytes);

    // Logits buffer: HOST_VISIBLE so test harness can read directly
    size_t logitsBytes = (size_t)model->vocabSize * sizeof(float);
    if (!createBufferHostVisible(ctx->device, ctx->physicalDevice,
                                 logitsBuffer, logitsMemory,
                                 (void**)&logitsMapped, logitsAddr,
                                 logitsBytes)) {
        return false;
    }
    fprintf(stderr, "[kernel-entry] logits=%llu (%zuMB, host-visible)\n",
            (unsigned long long)logitsAddr, logitsBytes / 1024 / 1024);

    return true;
}

void InferenceEngine::cleanupKernelEntryBuffers() {
    if (mailboxBuffer) vkDestroyBuffer(ctx->device, mailboxBuffer, nullptr);
    if (mailboxMemory) vkFreeMemory(ctx->device, mailboxMemory, nullptr);
    mailboxBuffer = VK_NULL_HANDLE;
    mailboxMemory = VK_NULL_HANDLE;
    mailboxAddr = 0;

    if (outputTokenBuffer) vkDestroyBuffer(ctx->device, outputTokenBuffer, nullptr);
    if (outputTokenMemory) vkFreeMemory(ctx->device, outputTokenMemory, nullptr);
    outputTokenBuffer = VK_NULL_HANDLE;
    outputTokenMemory = VK_NULL_HANDLE;
    outputTokenAddr = 0;

    if (scratchBuffer) vkDestroyBuffer(ctx->device, scratchBuffer, nullptr);
    if (scratchMemory) vkFreeMemory(ctx->device, scratchMemory, nullptr);
    scratchBuffer = VK_NULL_HANDLE;
    scratchMemory = VK_NULL_HANDLE;
    scratchAddr = 0;

    if (hiddenStateBuffer) vkDestroyBuffer(ctx->device, hiddenStateBuffer, nullptr);
    if (hiddenStateMemory) vkFreeMemory(ctx->device, hiddenStateMemory, nullptr);
    hiddenStateBuffer = VK_NULL_HANDLE;
    hiddenStateMemory = VK_NULL_HANDLE;
    hiddenStateAddr = 0;

    if (logitsBuffer) vkDestroyBuffer(ctx->device, logitsBuffer, nullptr);
    if (logitsMemory) vkFreeMemory(ctx->device, logitsMemory, nullptr);
    logitsBuffer = VK_NULL_HANDLE;
    logitsMemory = VK_NULL_HANDLE;
    logitsAddr = 0;
    logitsMapped = nullptr;

    if (layerParamsBuffer) vkDestroyBuffer(ctx->device, layerParamsBuffer, nullptr);
    if (layerParamsMemory) vkFreeMemory(ctx->device, layerParamsMemory, nullptr);
    layerParamsBuffer = VK_NULL_HANDLE;
    layerParamsMemory = VK_NULL_HANDLE;
    layerParamsAddr = 0;
}

// ============================================================================
// cleanup — release all engine-owned GPU resources.
// ============================================================================
void InferenceEngine::cleanup() {
    cleanupKernelEntryBuffers();
    cleanupWeightBuffer();
    cleanupDequantBuffer();
    cleanupEmbedCache();
}

uint32_t InferenceEngine::forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers) {
    uint32_t dim = model->embeddingLength;
    uint32_t headDim = dim / model->headCount;
    uint32_t hiddenDim = model->feedForwardLength;

    allocator->reset();

    uint32_t seqLen = seqPos + 1;
    size_t hiddenSize = dim * sizeof(float);
    size_t qkvSize = (size_t)seqLen * headDim * model->headCount * sizeof(float);
    size_t kvSize = (size_t)seqLen * headDim * model->headCountKv * sizeof(float);
    size_t attnOutSize = dim * sizeof(float);
    size_t mlpOutSize = dim * sizeof(float);
    size_t logitsSize = (size_t)model->vocabSize * sizeof(float);
    size_t sampleSize = 16;

    uint64_t hiddenAddr = allocator->alloc(hiddenSize);
    uint64_t qAddr = allocator->alloc(qkvSize);
    uint64_t kAddr = allocator->alloc(kvSize);
    uint64_t vAddr = allocator->alloc(kvSize);
    uint64_t attnOutAddr = allocator->alloc(attnOutSize);
    uint64_t mlpOutAddr = allocator->alloc(mlpOutSize);
    uint64_t logitsAddr = allocator->alloc(logitsSize);
    uint64_t sampleOutAddr = allocator->alloc(sampleSize);

    if (!hiddenAddr || !qAddr || !kAddr || !vAddr || !attnOutAddr || !mlpOutAddr || !logitsAddr || !sampleOutAddr) {
        fprintf(stderr, "[FATAL] Ring allocator overflow at seqPos=%u\n", seqPos);
        return 0;
    }

    uint64_t qRowAddr = qAddr + (uint64_t)seqPos * dim * sizeof(float);
    uint64_t kRowAddr = kAddr + (uint64_t)seqPos * headDim * model->headCountKv * sizeof(float);
    uint64_t vRowAddr = vAddr + (uint64_t)seqPos * headDim * model->headCountKv * sizeof(float);

    auto fwdStart = std::chrono::steady_clock::now();

    // === ONE BATCH PER LAYER to avoid AMD WDDM TDR on large models ===

    // === Embedding Batch ===
    scheduler->beginBatch(0);
    uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
    if (embedAddr) {
        const TensorDesc* embedTensor = findTensor(*model, "token_embd.weight");
        bool useQ8Embed = (embedTensor && embedTensor->format == QuantFormat::Q8_0 && !embedCacheReady);

        if (useQ8Embed) {
            EmbedPushConstants embedPC = {embedAddr, hiddenAddr, tokenId, seqPos, dim};
            scheduler->dispatchInBatch(pipelines->getPipeline("embed_q8_0"), pipelines->getLayout("embed_q8_0"),
                &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        } else {
            uint64_t embedAddr_dq = embedCacheReady ? embedCacheAddr : 0;
            if (!embedAddr_dq) {
                embedAddr_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight");
            }
            EmbedPushConstants embedPC = {embedAddr_dq ? embedAddr_dq : embedAddr, hiddenAddr, tokenId, seqPos, dim};
            scheduler->dispatchInBatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        }
        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAllThrottled(0.8);

    uint32_t layersToRun = std::min(maxLayers, model->blockCount);

    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);
        scheduler->beginBatch(0);

        // === Attention Dequant ===
        // Each tensor gets a NON-OVERLAPPING region of the dequant buffer.
        // Layout: attn_norm | Q | K | V | O
        size_t attnNormSize = getF16OutputSize(*model, prefix + ".attn_norm.weight");
        size_t qSize = (size_t)dim * dim * sizeof(float);
        size_t kSize = (size_t)(headDim * model->headCountKv) * dim * sizeof(float);
        size_t vSize = kSize;
        size_t oSize = getF16OutputSize(*model, prefix + ".attn_output.weight");

        size_t attnOffNorm = 0;
        size_t attnOffQ     = attnOffNorm + attnNormSize;
        size_t attnOffK     = attnOffQ + qSize;
        size_t attnOffV     = attnOffK + kSize;
        size_t attnOffO     = attnOffV + vSize;

        uint64_t addrAttnNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_norm.weight", attnOffNorm);
        uint64_t addrQW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_q.weight", attnOffQ);
        uint64_t addrKW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_k.weight", attnOffK);
        uint64_t addrVW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_v.weight", attnOffV);
        uint64_t addrOW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_output.weight", attnOffO);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === Attention Compute ===
        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        RmsNormPushConstants normPC = {hiddenAddr, addrAttnNorm_dq ? addrAttnNorm_dq : addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &normPC, sizeof(normPC), 1, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint64_t normHidden = attnOutAddr;
        uint64_t addrQW = findTensorAddr(*model, prefix + ".attn_q.weight");
        uint64_t addrKW = findTensorAddr(*model, prefix + ".attn_k.weight");
        uint64_t addrVW = findTensorAddr(*model, prefix + ".attn_v.weight");

        GemmPushConstants qPC = {normHidden, addrQW_dq ? addrQW_dq : addrQW, qRowAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &qPC, sizeof(qPC), (dim + 31) / 32, 1, 1);
        GemmPushConstants kPC = {normHidden, addrKW_dq ? addrKW_dq : addrKW, kRowAddr, 1, (uint32_t)(headDim * model->headCountKv), dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &kPC, sizeof(kPC), ((uint32_t)(headDim * model->headCountKv) + 31) / 32, 1, 1);
        GemmPushConstants vPC = {normHidden, addrVW_dq ? addrVW_dq : addrVW, vRowAddr, 1, (uint32_t)(headDim * model->headCountKv), dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &vPC, sizeof(vPC), ((uint32_t)(headDim * model->headCountKv) + 31) / 32, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint32_t ropeTotal = model->headCount * headDim;
        RopePushConstants ropePC = {qRowAddr, kRowAddr, seqLen, headDim, model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
            &ropePC, sizeof(ropePC), (ropeTotal + 31) / 32, 1, 1);

        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kRowAddr, vRowAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatchInBatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);
        }

        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {qAddr, kCacheAddr, vCacheAddr, attnOutAddr, seqLen, headDim,
                model->headCount, model->headCountKv, h, 1.0f / std::sqrt(static_cast<float>(headDim))};
            scheduler->dispatchInBatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                &attnPC, sizeof(attnPC), 1, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint64_t addrOW = findTensorAddr(*model, prefix + ".attn_output.weight");
        GemmPushConstants outPC = {attnOutAddr, addrOW_dq ? addrOW_dq : addrOW, mlpOutAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &outPC, sizeof(outPC), (dim + 31) / 32, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC1, sizeof(addPC1), (dim + 255) / 256, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

        // === FFN Dequant ===
        // Layout: ffn_norm | up | gate | down
        size_t upSize = getF16OutputSize(*model, prefix + ".ffn_up.weight");
        size_t gateSize = getF16OutputSize(*model, prefix + ".ffn_gate.weight");
        size_t downSize = getF16OutputSize(*model, prefix + ".ffn_down.weight");
        size_t ffnNormSize = getF16OutputSize(*model, prefix + ".ffn_norm.weight");

        size_t ffnOffNorm = 0;
        size_t ffnOffUp   = ffnOffNorm + ffnNormSize;
        size_t ffnOffGate = ffnOffUp + upSize;
        size_t ffnOffDown = ffnOffGate + gateSize;

        uint64_t addrFfnNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_norm.weight", ffnOffNorm);
        uint64_t addrUpW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_up.weight", ffnOffUp);
        uint64_t addrGateW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_gate.weight", ffnOffGate);
        uint64_t addrDownW_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_down.weight", ffnOffDown);

        if (layer == 0) {
            fprintf(stderr, "[diag] L0 attn dequant: norm@0x%llx(off=%zu) Q@0x%llx(off=%zu) K@0x%llx(off=%zu) V@0x%llx(off=%zu) O@0x%llx(off=%zu)\n",
                (unsigned long long)addrAttnNorm_dq, attnOffNorm,
                (unsigned long long)addrQW_dq, attnOffQ,
                (unsigned long long)addrKW_dq, attnOffK,
                (unsigned long long)addrVW_dq, attnOffV,
                (unsigned long long)addrOW_dq, attnOffO);
            fprintf(stderr, "[diag] L0 ffn   dequant: norm@0x%llx(off=%zu) up@0x%llx(off=%zu) gate@0x%llx(off=%zu) down@0x%llx(off=%zu)\n",
                (unsigned long long)addrFfnNorm_dq, ffnOffNorm,
                (unsigned long long)addrUpW_dq, ffnOffUp,
                (unsigned long long)addrGateW_dq, ffnOffGate,
                (unsigned long long)addrDownW_dq, ffnOffDown);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === FFN Compute ===
        uint64_t addrFfnNorm = findTensorAddr(*model, prefix + ".ffn_norm.weight");
        RmsNormPushConstants ffnNormPC = {hiddenAddr, addrFfnNorm_dq ? addrFfnNorm_dq : addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &ffnNormPC, sizeof(ffnNormPC), 1, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint64_t addrUpW = findTensorAddr(*model, prefix + ".ffn_up.weight");
        uint64_t addrGateW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW = findTensorAddr(*model, prefix + ".ffn_down.weight");

        uint64_t gateScratchAddr = logitsAddr;
        uint64_t upScratchAddr   = logitsAddr + (uint64_t)hiddenDim * sizeof(float);
        uint64_t interScratchAddr = logitsAddr;

        GemmPushConstants gatePC = {attnOutAddr,
            addrGateW_dq ? addrGateW_dq : addrGateW,
            gateScratchAddr, 1, hiddenDim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &gatePC, sizeof(gatePC), (hiddenDim + 31) / 32, 1, 1);

        GemmPushConstants upPC = {attnOutAddr,
            addrUpW_dq ? addrUpW_dq : addrUpW,
            upScratchAddr, 1, hiddenDim, dim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &upPC, sizeof(upPC), (hiddenDim + 31) / 32, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        SiluMulPushConstants siluPC = {gateScratchAddr, upScratchAddr, interScratchAddr, hiddenDim};
        scheduler->dispatchInBatch(pipelines->getPipeline("silu_mul"), pipelines->getLayout("silu_mul"),
            &siluPC, sizeof(siluPC), (hiddenDim + 255) / 256, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        GemmPushConstants downPC = {interScratchAddr,
            addrDownW_dq ? addrDownW_dq : addrDownW,
            mlpOutAddr, 1, dim, hiddenDim, 1.0f, 0};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &downPC, sizeof(downPC), (dim + 31) / 32, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC2, sizeof(addPC2), (dim + 255) / 256, 1, 1);

        scheduler->endBatch(VK_NULL_HANDLE);
        scheduler->syncAllThrottled(0.8);

        // Diagnostic: read back dequant buffer after layer 0 completes
        if (layer == 0 && dequantMemory) {
            void* mapped = nullptr;
            VkResult r = vkMapMemory(ctx->device, dequantMemory, 0, dequantCapacity, 0, &mapped);
            if (r == VK_SUCCESS && mapped) {
                float* df = (float*)mapped;
                fprintf(stderr, "[diag] L0 post-sync dequant buf[0..9]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9]);
                fprintf(stderr, "[diag] Q-dq@+qOff (%zu bytes in): ", attnOffQ);
                for (int j = 0; j < 10; ++j) fprintf(stderr, "%.6f ", df[attnOffQ / sizeof(float) + j]);
                fprintf(stderr, "\n");
                fprintf(stderr, "[diag] K-dq@+kOff (%zu bytes in): ", attnOffK);
                for (int j = 0; j < 10; ++j) fprintf(stderr, "%.6f ", df[attnOffK / sizeof(float) + j]);
                fprintf(stderr, "\n");
                vkUnmapMemory(ctx->device, dequantMemory);
            }
        }
    }

    // === Final Norm + LM Head Batch ===
    scheduler->beginBatch(0);
    uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
    if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");
    size_t outNormSize = getF16OutputSize(*model, "output_norm.weight");
    size_t finalOffNorm = 0;

    uint64_t addrOutNorm_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output_norm.weight", finalOffNorm);

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm_dq ? addrOutNorm_dq : addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
    scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
        &finalNorm, sizeof(finalNorm), 1, 1, 1);

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    size_t finalOffLMHead = finalOffNorm + outNormSize;
    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    uint64_t addrLMHead_dq = 0;
    uint32_t lmTransB = 0;
    if (addrLMHead) {
        addrLMHead_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output.weight", finalOffLMHead);
    } else {
        addrLMHead = findTensorAddr(*model, "token_embd.weight");
        if (embedCacheReady) {
            addrLMHead_dq = embedCacheAddr;
        } else {
            addrLMHead_dq = dequantWeightInBatch(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight", finalOffLMHead);
        }
        lmTransB = 1;
    }

    GemmPushConstants lmPC = {attnOutAddr, addrLMHead_dq ? addrLMHead_dq : addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f, lmTransB};
    scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
        &lmPC, sizeof(lmPC), (model->vocabSize + 31) / 32, 1, 1);

    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAll();

    auto fwdEnd = std::chrono::steady_clock::now();
    auto fwdMs = std::chrono::duration_cast<std::chrono::milliseconds>(fwdEnd - fwdStart).count();
    fprintf(stderr, "[fwd] forward layers: %lld ms\n", (long long)fwdMs);

    // Update KV cache seq lengths for next token (after GPU is done)
    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        kvCache->incrementSeqLen(layer);
    }

    // === Diagnostic: readback hidden state after embed ===
    if (allocator->mappedPtr && embedAddr) {
        uint64_t hOff = hiddenAddr - allocator->baseAddress;
        VkMappedMemoryRange r2 = {};
        r2.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r2.memory = allocator->memory;
        r2.offset = hOff & ~((uint64_t)4095);
        r2.size = (dim * sizeof(float) + 4095) & ~((uint64_t)4095);
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &r2);
        float* h = reinterpret_cast<float*>(allocator->mappedPtr + hOff);
        int nanCount = 0;
        float hsum = 0, hmax = -1e30f, hmin = 1e30f;
        for (uint32_t i = 0; i < dim; ++i) {
            if (std::isnan(h[i])) nanCount++;
            else { hsum += h[i]; if (h[i] > hmax) hmax = h[i]; if (h[i] < hmin) hmin = h[i]; }
        }
        fprintf(stderr, "[diag] hidden after layers[0..3]: %f %f %f %f nan=%d sum=%f min=%f max=%f\n",
            h[0], h[1], h[2], h[3], nanCount, hsum, hmin, hmax);
    }

    // === Diagnostic: readback logits before sampling ===
    if (allocator->mappedPtr) {
        size_t logOff = logitsAddr - allocator->baseAddress;
        VkMappedMemoryRange lr = {};
        lr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        lr.memory = allocator->memory;
        lr.offset = logOff & ~(static_cast<uint64_t>(4095));
        lr.size = (logitsSize + 4095) & ~(static_cast<uint64_t>(4095));
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &lr);
        float* logPtr = reinterpret_cast<float*>(allocator->mappedPtr + logOff);
        int nanL = 0;
        float logMax = -1e30f;
        uint32_t logArgmax = 0;
        for (uint32_t i = 0; i < model->vocabSize; ++i) {
            if (std::isnan(logPtr[i])) { nanL++; }
            else if (logPtr[i] > logMax) { logMax = logPtr[i]; logArgmax = i; }
        }
        fprintf(stderr, "[diag] logits[0..4]: %f %f %f %f %f nan=%d argmax=%u max=%f\n",
            logPtr[0], logPtr[1], logPtr[2], logPtr[3], logPtr[4], nanL, logArgmax, logMax);
    }

    // === CPU Sampling ===
    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t logitsOffset = logitsAddr - allocator->baseAddress;
        size_t alignedOff = logitsOffset & ~(static_cast<uint64_t>(4095));
        size_t alignedEnd = (logitsOffset + logitsSize + 4095) & ~(static_cast<uint64_t>(4095));
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = alignedOff;
        range.size = alignedEnd - alignedOff;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        float* cpuLogits = reinterpret_cast<float*>(allocator->mappedPtr + logitsOffset);
        if (cpuLogits) {
            nextToken = sampleArgmax(cpuLogits, model->vocabSize);
        }
    }

    lastLogitsAddr = logitsAddr;
    lastLogitsOffset = logitsAddr - allocator->baseAddress;
    return nextToken;
}

// ============================================================================

// forwardKernelEntry — dispatch kernel_entry.comp for a single forward pass.
// Uses the one-dispatch persistent kernel with LayerParams.
// ============================================================================
uint32_t InferenceEngine::forwardKernelEntry(uint32_t tokenId, uint32_t seqPos) {
    uint32_t dim = model->embeddingLength;
    uint32_t headDim = dim / model->headCount;
    uint32_t kvDim = model->headCountKv * headDim;

    // 1. Write mailbox: tokenReady=1, tokenId, seqLen
    {
        uint32_t mailboxData[6] = {};
        mailboxData[0] = 1;          // tokenReady
        mailboxData[1] = 0;          // tokenAck
        mailboxData[2] = tokenId;
        mailboxData[3] = seqPos + 1; // seqLen (1-based)
        mailboxData[4] = 0;          // layerIndex
        mailboxData[5] = 0;          // phase

        void* mapped = nullptr;
        VkResult r = vkMapMemory(ctx->device, mailboxMemory, 0, sizeof(mailboxData), 0, &mapped);
        if (r != VK_SUCCESS || !mapped) {
            fprintf(stderr, "[kernel-entry] FAILED to map mailbox\n");
            return 0;
        }
        memcpy(mapped, mailboxData, sizeof(mailboxData));
        vkUnmapMemory(ctx->device, mailboxMemory);
    }

    // 2. Build push constants
    KernelEntryPushConstants pc = {};
    pc.addrMailbox     = mailboxAddr;
    pc.addrTokenEmbed  = embedCacheReady ? embedCacheAddr : findTensorAddr(*model, "token_embd.weight");
    pc.addrOutputNorm  = findTensorAddr(*model, "output_norm.weight");
    if (!pc.addrOutputNorm) pc.addrOutputNorm = findTensorAddr(*model, "norm.weight");
    pc.addrLMHead      = findTensorAddr(*model, "output.weight");
    if (!pc.addrLMHead) {
        // Weight-tied: use token_embd as LM head
        pc.addrLMHead = embedCacheReady ? embedCacheAddr : findTensorAddr(*model, "token_embd.weight");
    }
    pc.addrHiddenState = hiddenStateAddr;  // persistent activation buffer
    pc.addrOutput      = outputTokenAddr;
    pc.addrLayerParams = layerParamsAddr;
    pc.addrScratch     = scratchAddr;
    pc.addrLogits      = logitsAddr;
    pc.vocabSize       = model->vocabSize;
    pc.embeddingDim    = dim;
    pc.nLayers         = model->blockCount;
    pc.headDim         = headDim;
    pc.nHeads          = model->headCount;
    pc.nKvHeads        = model->headCountKv;
    pc.ropeBase        = 10000.0f;
    pc.ropeScale       = 1.0f;

    fprintf(stderr, "[kernel-entry] dispatch: tokenId=%u seqPos=%u nLayers=%u dim=%u headDim=%u\n",
            tokenId, seqPos, model->blockCount, dim, headDim);
    fprintf(stderr, "[kernel-entry] embed=%llu outputNorm=%llu lmHead=%llu layerParams=%llu scratch=%llu\n",
            (unsigned long long)pc.addrTokenEmbed, (unsigned long long)pc.addrOutputNorm,
            (unsigned long long)pc.addrLMHead, (unsigned long long)pc.addrLayerParams,
            (unsigned long long)pc.addrScratch);

    // 3. Dispatch ONE workgroup — kernel_entry.comp polls mailbox internally
    scheduler->dispatch(pipelines->getPipeline("kernel_entry"), pipelines->getLayout("kernel_entry"),
                         &pc, sizeof(pc), 1, 1, 1);
    scheduler->syncAllThrottled();

    // 4. Read output token from output buffer
    uint32_t nextToken = 0;
    if (outputTokenMemory) {
        void* mapped = nullptr;
        VkResult r = vkMapMemory(ctx->device, outputTokenMemory, 0, sizeof(uint32_t), 0, &mapped);
        if (r == VK_SUCCESS && mapped) {
            memcpy(&nextToken, mapped, sizeof(uint32_t));
            vkUnmapMemory(ctx->device, outputTokenMemory);
        }
    }

    fprintf(stderr, "[kernel-entry] output token: %u\n", nextToken);
    // Logits are in the host-visible logits buffer
    lastLogitsAddr = logitsAddr;
    lastLogitsOffset = 0;  // logits start at offset 0 in their own buffer
    return nextToken;
}

uint32_t InferenceEngine::forward(uint32_t tokenId, uint32_t seqPos) {
    if (kernelEntryReady) {
        return forwardKernelEntry(tokenId, seqPos);
    }
    return forwardPartial(tokenId, seqPos, model->blockCount);
}

std::vector<uint32_t> InferenceEngine::generate(const std::string& prompt, uint32_t maxTokens) {
    std::vector<uint32_t> tokens = tokenizer->encode(prompt);

    std::cout << "Prompt tokens (" << tokens.size() << "): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        forward(tokens[i], static_cast<uint32_t>(i));
    }

    uint32_t seqPos = static_cast<uint32_t>(tokens.size());
    auto startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::minutes(2);

    for (uint32_t i = 0; i < maxTokens; ++i) {
        auto now = std::chrono::steady_clock::now();
        if (now - startTime > timeout) {
            std::cerr << "\n[TIMEOUT] Generation exceeded 2 minutes, stopping.\n";
            break;
        }

        uint32_t nextToken = forward(tokens.back(), seqPos);
        tokens.push_back(nextToken);
        seqPos++;

        std::cout << "Token " << seqPos << ": " << nextToken << "\n";
        if (nextToken == tokenizer->eosTokenId()) break;
    }

    return tokens;
}

std::vector<uint32_t> InferenceEngine::forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft) {
    std::vector<uint32_t> accepted;

    uint32_t draftLayers = std::min(2u, model->blockCount);
    std::vector<uint32_t> draftTokens;
    uint32_t currentToken = tokenId;
    for (uint32_t i = 0; i < nDraft; ++i) {
        currentToken = draftForward(currentToken, seqPos + i, draftLayers);
        draftTokens.push_back(currentToken);
    }

    std::cout << "Draft tokens: ";
    for (auto t : draftTokens) std::cout << t << " ";
    std::cout << "\n";

    for (uint32_t i = 0; i < nDraft; ++i) {
        uint32_t inputTok = (i == 0) ? tokenId : draftTokens[i - 1];
        uint32_t fullToken = forward(inputTok, seqPos + i);
        if (fullToken == draftTokens[i]) {
            accepted.push_back(draftTokens[i]);
        } else {
            accepted.push_back(fullToken);
            break;
        }
    }

    std::cout << "Accepted " << accepted.size() << "/" << nDraft << " draft tokens\n";
    return accepted;
}

std::vector<uint32_t> InferenceEngine::generateSpeculative(const std::string& prompt, uint32_t maxTokens, uint32_t nDraft) {
    std::vector<uint32_t> tokens = tokenizer->encode(prompt);

    std::cout << "Prompt tokens (" << tokens.size() << "): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        forward(tokens[i], static_cast<uint32_t>(i));
    }

    uint32_t seqPos = static_cast<uint32_t>(tokens.size());
    uint32_t generated = 0;
    auto startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::minutes(2);

    while (generated < maxTokens) {
        auto now = std::chrono::steady_clock::now();
        if (now - startTime > timeout) {
            std::cerr << "\n[TIMEOUT] Speculative generation exceeded 2 minutes, stopping.\n";
            break;
        }

        auto accepted = forwardSpeculative(tokens.back(), seqPos, nDraft);

        for (auto t : accepted) {
            tokens.push_back(t);
            seqPos++;
            generated++;
            std::cout << "Token " << seqPos << ": " << t << " (speculative)\n";
            if (t == tokenizer->eosTokenId() || generated >= maxTokens) break;
        }

        if (generated >= maxTokens) break;
    }

    return tokens;
}

uint32_t InferenceEngine::draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers) {
    return forwardPartial(tokenId, seqPos, nLayers);
}

uint32_t InferenceEngine::verifyDraftToken(uint32_t draftToken, uint32_t seqPos) {
    return forward(draftToken, seqPos);
}

uint32_t InferenceEngine::sampleGpu(uint64_t logitsAddr, uint32_t vocabSize,
                                      uint64_t sampleOutAddr, uint64_t scratchAddr) {
    uint32_t seed = static_cast<uint32_t>(std::random_device{}());

    uint32_t effectiveK = 40;
    float topP = 0.9f;

    TopKPushConstants pc = {};
    pc.addrLogits = logitsAddr;
    pc.addrOutput = sampleOutAddr;
    pc.addrScratch = scratchAddr;
    pc.vocabSize = vocabSize;
    pc.temperature = 1.0f;
    pc.topK = effectiveK;
    pc.topP = topP;
    pc.seed = seed;

    uint32_t wgCount = (vocabSize + 255) / 256;
    scheduler->dispatch(pipelines->getPipeline("topk"), pipelines->getLayout("topk"),
                         &pc, sizeof(pc), wgCount, 1, 1);
    scheduler->syncAllThrottled();

    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t localOffset = sampleOutAddr - allocator->baseAddress;

        size_t alignedOffset = localOffset & ~(static_cast<size_t>(4096) - 1);
        size_t alignedEnd = (localOffset + sizeof(uint32_t) + 4095) & ~(static_cast<size_t>(4095));

        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = alignedOffset;
        range.size = alignedEnd - alignedOffset;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);

        std::memcpy(&nextToken, allocator->mappedPtr + localOffset, sizeof(uint32_t));
    }

    return nextToken;
}

uint32_t InferenceEngine::sampleArgmax(const float* logits, uint32_t vocabSize) {
    if (!logits) return 0;
    uint32_t best = 0;
    float bestVal = logits[0];
    for (uint32_t i = 1; i < vocabSize; ++i) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            best = i;
        }
    }
    return best;
}

} // namespace rdna4
