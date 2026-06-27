#include "rdna4_engine.hpp"
#include "rdna4_types.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include "rdna4_kv_cache.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

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
                               size_t outOffset = 0) {
    const TensorDesc* t = findTensor(model, name);
    if (!t) return 0;
    // F16/F32 weights don't need dequantization
    if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) return t->gpuAddress;

    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;

    size_t outBytes = (size_t)nElements * sizeof(float);
    if (outOffset + outBytes > dequantBufSize) return 0;  // Too large

    // dispatch grid: local_size_x=32, one thread per element
    // Cap workgroups per dispatch to avoid GPU hang (9.7M workgroups caused hang)
    const uint32_t MAX_WG_PER_DISPATCH = 1024 * 1024;  // 1M workgroups = 32M threads
    uint32_t totalWorkgroups = (nElements + 31) / 32;
    if (totalWorkgroups == 0) totalWorkgroups = 1;

    uint32_t elementsPerChunk = MAX_WG_PER_DISPATCH * 32;
    uint32_t offset = 0;

    fprintf(stderr, "[dequant] %s: nElem=%u total_wg=%u (chunking %u at a time)\n",
            name.c_str(), nElements, totalWorkgroups, MAX_WG_PER_DISPATCH);
    fflush(stderr);

    while (offset < nElements) {
        uint32_t chunkSize = std::min(elementsPerChunk, nElements - offset);
        uint32_t chunkWg = (chunkSize + 31) / 32;

        DequantizePushConstants pc = {};
        pc.addrQuant = t->gpuAddress + (uint64_t)offset * 2;  // Quantized data is 2 bytes/element
        pc.addrOut = dequantBufAddr + outOffset + (uint64_t)offset * sizeof(float);
        pc.nElements = chunkSize;
        pc.quantFormat = static_cast<uint32_t>(t->format);
        pc.blockSize = t->blockSize;
        pc.blockElements = t->blockElements;
        pc.totalThreads = chunkWg * 32;

        sched->dispatch(pipes->getPipeline("dequantize"), pipes->getLayout("dequantize"),
                        &pc, sizeof(pc), chunkWg, 1, 1);
        sched->syncAll();

        offset += chunkSize;
    }

    return dequantBufAddr + outOffset;
}

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

bool InferenceEngine::initDequantBuffer() {
    // Find the largest weight tensor to determine buffer size
    // Cap at 128 MB — embedding table uses separate persistent cache
    size_t maxSize = 0;
    for (const auto& t : model->tensors) {
        if (t.format == QuantFormat::F16 || t.format == QuantFormat::F32) continue;
        // Skip token_embd — it uses the persistent embed cache
        if (t.name.find("token_embd") != std::string::npos) continue;
        uint32_t nElements = 1;
        for (auto d : t.shape) nElements *= d;
        size_t f32Bytes = (size_t)nElements * sizeof(float);
        if (f32Bytes > maxSize) maxSize = f32Bytes;
    }
    if (maxSize == 0) maxSize = 16 * 1024 * 1024;  // Minimum 16 MB
    if (maxSize > 128 * 1024 * 1024) maxSize = 128 * 1024 * 1024;  // Cap at 128 MB

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

    fprintf(stderr, "[dequant] staging buffer: %zu MB @ 0x%llx memType=%u props=0x%x\n",
            maxSize / 1024 / 1024, (unsigned long long)dequantAddr,
            memTypeIndex, memProps.memoryTypes[memTypeIndex].propertyFlags);
    return true;
}

void InferenceEngine::cleanupDequantBuffer() {
    if (dequantBuffer) vkDestroyBuffer(ctx->device, dequantBuffer, nullptr);
    if (dequantMemory) vkFreeMemory(ctx->device, dequantMemory, nullptr);
    dequantBuffer = VK_NULL_HANDLE;
    dequantMemory = VK_NULL_HANDLE;
    dequantAddr = 0;
}

bool InferenceEngine::initEmbedCache() {
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

    // Dequantize embedding table once, into the dequant staging buffer first,
    // then copy to the persistent embed cache
    // We use the dequant staging buffer as intermediate, then the embed cache
    // is the final resting place. Since both are DEVICE_LOCAL, we dispatch
    // dequant directly into the embed cache.
    fprintf(stderr, "[embed-cache] dequantizing embedding table (%u elements, %u workgroups)...\n",
            nElements, (nElements + 31) / 32);

    // Dequantize in chunks (same as dequantWeight)
    const uint32_t MAX_WG = 1024 * 1024;
    uint32_t totalWg = (nElements + 31) / 32;
    uint32_t elemPerChunk = MAX_WG * 32;
    uint32_t offset = 0;

    while (offset < nElements) {
        uint32_t chunk = std::min(elemPerChunk, nElements - offset);
        uint32_t chunkWg = (chunk + 31) / 32;

        DequantizePushConstants pc = {};
        pc.addrQuant = t->gpuAddress + (uint64_t)offset * 2;
        pc.addrOut = embedCacheAddr + (uint64_t)offset * sizeof(float);
        pc.nElements = chunk;
        pc.quantFormat = static_cast<uint32_t>(t->format);
        pc.blockSize = t->blockSize;
        pc.blockElements = t->blockElements;
        pc.totalThreads = chunkWg * 32;

        scheduler->dispatch(pipelines->getPipeline("dequantize"), pipelines->getLayout("dequantize"),
                            &pc, sizeof(pc), chunkWg, 1, 1);
        scheduler->syncAll();

        offset += chunk;
        fprintf(stderr, "[embed-cache] dequantized %u / %u elements\n", offset, nElements);
    }

    embedCacheReady = true;
    fprintf(stderr, "[embed-cache] ready @ 0x%llx (%zu MB)\n",
            (unsigned long long)embedCacheAddr, embedCacheSize / 1024 / 1024);
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

    fprintf(stderr, "[fwd] forwardPartial: tokenId=%u seqPos=%u maxLayers=%u\n", tokenId, seqPos, maxLayers);
    fprintf(stderr, "[fwd] allocs: hidden=%llx q=%llx k=%llx v=%llx attnOut=%llx mlpOut=%llx logits=%llx sample=%llx\n",
            (unsigned long long)hiddenAddr, (unsigned long long)qAddr, (unsigned long long)kAddr,
            (unsigned long long)vAddr, (unsigned long long)attnOutAddr, (unsigned long long)mlpOutAddr,
            (unsigned long long)logitsAddr, (unsigned long long)sampleOutAddr);

    // === Embedding ===
    uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
    fprintf(stderr, "[fwd] embedAddr=%llx embedCacheReady=%d embedCacheAddr=%llx\n",
            (unsigned long long)embedAddr, embedCacheReady, (unsigned long long)embedCacheAddr);
    if (embedAddr) {
        uint64_t embedAddr_dq = 0;
        if (embedCacheReady) {
            embedAddr_dq = embedCacheAddr;
        } else {
            embedAddr_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight");
        }

        EmbedPushConstants embedPC = {embedAddr_dq ? embedAddr_dq : embedAddr, hiddenAddr, tokenId, seqPos, dim};
        scheduler->dispatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                             &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        scheduler->syncAll();
    }

    // === Transformer blocks (limited to maxLayers) ===
    uint32_t layersToRun = std::min(maxLayers, model->blockCount);
    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);

        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        uint64_t addrFfnNorm  = findTensorAddr(*model, prefix + ".ffn_norm.weight");

        fprintf(stderr, "[fwd] layer %u/%u\n", layer, layersToRun);

        // --- Pre-attention RMS norm ---
        uint64_t addrAttnNorm_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_norm.weight");
        fprintf(stderr, "[fwd]   attnNorm addr=%llx dq=%llx\n", (unsigned long long)addrAttnNorm, (unsigned long long)addrAttnNorm_dq);
        RmsNormPushConstants normPC1 = {hiddenAddr, addrAttnNorm_dq ? addrAttnNorm_dq : addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                             &normPC1, sizeof(normPC1), 1, 1, 1);
        scheduler->syncAll();
        uint64_t normHidden = attnOutAddr;

        // --- Q GEMM ---
        uint64_t addrQW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_q.weight");
        uint64_t addrQW = findTensorAddr(*model, prefix + ".attn_q.weight");
        fprintf(stderr, "[fwd]   Q gemm: in=%llx w=%llx dq=%llx out=%llx\n",
                (unsigned long long)normHidden, (unsigned long long)addrQW, (unsigned long long)addrQW_dq, (unsigned long long)qRowAddr);
        GemmPushConstants qpc = {normHidden, addrQW_dq ? addrQW_dq : addrQW, qRowAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                             &qpc, sizeof(qpc), (dim+31)/32, 1, 1);
        scheduler->syncAll();

        // --- K GEMM ---
        uint64_t addrKW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_k.weight");
        uint64_t addrKW = findTensorAddr(*model, prefix + ".attn_k.weight");
        GemmPushConstants kpc = {normHidden, addrKW_dq ? addrKW_dq : addrKW, kRowAddr, 1, headDim * model->headCountKv, dim, 1.0f, 0};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                             &kpc, sizeof(kpc), (headDim*model->headCountKv+31)/32, 1, 1);
        scheduler->syncAll();

        // --- V GEMM ---
        uint64_t addrVW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_v.weight");
        uint64_t addrVW = findTensorAddr(*model, prefix + ".attn_v.weight");
        GemmPushConstants vpc = {normHidden, addrVW_dq ? addrVW_dq : addrVW, vRowAddr, 1, headDim * model->headCountKv, dim, 1.0f, 0};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                             &vpc, sizeof(vpc), (headDim*model->headCountKv+31)/32, 1, 1);
        scheduler->syncAll();

        // --- RoPE ---
        uint32_t ropeTotal = model->headCount * headDim;
        RopePushConstants ropePC = {qRowAddr, kRowAddr, seqLen, headDim,
                                     model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
                             &ropePC, sizeof(ropePC), (ropeTotal + 31) / 32, 1, 1);
        scheduler->syncAll();

        // --- KV cache write ---
        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kRowAddr, vRowAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                                 &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);
            scheduler->syncAll();
        }

        // --- Attention ---
        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {
                qAddr, kCacheAddr, vCacheAddr, attnOutAddr,
                seqLen, headDim,
                model->headCount, model->headCountKv, h,
                1.0f / std::sqrt(static_cast<float>(headDim))
            };
            scheduler->dispatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                                 &attnPC, sizeof(attnPC), 1, 1, 1);
        }
        scheduler->syncAll();

        // --- Output projection ---
        uint64_t addrOW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_output.weight");
        uint64_t addrOW = findTensorAddr(*model, prefix + ".attn_output.weight");
        GemmPushConstants outPC = {attnOutAddr, addrOW_dq ? addrOW_dq : addrOW, mlpOutAddr, 1, dim, dim, 1.0f, 0};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                             &outPC, sizeof(outPC), (dim+31)/32, 1, 1);
        scheduler->syncAll();

        // --- Residual add ---
        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                             &addPC1, sizeof(addPC1), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        // --- Pre-FFN RMS norm ---
        uint64_t addrFfnNorm_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_norm.weight");
        RmsNormPushConstants normPC2 = {hiddenAddr, addrFfnNorm_dq ? addrFfnNorm_dq : addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                             &normPC2, sizeof(normPC2), 1, 1, 1);
        scheduler->syncAll();
        normHidden = attnOutAddr;

        // --- MLP ---
        size_t upSize = getF16OutputSize(*model, prefix + ".ffn_up.weight");
        size_t gateSize = getF16OutputSize(*model, prefix + ".ffn_gate.weight");

        uint64_t addrUpW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_up.weight", 0);
        uint64_t addrGateW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_gate.weight", upSize);
        uint64_t addrDownW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_down.weight", upSize + gateSize);

        uint64_t addrUpW = findTensorAddr(*model, prefix + ".ffn_up.weight");
        uint64_t addrGateW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW = findTensorAddr(*model, prefix + ".ffn_down.weight");

        MlpPushConstants mlpPC = {
            normHidden,
            addrUpW_dq ? addrUpW_dq : addrUpW,
            addrGateW_dq ? addrGateW_dq : addrGateW,
            addrDownW_dq ? addrDownW_dq : addrDownW,
            mlpOutAddr,
            dim, hiddenDim, 0, 1.0f
        };
        scheduler->dispatch(pipelines->getPipeline("mlp"), pipelines->getLayout("mlp"),
                             &mlpPC, sizeof(mlpPC), 1, 1, 1);
        scheduler->syncAll();

        // --- Residual add ---
        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                             &addPC2, sizeof(addPC2), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        kvCache->incrementSeqLen(layer);
    }

    // === Final norm ===
    uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
    if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");

    uint64_t addrOutNorm_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output_norm.weight");
    RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm_dq ? addrOutNorm_dq : addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
    scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                         &finalNorm, sizeof(finalNorm), 1, 1, 1);
    scheduler->syncAll();
    uint64_t normHidden = attnOutAddr;

    // === LM head ===
    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    uint64_t addrLMHead_dq = 0;
    uint32_t lmTransB = 0;
    if (addrLMHead) {
        addrLMHead_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output.weight");
    } else {
        addrLMHead = findTensorAddr(*model, "token_embd.weight");
        if (embedCacheReady) {
            addrLMHead_dq = embedCacheAddr;
        } else {
            addrLMHead_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight");
        }
        lmTransB = 1;
    }

    GemmPushConstants lmPC = {normHidden, addrLMHead_dq ? addrLMHead_dq : addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f, lmTransB};
    scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                         &lmPC, sizeof(lmPC), (model->vocabSize+31)/32, 1, 1);
    scheduler->syncAll();

    // === Sample ===
    size_t sampleScratchSize = (size_t)model->vocabSize * sizeof(float);
    uint64_t sampleScratchAddr = allocator->alloc(sampleScratchSize);

    uint32_t nextToken = 0;
    if (sampleScratchAddr && allocator->mappedPtr) {
        nextToken = sampleGpu(logitsAddr, model->vocabSize,
                              sampleOutAddr, sampleScratchAddr);
    }
    if (!nextToken && allocator->mappedPtr) {
        size_t logitsOffset = logitsAddr - allocator->baseAddress;
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = logitsOffset & ~(static_cast<size_t>(4096) - 1);
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        float* cpuLogits = reinterpret_cast<float*>(allocator->mappedPtr + logitsOffset);
        if (cpuLogits) {
            nextToken = sampleArgmax(cpuLogits, model->vocabSize);
        }
    }

    return nextToken;
}

uint32_t InferenceEngine::forward(uint32_t tokenId, uint32_t seqPos) {
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
    for (uint32_t i = 0; i < maxTokens; ++i) {
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

    while (generated < maxTokens) {
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
    scheduler->syncAll();

    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t localOffset = sampleOutAddr - allocator->baseAddress;

        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = localOffset & ~(static_cast<size_t>(4096) - 1);
        range.size = VK_WHOLE_SIZE;
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
