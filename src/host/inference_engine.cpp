#include "rdna4_engine.hpp"
#include "rdna4_types.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include "rdna4_kv_cache.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

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

// Dequantize a weight tensor to float16 using the dedicated staging buffer
static uint64_t dequantWeight(Scheduler* sched, PipelineBuilder* pipes,
                               uint64_t dequantBufAddr, size_t dequantBufSize,
                               const ModelDesc& model, const std::string& name) {
    const TensorDesc* t = findTensor(model, name);
    if (!t) return 0;
    // F16/F32 weights don't need dequantization
    if (t->format == QuantFormat::F16 || t->format == QuantFormat::F32) return t->gpuAddress;

    uint32_t nElements = 1;
    for (auto d : t->shape) nElements *= d;

    size_t outBytes = nElements * sizeof(uint16_t);
    if (outBytes > dequantBufSize) return 0;  // Too large

    DequantizePushConstants pc = {};
    pc.addrQuant = t->gpuAddress;
    pc.addrOut = dequantBufAddr;
    pc.nElements = nElements;
    pc.quantFormat = static_cast<uint32_t>(t->format);
    pc.blockSize = t->blockSize;
    pc.blockElements = t->blockElements;

    uint32_t nBlocks = (nElements + t->blockElements - 1) / t->blockElements;
    uint32_t nWorkgroups = (nBlocks + 31) / 32;
    if (nWorkgroups == 0) nWorkgroups = 1;

    fprintf(stderr, "[dequant] dispatching: quant=0x%llx out=0x%llx nElem=%u fmt=%u bs=%u be=%u wg=%u\n",
            (unsigned long long)pc.addrQuant, (unsigned long long)pc.addrOut,
            pc.nElements, pc.quantFormat, pc.blockSize, pc.blockElements, nWorkgroups);
    fflush(stderr);

    sched->dispatch(pipes->getPipeline("dequantize"), pipes->getLayout("dequantize"),
                    &pc, sizeof(pc), nWorkgroups, 1, 1);
    sched->syncAll();

    return dequantBufAddr;
}

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

bool InferenceEngine::initDequantBuffer() {
    // Find the largest weight tensor to determine buffer size
    // Cap at 128 MB — anything larger (like token_embd) will use ring allocator
    size_t maxSize = 0;
    for (const auto& t : model->tensors) {
        if (t.format == QuantFormat::F16 || t.format == QuantFormat::F32) continue;
        uint32_t nElements = 1;
        for (auto d : t.shape) nElements *= d;
        size_t f16Bytes = nElements * sizeof(uint16_t);
        if (f16Bytes > maxSize && f16Bytes <= 128 * 1024 * 1024) maxSize = f16Bytes;
    }
    if (maxSize == 0) maxSize = 16 * 1024 * 1024;  // Minimum 16 MB

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

    // Map for debug readback
    vkMapMemory(ctx->device, dequantMemory, 0, maxSize, 0, reinterpret_cast<void**>(&dequantMapped));

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

void InferenceEngine::invalidateDequantBuffer() {
    if (!dequantMapped) return;
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = dequantMemory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
}

uint32_t InferenceEngine::forward(uint32_t tokenId, uint32_t seqPos) {
    uint32_t dim = model->embeddingLength;
    uint32_t headDim = dim / model->headCount;
    uint32_t hiddenDim = model->feedForwardLength;

    allocator->reset();

    size_t hiddenSize = dim * sizeof(float);
    size_t qkvSize = headDim * model->headCount * sizeof(float);
    size_t kvSize = headDim * model->headCountKv * sizeof(float);
    size_t attnOutSize = dim * sizeof(float);
    size_t mlpOutSize = dim * sizeof(float);
    size_t logitsSize = model->vocabSize * sizeof(float);
    size_t sampleSize = 16;

    uint64_t hiddenAddr = allocator->alloc(hiddenSize);
    uint64_t qAddr = allocator->alloc(qkvSize);
    uint64_t kAddr = allocator->alloc(kvSize);
    uint64_t vAddr = allocator->alloc(kvSize);
    uint64_t attnOutAddr = allocator->alloc(attnOutSize);
    uint64_t mlpOutAddr = allocator->alloc(mlpOutSize);
    uint64_t logitsAddr = allocator->alloc(logitsSize);
    uint64_t sampleOutAddr = allocator->alloc(sampleSize);

    // Embedding
    uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
    if (embedAddr) {
        // Debug: check if raw weight data is non-zero on GPU
        const TensorDesc* embedTensor = findTensor(*model, "token_embd.weight");
        if (embedTensor && dequantMapped) {
            // Read raw quantized bytes from GPU via staging (not possible directly)
            // But we can check the dequantized output
        }

        // Dequantize embedding weights
        uint64_t embedAddr_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "token_embd.weight");

        // Debug: check dequantized embedding
        if (dequantMapped) {
            invalidateDequantBuffer();
            uint16_t f16vals[8] = {0};
            std::memcpy(f16vals, dequantMapped, sizeof(f16vals));
            fprintf(stderr, "[debug] dequant test pattern: %u %u (expect ~42.0=0x5140 ~-3.5=0xC333)\n",
                    f16vals[0], f16vals[1]);
            fprintf(stderr, "[debug] dequant embed[0..7] raw f16: %u %u %u %u %u %u %u %u\n",
                    f16vals[0], f16vals[1], f16vals[2], f16vals[3],
                    f16vals[4], f16vals[5], f16vals[6], f16vals[7]);
            fflush(stderr);
        }

        EmbedPushConstants embedPC = {embedAddr_dq ? embedAddr_dq : embedAddr, hiddenAddr, tokenId, seqPos, dim};
        scheduler->dispatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                             &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        scheduler->syncAll();

        // Debug: read back first few elements of hidden state
        if (allocator->mappedPtr) {
            size_t localOff = hiddenAddr - allocator->baseAddress;
            float vals[4] = {0};
            std::memcpy(vals, allocator->mappedPtr + localOff, sizeof(vals));
            fprintf(stderr, "[debug] hidden after embed: %f %f %f %f\n", vals[0], vals[1], vals[2], vals[3]);
            fflush(stderr);
        }
    }

    // Transformer blocks
    for (uint32_t layer = 0; layer < model->blockCount; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);

        uint64_t addrQW = findTensorAddr(*model, prefix + ".attn_q.weight");
        uint64_t addrKW = findTensorAddr(*model, prefix + ".attn_k.weight");
        uint64_t addrVW = findTensorAddr(*model, prefix + ".attn_v.weight");
        uint64_t addrOW = findTensorAddr(*model, prefix + ".attn_output.weight");
        uint64_t addrUpW   = findTensorAddr(*model, prefix + ".ffn_up.weight");
        uint64_t addrGateW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW = findTensorAddr(*model, prefix + ".ffn_down.weight");
        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        uint64_t addrFfnNorm  = findTensorAddr(*model, prefix + ".ffn_norm.weight");

        if (!addrQW) addrQW = findTensorAddr(*model, prefix + ".attn_qkv.weight");

        // Pre-attention norm
        RmsNormPushConstants normPC1 = {hiddenAddr, addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                             &normPC1, sizeof(normPC1), 1, 1, 1);
        scheduler->syncAll();

        uint64_t normHidden = attnOutAddr;

        // QKV projections — dequantize weights to float16 first
        uint64_t addrQW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_q.weight");

        // Debug: read back first few dequantized weight values
        if (dequantMapped && addrQW_dq) {
            invalidateDequantBuffer();
            uint16_t f16vals[8] = {0};
            std::memcpy(f16vals, dequantMapped, sizeof(f16vals));
            fprintf(stderr, "[debug] dequant Q[0..7] raw f16: %u %u %u %u %u %u %u %u\n",
                    f16vals[0], f16vals[1], f16vals[2], f16vals[3],
                    f16vals[4], f16vals[5], f16vals[6], f16vals[7]);
            // Check test pattern at buffer start (every thread wrote globalId + 0x10000)
            uint32_t uintVals[8] = {0};
            std::memcpy(uintVals, dequantMapped, sizeof(uintVals));
            fprintf(stderr, "[debug] dequant BDA test[0..7]: 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x\n",
                    uintVals[0], uintVals[1], uintVals[2], uintVals[3],
                    uintVals[4], uintVals[5], uintVals[6], uintVals[7]);
            fprintf(stderr, "[debug] (expect 0x00010000 0x00010001 0x00010002 ...)\n");
            fflush(stderr);
        }

        uint64_t addrKW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_k.weight");
        uint64_t addrVW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_v.weight");

        GemmPushConstants qpc = {normHidden, addrQW_dq ? addrQW_dq : addrQW, qAddr, 1, dim, dim, 1.0f};
        GemmPushConstants kpc = {normHidden, addrKW_dq ? addrKW_dq : addrKW, kAddr, 1, headDim * model->headCountKv, dim, 1.0f};
        GemmPushConstants vpc = {normHidden, addrVW_dq ? addrVW_dq : addrVW, vAddr, 1, headDim * model->headCountKv, dim, 1.0f};

        scheduler->dispatchMulti({
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &qpc, sizeof(qpc), (dim+31)/32, 1, 1},
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &kpc, sizeof(kpc), (headDim*model->headCountKv+31)/32, 1, 1},
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &vpc, sizeof(vpc), (headDim*model->headCountKv+31)/32, 1, 1},
        });
        scheduler->syncAll();

        // RoPE
        RopePushConstants ropePC = {qAddr, kAddr, seqPos + 1, headDim,
                                     model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
                             &ropePC, sizeof(ropePC), model->headCount, 1, 1);
        scheduler->syncAll();

        // KV cache write — use buffer addresses from KVCacheManager
        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kAddr, vAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                                 &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);
            scheduler->syncAll();
        }

        // Attention per head
        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {
                qAddr, kAddr, vAddr, attnOutAddr,
                seqPos + 1, headDim,
                model->headCount, model->headCountKv, h,
                1.0f / std::sqrt(static_cast<float>(headDim))
            };
            scheduler->dispatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                                 &attnPC, sizeof(attnPC), 32, 1, 1, 0);
        }
        scheduler->syncAll();

        // Output projection — dequantize weights
        uint64_t addrOW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".attn_output.weight");
        GemmPushConstants outPC = {attnOutAddr, addrOW_dq ? addrOW_dq : addrOW, mlpOutAddr, 1, dim, dim, 1.0f};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                             &outPC, sizeof(outPC), (dim+31)/32, 1, 1);
        scheduler->syncAll();

        // Residual
        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                             &addPC1, sizeof(addPC1), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        // Pre-FFN norm
        RmsNormPushConstants normPC2 = {hiddenAddr, addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                             &normPC2, sizeof(normPC2), 1, 1, 1);
        scheduler->syncAll();

        normHidden = attnOutAddr;

        // MLP — dequantize weights to float16
        uint64_t addrUpW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_up.weight");
        uint64_t addrGateW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, prefix + ".ffn_down.weight");

        MlpPushConstants mlpPC = {
            normHidden,
            addrUpW_dq ? addrUpW_dq : addrUpW,
            addrGateW_dq ? addrGateW_dq : addrGateW,
            addrDownW_dq ? addrDownW_dq : addrDownW,
            mlpOutAddr,
            dim, hiddenDim, 0, 1.0f
        };
        scheduler->dispatch(pipelines->getPipeline("mlp"), pipelines->getLayout("mlp"),
                             &mlpPC, sizeof(mlpPC), 32, 1, 1);
        scheduler->syncAll();

        // Residual
        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                             &addPC2, sizeof(addPC2), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        kvCache->incrementSeqLen(layer);
    }

    // Final norm
    uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
    if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");

    RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
    scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                         &finalNorm, sizeof(finalNorm), 1, 1, 1);
    scheduler->syncAll();

    uint64_t normHidden = attnOutAddr;

    // LM head — dequantize weights
    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    if (!addrLMHead) addrLMHead = findTensorAddr(*model, "token_embd.weight");
    uint64_t addrLMHead_dq = dequantWeight(scheduler, pipelines, dequantAddr, dequantCapacity, *model, "output.weight");

    GemmPushConstants lmPC = {normHidden, addrLMHead_dq ? addrLMHead_dq : addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f};
    scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                         &lmPC, sizeof(lmPC), (model->vocabSize+31)/32, 1, 1);
    scheduler->syncAll();

    // Sample
    TopKPushConstants topkPC = {logitsAddr, sampleOutAddr, model->vocabSize, 1.0f, 1};
    scheduler->dispatch(pipelines->getPipeline("topk"), pipelines->getLayout("topk"),
                         &topkPC, sizeof(topkPC), 256, 1, 1);
    scheduler->syncAll();

    // Read back sampled token
    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t localOffset = sampleOutAddr - allocator->baseAddress;

        // Invalidate mapped memory to ensure GPU writes are visible to host
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = localOffset & ~(4096 - 1); // align to nonCoherentAtomSize
        range.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);

        std::memcpy(&nextToken, allocator->mappedPtr + localOffset, sizeof(uint32_t));
    } else {
        nextToken = sampleArgmax(nullptr, model->vocabSize);
    }

    return nextToken;
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

    std::vector<bool> verified(nDraft, false);
    for (uint32_t i = 0; i < nDraft && i < 3; ++i) {
        uint32_t fullToken = forward(tokenId, seqPos + i);
        verified[i] = (fullToken == draftTokens[i]);
        if (!verified[i]) {
            accepted.push_back(fullToken);
            break;
        } else {
            accepted.push_back(draftTokens[i]);
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
    (void)nLayers;
    return forward(tokenId, seqPos);
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
