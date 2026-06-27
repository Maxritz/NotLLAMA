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

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

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
        EmbedPushConstants embedPC = {embedAddr, hiddenAddr, tokenId, seqPos, dim};
        scheduler->dispatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                             &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        scheduler->syncAll();
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

        // QKV projections (parallel)
        GemmPushConstants qpc = {normHidden, addrQW, qAddr, 1, dim, dim, 1.0f};
        GemmPushConstants kpc = {normHidden, addrKW, kAddr, 1, headDim * model->headCountKv, dim, 1.0f};
        GemmPushConstants vpc = {normHidden, addrVW, vAddr, 1, headDim * model->headCountKv, dim, 1.0f};

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

        // Output projection
        GemmPushConstants outPC = {attnOutAddr, addrOW, mlpOutAddr, 1, dim, dim, 1.0f};
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

        // MLP
        MlpPushConstants mlpPC = {
            normHidden, addrUpW, addrGateW, addrDownW, mlpOutAddr,
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

    // LM head
    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    if (!addrLMHead) addrLMHead = findTensorAddr(*model, "token_embd.weight");

    GemmPushConstants lmPC = {normHidden, addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f};
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
