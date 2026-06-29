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

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

void InferenceEngine::cleanup() {
    // Nothing to clean up — dequant/embed cache removed, weights are GPU-resident F32
}

// ============================================================================
// forwardPartial — single command buffer per token, no dequant, no diagnostics.
// All weights are pre-dequantized to F32 at load time by WeightUploader.
// One beginBatch/endBatch wraps ALL dispatches for the entire forward pass.
// ============================================================================
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

    // === SINGLE BATCH: all dispatches for entire forward pass ===
    scheduler->beginBatch(0);

    // === Embedding ===
    {
        uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
        if (embedAddr) {
            EmbedPushConstants embedPC = {embedAddr, hiddenAddr, tokenId, seqPos, dim};
            scheduler->dispatchInBatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                &embedPC, sizeof(embedPC), (dim + 255) / 256, 1, 1);
            scheduler->barrierBetweenGroups(
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
    }

    uint32_t layersToRun = std::min(maxLayers, model->blockCount);

    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);

        // === Attention RMS Norm ===
        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        RmsNormPushConstants normPC = {hiddenAddr, addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &normPC, sizeof(normPC), 1, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === QKV Projections (F32 weights, no dequant) ===
        uint64_t normHidden = attnOutAddr;
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".attn_q.weight");
            GemmPushConstants gpc = {normHidden, addrW, qRowAddr, 1, dim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (dim + 31) / 32, 1, 1);
        }
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".attn_k.weight");
            uint32_t kvDim = headDim * model->headCountKv;
            GemmPushConstants gpc = {normHidden, addrW, kRowAddr, 1, kvDim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (kvDim + 31) / 32, 1, 1);
        }
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".attn_v.weight");
            uint32_t kvDim = headDim * model->headCountKv;
            GemmPushConstants gpc = {normHidden, addrW, vRowAddr, 1, kvDim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (kvDim + 31) / 32, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === RoPE ===
        uint32_t ropeTotal = model->headCount * headDim;
        RopePushConstants ropePC = {qRowAddr, kRowAddr, seqLen, headDim, model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
            &ropePC, sizeof(ropePC), (ropeTotal + 31) / 32, 1, 1);

        // === KV Cache Write ===
        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kRowAddr, vRowAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatchInBatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);

            scheduler->barrierBetweenGroups(
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }

        // === Attention (per-head) ===
        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {qAddr, kCacheAddr, vCacheAddr, attnOutAddr, seqLen, headDim,
                model->headCount, model->headCountKv, h, 1.0f / std::sqrt(static_cast<float>(headDim))};
            scheduler->dispatchInBatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                &attnPC, sizeof(attnPC), 1, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === Attention Output Projection ===
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".attn_output.weight");
            GemmPushConstants gpc = {attnOutAddr, addrW, mlpOutAddr, 1, dim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (dim + 31) / 32, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === Residual Add (attention) ===
        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC1, sizeof(addPC1), (dim + 255) / 256, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

        // === FFN RMS Norm ===
        uint64_t addrFfnNorm = findTensorAddr(*model, prefix + ".ffn_norm.weight");
        RmsNormPushConstants ffnNormPC = {hiddenAddr, addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &ffnNormPC, sizeof(ffnNormPC), 1, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === FFN Gate + Up ===
        uint64_t gateScratchAddr  = logitsAddr;
        uint64_t upScratchAddr    = logitsAddr + (uint64_t)hiddenDim * sizeof(float);
        uint64_t interScratchAddr = logitsAddr + (uint64_t)hiddenDim * 2 * sizeof(float);
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
            GemmPushConstants gpc = {attnOutAddr, addrW, gateScratchAddr, 1, hiddenDim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (hiddenDim + 31) / 32, 1, 1);
        }
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".ffn_up.weight");
            GemmPushConstants gpc = {attnOutAddr, addrW, upScratchAddr, 1, hiddenDim, dim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (hiddenDim + 31) / 32, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === SiLU + Mul ===
        SiluMulPushConstants siluPC = {gateScratchAddr, upScratchAddr, interScratchAddr, hiddenDim};
        scheduler->dispatchInBatch(pipelines->getPipeline("silu_mul"), pipelines->getLayout("silu_mul"),
            &siluPC, sizeof(siluPC), (hiddenDim + 255) / 256, 1, 1);

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === FFN Down ===
        {
            uint64_t addrW = findTensorAddr(*model, prefix + ".ffn_down.weight");
            GemmPushConstants gpc = {interScratchAddr, addrW, mlpOutAddr, 1, dim, hiddenDim, 1.0f, 0};
            scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                &gpc, sizeof(gpc), (dim + 31) / 32, 1, 1);
        }

        scheduler->barrierBetweenGroups(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // === Residual Add (FFN) ===
        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatchInBatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
            &addPC2, sizeof(addPC2), (dim + 255) / 256, 1, 1);
    }

    // === Final Norm + LM Head ===
    {
        uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
        if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");
        RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatchInBatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
            &finalNorm, sizeof(finalNorm), 1, 1, 1);
    }

    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    {
        uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
        if (!addrLMHead) {
            addrLMHead = findTensorAddr(*model, "token_embd.weight");
        }
        GemmPushConstants gpc = {attnOutAddr, addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f, 1};
        scheduler->dispatchInBatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
            &gpc, sizeof(gpc), (model->vocabSize + 31) / 32, 1, 1);
    }

    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAll();

    auto fwdEnd = std::chrono::steady_clock::now();
    auto fwdMs = std::chrono::duration_cast<std::chrono::milliseconds>(fwdEnd - fwdStart).count();
    fprintf(stderr, "[fwd] %u layers: %lld ms\n", layersToRun, (long long)fwdMs);

    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        kvCache->incrementSeqLen(layer);
    }

    // === CPU Sampling ===
    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t logOff = logitsAddr - allocator->baseAddress;
        size_t alignedOff = logOff & ~(static_cast<uint64_t>(4095));
        size_t alignedEnd = (logOff + logitsSize + 4095) & ~(static_cast<uint64_t>(4095));
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = alignedOff;
        range.size = alignedEnd - alignedOff;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);
        float* cpuLogits = reinterpret_cast<float*>(allocator->mappedPtr + logOff);
        if (cpuLogits) {
            nextToken = sampleArgmax(cpuLogits, model->vocabSize);
        }
    }

    lastLogitsAddr = logitsAddr;
    lastLogitsOffset = logitsAddr - allocator->baseAddress;
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
