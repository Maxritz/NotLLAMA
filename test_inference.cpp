#include "rdna4.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_types.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_kv_cache.hpp"
#include "rdna4_pipeline.hpp"
#include "rdna4_tokenizer.hpp"
#include "rdna4_engine.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include "cpu_reference.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace rdna4;

static size_t computeLogitsOffset(uint32_t dim, uint32_t headCount,
                                   uint32_t headCountKv, uint32_t headDim,
                                   uint32_t vocabSize, uint32_t seqLen) {
    size_t hiddenSize = (size_t)dim * sizeof(float);
    size_t qkvSize = (size_t)seqLen * headDim * headCount * sizeof(float);
    size_t kvSize = (size_t)seqLen * headDim * headCountKv * sizeof(float);
    size_t attnOutSize = (size_t)dim * sizeof(float);
    size_t mlpOutSize = (size_t)dim * sizeof(float);
    const size_t align = 256;

    size_t off = 0;
    auto al = [&](size_t s) { off = (off + align - 1) & ~(align - 1); off += s; };

    al(hiddenSize);   // hidden
    al(qkvSize);      // Q
    al(kvSize);       // K
    al(kvSize);       // V
    al(attnOutSize);  // attnOut
    al(mlpOutSize);   // mlpOut

    return off; // logits offset
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test_inference <model.json> <model.bin>\n");
        return 1;
    }

    printf("=== Inference Test ===\n");

    std::string jsonPath = argv[1];
    std::string binPath = argv[2];

    // --- Vulkan setup (same as main.cpp) ---
    printf("Loading model...\n");

    VulkanContext ctx;
    if (!ctx.init()) {
        fprintf(stderr, "Failed to initialize Vulkan\n");
        return 1;
    }

    WeightUploader uploader(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);
    ModelDesc model = uploader.load(jsonPath, binPath);
    printf("Model loaded: %zu tensors\n", model.tensors.size());

    std::ifstream jsonFile(jsonPath);
    nlohmann::json fullJson;
    jsonFile >> fullJson;

    Tokenizer tokenizer;
    if (fullJson.contains("tokenizer")) {
        uploader.loadTokenizer(tokenizer, fullJson["tokenizer"]);
    }

    uint32_t headDim = model.embeddingLength / model.headCount;
    uint32_t maxContext = std::min(model.contextLength, 2048u);
    KVCacheManager kvCache(ctx.device, ctx.physicalDevice,
                           maxContext, model.blockCount,
                           model.headCountKv, headDim);
    if (!kvCache.allocate()) {
        fprintf(stderr, "KV cache OOM\n");
        return 1;
    }

    size_t ringSize = 64 * 1024 * 1024;
    RingAllocator allocator(ctx.device, ctx.physicalDevice, ringSize);

    PipelineBuilder pipelines(ctx.device);
    std::string spvDir = "shaders/";
    auto loadPipe = [&](const char* name, size_t pcSize) {
        pipelines.loadShader(name, spvDir + std::string(name) + ".spv");
        pipelines.createComputePipeline(name, pcSize);
    };

    loadPipe("gemm", sizeof(GemmPushConstants));
    loadPipe("attention", sizeof(AttentionPushConstants));
    loadPipe("flash_attention", sizeof(FlashAttentionPushConstants));
    loadPipe("mlp", sizeof(MlpPushConstants));
    loadPipe("rope", sizeof(RopePushConstants));
    loadPipe("topk", sizeof(TopKPushConstants));
    loadPipe("add", sizeof(AddPushConstants));
    loadPipe("rms_norm", sizeof(RmsNormPushConstants));
    loadPipe("embed", sizeof(EmbedPushConstants));
    loadPipe("kv_cache_write", sizeof(KVCacheWritePushConstants));
    loadPipe("dequantize", sizeof(DequantizePushConstants));

    Scheduler scheduler(ctx.device, ctx.queues, ctx.queueFamilyIndex);

    InferenceEngine engine(&ctx, &model, &kvCache, &pipelines, &tokenizer,
                           &scheduler, &allocator);
    engine.initDequantBuffer();
    engine.initEmbedCache();

    // --- Run GPU forward pass ---
    uint32_t tokenId = 1;
    uint32_t seqPos = 0;
    printf("Running forward pass for token %u (seqPos=%u)...\n", tokenId, seqPos);

    uint32_t nextToken = engine.forward(tokenId, seqPos);
    printf("GPU next token: %u\n", nextToken);

    // --- Read back logits from ring allocator ---
    size_t logitsOff = computeLogitsOffset(model.embeddingLength, model.headCount,
                                            model.headCountKv, headDim,
                                            model.vocabSize, seqPos + 1);

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = allocator.memory;
    range.offset = logitsOff & ~(static_cast<size_t>(4096) - 1);
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(ctx.device, 1, &range);

    float* gpuLogits = reinterpret_cast<float*>(allocator.mappedPtr + logitsOff);

    printf("GPU logits[0..9]: ");
    for (int i = 0; i < 10; ++i) {
        printf("%.6f ", gpuLogits[i]);
    }
    printf("\n");

    uint32_t gpuArgmax = 0;
    float gpuMax = gpuLogits[0];
    for (uint32_t i = 1; i < model.vocabSize; ++i) {
        if (gpuLogits[i] > gpuMax) {
            gpuMax = gpuLogits[i];
            gpuArgmax = i;
        }
    }
    printf("GPU argmax: token %u = %.6f\n", gpuArgmax, gpuMax);

    // --- CPU reference ---
    CpuReference cpuRef;
    if (!cpuRef.load(jsonPath, binPath)) {
        fprintf(stderr, "Failed to load CPU reference model\n");
        return 1;
    }

    printf("Running CPU reference forward pass...\n");
    std::vector<float> cpuLogits = cpuRef.forward(tokenId);

    printf("CPU logits[0..9]: ");
    for (int i = 0; i < 10; ++i) {
        printf("%.6f ", cpuLogits[i]);
    }
    printf("\n");

    auto [cpuArgmax, cpuMax] = CpuReference::argmax(cpuLogits);
    printf("CPU argmax: token %u = %.6f\n", cpuArgmax, cpuMax);

    // --- Compare ---
    float maxErr = 0.0f;
    double sumErr = 0.0;
    for (uint32_t i = 0; i < model.vocabSize; ++i) {
        float err = std::abs(gpuLogits[i] - cpuLogits[i]);
        if (err > maxErr) maxErr = err;
        sumErr += err;
    }
    float meanErr = (float)(sumErr / model.vocabSize);

    printf("Max absolute error: %.6f\n", maxErr);
    printf("Mean absolute error: %.6f\n", meanErr);

    bool pass = (maxErr < 0.01f);
    printf("%s\n", pass ? "PASS" : "FAIL");

    // --- Cleanup ---
    scheduler.cleanup();
    pipelines.cleanup();
    kvCache.free();
    engine.cleanupEmbedCache();
    engine.cleanupDequantBuffer();
    uploader.freeAll(model);
    ctx.cleanup();

    return pass ? 0 : 1;
}
