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
#include <chrono>
#include <Windows.h>
#include <future>

using namespace rdna4;

static LONG WINAPI crashHandler(EXCEPTION_POINTERS* ep) {
    fprintf(stderr, "\n=== CRASH: 0x%08X at addr %p ===\n",
            ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
    fprintf(stderr, "RAX=%p RBX=%p RCX=%p RDX=%p\n",
            (void*)ep->ContextRecord->Rax, (void*)ep->ContextRecord->Rbx,
            (void*)ep->ContextRecord->Rcx, (void*)ep->ContextRecord->Rdx);
    fprintf(stderr, "RSI=%p RDI=%p RBP=%p RSP=%p\n",
            (void*)ep->ContextRecord->Rsi, (void*)ep->ContextRecord->Rdi,
            (void*)ep->ContextRecord->Rbp, (void*)ep->ContextRecord->Rsp);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

static void printTopK(const float* logits, uint32_t vocabSize, int k, const char* label) {
    k = std::min(k, (int)vocabSize);
    std::vector<std::pair<float, uint32_t>> indexed(vocabSize);
    for (uint32_t i = 0; i < vocabSize; ++i) indexed[i] = {logits[i], i};
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    printf("%s top-%d:", label, k);
    for (int i = 0; i < k; ++i) {
        printf(" %u=%.6f", indexed[i].second, indexed[i].first);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    SetUnhandledExceptionFilter(crashHandler);

    if (argc < 3) {
        fprintf(stderr, "Usage: test_inference <model.json> <model.bin>\n");
        return 1;
    }

    printf("=== Inference Test ===\n");

    std::string jsonPath = argv[1];
    std::string binPath = argv[2];

    printf("Loading model: %s\n", jsonPath.c_str());
    fflush(stdout);

    VulkanContext ctx;
    if (!ctx.init()) {
        fprintf(stderr, "Failed to initialize Vulkan\n");
        return 1;
    }

    WeightUploader uploader(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);
    ModelDesc model;
    try {
        model = uploader.load(jsonPath, binPath);
    } catch (const std::exception& e) {
        fprintf(stderr, "FATAL: uploader.load() threw: %s\n", e.what());
        ctx.cleanup();
        return 1;
    }
    if (model.tensors.empty()) {
        fprintf(stderr, "Failed to load model tensors\n");
        ctx.cleanup();
        return 1;
    }
    printf("Model loaded: %zu tensors\n", model.tensors.size());

    std::ifstream jsonFile(jsonPath);
    nlohmann::json fullJson;
    if (jsonFile.is_open()) jsonFile >> fullJson;

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
        ctx.cleanup();
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
    loadPipe("mlp_fused_gateup", sizeof(MlpFusedGateUpPushConstants));
    loadPipe("rope", sizeof(RopePushConstants));
    loadPipe("topk", sizeof(TopKPushConstants));
    loadPipe("add", sizeof(AddPushConstants));
    loadPipe("silu_mul", sizeof(SiluMulPushConstants));
    loadPipe("rms_norm", sizeof(RmsNormPushConstants));
    loadPipe("embed", sizeof(EmbedPushConstants));
    loadPipe("embed_q8_0", sizeof(EmbedPushConstants));
    loadPipe("kv_cache_write", sizeof(KVCacheWritePushConstants));
    loadPipe("dequantize", sizeof(DequantizePushConstants));
    loadPipe("dequantize_test", sizeof(DequantizePushConstants));
    loadPipe("kernel_entry", sizeof(KernelEntryPushConstants));

    Scheduler scheduler(ctx.device, ctx.queues, ctx.queueFamilyIndex);
    FencePool fencePool(ctx.device);
    scheduler.fencePool = &fencePool;

    InferenceEngine engine(&ctx, &model, &kvCache, &pipelines, &tokenizer,
                           &scheduler, &allocator);
    if (!engine.initDequantBuffer()) {
        fprintf(stderr, "Failed to allocate dequant staging buffer\n");
        scheduler.cleanup();
        pipelines.cleanup();
        kvCache.free();
        ctx.cleanup();
        return 1;
    }

    uint32_t tokenId = 1;
    uint32_t seqPos = 0;
    printf("Running GPU workload for token %u (seqPos=%u)...\n", tokenId, seqPos);
    fflush(stdout);

    // Wrap ALL GPU work (embed cache dequant + forward) in a single 2-minute timeout
    uint32_t nextToken = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto gpuFuture = std::async(std::launch::async, [&]() {
        try {
            if (!engine.initEmbedCache()) {
                fprintf(stderr, "Warning: embedding cache initialization failed, continuing without cache\n");
            }
            return engine.forward(tokenId, seqPos);
        } catch (...) {
            return 0u;
        }
    });
    auto status = gpuFuture.wait_for(std::chrono::seconds(120));
    if (status == std::future_status::timeout) {
        fprintf(stderr, "[TIMEOUT] GPU workload exceeded 2 minutes, GPU may be hung\n");
        fflush(stderr);
        scheduler.cleanup();
        pipelines.cleanup();
        kvCache.free();
        engine.cleanup();
        uploader.freeAll(model);
        ctx.cleanup();
        return 1;
    }
    nextToken = gpuFuture.get();
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    printf("GPU next token: %u (forward took %lld ms)\n", nextToken, (long long)ms);
    fflush(stdout);

    VkResult waitResult = vkDeviceWaitIdle(ctx.device);
    if (waitResult != VK_SUCCESS) {
        fprintf(stderr, "[ERROR] vkDeviceWaitIdle failed: %d — GPU may have crashed\n", (int)waitResult);
        scheduler.cleanup();
        pipelines.cleanup();
        kvCache.free();
        engine.cleanup();
        uploader.freeAll(model);
        ctx.cleanup();
        return 1;
    }

    // --- GPU readback ---
    float maxErr = 99.0f;
    float meanErr = 99.0f;
    bool pass = false;
    std::vector<float> gpuLogitsCopy;

    // Determine logits source: kernel_entry path uses logitsMapped, forwardPartial uses ring allocator
    const float* gpuLogits = nullptr;
    if (engine.logitsMapped) {
        gpuLogits = engine.logitsMapped;
        fprintf(stderr, "[readback] using kernel_entry logits buffer\n");
    } else if (allocator.mappedPtr && engine.lastLogitsOffset != 0) {
        size_t logitsSize = (size_t)model.vocabSize * sizeof(float);
        size_t logitsOff = engine.lastLogitsOffset;
        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator.memory;
        range.offset = logitsOff & ~(static_cast<size_t>(4096) - 1);
        range.size = ((logitsSize + 4095) & ~static_cast<size_t>(4095));
        vkInvalidateMappedMemoryRanges(ctx.device, 1, &range);
        gpuLogits = reinterpret_cast<float*>(allocator.mappedPtr + logitsOff);
        fprintf(stderr, "[readback] using ring allocator logits at offset %zu\n", logitsOff);
    }

    if (!gpuLogits) {
        fprintf(stderr, "[ERROR] No valid logits pointer — forward may have failed\n");
    } else {
        bool gpuHasNan = false;
        for (uint32_t i = 0; i < model.vocabSize; ++i) {
            if (std::isnan(gpuLogits[i]) || std::isinf(gpuLogits[i])) {
                gpuHasNan = true;
                break;
            }
        }

        gpuLogitsCopy.assign(gpuLogits, gpuLogits + model.vocabSize);

        printf("GPU logits[0..9]: ");
        for (int i = 0; i < 10 && i < (int)model.vocabSize; ++i) {
            printf("%.6f ", gpuLogitsCopy[i]);
        }
        printf(gpuHasNan ? " [NAN/INF DETECTED!]\n" : "\n");
        printTopK(gpuLogitsCopy.data(), model.vocabSize, 5, "GPU");

        uint32_t gpuArgmax = 0;
        float gpuMax = gpuLogitsCopy[0];
        for (uint32_t i = 1; i < model.vocabSize; ++i) {
            if (gpuLogitsCopy[i] > gpuMax) {
                gpuMax = gpuLogitsCopy[i];
                gpuArgmax = i;
            }
        }
        printf("GPU argmax: token %u = %.6f\n", gpuArgmax, gpuMax);

        // --- CPU reference ---
        CpuReference cpuRef;
        if (!cpuRef.load(jsonPath, binPath)) {
            fprintf(stderr, "Failed to load CPU reference model\n");
        } else {
            printf("Running CPU reference forward pass...\n");
            fflush(stdout);
            std::vector<float> cpuLogits = cpuRef.forward(tokenId);
            if (cpuLogits.size() != model.vocabSize) {
                fprintf(stderr, "[ERROR] CPU reference returned %zu logits, expected %u\n",
                        cpuLogits.size(), model.vocabSize);
            } else {
                printf("CPU logits[0..9]: ");
                for (int i = 0; i < 10 && i < (int)model.vocabSize; ++i) {
                    printf("%.6f ", cpuLogits[i]);
                }
                printf("\n");
                printTopK(cpuLogits.data(), model.vocabSize, 5, "CPU");

                auto [cpuArgmax, cpuMax] = CpuReference::argmax(cpuLogits);
                printf("CPU argmax: token %u = %.6f\n", cpuArgmax, cpuMax);

                maxErr = 0.0f;
                double sumErr = 0.0;
                for (uint32_t i = 0; i < model.vocabSize; ++i) {
                    float err = std::abs(gpuLogitsCopy[i] - cpuLogits[i]);
                    if (err > maxErr) maxErr = err;
                    sumErr += err;
                }
                meanErr = (float)(sumErr / model.vocabSize);
                pass = (maxErr < 0.01f);
            }
        }
    }

    printf("Max absolute error: %.6f\n", maxErr);
    printf("Mean absolute error: %.6f\n", meanErr);
    printf("%s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);
    fflush(stderr);

    vkDeviceWaitIdle(ctx.device);
    uploader.freeAll(model);
    scheduler.cleanup();
    pipelines.cleanup();
    kvCache.free();
    engine.cleanup();
    ctx.cleanup();

    // Use _exit to skip CRT cleanup (avoids driver heap corruption at shutdown)
    _exit(pass ? 0 : 1);
    return pass ? 0 : 1;
}
