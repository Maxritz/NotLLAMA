#include "rdna4.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_types.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_kv_cache.hpp"
#include "rdna4_pipeline.hpp"
#include "rdna4_tokenizer.hpp"
#include "rdna4_engine.hpp"
#include "rdna4_profiler.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <Windows.h>

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
    fprintf(stderr, "R8=%p R9=%p R10=%p R11=%p\n",
            (void*)ep->ContextRecord->R8, (void*)ep->ContextRecord->R9,
            (void*)ep->ContextRecord->R10, (void*)ep->ContextRecord->R11);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char** argv) {
    SetUnhandledExceptionFilter(crashHandler);

    FILE* dbg = fopen("debug.log", "w");
    if (!dbg) dbg = stderr;
    auto log = [&](const char* msg) { fprintf(dbg, "%s\n", msg); fflush(dbg); };

    log("start");
    printf("RDNA4 LLaMA starting...\n");
    std::cout << "NotLLAMA Vulkan Inference Engine\n";
    std::cout << "================================\n\n";

    log("creating context");
    VulkanContext ctx;
    log("init context");
    if (!ctx.init()) {
        log("context init failed");
        std::cerr << "Failed to initialize Vulkan\n";
        fclose(dbg);
        return 1;
    }
    log("context OK");
    std::cout << "Vulkan initialized with 4 ACE queues\n";

    Profiler profiler(ctx.device, ctx.physicalDevice);

    if (argc < 3) {
        std::cout << "Usage: rdna4_llama <model.weights.json> <model.weights.bin> [prompt] [--speculative N]\n";
        profiler.cleanup();
        ctx.cleanup();
        fclose(dbg);
        return 0;
    }

    std::string jsonPath = argv[1];
    std::string binPath = argv[2];
    std::string prompt = "Hello world";
    bool useSpeculative = false;
    uint32_t nDraft = 3;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--speculative" && i + 1 < argc) {
            useSpeculative = true;
            nDraft = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            prompt = arg;
        }
    }

    log("opening json");
    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        log("json open failed");
        std::cerr << "Failed to open: " << jsonPath << "\n";
        profiler.cleanup();
        ctx.cleanup();
        fclose(dbg);
        return 1;
    }
    log("parsing json");
    nlohmann::json fullJson;
    jsonFile >> fullJson;
    log("json parsed");

    profiler.beginCpu("load_weights");
    log("creating uploader");
    WeightUploader uploader(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);
    log("loading weights");
    ModelDesc model;
    try {
        model = uploader.load(jsonPath, binPath);
    } catch (const std::exception& e) {
        log(("uploader.load() threw: " + std::string(e.what())).c_str());
        std::cerr << "FATAL: " << e.what() << "\n";
        profiler.cleanup();
        ctx.cleanup();
        fclose(dbg);
        return 1;
    }
    if (model.tensors.empty()) {
        log("model has no tensors");
        std::cerr << "FATAL: Failed to load model tensors\n";
        profiler.cleanup();
        ctx.cleanup();
        fclose(dbg);
        return 1;
    }
    log("weights loaded");
    profiler.endCpu("load_weights");

    std::cout << "\nModel: " << model.architecture
              << " | Blocks: " << model.blockCount
              << " | Embed: " << model.embeddingLength
              << " | Heads: " << model.headCount << "/" << model.headCountKv
              << " | Vocab: " << model.vocabSize
              << " | Tensors: " << model.tensors.size() << "\n" << std::flush;

    log("loading tokenizer");
    Tokenizer tokenizer;
    if (fullJson.contains("tokenizer")) {
        uploader.loadTokenizer(tokenizer, fullJson["tokenizer"]);
    }
    log("tokenizer done");

    log("kv cache alloc");
    uint32_t headDim = model.embeddingLength / model.headCount;
    // Cap context to what fits in VRAM (131072 would need 4.5GB for KV alone)
    uint32_t maxContext = std::min(model.contextLength, 2048u);
    fprintf(stderr, "[kv_cache] context capped: %u -> %u (was %.1f GB for KV)\n",
            model.contextLength, maxContext,
            (double)model.contextLength * model.blockCount * model.headCountKv * headDim * 2 * 2 / (1024.0*1024*1024));
    fflush(stderr);
    KVCacheManager kvCache(ctx.device, ctx.physicalDevice,
                           maxContext, model.blockCount,
                           model.headCountKv, headDim);
    if (!kvCache.allocate()) { log("kv cache OOM"); std::cerr << "KV cache OOM\n"; fclose(dbg); return 1; }
    log("kv cache OK");

    log("ring alloc");
    // 80% VRAM cap for safety
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memProps);
    VkDeviceSize totalVram = 0;
    bool heapVisited[VK_MAX_MEMORY_HEAPS] = {};
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            !heapVisited[memProps.memoryTypes[i].heapIndex]) {
            heapVisited[memProps.memoryTypes[i].heapIndex] = true;
            totalVram += memProps.memoryHeaps[memProps.memoryTypes[i].heapIndex].size;
        }
    }
    size_t ringSize = std::max<size_t>(64 * 1024 * 1024,
        (size_t)model.vocabSize * sizeof(float) * 2 +
        (size_t)model.embeddingLength * sizeof(float) * 8);
    ringSize = std::min<size_t>(ringSize, (size_t)(totalVram * 0.8));
    fprintf(stderr, "[ring] VRAM: %.0f MB, ring capped at 80%%: %zu MB\n",
            (double)totalVram / (1024.0*1024.0), ringSize / (1024*1024));
    RingAllocator allocator(ctx.device, ctx.physicalDevice, ringSize);
    log("ring OK");

    log("creating pipelines");
    PipelineBuilder pipelines(ctx.device);
    std::string spvDir = "shaders/";

    auto loadPipe = [&](const char* name, size_t pcSize) {
        log(("loading shader: " + std::string(name)).c_str());
        pipelines.loadShader(name, spvDir + std::string(name) + ".spv");
        pipelines.createComputePipeline(name, pcSize);
        log(("pipeline OK: " + std::string(name)).c_str());
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
    loadPipe("dequant_turbo", sizeof(DequantizePushConstants));
    loadPipe("gemm_turbo", sizeof(GemmPushConstants));
    loadPipe("dequantize", sizeof(DequantizePushConstants));
    loadPipe("gemm_q4k", sizeof(GemmPushConstants));
    loadPipe("gemm_q6k", sizeof(GemmPushConstants));

    log("all pipelines OK");
    std::cout << "Pipelines loaded (including Flash Attention)\n";

    log("creating scheduler");
    Scheduler scheduler(ctx.device, ctx.queues, ctx.queueFamilyIndex);
    FencePool fencePool(ctx.device);
    scheduler.fencePool = &fencePool;

    // === BDA WRITE TEST ===
    // Allocate a small test buffer, dispatch a shader that writes DEADBEEF+tid via BDA,
    // then read back to verify BDA writes work at all.
    {
        log("BDA test: creating buffer");
        const uint32_t testN = 256;
        const size_t testBytes = testN * sizeof(uint32_t);

        VkBuffer testBuf = VK_NULL_HANDLE;
        VkDeviceMemory testMem = VK_NULL_HANDLE;
        VkBufferCreateInfo bufInfo = {};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = testBytes;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r = vkCreateBuffer(ctx.device, &bufInfo, nullptr, &testBuf);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[BDA_TEST] vkCreateBuffer failed: %d\n", r);
            log("BDA test: FAILED (vkCreateBuffer)");
            goto bda_test_done;
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(ctx.device, testBuf, &memReq);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &memProps);

        // Try DEVICE_LOCAL+HOST_VISIBLE first, fallback to HOST_VISIBLE+HOST_COHERENT
        uint32_t memTypeIdx = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                memTypeIdx = i; break;
            }
        }
        if (memTypeIdx == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((memReq.memoryTypeBits & (1 << i)) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                    (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    memTypeIdx = i; break;
                }
            }
        }
        if (memTypeIdx == UINT32_MAX) {
            fprintf(stderr, "[BDA_TEST] No suitable memory type found\n");
            vkDestroyBuffer(ctx.device, testBuf, nullptr);
            log("BDA test: FAILED (no memory type)");
            goto bda_test_done;
        }

        VkMemoryAllocateFlagsInfo flagsInfo = {};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memTypeIdx;
        allocInfo.pNext = &flagsInfo;
        r = vkAllocateMemory(ctx.device, &allocInfo, nullptr, &testMem);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[BDA_TEST] vkAllocateMemory failed: %d\n", r);
            vkDestroyBuffer(ctx.device, testBuf, nullptr);
            log("BDA test: FAILED (vkAllocateMemory)");
            goto bda_test_done;
        }
        r = vkBindBufferMemory(ctx.device, testBuf, testMem, 0);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[BDA_TEST] vkBindBufferMemory failed: %d\n", r);
            vkFreeMemory(ctx.device, testMem, nullptr);
            vkDestroyBuffer(ctx.device, testBuf, nullptr);
            log("BDA test: FAILED (vkBindBufferMemory)");
            goto bda_test_done;
        }

        VkBufferDeviceAddressInfo addrInfo = {};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = testBuf;
        uint64_t testBda = vkGetBufferDeviceAddress(ctx.device, &addrInfo);

        fprintf(stderr, "[BDA_TEST] buffer addr=0x%llx memType=%u\n",
                (unsigned long long)testBda, memTypeIdx);
        fflush(stderr);

        // Load the test shader
        log("BDA test: loading pipeline");
        pipelines.loadShader("bda_test", "shaders/bda_test.spv");
        pipelines.createComputePipeline("bda_test", sizeof(BdaTestPushConstants));

        // Dispatch
        log("BDA test: dispatching");
        BdaTestPushConstants pc = {};
        pc.addrOut = testBda;
        pc.nElements = testN;
        scheduler.dispatch(pipelines.getPipeline("bda_test"), pipelines.getLayout("bda_test"),
                           &pc, sizeof(pc), 1, 1, 1);
        scheduler.syncAll();

        // Read back
        void* mapped = nullptr;
        r = vkMapMemory(ctx.device, testMem, 0, testBytes, 0, &mapped);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[BDA_TEST] vkMapMemory failed: %d\n", r);
            vkDestroyBuffer(ctx.device, testBuf, nullptr);
            vkFreeMemory(ctx.device, testMem, nullptr);
            log("BDA test: FAILED (vkMapMemory)");
            goto bda_test_done;
        }
        uint32_t* data = (uint32_t*)mapped;
        bool allZero = true;
        bool allCorrect = true;
        for (uint32_t i = 0; i < testN; ++i) {
            if (data[i] != 0) allZero = false;
            if (data[i] != 0xDEADBEEF + i) allCorrect = false;
        }
        fprintf(stderr, "[BDA_TEST] results: allZero=%d allCorrect=%d first[0..3]=%08X %08X %08X %08X\n",
                allZero, allCorrect, data[0], data[1], data[2], data[3]);
        fflush(stderr);

        if (allCorrect) {
            printf("BDA TEST PASSED: shader wrote correct values via buffer device address\n");
        } else if (allZero) {
            printf("BDA TEST FAILED: all zeros — BDA writes NOT working\n");
        } else {
            printf("BDA TEST PARTIAL: some writes worked, some didn't\n");
        }

        vkUnmapMemory(ctx.device, testMem);
        vkDestroyBuffer(ctx.device, testBuf, nullptr);
        vkFreeMemory(ctx.device, testMem, nullptr);
        log("BDA test: done");
    }
    // === END BDA TEST ===
bda_test_done:;

    log("creating engine");
    InferenceEngine engine(&ctx, &model, &kvCache, &pipelines, &tokenizer, &scheduler, &allocator);
    if (!engine.initDequantBuffer()) {
        log("dequant buffer init failed");
        std::cerr << "Failed to initialize dequantization buffer\n";
    }
    if (!engine.initEmbedCache()) {
        log("embed cache init failed (will dequantize per-token)");
    }
    std::cout << "\nPrompt: \"" << prompt << "\"\n";
    if (useSpeculative) {
        std::cout << "Mode: Speculative decode (draft=" << nDraft << ")\n\n";
    } else {
        std::cout << "Mode: Standard autoregressive\n\n";
    }

    log("starting generate");
    std::cout << std::flush;
    std::vector<uint32_t> tokens;
    if (useSpeculative) {
        tokens = engine.generateSpeculative(prompt, 32, nDraft);
    } else {
        tokens = engine.generate(prompt, 32);
    }
    log("generate done");

    std::cout << "\nGenerated (" << tokens.size() << " tokens): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    std::cout << "\nDecoded: \"" << tokenizer.decode(tokens) << "\"\n" << std::flush;

    profiler.report();

    vkDeviceWaitIdle(ctx.device);
    scheduler.cleanup();
    pipelines.cleanup();
    kvCache.free();
    engine.cleanupEmbedCache();
    engine.cleanupDequantBuffer();
    uploader.freeAll(model);
    profiler.cleanup();
    ctx.cleanup();
    log("all cleaned up");
    fclose(dbg);
    return 0;
}
