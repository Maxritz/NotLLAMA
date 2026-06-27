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
    auto log = [&](const char* msg) { fprintf(dbg, "%s\n", msg); fflush(dbg); };

    log("start");
    printf("RDNA4 LLaMA starting...\n");
    std::cout << "RDNA4 LLaMA Inference Engine\n";
    std::cout << "=============================\n\n";

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
    ModelDesc model = uploader.load(jsonPath, binPath);
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
    size_t ringSize = 64 * 1024 * 1024;  // 64 MB for activations (enough for single token forward)
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
    loadPipe("mlp", sizeof(MlpPushConstants));
    loadPipe("rope", sizeof(RopePushConstants));
    loadPipe("topk", sizeof(TopKPushConstants));
    loadPipe("add", sizeof(AddPushConstants));
    loadPipe("rms_norm", sizeof(RmsNormPushConstants));
    loadPipe("embed", sizeof(EmbedPushConstants));
    loadPipe("kv_cache_write", sizeof(KVCacheWritePushConstants));
    loadPipe("dequantize", sizeof(DequantizePushConstants));

    log("all pipelines OK");
    std::cout << "Pipelines loaded (including Flash Attention)\n";

    log("creating scheduler");
    Scheduler scheduler(ctx.device, ctx.queues, ctx.queueFamilyIndex);

    log("creating engine");
    InferenceEngine engine(&ctx, &model, &kvCache, &pipelines, &tokenizer, &scheduler, &allocator);
    if (!engine.initDequantBuffer()) {
        log("dequant buffer init failed");
        std::cerr << "Failed to initialize dequantization buffer\n";
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

    pipelines.cleanup();
    kvCache.free();
    engine.cleanupDequantBuffer();
    uploader.freeAll(model);
    profiler.cleanup();
    ctx.cleanup();
    log("all cleaned up");
    fclose(dbg);
    return 0;
}
