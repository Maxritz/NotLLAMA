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

using namespace rdna4;

int main(int argc, char** argv) {
    std::cout << "RDNA4 LLaMA Inference Engine\n";
    std::cout << "=============================\n\n";

    VulkanContext ctx;
    if (!ctx.init()) {
        std::cerr << "Failed to initialize Vulkan\n";
        return 1;
    }
    std::cout << "Vulkan initialized with 4 ACE queues\n";

    Profiler profiler(ctx.device, ctx.physicalDevice);

    if (argc < 3) {
        std::cout << "Usage: rdna4_llama <model.weights.json> <model.weights.bin> [prompt] [--speculative N]\n";
        ctx.cleanup();
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

    std::ifstream jsonFile(jsonPath);
    nlohmann::json fullJson;
    jsonFile >> fullJson;

    profiler.beginCpu("load_weights");
    WeightUploader uploader(ctx.device, ctx.physicalDevice);
    ModelDesc model = uploader.load(jsonPath, binPath);
    profiler.endCpu("load_weights");

    std::cout << "\nModel: " << model.architecture
              << " | Blocks: " << model.blockCount
              << " | Embed: " << model.embeddingLength
              << " | Heads: " << model.headCount << "/" << model.headCountKv
              << " | Vocab: " << model.vocabSize
              << " | Tensors: " << model.tensors.size() << "\n";

    Tokenizer tokenizer;
    if (fullJson.contains("tokenizer")) {
        uploader.loadTokenizer(tokenizer, fullJson["tokenizer"]);
    }

    profiler.beginCpu("kv_cache_alloc");
    uint32_t headDim = model.embeddingLength / model.headCount;
    KVCacheManager kvCache(ctx.device, ctx.physicalDevice,
                           model.contextLength, model.blockCount,
                           model.headCountKv, headDim);
    if (!kvCache.allocate()) { std::cerr << "KV cache OOM\n"; return 1; }
    profiler.endCpu("kv_cache_alloc");

    profiler.beginCpu("ring_alloc");
    size_t ringSize = 512 * 1024 * 1024;
    RingAllocator allocator(ctx.device, ctx.physicalDevice, ringSize);
    profiler.endCpu("ring_alloc");

    profiler.beginCpu("create_pipelines");
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

    profiler.endCpu("create_pipelines");
    std::cout << "Pipelines loaded (including Flash Attention)\n";

    Scheduler scheduler(ctx.device, ctx.queues, ctx.queueFamilyIndex);

    profiler.beginCpu("inference");
    InferenceEngine engine(&ctx, &model, &kvCache, &pipelines, &tokenizer, &scheduler, &allocator);
    std::cout << "\nPrompt: \"" << prompt << "\"\n";
    if (useSpeculative) {
        std::cout << "Mode: Speculative decode (draft=" << nDraft << ")\n\n";
    } else {
        std::cout << "Mode: Standard autoregressive\n\n";
    }

    std::vector<uint32_t> tokens;
    if (useSpeculative) {
        tokens = engine.generateSpeculative(prompt, 32, nDraft);
    } else {
        tokens = engine.generate(prompt, 32);
    }
    profiler.endCpu("inference");

    std::cout << "\nGenerated (" << tokens.size() << " tokens): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    std::cout << "\nDecoded: \"" << tokenizer.decode(tokens) << "\"\n";

    profiler.report();

    pipelines.cleanup();
    kvCache.free();
    ctx.cleanup();
    return 0;
}
