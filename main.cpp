#include "rdna4.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"
#include "rdna4_engine.hpp"
#include "rdna4_kv_cache.hpp"
#include "engine/engine.hpp"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <chrono>

using namespace rdna4;

int main(int argc, char** argv) {
    printf("RDNA4 LLaMA Inference Engine\n");
    printf("============================\n\n");

    // Parse flags and positional arguments.
    bool clear_shaders = false;
    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--clear-shaders") {
            clear_shaders = true;
        } else {
            positionals.push_back(arg);
        }
    }

    if (positionals.size() < 2) {
        printf("Usage: rdna4_llama [--clear-shaders] <model.weights.json> <model.weights.bin> [prompt]\n");
        return 0;
    }

    std::string jsonPath = positionals[0];
    std::string binPath = positionals[1];
    std::string prompt = (positionals.size() > 2) ? positionals[2] : "Hello";

    VulkanContext ctx;
    if (!ctx.init()) { std::cerr << "Failed to init Vulkan\n"; return 1; }
    printf("Vulkan initialized (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    // ── Shader library ──
    std::string shader_dir = "shaders";
    {
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        fs::path cwd = fs::current_path();
        std::vector<fs::path> candidates = {
            exe_dir / "shaders",
            exe_dir / ".." / "shaders",
            exe_dir / ".." / ".." / "shaders",
            cwd / "shaders",
            cwd / ".." / "shaders",
            cwd / ".." / ".." / "shaders",
        };
        for (const auto& c : candidates) {
            fs::path canon = fs::weakly_canonical(c);
            if (fs::exists(canon / "rms_norm.comp")) {
                shader_dir = canon.string();
                break;
            }
        }
    }

    // Optional: force a fresh compile of all shaders by clearing the SPIR-V cache.
    if (clear_shaders) {
        namespace fs = std::filesystem;
        fs::path cache_dir = fs::path(shader_dir) / "cache";
        size_t removed = 0;
        if (fs::exists(cache_dir)) {
            for (const auto& entry : fs::directory_iterator(cache_dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".spv") {
                    fs::remove(entry.path());
                    ++removed;
                }
            }
        }
        fprintf(stderr, "[main] Cleared %zu cached SPIR-V files from %s\n",
                removed, cache_dir.string().c_str());
    }

    // ── Load model via ModelAdapter (streaming path) ──
    bool is_gguf = jsonPath.size() >= 5 && jsonPath.compare(jsonPath.size() - 5, 5, ".gguf") == 0;
    notllama::ModelAdapter model_adapter(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);
    bool loaded = is_gguf ? model_adapter.LoadFromGGUF(jsonPath)
                          : model_adapter.LoadFromPath(jsonPath, nullptr);
    if (!loaded) {
        std::cerr << "FATAL: Failed to load model from " << jsonPath << "\n";
        ctx.cleanup(); return 1;
    }

    printf("Model: %s | Layers: %zu | Heads: %zu/%zu | Embed: %zu\n",
           model_adapter.GetArchitecture().c_str(),
           model_adapter.GetNumLayers(),
           model_adapter.GetHeadDim() ? model_adapter.GetNumKVHeads() * model_adapter.GetHeadDim() / model_adapter.GetHeadDim() : 0,
           model_adapter.GetNumKVHeads(),
           model_adapter.GetHeadDim() * model_adapter.GetNumKVHeads());

    // Stream all layers to GPU (one at a time)
    for (uint32_t l = 0; l < model_adapter.GetNumLayers(); l++) {
        if (!model_adapter.StreamLayerWeights(l, nullptr)) {
            std::cerr << "FATAL: Failed to stream layer " << l << "\n";
            ctx.cleanup(); return 1;
        }
    }

    // Upload global tensors (embed, output_norm, output)
    if (!model_adapter.UploadGlobalWeights()) {
        std::cerr << "FATAL: Failed to upload global weights\n";
        ctx.cleanup(); return 1;
    }

    // ── KV cache ──
    uint32_t n_layers   = (uint32_t)model_adapter.GetNumLayers();
    uint32_t n_kv_heads = (uint32_t)model_adapter.GetNumKVHeads();
    uint32_t hd         = (uint32_t)model_adapter.GetHeadDim();
    rdna4::KVCacheManager kv_cache(ctx.device, ctx.physicalDevice, 2048, n_layers, n_kv_heads, hd);
    if (n_layers > 0) {
        if (!kv_cache.allocate()) {
            std::cerr << "FATAL: KV cache allocation failed\n";
            ctx.cleanup(); return 1;
        }
    }

    // ── Load tokenizer ──
    Tokenizer tokenizer;
    if (is_gguf) {
        tokenizer = model_adapter.GetTokenizer();
    } else {
        std::ifstream jf(jsonPath);
        nlohmann::json fullJson;
        if (jf.is_open()) { jf >> fullJson; if (fullJson.contains("tokenizer")) {
            WeightUploader tmp(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);
            tmp.loadTokenizer(tokenizer, fullJson["tokenizer"]);
        } }
    }
    printf("Tokenizer: %zu tokens\n", tokenizer.idToToken.size());

    // ── Encode prompt ──
    std::vector<uint32_t> input_ids = tokenizer.encode(prompt);
    if (input_ids.empty()) input_ids = {1};
    printf("Prompt '%s' -> %zu tokens\n", prompt.c_str(), input_ids.size());

    notllama::VendorProfile profile;
    profile.vendor = ctx.isAmd() ? notllama::VendorID::AMD : notllama::VendorID::UNKNOWN;
    profile.subgroup_size = ctx.subgroupSize;
    profile.wave32 = ctx.isWave32();
    profile.cooperative_matrix = false;

    notllama::VulkanShaderLibrary shader_lib(ctx.device, shader_dir, profile);
    notllama::SpecializationMap spec{};
    spec.subgroup_size = profile.subgroup_size;
    {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = shader_lib.PrecompileAll(spec);
        auto t1 = std::chrono::steady_clock::now();
        printf("Shader precompile: %s (%.2fs)\n", ok ? "OK" : "FAILED",
               std::chrono::duration<double>(t1 - t0).count());
    }

    // ── Engine ──
    {
    notllama::RingAllocatorAdapter allocator(ctx.device, ctx.physicalDevice,
                                              512 * 1024 * 1024,
                                              256 * 1024 * 1024,
                                              512 * 1024 * 1024);
    notllama::VulkanDescriptorManager desc_mgr(ctx.device, ctx.physicalDevice, &allocator);
    if (!desc_mgr.IsValid()) { std::cerr << "Descriptor manager init failed\n"; ctx.cleanup(); return 1; }
    shader_lib.SetBindlessSetLayout(desc_mgr.GetBindlessSetLayout());

    notllama::VulkanComputeEngine engine(ctx.device, ctx.physicalDevice, ctx.queues[0],
                                          ctx.queueFamilyIndex, &shader_lib, &desc_mgr,
                                           &allocator, static_cast<uint32_t>(model_adapter.GetHeadDim() * model_adapter.GetNumKVHeads()));
    engine.SetModel(&model_adapter);
    engine.SetKVCache(&kv_cache, 2048);

    // ── Pre-fill prompt (first token) ──
    if (!engine.AddSequence(0, input_ids)) {
        std::cerr << "Failed to add sequence\n"; ctx.cleanup(); return 1;
    }

    // ── Generation loop ──
    printf("\nGenerating...\n");
    fprintf(stderr, "[main] Starting inference loop\n");
    auto t0 = std::chrono::steady_clock::now();
    uint32_t tokens_generated = 0;
    const uint32_t max_tokens = 128;
    std::vector<uint32_t> output_ids;
    output_ids.reserve(max_tokens);
    uint32_t prev_token = UINT32_MAX;
    while (tokens_generated < max_tokens) {
        if (!engine.StepBatch()) {
            printf("[engine stopped at token %u]\n", tokens_generated);
            break;
        }
        uint32_t last_id = engine.LastTokenId();
        if (last_id != prev_token) {
            output_ids.push_back(last_id);
            tokens_generated++;
            prev_token = last_id;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("\nGenerated %u tokens in %.2fs (%.1f tok/s)\n",
           tokens_generated, elapsed, tokens_generated / elapsed);

    std::string output = tokenizer.decode(output_ids);
    printf("\n--- Output ---\n%s\n---\n", output.c_str());
    } // engine + allocator + desc_mgr destroyed here

    // ── Cleanup ──
    kv_cache.free();
    ctx.cleanup();
    return 0;
}
