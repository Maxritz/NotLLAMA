#include "cli_parser.hpp"
#include "multi_model_manager.hpp"
#include "model_router.hpp"
#include "mtp_engine.hpp"
#include "web_server.hpp"
#include "engine/engine.hpp"
#include "rdna4_vulkan.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <chrono>
#include <cstring>
#include <signal.h>

using namespace notllama;

static std::atomic<bool> g_running{true};
static void SignalHandler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        fprintf(stderr, "\n[main] Shutdown signal received\n");
        g_running = false;
    }
}

// Print colored text
static void PrintColor(const char* text, const char* color_code) {
    fprintf(stderr, "%s%s\033[0m", color_code, text);
}

static std::string ReadPromptFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

static void PrintBanner() {
    fprintf(stderr,
        "\n"
        "+=============================================+\n"
        "|  NotLLAMA v0.2.0 - Vulkan Compute LLM       |\n"
        "|  Multi-Model | MTP | Web API | llama.cpp CLI|\n"
        "+=============================================+\n"
        "\n");
}

static void RunInteractiveMode(MultiModelManager* mgr, ModelRouter* router,
                                MTPEngine* mtp, const CLIOptions& opts) {
    fprintf(stderr, "\n=== Interactive Mode ===\n");
    fprintf(stderr, "Type 'quit' or 'exit' to stop\n");
    fprintf(stderr, "Type '/models' to list loaded models\n");
    fprintf(stderr, "Type '/switch <model_id>' to change model\n");
    fprintf(stderr, "Type '/route <prompt>' to see routing decision\n");
    fprintf(stderr, "Type '/clear' to clear context\n");
    fprintf(stderr, "========================\n\n");

    auto models = mgr->ListModels();
    std::string current_model = models.empty() ? "" : models[0];
    std::vector<uint32_t> context_tokens;

    while (g_running) {
        if (opts.color) PrintColor("\nYou: ", "\033[1;36m");
        else fprintf(stderr, "\nYou: ");

        std::string input;
        if (opts.multiline_input) {
            // Multi-line: read until empty line
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) break;
                if (!input.empty()) input += "\n";
                input += line;
            }
        } else {
            if (!std::getline(std::cin, input)) break;
        }

        // Commands
        if (input == "quit" || input == "exit") break;
        if (input == "/clear") {
            context_tokens.clear();
            fprintf(stderr, "[Context cleared]\n");
            continue;
        }
        if (input == "/models") {
            fprintf(stderr, "Loaded models:\n");
            for (const auto& id : mgr->ListModels()) {
                auto* inst = mgr->GetModel(id);
                fprintf(stderr, "  %s - %s (tags: %s)\n",
                        id.c_str(), inst ? inst->name.c_str() : "?",
                        inst ? inst->tags.c_str() : "");
            }
            continue;
        }
        if (input.starts_with("/switch ")) {
            std::string new_id = input.substr(8);
            if (mgr->IsLoaded(new_id)) {
                current_model = new_id;
                context_tokens.clear();
                fprintf(stderr, "[Switched to model: %s]\n", new_id.c_str());
            } else {
                fprintf(stderr, "[Model not found: %s]\n", new_id.c_str());
            }
            continue;
        }
        if (input.starts_with("/route ")) {
            std::string prompt = input.substr(7);
            auto decision = router->Route(prompt);
            fprintf(stderr, "[Router] Primary: %s (confidence: %.2f)\n",
                    decision.primary_model_id.c_str(), decision.confidence);
            fprintf(stderr, "[Router] Reason: %s\n", decision.reason.c_str());
            continue;
        }
        if (input.empty()) continue;

        // Route to best model
        auto decision = router->Route(input);
        std::string model_id = current_model.empty() ? decision.primary_model_id : current_model;

        if (model_id.empty()) {
            fprintf(stderr, "[Error: No model loaded]\n");
            continue;
        }

        auto* inst = mgr->GetModel(model_id);
        if (!inst) {
            fprintf(stderr, "[Error: Model '%s' not available]\n", model_id.c_str());
            continue;
        }

        if (!decision.reason.empty() && model_id != current_model) {
            fprintf(stderr, "[Routed to '%s' - %s]\n", model_id.c_str(), decision.reason.c_str());
        }

        // Build prompt with system prompt
        std::string full_prompt = opts.system_prompt + "\n" + input + "\n";

        // Tokenize prompt (char-level for now)
        std::vector<uint32_t> prompt_tokens;
        for (char c : full_prompt) {
            prompt_tokens.push_back(static_cast<uint8_t>(c));
        }

        // Set MTP config
        MTPConfig mtp_cfg;
        mtp_cfg.n_predict = static_cast<uint32_t>(opts.n_predict > 0 ? opts.n_predict : 256);
        mtp_cfg.use_draft_model = !opts.draft_model.empty();
        mtp_cfg.draft_model_id = opts.draft_model;
        mtp_cfg.n_draft = opts.enable_mtp ? opts.mtp_n_draft : 1;
        mtp->SetConfig(mtp_cfg);
        mtp->SetSeed(opts.seed);

        // Generate
        uint32_t n_predict = opts.n_predict > 0 ? static_cast<uint32_t>(opts.n_predict) : 256;
        auto start = std::chrono::steady_clock::now();
        auto generated = mtp->Generate(model_id, prompt_tokens, n_predict);
        auto end = std::chrono::steady_clock::now();

        // Convert tokens to text
        std::string response_text;
        const auto& tok = inst->adapter->GetTokenizer();
        if (!tok.idToToken.empty()) {
            for (uint32_t t : generated) {
                if (t < tok.idToToken.size()) response_text += tok.idToToken[t];
                else response_text += "?";
            }
        } else {
            for (uint32_t t : generated) {
                if (t < 256) response_text += static_cast<char>(t);
            }
        }

        // Print response
        if (opts.color) PrintColor("\nAssistant: ", "\033[1;32m");
        else fprintf(stderr, "\nAssistant: ");
        fprintf(stderr, "%s\n", response_text.c_str());

        // Update context
        for (uint32_t t : prompt_tokens) context_tokens.push_back(t);
        for (uint32_t t : generated) context_tokens.push_back(t);

        // Context window management
        if (context_tokens.size() > static_cast<size_t>(opts.ctx_size)) {
            size_t keep = opts.ctx_size / 2;
            context_tokens = std::vector<uint32_t>(
                context_tokens.end() - keep, context_tokens.end());
        }

        // Stats
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double tok_per_sec = (elapsed_ms > 0) ? (generated.size() * 1000.0 / elapsed_ms) : 0;
        fprintf(stderr, "  [%zu tokens, %.1f ms, %.1f tok/s]\n",
                generated.size(), elapsed_ms, tok_per_sec);
    }
}

static void RunSingleShot(MultiModelManager* mgr, ModelRouter* router,
                          MTPEngine* mtp, const CLIOptions& opts) {
    if (opts.prompt.empty()) {
        fprintf(stderr, "[Error: No prompt specified. Use -p/--prompt or -i for interactive mode]\n");
        return;
    }

    if (opts.verbose_prompt && !opts.no_display_prompt) {
        fprintf(stderr, "Prompt:\n%s\n\n", opts.prompt.c_str());
    }

    // Route to best model
    auto decision = router->Route(opts.prompt);
    std::string model_id = decision.primary_model_id;
    if (model_id.empty()) {
        fprintf(stderr, "[Error: No models loaded]\n");
        return;
    }

    if (!decision.reason.empty()) {
        fprintf(stderr, "[Router] %s\n", decision.reason.c_str());
    }

    auto* inst = mgr->GetModel(model_id);
    if (!inst) {
        fprintf(stderr, "[Error: Model '%s' not available]\n", model_id.c_str());
        return;
    }

    // Tokenize
    std::vector<uint32_t> prompt_tokens;
    for (char c : opts.prompt) {
        prompt_tokens.push_back(static_cast<uint8_t>(c));
    }

    // Configure MTP
    MTPConfig mtp_cfg;
    mtp_cfg.n_predict = static_cast<uint32_t>(opts.n_predict > 0 ? opts.n_predict : 128);
    mtp_cfg.temperature = opts.temperature;
    mtp_cfg.use_draft_model = !opts.draft_model.empty();
    mtp_cfg.draft_model_id = opts.draft_model;
    mtp_cfg.n_draft = opts.enable_mtp ? opts.mtp_n_draft : 1;
    mtp->SetConfig(mtp_cfg);
    mtp->SetSeed(opts.seed);

    // Generate
    auto start = std::chrono::steady_clock::now();
    auto generated = mtp->Generate(model_id, prompt_tokens,
                                     opts.n_predict > 0 ? static_cast<uint32_t>(opts.n_predict) : 128);
    auto end = std::chrono::steady_clock::now();

    // Convert to text
    std::string output;
    const auto& tok = inst->adapter->GetTokenizer();
    if (!tok.idToToken.empty()) {
        for (uint32_t t : generated) {
            if (t < tok.idToToken.size()) output += tok.idToToken[t];
            else output += "?";
        }
    } else {
        for (uint32_t t : generated) {
            if (t < 256) output += static_cast<char>(t);
        }
    }

    // Output
    if (opts.output_format == "json") {
        nlohmann::json j;
        j["content"] = output;
        j["tokens_generated"] = generated.size();
        j["model"] = model_id;
        auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        j["generation_time_ms"] = elapsed_ms;
        j["tokens_per_second"] = (elapsed_ms > 0) ? (generated.size() * 1000.0 / elapsed_ms) : 0;
        std::cout << j.dump(2) << std::endl;
    } else {
        if (!opts.no_display_prompt) {
            fprintf(stderr, "%s", output.c_str());
        } else {
            printf("%s", output.c_str());
        }
        printf("\n");
    }
}

static void RunBenchmark(MultiModelManager* mgr, MTPEngine* mtp, const CLIOptions& opts) {
    fprintf(stderr, "\n=== Benchmark Mode ===\n");
    auto models = mgr->ListModels();
    if (models.empty()) {
        fprintf(stderr, "[Error: No models loaded for benchmark]\n");
        return;
    }

    std::string model_id = models[0];
    auto* inst = mgr->GetModel(model_id);
    if (!inst) return;

    // Warm-up
    fprintf(stderr, "Warming up...\n");
    std::vector<uint32_t> warmup(32, 1); // 32 BOS tokens
    mtp->Generate(model_id, warmup, 16);

    // Benchmark
    fprintf(stderr, "\nRunning %d iterations...\n", opts.benchmark_iterations);
    std::vector<double> latencies;
    latencies.reserve(opts.benchmark_iterations);

    std::vector<uint32_t> prompt_tokens;
    if (opts.prompt_benchmark) {
        // Benchmark prompt processing (prefill)
        for (int i = 0; i < 256; i++) prompt_tokens.push_back((i % 256) + 1);
    } else {
        // Benchmark token generation (decode)
        prompt_tokens.push_back(1); // BOS
    }

    MTPConfig mtp_cfg;
    mtp_cfg.n_predict = 1;
    mtp->SetConfig(mtp_cfg);

    for (int i = 0; i < opts.benchmark_iterations; i++) {
        auto start = std::chrono::steady_clock::now();
        auto generated = mtp->Generate(model_id, prompt_tokens,
            opts.prompt_benchmark ? 1 : 64);
        auto end = std::chrono::steady_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double tok_per_sec = (elapsed_ms > 0) ? (generated.size() * 1000.0 / elapsed_ms) : 0;
        latencies.push_back(elapsed_ms);

        fprintf(stderr, "  Iter %d: %.1f ms, %.1f tok/s\n", i + 1, elapsed_ms, tok_per_sec);
    }

    // Statistics
    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double min_lat = latencies.front();
    double max_lat = latencies.back();
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    fprintf(stderr, "\n--- Results ---\n");
    fprintf(stderr, "Model: %s\n", model_id.c_str());
    fprintf(stderr, "Iterations: %d\n", opts.benchmark_iterations);
    fprintf(stderr, "Avg latency: %.1f ms\n", avg);
    fprintf(stderr, "Median: %.1f ms\n", median);
    fprintf(stderr, "Min: %.1f ms\n", min_lat);
    fprintf(stderr, "Max: %.1f ms\n", max_lat);
    fprintf(stderr, "Avg throughput: %.1f tok/s\n",
            (avg > 0) ? (1000.0 / avg) : 0);
}

int main(int argc, char** argv) {
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);

    // Parse CLI
    CLIOptions opts;
    if (!CLIParser::Parse(argc, argv, opts)) {
        if (opts.show_help) {
            CLIParser::PrintHelp(argv[0]);
            return 0;
        }
        if (opts.show_version) {
            CLIParser::PrintVersion();
            return 0;
        }
        return 1;
    }

    PrintBanner();

    // Show parsed options
    fprintf(stderr, "[Config] Models: %zu | Router: %s | MTP: %s | Server: %s\n",
            opts.model_paths.size(),
            opts.router_mode.c_str(),
            opts.enable_mtp ? "enabled" : "disabled",
            opts.server_mode ? "enabled" : "disabled");

    // Initialize Vulkan
    fprintf(stderr, "[Init] Initializing Vulkan...\n");
    rdna4::VulkanContext ctx;
    if (!ctx.init()) {
        fprintf(stderr, "[FATAL] Vulkan initialization failed\n");
        return 1;
    }
    fprintf(stderr, "[Init] Vulkan OK (wave32=%d, subgroup=%u)\n",
            ctx.isWave32() ? 1 : 0, ctx.subgroupSize);

    // Find shader directory
    std::string shader_dir = "shaders";
    {
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        std::vector<fs::path> candidates = {
            exe_dir / "shaders",
            exe_dir / ".." / "shaders",
            fs::current_path() / "shaders",
        };
        for (const auto& c : candidates) {
            if (fs::exists(c / "rms_norm.comp")) {
                shader_dir = fs::weakly_canonical(c).string();
                break;
            }
        }
    }
    fprintf(stderr, "[Init] Shader dir: %s\n", shader_dir.c_str());

    // Create engine infrastructure
    VulkanShaderLibrary shader_lib(ctx.device, shader_dir,
        VulkanDevice(&ctx).GetVendorProfile());

    RingAllocatorAdapter allocator(ctx.device, ctx.physicalDevice,
        256 * 1024 * 1024,   // weight ring
        256 * 1024 * 1024,   // transient ring
        256 * 1024 * 1024);  // KV ring

    VulkanDescriptorManager desc_mgr(ctx.device, ctx.physicalDevice, &allocator);

    // Create multi-model manager
    MultiModelManager model_mgr(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex,
                                 &shader_lib, &desc_mgr, &allocator);

    // Load all models specified on command line
    for (size_t i = 0; i < opts.model_paths.size(); i++) {
        std::string path = opts.model_paths[i];
        std::string id = (i < opts.model_ids.size()) ? opts.model_ids[i] : "";
        std::string tags = (i < opts.model_tags.size()) ? opts.model_tags[i] : "";

        // Derive ID from filename if not specified
        if (id.empty()) {
            size_t last_slash = path.find_last_of("/\\");
            size_t last_dot = path.find_last_of('.');
            if (last_dot == std::string::npos || last_dot < last_slash)
                last_dot = path.length();
            id = path.substr(last_slash == std::string::npos ? 0 : last_slash + 1,
                             last_dot - (last_slash == std::string::npos ? 0 : last_slash + 1));
        }

        uint32_t gpu_layers = (opts.gpu_layers < 0) ? (uint32_t)-1 : static_cast<uint32_t>(opts.gpu_layers);

        std::string loaded_id = model_mgr.LoadModel(path, id, id, tags,
                                                      gpu_layers,
                                                      static_cast<size_t>(opts.ctx_size),
                                                      opts.temperature,
                                                      opts.top_p,
                                                      opts.top_k);
        if (loaded_id.empty()) {
            fprintf(stderr, "[WARN] Failed to load model: %s\n", path.c_str());
        }
    }

    if (model_mgr.Count() == 0) {
        fprintf(stderr, "[WARN] No models loaded. Specify with -m/--model\n");
        if (!opts.server_mode && opts.prompt.empty()) {
            fprintf(stderr, "[FATAL] No models and no prompt. Use --help for usage.\n");
            ctx.cleanup();
            return 1;
        }
    }

    // Create router
    ModelRouter router(&model_mgr);
    if (opts.router_mode == "ensemble") router.SetMode(ModelRouter::Mode::ENSEMBLE);
    else if (opts.router_mode == "cascade") router.SetMode(ModelRouter::Mode::CASCADE);

    // Auto-detect capabilities for all loaded models
    for (const auto& id : model_mgr.ListModels()) {
        router.AutoDetectCapabilities(id);
    }

    // Create MTP engine
    MTPEngine mtp(&model_mgr);
    MTPConfig mtp_cfg;
    mtp_cfg.n_draft = opts.mtp_n_draft;
    mtp_cfg.use_draft_model = !opts.draft_model.empty();
    mtp_cfg.draft_model_id = opts.draft_model;
    mtp.SetConfig(mtp_cfg);
    mtp.SetSeed(opts.seed);

    // Server mode
    if (opts.server_mode) {
        fprintf(stderr, "\n[Server] Starting HTTP API server...\n");
        WebServer server(&model_mgr, &router, &mtp,
                          opts.host, opts.port, opts.timeout);
        if (!opts.api_key.empty()) server.SetApiKey(opts.api_key);
        server.EnableEmbeddings(opts.embedding_endpoint);
        server.EnableReranking(opts.reranking_endpoint);

        if (!server.Start()) {
            fprintf(stderr, "[FATAL] Failed to start server\n");
            ctx.cleanup();
            return 1;
        }

        fprintf(stderr, "[Server] Ready at http://%s:%d\n", opts.host.c_str(), opts.port);
        fprintf(stderr, "[Server] Endpoints:\n");
        fprintf(stderr, "  POST /v1/completions\n");
        fprintf(stderr, "  POST /v1/chat/completions\n");
        fprintf(stderr, "  GET  /v1/models\n");
        fprintf(stderr, "  GET  /health\n");
        if (opts.embedding_endpoint) fprintf(stderr, "  POST /v1/embeddings\n");
        fprintf(stderr, "  POST /tokenize\n");
        fprintf(stderr, "  POST /detokenize\n");
        fprintf(stderr, "\nPress Ctrl+C to stop\n");

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        server.Stop();
        ctx.cleanup();
        return 0;
    }

    // Benchmark mode
    if (opts.benchmark) {
        RunBenchmark(&model_mgr, &mtp, opts);
        ctx.cleanup();
        return 0;
    }

    // Interactive or single-shot mode
    if (opts.interactive) {
        RunInteractiveMode(&model_mgr, &router, &mtp, opts);
    } else {
        RunSingleShot(&model_mgr, &router, &mtp, opts);
    }

    fprintf(stderr, "\n[main] Cleanup...\n");
    ctx.cleanup();
    fprintf(stderr, "[main] Done.\n");
    return 0;
}
