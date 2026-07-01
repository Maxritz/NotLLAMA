#include "cli_parser.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <random>
#include <string>

namespace notllama {

bool CLIParser::ParseInt(const char* s, int32_t& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int32_t>(v);
    return true;
}

bool CLIParser::ParseUInt(const char* s, uint32_t& out) {
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<uint32_t>(v);
    return true;
}

bool CLIParser::ParseFloat(const char* s, float& out) {
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (end == s || *end != '\0') return false;
    out = v;
    return true;
}

bool CLIParser::ParseStringList(const char* s, std::vector<std::string>& out) {
    std::string str(s);
    size_t start = 0;
    size_t end = str.find(',');
    while (end != std::string::npos) {
        out.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(',', start);
    }
    out.push_back(str.substr(start));
    return true;
}

bool CLIParser::Parse(int argc, char** argv, CLIOptions& out) {
    // First pass: collect all --model entries and their metadata
    struct ModelEntry {
        std::string path;
        std::string id;
        std::string tags;
    };
    std::vector<ModelEntry> model_entries;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            ModelEntry entry;
            entry.path = argv[++i];
            // Default ID from filename
            size_t last_slash = entry.path.find_last_of("/\\");
            size_t last_dot = entry.path.find_last_of('.');
            if (last_dot == std::string::npos || last_dot < last_slash)
                last_dot = entry.path.length();
            entry.id = entry.path.substr(
                last_slash == std::string::npos ? 0 : last_slash + 1,
                last_dot - (last_slash == std::string::npos ? 0 : last_slash + 1));
            model_entries.push_back(entry);
        }
    }

    // Second pass: parse all arguments
    size_t current_model_idx = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // --- Help / Version ---
        if (arg == "--help" || arg == "-h") { out.show_help = true; return false; }
        if (arg == "--version" || arg == "-v") { out.show_version = true; return false; }

        // --- Model options ---
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            out.model_paths.push_back(argv[++i]);
            // Assign corresponding ID and tags if available
            if (current_model_idx < out.model_ids.size()) current_model_idx++;
        }
        else if (arg == "--model-id" && i + 1 < argc) {
            out.model_ids.push_back(argv[++i]);
            if (current_model_idx < out.model_paths.size()) current_model_idx++;
        }
        else if (arg == "--model-tags" && i + 1 < argc) {
            out.model_tags.push_back(argv[++i]);
        }
        else if (arg == "--draft-model" && i + 1 < argc) {
            out.draft_model = argv[++i];
        }
        else if ((arg == "--gpu-layers" || arg == "-ngl") && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.gpu_layers)) {
                fprintf(stderr, "Error: Invalid --gpu-layers value\n"); return false;
            }
        }

        // --- Prompt options ---
        else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            out.prompt = argv[++i];
        }
        else if ((arg == "--file" || arg == "-f") && i + 1 < argc) {
            out.prompt_file = argv[++i];
        }
        else if (arg == "--system-prompt" && i + 1 < argc) {
            out.system_prompt = argv[++i];
        }
        else if (arg == "--chat-template" && i + 1 < argc) {
            out.chat_template = argv[++i];
        }
        else if (arg == "--interactive" || arg == "-i") {
            out.interactive = true;
        }
        else if (arg == "--multiline-input") {
            out.multiline_input = true;
        }

        // --- Generation options ---
        else if ((arg == "--n-predict" || arg == "-n") && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.n_predict)) {
                fprintf(stderr, "Error: Invalid --n-predict value\n"); return false;
            }
        }
        else if ((arg == "--ctx-size" || arg == "-c") && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.ctx_size)) {
                fprintf(stderr, "Error: Invalid --ctx-size value\n"); return false;
            }
        }
        else if ((arg == "--batch-size" || arg == "-b") && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.batch_size)) {
                fprintf(stderr, "Error: Invalid --batch-size value\n"); return false;
            }
        }
        else if (arg == "--ubatch-size" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.ubatch_size)) {
                fprintf(stderr, "Error: Invalid --ubatch-size value\n"); return false;
            }
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.temperature)) {
                fprintf(stderr, "Error: Invalid --temperature value\n"); return false;
            }
        }
        else if (arg == "--top-p" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.top_p)) {
                fprintf(stderr, "Error: Invalid --top-p value\n"); return false;
            }
        }
        else if (arg == "--top-k" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.top_k)) {
                fprintf(stderr, "Error: Invalid --top-k value\n"); return false;
            }
        }
        else if (arg == "--repeat-penalty" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.repeat_penalty)) {
                fprintf(stderr, "Error: Invalid --repeat-penalty value\n"); return false;
            }
        }
        else if (arg == "--repeat-last-n" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.repeat_last_n)) {
                fprintf(stderr, "Error: Invalid --repeat-last-n value\n"); return false;
            }
        }
        else if (arg == "--frequency-penalty" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.frequency_penalty)) {
                fprintf(stderr, "Error: Invalid --frequency-penalty value\n"); return false;
            }
        }
        else if (arg == "--presence-penalty" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.presence_penalty)) {
                fprintf(stderr, "Error: Invalid --presence-penalty value\n"); return false;
            }
        }
        else if (arg == "--sampler-seq" && i + 1 < argc) {
            out.sampler = argv[++i];
        }
        else if (arg == "--min-p" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.min_p)) {
                fprintf(stderr, "Error: Invalid --min-p value\n"); return false;
            }
        }
        else if (arg == "--dynatemp-range" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.dynatemp_range)) {
                fprintf(stderr, "Error: Invalid --dynatemp-range value\n"); return false;
            }
        }
        else if (arg == "--dynatemp-exponent" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.dynatemp_exponent)) {
                fprintf(stderr, "Error: Invalid --dynatemp-exponent value\n"); return false;
            }
        }
        else if (arg == "--mirostat" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.mirostat)) {
                fprintf(stderr, "Error: Invalid --mirostat value\n"); return false;
            }
        }
        else if (arg == "--mirostat-tau" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.mirostat_tau)) {
                fprintf(stderr, "Error: Invalid --mirostat-tau value\n"); return false;
            }
        }
        else if (arg == "--mirostat-eta" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], out.mirostat_eta)) {
                fprintf(stderr, "Error: Invalid --mirostat-eta value\n"); return false;
            }
        }
        else if (arg == "--seed" && i + 1 < argc) {
            if (!ParseUInt(argv[++i], out.seed)) {
                fprintf(stderr, "Error: Invalid --seed value\n"); return false;
            }
        }

        // --- Multi-model / routing ---
        else if (arg == "--router-mode" && i + 1 < argc) {
            out.router_mode = argv[++i];
        }
        else if (arg == "--mtp") {
            out.enable_mtp = true;
        }
        else if (arg == "--mtp-n-draft" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.mtp_n_draft)) {
                fprintf(stderr, "Error: Invalid --mtp-n-draft value\n"); return false;
            }
        }

        // --- Server options ---
        else if (arg == "--server") {
            out.server_mode = true;
        }
        else if (arg == "--host" && i + 1 < argc) {
            out.host = argv[++i];
        }
        else if (arg == "--port" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.port)) {
                fprintf(stderr, "Error: Invalid --port value\n"); return false;
            }
        }
        else if (arg == "--timeout" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.timeout)) {
                fprintf(stderr, "Error: Invalid --timeout value\n"); return false;
            }
        }
        else if (arg == "--api-key" && i + 1 < argc) {
            out.api_key = argv[++i];
        }
        else if (arg == "--embeddings") {
            out.embedding_endpoint = true;
        }
        else if (arg == "--reranking") {
            out.reranking_endpoint = true;
        }

        // --- Formatting ---
        else if (arg == "--verbose-prompt") {
            out.verbose_prompt = true;
        }
        else if (arg == "--no-display-prompt") {
            out.no_display_prompt = true;
        }
        else if (arg == "--log-disable") {
            out.log_disable = true;
        }
        else if (arg == "--log-file" && i + 1 < argc) {
            out.log_file = argv[++i];
        }
        else if (arg == "--color") {
            out.color = true;
        }
        else if (arg == "--no-color") {
            out.color = false;
        }
        else if (arg == "--special") {
            out.special = true;
        }
        else if (arg == "--output-format" && i + 1 < argc) {
            out.output_format = argv[++i];
        }

        // --- Performance ---
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.threads)) {
                fprintf(stderr, "Error: Invalid --threads value\n"); return false;
            }
        }
        else if (arg == "--threads-batch" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.threads_batch)) {
                fprintf(stderr, "Error: Invalid --threads-batch value\n"); return false;
            }
        }
        else if (arg == "--flash-attn") {
            out.flash_attn = true;
        }
        else if (arg == "--no-flash-attn") {
            out.flash_attn = false;
        }
        else if (arg == "--no-kv-offload") {
            out.no_kv_offload = true;
        }
        else if (arg == "--tensor-split" && i + 1 < argc) {
            out.tensor_split = argv[++i];
        }
        else if (arg == "--main-gpu" && i + 1 < argc) {
            out.main_gpu = argv[++i];
        }

        // --- Benchmark ---
        else if (arg == "--benchmark") {
            out.benchmark = true;
        }
        else if (arg == "--benchmark-iterations" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.benchmark_iterations)) {
                fprintf(stderr, "Error: Invalid --benchmark-iterations value\n"); return false;
            }
        }
        else if (arg == "--prompt-benchmark") {
            out.prompt_benchmark = true;
        }

        // --- Distributed Agent ---
        else if (arg == "--agent-name" && i + 1 < argc) {
            out.agent_name = argv[++i];
        }
        else if (arg == "--agent-port" && i + 1 < argc) {
            if (!ParseInt(argv[++i], out.agent_port)) {
                fprintf(stderr, "Error: Invalid --agent-port value\n"); return false;
            }
        }
        else if (arg == "--agent-peer" && i + 1 < argc) {
            out.agent_peers.push_back(argv[++i]);
        }
        else if (arg == "--enable-reason-sharing") {
            out.enable_reason_sharing = true;
        }
        else if (arg == "--enable-model-distill") {
            out.enable_model_distill = true;
        }

        // --- External tool awareness ---
        else if (arg == "--use-graphify") {
            out.use_graphify = true;
        }
        else if (arg == "--graphify-url" && i + 1 < argc) {
            out.use_graphify = true;
            out.graphify_url = argv[++i];
        }
        else if (arg == "--use-mcp") {
            out.use_mcp = true;
        }
        else if (arg == "--mcp-url" && i + 1 < argc) {
            out.use_mcp = true;
            out.mcp_url = argv[++i];
        }

        // --- Model creation ---
        else if (arg == "--create-model" && i + 1 < argc) {
            out.create_model_name = argv[++i];
        }
        else if (arg == "--create-model-type" && i + 1 < argc) {
            out.create_model_type = argv[++i];
        }
        else if (arg == "--create-model-size" && i + 1 < argc) {
            char* end = nullptr;
            out.create_model_size_mb = std::strtoul(argv[++i], &end, 10);
        }
        else if (arg == "--create-model-quant" && i + 1 < argc) {
            out.create_model_quant = argv[++i];
        }
        else if (arg == "--create-model-arch" && i + 1 < argc) {
            out.create_model_arch = argv[++i];
        }

        // --- Unknown ---
        else if (arg.starts_with("--")) {
            // Store unknown args for forward compatibility
            std::string key = arg.substr(2);
            if (i + 1 < argc && !std::string(argv[i + 1]).starts_with("-")) {
                out.extra[key] = argv[++i];
            } else {
                out.extra[key] = "true";
            }
        }
        else {
            // Positional argument: prompt
            if (out.prompt.empty()) {
                out.prompt = arg;
            }
        }
    }

    // Validate: if --file specified, read prompt from file
    if (!out.prompt_file.empty()) {
        FILE* f = fopen(out.prompt_file.c_str(), "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            out.prompt.resize(size);
            fread(out.prompt.data(), 1, size, f);
            fclose(f);
        } else {
            fprintf(stderr, "Warning: Cannot read prompt file: %s\n", out.prompt_file.c_str());
        }
    }

    // Default seed: random if 0xFFFFFFFF
    if (out.seed == 0xFFFFFFFF) {
        out.seed = static_cast<uint32_t>(std::random_device{}());
    }

    return true;
}

void CLIParser::PrintHelp(const char* program_name) {
    fprintf(stderr,
    "NotLLAMA - Vulkan Compute LLM Inference Engine\n"
    "\n"
    "Usage: %s [options]\n"
    "\n"
    "MODEL OPTIONS:\n"
    "  -m, --model FNAME         Model path (can specify multiple times)\n"
    "      --model-id ID         Label for the model (default: filename)\n"
    "      --model-tags TAGS     Comma-separated tags (e.g., \"code,python\")\n"
    "      --draft-model FNAME   Draft model for speculative decoding\n"
    "  -ngl, --gpu-layers N      GPU layers to offload (-1 = all)\n"
    "\n"
    "PROMPT OPTIONS:\n"
    "  -p, --prompt PROMPT       Prompt text\n"
    "  -f, --file FNAME          Read prompt from file\n"
    "      --system-prompt TEXT  System prompt for chat mode\n"
    "      --chat-template NAME  Chat template name\n"
    "  -i, --interactive         Interactive mode\n"
    "      --multiline-input     Allow multiline input\n"
    "\n"
    "GENERATION OPTIONS:\n"
    "  -n, --n-predict N         Tokens to predict (-1 = infinite)\n"
    "  -c, --ctx-size N          Context window size (default: 4096)\n"
    "  -b, --batch-size N        Batch size (default: 2048)\n"
    "      --ubatch-size N       Micro-batch size (default: 512)\n"
    "      --temperature T       Sampling temperature (default: 0.8)\n"
    "      --top-p P             Nucleus sampling (default: 0.95)\n"
    "      --top-k K             Top-k sampling (default: 40)\n"
    "      --repeat-penalty P    Repeat penalty (default: 1.1)\n"
    "      --repeat-last-n N     Penalty window (default: 64)\n"
    "      --frequency-penalty P (default: 0.0)\n"
    "      --presence-penalty P  (default: 0.0)\n"
    "      --seed N              RNG seed (default: random)\n"
    "      --mirostat N          Mirostat mode (0=off, 1/2) (default: 0)\n"
    "      --mirostat-tau T      (default: 5.0)\n"
    "      --mirostat-eta E      (default: 0.1)\n"
    "      --dynatemp-range R    Dynamic temperature range\n"
    "      --dynatemp-exponent E (default: 1.0)\n"
    "      --min-p N             Minimum probability threshold\n"
    "\n"
    "MULTI-MODEL OPTIONS:\n"
    "      --router-mode MODE    single|ensemble|cascade (default: single)\n"
    "      --mtp                 Enable multi-token prediction\n"
    "      --mtp-n-draft N       Draft tokens per step (default: 4)\n"
    "\n"
    "SERVER OPTIONS:\n"
    "      --server              Run HTTP API server\n"
    "      --host ADDR           Bind address (default: 127.0.0.1)\n"
    "      --port PORT           Port (default: 8080)\n"
    "      --timeout SEC         Request timeout (default: 600)\n"
    "      --api-key KEY         API key for authentication\n"
    "      --embeddings          Enable embeddings endpoint\n"
    "      --reranking           Enable reranking endpoint\n"
    "\n"
    "FORMAT OPTIONS:\n"
    "      --verbose-prompt      Print prompt before generation\n"
    "      --no-display-prompt   Do not echo prompt\n"
    "      --color               Enable colored output (default)\n"
    "      --no-color            Disable colored output\n"
    "      --special             Show special tokens\n"
    "      --output-format FMT   text|json (default: text)\n"
    "\n"
    "PERFORMANCE OPTIONS:\n"
    "  -t, --threads N           CPU threads (-1 = auto)\n"
    "      --threads-batch N     Batch processing threads\n"
    "      --flash-attn          Enable Flash Attention (default)\n"
    "      --no-flash-attn       Disable Flash Attention\n"
    "      --no-kv-offload       Keep KV cache on CPU\n"
    "\n"
    "BENCHMARK OPTIONS:\n"
    "      --benchmark           Run benchmark\n"
    "      --benchmark-iterations N (default: 10)\n"
    "      --prompt-benchmark    Benchmark prompt processing\n"
    "\n"
    "DISTRIBUTED AGENT OPTIONS:\n"
    "      --agent-name NAME     Unique name for this node (default: notllama-agent)\n"
    "      --agent-port PORT     Enable agent listener on this port (0 = disabled)\n"
    "      --agent-peer HOST,PORT,NAME,TAGS  Add a peer agent (repeatable)\n"
    "      --enable-reason-sharing  Allow agents to share reasoning\n"
    "      --enable-model-distill   Allow cross-agent model distillation\n"
    "\n"
    "EXTERNAL TOOL AWARENESS:\n"
    "      --use-graphify        Enable Graphify integration\n"
    "      --graphify-url URL    Graphify endpoint URL\n"
    "      --use-mcp             Enable MCP integration\n"
    "      --mcp-url URL         MCP server URL\n"
    "\n"
    "MODEL CREATION:\n"
    "      --create-model NAME   Create a new model (requires --create-model-type)\n"
    "      --create-model-type   Base model type (llama, qwen, gemma)\n"
    "      --create-model-size N Target size in MB\n"
    "      --create-model-quant  Quantization (Q4_K, Q6_K, Q8_0)\n"
    "      --create-model-arch   Architecture: dense, moe, mixed\n"
    "\n"
    "OTHER:\n"
    "  -h, --help                Show this help\n"
    "  -v, --version             Show version\n"
    "\n"
    "EXAMPLES:\n"
    "  # Single model inference\n"
    "  %s -m model.gguf -p \"Hello world\" -n 128\n"
    "\n"
    "  # Multi-model with routing\n"
    "  %s -m code_model.gguf --model-id coder --model-tags code,python \\\n"
    "     -m chat_model.gguf --model-id chatbot --model-tags english,chat \\\n"
    "     --router-mode single -p \"Write a Python function\" -n 256\n"
    "\n"
    "  # Server mode\n"
    "  %s -m model.gguf --server --host 0.0.0.0 --port 8080\n"
    "\n"
    "  # Interactive chat with MTP\n"
    "  %s -m model.gguf -i --mtp --temperature 0.7\n"
    "\n",
    program_name, program_name, program_name, program_name, program_name);
}

void CLIParser::PrintVersion() {
    fprintf(stderr,
    "NotLLAMA version 0.2.0 (Patch 2: Multi-Model + MTP + Web)\n"
    "Vulkan Compute LLM Inference Engine for AMD RDNA4/RDNA3\n"
    "\n"
    "Features:\n"
    "  - Multi-model loading with VRAM management\n"
    "  - Model routing (single/ensemble/cascade)\n"
    "  - Multi-token prediction (speculative decoding)\n"
    "  - OpenAI-compatible HTTP API server\n"
    "  - llama.cpp-compatible CLI\n"
    "  - All quantization formats (Q4_0, Q8_0, Q4_K, Q6_K, K-quants)\n"
    "  - Flash Attention\n"
    "  - Wave32 compute shaders\n"
    "\n");
}

} // namespace notllama
