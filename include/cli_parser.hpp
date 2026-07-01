#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace notllama {

// llama.cpp-compatible CLI arguments
struct CLIOptions {
    // Model options
    std::vector<std::string> model_paths;         // --model (can specify multiple)
    std::vector<std::string> model_ids;           // --model-id (optional labels)
    std::vector<std::string> model_tags;          // --model-tags (e.g., "code,python")
    std::string draft_model;                      // --draft-model for speculative decode
    int32_t gpu_layers = -1;                      // --gpu-layers (-1 = all)

    // Prompt options
    std::string prompt;                           // --prompt / -p
    std::string prompt_file;                      // --file / -f
    std::string system_prompt;                    // --system-prompt
    std::string chat_template;                    // --chat-template
    bool interactive = false;                     // --interactive / -i
    bool multiline_input = false;                 // --multiline-input

    // Generation options
    int32_t n_predict = -1;                       // --n-predict / -n (-1 = infinite)
    int32_t ctx_size = 4096;                      // --ctx-size / -c
    int32_t batch_size = 2048;                    // --batch-size / -b
    int32_t ubatch_size = 512;                    // --ubatch-size
    float temperature = 0.8f;                     // --temperature
    float top_p = 0.95f;                          // --top-p
    int32_t top_k = 40;                           // --top-k
    float repeat_penalty = 1.1f;                  // --repeat-penalty
    int32_t repeat_last_n = 64;                   // --repeat-last-n
    float frequency_penalty = 0.0f;               // --frequency-penalty
    float presence_penalty = 0.0f;                // --presence-penalty
    std::string sampler = "default";              // --sampler-seq
    int32_t min_p = 0;                            // --min-p
    float dynatemp_range = 0.0f;                  // --dynatemp-range
    float dynatemp_exponent = 1.0f;               // --dynatemp-exponent
    int32_t mirostat = 0;                         // --mirostat (0=off, 1/2=on)
    float mirostat_tau = 5.0f;                    // --mirostat-tau
    float mirostat_eta = 0.1f;                    // --mirostat-eta
    uint32_t seed = 0xFFFFFFFF;                   // --seed (-1 = random)

    // Multi-model / routing options
    std::string router_mode = "single";           // --router-mode (single/ensemble/cascade)
    bool enable_mtp = false;                      // --mtp (enable multi-token prediction)
    int32_t mtp_n_draft = 4;                      // --mtp-n-draft

    // Web server options
    bool server_mode = false;                     // --server
    std::string host = "127.0.0.1";               // --host
    int32_t port = 8080;                          // --port
    int32_t timeout = 600;                        // --timeout (seconds)
    std::string api_key;                          // --api-key
    bool embedding_endpoint = true;               // --embeddings
    bool reranking_endpoint = false;              // --reranking

    // Formatting options
    bool verbose_prompt = false;                  // --verbose-prompt
    bool no_display_prompt = false;               // --no-display-prompt
    bool log_disable = false;                     // --log-disable
    std::string log_file;                         // --log-file
    bool color = true;                            // --color / --no-color
    bool special = false;                         // --special
    std::string output_format = "text";           // --output-format (text/json)

    // Performance options
    int32_t threads = -1;                         // --threads / -t (-1 = auto)
    int32_t threads_batch = -1;                   // --threads-batch
    bool flash_attn = true;                       // --flash-attn
    bool no_kv_offload = false;                   // --no-kv-offload
    std::string tensor_split;                     // --tensor-split
    std::string main_gpu;                         // --main-gpu

    // Benchmark / debug
    bool benchmark = false;                       // --benchmark
    int32_t benchmark_iterations = 10;            // --benchmark-iterations
    bool prompt_benchmark = false;                // --prompt-benchmark

    // Distributed Agent options
    std::string agent_name;                       // --agent-name (e.g., "deepseek-node")
    int32_t agent_port = 0;                       // --agent-port (0 = disabled)
    std::vector<std::string> agent_peers;         // --agent-peer (host,port,name,tags)
    bool enable_reason_sharing = true;            // --enable-reason-sharing
    bool enable_model_distill = false;            // --enable-model-distill

    // External tool awareness (not built-in, just flags)
    bool use_graphify = false;                    // --use-graphify
    std::string graphify_url;                     // --graphify-url
    bool use_mcp = false;                         // --use-mcp
    std::string mcp_url;                          // --mcp-url

    // Model creation
    std::string create_model_name;                // --create-model <name>
    std::string create_model_type;                // --create-model-type
    size_t create_model_size_mb = 0;              // --create-model-size
    std::string create_model_quant;               // --create-model-quant
    std::string create_model_arch;                // --create-model-arch

    // Help
    bool show_help = false;                       // --help / -h
    bool show_version = false;                    // --version / -v

    // Extra key-value pairs for forward compatibility
    std::unordered_map<std::string, std::string> extra;
};

class CLIParser {
public:
    // Parse argc/argv into options. Returns false on error or --help.
    static bool Parse(int argc, char** argv, CLIOptions& out);

    // Print help text (llama.cpp style)
    static void PrintHelp(const char* program_name);

    // Print version
    static void PrintVersion();

private:
    static bool ParseInt(const char* s, int32_t& out);
    static bool ParseUInt(const char* s, uint32_t& out);
    static bool ParseFloat(const char* s, float& out);
    static bool ParseStringList(const char* s, std::vector<std::string>& out);
};

} // namespace notllama
