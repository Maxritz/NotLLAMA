#pragma once
#include "multi_model_manager.hpp"
#include "model_router.hpp"
#include "inference_runner.hpp"
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>

// nlohmann/json forward
#include <nlohmann/json.hpp>

namespace notllama {

// llama.cpp-compatible API request/response structures
struct CompletionRequest {
    std::string model;               // Model ID to use
    std::vector<std::map<std::string, std::string>> messages; // Chat messages
    std::string prompt;              // Or raw prompt
    int32_t max_tokens = -1;         // Max tokens to generate
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float repeat_penalty = 1.1f;
    int32_t seed = -1;
    bool stream = false;
    std::vector<std::string> stop;
    std::string user;                // End-user ID for tracking
    int32_t n_predict = -1;
    bool logprobs = false;
    int32_t top_logprobs = 0;
};

struct CompletionResponse {
    std::string id;
    std::string object;              // "text_completion" or "chat.completion"
    int64_t created;
    std::string model;
    struct Choice {
        int32_t index = 0;
        std::string text;
        std::string finish_reason;   // "stop", "length", "null"
        std::map<std::string, nlohmann::json> logprobs;
    };
    std::vector<Choice> choices;
    struct Usage {
        int32_t prompt_tokens = 0;
        int32_t completion_tokens = 0;
        int32_t total_tokens = 0;
    } usage;
    bool done = false;
};

// HTTP server: llama.cpp-compatible REST API
// Endpoints:
//   POST /v1/completions        - Text completion (OpenAI compatible)
//   POST /v1/chat/completions   - Chat completion (OpenAI compatible)
//   POST /v1/embeddings         - Text embedding
//   POST /v1/rerank             - Reranking
//   GET  /v1/models             - List loaded models
//   GET  /health                - Health check
//   GET  /props                 - Server properties
//   POST /tokenize              - Tokenize text
//   POST /detokenize            - Detokenize IDs
class WebServer {
public:
    WebServer(MultiModelManager* model_mgr,
              ModelRouter* router,
              InferenceRunner* inference,
              const std::string& host = "127.0.0.1",
              int32_t port = 8080,
              int32_t timeout_seconds = 600);
    ~WebServer();

    // Start server in background thread
    bool Start();

    // Stop server
    void Stop();

    // Check if running
    bool IsRunning() const { return running_.load(); }

    // Set API key for authentication (empty = no auth)
    void SetApiKey(const std::string& key) { api_key_ = key; }

    // Enable/disable endpoints
    void EnableEmbeddings(bool enable) { enable_embeddings_ = enable; }
    void EnableReranking(bool enable) { enable_reranking_ = enable; }

    // Agent endpoint handler callback: (subpath, body_json) -> response_json_string
    using AgentHandlerFn = std::function<std::string(const std::string&, const nlohmann::json&)>;
    void SetAgentHandler(AgentHandlerFn fn) { agent_handler_ = fn; }

private:
    MultiModelManager* model_mgr_;
    AgentHandlerFn agent_handler_;
    ModelRouter* router_;
    InferenceRunner* inference_;
    std::string host_;
    int32_t port_;
    int32_t timeout_seconds_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_flag_{false};
    std::thread server_thread_;
    std::string api_key_;
    bool enable_embeddings_ = true;
    bool enable_reranking_ = false;

    int server_fd_ = -1;

    // Server loop
    void Run();

    // Request handlers
    void HandleRequest(int client_fd);
    std::string RouteRequest(const std::string& method,
                              const std::string& path,
                              const std::string& body,
                              const std::string& auth_header);

    // API endpoint handlers
    std::string HandleCompletions(const nlohmann::json& req);
    std::string HandleChatCompletions(const nlohmann::json& req);
    std::string HandleEmbeddings(const nlohmann::json& req);
    std::string HandleRerank(const nlohmann::json& req);
    std::string HandleListModels();
    std::string HandleHealth();
    std::string HandleProps();
    std::string HandleTokenize(const nlohmann::json& req);
    std::string HandleDetokenize(const nlohmann::json& req);

    // Streaming response
    void SendStreamChunk(int client_fd, const std::string& data, bool done);

    // Helpers
    std::string ParsePath(const std::string& request);
    std::string ParseMethod(const std::string& request);
    std::string ParseBody(const std::string& request);
    std::string ParseHeader(const std::string& request, const std::string& header);
    bool CheckAuth(const std::string& auth_header);
    std::string MakeJsonResponse(int code, const nlohmann::json& data);
    std::string MakeErrorResponse(int code, const std::string& message,
                                   const std::string& type);

    // Generate completion using the engine
    CompletionResponse DoCompletion(const CompletionRequest& req);

    int MakeSocket();
};

} // namespace notllama
