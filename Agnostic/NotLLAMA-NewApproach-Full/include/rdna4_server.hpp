#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace rdna4 {

struct InferenceEngine;
struct ModelHParams;

// ============================================================================
// HTTP server config (REST API parity with llama.cpp server)
// ============================================================================
struct ServerConfig {
    uint16_t port = 8080;
    std::string host = "127.0.0.1";
    uint32_t maxClients = 64;
    uint32_t requestTimeoutMs = 60000;
    bool enableCors = true;
    bool enableHealthEndpoint = true;
    bool enableMetricsEndpoint = true;
    bool enableSlotsEndpoint = true;
    bool enableEmbeddingEndpoint = true;
    bool enableCompletionEndpoint = true;
    bool enableChatEndpoint = true;
    bool enableInfillEndpoint = true;
    bool enableTokenizeEndpoint = true;
    bool enableDetokenizeEndpoint = true;
};

// ============================================================================
// Slot state (llama.cpp server parity)
// ============================================================================
struct ServerSlot {
    uint32_t id = 0;
    bool     occupied = false;
    uint32_t seqId = 0;
    std::string clientId;
    std::vector<uint32_t> promptTokens;
    std::vector<uint32_t> generatedTokens;
    bool     isProcessing = false;
    bool     isStreaming = false;
    float    temperature = 0.8f;
    uint32_t topK = 40;
    float    topP = 0.95f;
    uint32_t maxTokens = 256;
    std::string grammar;
    std::vector<std::string> stopSequences;
    uint64_t tStartUs = 0;
    uint64_t tPromptUs = 0;
    uint64_t tGenerateUs = 0;
};

// ============================================================================
// HTTP server — OpenAI-compatible API + llama.cpp extensions
// ============================================================================
class HttpServer {
public:
    explicit HttpServer(const ServerConfig& cfg);
    ~HttpServer();

    bool start();
    void stop();
    bool isRunning() const;

    // Bind inference engine (called after model is loaded)
    void setEngine(InferenceEngine* engine);

    // Bind tokenizer (for /tokenize, /detokenize)
    // void setTokenizer(Tokenizer* tok); // if needed

    // Slot management
    uint32_t allocateSlot();
    void releaseSlot(uint32_t slotId);
    ServerSlot* getSlot(uint32_t slotId);

    // Stats
    uint64_t getRequestCount() const;
    uint64_t getErrorCount() const;
    float    getAvgLatencyMs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rdna4
