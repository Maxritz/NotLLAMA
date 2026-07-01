#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace rdna4 {

struct InferenceEngine;
struct ModelHParams;
struct ContextManager;

// ============================================================================
// Batched sequence slot
// ============================================================================
struct BatchSlot {
    uint32_t slotId = 0;
    uint32_t seqId = 0;          // maps to ContextManager sequence
    std::vector<uint32_t> promptTokens;
    std::vector<uint32_t> generatedTokens;

    // State
    bool isPrefill = true;
    bool isGenerating = false;
    bool isComplete = false;
    uint32_t pos = 0;            // current position in KV cache

    // Sampling params (per-slot override)
    float temperature = 0.8f;
    uint32_t topK = 40;
    float topP = 0.95f;
    uint32_t maxTokens = 256;

    // Callback for streaming output
    using TokenCallback = std::function<void(uint32_t slotId, uint32_t token, const std::string& text, bool isLast)>;
    TokenCallback onToken;
};

// ============================================================================
// Continuous batching manager
// ============================================================================
class BatchManager {
public:
    struct Config {
        uint32_t maxBatchSize = 8;           // max concurrent sequences
        uint32_t maxContextLength = 4096;
        bool     continuousBatching = true;  // pause/generate interleave
        bool     chunkedPrefill = true;      // split long prefill into chunks
        uint32_t chunkSize = 512;            // tokens per prefill chunk
        bool     priorityBoost = true;       // prioritize short prompts
    };

    explicit BatchManager(InferenceEngine* engine,
                          ContextManager* ctxMgr,
                          const Config& cfg);

    // Add a new request to the batch queue
    uint32_t submitRequest(const std::vector<uint32_t>& promptTokens,
                           const BatchSlot::TokenCallback& onToken = nullptr);

    // Run one scheduling iteration (prefill + generate step)
    // Returns number of active sequences after this step
    uint32_t step();

    // Check if any sequences are still active
    bool hasActive() const;

    // Get completions (sequences that finished this step)
    std::vector<uint32_t> getCompletedSlots() const;

    // Cancel a running sequence
    bool cancelRequest(uint32_t slotId);

    // Wait for all sequences to complete
    void waitAll();

    // Stats
    uint32_t getQueueDepth() const;
    uint32_t getActiveCount() const;
    float    getAvgTokensPerSec() const;

private:
    InferenceEngine* engine_;
    ContextManager* ctxMgr_;
    Config cfg_;

    std::vector<std::unique_ptr<BatchSlot>> slots_;
    std::vector<uint32_t> freeSlots_;
    std::vector<std::unique_ptr<BatchSlot>> pendingQueue_;

    uint32_t nextSlotId_ = 1;
    uint64_t totalTokensGenerated_ = 0;
    uint64_t totalSteps_ = 0;

    void runPrefill(BatchSlot& slot);
    void runGenerate(BatchSlot& slot);
    bool tryAddToBatch();
};

} // namespace rdna4
