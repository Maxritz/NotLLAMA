#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <memory>
#include <functional>
#include <mutex>

namespace rdna4 {

struct ModelHParams;
struct InferenceEngine;

// ============================================================================
// Context compaction strategy
// ============================================================================
enum class CompactionStrategy : uint32_t {
    NONE          = 0,  // Hard stop at context limit
    SLIDING_WINDOW = 1, // Keep last N tokens, drop front
    HALF_SLIDE    = 2,  // Drop first 50% when threshold reached
    FIFO          = 3,  // First-in-first-out eviction
    IMPORTANCE    = 4,  // Keep high-importance tokens (system prompt, etc.)
    SUMMARY       = 5,  // Summarize dropped section and inject as system msg
};

// ============================================================================
// Per-sequence context state
// ============================================================================
struct SequenceContext {
    uint32_t seqId = 0;
    std::vector<uint32_t> tokens;
    std::vector<float>    tokenImportance; // per-token importance score [0,1]

    // KV cache bookkeeping (layer, head, pos offsets)
    uint32_t kvCachePos = 0;      // current write position in KV cache
    uint32_t kvCacheBase = 0;     // base offset (after compaction shifts)

    // Metadata
    uint64_t lastActivityUs = 0;
    uint32_t nGenerated = 0;
    bool     isPrefill = true;
};

// ============================================================================
// Context manager — prevents context window breaches via compaction
// ============================================================================
class ContextManager {
public:
    struct Config {
        uint32_t maxContextLength = 4096;
        float    compactionThreshold = 0.85f; // compact when usage > 85%
        float    compactionTarget = 0.50f;    // compact down to 50% usage
        CompactionStrategy strategy = CompactionStrategy::HALF_SLIDE;
        bool     preserveSystemPrompt = true;
        uint32_t systemPromptTokens = 0;      // number of leading tokens to protect
        bool     enableImportanceScoring = false;
    };

    explicit ContextManager(const Config& cfg);

    // Register a new sequence; returns seqId
    uint32_t createSequence(uint32_t systemPromptLen = 0);

    // Append token(s) to a sequence; returns false if sequence was compacted
    bool appendTokens(uint32_t seqId, const std::vector<uint32_t>& tokens);
    bool appendToken(uint32_t seqId, uint32_t token);

    // Check if compaction is needed and execute it
    bool maybeCompact(uint32_t seqId);

    // Force immediate compaction
    void compactNow(uint32_t seqId);

    // Get current token window (after any compaction)
    const std::vector<uint32_t>& getTokens(uint32_t seqId) const;
    uint32_t getTokenCount(uint32_t seqId) const;
    float    getUsageRatio(uint32_t seqId) const;

    // KV cache shift notification (called by engine after compaction)
    using KvShiftCallback = std::function<void(uint32_t seqId, uint32_t oldBase, uint32_t newBase, uint32_t newLen)>;
    void setKvShiftCallback(KvShiftCallback cb);

    // Importance scoring hooks
    void setTokenImportance(uint32_t seqId, uint32_t pos, float importance);
    void scoreSystemPromptHigh(uint32_t seqId, uint32_t len);

    // Reset / remove sequence
    void resetSequence(uint32_t seqId);
    void removeSequence(uint32_t seqId);

    // Global stats
    uint32_t getSequenceCount() const;
    uint32_t getTotalTokens() const;
    uint32_t getCompactionCount() const;

private:
    Config cfg_;
    std::unordered_map<uint32_t, std::unique_ptr<SequenceContext>> sequences_;
    uint32_t nextSeqId_ = 1;
    uint32_t compactionCount_ = 0;
    KvShiftCallback kvShiftCb_;
    mutable std::mutex mutex_;

    void doSlidingWindowCompact(SequenceContext& seq);
    void doHalfSlideCompact(SequenceContext& seq);
    void doFifoCompact(SequenceContext& seq);
    void doImportanceCompact(SequenceContext& seq);
    void doSummaryCompact(SequenceContext& seq);
};

} // namespace rdna4
