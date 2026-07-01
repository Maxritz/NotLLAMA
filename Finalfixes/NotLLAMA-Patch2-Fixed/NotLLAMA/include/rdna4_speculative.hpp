#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rdna4 {

struct InferenceEngine;
struct ModelHParams;
struct Tokenizer;

// ============================================================================
// Draft model slot for speculative decoding
// ============================================================================
struct DraftModel {
    std::string modelId;
    InferenceEngine* engine = nullptr;
    float draftAcceptanceThreshold = 0.5f;
    uint32_t maxDraftTokens = 4;     // K in speculative decoding
    uint32_t vocabSize = 0;
    bool     sharedVocab = true;     // target and draft share tokenizer
};

// ============================================================================
// Speculative decoding manager
// ============================================================================
class SpeculativeDecoder {
public:
    struct Config {
        uint32_t maxDraftTokens = 4;
        float    minAcceptanceRate = 0.3f; // disable if below this
        bool     adaptiveDraftLen = true;  // adjust K based on acceptance rate
        uint32_t minDraftTokens = 1;
        uint32_t maxAdaptiveDraft = 8;
    };

    SpeculativeDecoder(InferenceEngine* targetEngine,
                       Tokenizer* tokenizer,
                       const Config& cfg);

    // Attach a draft model (smaller, same architecture)
    bool setDraftModel(const DraftModel& draft);

    // Remove draft model (fallback to autoregressive)
    void clearDraftModel();

    // Generate one token using speculative decoding
    // Internally: draft K tokens -> target verifies -> accept N, reject 1
    uint32_t generateToken(const std::vector<uint32_t>& context,
                           float temperature,
                           uint32_t topK,
                           float topP);

    // Stats
    float getAcceptanceRate() const;
    uint64_t getTotalDrafted() const;
    uint64_t getTotalAccepted() const;
    float getSpeedup() const;

private:
    InferenceEngine* target_;
    Tokenizer* tokenizer_;
    Config cfg_;
    std::unique_ptr<DraftModel> draft_;

    uint64_t totalDrafted_ = 0;
    uint64_t totalAccepted_ = 0;
    uint64_t totalTargetRuns_ = 0;
    float    currentAcceptanceRate_ = 0.0f;
    uint32_t currentDraftLen_ = 4;

    std::vector<uint32_t> draftTokens(const std::vector<uint32_t>& context, uint32_t k);
    uint32_t verifyAndSample(const std::vector<uint32_t>& context,
                             const std::vector<uint32_t>& drafted,
                             float temperature, uint32_t topK, float topP);
};

} // namespace rdna4
