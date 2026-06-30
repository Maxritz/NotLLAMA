#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace rdna4 {

// ============================================================================
// Grammar constraint types (llama.cpp parity)
// ============================================================================
enum class GrammarType : uint32_t {
    NONE     = 0,
    JSON     = 1,  // Strict JSON output
    JSON_SCHEMA = 2, // JSON with schema validation
    REGEX    = 3,  // Regular expression constraint
    BNF      = 4,  // Backus-Naur form grammar
    CNF      = 5,  // Conjunctive normal form
    COUNTER  = 6,  // Token counter / length limiter
};

// ============================================================================
// Logit bias — per-token probability adjustment
// ============================================================================
struct LogitBias {
    uint32_t tokenId = 0;
    float bias = 0.0f;    // added to logit before softmax
    bool ban = false;     // if true, set probability to 0
    float scale = 1.0f;   // multiply probability (applied after softmax)
};

// ============================================================================
// Sampler configuration
// ============================================================================
struct SamplerConfig {
    float temperature = 0.8f;
    uint32_t topK = 40;
    float topP = 0.95f;
    float minP = 0.05f;
    float typicalP = 1.0f;
    float tfsZ = 1.0f;           // tail free sampling
    float repeatPenalty = 1.1f;
    uint32_t repeatLastN = 64;
    float presencePenalty = 0.0f;
    float frequencyPenalty = 0.0f;
    float penaltyDecay = 0.0f;
    bool penaltyPromptTokens = false;
    bool ignoreEos = false;
    uint32_t seed = 0;
    GrammarType grammarType = GrammarType::NONE;
    std::string grammarSource;   // JSON schema, regex, BNF, etc.
    std::vector<LogitBias> logitBias;
    uint32_t mirostatMode = 0;   // 0=off, 1=v1, 2=v2
    float mirostatTau = 5.0f;
    float mirostatEta = 0.1f;
    bool dryRun = false;         // compute but don't modify logits
};

// ============================================================================
// Chat message (for chat template application)
// ============================================================================
struct ChatMessage {
    std::string role;    // "system", "user", "assistant", "tool"
    std::string content;
    std::string name;    // optional name (for tool calls)
    std::vector<std::unordered_map<std::string, std::string>> toolCalls;
};

// ============================================================================
// Advanced sampler — grammar, logit bias, chat templates, penalties
// ============================================================================
class AdvancedSampler {
public:
    explicit AdvancedSampler(const SamplerConfig& cfg);

    // Apply all samplers to logits in place
    // logits: [vocabSize] float array
    void sample(float* logits, uint32_t vocabSize,
                const std::vector<uint32_t>& context,
                uint32_t& outToken);

    // Grammar-constrained sampling
    void setGrammar(GrammarType type, const std::string& source);
    void clearGrammar();

    // Logit bias management
    void setBias(const std::vector<LogitBias>& biases);
    void addBias(const LogitBias& bias);
    void clearBias();

    // Chat template
    bool loadChatTemplate(const std::string& templateString);
    std::string applyChatTemplate(const std::vector<ChatMessage>& messages) const;

    // Token healing — fix token boundary artifacts
    uint32_t healToken(const std::vector<uint32_t>& context,
                       const float* logits, uint32_t vocabSize,
                       const std::string& partialUtf8);

    // Repeat penalty application (separable for inspection)
    void applyRepeatPenalty(float* logits, uint32_t vocabSize,
                            const std::vector<uint32_t>& context);

    // Temperature + nucleus (top-p) + top-k
    void applyTemperatureAndNucleus(float* logits, uint32_t vocabSize);

    // Mirostat sampling
    void applyMirostat(float* logits, uint32_t vocabSize, uint32_t& outToken);

    // Dry sampler (logits inspection without modification)
    std::vector<std::pair<uint32_t, float>> inspectLogits(const float* logits,
                                                           uint32_t vocabSize,
                                                           uint32_t topN = 10);

private:
    SamplerConfig cfg_;
    std::vector<LogitBias> biases_;
    std::string chatTemplate_;
    GrammarType grammarType_ = GrammarType::NONE;
    std::string grammarSource_;

    // Mirostat state
    float mirostatMu_ = 0.0f;

    // Grammar state machine (stub)
    struct GrammarState;
    std::unique_ptr<GrammarState> grammarState_;
};

} // namespace rdna4
