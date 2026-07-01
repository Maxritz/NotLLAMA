#pragma once
#include "multi_model_manager.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>

namespace notllama {

// Routing decision: which model(s) to use
struct RouteDecision {
    std::string primary_model_id;
    std::vector<std::string> fallback_model_ids;
    float confidence = 1.0f;     // How confident we are in this route (0-1)
    std::string reason;          // Human-readable routing reason
};

// Model capability profile for routing decisions
struct ModelCapability {
    std::string model_id;
    float code_score = 0.0f;         // Proficiency at code (0-1)
    float english_score = 0.0f;      // Proficiency at English text (0-1)
    float math_score = 0.0f;         // Proficiency at math (0-1)
    float reasoning_score = 0.0f;    // Proficiency at reasoning (0-1)
    float creative_score = 0.0f;     // Proficiency at creative writing (0-1)
    float technical_score = 0.0f;    // Proficiency at technical docs (0-1)
    float chat_score = 0.0f;         // Proficiency at conversation (0-1)
    size_t context_window = 2048;    // Max context length
    float avg_latency_ms = 0.0f;     // Average inference latency
};

// Content type detected from prompt
enum class ContentType {
    CODE,           // Programming code
    ENGLISH,        // General English text
    MATH,           // Mathematical content
    REASONING,      // Logical reasoning
    CREATIVE,       // Creative writing
    TECHNICAL,      // Technical documentation
    CHAT,           // Casual conversation
    MIXED,          // Mixed/unclear
};

// Model router: decides which model should handle a given prompt
// Supports ensemble mode (multiple models vote) and specialist mode (best model)
class ModelRouter {
public:
    enum class Mode {
        SINGLE_BEST,    // Route to single best model
        ENSEMBLE,       // Query multiple models, blend outputs
        CASCADE,        // Try primary, fall back on low confidence
    };

    ModelRouter(MultiModelManager* manager);

    // Set routing mode
    void SetMode(Mode mode) { mode_ = mode; }
    Mode GetMode() const { return mode_; }

    // Register capability profile for a model
    void RegisterCapability(const std::string& model_id, const ModelCapability& cap);

    // Auto-detect capabilities from model name/tags
    void AutoDetectCapabilities(const std::string& model_id);

    // Route a prompt to the best model(s)
    RouteDecision Route(const std::string& prompt);

    // Route with explicit content type hint
    RouteDecision RouteWithHint(const std::string& prompt, ContentType hint);

    // Detect content type from prompt text
    ContentType DetectContentType(const std::string& prompt);

    // Ensemble: query multiple models and return best response
    // (Returns token ID from the model with highest confidence)
    uint32_t EnsembleForward(const std::vector<std::string>& model_ids,
                              uint32_t token, uint32_t position);

    // Set weights for routing criteria (must sum to 1.0)
    void SetWeights(float code_w, float english_w, float math_w,
                    float reasoning_w, float creative_w, float technical_w, float chat_w);

private:
    MultiModelManager* manager_;
    Mode mode_ = Mode::SINGLE_BEST;
    std::unordered_map<std::string, ModelCapability> capabilities_;

    // Routing weights
    float w_code_ = 0.20f;
    float w_english_ = 0.15f;
    float w_math_ = 0.15f;
    float w_reasoning_ = 0.15f;
    float w_creative_ = 0.10f;
    float w_technical_ = 0.15f;
    float w_chat_ = 0.10f;

    // Keyword patterns for content detection
    static const std::vector<std::pair<std::regex, ContentType>> patterns_;

    float ScoreModelForContent(const ModelCapability& cap, ContentType content);
    RouteDecision PickBestModel(ContentType content);
    RouteDecision PickEnsemble(ContentType content);
};

} // namespace notllama
