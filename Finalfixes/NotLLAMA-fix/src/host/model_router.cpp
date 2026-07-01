#include "model_router.hpp"
#include <cctype>
#include <algorithm>
#include <cstdio>

namespace notllama {

// Keyword patterns for content type detection
const std::vector<std::pair<std::regex, ContentType>> ModelRouter::patterns_ = {
    // Code patterns
    {std::regex(R"(\b(def|function|class|import|from|#include|#define|const|let|var|=>|\{|\}|\[code\]|```)\b)", std::regex::icase), ContentType::CODE},
    {std::regex(R"(\b(python|javascript|typescript|rust|go|c\+\+|java|kotlin|swift|sql|html|css|bash|shell|regex|api|json|xml|docker|git|npm|pip|cargo)\b)", std::regex::icase), ContentType::CODE},
    {std::regex(R"(function\s*\(|\bfn\s+|\bdef\s+\w+\s*\(|=>\s*\{|class\s+\w+\s*\{)", std::regex::icase), ContentType::CODE},
    // Math patterns
    {std::regex(R"([\d]+[\+\-\*/\^][\d]+|=\s*\d+|\b(solve|equation|integral|derivative|calculate|compute|formula|algebra|geometry|calculus|matrix|vector|sum|product|log|sin|cos|tan)\b)", std::regex::icase), ContentType::MATH},
    {std::regex(R"(\$\s*[\d,.]+|\b(percent|ratio|probability|statistics|mean|median|variance)\b)", std::regex::icase), ContentType::MATH},
    // Reasoning patterns
    {std::regex(R"(\b(why|how|explain|because|therefore|reason|logic|deduce|infer|conclusion|premise|argument|evidence|prove|demonstrate|analyze)\b)", std::regex::icase), ContentType::REASONING},
    {std::regex(R"(\b(step by step|let's think|what if|consider|suppose|assume|given that|if then)\b)", std::regex::icase), ContentType::REASONING},
    // Technical patterns
    {std::regex(R"(\b(documentation|specification|RFC|protocol|architecture|system|database|server|client|network|protocol|interface|component|module|service)\b)", std::regex::icase), ContentType::TECHNICAL},
    // Creative patterns
    {std::regex(R"(\b(story|poem|write|creative|imagine|fiction|novel|character|scene|dialogue|narrative|describe|vivid|metaphor)\b)", std::regex::icase), ContentType::CREATIVE},
    // Chat patterns
    {std::regex(R"(\b(hello|hi|hey|thanks|please|sorry|help|can you|would you|what's up|how are|nice to|bye|goodbye)\b)", std::regex::icase), ContentType::CHAT},
};

ModelRouter::ModelRouter(MultiModelManager* manager) : manager_(manager) {}

void ModelRouter::RegisterCapability(const std::string& model_id, const ModelCapability& cap) {
    capabilities_[model_id] = cap;
}

void ModelRouter::AutoDetectCapabilities(const std::string& model_id) {
    auto* inst = manager_->GetModel(model_id);
    if (!inst) return;

    ModelCapability cap;
    cap.model_id = model_id;
    cap.context_window = inst->max_context;

    std::string lower_name = inst->name;
    std::string lower_tags = inst->tags;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    std::transform(lower_tags.begin(), lower_tags.end(), lower_tags.begin(), ::tolower);

    // Code models
    if (lower_name.find("code") != std::string::npos ||
        lower_tags.find("code") != std::string::npos ||
        lower_name.find("deepseek") != std::string::npos ||
        lower_name.find("codellama") != std::string::npos ||
        lower_name.find("starcoder") != std::string::npos ||
        lower_name.find("qwen") != std::string::npos) {
        cap.code_score = 0.95f;
        cap.technical_score = 0.90f;
        cap.reasoning_score = 0.80f;
        cap.english_score = 0.70f;
    }
    // English/general models
    else if (lower_name.find("llama") != std::string::npos ||
             lower_name.find("mistral") != std::string::npos ||
             lower_name.find("yi") != std::string::npos ||
             lower_name.find("zephyr") != std::string::npos) {
        cap.english_score = 0.95f;
        cap.chat_score = 0.90f;
        cap.creative_score = 0.85f;
        cap.reasoning_score = 0.85f;
        cap.code_score = 0.60f;
    }
    // Math/reasoning models
    else if (lower_name.find("math") != std::string::npos ||
             lower_name.find("wizard") != std::string::npos ||
             lower_name.find("phi") != std::string::npos) {
        cap.math_score = 0.95f;
        cap.reasoning_score = 0.95f;
        cap.code_score = 0.75f;
        cap.english_score = 0.80f;
    }
    // Default
    else {
        cap.english_score = 0.80f;
        cap.code_score = 0.50f;
        cap.chat_score = 0.75f;
        cap.reasoning_score = 0.70f;
        cap.math_score = 0.60f;
        cap.creative_score = 0.70f;
        cap.technical_score = 0.60f;
    }

    // Override with explicit tags
    if (lower_tags.find("code") != std::string::npos) cap.code_score = std::max(cap.code_score, 0.90f);
    if (lower_tags.find("english") != std::string::npos) cap.english_score = std::max(cap.english_score, 0.90f);
    if (lower_tags.find("math") != std::string::npos) cap.math_score = std::max(cap.math_score, 0.90f);
    if (lower_tags.find("chat") != std::string::npos) cap.chat_score = std::max(cap.chat_score, 0.90f);

    capabilities_[model_id] = cap;
    fprintf(stderr, "[Router] Auto-detected for '%s': code=%.2f english=%.2f math=%.2f chat=%.2f\n",
            model_id.c_str(), cap.code_score, cap.english_score, cap.math_score, cap.chat_score);
}

ContentType ModelRouter::DetectContentType(const std::string& prompt) {
    std::unordered_map<ContentType, int> scores;

    for (const auto& [pattern, ctype] : patterns_) {
        try {
            if (std::regex_search(prompt, pattern)) {
                scores[ctype] += 2;
            }
        } catch (...) {
            // regex exception - skip this pattern
        }
    }

    // Additional heuristics
    // Count code indicators: brackets, semicolons, indentation
    int code_chars = 0;
    for (char c : prompt) {
        if (c == '{' || c == '}' || c == ';' || c == '(' || c == ')') code_chars++;
    }
    if (code_chars > prompt.length() / 20) { // > 5% code chars
        scores[ContentType::CODE] += 3;
    }

    // Count math symbols
    int math_chars = 0;
    for (char c : prompt) {
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '^') math_chars++;
    }
    if (math_chars > prompt.length() / 15) {
        scores[ContentType::MATH] += 3;
    }

    if (scores.empty()) return ContentType::MIXED;

    // Pick highest score
    auto best = std::max_element(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    return best->first;
}

float ModelRouter::ScoreModelForContent(const ModelCapability& cap, ContentType content) {
    switch (content) {
        case ContentType::CODE:       return cap.code_score;
        case ContentType::ENGLISH:    return cap.english_score;
        case ContentType::MATH:       return cap.math_score;
        case ContentType::REASONING:  return cap.reasoning_score;
        case ContentType::CREATIVE:   return cap.creative_score;
        case ContentType::TECHNICAL:  return cap.technical_score;
        case ContentType::CHAT:       return cap.chat_score;
        case ContentType::MIXED:      // Average of all scores
            return (cap.code_score + cap.english_score + cap.math_score +
                    cap.reasoning_score + cap.creative_score +
                    cap.technical_score + cap.chat_score) / 7.0f;
    }
    return 0.0f;
}

RouteDecision ModelRouter::PickBestModel(ContentType content) {
    RouteDecision decision;
    decision.confidence = 0.0f;

    auto model_ids = manager_->ListModels();
    if (model_ids.empty()) {
        decision.reason = "No models loaded";
        return decision;
    }

    std::string best_id;
    float best_score = -1.0f;

    for (const auto& id : model_ids) {
        auto it = capabilities_.find(id);
        if (it == capabilities_.end()) {
            // Auto-detect if not registered
            AutoDetectCapabilities(id);
            it = capabilities_.find(id);
        }
        if (it == capabilities_.end()) continue;

        float score = ScoreModelForContent(it->second, content);
        if (score > best_score) {
            best_score = score;
            best_id = id;
        }
    }

    if (!best_id.empty()) {
        decision.primary_model_id = best_id;
        decision.confidence = best_score;
        decision.reason = "Routed to " + best_id + " (score=" + std::to_string((int)(best_score * 100)) + "%)";
    } else {
        decision.primary_model_id = model_ids[0];
        decision.reason = "Fallback to first model (no capability match)";
    }

    return decision;
}

RouteDecision ModelRouter::PickEnsemble(ContentType content) {
    RouteDecision decision;
    auto model_ids = manager_->ListModels();
    if (model_ids.empty()) return decision;

    // Sort models by score for this content type
    std::vector<std::pair<std::string, float>> scored;
    for (const auto& id : model_ids) {
        auto it = capabilities_.find(id);
        if (it == capabilities_.end()) {
            AutoDetectCapabilities(id);
            it = capabilities_.find(id);
        }
        float score = (it != capabilities_.end()) ? ScoreModelForContent(it->second, content) : 0.5f;
        scored.push_back({id, score});
    }

    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Primary = best, fallbacks = next 2 (or fewer if not enough models)
    decision.primary_model_id = scored[0].first;
    decision.confidence = scored[0].second;
    for (size_t i = 1; i < scored.size() && i < 3; i++) {
        decision.fallback_model_ids.push_back(scored[i].first);
    }
    decision.reason = "Ensemble: primary=" + scored[0].first;
    return decision;
}

RouteDecision ModelRouter::Route(const std::string& prompt) {
    ContentType content = DetectContentType(prompt);

    switch (mode_) {
        case Mode::SINGLE_BEST:
            return PickBestModel(content);
        case Mode::ENSEMBLE:
            return PickEnsemble(content);
        case Mode::CASCADE:
            return PickBestModel(content); // Same as single, but with fallback
    }
    return PickBestModel(content);
}

RouteDecision ModelRouter::RouteWithHint(const std::string& prompt, ContentType hint) {
    (void)prompt;
    return PickBestModel(hint);
}

uint32_t ModelRouter::EnsembleForward(const std::vector<std::string>& model_ids,
                                        uint32_t token, uint32_t position) {
    // Simple ensemble: each model votes, return most popular token
    // For a real implementation, you'd average logits across models
    std::unordered_map<uint32_t, int> votes;
    uint32_t best_token = token;
    int best_count = 0;

    for (const auto& id : model_ids) {
        auto* inst = manager_->GetModel(id);
        if (!inst || !inst->adapter) continue;

        // TODO: Run forward pass and get logits
        // For now, return the input token (passthrough)
        votes[token]++;
    }

    for (const auto& [tok, count] : votes) {
        if (count > best_count) {
            best_count = count;
            best_token = tok;
        }
    }
    return best_token;
}

void ModelRouter::SetWeights(float code_w, float english_w, float math_w,
                              float reasoning_w, float creative_w, float technical_w, float chat_w) {
    w_code_ = code_w;
    w_english_ = english_w;
    w_math_ = math_w;
    w_reasoning_ = reasoning_w;
    w_creative_ = creative_w;
    w_technical_ = technical_w;
    w_chat_ = chat_w;
}

} // namespace notllama
