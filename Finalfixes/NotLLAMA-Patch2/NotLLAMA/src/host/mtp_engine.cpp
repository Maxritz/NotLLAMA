#include "mtp_engine.hpp"
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>

namespace notllama {

MTPEngine::MTPEngine(MultiModelManager* manager)
    : manager_(manager), rng_(std::random_device{}()) {}

void MTPEngine::SetConfig(const MTPConfig& cfg) { config_ = cfg; }

uint32_t MTPEngine::GenerateNext(const std::string& model_id,
                                   const std::vector<uint32_t>& prompt_tokens,
                                   uint32_t position) {
    auto* inst = manager_->GetModel(model_id);
    if (!inst || !inst->adapter) {
        fprintf(stderr, "[MTP] Model '%s' not found\n", model_id.c_str());
        return 0;
    }

    if (config_.use_draft_model && !config_.draft_model_id.empty()) {
        // Speculative decoding path
        std::vector<uint32_t> accepted;
        uint32_t n_accepted = SpeculativeDecode(model_id, config_.draft_model_id,
                                                  prompt_tokens, position, accepted);
        if (n_accepted > 0 && !accepted.empty()) {
            return accepted[0];
        }
        // Fall through to direct generation if speculative failed
    }

    // Direct single-token generation via the model's forward pass
    // The Vulkan inference engine handles the actual token prediction
    // For now, we return a deterministic token based on context
    // (Full implementation would query the GPU for logits)

    // Simple fallback: use position-based token for testing
    uint32_t vocab_size = static_cast<uint32_t>(inst->adapter->GetEmbeddingDim());
    if (vocab_size == 0) vocab_size = 32000;

    // In a real implementation, this would:
    // 1. Run the model forward pass to get logits
    // 2. Apply temperature, top-p, top-k
    // 3. Sample a token
    // For now, return a deterministic token
    uint32_t token = (position + 1) % vocab_size;
    if (token == 0) token = 1;

    inst->total_tokens_generated++;
    return token;
}

std::vector<uint32_t> MTPEngine::Generate(const std::string& model_id,
                                            const std::vector<uint32_t>& prompt_tokens,
                                            uint32_t max_new_tokens,
                                            uint32_t eos_token) {
    std::vector<uint32_t> result;
    result.reserve(max_new_tokens);

    uint32_t position = static_cast<uint32_t>(prompt_tokens.size());
    std::vector<uint32_t> context = prompt_tokens;

    auto* inst = manager_->GetModel(model_id);
    if (!inst) {
        fprintf(stderr, "[MTP] Model '%s' not available\n", model_id.c_str());
        return result;
    }

    fprintf(stderr, "[MTP] Generating up to %u tokens with model '%s'\n",
            max_new_tokens, model_id.c_str());

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < max_new_tokens; i++) {
        uint32_t next_token = GenerateNext(model_id, context, position);

        if (next_token == 0) break;          // Error
        if (next_token == eos_token) break;   // End of sequence

        result.push_back(next_token);
        context.push_back(next_token);
        position++;

        // Context window management
        if (context.size() > inst->max_context) {
            // Keep last N tokens
            size_t keep = inst->max_context / 2;
            context = std::vector<uint32_t>(context.end() - keep, context.end());
            position = static_cast<uint32_t>(context.size());
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double tok_per_sec = (elapsed_ms > 0) ? (result.size() * 1000.0 / elapsed_ms) : 0;

    fprintf(stderr, "[MTP] Generated %zu tokens in %.1f ms (%.1f tok/s)\n",
            result.size(), elapsed_ms, tok_per_sec);

    inst->total_tokens_generated += result.size();
    inst->total_prompts++;

    return result;
}

uint32_t MTPEngine::SpeculativeDecode(const std::string& target_model_id,
                                        const std::string& draft_model_id,
                                        const std::vector<uint32_t>& context,
                                        uint32_t position,
                                        std::vector<uint32_t>& accepted_tokens) {
    auto* target = manager_->GetModel(target_model_id);
    auto* draft = manager_->GetModel(draft_model_id);

    if (!target || !draft) {
        fprintf(stderr, "[MTP] Speculative decode: models not available\n");
        return 0;
    }

    // 1. Draft model generates N candidate tokens
    std::vector<uint32_t> draft_tokens = DraftTokens(draft, context, position, config_.n_draft);
    if (draft_tokens.empty()) return 0;

    // 2. Target model verifies draft tokens in parallel
    // In a real implementation, the target model would evaluate all draft positions
    // and return which tokens are accepted
    uint32_t n_accepted = VerifyDrafts(target, context, draft_tokens, position);

    // 3. Return accepted tokens
    accepted_tokens.clear();
    for (uint32_t i = 0; i < n_accepted && i < draft_tokens.size(); i++) {
        accepted_tokens.push_back(draft_tokens[i]);
    }

    fprintf(stderr, "[MTP] Speculative: %u/%zu tokens accepted\n",
            n_accepted, draft_tokens.size());

    return n_accepted;
}

std::vector<uint32_t> MTPEngine::DraftTokens(ModelInstance* draft_model,
                                               const std::vector<uint32_t>& context,
                                               uint32_t position, uint32_t n_draft) {
    std::vector<uint32_t> drafts;
    drafts.reserve(n_draft);

    std::vector<uint32_t> draft_context = context;
    uint32_t pos = position;

    for (uint32_t i = 0; i < n_draft; i++) {
        uint32_t tok = GenerateNext(draft_model->id, draft_context, pos);
        if (tok == 0) break;
        drafts.push_back(tok);
        draft_context.push_back(tok);
        pos++;
    }

    return drafts;
}

uint32_t MTPEngine::VerifyDrafts(ModelInstance* target_model,
                                   const std::vector<uint32_t>& context,
                                   const std::vector<uint32_t>& draft_tokens,
                                   uint32_t position) {
    // Simplified verification: accept tokens probabilistically
    // In a real implementation, the target model evaluates each draft position
    // and returns the position where the first mismatch occurs

    if (!config_.verify_drafts) {
        return static_cast<uint32_t>(draft_tokens.size()); // Accept all without verification
    }

    // For now, use a probabilistic acceptance based on the threshold
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    uint32_t accepted = 0;

    for (uint32_t i = 0; i < draft_tokens.size(); i++) {
        float r = dist(rng_);
        if (r < config_.draft_accept_threshold) {
            accepted++;
        } else {
            break; // Reject this and all subsequent tokens
        }
    }

    return accepted;
}

uint32_t MTPEngine::Sample(const float* logits, uint32_t vocab_size,
                            float temperature, float top_p, int32_t top_k) {
    if (vocab_size == 0) return 0;

    // Copy logits for modification
    std::vector<float> probs(logits, logits + vocab_size);

    ApplySampler(probs, temperature, top_p, top_k);

    // Sample from the distribution
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

void MTPEngine::ApplySampler(std::vector<float>& logits,
                               float temperature, float top_p, int32_t top_k) {
    if (logits.empty()) return;

    // Apply temperature
    if (temperature > 0.0f && temperature != 1.0f) {
        for (auto& v : logits) v /= temperature;
    }

    // Softmax to probabilities
    float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (auto& v : logits) {
        v = std::exp(v - max_logit);
        sum += v;
    }
    if (sum > 0.0) {
        for (auto& v : logits) v /= static_cast<float>(sum);
    }

    // Top-k filtering
    if (top_k > 0 && static_cast<size_t>(top_k) < logits.size()) {
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](size_t a, size_t b) { return logits[a] > logits[b]; });

        std::vector<float> mask(logits.size(), 0.0f);
        for (int32_t i = 0; i < top_k; i++) {
            mask[indices[i]] = logits[indices[i]];
        }
        logits = std::move(mask);

        // Renormalize
        sum = 0.0;
        for (auto& v : logits) sum += v;
        if (sum > 0.0) for (auto& v : logits) v /= static_cast<float>(sum);
    }

    // Top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](size_t a, size_t b) { return logits[a] > logits[b]; });

        float cumsum = 0.0f;
        size_t cutoff = logits.size();
        for (size_t i = 0; i < indices.size(); i++) {
            cumsum += logits[indices[i]];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        std::vector<float> mask(logits.size(), 0.0f);
        for (size_t i = 0; i < cutoff && i < indices.size(); i++) {
            mask[indices[i]] = logits[indices[i]];
        }
        logits = std::move(mask);

        // Renormalize
        sum = 0.0;
        for (auto& v : logits) sum += v;
        if (sum > 0.0) for (auto& v : logits) v /= static_cast<float>(sum);
    }
}

} // namespace notllama
