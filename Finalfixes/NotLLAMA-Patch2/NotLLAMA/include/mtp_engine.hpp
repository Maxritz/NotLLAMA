#pragma once
#include "multi_model_manager.hpp"
#include <vector>
#include <cstdint>
#include <random>

namespace notllama {

// Multi-Token Prediction (MTP) engine
// Predicts N tokens per forward pass using speculative decoding
// Can use a draft model for faster token generation
struct MTPConfig {
    uint32_t n_predict = 1;          // Tokens to predict per forward (1 = normal)
    uint32_t n_draft = 4;            // Draft tokens to generate speculatively
    float draft_accept_threshold = 0.9f;
    bool use_draft_model = false;    // Use a separate small model for drafting
    std::string draft_model_id;      // Which model to use as drafter
    bool verify_drafts = true;       // Verify draft tokens with target model
};

class MTPEngine {
public:
    MTPEngine(MultiModelManager* manager);

    void SetConfig(const MTPConfig& cfg) { config_ = cfg; }
    const MTPConfig& GetConfig() const { return config_; }

    // Generate the next token using MTP
    // Returns the accepted token ID
    uint32_t GenerateNext(const std::string& model_id,
                           const std::vector<uint32_t>& prompt_tokens,
                           uint32_t position);

    // Generate multiple tokens autoregressively with MTP acceleration
    std::vector<uint32_t> Generate(const std::string& model_id,
                                     const std::vector<uint32_t>& prompt_tokens,
                                     uint32_t max_new_tokens,
                                     uint32_t eos_token = 2);

    // Speculative decode: draft model generates candidates, target verifies
    // Returns number of accepted tokens (0 = none accepted, fall back to target)
    uint32_t SpeculativeDecode(const std::string& target_model_id,
                                const std::string& draft_model_id,
                                const std::vector<uint32_t>& context,
                                uint32_t position,
                                std::vector<uint32_t>& accepted_tokens);

    // Temperature sampling from logits
    uint32_t Sample(const float* logits, uint32_t vocab_size,
                     float temperature, float top_p, int32_t top_k);

    // Set seed for reproducibility
    void SetSeed(uint32_t seed) { rng_.seed(seed); }

private:
    MultiModelManager* manager_;
    MTPConfig config_;
    std::mt19937 rng_;

    // Draft N tokens using the draft model
    std::vector<uint32_t> DraftTokens(ModelInstance* draft_model,
                                        const std::vector<uint32_t>& context,
                                        uint32_t position, uint32_t n_draft);

    // Verify draft tokens against target model
    uint32_t VerifyDrafts(ModelInstance* target_model,
                           const std::vector<uint32_t>& context,
                           const std::vector<uint32_t>& draft_tokens,
                           uint32_t position);

    // Apply temperature + top-p + top-k filtering
    void ApplySampler(std::vector<float>& logits,
                       float temperature, float top_p, int32_t top_k);
};

} // namespace notllama
