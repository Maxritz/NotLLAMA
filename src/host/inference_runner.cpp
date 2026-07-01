#include "inference_runner.hpp"
#include <cstdio>
#include <cstring>

namespace notllama {

InferenceRunner::InferenceRunner(VkDevice device, VkPhysicalDevice physicalDevice,
                                 VkQueue queue, uint32_t queue_family_index,
                                 VulkanShaderLibrary* shader_lib,
                                 VulkanDescriptorManager* desc_mgr,
                                 RingAllocatorAdapter* allocator,
                                 uint32_t embed_dim)
    : device_(device), physical_device_(physicalDevice),
      queue_(queue), queue_family_index_(queue_family_index),
      shader_lib_(shader_lib), desc_mgr_(desc_mgr),
      allocator_(allocator), embed_dim_(embed_dim) {}

InferenceRunner::~InferenceRunner() {
    // Engine cleans up its own Vulkan resources
}

bool InferenceRunner::SetupModel(const std::string& model_id,
                                 MultiModelManager* model_mgr) {
    if (!model_mgr) {
        fprintf(stderr, "[InferenceRunner] model_mgr is null\n");
        return false;
    }

    auto* inst = model_mgr->GetModel(model_id);
    if (!inst) {
        fprintf(stderr, "[InferenceRunner] Model '%s' not found\n", model_id.c_str());
        return false;
    }
    if (!inst->adapter) {
        fprintf(stderr, "[InferenceRunner] Model '%s' has no adapter\n", model_id.c_str());
        return false;
    }
    if (!inst->kv_cache) {
        fprintf(stderr, "[InferenceRunner] Model '%s' has no KV cache\n", model_id.c_str());
        return false;
    }

    // Re-create engine if model changed (to ensure clean state)
    if (current_model_id_ != model_id || !engine_) {
        engine_.reset();
        engine_ = std::make_unique<VulkanComputeEngine>(
            device_, physical_device_, queue_, queue_family_index_,
            shader_lib_, desc_mgr_, allocator_, embed_dim_);

        current_model_ = inst->adapter.get();
        current_kv_cache_ = inst->kv_cache.get();
        current_max_seq_len_ = static_cast<uint32_t>(inst->max_context);
        current_model_id_ = model_id;

        engine_->SetModel(current_model_);
        engine_->SetKVCache(current_kv_cache_, current_max_seq_len_);

        fprintf(stderr, "[InferenceRunner] Model '%s' configured: layers=%zu embed=%u ctx=%u\n",
                model_id.c_str(), current_model_->GetNumLayers(),
                static_cast<uint32_t>(current_model_->GetEmbeddingDim()),
                current_max_seq_len_);
    }

    return true;
}

bool InferenceRunner::Prime(const std::string& model_id,
                            MultiModelManager* model_mgr,
                            const std::vector<uint32_t>& prompt_tokens) {
    if (prompt_tokens.empty()) {
        fprintf(stderr, "[InferenceRunner] Empty prompt\n");
        return false;
    }

    if (!SetupModel(model_id, model_mgr)) return false;

    // Reset engine state for new sequence
    engine_->ResetExecutionEngine();

    // Add the prompt tokens as sequence 0
    if (!engine_->AddSequence(0, prompt_tokens)) {
        fprintf(stderr, "[InferenceRunner] AddSequence failed\n");
        return false;
    }

    last_token_ = 0;
    tokens_generated_ = 0;
    primed_ = true;

    fprintf(stderr, "[InferenceRunner] Primed with %zu prompt tokens\n", prompt_tokens.size());
    return true;
}

uint32_t InferenceRunner::GenerateOne() {
    if (!primed_ || !engine_) {
        fprintf(stderr, "[InferenceRunner] Not primed. Call Prime() first.\n");
        return 0;
    }

    if (!RunUntilToken()) {
        return 0;
    }

    return last_token_;
}

std::vector<uint32_t> InferenceRunner::Generate(const std::string& model_id,
                                                MultiModelManager* model_mgr,
                                                const std::vector<uint32_t>& prompt_tokens,
                                                uint32_t max_new_tokens,
                                                uint32_t eos_token_id) {
    std::vector<uint32_t> result;
    result.reserve(max_new_tokens);

    if (!Prime(model_id, model_mgr, prompt_tokens)) {
        fprintf(stderr, "[InferenceRunner] Prime failed\n");
        return result;
    }

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < max_new_tokens; i++) {
        if (!RunUntilToken()) {
            fprintf(stderr, "[InferenceRunner] RunUntilToken failed at token %u\n", i);
            break;
        }

        if (last_token_ == 0) {
            fprintf(stderr, "[InferenceRunner] Got token 0, stopping\n");
            break;
        }

        result.push_back(last_token_);

        if (last_token_ == eos_token_id) {
            fprintf(stderr, "[InferenceRunner] EOS token %u reached\n", eos_token_id);
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double tok_per_sec = (elapsed_ms > 0) ? (result.size() * 1000.0 / elapsed_ms) : 0;

    fprintf(stderr, "[InferenceRunner] Generated %zu tokens in %.1f ms (%.1f tok/s)\n",
            result.size(), elapsed_ms, tok_per_sec);

    return result;
}

bool InferenceRunner::RunUntilToken() {
    if (!engine_) return false;

    // Phase values: 0=VALIDATE 1=LOAD_WEIGHTS 2=IDLE 3=EMBED ... 20=LM_HEAD 21=TOPK
    const int TOPK_PHASE = 21;
    const int IDLE_PHASE = 2;

    int prev_phase = engine_->GetPhase();
    int safety = 0;
    const int MAX_STEPS = 100000; // layers * phases + margin

    while (safety < MAX_STEPS) {
        if (!engine_->StepBatch()) {
            fprintf(stderr, "[InferenceRunner] StepBatch failed at phase=%d\n", engine_->GetPhase());
            return false;
        }

        int current_phase = engine_->GetPhase();

        // Detect TOPK -> IDLE transition: a token was just generated
        if (prev_phase == TOPK_PHASE && current_phase == IDLE_PHASE) {
            last_token_ = engine_->LastTokenId();
            tokens_generated_++;
            return true;
        }

        // Also handle the case where we're in IDLE and a token was set
        // (can happen on first pass through)
        if (current_phase == IDLE_PHASE && engine_->LastTokenId() != 0 &&
            engine_->LastTokenId() != last_token_) {
            last_token_ = engine_->LastTokenId();
            tokens_generated_++;
            return true;
        }

        prev_phase = current_phase;
        safety++;
    }

    fprintf(stderr, "[InferenceRunner] Safety limit (%d steps) exceeded\n", MAX_STEPS);
    return false;
}

bool InferenceRunner::IsReady() const {
    return engine_ != nullptr && current_model_ != nullptr;
}

void InferenceRunner::Reset() {
    if (engine_) {
        engine_->ResetExecutionEngine();
    }
    current_model_id_.clear();
    current_model_ = nullptr;
    current_kv_cache_ = nullptr;
    last_token_ = 0;
    tokens_generated_ = 0;
    primed_ = false;
}

} // namespace notllama
