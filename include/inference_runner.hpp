#pragma once
#include "engine/engine.hpp"
#include "multi_model_manager.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace notllama {

// InferenceRunner wraps VulkanComputeEngine to provide a clean,
// high-level Generate() API. It runs the real GPU forward pass
// via StepBatch() instead of returning fake tokens.
class InferenceRunner {
public:
    InferenceRunner(VkDevice device, VkPhysicalDevice physicalDevice,
                    VkQueue queue, uint32_t queue_family_index,
                    VulkanShaderLibrary* shader_lib,
                    VulkanDescriptorManager* desc_mgr,
                    RingAllocatorAdapter* allocator,
                    uint32_t embed_dim = 4096);
    ~InferenceRunner();

    // Generate tokens using the real Vulkan compute engine.
    // Returns generated token IDs (not including prompt tokens).
    std::vector<uint32_t> Generate(const std::string& model_id,
                                   MultiModelManager* model_mgr,
                                   const std::vector<uint32_t>& prompt_tokens,
                                   uint32_t max_new_tokens,
                                   uint32_t eos_token_id = 2);

    // Generate a single token. Returns the token ID, or 0 on error.
    // Must call Prime() first.
    uint32_t GenerateOne();

    // Prime the engine with prompt tokens. Call before GenerateOne().
    // Returns true on success.
    bool Prime(const std::string& model_id,
               MultiModelManager* model_mgr,
               const std::vector<uint32_t>& prompt_tokens);

    // Check if the engine is ready for inference
    bool IsReady() const;

    // Reset the engine state (clears sequences)
    void Reset();

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;
    uint32_t queue_family_index_;
    VulkanShaderLibrary* shader_lib_;
    VulkanDescriptorManager* desc_mgr_;
    RingAllocatorAdapter* allocator_;
    uint32_t embed_dim_;

    std::unique_ptr<VulkanComputeEngine> engine_;

    // Current model being used
    std::string current_model_id_;
    IModel* current_model_ = nullptr;
    rdna4::KVCacheManager* current_kv_cache_ = nullptr;
    uint32_t current_max_seq_len_ = 4096;

    // Internal state for token-by-token generation
    uint32_t last_token_ = 0;
    uint32_t tokens_generated_ = 0;
    bool primed_ = false;

    // Run StepBatch until a token is produced (phase transitions from TOPK to IDLE)
    // Returns true if a new token was generated, false on error
    bool RunUntilToken();

    // Setup the engine for a specific model
    bool SetupModel(const std::string& model_id, MultiModelManager* model_mgr);
};

} // namespace notllama
