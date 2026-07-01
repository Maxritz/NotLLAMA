#pragma once
#include "engine/icompute_engine.hpp"
#include "engine/ishader_library.hpp"
#include "engine/idescriptor_manager.hpp"
#include "engine/imemory_allocator.hpp"
#include "engine/imodel.hpp"
#include "rdna4_kv_cache.hpp"
#include "rdna4_types.hpp"
#include <vulkan/vulkan.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace notllama {

class VulkanComputeEngine : public IComputeEngine {
public:
    VulkanComputeEngine(VkDevice device, VkPhysicalDevice physical_device,
                        VkQueue queue, uint32_t queue_family,
                        IShaderLibrary* shader_lib, IDescriptorManager* desc_mgr,
                        IMemoryAllocator* allocator, uint32_t embed_dim = 4096);
    ~VulkanComputeEngine() override;

    // Set model weights and KV cache: call before StepBatch
    void SetModel(IModel* model);
    void SetKVCache(rdna4::KVCacheManager* kv_cache, uint32_t max_seq_len = 4096);

    bool AddSequence(uint32_t seq_id, const std::vector<uint32_t>& tokens) override;
    void RemoveSequence(uint32_t seq_id) override;
    bool StepBatch() override;

    void SetMaxUtilization(float percent) override;
    void Throttle() override;
    void StartWatchdog() override;
    void StopWatchdog() override;
    WatchdogStatus GetLastFrameStatus() override;
    void ResetExecutionEngine() override;
    void EnableProfiling(bool enable) override;

    // Results: valid after TOPK phase
    uint32_t LastTokenId() const { return last_token_id_; }
    int GetPhase() const { return static_cast<int>(step_phase_); }

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueue queue_;
    uint32_t queue_family_;
    IShaderLibrary* shader_lib_;
    IDescriptorManager* desc_mgr_;
    IMemoryAllocator* allocator_;
    IModel* model_ = nullptr;
    class rdna4::KVCacheManager* kv_cache_ = nullptr;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    // Model dimensions (set by SetModel)
    uint32_t n_layers_ = 0;
    uint32_t n_heads_ = 0;
    uint32_t n_kv_heads_ = 0;
    uint32_t head_dim_ = 0;
    uint32_t hidden_dim_ = 0;
    uint32_t vocab_size_ = 0;
    uint32_t max_seq_len_ = 4096;
    uint32_t embed_dim_ = 4096;

    // Per-layer weight device addresses (flat buffer)
    // Layout for each layer (each entry = device address):
    //   attn_norm, q_proj, k_proj, v_proj, attn_out, ffn_norm, ffn_up, ffn_gate, ffn_down
    static constexpr uint32_t WEIGHTS_PER_LAYER = 9;
    std::vector<std::array<VkDeviceAddress, WEIGHTS_PER_LAYER>> layer_weights_;
    std::vector<std::array<rdna4::QuantFormat, WEIGHTS_PER_LAYER>> layer_weight_formats_;

    // Layer weight GPU allocations (for cleanup)
    std::vector<GpuAllocation> layer_weight_allocs_;

    // MoE layer detection
    std::vector<bool> layer_is_moe_;
    std::vector<VkDeviceAddress> layer_moe_gate_;
    std::vector<rdna4::QuantFormat> layer_moe_gate_fmt_;

    // Global weight addresses
    VkDeviceAddress addr_embed_ = 0;   // token_embd.weight [vocab, dim]
    VkDeviceAddress addr_lm_head_ = 0; // output.weight [dim, vocab]
    VkDeviceAddress addr_output_norm_ = 0;
    rdna4::QuantFormat lm_head_dtype_ = rdna4::QuantFormat::F32;

    // Scratch buffers
    GpuAllocation scratch_q_;      // [n_heads * head_dim]
    GpuAllocation scratch_k_;      // [n_kv_heads * head_dim]
    GpuAllocation scratch_v_;      // [n_kv_heads * head_dim]
    GpuAllocation scratch_attn_;   // [n_heads * head_dim] (attention output)
    GpuAllocation scratch_attn_out_;// [embed_dim] (attention-output GEMM output)
    GpuAllocation scratch_norm_;   // [embed_dim] (RMS_NORM output, preserves hidden_state as residual)
    GpuAllocation scratch_ffn_;    // [hidden_dim] (FFN intermediate)
    GpuAllocation scratch_ffn_gate_;// [hidden_dim] (gate output)
    GpuAllocation scratch_logits_; // [vocab_size]

    GpuAllocation rms_weight_;     // reused for both attn_norm and ffn_norm
    bool rms_weight_loaded_ = false;

    // TOPK output buffer (host-visible, stores {tokenId, tokenProb})
    VkBuffer topk_output_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory topk_output_mem_ = VK_NULL_HANDLE;
    VkDeviceAddress topk_output_addr_ = 0;
    uint32_t* topk_output_mapped_ = nullptr;

    // Sequence state
    struct ActiveSequence {
        std::vector<uint32_t> tokens;
        uint32_t position = 0;
        GpuAllocation hidden_state;
    };
    std::unordered_map<uint32_t, ActiveSequence> sequences_;

    // Pipeline progress
    enum class StepPhase {
        VALIDATE, LOAD_WEIGHTS, IDLE,
        EMBED,
        LAYER_RMS_ATTN, LAYER_QKV_Q, LAYER_QKV_K, LAYER_QKV_V,
        LAYER_ROPE,
        LAYER_KV_CACHE_WRITE, LAYER_FLASH_ATTN, LAYER_ATTN_OUT, LAYER_ATTN_RESIDUAL,
        LAYER_RMS_FFN,
        LAYER_FFN_UP, LAYER_FFN_GATE, LAYER_SILU_MUL, LAYER_FFN_DOWN, LAYER_FFN_RESIDUAL,
        LAYER_NEXT,
        LM_HEAD, TOPK
    };
    StepPhase step_phase_ = StepPhase::VALIDATE;
    uint32_t current_layer_ = 0;
    uint32_t current_seq_id_ = 0;
    uint32_t next_seq_index_ = 0;
    uint32_t last_token_id_ = 0;
    uint64_t xors_state_ = 12345;

    bool rms_norm_validated_ = false;
    bool model_weights_loaded_ = false;
    bool scratch_allocated_ = false;

    float max_utilization_ = 0.8f;
    bool profiling_ = false;
    bool watchdog_running_ = false;
    WatchdogStatus last_status_ = WatchdogStatus::OK;

    // Resource management
    bool EnsureResources();
    bool ValidateRmsNormDispatch();
    GpuAllocation AllocScratch(size_t size);
    bool AllocScratchBuffers();
    bool LoadModelWeights();

    // Dispatch helpers (all record into cmd_buffer_ + submit + wait)
    bool DispatchEmbed(uint32_t token_id, uint32_t token_pos, VkDeviceAddress hidden_addr);
    bool DispatchRmsNorm(VkDeviceAddress in_addr, VkDeviceAddress weight_addr,
                         VkDeviceAddress out_addr, uint32_t row_size, uint32_t n_rows);
    bool DispatchGemm(VkDeviceAddress a_addr, VkDeviceAddress b_addr, VkDeviceAddress c_addr,
                       uint32_t M, uint32_t N, uint32_t K, bool transB = false,
                       rdna4::QuantFormat weightFormat = rdna4::QuantFormat::F32);
    bool DispatchAdd(VkDeviceAddress a_addr, VkDeviceAddress b_addr, VkDeviceAddress c_addr,
                     uint32_t n_elements);
    bool DispatchRope(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
                      uint32_t seq_pos, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim);
    bool DispatchFlashAttn(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
                            VkDeviceAddress v_addr, VkDeviceAddress out_addr,
                            uint32_t seq_len, uint32_t n_q_rows);
    bool DispatchSiluMul(VkDeviceAddress gate_addr, VkDeviceAddress up_addr,
                          VkDeviceAddress out_addr, uint32_t n_elements);
    bool DispatchTopK(VkDeviceAddress logits_addr, VkDeviceAddress output_addr,
                       VkDeviceAddress scratch_addr);
    bool DispatchKvCacheWrite(VkDeviceAddress k_in, VkDeviceAddress v_in,
                               VkDeviceAddress k_cache, VkDeviceAddress v_cache,
                               uint32_t seq_pos, uint32_t max_seq);
};

} // namespace notllama
