#pragma once
#include "engine/iattention_scheduler.hpp"
#include "engine/ishader_library.hpp"
#include "rdna4_kv_cache.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace notllama {

// Bridge adapter: maps IAttentionScheduler onto rdna4::KVCacheManager.
// FlashDecodeBatch dispatches flash_attention.comp with KV cache buffers.
class KVCacheAdapter : public IAttentionScheduler {
public:
    KVCacheAdapter(VkDevice device, VkPhysicalDevice physical_device,
                   uint32_t max_seq, uint32_t n_layers, uint32_t n_kv_heads, uint32_t head_dim,
                   IShaderLibrary* shader_lib = nullptr, VkQueue queue = VK_NULL_HANDLE);
    ~KVCacheAdapter() override;

    bool InitializeKVSlab(size_t slab_size, IMemoryAllocator* allocator) override;

    uint32_t AllocateBlock() override;
    void FreeBlock(uint32_t block_id) override;

    void SetSequenceBlockTable(uint32_t seq_id, const std::vector<uint32_t>& block_table) override;
    void RemoveSequence(uint32_t seq_id) override;

    void RotateAndCache(uint32_t layer, uint32_t seq_id, uint32_t token_pos,
                        const TensorMeta& K, const TensorMeta& V) override;

    void FlashDecodeBatch(uint32_t layer, const BatchMetadata& batch,
                          const TensorMeta& Q, TensorMeta& output) override;

    void EvictBlocksLRU(float keep_percent, VkQueue transfer_queue, VkSemaphore timeline_semaphore) override;
    uint32_t GetRemainingBlocks() const override;

private:
    std::unique_ptr<rdna4::KVCacheManager> kv_;
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    IShaderLibrary* shader_lib_;
    VkQueue queue_;
    uint32_t max_seq_;
    uint32_t n_layers_;
    uint32_t n_kv_heads_;
    uint32_t head_dim_;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool resources_ready_ = false;

    // Block pool management
    struct BlockEntry {
        bool in_use = false;
        uint64_t last_access = 0;
    };
    std::vector<BlockEntry> block_pool_;
    uint32_t next_block_id_ = 0;

    // Per-sequence block tracking
    std::unordered_map<uint32_t, std::vector<uint32_t>> seq_blocks_;
    std::unordered_map<uint32_t, uint64_t> seq_last_access_;

    bool EnsureResources();
    bool EnsureKvCacheWritePipeline(VkPipeline& pipeline);
    bool DispatchKvCacheWrite(VkPipeline pipeline, uint32_t layer,
                               VkDeviceAddress k_in, VkDeviceAddress v_in,
                               VkDeviceAddress k_cache, VkDeviceAddress v_cache,
                               uint32_t seq_pos);
};

} // namespace notllama
