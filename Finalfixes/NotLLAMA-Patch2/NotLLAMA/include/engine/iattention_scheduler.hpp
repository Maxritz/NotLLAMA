#pragma once
#include "types.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace notllama {

class IMemoryAllocator;

class IAttentionScheduler {
public:
    virtual ~IAttentionScheduler() = default;

    virtual bool InitializeKVSlab(size_t slab_size, IMemoryAllocator* allocator) = 0;

    virtual uint32_t AllocateBlock() = 0;
    virtual void FreeBlock(uint32_t block_id) = 0;

    virtual void SetSequenceBlockTable(uint32_t seq_id, const std::vector<uint32_t>& block_table) = 0;
    virtual void RemoveSequence(uint32_t seq_id) = 0;

    virtual void RotateAndCache(uint32_t layer, uint32_t seq_id, uint32_t token_pos,
                                const TensorMeta& K, const TensorMeta& V) = 0;

    virtual void FlashDecodeBatch(uint32_t layer, const BatchMetadata& batch,
                                  const TensorMeta& Q, TensorMeta& output) = 0;

    virtual void EvictBlocksLRU(float keep_percent, VkQueue transfer_queue, VkSemaphore timeline_semaphore) = 0;
    virtual uint32_t GetRemainingBlocks() const = 0;
};

} // namespace notllama
