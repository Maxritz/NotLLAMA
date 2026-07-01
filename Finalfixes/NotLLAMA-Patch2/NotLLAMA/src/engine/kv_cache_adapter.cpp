#include "engine/kv_cache_adapter.hpp"
#include "engine/imemory_allocator.hpp"
#include "rdna4_types.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace notllama {

KVCacheAdapter::KVCacheAdapter(VkDevice device, VkPhysicalDevice physical_device,
                               uint32_t max_seq, uint32_t n_layers,
                               uint32_t n_kv_heads, uint32_t head_dim,
                               IShaderLibrary* shader_lib, VkQueue queue)
    : device_(device), physical_device_(physical_device),
      shader_lib_(shader_lib), queue_(queue),
      max_seq_(max_seq), n_layers_(n_layers), n_kv_heads_(n_kv_heads), head_dim_(head_dim) {}

KVCacheAdapter::~KVCacheAdapter() {
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (cmd_buffer_ != VK_NULL_HANDLE && cmd_pool_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buffer_);
    if (cmd_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (kv_) kv_->free();
}

bool KVCacheAdapter::EnsureResources() {
    if (resources_ready_) return true;

    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = 0;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &pool_ci, nullptr, &cmd_pool_) != VK_SUCCESS)
        return false;

    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = cmd_pool_;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &cmd_ai, &cmd_buffer_) != VK_SUCCESS)
        return false;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_, &fci, nullptr, &fence_) != VK_SUCCESS)
        return false;

    resources_ready_ = true;
    return true;
}

bool KVCacheAdapter::InitializeKVSlab(size_t slab_size, IMemoryAllocator* allocator) {
    (void)slab_size;
    (void)allocator;
    kv_ = std::make_unique<rdna4::KVCacheManager>(device_, physical_device_,
                                                   max_seq_, n_layers_,
                                                   n_kv_heads_, head_dim_);
    if (!kv_->allocate()) return false;
    // Pre-allocate block pool: one block = 32 tokens (tiled KV pattern)
    uint32_t num_blocks = (max_seq_ + 31) / 32;
    block_pool_.resize(num_blocks);
    return true;
}

uint32_t KVCacheAdapter::AllocateBlock() {
    for (uint32_t i = 0; i < block_pool_.size(); i++) {
        if (!block_pool_[i].in_use) {
            block_pool_[i].in_use = true;
            block_pool_[i].last_access = 0;
            return i;
        }
    }
    // Pool exhausted
    return UINT32_MAX;
}

void KVCacheAdapter::FreeBlock(uint32_t block_id) {
    if (block_id < block_pool_.size())
        block_pool_[block_id].in_use = false;
}

void KVCacheAdapter::SetSequenceBlockTable(uint32_t seq_id, const std::vector<uint32_t>& block_table) {
    seq_blocks_[seq_id] = block_table;
    seq_last_access_[seq_id] = 0;
}

void KVCacheAdapter::RemoveSequence(uint32_t seq_id) {
    auto it = seq_blocks_.find(seq_id);
    if (it != seq_blocks_.end()) {
        for (uint32_t bid : it->second)
            FreeBlock(bid);
        seq_blocks_.erase(it);
    }
    seq_last_access_.erase(seq_id);
}

bool KVCacheAdapter::EnsureKvCacheWritePipeline(VkPipeline& pipeline) {
    if (!shader_lib_) return false;
    SpecializationMap spec{};
    spec.subgroup_size = 32;
    spec.head_dim = head_dim_;
    pipeline = shader_lib_->GetPipeline(KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
    return pipeline != VK_NULL_HANDLE;
}

bool KVCacheAdapter::DispatchKvCacheWrite(VkPipeline pipeline, uint32_t layer,
                                           VkDeviceAddress k_in, VkDeviceAddress v_in,
                                           VkDeviceAddress k_cache, VkDeviceAddress v_cache,
                                           uint32_t seq_pos) {
    if (!EnsureResources() || !kv_ || layer >= n_layers_) return false;

    if (!shader_lib_) return false;
    SpecializationMap spec{};
    spec.subgroup_size = 32;
    spec.head_dim = head_dim_;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(
        KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
    if (pl == VK_NULL_HANDLE) return false;

    rdna4::KVCacheWritePushConstants push{};
    push.addrKIn = k_in;
    push.addrVIn = v_in;
    push.addrKCache = k_cache;
    push.addrVCache = v_cache;
    push.seqPos = seq_pos;
    push.headDim = head_dim_;
    push.nKvHeads = n_kv_heads_;
    push.maxSeq = max_seq_;

    {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmd_buffer_, &bi) != VK_SUCCESS) return false;

        vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        uint32_t total = n_kv_heads_ * head_dim_;
        vkCmdDispatch(cmd_buffer_, (total + 31) / 32, 1, 1);

        if (vkEndCommandBuffer(cmd_buffer_) != VK_SUCCESS) return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

void KVCacheAdapter::RotateAndCache(uint32_t layer, uint32_t seq_id, uint32_t token_pos,
                                    const TensorMeta& K, const TensorMeta& V) {
    if (!kv_ || !shader_lib_ || layer >= n_layers_) return;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (!EnsureKvCacheWritePipeline(pipeline)) return;

    VkDeviceAddress k_cache = kv_->getKBufferAddress(layer);
    VkDeviceAddress v_cache = kv_->getVBufferAddress(layer);

    if (!DispatchKvCacheWrite(pipeline, layer,
                              K.alloc.device_address, V.alloc.device_address,
                              k_cache, v_cache, token_pos))
        return;

    kv_->incrementSeqLen(layer);

    // Update LRU tracking
    seq_last_access_[seq_id] = token_pos;
}

void KVCacheAdapter::FlashDecodeBatch(uint32_t layer, const BatchMetadata& batch,
                                      const TensorMeta& Q, TensorMeta& output) {
    if (!kv_ || !shader_lib_ || layer >= n_layers_) return;
    if (!EnsureResources()) return;

    uint32_t seq_len = kv_->getSeqLen(layer);
    if (seq_len == 0) return;

    rdna4::FlashAttentionPushConstants push{};
    push.addrQ = Q.alloc.device_address;
    push.addrKCache = kv_->getKBufferAddress(layer);
    push.addrVCache = kv_->getVBufferAddress(layer);
    push.addrOut = output.alloc.device_address;
    push.seqLen = seq_len;
    push.headDim = head_dim_;
    push.qRowStart = 0;
    push.qRowCount = batch.num_active_sequences;
    push.kvTileSize = 32;
    push.invSqrtHeadDim = 1.0f / sqrtf(static_cast<float>(head_dim_));

    SpecializationMap spec{};
    spec.subgroup_size = 32;
    spec.head_dim = head_dim_;

    VkPipeline pipeline = shader_lib_->GetPipeline(
        KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
    if (pipeline == VK_NULL_HANDLE) {
        fprintf(stderr, "FlashDecodeBatch: failed to get FLASH_ATTN pipeline\n");
        return;
    }

    // Each workgroup handles 32 query rows (one per thread).
    // Multiple dispatches needed when qRowCount > 32.
    VkPipelineLayout pipeline_pl = shader_lib_->GetPipelineLayout(
        KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
    if (pipeline_pl == VK_NULL_HANDLE) return;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(cmd_buffer_, &bi) != VK_SUCCESS) return;

    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    uint32_t remaining = push.qRowCount;
    uint32_t start = push.qRowStart;
    while (remaining > 0) {
        uint32_t chunk = remaining > 32 ? 32 : remaining;
        push.qRowStart = start;
        push.qRowCount = chunk;
        vkCmdPushConstants(cmd_buffer_, pipeline_pl, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(cmd_buffer_, 1, 1, 1);
        start += chunk;
        remaining -= chunk;
    }

    if (vkEndCommandBuffer(cmd_buffer_) != VK_SUCCESS) return;

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buffer_;

    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
}

void KVCacheAdapter::EvictBlocksLRU(float keep_percent, VkQueue transfer_queue, VkSemaphore timeline_semaphore) {
    (void)transfer_queue;
    (void)timeline_semaphore;
    if (block_pool_.empty()) return;

    // Sort sequences by last access time, evict oldest
    uint32_t total = (uint32_t)block_pool_.size();
    uint32_t keep = (uint32_t)(total * keep_percent);
    if (keep >= total) return;
    uint32_t evict = total - keep;

    // Build sorted list of sequences by last access
    std::vector<std::pair<uint64_t, uint32_t>> sorted;
    for (auto& [seq_id, last_access] : seq_last_access_)
        sorted.push_back({last_access, seq_id});
    std::sort(sorted.begin(), sorted.end());

    // Evict oldest sequences until we've freed enough blocks
    uint32_t freed = 0;
    for (auto& [_, seq_id] : sorted) {
        if (freed >= evict) break;
        RemoveSequence(seq_id);
        freed++;
    }
}

uint32_t KVCacheAdapter::GetRemainingBlocks() const {
    uint32_t count = 0;
    for (auto& e : block_pool_)
        if (!e.in_use) count++;
    return count;
}

} // namespace notllama
