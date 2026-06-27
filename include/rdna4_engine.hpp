#pragma once
#include "rdna4.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_kv_cache.hpp"
#include "rdna4_pipeline.hpp"
#include "rdna4_tokenizer.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include <string>
#include <vector>

namespace rdna4 {

class InferenceEngine {
public:
    VulkanContext* ctx;
    ModelDesc* model;
    KVCacheManager* kvCache;
    PipelineBuilder* pipelines;
    Tokenizer* tokenizer;
    Scheduler* scheduler;
    RingAllocator* allocator;

    // Dequantization staging buffer (reused for every dequantize dispatch)
    VkBuffer dequantBuffer = VK_NULL_HANDLE;
    VkDeviceMemory dequantMemory = VK_NULL_HANDLE;
    VkDeviceAddress dequantAddr = 0;
    size_t dequantCapacity = 0;

    // Persistent embedding cache (dequantized once, reused across all tokens)
    VkBuffer embedCacheBuffer = VK_NULL_HANDLE;
    VkDeviceMemory embedCacheMemory = VK_NULL_HANDLE;
    VkDeviceAddress embedCacheAddr = 0;
    size_t embedCacheSize = 0;
    bool embedCacheReady = false;

    InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                      PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a);

    bool initDequantBuffer();  // Allocate reusable dequant staging buffer
    void cleanupDequantBuffer();
    bool initEmbedCache();  // Allocate persistent embedding cache
    void cleanupEmbedCache();

    uint32_t forward(uint32_t tokenId, uint32_t seqPos);

    // Speculative decode: draft N tokens, verify in parallel
    std::vector<uint32_t> forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft);

    std::vector<uint32_t> generate(const std::string& prompt, uint32_t maxTokens);

    // Generate with speculative decoding
    std::vector<uint32_t> generateSpeculative(const std::string& prompt, uint32_t maxTokens, uint32_t nDraft);

    uint32_t sampleGpu(uint64_t logitsAddr, uint32_t vocabSize, uint64_t sampleOutAddr, uint64_t scratchAddr);
    uint32_t sampleArgmax(const float* logits, uint32_t vocabSize);

private:
    // Core forward pass with optional layer count limit
    uint32_t forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers);

    // Fast draft forward using fewer layers (e.g., first 2 layers only)
    uint32_t draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers);

    // Verify a draft token against the full model — returns predicted next token
    uint32_t verifyDraftToken(uint32_t draftToken, uint32_t seqPos);
};

} // namespace rdna4
