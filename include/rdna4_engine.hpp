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

    InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                      PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a);

    bool initDequantBuffer();  // Allocate reusable dequant staging buffer
    void cleanupDequantBuffer();

    uint32_t forward(uint32_t tokenId, uint32_t seqPos);

    // Speculative decode: draft N tokens, verify in parallel
    std::vector<uint32_t> forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft);

    std::vector<uint32_t> generate(const std::string& prompt, uint32_t maxTokens);

    // Generate with speculative decoding
    std::vector<uint32_t> generateSpeculative(const std::string& prompt, uint32_t maxTokens, uint32_t nDraft);

    uint32_t sampleArgmax(const float* logits, uint32_t vocabSize);

private:
    // Fast draft forward using fewer layers (e.g., first 2 layers only)
    uint32_t draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers);

    // Verify a draft token against the full model
    bool verifyDraftToken(uint32_t draftToken, uint32_t expectedToken, uint32_t seqPos);
};

} // namespace rdna4
