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

    InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                      PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a);

    // Release all engine-owned GPU resources. Safe to call multiple times.
    void cleanup();

    uint32_t forward(uint32_t tokenId, uint32_t seqPos);

    // Address of the logits buffer from the most recent forward().
    uint64_t lastLogitsAddr = 0;
    size_t lastLogitsOffset = 0;

    std::vector<uint32_t> generate(const std::string& prompt, uint32_t maxTokens);

    uint32_t sampleArgmax(const float* logits, uint32_t vocabSize);

private:
    uint32_t forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers);
};

} // namespace rdna4
