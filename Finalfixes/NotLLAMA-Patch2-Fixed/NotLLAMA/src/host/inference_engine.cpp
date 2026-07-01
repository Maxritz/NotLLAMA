#include "rdna4_engine.hpp"
#include <cstdio>

namespace rdna4 {

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

bool InferenceEngine::initDequantBuffer() { return true; }
void InferenceEngine::cleanupDequantBuffer() {}
bool InferenceEngine::initEmbedCache() { return true; }
void InferenceEngine::cleanupEmbedCache() {}
bool InferenceEngine::initWeightBuffer() { return true; }
void InferenceEngine::cleanupWeightBuffer() {}
bool InferenceEngine::initLayerParams() { return true; }
bool InferenceEngine::initKernelEntryBuffers() { return true; }
void InferenceEngine::cleanupKernelEntryBuffers() {}
void InferenceEngine::cleanup() {}

uint32_t InferenceEngine::forward(uint32_t /*tokenId*/, uint32_t /*seqPos*/) {
    fprintf(stderr, "[InferenceEngine] forward() is disabled: GPU inference code pending rewrite\n");
    return 0;
}

std::vector<uint32_t> InferenceEngine::generate(const std::string& /*prompt*/, uint32_t /*maxTokens*/) {
    fprintf(stderr, "[InferenceEngine] generate() is disabled: GPU inference code pending rewrite\n");
    return {};
}

std::vector<uint32_t> InferenceEngine::generateSpeculative(const std::string& /*prompt*/, uint32_t /*maxTokens*/, uint32_t /*nDraft*/) {
    fprintf(stderr, "[InferenceEngine] generateSpeculative() is disabled: GPU inference code pending rewrite\n");
    return {};
}

uint32_t InferenceEngine::forwardPartial(uint32_t /*tokenId*/, uint32_t /*seqPos*/, uint32_t /*maxLayers*/) {
    fprintf(stderr, "[InferenceEngine] forwardPartial() is disabled: GPU inference code pending rewrite\n");
    return 0;
}

uint32_t InferenceEngine::forwardKernelEntry(uint32_t /*tokenId*/, uint32_t /*seqPos*/) {
    fprintf(stderr, "[InferenceEngine] forwardKernelEntry() is disabled: GPU inference code pending rewrite\n");
    return 0;
}

uint32_t InferenceEngine::draftForward(uint32_t /*tokenId*/, uint32_t /*seqPos*/, uint32_t /*nLayers*/) {
    return 0;
}

uint32_t InferenceEngine::verifyDraftToken(uint32_t /*draftToken*/, uint32_t /*seqPos*/) {
    return 0;
}

uint32_t InferenceEngine::sampleGpu(uint64_t /*logitsAddr*/, uint32_t /*vocabSize*/, uint64_t /*sampleOutAddr*/, uint64_t /*scratchAddr*/) {
    return 0;
}

uint32_t InferenceEngine::sampleArgmax(const float* logits, uint32_t vocabSize) {
    if (!logits || vocabSize == 0) return 0;
    uint32_t best = 0;
    float bestVal = logits[0];
    for (uint32_t i = 1; i < vocabSize; ++i) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            best = i;
        }
    }
    return best;
}

} // namespace rdna4
