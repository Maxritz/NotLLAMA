#include "rdna4_compression_dispatch.hpp"
#include "rdna4_compression.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_pipeline.hpp"
#include <iostream>

namespace rdna4 {

void loadCompressionPipelines(PipelineBuilder& pipeline) {
    // TODO: implement when compression shaders are ready
    (void)pipeline;
}

void dispatchCompressContext(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcKVCache,
    VkDeviceAddress dstKVCache,
    uint32_t srcLen,
    uint32_t dstLen,
    uint32_t numLayers)
{
    // TODO: implement when compression shaders are ready
    (void)scheduler;
    (void)pipeline;
    (void)srcKVCache;
    (void)dstKVCache;
    (void)srcLen;
    (void)dstLen;
    (void)numLayers;
    std::cerr << "dispatchCompressContext: not yet implemented\n";
}

void dispatchKVCacheQuantize(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcF16,
    VkDeviceAddress dstQuantized,
    uint32_t numElements,
    uint32_t bitsPerElement)
{
    // TODO: implement when compression shaders are ready
    (void)scheduler;
    (void)pipeline;
    (void)srcF16;
    (void)dstQuantized;
    (void)numElements;
    (void)bitsPerElement;
    std::cerr << "dispatchKVCacheQuantize: not yet implemented\n";
}

void dispatchKVCacheDequant(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcQuantized,
    VkDeviceAddress dstF16,
    uint32_t numElements,
    uint32_t bitsPerElement)
{
    // TODO: implement when compression shaders are ready
    (void)scheduler;
    (void)pipeline;
    (void)srcQuantized;
    (void)dstF16;
    (void)numElements;
    (void)bitsPerElement;
    std::cerr << "dispatchKVCacheDequant: not yet implemented\n";
}

} // namespace rdna4
