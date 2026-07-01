#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace rdna4 {

class Scheduler;
class PipelineBuilder;

void dispatchCompressContext(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcKVCache,
    VkDeviceAddress dstKVCache,
    uint32_t srcLen,
    uint32_t dstLen,
    uint32_t numLayers);

void dispatchKVCacheQuantize(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcF16,
    VkDeviceAddress dstQuantized,
    uint32_t numElements,
    uint32_t bitsPerElement);

void dispatchKVCacheDequant(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcQuantized,
    VkDeviceAddress dstF16,
    uint32_t numElements,
    uint32_t bitsPerElement);

void loadCompressionPipelines(PipelineBuilder& pipeline);

} // namespace rdna4
