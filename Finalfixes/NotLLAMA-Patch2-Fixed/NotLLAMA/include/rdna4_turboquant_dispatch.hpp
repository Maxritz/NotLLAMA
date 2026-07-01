#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace rdna4 {

class Scheduler;
class PipelineBuilder;

bool dequantTurboWeight(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcAddr,
    VkDeviceAddress dstAddr,
    uint32_t n,
    uint32_t blockSize,
    uint32_t bits,
    uint32_t scaleBits,
    uint32_t zeroPoint,
    uint32_t workgroupChunk,
    bool sync);

bool dequantTurboWeightInBatch(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcAddr,
    VkDeviceAddress dstAddr,
    uint32_t n,
    uint32_t blockSize,
    uint32_t bits,
    uint32_t scaleBits,
    uint32_t zeroPoint);

void dispatchGemmTurbo(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress addrA,
    VkDeviceAddress addrB,
    VkDeviceAddress addrC,
    uint32_t M, uint32_t K, uint32_t N,
    uint32_t blockSize,
    uint32_t bits,
    uint32_t scaleBits,
    uint32_t zeroPoint,
    float alpha);

void loadTurboQuantPipelines(PipelineBuilder& pipeline);

} // namespace rdna4
