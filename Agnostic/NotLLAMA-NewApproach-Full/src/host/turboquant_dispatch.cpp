#include "rdna4_turboquant_dispatch.hpp"
#include "rdna4_types.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_pipeline.hpp"
#include <algorithm>
#include <cstdint>

namespace rdna4 {

void loadTurboQuantPipelines(PipelineBuilder& pipeline) {
    std::string spvDir = "shaders/";
    pipeline.loadShader("dequant_turbo", spvDir + "dequant_turbo.spv");
    pipeline.createComputePipeline("dequant_turbo", sizeof(DequantTurboPushConstants));

    pipeline.loadShader("gemm_turbo", spvDir + "gemm_turbo.spv");
    pipeline.createComputePipeline("gemm_turbo", sizeof(GemmTurboPushConstants));
}

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
    bool sync)
{
    if (n == 0) return false;

    uint32_t totalWorkgroups = (n + blockSize - 1) / blockSize;
    if (totalWorkgroups == 0) totalWorkgroups = 1;

    uint32_t offset = 0;
    uint32_t elementsPerChunk = workgroupChunk * blockSize;

    while (offset < n) {
        uint32_t chunkSize = std::min(elementsPerChunk, n - offset);
        uint32_t chunkWg = (chunkSize + blockSize - 1) / blockSize;

        DequantTurboPushConstants pc = {};
        pc.addrSrc = srcAddr + (uint64_t)offset * bits / 8;
        pc.addrDst = dstAddr + (uint64_t)offset * sizeof(uint16_t);
        pc.n = chunkSize;
        pc.blockSize = blockSize;
        pc.bits = bits;
        pc.scaleBits = scaleBits;
        pc.zeroPoint = zeroPoint;
        pc.scale = 1.0f;

        scheduler->dispatch(pipeline.getPipeline("dequant_turbo"),
                           pipeline.getLayout("dequant_turbo"),
                           &pc, sizeof(pc), chunkWg, 1, 1);

        offset += chunkSize;
    }

    if (sync) scheduler->syncAllThrottled();
    return true;
}

bool dequantTurboWeightInBatch(
    Scheduler* scheduler,
    PipelineBuilder& pipeline,
    VkDeviceAddress srcAddr,
    VkDeviceAddress dstAddr,
    uint32_t n,
    uint32_t blockSize,
    uint32_t bits,
    uint32_t scaleBits,
    uint32_t zeroPoint)
{
    if (n == 0) return false;

    const uint32_t MAX_WG_PER_DISPATCH = 1024 * 1024;
    uint32_t elementsPerChunk = MAX_WG_PER_DISPATCH * blockSize;
    uint32_t offset = 0;

    while (offset < n) {
        uint32_t chunkSize = std::min(elementsPerChunk, n - offset);
        uint32_t chunkWg = (chunkSize + blockSize - 1) / blockSize;

        DequantTurboPushConstants pc = {};
        pc.addrSrc = srcAddr + (uint64_t)offset * bits / 8;
        pc.addrDst = dstAddr + (uint64_t)offset * sizeof(uint16_t);
        pc.n = chunkSize;
        pc.blockSize = blockSize;
        pc.bits = bits;
        pc.scaleBits = scaleBits;
        pc.zeroPoint = zeroPoint;
        pc.scale = 1.0f;

        scheduler->dispatchInBatch(pipeline.getPipeline("dequant_turbo"),
                                  pipeline.getLayout("dequant_turbo"),
                                  &pc, sizeof(pc), chunkWg, 1, 1);

        offset += chunkSize;
    }

    return true;
}

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
    float alpha)
{
    GemmTurboPushConstants pc = {};
    pc.addrA = addrA;
    pc.addrB = addrB;
    pc.addrC = addrC;
    pc.M = M;
    pc.K = K;
    pc.N = N;
    pc.blockSize = blockSize;
    pc.bits = bits;
    pc.scaleBits = scaleBits;
    pc.zeroPoint = zeroPoint;
    pc.alpha = alpha;

    uint32_t gx = (N + 31) / 32;
    uint32_t gy = (M + 31) / 32;

    scheduler->dispatch(pipeline.getPipeline("gemm_turbo"),
                       pipeline.getLayout("gemm_turbo"),
                       &pc, sizeof(pc), gx, gy, 1);
}

} // namespace rdna4
