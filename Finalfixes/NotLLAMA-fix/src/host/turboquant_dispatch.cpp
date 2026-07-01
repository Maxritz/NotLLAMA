#include "rdna4_turboquant_dispatch.hpp"
#include "rdna4_types.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_pipeline.hpp"
#include "rdna4_scheduler.hpp"
#include <cstdio>

namespace rdna4 {

void loadTurboQuantPipelines(PipelineBuilder&) {
    fprintf(stderr, "[TurboQuant] loadTurboQuantPipelines disabled: code pending rewrite\n");
}

bool dequantTurboWeight(Scheduler*, PipelineBuilder&, VkDeviceAddress, VkDeviceAddress,
                        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool) {
    fprintf(stderr, "[TurboQuant] dequantTurboWeight disabled: code pending rewrite\n");
    return false;
}

bool dequantTurboWeightInBatch(Scheduler*, PipelineBuilder&, VkDeviceAddress, VkDeviceAddress,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
    fprintf(stderr, "[TurboQuant] dequantTurboWeightInBatch disabled: code pending rewrite\n");
    return false;
}

void dispatchGemmTurbo(Scheduler*, PipelineBuilder&, VkDeviceAddress, VkDeviceAddress, VkDeviceAddress,
                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) {
    fprintf(stderr, "[TurboQuant] dispatchGemmTurbo disabled: code pending rewrite\n");
}

} // namespace rdna4
