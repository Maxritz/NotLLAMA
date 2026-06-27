#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <string>

namespace rdna4 {

class Profiler;

class Scheduler {
public:
    VkDevice device;
    VkQueue queues[4];
    VkCommandPool cmdPools[4];
    uint32_t queueFamilyIndex;

    Scheduler(VkDevice dev, VkQueue q[4], uint32_t qfIndex);
    ~Scheduler();

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  const void* pushConstants, size_t pcSize,
                  uint32_t gx, uint32_t gy, uint32_t gz,
                  int aceIndex = 0,
                  VkFence fence = VK_NULL_HANDLE);

    // Dispatch with GPU timestamp profiling
    void dispatchTimed(const std::string& name, Profiler* profiler,
                        VkPipeline pipeline, VkPipelineLayout layout,
                        const void* pushConstants, size_t pcSize,
                        uint32_t gx, uint32_t gy, uint32_t gz,
                        int aceIndex = 0);

    void dispatchMulti(const std::vector<std::tuple<VkPipeline, VkPipelineLayout,
                                                       const void*, size_t,
                                                       uint32_t, uint32_t, uint32_t>>& dispatches);

    // Batch multiple dispatches into a single command buffer submission
    struct DispatchDesc {
        VkPipeline pipeline;
        VkPipelineLayout layout;
        const void* pushConstants;
        size_t pcSize;
        uint32_t gx, gy, gz;
    };
    void dispatchBatch(const std::vector<DispatchDesc>& dispatches, int aceIndex = 0);

    void syncAll();

    void speculativeDecode(const std::vector<uint32_t>& draftTokens,
                           VkPipeline verifyPipeline, VkPipelineLayout verifyLayout,
                           const void* verifyPC, size_t verifyPCSize);

private:
    VkCommandBuffer allocateCmd(int aceIndex);
};

} // namespace rdna4
