#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include "rdna4_fence_pool.hpp"

namespace rdna4 {

class Profiler;

class Scheduler {
public:
    VkDevice device;
    VkQueue queues[4];
    VkCommandPool cmdPools[4];
    uint32_t queueFamilyIndex;

    // Fence pool replaces per-submit vkCreateFence/vkDestroyFence.
    // Fences are acquired on submit, waited on during sync, released back to pool.
    FencePool* fencePool = nullptr;

    Scheduler(VkDevice dev, VkQueue q[4], uint32_t qfIndex, FencePool* pool = nullptr);
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

    // Batch multiple dispatches into a single command buffer submission.
    // Push constants are stored INLINE (not as pointers) to prevent
    // invalidation from vector reallocation.
    struct DispatchDesc {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        uint8_t pushConstantData[128];
        size_t pcSize = 0;
        uint32_t gx = 0, gy = 0, gz = 0;

        DispatchDesc() { memset(pushConstantData, 0, 128); }

        template<typename T>
        DispatchDesc(VkPipeline p, VkPipelineLayout l, const T& pc,
                     uint32_t x, uint32_t y, uint32_t z)
            : pipeline(p), layout(l), pcSize(sizeof(T)), gx(x), gy(y), gz(z) {
            static_assert(sizeof(T) <= 128, "Push constants exceed 128-byte Vulkan limit");
            memcpy(pushConstantData, &pc, sizeof(T));
        }
    };
    void dispatchBatch(const std::vector<DispatchDesc>& dispatches, int aceIndex = 0, VkFence fence = VK_NULL_HANDLE);

    // Pipeline barrier to insert between dispatch groups in a batch.
    // All members are stored INLINE to prevent dangling pointers from
    // stack-local variables when the struct is copied into a vector.
    struct PipelineBarrier {
        VkPipelineStageFlags srcStageMask = 0;
        VkPipelineStageFlags dstStageMask = 0;
        VkDependencyFlags dependencyFlags = 0;
        VkMemoryBarrier memoryBarrier = {};
        uint32_t bufferMemoryBarrierCount = 0;
        VkBufferMemoryBarrier bufferMemoryBarriers[4] = {};
        uint32_t imageMemoryBarrierCount = 0;
        VkImageMemoryBarrier imageMemoryBarriers[4] = {};
    };
// Batch dispatches with pipeline barriers between groups.
    // barriers[i] is inserted after dispatches[groupEnds[i]].
    void dispatchBatchBarriers(const std::vector<DispatchDesc>& dispatches,
                                 const std::vector<uint32_t>& groupEnds,
                                 const std::vector<PipelineBarrier>& barriers,
                                 int aceIndex = 0, VkFence fence = VK_NULL_HANDLE);

    // Batch mode: begin → multiple dispatches → barrier → end (single vkQueueSubmit)
    void beginBatch(int aceIndex = 0);
    void dispatchInBatch(VkPipeline pipeline, VkPipelineLayout layout,
                         const void* pushConstants, size_t pcSize,
                         uint32_t gx, uint32_t gy, uint32_t gz);
    void barrierBetweenGroups(VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess);
    void endBatch(VkFence fence = VK_NULL_HANDLE);

    // Fence-based sync: waits on the latest fence per queue, then resets pools.
    // Replaces vkQueueWaitIdle with fence-based waits (~2s → ~0.1ms on AMD RDNA4).
    void syncAll();

    // syncAll + throttle: sleeps after sync to cap GPU utilization at ~80%.
    // targetUtilization: 0.0-1.0, default 0.8
    void syncAllThrottled(double targetUtilization = 0.8);

    // Fence-based per-layer sync: wait on the layerFence, reset it, reset the
    // command pool for the given ACE.  Must call createLayerFence() first.
    void createLayerFence();
    void syncLayer(int aceIndex = 0);

    // Must be called before VkDevice is destroyed. Waits for idle, destroys
    // command pools, and nulls handles so the destructor is safe to run after.
    void cleanup();

    void speculativeDecode(const std::vector<uint32_t>& draftTokens,
                           VkPipeline verifyPipeline, VkPipelineLayout verifyLayout,
                           const void* verifyPC, size_t verifyPCSize);

    VkFence layerFence = VK_NULL_HANDLE;

    // Per-queue fence tracking for syncAll/syncAllThrottled.
    // Each submit pushes its fence to the per-queue vector.
    // syncAll waits on all, releases them all, then clears.
    std::array<std::vector<VkFence>, 4> queueFences_;

    // Batch mode state
    int batchAceIndex = 0;
    VkCommandBuffer batchCmdBuffer = VK_NULL_HANDLE;

private:
    VkCommandBuffer allocateCmd(int aceIndex);
};

} // namespace rdna4
