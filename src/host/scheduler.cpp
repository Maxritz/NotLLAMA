#include "rdna4_scheduler.hpp"
#include "rdna4_profiler.hpp"
#include <iostream>
#include <tuple>

namespace rdna4 {

Scheduler::Scheduler(VkDevice dev, VkQueue q[4], uint32_t qfIndex)
    : device(dev), queueFamilyIndex(qfIndex) {
    for (int i = 0; i < 4; ++i) {
        queues[i] = q[i];
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = qfIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPools[i]);
    }
}

Scheduler::~Scheduler() {
    for (int i = 0; i < 4; ++i) {
        if (cmdPools[i]) vkDestroyCommandPool(device, cmdPools[i], nullptr);
    }
}

VkCommandBuffer Scheduler::allocateCmd(int aceIndex) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPools[aceIndex];
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);
    return cmd;
}

void Scheduler::dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                            const void* pushConstants, size_t pcSize,
                            uint32_t gx, uint32_t gy, uint32_t gz,
                            int aceIndex, VkFence fence) {
    VkCommandBuffer cmd = allocateCmd(aceIndex);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstants && pcSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pushConstants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
}

void Scheduler::dispatchTimed(const std::string& name, Profiler* profiler,
                                 VkPipeline pipeline, VkPipelineLayout layout,
                                 const void* pushConstants, size_t pcSize,
                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                 int aceIndex) {
    VkCommandBuffer cmd = allocateCmd(aceIndex);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    uint32_t qStart = profiler->allocateQueryRange(2);
    profiler->writeTimestamp(cmd, qStart, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstants && pcSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pushConstants);
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    profiler->writeTimestamp(cmd, qStart + 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);

    vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    profiler->recordGpuTime(name, qStart, qStart + 1, aceIndex);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
}

void Scheduler::dispatchMulti(const std::vector<std::tuple<VkPipeline, VkPipelineLayout,
                                                              const void*, size_t,
                                                              uint32_t, uint32_t, uint32_t>>& dispatches) {
    for (size_t i = 0; i < dispatches.size() && i < 4; ++i) {
        const auto& d = dispatches[i];
        dispatch(std::get<0>(d), std::get<1>(d),
                 std::get<2>(d), std::get<3>(d),
                 std::get<4>(d), std::get<5>(d), std::get<6>(d),
                 static_cast<int>(i));
    }
}

void Scheduler::syncAll() {
    for (int i = 0; i < 4; ++i) {
        vkQueueWaitIdle(queues[i]);
    }
}

void Scheduler::speculativeDecode(const std::vector<uint32_t>& draftTokens,
                                   VkPipeline verifyPipeline, VkPipelineLayout verifyLayout,
                                   const void* verifyPC, size_t verifyPCSize) {
    int nDraft = static_cast<int>(draftTokens.size());
    for (int i = 0; i < nDraft && i < 3; ++i) {
        dispatch(verifyPipeline, verifyLayout, verifyPC, verifyPCSize,
                 1, 1, 1, i + 1);
    }
}

} // namespace rdna4
