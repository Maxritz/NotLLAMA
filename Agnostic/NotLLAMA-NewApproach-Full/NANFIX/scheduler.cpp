#include "rdna4_scheduler.hpp"
#include "rdna4_profiler.hpp"
#include <iostream>
#include <cstdio>
#include <tuple>
#include <chrono>
#include <thread>

namespace rdna4 {

Scheduler::Scheduler(VkDevice dev, VkQueue q[4], uint32_t qfIndex, FencePool* pool)
    : device(dev), queueFamilyIndex(qfIndex), fencePool(pool) {
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
    cleanup();
}

void Scheduler::cleanup() {
    syncAll();
    for (int i = 0; i < 4; ++i) {
        if (cmdPools[i]) {
            vkDestroyCommandPool(device, cmdPools[i], nullptr);
            cmdPools[i] = VK_NULL_HANDLE;
        }
    }
}

VkCommandBuffer Scheduler::allocateCmd(int aceIndex) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = cmdPools[aceIndex];
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &cmd);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] vkAllocateCommandBuffers failed: %d\n", result);
        return VK_NULL_HANDLE;
    }
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
    VkResult beginResult = vkBeginCommandBuffer(cmd, &beginInfo);
    if (beginResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] vkBeginCommandBuffer failed: %d\n", beginResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstants && pcSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pushConstants);
    }
    if (gx > 65535 || gy > 65535 || gz > 65535) {
        fprintf(stderr, "[Scheduler::dispatch] FATAL: workgroup count %u x %u x %u exceeds Vulkan limit 65535\n", gx, gy, gz);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatch: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    // Auto-acquire fence from pool if none provided
    bool ownsFence = (fence == VK_NULL_HANDLE && fencePool);
    if (ownsFence) {
        fence = fencePool->acquire();
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkResult submitResult = vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] vkQueueSubmit failed: %d\n", submitResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        if (ownsFence) fencePool->release(fence);
        return;
    }

    // Track fences per queue for syncAll
    if (ownsFence && aceIndex >= 0 && aceIndex < 4) {
        queueFences_[aceIndex].push_back(fence);
    }
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
    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchTimed: vkBeginCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    uint32_t qStart = profiler->allocateQueryRange(2);
    profiler->writeTimestamp(cmd, qStart, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstants && pcSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pushConstants);
    }
    if (gx > 65535 || gy > 65535 || gz > 65535) {
        fprintf(stderr, "[Scheduler::dispatch] FATAL: workgroup count %u x %u x %u exceeds Vulkan limit 65535\n", gx, gy, gz);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }
    vkCmdDispatch(cmd, gx, gy, gz);

    profiler->writeTimestamp(cmd, qStart + 1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchTimed: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    // Use fence pool instead of per-submit create/destroy
    VkFence fence = fencePool ? fencePool->acquire() : VK_NULL_HANDLE;
    if (fence == VK_NULL_HANDLE) {
        // Fallback: create fence inline if pool unavailable
        VkFenceCreateInfo fi = {};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fi, nullptr, &fence);
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkResult submitResult = vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchTimed: vkQueueSubmit failed: %d\n", submitResult);
        if (fencePool) fencePool->release(fence); else vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    VkResult waitResult = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (waitResult == VK_TIMEOUT) {
        fprintf(stderr, "[Scheduler] dispatchTimed: vkWaitForFences timed out\n");
    } else if (waitResult == VK_ERROR_DEVICE_LOST) {
        fprintf(stderr, "[Scheduler] dispatchTimed: VK_ERROR_DEVICE_LOST\n");
        if (fencePool) fencePool->release(fence); else vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    } else if (waitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchTimed: vkWaitForFences failed: %d\n", waitResult);
    }

    profiler->recordGpuTime(name, qStart, qStart + 1, aceIndex);

    if (fencePool) {
        fencePool->release(fence);
    } else {
        vkDestroyFence(device, fence, nullptr);
    }
    vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
}

void Scheduler::dispatchMulti(const std::vector<std::tuple<VkPipeline, VkPipelineLayout,
                                                              const void*, size_t,
                                                              uint32_t, uint32_t, uint32_t>>& dispatches) {
    if (dispatches.size() > 4) {
        fprintf(stderr, "[Scheduler] dispatchMulti: %zu dispatches exceeds ACE count (4), dropping %zu\n",
                dispatches.size(), dispatches.size() - 4);
    }
    for (size_t i = 0; i < dispatches.size() && i < 4; ++i) {
        const auto& d = dispatches[i];
        dispatch(std::get<0>(d), std::get<1>(d),
                 std::get<2>(d), std::get<3>(d),
                 std::get<4>(d), std::get<5>(d), std::get<6>(d),
                 static_cast<int>(i));
    }
}

void Scheduler::dispatchBatch(const std::vector<DispatchDesc>& dispatches, int aceIndex, VkFence fence) {
    if (dispatches.empty()) return;

    VkCommandBuffer cmd = allocateCmd(aceIndex);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginResult = vkBeginCommandBuffer(cmd, &beginInfo);
    if (beginResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatch: vkBeginCommandBuffer failed: %d\n", beginResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    for (const auto& d : dispatches) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
        if (d.pcSize > 0) {
            vkCmdPushConstants(cmd, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(d.pcSize), d.pushConstantData);
        }
        vkCmdDispatch(cmd, d.gx, d.gy, d.gz);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatch: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    // Auto-acquire fence from pool if none provided
    bool ownsFence = (fence == VK_NULL_HANDLE && fencePool);
    if (ownsFence) {
        fence = fencePool->acquire();
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkResult submitResult = vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatch: vkQueueSubmit failed: %d\n", submitResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        if (ownsFence) fencePool->release(fence);
        return;
    }

    // Track fences per queue for syncAll
    if (ownsFence && aceIndex >= 0 && aceIndex < 4) {
        queueFences_[aceIndex].push_back(fence);
    }
}

void Scheduler::syncAll() {
    if (fencePool) {
        // Fence-based sync: wait on all in-flight fences, release back to pool.
        auto t0 = std::chrono::steady_clock::now();
        // Collect all in-flight fences from all queues
        std::vector<VkFence> allFences;
        for (int i = 0; i < 4; ++i) {
            allFences.insert(allFences.end(), queueFences_[i].begin(), queueFences_[i].end());
        }
        if (!allFences.empty()) {
            vkWaitForFences(device, static_cast<uint32_t>(allFences.size()), allFences.data(), VK_TRUE, UINT64_MAX);
        }
        auto t1 = std::chrono::steady_clock::now();
        // Release all fences back to pool and clear
        for (int i = 0; i < 4; ++i) {
            for (VkFence f : queueFences_[i]) {
                fencePool->release(f);
            }
            queueFences_[i].clear();
        }
        // Reset command pools
        for (int i = 0; i < 4; ++i) {
            vkResetCommandPool(device, cmdPools[i], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
        auto t2 = std::chrono::steady_clock::now();
        auto waitMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto resetMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (waitMs > 100.0) {
            fprintf(stderr, "[sync] fence-wait=%.1fms reset=%.1fms pool=%u\n",
                    waitMs, resetMs, fencePool->available());
        }
    } else {
        // Fallback: original vkQueueWaitIdle path (no fence pool)
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; ++i) {
            vkQueueWaitIdle(queues[i]);
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; ++i) {
            vkResetCommandPool(device, cmdPools[i], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
        auto t2 = std::chrono::steady_clock::now();
        auto waitMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto resetMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (waitMs > 100.0) {
            fprintf(stderr, "[sync] queue-idle=%.1fms reset=%.1fms\n", waitMs, resetMs);
        }
    }
}

void Scheduler::syncAllThrottled(double targetUtilization) {
    if (fencePool) {
        auto t0 = std::chrono::steady_clock::now();
        // Collect all in-flight fences from all queues
        std::vector<VkFence> allFences;
        for (int i = 0; i < 4; ++i) {
            allFences.insert(allFences.end(), queueFences_[i].begin(), queueFences_[i].end());
        }
        if (!allFences.empty()) {
            vkWaitForFences(device, static_cast<uint32_t>(allFences.size()), allFences.data(), VK_TRUE, UINT64_MAX);
        }
        auto t1 = std::chrono::steady_clock::now();
        // Release all fences back to pool and clear
        for (int i = 0; i < 4; ++i) {
            for (VkFence f : queueFences_[i]) {
                fencePool->release(f);
            }
            queueFences_[i].clear();
        }
        // Reset command pools
        for (int i = 0; i < 4; ++i) {
            vkResetCommandPool(device, cmdPools[i], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
        // Throttle sleep
        double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (gpuMs > 0.5 && targetUtilization > 0.0 && targetUtilization < 1.0) {
            double sleepMs = gpuMs * (1.0 / targetUtilization - 1.0);
            if (sleepMs > 0.5) {
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(sleepMs * 1000)));
            }
        }
    } else {
        // Fallback: original vkQueueWaitIdle path
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; ++i) {
            vkQueueWaitIdle(queues[i]);
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; ++i) {
            vkResetCommandPool(device, cmdPools[i], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
        double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (gpuMs > 0.5 && targetUtilization > 0.0 && targetUtilization < 1.0) {
            double sleepMs = gpuMs * (1.0 / targetUtilization - 1.0);
            if (sleepMs > 0.5) {
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>(sleepMs * 1000)));
            }
        }
    }
}

void Scheduler::dispatchBatchBarriers(const std::vector<DispatchDesc>& dispatches,
                                        const std::vector<uint32_t>& groupEnds,
                                        const std::vector<PipelineBarrier>& barriers,
                                        int aceIndex, VkFence fence) {
    if (dispatches.empty()) return;

    VkCommandBuffer cmd = allocateCmd(aceIndex);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginResult = vkBeginCommandBuffer(cmd, &beginInfo);
    if (beginResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatchBarriers: vkBeginCommandBuffer failed: %d\n", beginResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    uint32_t barrierIdx = 0;
    for (uint32_t di = 0; di < static_cast<uint32_t>(dispatches.size()); ++di) {
        const auto& d = dispatches[di];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
        if (d.pcSize > 0) {
            vkCmdPushConstants(cmd, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(d.pcSize), d.pushConstantData);
        }
        vkCmdDispatch(cmd, d.gx, d.gy, d.gz);

        // Insert barrier after this dispatch if it's at a group boundary
        if (barrierIdx < static_cast<uint32_t>(barriers.size()) && di == groupEnds[barrierIdx]) {
            const auto& b = barriers[barrierIdx];
            const VkMemoryBarrier* mbPtr = (b.memoryBarrier.sType != 0) ? &b.memoryBarrier : nullptr;
            vkCmdPipelineBarrier(cmd, b.srcStageMask, b.dstStageMask,
                                   b.dependencyFlags,
                                   mbPtr ? 1 : 0, mbPtr,
                                   b.bufferMemoryBarrierCount, b.bufferMemoryBarrierCount > 0 ? b.bufferMemoryBarriers : nullptr,
                                   b.imageMemoryBarrierCount, b.imageMemoryBarrierCount > 0 ? b.imageMemoryBarriers : nullptr);
            barrierIdx++;
        }
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatchBarriers: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        return;
    }

    bool ownsFence = (fence == VK_NULL_HANDLE && fencePool);
    if (ownsFence) {
        fence = fencePool->acquire();
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkResult submitResult = vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] dispatchBatchBarriers: vkQueueSubmit failed: %d\n", submitResult);
        vkFreeCommandBuffers(device, cmdPools[aceIndex], 1, &cmd);
        if (ownsFence) fencePool->release(fence);
        return;
    }

    if (ownsFence && aceIndex >= 0 && aceIndex < 4) {
        queueFences_[aceIndex].push_back(fence);
    }
}

void Scheduler::beginBatch(int aceIndex) {
    batchAceIndex = aceIndex;
    batchCmdBuffer = allocateCmd(aceIndex);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginResult = vkBeginCommandBuffer(batchCmdBuffer, &beginInfo);
    if (beginResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] beginBatch: vkBeginCommandBuffer failed: %d\n", beginResult);
        batchCmdBuffer = VK_NULL_HANDLE;
    }
}

void Scheduler::dispatchInBatch(VkPipeline pipeline, VkPipelineLayout layout,
                                 const void* pushConstants, size_t pcSize,
                                 uint32_t gx, uint32_t gy, uint32_t gz) {
    if (batchCmdBuffer == VK_NULL_HANDLE) return;

    // Vulkan minimum guaranteed limit per dimension is 65535
    if (gx > 65535 || gy > 65535 || gz > 65535) {
        fprintf(stderr, "[Scheduler::dispatchInBatch] FATAL: workgroup count %u x %u x %u exceeds Vulkan limit 65535\n", gx, gy, gz);
        return;
    }

    vkCmdBindPipeline(batchCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (pushConstants && pcSize > 0) {
        vkCmdPushConstants(batchCmdBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(pcSize), pushConstants);
    }
    vkCmdDispatch(batchCmdBuffer, gx, gy, gz);
}

void Scheduler::barrierBetweenGroups(VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                      VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    if (batchCmdBuffer == VK_NULL_HANDLE) return;

    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = srcAccess;
    mb.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(batchCmdBuffer, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Scheduler::endBatch(VkFence fence) {
    if (batchCmdBuffer == VK_NULL_HANDLE) return;

    if (vkEndCommandBuffer(batchCmdBuffer) != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] endBatch: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(device, cmdPools[batchAceIndex], 1, &batchCmdBuffer);
        batchCmdBuffer = VK_NULL_HANDLE;
        return;
    }

    bool ownsFence = (fence == VK_NULL_HANDLE && fencePool);
    if (ownsFence) {
        fence = fencePool->acquire();
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &batchCmdBuffer;

    VkResult submitResult = vkQueueSubmit(queues[batchAceIndex], 1, &submitInfo, fence);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[Scheduler] endBatch: vkQueueSubmit failed: %d\n", submitResult);
        vkFreeCommandBuffers(device, cmdPools[batchAceIndex], 1, &batchCmdBuffer);
        if (ownsFence) fencePool->release(fence);
    } else if (ownsFence) {
        queueFences_[batchAceIndex].push_back(fence);
    }

    batchCmdBuffer = VK_NULL_HANDLE;
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
