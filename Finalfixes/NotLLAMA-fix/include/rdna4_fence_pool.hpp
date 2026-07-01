#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <cstdio>

namespace rdna4 {

// Fence pool: pre-allocates reusable VkFences to avoid per-submit create/destroy.
//
// Lifecycle per fence:
//   acquire() → [unsignaled, ready] → vkQueueSubmit(fence) → GPU signals it
//   → wait for completion → release() → back in pool
//
// Thread safety: none (single-threaded host, as is the rest of the engine).
class FencePool {
public:
    VkDevice device;
    static constexpr uint32_t POOL_SIZE = 64;

    explicit FencePool(VkDevice dev) : device(dev) {
        pool_.reserve(POOL_SIZE);
        VkFenceCreateInfo fi = {};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (uint32_t i = 0; i < POOL_SIZE; ++i) {
            VkFence f = VK_NULL_HANDLE;
            VkResult r = vkCreateFence(device, &fi, nullptr, &f);
            if (r == VK_SUCCESS && f != VK_NULL_HANDLE) {
                pool_.push_back(f);
            }
        }
    }

    ~FencePool() {
        for (VkFence f : pool_) {
            if (f != VK_NULL_HANDLE) vkDestroyFence(device, f, nullptr);
        }
        for (VkFence f : inFlight_) {
            if (f != VK_NULL_HANDLE) vkDestroyFence(device, f, nullptr);
        }
    }

    // Non-copyable, non-movable
    FencePool(const FencePool&) = delete;
    FencePool& operator=(const FencePool&) = delete;

    // Acquire a fence from the pool (or create a new one if empty).
    // The returned fence is in UNSIGNALED state, ready for vkQueueSubmit.
    VkFence acquire() {
        if (!pool_.empty()) {
            VkFence f = pool_.back();
            pool_.pop_back();
            // Reset to unsignaled (safe: caller waited before releasing)
            vkResetFences(device, 1, &f);
            return f;
        }
        // Pool exhausted — create on demand
        VkFenceCreateInfo fi = {};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence f = VK_NULL_HANDLE;
        VkResult r = vkCreateFence(device, &fi, nullptr, &f);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "[FencePool] vkCreateFence failed: %d\n", r);
            return VK_NULL_HANDLE;
        }
        return f;
    }

    // Release a fence back to the pool after its work has completed.
    // The fence MUST be in signaled state (caller waited on it first).
    void release(VkFence f) {
        if (f == VK_NULL_HANDLE) return;
        pool_.push_back(f);
    }

    // Wait for a single fence, then release it back to the pool.
    void waitAndRelease(VkFence f) {
        if (f == VK_NULL_HANDLE) return;
        vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX);
        release(f);
    }

    // Wait for multiple fences, then release them all.
    void waitAndReleaseAll(const VkFence* fences, uint32_t count) {
        if (!fences || count == 0) return;
        vkWaitForFences(device, count, fences, VK_TRUE, UINT64_MAX);
        for (uint32_t i = 0; i < count; ++i) {
            release(fences[i]);
        }
    }

    // How many fences are currently available in the pool.
    uint32_t available() const { return static_cast<uint32_t>(pool_.size()); }

private:
    std::vector<VkFence> pool_;    // available fences (unsignaled)
    std::vector<VkFence> inFlight_; // reserved for future use (currently unused)
};

} // namespace rdna4
