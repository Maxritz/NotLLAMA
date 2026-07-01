#pragma once
#include "rdna4.hpp"
#include <vulkan/vulkan.h>

namespace rdna4 {

struct GpuMailbox {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceAddress gpuAddress = 0;
    VkDevice device = VK_NULL_HANDLE;

    bool init(VkDevice dev, VkPhysicalDevice pdev);
    void destroy();

    VkDeviceAddress addr() const { return gpuAddress; }

    // Host-side polling
    uint32_t readToken();
    void acknowledge();

    // GPU-side: shader writes token here via buffer_reference
    // Layout: [0] = token_id (uint32), [4] = ready flag (uint32)
};

} // namespace rdna4
