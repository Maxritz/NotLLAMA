#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <string>

namespace rdna4 {

class VulkanContext {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queues[4] = {};
    uint32_t queueFamilyIndex = 0xFFFFFFFF;

    bool init();
    void cleanup();
};

} // namespace rdna4
