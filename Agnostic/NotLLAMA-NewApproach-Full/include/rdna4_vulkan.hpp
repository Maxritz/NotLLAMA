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

    // GPU properties detected at init (for shader tuning)
    uint32_t subgroupSize = 64;       // Actual subgroup/wave size (32 or 64)
    uint32_t vendorID = 0;            // GPU vendor (0x1002 = AMD, 0x10DE = NV)
    uint32_t deviceApiVersion = 0;    // Vulkan version the device supports
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};

    bool init();
    void cleanup();

    // Convenience
    bool isAmd() const { return vendorID == 0x1002; }
    bool isNvidia() const { return vendorID == 0x10DE; }
    bool isWave32() const { return subgroupSize <= 32; }
    bool isWave64() const { return subgroupSize >= 64; }
};

} // namespace rdna4
