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

    // GPU properties detected at init (for shader tuning and wave size detection)
    uint32_t subgroupSize = 64;
    uint32_t vendorID = 0;
    uint32_t deviceApiVersion = 0;
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};

    // RDNA generation + wave32 capability (AMD only)
    uint32_t amdGeneration = 0;  // 0=unknown, 1=GCN, 2=RDNA2, 3=RDNA3, 4=RDNA4
    bool forceWave32 = false;    // can force wave32 via pipeline subgroup size control

    // Available AMD-specific extensions (enabled at create device)
    bool hasAMDShaderHalfFloat = false;
    bool hasAMDShaderBallot = false;
    bool hasShaderSubgroupRotate = false;

    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;

    bool init();
    void cleanup();

    bool isAmd() const { return vendorID == 0x1002; }
    bool isNvidia() const { return vendorID == 0x10DE; }
    bool isGCN() const { return amdGeneration == 1; }
    bool isRDNA2() const { return amdGeneration == 2; }
    bool isRDNA3() const { return amdGeneration == 3; }
    bool isRDNA4() const { return amdGeneration == 4; }
    bool isWave32() const { return subgroupSize <= 32 || forceWave32; }
    bool isWave64() const { return !isWave32(); }
};

} // namespace rdna4
