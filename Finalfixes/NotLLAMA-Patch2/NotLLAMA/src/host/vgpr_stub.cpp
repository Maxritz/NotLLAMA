#include "rdna4_vgpr.hpp"
#include <iostream>
#include <vector>
#include <string>



namespace rdna4 {

DynamicVGPRExtension::DynamicVGPRExtension(VkDevice dev, VkPhysicalDevice pdev) : device(dev), physicalDevice(pdev) {
    available = checkSupport();
    if (available) {
        std::cout << "VK_AMD_shader_info available - Dynamic VGPR possible\n";
    } else {
        std::cout << "VK_AMD_shader_info not available - Dynamic VGPR disabled\n";
        std::cout << "  (Requires AMD proprietary driver or custom ISA loader)\n";
    }
}

bool DynamicVGPRExtension::checkSupport() {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());

    for (const auto& ext : exts) {
        if (std::string(ext.extensionName) == VK_AMD_SHADER_INFO_EXTENSION_NAME) {
            return true;
        }
    }
    return false;
}

VkShaderModule DynamicVGPRExtension::loadAMDShader(const uint32_t* isaBinary, size_t wordCount) {
    // RDNA4 ISA loading via standard Vulkan isn't supported.
    // Raw AMD ISA binaries require vendor-specific driver extensions.
    // Use glslc-compiled SPIR-V for compute shaders instead.
    std::cerr << "loadAMDShader: Raw ISA loading not supported via standard Vulkan.\n";
    std::cerr << "  Use glslc to compile GLSL → SPIR-V for compute shaders.\n";
    return VK_NULL_HANDLE;
}

uint32_t DynamicVGPRExtension::queryVGPRUsage(VkPipeline pipeline) {
    if (!available || !pipeline) return 0;

    // Query actual VGPR usage from the AMD shader info extension.
    // This calls vkGetShaderInfoAMD which returns hardware-specific resource usage data.
    using PFN_GetShaderInfoAMD = VkResult(VKAPI_CALL*)(
        VkDevice, VkPipeline, VkShaderStageFlagBits, VkShaderInfoTypeAMD, size_t*, void*);
    auto pfn = reinterpret_cast<PFN_GetShaderInfoAMD>(
        vkGetDeviceProcAddr(device, "vkGetShaderInfoAMD"));
    if (!pfn) {
        std::cerr << "queryVGPRUsage: vkGetDeviceProcAddr failed\n";
        return 0;
    }

    // Query shader statistics (includes VGPR/SGPR usage)
    size_t dataSize = 0;
    VkResult result = pfn(device, pipeline, VK_SHADER_STAGE_COMPUTE_BIT,
                          VK_SHADER_INFO_TYPE_STATISTICS_AMD, &dataSize, nullptr);
    if (result != VK_SUCCESS || dataSize == 0) return 0;

    std::vector<uint8_t> data(dataSize);
    result = pfn(device, pipeline, VK_SHADER_STAGE_COMPUTE_BIT,
                 VK_SHADER_INFO_TYPE_STATISTICS_AMD, &dataSize, data.data());
    if (result != VK_SUCCESS) return 0;

    // VkShaderStatisticsInfoAMD layout:
    //   VkShaderStageFlags shaderStageMask
    //   uint32_t resourceUsageNumBindings
    //   uint32_t resourceUsageNumDescriptors
    //   uint32_t resourceUsageNumInstructions
    //   uint32_t resourceUsageNumVgprs
    //   uint32_t resourceUsageNumSgprs
    //   uint32_t resourceUsageLdsSizePerLocalWorkGroup
    //   ... more fields
    // Offset 4 = stageMask, offset 8 = numBindings, offset 12 = numDescriptors,
    // offset 16 = numInstructions, offset 20 = numVgprs
    if (dataSize >= 24) {
        uint32_t vgprCount = *reinterpret_cast<uint32_t*>(data.data() + 20);
        std::cout << "VGPR usage: " << vgprCount << " VGPRs\n";
        return vgprCount;
    }

    return 0;
}

} // namespace rdna4
