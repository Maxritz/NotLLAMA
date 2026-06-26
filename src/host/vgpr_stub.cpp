#include "rdna4_vgpr.hpp"
#include <iostream>

namespace rdna4 {

DynamicVGPRExtension::DynamicVGPRExtension(VkDevice dev) : device(dev) {
    available = checkSupport();
    if (available) {
        std::cout << "VK_AMD_shader_info available — Dynamic VGPR possible\n";
    } else {
        std::cout << "VK_AMD_shader_info not available — Dynamic VGPR disabled\n";
        std::cout << "  (Requires AMD proprietary driver or custom ISA loader)\n";
    }
}

bool DynamicVGPRExtension::checkSupport() {
    // Check for VK_AMD_shader_info extension
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());

    for (const auto& ext : exts) {
        if (std::string(ext.extensionName) == VK_AMD_SHADER_INFO_EXTENSION_NAME) {
            return true;
        }
    }
    return false;
}

VkShaderModule DynamicVGPRExtension::loadAMDShader(const uint32_t* isaBinary, size_t wordCount) {
    // TODO: Use vkCreateShaderModule with AMD-specific shader binary
    // This requires the driver to accept raw GCN/RDNA4 ISA
    std::cerr << "loadAMDShader: Not implemented — requires custom ISA assembler\n";
    return VK_NULL_HANDLE;
}

uint32_t DynamicVGPRExtension::queryVGPRUsage(VkPipeline pipeline) {
    // TODO: Use vkGetShaderInfoAMD to query resource usage
    std::cerr << "queryVGPRUsage: Not implemented\n";
    return 0;
}

} // namespace rdna4
