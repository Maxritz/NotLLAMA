#include "engine/vulkan_debug_context.hpp"
#include <cstring>

namespace notllama {

VulkanDebugContext::VulkanDebugContext(VkInstance instance, VkDevice device)
    : instance_(instance), device_(device) {
    if (!instance_ || !device_) return;

    vkSetDebugUtilsObjectNameEXT_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(instance_, "vkSetDebugUtilsObjectNameEXT"));
    vkCmdBeginDebugUtilsLabelEXT_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(instance_, "vkCmdBeginDebugUtilsLabelEXT"));
    vkCmdEndDebugUtilsLabelEXT_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(instance_, "vkCmdEndDebugUtilsLabelEXT"));

    debug_utils_available_ = vkSetDebugUtilsObjectNameEXT_ &&
                              vkCmdBeginDebugUtilsLabelEXT_ &&
                              vkCmdEndDebugUtilsLabelEXT_;
}

bool VulkanDebugContext::EnableValidationLayers() {
    // Validation layers must be enabled at VkInstance creation time.
    // This method reports whether the current instance supports debug utils.
    return debug_utils_available_;
}

void VulkanDebugContext::NameObject(uint64_t handle, VkObjectType type, const char* name) {
    if (!debug_utils_available_ || handle == 0 || !name) return;

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    vkSetDebugUtilsObjectNameEXT_(device_, &info);
}

void VulkanDebugContext::BeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b) {
    if (!debug_utils_available_ || cmd == VK_NULL_HANDLE || !name) return;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT_(cmd, &label);
}

void VulkanDebugContext::EndLabel(VkCommandBuffer cmd) {
    if (!debug_utils_available_ || cmd == VK_NULL_HANDLE) return;
    vkCmdEndDebugUtilsLabelEXT_(cmd);
}

void VulkanDebugContext::CaptureFrame(const std::string& path) {
    (void)path;
    // TODO: RenderDoc integration if available
}

} // namespace notllama
