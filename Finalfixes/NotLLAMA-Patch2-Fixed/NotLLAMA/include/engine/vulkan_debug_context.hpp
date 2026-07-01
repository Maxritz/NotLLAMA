#pragma once
#include "engine/idebug_context.hpp"
#include <vulkan/vulkan.h>

namespace notllama {

class VulkanDebugContext : public IDebugContext {
public:
    explicit VulkanDebugContext(VkInstance instance, VkDevice device);
    ~VulkanDebugContext() override = default;

    bool EnableValidationLayers() override;
    void NameObject(uint64_t handle, VkObjectType type, const char* name) override;
    void BeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b) override;
    void EndLabel(VkCommandBuffer cmd) override;
    void CaptureFrame(const std::string& path) override;

private:
    VkInstance instance_;
    VkDevice device_;

    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT_ = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT_ = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT_ = nullptr;

    bool debug_utils_available_ = false;
};

} // namespace notllama
