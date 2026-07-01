#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>

namespace notllama {

class IDebugContext {
public:
    virtual ~IDebugContext() = default;

    virtual bool EnableValidationLayers() = 0;
    virtual void NameObject(uint64_t handle, VkObjectType type, const char* name) = 0;
    virtual void BeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b) = 0;
    virtual void EndLabel(VkCommandBuffer cmd) = 0;
    virtual void CaptureFrame(const std::string& path) = 0;
};

} // namespace notllama
