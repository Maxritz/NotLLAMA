#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace rdna4 {

// Loads SPIR-V from disk and creates VkPipeline + VkPipelineLayout
class PipelineBuilder {
public:
    VkDevice device;
    std::unordered_map<std::string, VkPipeline> pipelines;
    std::unordered_map<std::string, VkPipelineLayout> layouts;
    std::unordered_map<std::string, VkShaderModule> modules;

    PipelineBuilder(VkDevice dev) : device(dev) {}

    bool loadShader(const std::string& name, const std::string& spvPath);
    bool createComputePipeline(const std::string& name, size_t pushConstantSize);

    VkPipeline getPipeline(const std::string& name) const;
    VkPipelineLayout getLayout(const std::string& name) const;

    void cleanup();
};

} // namespace rdna4
