#include "rdna4_pipeline.hpp"
#include <fstream>
#include <iostream>
#include <vector>

namespace rdna4 {

bool PipelineBuilder::loadShader(const std::string& name, const std::string& spvPath) {
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open SPIR-V: " << spvPath << "\n";
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = code.data();

    VkShaderModule module;
    VkResult r = vkCreateShaderModule(device, &createInfo, nullptr, &module);
    if (r != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << name << "\n";
        return false;
    }

    modules[name] = module;
    return true;
}

bool PipelineBuilder::createComputePipeline(const std::string& name, size_t pushConstantSize) {
    auto it = modules.find(name);
    if (it == modules.end()) {
        std::cerr << "Shader module not loaded: " << name << "\n";
        return false;
    }

    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout layout;
    VkResult r = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout);
    if (r != VK_SUCCESS) return false;
    layouts[name] = layout;

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = it->second;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (r != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, layout, nullptr);
        return false;
    }

    pipelines[name] = pipeline;
    std::cout << "Pipeline created: " << name << " (PC size=" << pushConstantSize << ")\n";
    return true;
}

VkPipeline PipelineBuilder::getPipeline(const std::string& name) const {
    auto it = pipelines.find(name);
    return it != pipelines.end() ? it->second : VK_NULL_HANDLE;
}

VkPipelineLayout PipelineBuilder::getLayout(const std::string& name) const {
    auto it = layouts.find(name);
    return it != layouts.end() ? it->second : VK_NULL_HANDLE;
}

void PipelineBuilder::cleanup() {
    for (auto& [name, pipe] : pipelines) vkDestroyPipeline(device, pipe, nullptr);
    for (auto& [name, layout] : layouts) vkDestroyPipelineLayout(device, layout, nullptr);
    for (auto& [name, mod] : modules) vkDestroyShaderModule(device, mod, nullptr);
    pipelines.clear();
    layouts.clear();
    modules.clear();
}

} // namespace rdna4
