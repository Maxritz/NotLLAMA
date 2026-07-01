#pragma once
#include <vulkan/vulkan.h>

VkResult create_pipeline_seh(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t count,
    const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator,
    VkPipeline* pipelines);

VkShaderModule create_shader_module(VkDevice device, const uint32_t* code, size_t code_size);
