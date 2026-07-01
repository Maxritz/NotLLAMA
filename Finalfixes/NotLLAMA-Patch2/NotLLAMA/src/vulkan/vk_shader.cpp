#include <vulkan/vulkan.h>
#include <cstdio>
#include <vector>
#ifdef _MSC_VER
#include <windows.h>
#endif

static VkResult create_pipeline_seh_impl(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t count,
    const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator,
    VkPipeline* pipelines)
{
#ifdef _MSC_VER
    __try {
        return vkCreateComputePipelines(device, cache, count, infos, allocator, pipelines);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        DWORD code = GetExceptionCode();
        fprintf(stderr, "SEH exception 0x%08X in vkCreateComputePipelines\n", code);
        for (uint32_t i = 0; i < count; i++) pipelines[i] = VK_NULL_HANDLE;
        return VK_ERROR_UNKNOWN;
    }
#else
    return vkCreateComputePipelines(device, cache, count, infos, allocator, pipelines);
#endif
}

VkResult create_pipeline_seh(
    VkDevice device,
    VkPipelineCache cache,
    uint32_t count,
    const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator,
    VkPipeline* pipelines)
{
    return create_pipeline_seh_impl(device, cache, count, infos, allocator, pipelines);
}

VkShaderModule create_shader_module(VkDevice device, const uint32_t* code, size_t code_size) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code_size;
    ci.pCode = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(device, &ci, nullptr, &mod);
    if (res != VK_SUCCESS) { fprintf(stderr, "vkCreateShaderModule failed: %d\n", res); return VK_NULL_HANDLE; }
    return mod;
}
