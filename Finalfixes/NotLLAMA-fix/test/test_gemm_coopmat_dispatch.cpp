#include "rdna4_vulkan.hpp"
#include "engine/shader_compiler.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <vector>
#include <string>

static std::string FindShaderDir(const char* file) {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();
    for (const auto& p : {fs::path("shaders"), cwd / "shaders", cwd / ".." / ".." / "shaders"}) {
        if (fs::exists(p / file)) return fs::absolute(p).string();
    }
    return "shaders";
}

int main() {
    fprintf(stderr, "Testing gemm_coopmat.comp (wave32 cooperative GEMM)\n");

    rdna4::VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext\n"); return 1; }

    notllama::ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc\n"); return 0; }

    std::string dir = FindShaderDir("gemm_coopmat.comp");
    notllama::ShaderCompileOptions opts;
    opts.src_path = dir + "/gemm_coopmat.comp";
    opts.cache_dir = dir + "/cache";
    opts.target_env = "vulkan1.2";
    opts.defines.push_back("SUBGROUP_SIZE=32");

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: compile: %s\n", log.c_str()); return 1;
    }
    fprintf(stderr, "Compiled (%zu words)\n", spv.size());

    // Shader module
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv.size() * sizeof(uint32_t);
    sm_ci.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &sm_ci, nullptr, &module) != VK_SUCCESS) { fprintf(stderr, "FAIL: CreateShaderModule\n"); return 1; }

    // Pipeline layout (push constants only, no descriptor sets)
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 0;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 48;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc_range;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(ctx.device, &pl_ci, nullptr, &pl) != VK_SUCCESS) { fprintf(stderr, "FAIL: CreatePipelineLayout\n"); return 1; }

    // Pipeline with wave32 control
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo wave32_ci{};
    wave32_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    wave32_ci.requiredSubgroupSize = ctx.subgroupSize;
    VkPipelineShaderStageCreateInfo stage_ci{};
    stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = module;
    stage_ci.pName = "main";
    stage_ci.pNext = ctx.isWave32() ? &wave32_ci : nullptr;
    VkComputePipelineCreateInfo pipe_ci{};
    pipe_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage = stage_ci;
    pipe_ci.layout = pl;
    pipe_ci.basePipelineHandle = VK_NULL_HANDLE;
    pipe_ci.basePipelineIndex = -1;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &pipeline) != VK_SUCCESS) { fprintf(stderr, "FAIL: CreatePipeline\n"); return 1; }
    fprintf(stderr, "Pipeline OK\n");

    // Allocate buffers
    const uint32_t M = 32, N = 32, K = 32;
    auto createBuf = [&](VkBuffer& buf, VkDeviceMemory& mem, size_t sz) -> bool {
        VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx.device, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(ctx.device, buf, &mr);
        VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &pm);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(ctx.device, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        return vkBindBufferMemory(ctx.device, buf, mem, 0) == VK_SUCCESS;
    };
    VkBuffer bufA, bufB, bufC;
    VkDeviceMemory memA, memB, memC;
    size_t sA = M*K*4, sB = K*N*4, sC = M*N*4;
    if (!createBuf(bufA, memA, sA) || !createBuf(bufB, memB, sB) || !createBuf(bufC, memC, sC)) { fprintf(stderr, "FAIL: Buffers\n"); return 1; }

    VkDeviceAddress addrA, addrB, addrC;
    { VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = bufA; addrA = vkGetBufferDeviceAddress(ctx.device, &bdai); }
    { VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = bufB; addrB = vkGetBufferDeviceAddress(ctx.device, &bdai); }
    { VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = bufC; addrC = vkGetBufferDeviceAddress(ctx.device, &bdai); }

    float* a, * b, * c;
    vkMapMemory(ctx.device, memA, 0, sA, 0, (void**)&a);
    vkMapMemory(ctx.device, memB, 0, sB, 0, (void**)&b);
    vkMapMemory(ctx.device, memC, 0, sC, 0, (void**)&c);
    srand(42);
    for (int i = 0; i < M*K; i++) a[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K*N; i++) b[i] = (float)(rand() % 100) / 10.0f;
    memset(c, 0, sC);
    vkUnmapMemory(ctx.device, memA); vkUnmapMemory(ctx.device, memB); vkUnmapMemory(ctx.device, memC);

    // Dispatch
    VkCommandPoolCreateInfo pool_ci{}; pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; pool_ci.queueFamilyIndex = ctx.queueFamilyIndex;
    VkCommandPool pool; vkCreateCommandPool(ctx.device, &pool_ci, nullptr, &pool);
    VkCommandBufferAllocateInfo cmd_ai{}; cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cmd_ai.commandPool = pool; cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmd_ai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(ctx.device, &cmd_ai, &cmd);

    struct PC { uint64_t aA, aB, aC; uint32_t M, N, K; float alpha; } push = {addrA, addrB, addrC, M, N, K, 1.0f};
    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, N, M, 1);
        vkEndCommandBuffer(cmd);
    }
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);
    vkQueueSubmit(ctx.queues[0], 1, &si, fence);
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    // Verify
    vkMapMemory(ctx.device, memC, 0, sC, 0, (void**)&c);
    vkMapMemory(ctx.device, memA, 0, sA, 0, (void**)&a);
    vkMapMemory(ctx.device, memB, 0, sB, 0, (void**)&b);

    size_t errors = 0;
    double max_diff = 0.0;
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            float exp = 0;
            for (uint32_t k = 0; k < K; k++) exp += a[i*K+k] * b[k*N+j];
            double d = std::abs((double)(exp - c[i*N+j]));
            if (d > max_diff) max_diff = d;
            if (d > 0.01f && errors++ < 5) fprintf(stderr, "  MISMATCH[%u,%u]: got %f exp %f\n", i, j, c[i*N+j], exp);
        }

    vkUnmapMemory(ctx.device, memA); vkUnmapMemory(ctx.device, memB); vkUnmapMemory(ctx.device, memC);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pl, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, bufA, nullptr); vkFreeMemory(ctx.device, memA, nullptr);
    vkDestroyBuffer(ctx.device, bufB, nullptr); vkFreeMemory(ctx.device, memB, nullptr);
    vkDestroyBuffer(ctx.device, bufC, nullptr); vkFreeMemory(ctx.device, memC, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    ctx.cleanup();

    if (errors) { fprintf(stderr, "FAIL: %zu mismatches, max diff %f\n", errors, max_diff); return 1; }
    fprintf(stderr, "PASS: %ux%u (%zu mismatches, max diff=%f)\n", M, N, errors, max_diff);
    return 0;
}
