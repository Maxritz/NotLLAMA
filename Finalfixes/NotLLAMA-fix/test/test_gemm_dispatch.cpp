#include "engine/engine.hpp"
#include "rdna4_vulkan.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <vector>
#include <string>

using namespace notllama;

static size_t errors = 0;
static double max_diff = 0.0;

static void check(const char* label, float expected, float actual, size_t i, size_t j) {
    double d = std::abs((double)(expected - actual));
    if (d > max_diff) max_diff = d;
    if (d > 0.01f) {
        if (errors < 5)
            fprintf(stderr, "  MISMATCH %s[%zu,%zu]: got %f expected %f\n", label, i, j, actual, expected);
        errors++;
    }
}

int main() {
    fprintf(stderr, "Testing cooperative_gemm.comp (wave32 DPP GEMM)\n");

    // Find shader directory relative to exe or cwd
    std::string shader_dir = "shaders";
    {
        namespace fs = std::filesystem;
        fs::path cwd = fs::current_path();
        for (const auto& p : {fs::path("shaders"), cwd / "shaders", cwd / ".." / ".." / "shaders"}) {
            if (fs::exists(p / "cooperative_gemm.comp")) { shader_dir = fs::absolute(p).string(); break; }
        }
    }

    // 1. Vulkan context
    rdna4::VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext\n"); return 1; }
    fprintf(stderr, "Vulkan OK (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    // 2. Compile cooperative_gemm.comp
    ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc\n"); return 0; }

    ShaderCompileOptions opts;
    opts.src_path = shader_dir + "/cooperative_gemm.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.target_env = "vulkan1.2";
    opts.defines.push_back("SUBGROUP_SIZE=32");

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: compile: %s\n", log.c_str()); return 1;
    }
    fprintf(stderr, "Compiled (%zu words)\n", spv.size());

    // 3. Pipeline: shader module → pipeline layout → pipeline
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv.size() * sizeof(uint32_t);
    sm_ci.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &sm_ci, nullptr, &module) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateShaderModule\n"); return 1;
    }

    // No descriptor sets — uses buffer references. Push constants only.
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 0;
    pl_ci.pSetLayouts = nullptr;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 48;  // 2 uint64_t + 4 uint32_t + 1 float = 36, rounded to 48
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc_range;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(ctx.device, &pl_ci, nullptr, &pl) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreatePipelineLayout\n"); return 1;
    }

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
    VkResult r = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &pipeline);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateComputePipeline (r=%d)\n", r); return 1;
    }
    fprintf(stderr, "Pipeline OK\n");

    // 4. Allocate buffers A(MxK), B(KxN), C(MxN) — use M=32,N=32,K=32
    const uint32_t M = 32, N = 32, K = 32;
    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    auto createBufBDA = [&](VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr, size_t sz) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx.device, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(ctx.device, buf, &mr);
        VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &pm);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            { mt = i; break; }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(ctx.device, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(ctx.device, buf, mem, 0) != VK_SUCCESS) return false;
        VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = buf;
        addr = vkGetBufferDeviceAddress(ctx.device, &bdai);
        return true;
    };

    VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE, bufC = VK_NULL_HANDLE;
    VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE, memC = VK_NULL_HANDLE;
    VkDeviceAddress addrA = 0, addrB = 0, addrC = 0;

    if (!createBufBDA(bufA, memA, addrA, sizeA) || !createBufBDA(bufB, memB, addrB, sizeB) || !createBufBDA(bufC, memC, addrC, sizeC)) {
        fprintf(stderr, "FAIL: CreateBuffers\n"); return 1;
    }

    // 5. Fill A and B with known values, zero C
    float* a_ptr, * b_ptr, * c_ptr;
    vkMapMemory(ctx.device, memA, 0, sizeA, 0, (void**)&a_ptr);
    vkMapMemory(ctx.device, memB, 0, sizeB, 0, (void**)&b_ptr);
    vkMapMemory(ctx.device, memC, 0, sizeC, 0, (void**)&c_ptr);
    srand(42);
    for (size_t i = 0; i < M * K; i++) a_ptr[i] = (float)(rand() % 100) / 10.0f;
    for (size_t i = 0; i < K * N; i++) b_ptr[i] = (float)(rand() % 100) / 10.0f;
    memset(c_ptr, 0, sizeC);
    vkUnmapMemory(ctx.device, memA); vkUnmapMemory(ctx.device, memB); vkUnmapMemory(ctx.device, memC);

    // 6. Command buffer: dispatch with push constants
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = ctx.queueFamilyIndex;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(ctx.device, &pool_ci, nullptr, &pool);
    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = pool; cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmd_ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device, &cmd_ai, &cmd);

    struct GemmPC { uint64_t addrA, addrB, addrC; uint32_t M, N, K; float alpha; uint32_t transB; } push = {addrA, addrB, addrC, M, N, K, 1.0f, 0};

    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, N, M, 1);  // one workgroup per output element
        vkEndCommandBuffer(cmd);
    }

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);
    if (vkQueueSubmit(ctx.queues[0], 1, &si, fence) != VK_SUCCESS) { fprintf(stderr, "FAIL: QueueSubmit\n"); return 1; }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    // 7. Read back C and compare with CPU reference
    vkMapMemory(ctx.device, memC, 0, sizeC, 0, (void**)&c_ptr);
    vkMapMemory(ctx.device, memA, 0, sizeA, 0, (void**)&a_ptr);
    vkMapMemory(ctx.device, memB, 0, sizeB, 0, (void**)&b_ptr);

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float expected = 0.0f;
            for (uint32_t k = 0; k < K; k++) expected += a_ptr[i * K + k] * b_ptr[k * N + j];
            check("C", expected, c_ptr[i * N + j], i, j);
        }
    }

    vkUnmapMemory(ctx.device, memA); vkUnmapMemory(ctx.device, memB); vkUnmapMemory(ctx.device, memC);

    // 8. Cleanup
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pl, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, bufA, nullptr); vkFreeMemory(ctx.device, memA, nullptr);
    vkDestroyBuffer(ctx.device, bufB, nullptr); vkFreeMemory(ctx.device, memB, nullptr);
    vkDestroyBuffer(ctx.device, bufC, nullptr); vkFreeMemory(ctx.device, memC, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    ctx.cleanup();

    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu mismatches, max diff %f\n", errors, max_diff);
        return 1;
    }
    fprintf(stderr, "PASS: all %ux%u elements match (max diff=%f)\n", M, N, max_diff);
    return 0;
}
