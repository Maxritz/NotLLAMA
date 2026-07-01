#include "rdna4_vulkan.hpp"
#include "rdna4_types.hpp"
#include "engine/shader_compiler.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <vector>
#include <string>

using namespace rdna4;

static std::string FindShaderDir() {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();
    for (const auto& p : {cwd / "shaders", cwd / ".." / ".." / "shaders", fs::path("shaders")}) {
        fs::path canon = fs::weakly_canonical(p);
        if (fs::exists(canon / "silu_mul.comp")) return canon.string();
    }
    return "shaders";
}

int main() {
    fprintf(stderr, "Testing silu_mul.comp (SiLU(gate) * up)\n");

    VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext init\n"); return 1; }
    fprintf(stderr, "Vulkan OK (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    std::string shader_dir = FindShaderDir();

    notllama::ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc unavailable\n"); return 0; }

    notllama::ShaderCompileOptions opts;
    opts.src_path = shader_dir + "/silu_mul.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.defines.push_back("SUBGROUP_SIZE=" + std::to_string(ctx.subgroupSize));

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: compile silu_mul.comp: %s\n", log.c_str()); return 1;
    }
    fprintf(stderr, "Compiled (%zu words)\n", spv.size());

    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv.size() * sizeof(uint32_t);
    sm_ci.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &sm_ci, nullptr, &module) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateShaderModule\n"); return 1;
    }

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 0;
    pl_ci.pSetLayouts = nullptr;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = sizeof(SiluMulPushConstants);
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
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &pipeline) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateComputePipeline\n"); return 1;
    }
    fprintf(stderr, "Pipeline OK\n");

    // Test: 1024 elements
    const uint32_t nElements = 1024;
    const size_t buf_size = nElements * sizeof(float);

    auto create_buf = [&](VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr, size_t sz) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx.device, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(ctx.device, buf, &mr);
        VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &pm);
        uint32_t mt = UINT32_MAX;
        VkMemoryPropertyFlags hf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & hf) == hf)
            { mt = i; break; }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(ctx.device, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(ctx.device, buf, mem, 0) != VK_SUCCESS) return false;
        VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = buf;
        addr = vkGetBufferDeviceAddress(ctx.device, &bdai);
        return true;
    };

    VkBuffer buf_gate = VK_NULL_HANDLE, buf_up = VK_NULL_HANDLE, buf_out = VK_NULL_HANDLE;
    VkDeviceMemory mem_gate = VK_NULL_HANDLE, mem_up = VK_NULL_HANDLE, mem_out = VK_NULL_HANDLE;
    VkDeviceAddress addr_gate = 0, addr_up = 0, addr_out = 0;

    if (!create_buf(buf_gate, mem_gate, addr_gate, buf_size) ||
        !create_buf(buf_up, mem_up, addr_up, buf_size) ||
        !create_buf(buf_out, mem_out, addr_out, buf_size)) {
        fprintf(stderr, "FAIL: CreateBuffers\n"); return 1;
    }

    float* gate_ptr, * up_ptr, * out_ptr;
    vkMapMemory(ctx.device, mem_gate, 0, buf_size, 0, (void**)&gate_ptr);
    vkMapMemory(ctx.device, mem_up, 0, buf_size, 0, (void**)&up_ptr);
    vkMapMemory(ctx.device, mem_out, 0, buf_size, 0, (void**)&out_ptr);
    srand(42);
    for (size_t i = 0; i < nElements; i++) {
        gate_ptr[i] = (float)(rand() % 2000 - 1000) / 100.0f;
        up_ptr[i] = (float)(rand() % 1000) / 100.0f;
    }
    memset(out_ptr, 0, buf_size);
    vkUnmapMemory(ctx.device, mem_gate); vkUnmapMemory(ctx.device, mem_up); vkUnmapMemory(ctx.device, mem_out);

    // CPU reference
    float* expected = new float[nElements];
    for (size_t i = 0; i < nElements; i++) {
        float silu = gate_ptr[i] / (1.0f + expf(-gate_ptr[i]));
        expected[i] = silu * up_ptr[i];
    }

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

    SiluMulPushConstants push{};
    push.addrGate = addr_gate;
    push.addrUp = addr_up;
    push.addrOut = addr_out;
    push.nElements = nElements;

    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (nElements + 255) / 256, 1, 1);
        vkEndCommandBuffer(cmd);
    }

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);
    if (vkQueueSubmit(ctx.queues[0], 1, &si, fence) != VK_SUCCESS) { fprintf(stderr, "FAIL: QueueSubmit\n"); return 1; }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    vkMapMemory(ctx.device, mem_out, 0, buf_size, 0, (void**)&out_ptr);

    size_t errors = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < nElements; i++) {
        double diff = std::abs((double)(expected[i] - out_ptr[i]));
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            if (errors < 5) fprintf(stderr, "  MISMATCH[%zu]: expected %f, got %f\n", i, expected[i], out_ptr[i]);
            errors++;
        }
    }

    delete[] expected;
    vkUnmapMemory(ctx.device, mem_out);
    vkUnmapMemory(ctx.device, mem_gate);
    vkUnmapMemory(ctx.device, mem_up);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pl, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, buf_gate, nullptr); vkFreeMemory(ctx.device, mem_gate, nullptr);
    vkDestroyBuffer(ctx.device, buf_up, nullptr); vkFreeMemory(ctx.device, mem_up, nullptr);
    vkDestroyBuffer(ctx.device, buf_out, nullptr); vkFreeMemory(ctx.device, mem_out, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    ctx.cleanup();

    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu mismatches, max diff %f\n", errors, max_diff);
        return 1;
    }
    fprintf(stderr, "PASS: all %u elements match (max diff=%f)\n", nElements, max_diff);
    return 0;
}
