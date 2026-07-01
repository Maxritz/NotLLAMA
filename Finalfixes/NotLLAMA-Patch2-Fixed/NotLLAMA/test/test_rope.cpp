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
        if (fs::exists(canon / "rope.comp")) return canon.string();
    }
    return "shaders";
}

int main() {
    fprintf(stderr, "Testing rope.comp (Rotary Position Embedding)\n");

    VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext init\n"); return 1; }
    fprintf(stderr, "Vulkan OK (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    std::string shader_dir = FindShaderDir();

    notllama::ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc unavailable\n"); return 0; }

    notllama::ShaderCompileOptions opts;
    opts.src_path = shader_dir + "/rope.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.defines.push_back("SUBGROUP_SIZE=" + std::to_string(ctx.subgroupSize));

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: compile rope.comp: %s\n", log.c_str()); return 1;
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
    pc_range.size = sizeof(RopePushConstants);
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

    // Test: 2 heads, headDim=64, seqLen=5 (so we're applying RoPE at pos=4)
    const uint32_t nHeads = 2;
    const uint32_t nKvHeads = 2;
    const uint32_t headDim = 64;
    const uint32_t seqLen = 5;
    const uint32_t dim = nHeads * headDim;        // = 128
    const size_t buf_size = seqLen * dim * sizeof(float);
    const float ropeBase = 10000.0f;
    const float ropeScale = 1.0f;

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
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &flagsInfo;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(ctx.device, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(ctx.device, buf, mem, 0) != VK_SUCCESS) return false;
        VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = buf;
        addr = vkGetBufferDeviceAddress(ctx.device, &bdai);
        return true;
    };

    VkBuffer buf_q = VK_NULL_HANDLE, buf_k = VK_NULL_HANDLE;
    VkDeviceMemory mem_q = VK_NULL_HANDLE, mem_k = VK_NULL_HANDLE;
    VkDeviceAddress addr_q = 0, addr_k = 0;

    if (!create_buf(buf_q, mem_q, addr_q, buf_size) || !create_buf(buf_k, mem_k, addr_k, buf_size)) {
        fprintf(stderr, "FAIL: CreateBuffers\n"); return 1;
    }

    float* q_ptr, * k_ptr;
    vkMapMemory(ctx.device, mem_q, 0, buf_size, 0, (void**)&q_ptr);
    vkMapMemory(ctx.device, mem_k, 0, buf_size, 0, (void**)&k_ptr);
    srand(42);
    for (size_t i = 0; i < seqLen * dim; i++) {
        q_ptr[i] = (float)(rand() % 1000) / 100.0f;
        k_ptr[i] = (float)(rand() % 1000) / 100.0f;
    }
    vkUnmapMemory(ctx.device, mem_q); vkUnmapMemory(ctx.device, mem_k);

    // Also keep CPU copy for reference
    float* q_cpu = new float[seqLen * dim];
    float* k_cpu = new float[seqLen * dim];
    vkMapMemory(ctx.device, mem_q, 0, buf_size, 0, (void**)&q_ptr);
    vkMapMemory(ctx.device, mem_k, 0, buf_size, 0, (void**)&k_ptr);
    memcpy(q_cpu, q_ptr, buf_size);
    memcpy(k_cpu, k_ptr, buf_size);
    vkUnmapMemory(ctx.device, mem_q); vkUnmapMemory(ctx.device, mem_k);

    // Compute CPU reference for RoPE at the last position (seqLen-1)
    // Process one even/odd pair at a time so we don't overwrite the even element
    // before computing the odd element.
    const uint32_t pos = seqLen - 1;
    for (uint32_t head = 0; head < nHeads; head++) {
        uint32_t kvHead = head / (nHeads / nKvHeads);
        for (uint32_t d = 0; d < headDim; d += 2) {
            float theta = powf(ropeBase, -(float)d / (float)headDim);
            float angle = (float)pos * theta * ropeScale;
            float cosA = cosf(angle);
            float sinA = sinf(angle);

            uint32_t qIdx = pos * dim + head * headDim + d;
            float q0 = q_cpu[qIdx];
            float q1 = q_cpu[qIdx + 1];
            q_cpu[qIdx]     = q0 * cosA - q1 * sinA;
            q_cpu[qIdx + 1] = q1 * cosA + q0 * sinA;

            uint32_t kIdx = pos * nKvHeads * headDim + kvHead * headDim + d;
            float k0 = k_cpu[kIdx];
            float k1 = k_cpu[kIdx + 1];
            k_cpu[kIdx]     = k0 * cosA - k1 * sinA;
            k_cpu[kIdx + 1] = k1 * cosA + k0 * sinA;
        }
    }

    // Dispatch
    VkCommandPoolCreateInfo pool_ci2{};
    pool_ci2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci2.queueFamilyIndex = ctx.queueFamilyIndex;
    pool_ci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(ctx.device, &pool_ci2, nullptr, &pool);
    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = pool; cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cmd_ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(ctx.device, &cmd_ai, &cmd);

    RopePushConstants push{};
    push.addrQ = addr_q;
    push.addrK = addr_k;
    push.seqLen = seqLen;
    push.headDim = headDim;
    push.nHeads = nHeads;
    push.nKvHeads = nKvHeads;
    push.ropeBase = ropeBase;
    push.ropeScale = ropeScale;

    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (nHeads * headDim / 2 + 31) / 32, 1, 1);
        vkEndCommandBuffer(cmd);
    }

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);
    if (vkQueueSubmit(ctx.queues[0], 1, &si, fence) != VK_SUCCESS) { fprintf(stderr, "FAIL: QueueSubmit\n"); return 1; }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    // Read back and verify
    vkMapMemory(ctx.device, mem_q, 0, buf_size, 0, (void**)&q_ptr);
    vkMapMemory(ctx.device, mem_k, 0, buf_size, 0, (void**)&k_ptr);

    size_t errors = 0;
    double max_diff = 0.0;
    // Only check the last position (where RoPE was applied)
    for (uint32_t head = 0; head < nHeads; head++) {
        for (uint32_t d = 0; d < headDim; d++) {
            uint32_t idx = pos * dim + head * headDim + d;
            double diff_q = std::abs((double)(q_cpu[idx] - q_ptr[idx]));
            double diff_k = std::abs((double)(k_cpu[idx] - k_ptr[idx]));
            if (diff_q > max_diff) max_diff = diff_q;
            if (diff_k > max_diff) max_diff = diff_k;
            if (diff_q > 0.01f) {
                if (errors < 5) fprintf(stderr, "  Q MISMATCH[%u,%u,%u]: expected %f, got %f\n", head, d, pos, q_cpu[idx], q_ptr[idx]);
                errors++;
            }
            if (diff_k > 0.01f) {
                if (errors < 5) fprintf(stderr, "  K MISMATCH[%u,%u,%u]: expected %f, got %f\n", head, d, pos, k_cpu[idx], k_ptr[idx]);
                errors++;
            }
        }
    }

    vkUnmapMemory(ctx.device, mem_q); vkUnmapMemory(ctx.device, mem_k);
    delete[] q_cpu; delete[] k_cpu;
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pl, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, buf_q, nullptr); vkFreeMemory(ctx.device, mem_q, nullptr);
    vkDestroyBuffer(ctx.device, buf_k, nullptr); vkFreeMemory(ctx.device, mem_k, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    ctx.cleanup();

    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu mismatches, max diff %f\n", errors, max_diff);
        return 1;
    }
    fprintf(stderr, "PASS: all elements match (max diff=%f)\n", max_diff);
    return 0;
}
