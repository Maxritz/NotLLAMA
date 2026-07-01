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
        if (fs::exists(canon / "embed.comp")) return canon.string();
    }
    return "shaders";
}

int main() {
    fprintf(stderr, "Testing embed.comp (embedding table lookup)\n");

    VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext init\n"); return 1; }
    fprintf(stderr, "Vulkan OK (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    std::string shader_dir = FindShaderDir();

    notllama::ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc unavailable\n"); return 0; }

    notllama::ShaderCompileOptions opts;
    opts.src_path = shader_dir + "/embed.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.defines.push_back("SUBGROUP_SIZE=" + std::to_string(ctx.subgroupSize));

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: compile embed.comp: %s\n", log.c_str()); return 1;
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
    pc_range.size = sizeof(EmbedPushConstants);
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

    // Test: vocab=100, dim=256, tokenId=42, tokenPos=3
    const uint32_t vocabSize = 100;
    const uint32_t dim = 256;
    const uint32_t tokenId = 42;
    const uint32_t tokenPos = 3;
    const size_t embed_size = vocabSize * dim * sizeof(float);
    const size_t hidden_size = (tokenPos + 1) * dim * sizeof(float);

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

    VkBuffer buf_embed = VK_NULL_HANDLE, buf_hidden = VK_NULL_HANDLE;
    VkDeviceMemory mem_embed = VK_NULL_HANDLE, mem_hidden = VK_NULL_HANDLE;
    VkDeviceAddress addr_embed = 0, addr_hidden = 0;

    if (!create_buf(buf_embed, mem_embed, addr_embed, embed_size) ||
        !create_buf(buf_hidden, mem_hidden, addr_hidden, hidden_size)) {
        fprintf(stderr, "FAIL: CreateBuffers\n"); return 1;
    }

    float* embed_ptr, * hidden_ptr;
    vkMapMemory(ctx.device, mem_embed, 0, embed_size, 0, (void**)&embed_ptr);
    vkMapMemory(ctx.device, mem_hidden, 0, hidden_size, 0, (void**)&hidden_ptr);
    srand(42);
    for (size_t i = 0; i < vocabSize * dim; i++)
        embed_ptr[i] = (float)(rand() % 1000) / 100.0f;
    memset(hidden_ptr, 0, hidden_size);
    vkUnmapMemory(ctx.device, mem_embed); vkUnmapMemory(ctx.device, mem_hidden);

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

    EmbedPushConstants push{};
    push.addrEmbedTable = addr_embed;
    push.addrHiddenState = addr_hidden;
    push.tokenId = tokenId;
    push.tokenPos = tokenPos;
    push.dim = dim;

    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, 1, 1, 1);
        vkEndCommandBuffer(cmd);
    }

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);
    if (vkQueueSubmit(ctx.queues[0], 1, &si, fence) != VK_SUCCESS) { fprintf(stderr, "FAIL: QueueSubmit\n"); return 1; }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    vkMapMemory(ctx.device, mem_hidden, 0, hidden_size, 0, (void**)&hidden_ptr);
    vkMapMemory(ctx.device, mem_embed, 0, embed_size, 0, (void**)&embed_ptr);

    // Verify: hidden_state[tokenPos*dim + d] = embed_table[tokenId*dim + d]
    size_t errors = 0;
    double max_diff = 0.0;
    for (uint32_t d = 0; d < dim; d++) {
        float expected = embed_ptr[tokenId * dim + d];
        float actual = hidden_ptr[tokenPos * dim + d];
        double diff = std::abs((double)(expected - actual));
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            if (errors < 5) fprintf(stderr, "  MISMATCH[%u]: expected %f, got %f\n", d, expected, actual);
            errors++;
        }
    }
    // Ensure other positions are untouched
    for (uint32_t p = 0; p < tokenPos; p++) {
        for (uint32_t d = 0; d < dim; d++) {
            if (hidden_ptr[p * dim + d] != 0.0f) {
                if (errors < 5) fprintf(stderr, "  DIRTY[%u,%u]: expected 0, got %f\n", p, d, hidden_ptr[p * dim + d]);
                errors++;
            }
        }
    }

    vkUnmapMemory(ctx.device, mem_embed); vkUnmapMemory(ctx.device, mem_hidden);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pl, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, buf_embed, nullptr); vkFreeMemory(ctx.device, mem_embed, nullptr);
    vkDestroyBuffer(ctx.device, buf_hidden, nullptr); vkFreeMemory(ctx.device, mem_hidden, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);
    ctx.cleanup();

    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu mismatches, max diff %f\n", errors, max_diff);
        return 1;
    }
    fprintf(stderr, "PASS: all %u dimensions match (max diff=%f), other positions clean\n", dim, max_diff);
    return 0;
}
