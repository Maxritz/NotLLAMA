#include "engine/engine.hpp"
#include "rdna4_vulkan.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>

using namespace notllama;

static std::string FindShaderDir() {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();
    for (const auto& p : {cwd / "shaders", cwd / ".." / ".." / "shaders", fs::path("shaders")}) {
        fs::path canon = fs::weakly_canonical(p);
        if (fs::exists(canon / "test_add.comp")) return canon.string();
    }
    return "shaders";
}

static bool ReadSpv(const std::string& path, std::vector<uint32_t>& code) {
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0 || sz % 4 != 0) { fclose(f); return false; }
    rewind(f);
    code.resize(sz / 4);
    bool ok = fread(code.data(), 1, sz, f) == (size_t)sz;
    fclose(f);
    return ok;
}

int main() {
    fprintf(stderr, "Kernel dispatch test: compile test_add.comp, dispatch, verify\n");

    rdna4::VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: VulkanContext init\n"); return 1; }
    fprintf(stderr, "Vulkan init OK (wave32=%d)\n", ctx.isWave32() ? 1 : 0);

    {
    // 2. Create command pool + command buffer
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = ctx.queueFamilyIndex;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx.device, &pool_ci, nullptr, &pool) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateCommandPool\n"); ctx.cleanup(); return 1;
    }

    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = pool;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.device, &cmd_ai, &cmd) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: AllocateCommandBuffers\n"); vkDestroyCommandPool(ctx.device, pool, nullptr); ctx.cleanup(); return 1;
    }

    // 3. Memory allocator + descriptor manager
    RingAllocatorAdapter allocator(ctx.device, ctx.physicalDevice, 32 * 1024 * 1024, 32 * 1024 * 1024, 4 * 1024 * 1024);
    VulkanDescriptorManager desc_mgr(ctx.device, ctx.physicalDevice, &allocator);
    if (!desc_mgr.IsValid()) { fprintf(stderr, "FAIL: DescriptorManager\n"); return 1; }

    // 4. Compile test_add.comp
    ShaderCompiler compiler;
    if (!compiler.IsAvailable()) { fprintf(stderr, "SKIP: glslc unavailable\n"); return 0; }

    std::string shader_dir = FindShaderDir();
    ShaderCompileOptions opts;
    opts.src_path = shader_dir + "/test_add.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.defines.push_back("SUBGROUP_SIZE=" + std::to_string(ctx.subgroupSize));

    std::vector<uint32_t> spv;
    std::string log;
    if (!compiler.Compile(opts, spv, &log)) {
        fprintf(stderr, "FAIL: Compile test_add.comp: %s\n", log.c_str()); return 1;
    }
    fprintf(stderr, "Compiled test_add.comp (%zu words)\n", spv.size());

    // 5. Create simple descriptor set layout with 3 storage buffers (set=0, bindings 0,1,2)
    // Avoids AMD driver crash with bindless (update-after-bind) + wave32 control.
    fprintf(stderr, "  creating descriptor set layout\n"); fflush(stderr);
    VkDescriptorSetLayoutBinding dsl_bindings[3] = {};
    for (int i = 0; i < 3; i++) {
        dsl_bindings[i].binding = i;
        dsl_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dsl_bindings[i].descriptorCount = 1;
        dsl_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 3;
    dsl_ci.pBindings = dsl_bindings;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(ctx.device, &dsl_ci, nullptr, &set_layout) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateDescriptorSetLayout\n"); return 1;
    }
    fprintf(stderr, "  created set_layout=%p\n", (void*)set_layout); fflush(stderr);
    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &set_layout;
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 128;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(ctx.device, &layout_ci, nullptr, &pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreatePipelineLayout\n"); return 1;
    }
    fprintf(stderr, "  pipeline layout created\n"); fflush(stderr);

    // 6. Shader module
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv.size() * sizeof(uint32_t);
    sm_ci.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &sm_ci, nullptr, &module) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateShaderModule\n"); return 1;
    }
    fprintf(stderr, "  shader module created\n"); fflush(stderr);

    // 7. Compute pipeline with wave32 subgroup size control
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
    pipe_ci.layout = pipeline_layout;
    pipe_ci.basePipelineHandle = VK_NULL_HANDLE;
    pipe_ci.basePipelineIndex = -1;

    VkPipeline pipeline = VK_NULL_HANDLE;
    fprintf(stderr, "  creating pipeline...\n"); fflush(stderr);
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &pipeline) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateComputePipeline\n"); return 1;
    }
    fprintf(stderr, "Pipeline created (wave32=%d)\n", ctx.isWave32() ? 1 : 0);
    fflush(stderr);

    // 8. Allocate SSBOs: A, B, C each 1024 floats
    fprintf(stderr, "  creating buffers...\n"); fflush(stderr);
    VkMemoryPropertyFlags host_visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const size_t buf_size = 1024 * sizeof(float);

    auto createBuf = [&](VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr) -> bool {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx.device, &bci, nullptr, &buf) != VK_SUCCESS) return false;

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(ctx.device, buf, &mem_req);

        VkPhysicalDeviceMemoryProperties phys_mem;
        vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &phys_mem);

        uint32_t mem_type = UINT32_MAX;
        for (uint32_t i = 0; i < phys_mem.memoryTypeCount; i++) {
            if ((mem_req.memoryTypeBits & (1u << i)) &&
                (phys_mem.memoryTypes[i].propertyFlags & host_visible) == host_visible) {
                mem_type = i; break;
            }
        }
        if (mem_type == UINT32_MAX) return false;

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mem_req.size;
        mai.memoryTypeIndex = mem_type;
        if (vkAllocateMemory(ctx.device, &mai, nullptr, &mem) != VK_SUCCESS) return false;

        if (vkBindBufferMemory(ctx.device, buf, mem, 0) != VK_SUCCESS) return false;

        VkBufferDeviceAddressInfo bdai{};
        bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdai.buffer = buf;
        addr = vkGetBufferDeviceAddress(ctx.device, &bdai);
        return true;
    };

    VkBuffer buf_a = VK_NULL_HANDLE, buf_b = VK_NULL_HANDLE, buf_c = VK_NULL_HANDLE;
    VkDeviceMemory mem_a = VK_NULL_HANDLE, mem_b = VK_NULL_HANDLE, mem_c = VK_NULL_HANDLE;
    VkDeviceAddress addr_a = 0, addr_b = 0, addr_c = 0;

    if (!createBuf(buf_a, mem_a, addr_a) || !createBuf(buf_b, mem_b, addr_b) || !createBuf(buf_c, mem_c, addr_c)) {
        fprintf(stderr, "FAIL: Create buffers\n"); return 1;
    }

    // 9. Fill A and B with random data
    float* a_ptr, * b_ptr, * c_ptr;
    vkMapMemory(ctx.device, mem_a, 0, buf_size, 0, (void**)&a_ptr);
    vkMapMemory(ctx.device, mem_b, 0, buf_size, 0, (void**)&b_ptr);
    vkMapMemory(ctx.device, mem_c, 0, buf_size, 0, (void**)&c_ptr);

    srand(42);
    for (size_t i = 0; i < 1024; i++) {
        a_ptr[i] = (float)(rand() % 1000) / 100.0f;
        b_ptr[i] = (float)(rand() % 1000) / 100.0f;
        c_ptr[i] = 0.0f;
    }
    // Ensure memory is visible (host coherent)
    vkUnmapMemory(ctx.device, mem_a);
    vkUnmapMemory(ctx.device, mem_b);
    vkUnmapMemory(ctx.device, mem_c);

    // 10. Allocate descriptor set with 3 storage buffer bindings
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(ctx.device, &dp_ci, nullptr, &desc_pool) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: CreateDescriptorPool\n"); return 1;
    }
    VkDescriptorSetAllocateInfo ds_ai{};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = desc_pool;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &set_layout;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(ctx.device, &ds_ai, &desc_set) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: AllocateDescriptorSets\n"); return 1;
    }

    // Write buffer descriptors at bindings 0, 1, 2
    VkDescriptorBufferInfo buf_info[3] = {};
    VkWriteDescriptorSet desc_writes[3] = {};
    VkBuffer bufs[3] = {buf_a, buf_b, buf_c};
    for (uint32_t i = 0; i < 3; i++) {
        buf_info[i].buffer = bufs[i];
        buf_info[i].offset = 0;
        buf_info[i].range = buf_size;
        desc_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        desc_writes[i].dstSet = desc_set;
        desc_writes[i].dstBinding = i;
        desc_writes[i].descriptorCount = 1;
        desc_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        desc_writes[i].pBufferInfo = &buf_info[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, desc_writes, 0, nullptr);

    // 11. Command buffer: dispatch test_add.comp with n=1024
    {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &begin);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
        uint32_t push_n = 1024;
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &push_n);
        vkCmdDispatch(cmd, (1024 + ctx.subgroupSize - 1) / ctx.subgroupSize, 1, 1);

        vkEndCommandBuffer(cmd);
    }

    // 12. Submit and wait
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fci, nullptr, &fence);

    if (vkQueueSubmit(ctx.queues[0], 1, &submit, fence) != VK_SUCCESS) {
        fprintf(stderr, "FAIL: QueueSubmit\n"); return 1;
    }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);

    // 13. Read back C (keep mapped for verification)
    vkMapMemory(ctx.device, mem_c, 0, buf_size, 0, (void**)&c_ptr);
    vkMapMemory(ctx.device, mem_a, 0, buf_size, 0, (void**)&a_ptr);
    vkMapMemory(ctx.device, mem_b, 0, buf_size, 0, (void**)&b_ptr);

    // 14. Verify against CPU reference
    size_t errors = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < 1024; i++) {
        float expected = a_ptr[i] + b_ptr[i];
        float actual = c_ptr[i];
        double diff = std::abs((double)(expected - actual));
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) {
            if (errors < 5)
                fprintf(stderr, "  MISMATCH[%zu]: expected %f, got %f (diff=%f)\n", i, expected, actual, diff);
            errors++;
        }
    }

    // 15. Cleanup (unmap after verification; then destroy resources)
    vkUnmapMemory(ctx.device, mem_a);
    vkUnmapMemory(ctx.device, mem_b);
    vkUnmapMemory(ctx.device, mem_c);

    vkDestroyDescriptorPool(ctx.device, desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyBuffer(ctx.device, buf_a, nullptr); vkFreeMemory(ctx.device, mem_a, nullptr);
    vkDestroyBuffer(ctx.device, buf_b, nullptr); vkFreeMemory(ctx.device, mem_b, nullptr);
    vkDestroyBuffer(ctx.device, buf_c, nullptr); vkFreeMemory(ctx.device, mem_c, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);

    if (errors > 0) {
        fprintf(stderr, "FAIL: %zu elements mismatched (max diff=%f)\n", errors, max_diff);
        ctx.cleanup();
        return 1;
    }
    fprintf(stderr, "PASS: all 1024 elements match (max diff=%f)\n", max_diff);
    }

    ctx.cleanup();
    return 0;
}
