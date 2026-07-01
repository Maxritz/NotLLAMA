#include "vulkan/vk_backend.h"
#include "vulkan/vk_shader.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>

// Inline binary shader data (compiled SPIR-V would be embedded here at build time)
// For now, these are loaded at runtime via shader_compiler.cpp

static VkResult create_desc_layout(VkDevice dev, uint32_t binding_count,
                                   VkDescriptorSetLayout* out) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
    for (uint32_t i = 0; i < binding_count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = binding_count;
    ci.pBindings = bindings.data();
    return vkCreateDescriptorSetLayout(dev, &ci, nullptr, out);
}

static uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t type_filter,
                              VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
        if ((type_filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

bool VulkanBackend::init(VkInstance instance) {
    dev = create_device(instance);
    if (!dev.device) return false;

    // Command pool
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = dev.compute_queue_family;
    if (vkCreateCommandPool(dev.device, &cpci, nullptr, &cmd_pool) != VK_SUCCESS)
        return false;

    // Command buffers: [0]=single, [1]=batch
    VkCommandBuffer cbs[2];
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    if (vkAllocateCommandBuffers(dev.device, &cbai, cbs) != VK_SUCCESS)
        return false;
    cmd = cbs[0];
    batch_cmd = cbs[1];

    // Fence
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(dev.device, &fci, nullptr, &fence) != VK_SUCCESS)
        return false;

    // Staging buffer (64MB)
    staging_size = 64 * 1024 * 1024;
    GpuBuffer staging = create_buffer(staging_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    staging_buf = staging.buffer;
    staging_mem = staging.memory;

    // Descriptor resources
    if (!create_descriptor_resources()) return false;

    printf("VulkanBackend initialized\n");
    return true;
}

bool VulkanBackend::create_descriptor_resources() {
    VkDevice d = dev.device;

    // Layouts
    if (create_desc_layout(d, 3, &desc_3buf_layout) != VK_SUCCESS) return false;
    if (create_desc_layout(d, 2, &desc_2buf_layout) != VK_SUCCESS) return false;
    if (create_desc_layout(d, 1, &desc_1buf_layout) != VK_SUCCESS) return false;
    if (create_desc_layout(d, 4, &desc_4buf_layout) != VK_SUCCESS) return false;

    // Pool: 128 sets × max 4 descriptors = 512
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 512;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 128;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(d, &dpci, nullptr, &desc_pool) != VK_SUCCESS) return false;

    // Allocate DESC_POOL_COUNT sets for each layout
    VkDescriptorSetLayout layouts[4] = {desc_3buf_layout, desc_2buf_layout, desc_1buf_layout, desc_4buf_layout};
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = desc_pool;
    dsai.descriptorSetCount = 1;

    for (int i = 0; i < DESC_POOL_COUNT; i++) {
        dsai.pSetLayouts = &layouts[0];
        if (vkAllocateDescriptorSets(d, &dsai, &desc_3buf_set[i]) != VK_SUCCESS) return false;
        dsai.pSetLayouts = &layouts[1];
        if (vkAllocateDescriptorSets(d, &dsai, &desc_2buf_set[i]) != VK_SUCCESS) return false;
        dsai.pSetLayouts = &layouts[2];
        if (vkAllocateDescriptorSets(d, &dsai, &desc_1buf_set[i]) != VK_SUCCESS) return false;
        dsai.pSetLayouts = &layouts[3];
        if (vkAllocateDescriptorSets(d, &dsai, &desc_4buf_set[i]) != VK_SUCCESS) return false;
    }
    return true;
}

void VulkanBackend::cleanup() {
    VkDevice d = dev.device;
    if (!d) return;

    auto destroy_pipe = [&](PipelineBundle& p) {
        if (p.pipeline) vkDestroyPipeline(d, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(d, p.layout, nullptr);
    };
    destroy_pipe(gemm_fp32_pipe);
    destroy_pipe(gemm_q4_0_pipe);
    destroy_pipe(gemm_q8_0_pipe);
    destroy_pipe(gemm_wmma_pipe);
    destroy_pipe(convert_f32_f16_pipe);
    destroy_pipe(rms_norm_pipe);
    destroy_pipe(rope_pipe);
    destroy_pipe(silu_mul_pipe);
    destroy_pipe(token_embed_pipe);
    destroy_pipe(add_pipe);
    destroy_pipe(attn_wmma_pipe);
    destroy_pipe(attn_softmax_pipe);
    destroy_pipe(attn_value_pipe);
    destroy_pipe(argmax_pipe);

    if (desc_3buf_layout) vkDestroyDescriptorSetLayout(d, desc_3buf_layout, nullptr);
    if (desc_2buf_layout) vkDestroyDescriptorSetLayout(d, desc_2buf_layout, nullptr);
    if (desc_1buf_layout) vkDestroyDescriptorSetLayout(d, desc_1buf_layout, nullptr);
    if (desc_4buf_layout) vkDestroyDescriptorSetLayout(d, desc_4buf_layout, nullptr);
    if (desc_pool) vkDestroyDescriptorPool(d, desc_pool, nullptr);

    if (staging_buf) vkDestroyBuffer(d, staging_buf, nullptr);
    if (staging_mem) vkFreeMemory(d, staging_mem, nullptr);

    if (cmd_pool) vkDestroyCommandPool(d, cmd_pool, nullptr);
    if (fence) vkDestroyFence(d, fence, nullptr);

    destroy_device(dev);
}

GpuBuffer VulkanBackend::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                        VkMemoryPropertyFlags mem_props, bool mapped) {
    GpuBuffer buf{};
    VkDevice d = dev.device;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d, &bci, nullptr, &buf.buffer) != VK_SUCCESS) return buf;

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(d, buf.buffer, &mr);

    uint32_t mem_type = find_mem_type(dev.phys_dev, mr.memoryTypeBits, mem_props);
    if (mem_type == UINT32_MAX) { vkDestroyBuffer(d, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; return buf; }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_type;
    if (vkAllocateMemory(d, &mai, nullptr, &buf.memory) != VK_SUCCESS) {
        vkDestroyBuffer(d, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; return buf;
    }

    vkBindBufferMemory(d, buf.buffer, buf.memory, 0);
    buf.size = size;
    if (mapped) vkMapMemory(d, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped);
    return buf;
}

void VulkanBackend::destroy_buffer(const GpuBuffer& buf) {
    VkDevice d = dev.device;
    if (buf.buffer) vkDestroyBuffer(d, buf.buffer, nullptr);
    if (buf.memory) {
        if (buf.mapped) vkUnmapMemory(d, buf.memory);
        vkFreeMemory(d, buf.memory, nullptr);
    }
}

void VulkanBackend::upload_to_buffer(const GpuBuffer& dst, const void* data, VkDeviceSize size) {
    VkDevice d = dev.device;
    if (staging_offset + size > staging_size) staging_offset = 0;
    void* map = nullptr;
    vkMapMemory(d, staging_mem, staging_offset, size, 0, &map);
    memcpy(map, data, (size_t)size);
    vkUnmapMemory(d, staging_mem);

    // Begin command buffer, record copy, submit, wait
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    VkBufferCopy bc{};
    bc.srcOffset = staging_offset;
    bc.dstOffset = 0;
    bc.size = size;
    vkCmdCopyBuffer(cmd, staging_buf, dst.buffer, 1, &bc);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(d, 1, &fence);
    vkQueueSubmit(dev.compute_queue, 1, &si, fence);
    vkWaitForFences(d, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetCommandPool(d, cmd_pool, 0);

    staging_offset += size;
}

void VulkanBackend::download_from_buffer(const GpuBuffer& src, void* data, VkDeviceSize size) {
    VkDevice d = dev.device;
    if (staging_offset + size > staging_size) staging_offset = 0;
    VkCommandBuffer cb = cmd;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferCopy bc{};
    bc.srcOffset = 0;
    bc.dstOffset = staging_offset;
    bc.size = size;
    vkCmdCopyBuffer(cb, src.buffer, staging_buf, 1, &bc);

    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkResetFences(d, 1, &fence);
    vkQueueSubmit(dev.compute_queue, 1, &si, fence);
    vkWaitForFences(d, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetCommandPool(d, cmd_pool, 0);

    void* map = nullptr;
    vkMapMemory(d, staging_mem, staging_offset, size, 0, &map);
    memcpy(data, map, (size_t)size);
    vkUnmapMemory(d, staging_mem);
}

void VulkanBackend::copy_buffer(const GpuBuffer& src, const GpuBuffer& dst, VkDeviceSize size) {
    VkBufferCopy bc{};
    bc.size = size;
    vkCmdCopyBuffer(current_cmd(), src.buffer, dst.buffer, 1, &bc);
}

void VulkanBackend::begin_batch() {
    batch_mode = true;
    desc_idx = 0;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(batch_cmd, &bi);
}

void VulkanBackend::end_batch() {
    batch_mode = false;
    vkEndCommandBuffer(batch_cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &batch_cmd;
    vkResetFences(dev.device, 1, &fence);
    vkQueueSubmit(dev.compute_queue, 1, &si, fence);
    vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetCommandPool(dev.device, cmd_pool, 0);
}

VkCommandBuffer VulkanBackend::current_cmd() {
    return batch_mode ? batch_cmd : cmd;
}

void VulkanBackend::push_constants(VkPipelineLayout layout, const void* data, uint32_t size) {
    vkCmdPushConstants(current_cmd(), layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void VulkanBackend::bind_descriptors(VkPipelineLayout layout, VkDescriptorSet set) {
    vkCmdBindDescriptorSets(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout, 0, 1, &set, 0, nullptr);
}

void VulkanBackend::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(current_cmd(), x, y, z);
}

void VulkanBackend::barrier() {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(current_cmd(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanBackend::update_desc_3buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1, VkBuffer b2,
                                      VkDeviceSize off0, VkDeviceSize off1, VkDeviceSize off2) {
    VkDescriptorBufferInfo infos[3] = {
        {b0, off0, VK_WHOLE_SIZE},
        {b1, off1, VK_WHOLE_SIZE},
        {b2, off2, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev.device, 3, writes, 0, nullptr);
}

void VulkanBackend::update_desc_2buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1) {
    VkDescriptorBufferInfo infos[2] = {{b0, 0, VK_WHOLE_SIZE}, {b1, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2] = {};
    for (int i = 0; i < 2; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev.device, 2, writes, 0, nullptr);
}

void VulkanBackend::update_desc_1buf(VkDescriptorSet set, VkBuffer b0) {
    VkDescriptorBufferInfo info{b0, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);
}

void VulkanBackend::update_desc_4buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1,
                                      VkBuffer b2, VkBuffer b3) {
    VkDescriptorBufferInfo infos[4] = {{b0, 0, VK_WHOLE_SIZE}, {b1, 0, VK_WHOLE_SIZE},
                                       {b2, 0, VK_WHOLE_SIZE}, {b3, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[4] = {};
    for (int i = 0; i < 4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev.device, 4, writes, 0, nullptr);
}

bool VulkanBackend::create_pipeline(const uint32_t* spirv, size_t size,
                                     VkDescriptorSetLayout desc_layout,
                                     uint32_t push_size, PipelineBundle& out,
                                     uint32_t required_subgroup_size) {
    VkDevice d = dev.device;

    VkShaderModule mod = create_shader_module(d, spirv, size);
    if (!mod) return false;

    // Pipeline layout
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = push_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &desc_layout;
    plci.pushConstantRangeCount = push_size ? 1 : 0;
    plci.pPushConstantRanges = push_size ? &push : nullptr;
    if (vkCreatePipelineLayout(d, &plci, nullptr, &out.layout) != VK_SUCCESS) {
        vkDestroyShaderModule(d, mod, nullptr);
        return false;
    }

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = out.layout;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req{};
    if (required_subgroup_size) {
        req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        req.requiredSubgroupSize = required_subgroup_size;
        cpci.stage.pNext = &req;
    }

    VkResult res = create_pipeline_seh(d, VK_NULL_HANDLE, 1, &cpci, nullptr, &out.pipeline);
    vkDestroyShaderModule(d, mod, nullptr);
    return res == VK_SUCCESS;
}

bool VulkanBackend::load_shader(const char* name, const uint32_t*, size_t, PipelineBundle&) {
    // Stub: shaders are loaded at runtime via shader_compiler
    fprintf(stderr, "load_shader(%s) — use runtime compilation\n", name);
    return false;
}

void VulkanBackend::gemm(const GpuBuffer& A, const GpuBuffer& B, const GpuBuffer& C,
                          uint32_t M, uint32_t N, uint32_t K,
                          bool A_is_fp16, bool B_is_fp16) {
    if (dev.coop_mat.supported && M >= 16 && N >= 16 && K >= 16 && A_is_fp16 && B_is_fp16) {
        uint32_t groups_x = (N + 15) / 16;
        uint32_t groups_y = (M + 15) / 16;
        begin_batch();
        int i = desc_idx++ % DESC_POOL_COUNT;
        update_desc_3buf(desc_3buf_set[i], A.buffer, B.buffer, C.buffer);
        vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, gemm_wmma_pipe.pipeline);
        bind_descriptors(gemm_wmma_pipe.layout, desc_3buf_set[i]);
        struct PC { uint32_t M, N, K; } pc = {M, N, K};
        push_constants(gemm_wmma_pipe.layout, &pc, sizeof(pc));
        dispatch(groups_x, groups_y);
        end_batch();
    } else {
        uint32_t groups_x = N;
        uint32_t groups_y = M;

        if (!batch_mode) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(cmd, &bi);
        }

        int i = desc_idx++ % DESC_POOL_COUNT;
        update_desc_3buf(desc_3buf_set[i], A.buffer, B.buffer, C.buffer);
        vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, gemm_fp32_pipe.pipeline);
        bind_descriptors(gemm_fp32_pipe.layout, desc_3buf_set[i]);
        struct PC { uint32_t M, N, K; } pc = {M, N, K};
        push_constants(gemm_fp32_pipe.layout, &pc, sizeof(pc));
        dispatch(groups_x, groups_y);

        if (!batch_mode) {
            vkEndCommandBuffer(cmd);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            vkResetFences(dev.device, 1, &fence);
            vkQueueSubmit(dev.compute_queue, 1, &si, fence);
            vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetCommandPool(dev.device, cmd_pool, 0);
        }
    }
}

void VulkanBackend::add(const GpuBuffer& a, const GpuBuffer& b, const GpuBuffer& c, uint32_t count) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], a.buffer, b.buffer, c.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, add_pipe.pipeline);
    bind_descriptors(add_pipe.layout, desc_3buf_set[i]);
    struct PC { uint32_t count; } pc = {count};
    push_constants(add_pipe.layout, &pc, sizeof(pc));
    dispatch((count + 255) / 256);
}

void VulkanBackend::rms_norm(const GpuBuffer& input, const GpuBuffer& weight,
                              const GpuBuffer& output, uint32_t dim, float eps, uint32_t rows) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], input.buffer, weight.buffer, output.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, rms_norm_pipe.pipeline);
    bind_descriptors(rms_norm_pipe.layout, desc_3buf_set[i]);
    struct PC { uint32_t dim; float eps; } pc = {dim, eps};
    push_constants(rms_norm_pipe.layout, &pc, sizeof(pc));
    dispatch(rows);
}

void VulkanBackend::rope(const GpuBuffer& data, uint32_t head_dim, uint32_t num_heads,
                          uint32_t position, float theta_base) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_1buf(desc_1buf_set[i], data.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, rope_pipe.pipeline);
    bind_descriptors(rope_pipe.layout, desc_1buf_set[i]);
    struct PC { uint32_t head_dim; uint32_t num_heads; uint32_t position; float theta_base; }
        pc = {head_dim, num_heads, position, theta_base};
    push_constants(rope_pipe.layout, &pc, sizeof(pc));
    dispatch(num_heads);
}

void VulkanBackend::silu_mul(const GpuBuffer& gate, const GpuBuffer& up,
                              const GpuBuffer& output, uint32_t count) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], gate.buffer, up.buffer, output.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, silu_mul_pipe.pipeline);
    bind_descriptors(silu_mul_pipe.layout, desc_3buf_set[i]);
    struct PC { uint32_t count; } pc = {count};
    push_constants(silu_mul_pipe.layout, &pc, sizeof(pc));
    dispatch((count + 255) / 256);
}

void VulkanBackend::token_embedding(const GpuBuffer& table, const GpuBuffer& output,
                                     uint32_t token_id, uint32_t embed_dim,
                                     uint32_t vocab_size, float scale) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_2buf(desc_2buf_set[i], table.buffer, output.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, token_embed_pipe.pipeline);
    bind_descriptors(token_embed_pipe.layout, desc_2buf_set[i]);
    struct PC { uint32_t token_id; uint32_t embed_dim; uint32_t vocab_size; float scale; }
        pc = {token_id, embed_dim, vocab_size, scale};
    push_constants(token_embed_pipe.layout, &pc, sizeof(pc));
    dispatch((embed_dim + 255) / 256);
}

void VulkanBackend::attention_scores_wmma(const GpuBuffer& Q, const GpuBuffer& K_cache,
                                           const GpuBuffer& scores, uint32_t head_dim,
                                           uint32_t seq_len, uint32_t max_seq, float scale,
                                           uint32_t num_q_heads, uint32_t gqa_ratio, uint32_t off_q) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], Q.buffer, K_cache.buffer, scores.buffer);

    struct PC {
        uint32_t head_dim, seq_len, max_seq;
        float scale;
        uint32_t num_q_heads, gqa_ratio, off_q;
    } pc = {head_dim, seq_len, max_seq, scale, num_q_heads, gqa_ratio, off_q};

    if (attn_scores_pipe.pipeline) {
        vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, attn_scores_pipe.pipeline);
        bind_descriptors(attn_scores_pipe.layout, desc_3buf_set[i]);
        push_constants(attn_scores_pipe.layout, &pc, sizeof(pc));
    } else {
        vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, attn_wmma_pipe.pipeline);
        bind_descriptors(attn_wmma_pipe.layout, desc_3buf_set[i]);
        push_constants(attn_wmma_pipe.layout, &pc, sizeof(pc));
    }
    dispatch(num_q_heads, (seq_len + 255) / 256);
}

void VulkanBackend::attention_softmax(const GpuBuffer& scores, uint32_t seq_len,
                                       uint32_t max_seq, uint32_t num_q_heads) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_1buf(desc_1buf_set[i], scores.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, attn_softmax_pipe.pipeline);
    bind_descriptors(attn_softmax_pipe.layout, desc_1buf_set[i]);
    struct PC { uint32_t seq_len; uint32_t max_seq; uint32_t num_q_heads; }
        pc = {seq_len, max_seq, num_q_heads};
    push_constants(attn_softmax_pipe.layout, &pc, sizeof(pc));
    dispatch(num_q_heads);
}

void VulkanBackend::attention_value(const GpuBuffer& scores, const GpuBuffer& V_cache,
                                     const GpuBuffer& output, uint32_t head_dim,
                                     uint32_t seq_len, uint32_t max_seq,
                                     uint32_t num_q_heads, uint32_t gqa_ratio) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], scores.buffer, V_cache.buffer, output.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, attn_value_pipe.pipeline);
    bind_descriptors(attn_value_pipe.layout, desc_3buf_set[i]);
    struct PC {
        uint32_t head_dim, seq_len, max_seq;
        uint32_t num_q_heads, gqa_ratio;
    } pc = {head_dim, seq_len, max_seq, num_q_heads, gqa_ratio};
    push_constants(attn_value_pipe.layout, &pc, sizeof(pc));
    dispatch(head_dim, num_q_heads);
}

static bool read_spv(const char* path, std::vector<uint32_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open SPIR-V: %s\n", path); return false; }
    size_t size = (size_t)f.tellg();
    if (size % 4) { fprintf(stderr, "SPIR-V size not multiple of 4: %s\n", path); return false; }
    out.resize(size / 4);
    f.seekg(0);
    f.read((char*)out.data(), size);
    return true;
}

bool VulkanBackend::load_pipelines(const char* spv_dir) {
    struct ShaderEntry {
        const char* name;
        PipelineBundle& bundle;
        VkDescriptorSetLayout layout;
        uint32_t push_size;
    };

    ShaderEntry required[] = {
        {"gemm_fp32",        gemm_fp32_pipe,        desc_3buf_layout, 12},
        {"gemm_q4_0",        gemm_q4_0_pipe,        desc_3buf_layout, 16},
        {"gemm_q8_0",        gemm_q8_0_pipe,        desc_3buf_layout, 16},
        {"convert_f32_f16",  convert_f32_f16_pipe,  desc_2buf_layout,  4},
        {"rms_norm",         rms_norm_pipe,         desc_3buf_layout,  8},
        {"rope",             rope_pipe,             desc_1buf_layout, 20},
        {"silu_mul",         silu_mul_pipe,         desc_3buf_layout,  4},
        {"add",              add_pipe,              desc_3buf_layout,  4},
        {"token_embedding",  token_embed_pipe,      desc_2buf_layout, 16},
        {"attention_scores", attn_scores_pipe,      desc_3buf_layout, 32},
        {"attention_softmax",attn_softmax_pipe,     desc_1buf_layout, 12},
        {"attention_value",  attn_value_pipe,       desc_3buf_layout, 24},
        {"argmax",           argmax_pipe,           desc_2buf_layout,  4},
    };

    for (auto& s : required) {
        std::string path = std::string(spv_dir) + "/" + s.name + ".spv";
        std::vector<uint32_t> code;
        if (!read_spv(path.c_str(), code)) return false;
        if (!create_pipeline(code.data(), code.size() * 4, s.layout, s.push_size, s.bundle)) {
            fprintf(stderr, "Failed to create pipeline: %s\n", s.name);
            return false;
        }
        printf("  Pipeline: %s\n", s.name);
    }

    // Optional WMMA pipelines (skip if SPV missing)
    ShaderEntry optional[] = {
        {"gemm_wmma",        gemm_wmma_pipe,        desc_3buf_layout, 12},
        {"attention_wmma",   attn_wmma_pipe,        desc_3buf_layout, 32},
    };
    for (auto& s : optional) {
        std::string path = std::string(spv_dir) + "/" + s.name + ".spv";
        std::vector<uint32_t> code;
        if (read_spv(path.c_str(), code))
            create_pipeline(code.data(), code.size() * 4, s.layout, s.push_size, s.bundle);
        else
            printf("  (skipped optional: %s)\n", s.name);
    }

    // Wave32 variants: recompile all pipelines with subgroup_size=32
    if (dev.wave32_supported) {
        struct Wave32Entry { const char* name; PipelineBundle& bundle; VkDescriptorSetLayout layout; uint32_t push; };
        Wave32Entry wave32_pipes[] = {
            {"gemm_fp32",        gemm_fp32_pipe,        desc_3buf_layout, 12},
            {"gemm_q4_0",        gemm_q4_0_pipe,        desc_3buf_layout, 16},
            {"gemm_q8_0",        gemm_q8_0_pipe,        desc_3buf_layout, 16},
            {"convert_f32_f16",  convert_f32_f16_pipe,  desc_2buf_layout,  4},
            {"rms_norm",         rms_norm_pipe,         desc_3buf_layout,  8},
            {"rope",             rope_pipe,             desc_1buf_layout, 20},
            {"silu_mul",         silu_mul_pipe,         desc_3buf_layout,  4},
            {"add",              add_pipe,              desc_3buf_layout,  4},
            {"token_embedding",  token_embed_pipe,      desc_2buf_layout, 16},
            {"attention_scores", attn_scores_pipe,      desc_3buf_layout, 32},
            {"attention_softmax",attn_softmax_pipe,     desc_1buf_layout, 12},
            {"attention_value",  attn_value_pipe,       desc_3buf_layout, 24},
            {"argmax",           argmax_pipe,           desc_2buf_layout,  4},
        };
        for (auto& s : wave32_pipes) {
            std::string path = std::string(spv_dir) + "/" + s.name + ".spv";
            std::vector<uint32_t> code;
            if (read_spv(path.c_str(), code)) {
                VkPipeline old_pipe = s.bundle.pipeline;
                VkPipelineLayout old_layout = s.bundle.layout;
                if (create_pipeline(code.data(), code.size() * 4, s.layout, s.push, s.bundle, 32)) {
                    // Success — destroy old wave64 pipeline
                    if (old_pipe) vkDestroyPipeline(dev.device, old_pipe, nullptr);
                    if (old_layout) vkDestroyPipelineLayout(dev.device, old_layout, nullptr);
                    printf("  Pipeline: %s (wave32)\n", s.name);
                } else {
                    // Restore old pipeline
                    s.bundle.pipeline = old_pipe;
                    s.bundle.layout = old_layout;
                }
            }
        }
    }
    return true;
}

GpuBuffer VulkanBackend::create_fp16_copy(const GpuBuffer& src_f32, uint32_t count) {
    GpuBuffer dst = create_buffer(count * sizeof(uint16_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_2buf(desc_2buf_set[i], src_f32.buffer, dst.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, convert_f32_f16_pipe.pipeline);
    bind_descriptors(convert_f32_f16_pipe.layout, desc_2buf_set[i]);
    struct PC { uint32_t count; } pc = {count};
    push_constants(convert_f32_f16_pipe.layout, &pc, sizeof(pc));
    dispatch((count + 255) / 256);

    return dst;
}

void VulkanBackend::gpu_argmax(const GpuBuffer& logits, const GpuBuffer& result, uint32_t vocab_size) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_2buf(desc_2buf_set[i], logits.buffer, result.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, argmax_pipe.pipeline);
    bind_descriptors(argmax_pipe.layout, desc_2buf_set[i]);
    struct PC { uint32_t vocab_size; } pc = {vocab_size};
    push_constants(argmax_pipe.layout, &pc, sizeof(pc));
    dispatch(1);
}

void VulkanBackend::gemm_q4_0(const GpuBuffer& A, const GpuBuffer& B_q, const GpuBuffer& C,
                                uint32_t M, uint32_t N, uint32_t K, uint32_t num_blocks) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], A.buffer, B_q.buffer, C.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, gemm_q4_0_pipe.pipeline);
    bind_descriptors(gemm_q4_0_pipe.layout, desc_3buf_set[i]);
    struct PC { uint32_t M, N, K, num_blocks; } pc = {M, N, K, num_blocks};
    push_constants(gemm_q4_0_pipe.layout, &pc, sizeof(pc));
    dispatch(N, M);
}

void VulkanBackend::gemm_q8_0(const GpuBuffer& A, const GpuBuffer& B_q, const GpuBuffer& C,
                                uint32_t M, uint32_t N, uint32_t K, uint32_t num_blocks) {
    int i = desc_idx++ % DESC_POOL_COUNT;
    update_desc_3buf(desc_3buf_set[i], A.buffer, B_q.buffer, C.buffer);
    vkCmdBindPipeline(current_cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, gemm_q8_0_pipe.pipeline);
    bind_descriptors(gemm_q8_0_pipe.layout, desc_3buf_set[i]);
    struct PC { uint32_t M, N, K, num_blocks; } pc = {M, N, K, num_blocks};
    push_constants(gemm_q8_0_pipe.layout, &pc, sizeof(pc));
    dispatch(N, M);
}
