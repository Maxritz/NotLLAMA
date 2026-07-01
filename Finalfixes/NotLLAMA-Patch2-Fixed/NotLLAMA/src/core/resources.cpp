#include "core/resources.h"
#include <fstream>
#include <cstring>

namespace notllama {

// --- Buffer ---

void Buffer::destroy(VkDevice device) {
    if (mapped) { vkUnmapMemory(device, memory); mapped = nullptr; }
    if (handle) vkDestroyBuffer(device, handle, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
    handle = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    address = 0;
    size = 0;
}

Buffer createBuffer(Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memProps, bool deviceAddress) {
    Buffer buf;
    buf.size = size;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(ctx.device(), &bufInfo, nullptr, &buf.handle));

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(ctx.device(), buf.handle, &memReqs);

    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    if (deviceAddress) flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = ctx.findMemoryType(memReqs.memoryTypeBits, memProps);
    allocInfo.pNext = &flagsInfo;

    VK_CHECK(vkAllocateMemory(ctx.device(), &allocInfo, nullptr, &buf.memory));
    VK_CHECK(vkBindBufferMemory(ctx.device(), buf.handle, buf.memory, 0));

    if (deviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buf.handle;
        buf.address = vkGetBufferDeviceAddress(ctx.device(), &addrInfo);
    }

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(ctx.device(), buf.memory, 0, size, 0, &buf.mapped));
    }

    return buf;
}

Buffer createBufferFromHost(Context& ctx, VkDeviceSize size, const void* data,
                            VkBufferUsageFlags usage) {
    // Staging
    Buffer staging = createBuffer(ctx, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(staging.mapped, data, size);

    // Device-local
    Buffer device = createBuffer(ctx, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);

    // Upload
    CommandBuffer cmd = beginSingleUse(ctx, 0);
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd.handle, staging.handle, device.handle, 1, &copyRegion);
    cmd.end();
    submitAndWait(ctx, cmd, 0);

    staging.destroy(ctx.device());
    return device;
}

void uploadBuffer(Context& ctx, Buffer& dst, const void* data, VkDeviceSize size) {
    Buffer staging = createBuffer(ctx, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(staging.mapped, data, size);

    CommandBuffer cmd = beginSingleUse(ctx, 0);
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd.handle, staging.handle, dst.handle, 1, &copyRegion);
    cmd.end();
    submitAndWait(ctx, cmd, 0);

    staging.destroy(ctx.device());
}

// --- Image ---

void Image::destroy(VkDevice device) {
    if (view) vkDestroyImageView(device, view, nullptr);
    if (handle) vkDestroyImage(device, handle, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
    handle = VK_NULL_HANDLE;
    view = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

Image createImage(Context& ctx, uint32_t w, uint32_t h, uint32_t d,
                  VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspects) {
    Image img;
    img.width = w; img.height = h; img.depth = d;
    img.format = format;

    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = (d > 1) ? VK_IMAGE_TYPE_3D : (h > 1) ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
    imgInfo.extent = {w, h, d};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = format;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = usage;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VK_CHECK(vkCreateImage(ctx.device(), &imgInfo, nullptr, &img.handle));

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(ctx.device(), img.handle, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = ctx.findImageMemoryType(memReqs.memoryTypeBits);

    VK_CHECK(vkAllocateMemory(ctx.device(), &allocInfo, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(ctx.device(), img.handle, img.memory, 0));

    // Image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = img.handle;
    viewInfo.viewType = (d > 1) ? VK_IMAGE_VIEW_TYPE_3D :
                        (h > 1) ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_1D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspects;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(ctx.device(), &viewInfo, nullptr, &img.view));

    return img;
}

// --- Pipeline ---

void Pipeline::destroy(VkDevice device) {
    if (handle) vkDestroyPipeline(device, handle, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (descLayout) vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    handle = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    descLayout = VK_NULL_HANDLE;
}

PipelineBuilder& PipelineBuilder::addBufferBinding(uint32_t binding, VkDescriptorType type, uint32_t count) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = count;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(b);
    return *this;
}

PipelineBuilder& PipelineBuilder::addImageBinding(uint32_t binding, VkDescriptorType type, uint32_t count) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = count;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(b);
    return *this;
}

PipelineBuilder& PipelineBuilder::addPushConstant(VkShaderStageFlags stage, uint32_t offset, uint32_t size) {
    VkPushConstantRange pc{};
    pc.stageFlags = stage;
    pc.offset = offset;
    pc.size = size;
    pushConstants.push_back(pc);
    return *this;
}

PipelineBuilder& PipelineBuilder::setLocalSize(uint32_t x, uint32_t y, uint32_t z) {
    localX = x; localY = y; localZ = z;
    return *this;
}

Pipeline PipelineBuilder::build() {
    Pipeline pipeline;

    // Descriptor set layout
    VkDescriptorSetLayoutCreateInfo descLayoutInfo{};
    descLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    descLayoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device(), &descLayoutInfo, nullptr, &pipeline.descLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &pipeline.descLayout;
    layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstants.size());
    layoutInfo.pPushConstantRanges = pushConstants.data();
    VK_CHECK(vkCreatePipelineLayout(ctx->device(), &layoutInfo, nullptr, &pipeline.layout));

    // Compute pipeline
    VkComputePipelineCreateInfo compInfo{};
    compInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.module = shaderModule;
    compInfo.stage.pName = "main";
    compInfo.layout = pipeline.layout;

    VK_CHECK(vkCreateComputePipelines(ctx->device(), VK_NULL_HANDLE, 1, &compInfo, nullptr, &pipeline.handle));

    return pipeline;
}

VkShaderModule createShaderModule(Context& ctx, const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(ctx.device(), &createInfo, nullptr, &module));
    return module;
}

std::vector<uint32_t> loadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open SPIR-V: " + path);

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

// --- Descriptor pool ---

void DescriptorPool::destroy(VkDevice device) {
    if (handle) vkDestroyDescriptorPool(device, handle, nullptr);
    handle = VK_NULL_HANDLE;
    set = VK_NULL_HANDLE;
}

DescriptorPool createDescriptorPool(Context& ctx, const Pipeline& pipeline) {
    DescriptorPool desc;

    // Count binding types
    uint32_t bufferCount = 0, imageCount = 0;
    // We'll create a pool big enough for any reasonable layout
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 4;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;

    VK_CHECK(vkCreateDescriptorPool(ctx.device(), &poolInfo, nullptr, &desc.handle));

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = desc.handle;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &pipeline.descLayout;

    VK_CHECK(vkAllocateDescriptorSets(ctx.device(), &allocInfo, &desc.set));
    return desc;
}

void updateDescriptorBuffer(DescriptorPool& desc, Context& ctx,
                            uint32_t binding, const Buffer& buf, VkDescriptorType type) {
    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = buf.handle;
    bufInfo.offset = 0;
    bufInfo.range = buf.size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = desc.set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = type;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(ctx.device(), 1, &write, 0, nullptr);
}

void updateDescriptorImage(DescriptorPool& desc, Context& ctx,
                           uint32_t binding, const Image& img, VkDescriptorType type) {
    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgInfo.imageView = img.view;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = desc.set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = type;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(ctx.device(), 1, &write, 0, nullptr);
}

// --- CommandBuffer ---

void CommandBuffer::begin(VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = flags;
    VK_CHECK(vkBeginCommandBuffer(handle, &info));
    recording = true;
}

void CommandBuffer::end() {
    VK_CHECK(vkEndCommandBuffer(handle));
    recording = false;
}

void CommandBuffer::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(handle, x, y, z);
}

void CommandBuffer::barrier(VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(handle, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void CommandBuffer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(handle, src, dst, 1, &region);
}

CommandBuffer beginSingleUse(Context& ctx, uint32_t aceIndex) {
    CommandBuffer cmd;
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.acePool(aceIndex);
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device(), &allocInfo, &cmd.handle));
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    return cmd;
}

void submitAndWait(Context& ctx, CommandBuffer& cmd, uint32_t aceIndex) {
    cmd.end();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(ctx.device(), &fenceInfo, nullptr, &fence));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd.handle;

    VK_CHECK(vkQueueSubmit(ctx.aceQueue(aceIndex), 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkFreeCommandBuffers(ctx.device(), ctx.acePool(aceIndex), 1, &cmd.handle);
}

// --- Fence ---

void Fence::destroy(VkDevice device) {
    if (handle) vkDestroyFence(device, handle, nullptr);
    handle = VK_NULL_HANDLE;
}

Fence createFence(Context& ctx, bool signaled) {
    Fence f;
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(ctx.device(), &info, nullptr, &f.handle));
    return f;
}

} // namespace notllama
