#pragma once
#include "core/context.h"
#include <vector>
#include <span>

namespace notllama {

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;

    void destroy(VkDevice device);
    bool valid() const { return handle != VK_NULL_HANDLE; }
};

Buffer createBuffer(Context& ctx, VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memProps,
                    bool deviceAddress = false);

Buffer createBufferFromHost(Context& ctx, VkDeviceSize size,
                            const void* data,
                            VkBufferUsageFlags usage);

void uploadBuffer(Context& ctx, Buffer& dst, const void* data, VkDeviceSize size);

struct Image {
    VkImage handle = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, depth = 0;

    void destroy(VkDevice device);
    bool valid() const { return handle != VK_NULL_HANDLE; }
};

Image createImage(Context& ctx, uint32_t w, uint32_t h, uint32_t d,
                  VkFormat format, VkImageUsageFlags usage,
                  VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT);

struct Pipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;

    void destroy(VkDevice device);
    bool valid() const { return handle != VK_NULL_HANDLE; }
};

struct PipelineBuilder {
    Context* ctx = nullptr;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkPushConstantRange> pushConstants;
    uint32_t localX = 1, localY = 1, localZ = 1;

    PipelineBuilder& setShader(VkShaderModule mod) { shaderModule = mod; return *this; }
    PipelineBuilder& addBufferBinding(uint32_t binding, VkDescriptorType type, uint32_t count = 1);
    PipelineBuilder& addImageBinding(uint32_t binding, VkDescriptorType type, uint32_t count = 1);
    PipelineBuilder& addPushConstant(VkShaderStageFlags stage, uint32_t offset, uint32_t size);
    PipelineBuilder& setLocalSize(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    Pipeline build();
};

VkShaderModule createShaderModule(Context& ctx, const std::vector<uint32_t>& spirv);

struct DescriptorPool {
    VkDescriptorPool handle = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;

    void destroy(VkDevice device);
};

DescriptorPool createDescriptorPool(Context& ctx, const Pipeline& pipeline);

void updateDescriptorBuffer(DescriptorPool& desc, Context& ctx,
                            uint32_t binding, const Buffer& buf,
                            VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

void updateDescriptorImage(DescriptorPool& desc, Context& ctx,
                           uint32_t binding, const Image& img,
                           VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

struct CommandBuffer {
    VkCommandBuffer handle = VK_NULL_HANDLE;
    VkCommandBufferBeginInfo beginInfo{};
    bool recording = false;

    void begin(VkCommandBufferUsageFlags flags = 0);
    void end();
    void dispatch(uint32_t x, uint32_t y, uint32_t z);
    void barrier(VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage);
    void bufferBarrier(VkBuffer buf, VkDeviceSize size,
                       uint32_t srcQueue, uint32_t dstQueue);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
};

CommandBuffer beginSingleUse(Context& ctx, uint32_t aceIndex = 0);
void submitAndWait(Context& ctx, CommandBuffer& cmd, uint32_t aceIndex = 0);

struct Fence {
    VkFence handle = VK_NULL_HANDLE;
    void destroy(VkDevice device);
};

Fence createFence(Context& ctx, bool signaled = false);

} // namespace notllama
