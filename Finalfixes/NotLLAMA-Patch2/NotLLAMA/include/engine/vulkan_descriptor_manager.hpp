#pragma once
#include "engine/idescriptor_manager.hpp"
#include "engine/imemory_allocator.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

namespace notllama {

// Bindless descriptor manager + per-frame metadata ring buffer.
// Uses Vulkan 1.2 descriptor indexing (update-after-bind, partially bound,
// variable descriptor count) to expose a single storage-buffer descriptor array.
class VulkanDescriptorManager : public IDescriptorManager {
public:
    static constexpr uint32_t kMaxDescriptors = 4096;
    static constexpr uint32_t kMaxFrames = 64;

    VulkanDescriptorManager(VkDevice device, VkPhysicalDevice physical_device,
                            IMemoryAllocator* allocator);
    ~VulkanDescriptorManager() override;

    bool IsValid() const;

    uint32_t RegisterBuffer(VkBuffer buffer) override;
    VkDescriptorSet GetBindlessSet() override;
    VkDescriptorSetLayout GetBindlessSetLayout() override;

    void UpdateMetadataRingBuffer(uint32_t frame_index, const FrameMetadata& meta) override;
    GpuAllocation GetMetadataBuffer(uint32_t frame_index) override;

    uint32_t GetCurrentFrameIndex() const override;

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    IMemoryAllocator* allocator_;
    bool valid_ = false;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;

    GpuAllocation metadata_alloc_;
    uint32_t current_frame_index_ = 0;
    uint32_t next_buffer_index_ = 0;

    bool CheckDescriptorIndexingSupport();
    bool CreateDescriptorSetLayout();
    bool CreateDescriptorPoolAndSet();
};

} // namespace notllama
