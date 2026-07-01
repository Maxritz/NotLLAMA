// VulkanDescriptorManager — bindless SSBO descriptor table + per-frame metadata ring.
// AMD workaround: update-after-bind flags removed to avoid crash with wave32
// forced via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.
// All descriptors are pre-written before binding (no update-after-bind).

#include "engine/vulkan_descriptor_manager.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace notllama {

VulkanDescriptorManager::VulkanDescriptorManager(VkDevice device,
                                                 VkPhysicalDevice physical_device,
                                                 IMemoryAllocator* allocator)
    : device_(device), physical_device_(physical_device), allocator_(allocator) {
    if (device_ == VK_NULL_HANDLE || physical_device_ == VK_NULL_HANDLE || !allocator_) {
        std::cerr << "VulkanDescriptorManager: invalid constructor arguments\n";
        return;
    }

    if (!CheckDescriptorIndexingSupport()) return;
    if (!CreateDescriptorSetLayout()) return;
    if (!CreateDescriptorPoolAndSet()) return;

    const size_t atom = allocator_->GetNonCoherentAtomSize();
    const size_t meta_stride = sizeof(FrameMetadata);
    const size_t alignment = std::max(atom, meta_stride);
    const size_t ring_size = kMaxFrames * alignment;

    metadata_alloc_ = allocator_->Allocate(MemoryType::TRANSIENT, ring_size, alignment);
    if (metadata_alloc_.buffer == VK_NULL_HANDLE) {
        std::cerr << "VulkanDescriptorManager: metadata ring allocation failed\n";
        return;
    }

    valid_ = true;
}

VulkanDescriptorManager::~VulkanDescriptorManager() {
    if (metadata_alloc_.buffer != VK_NULL_HANDLE && allocator_) {
        allocator_->Free(MemoryType::TRANSIENT, metadata_alloc_);
    }
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }
}

bool VulkanDescriptorManager::IsValid() const { return valid_; }

bool VulkanDescriptorManager::CheckDescriptorIndexingSupport() {
    VkPhysicalDeviceVulkan12Features feat12{};
    feat12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &feat12;

    vkGetPhysicalDeviceFeatures2(physical_device_, &features2);

    const bool ok = feat12.descriptorIndexing &&
                    feat12.descriptorBindingStorageBufferUpdateAfterBind &&
                    feat12.descriptorBindingPartiallyBound &&
                    feat12.runtimeDescriptorArray;
    if (!ok) {
        std::cerr << "VulkanDescriptorManager: required Vulkan 1.2 descriptor indexing features missing\n";
    }
    return ok;
}

bool VulkanDescriptorManager::CreateDescriptorSetLayout() {
    // AMD workaround: No update-after-bind flags. On AMD + wave32 forced via
    // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, update-after-bind
    // descriptor sets cause driver crashes. Pre-write all descriptors instead.
    VkDescriptorBindingFlags binding_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    // Note: VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT intentionally omitted.

    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
    binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    binding_flags_info.bindingCount = 1;
    binding_flags_info.pBindingFlags = &binding_flags;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = kMaxDescriptors;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &binding_flags_info;
    layout_info.flags = 0;  // No update-after-bind (AMD workaround)
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;

    VkResult r = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout_);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateDescriptorSetLayout failed: " << r << "\n";
        return false;
    }
    return true;
}

bool VulkanDescriptorManager::CreateDescriptorPoolAndSet() {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = kMaxDescriptors;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;  // No update-after-bind (AMD workaround)
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkResult r = vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_);
    if (r != VK_SUCCESS) {
        std::cerr << "vkCreateDescriptorPool failed: " << r << "\n";
        return false;
    }

    uint32_t variable_count = kMaxDescriptors;
    VkDescriptorSetVariableDescriptorCountAllocateInfo variable_count_info{};
    variable_count_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    variable_count_info.descriptorSetCount = 1;
    variable_count_info.pDescriptorCounts = &variable_count;

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pNext = &variable_count_info;
    alloc_info.descriptorPool = pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout_;

    r = vkAllocateDescriptorSets(device_, &alloc_info, &set_);
    if (r != VK_SUCCESS) {
        std::cerr << "vkAllocateDescriptorSets failed: " << r << "\n";
        return false;
    }
    return true;
}

uint32_t VulkanDescriptorManager::RegisterBuffer(VkBuffer buffer) {
    if (!valid_ || buffer == VK_NULL_HANDLE) return 0xFFFFFFFF;
    if (next_buffer_index_ >= kMaxDescriptors) {
        std::cerr << "VulkanDescriptorManager: descriptor table full (" << kMaxDescriptors << ")\n";
        return 0xFFFFFFFF;
    }

    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = next_buffer_index_;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    return next_buffer_index_++;
}

VkDescriptorSet VulkanDescriptorManager::GetBindlessSet() {
    return set_;
}

VkDescriptorSetLayout VulkanDescriptorManager::GetBindlessSetLayout() {
    return layout_;
}

void VulkanDescriptorManager::UpdateMetadataRingBuffer(uint32_t frame_index, const FrameMetadata& meta) {
    if (!valid_ || frame_index >= kMaxFrames || metadata_alloc_.buffer == VK_NULL_HANDLE) {
        return;
    }

    const size_t atom = allocator_->GetNonCoherentAtomSize();
    const size_t meta_stride = sizeof(FrameMetadata);
    const size_t alignment = std::max(atom, meta_stride);
    const size_t offset = frame_index * alignment;

    if (offset + meta_stride > metadata_alloc_.size) return;

    void* mapped = allocator_->Map(metadata_alloc_);
    if (!mapped) return;

    std::memcpy(static_cast<uint8_t*>(mapped) + offset, &meta, meta_stride);
    allocator_->Unmap(metadata_alloc_);
    current_frame_index_ = frame_index;
}

GpuAllocation VulkanDescriptorManager::GetMetadataBuffer(uint32_t frame_index) {
    if (!valid_ || frame_index >= kMaxFrames || metadata_alloc_.buffer == VK_NULL_HANDLE) {
        return {};
    }

    const size_t atom = allocator_->GetNonCoherentAtomSize();
    const size_t meta_stride = sizeof(FrameMetadata);
    const size_t alignment = std::max(atom, meta_stride);
    const size_t offset = frame_index * alignment;

    if (offset + meta_stride > metadata_alloc_.size) return {};

    GpuAllocation sub = metadata_alloc_;
    sub.offset += offset;
    sub.size = meta_stride;
    if (sub.device_address) {
        sub.device_address += offset;
    }
    return sub;
}

uint32_t VulkanDescriptorManager::GetCurrentFrameIndex() const {
    return current_frame_index_;
}

} // namespace notllama
