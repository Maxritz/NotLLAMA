#pragma once
#include "types.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

namespace notllama {

struct FrameMetadata;

class IDescriptorManager {
public:
    virtual ~IDescriptorManager() = default;

    virtual uint32_t RegisterBuffer(VkBuffer buffer) = 0;
    virtual VkDescriptorSet GetBindlessSet() = 0;
    virtual VkDescriptorSetLayout GetBindlessSetLayout() = 0;

    virtual void UpdateMetadataRingBuffer(uint32_t frame_index, const FrameMetadata& meta) = 0;
    virtual GpuAllocation GetMetadataBuffer(uint32_t frame_index) = 0;

    virtual uint32_t GetCurrentFrameIndex() const = 0;
};

} // namespace notllama
