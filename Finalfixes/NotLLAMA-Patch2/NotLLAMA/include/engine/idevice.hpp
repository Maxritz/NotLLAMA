#pragma once
#include "types.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

namespace notllama {

class IDevice {
public:
    virtual ~IDevice() = default;

    virtual VkPhysicalDevice GetPhysicalDevice() = 0;
    virtual VkDevice GetLogicalDevice() = 0;
    virtual VendorProfile GetVendorProfile() = 0;
    virtual uint32_t GetMaxSubgroupSize() = 0;
    virtual bool SupportsCooperativeMatrix() = 0;

    virtual uint32_t GetComputeQueueFamily() = 0;
    virtual uint32_t GetTransferQueueFamily() = 0;
    virtual VkQueue GetComputeQueue(uint32_t index) = 0;
    virtual VkQueue GetTransferQueue() = 0;

    virtual bool SupportsBufferDeviceAddress() = 0;
    virtual VkDeviceAddress GetBufferAddress(VkBuffer buffer) = 0;

    virtual size_t GetMaxSharedMemorySize() = 0;
    virtual void RecoverFromDeviceLoss() = 0;
};

} // namespace notllama
