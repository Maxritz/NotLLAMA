#pragma once
#include "engine/idevice.hpp"
#include "rdna4_vulkan.hpp"

namespace notllama {

class VulkanDevice : public IDevice {
public:
    explicit VulkanDevice(rdna4::VulkanContext* ctx);
    ~VulkanDevice() override = default;

    VkPhysicalDevice GetPhysicalDevice() override;
    VkDevice GetLogicalDevice() override;
    VendorProfile GetVendorProfile() override;
    uint32_t GetMaxSubgroupSize() override;
    bool SupportsCooperativeMatrix() override;

    uint32_t GetComputeQueueFamily() override;
    uint32_t GetTransferQueueFamily() override;
    VkQueue GetComputeQueue(uint32_t index) override;
    VkQueue GetTransferQueue() override;

    bool SupportsBufferDeviceAddress() override;
    VkDeviceAddress GetBufferAddress(VkBuffer buffer) override;

    size_t GetMaxSharedMemorySize() override;
    void RecoverFromDeviceLoss() override;

private:
    rdna4::VulkanContext* ctx_;
};

} // namespace notllama
