#include "engine/vulkan_device.hpp"
#include <cstring>
#include <vector>

namespace notllama {

VulkanDevice::VulkanDevice(rdna4::VulkanContext* ctx) : ctx_(ctx) {}

VkPhysicalDevice VulkanDevice::GetPhysicalDevice() {
    return ctx_ ? ctx_->physicalDevice : VK_NULL_HANDLE;
}

VkDevice VulkanDevice::GetLogicalDevice() {
    return ctx_ ? ctx_->device : VK_NULL_HANDLE;
}

VendorProfile VulkanDevice::GetVendorProfile() {
    VendorProfile p{};
    if (!ctx_) return p;

    p.vendor = ctx_->isAmd() ? VendorID::AMD :
               ctx_->isNvidia() ? VendorID::NVIDIA : VendorID::UNKNOWN;
    p.subgroup_size = ctx_->subgroupSize;
    p.wave32 = ctx_->isWave32();
    p.wave64 = ctx_->isWave64();
    p.cooperative_matrix = SupportsCooperativeMatrix();
    return p;
}

uint32_t VulkanDevice::GetMaxSubgroupSize() {
    return ctx_ ? ctx_->subgroupSize : 32;
}

bool VulkanDevice::SupportsCooperativeMatrix() {
    if (!ctx_ || !ctx_->physicalDevice) return false;

    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(ctx_->physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> ext(count);
    vkEnumerateDeviceExtensionProperties(ctx_->physicalDevice, nullptr, &count, ext.data());

    for (const auto& e : ext) {
        if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            return true;
        }
    }
    return false;
}

uint32_t VulkanDevice::GetComputeQueueFamily() {
    return ctx_ ? ctx_->queueFamilyIndex : 0;
}

uint32_t VulkanDevice::GetTransferQueueFamily() {
    // TODO: probe dedicated transfer queue family
    return ctx_ ? ctx_->queueFamilyIndex : 0;
}

VkQueue VulkanDevice::GetComputeQueue(uint32_t index) {
    if (!ctx_ || index >= 4) return VK_NULL_HANDLE;
    return ctx_->queues[index];
}

VkQueue VulkanDevice::GetTransferQueue() {
    // TODO: use dedicated transfer queue if available
    return ctx_ ? ctx_->queues[0] : VK_NULL_HANDLE;
}

bool VulkanDevice::SupportsBufferDeviceAddress() {
    // Existing codebase relies on BDA; assume enabled.
    return ctx_ != nullptr;
}

VkDeviceAddress VulkanDevice::GetBufferAddress(VkBuffer buffer) {
    if (!ctx_ || !ctx_->device || buffer == VK_NULL_HANDLE) return 0;
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(ctx_->device, &info);
}

size_t VulkanDevice::GetMaxSharedMemorySize() {
    if (!ctx_ || !ctx_->physicalDevice) return 32768;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_->physicalDevice, &props);
    return props.limits.maxComputeSharedMemorySize;
}

void VulkanDevice::RecoverFromDeviceLoss() {
    // TODO: implement 5-step rebuild
}

} // namespace notllama
