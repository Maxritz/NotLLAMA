#pragma once
#include "rdna4.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace rdna4 {

class RingAllocator {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress baseAddress = 0;
    uint8_t* mappedPtr = nullptr;

    size_t capacity;
    size_t offset;

    RingAllocator(VkDevice dev, VkPhysicalDevice pdev, size_t size);
    ~RingAllocator();

    uint64_t alloc(size_t size, size_t alignment = 256);
    void reset();
    void upload(uint64_t addr, const void* data, size_t size);
};

} // namespace rdna4
