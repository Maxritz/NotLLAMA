#include "rdna4_allocator.hpp"
#include <cstring>
#include <iostream>

namespace rdna4 {

RingAllocator::RingAllocator(VkDevice dev, VkPhysicalDevice pdev, size_t size)
    : device(dev), physicalDevice(pdev), capacity(size), offset(0) {

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateBuffer(device, &bufInfo, nullptr, &buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    // Prefer host-visible + device-local (resizable BAR) if available
    // Otherwise fall back to host-visible + coherent for easy readback
    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    // Fallback: host-visible + coherent
    if (memTypeIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }

    if (memTypeIndex == 0xFFFFFFFF) {
        std::cerr << "Failed to find suitable memory type for ring allocator\n";
        return;
    }

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = &flagsInfo;

    vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(device, buffer, memory, 0);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    baseAddress = vkGetBufferDeviceAddress(device, &addrInfo);

    VkResult r = vkMapMemory(device, memory, 0, size, 0, reinterpret_cast<void**>(&mappedPtr));
    if (r != VK_SUCCESS) {
        std::cerr << "Warning: Failed to map ring allocator memory. Readback will not work.\n";
        mappedPtr = nullptr;
    }

    std::cout << "Ring allocator: " << (size / 1024 / 1024) << " MB @ 0x" 
              << std::hex << baseAddress << std::dec;
    if (memProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        std::cout << " (device-local + host-visible)\n";
    } else {
        std::cout << " (host-visible only — slower)\n";
    }
}

RingAllocator::~RingAllocator() {
    if (mappedPtr) vkUnmapMemory(device, memory);
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
}

uint64_t RingAllocator::alloc(size_t size, size_t alignment) {
    size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
    if (aligned + size > capacity) {
        std::cerr << "Ring allocator OOM! Need " << size << ", have " 
                  << (capacity - aligned) << "\n";
        return 0;
    }
    uint64_t addr = baseAddress + aligned;
    offset = aligned + size;
    return addr;
}

void RingAllocator::reset() {
    offset = 0;
}

void RingAllocator::upload(uint64_t addr, const void* data, size_t size) {
    if (!mappedPtr) return;
    size_t localOffset = addr - baseAddress;
    std::memcpy(mappedPtr + localOffset, data, size);

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = localOffset;
    range.size = size;
    vkFlushMappedMemoryRanges(device, 1, &range);
}

} // namespace rdna4
