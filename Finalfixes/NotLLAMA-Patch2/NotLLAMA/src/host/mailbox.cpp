#include "rdna4_mailbox.hpp"
#include <cstring>
#include <iostream>

namespace rdna4 {

bool GpuMailbox::init(VkDevice dev, VkPhysicalDevice pdev) {
    device = dev;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = 8; // 4 bytes token + 4 bytes ready flag
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &buffer);
    if (r != VK_SUCCESS) return false;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pdev, &memProps);

    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = &flagsInfo;

    r = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (r != VK_SUCCESS) return false;

    vkBindBufferMemory(device, buffer, memory, 0);
    vkMapMemory(device, memory, 0, 8, 0, &mapped);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    gpuAddress = vkGetBufferDeviceAddress(device, &addrInfo);

    std::cout << "GPU mailbox initialized @ 0x" << std::hex << gpuAddress << std::dec << "\n";
    return true;
}

void GpuMailbox::destroy() {
    if (mapped) { vkUnmapMemory(device, memory); mapped = nullptr; }
    if (buffer) { vkDestroyBuffer(device, buffer, nullptr); buffer = VK_NULL_HANDLE; }
    if (memory) { vkFreeMemory(device, memory, nullptr); memory = VK_NULL_HANDLE; }
}

uint32_t GpuMailbox::readToken() {
    if (!mapped) return 0;
    uint32_t token = 0;
    std::memcpy(&token, static_cast<uint8_t*>(mapped) + 0, sizeof(uint32_t));
    return token;
}

void GpuMailbox::acknowledge() {
    if (!mapped) return;
    uint32_t zero = 0;
    std::memcpy(static_cast<uint8_t*>(mapped) + 4, &zero, sizeof(uint32_t));
}

} // namespace rdna4
