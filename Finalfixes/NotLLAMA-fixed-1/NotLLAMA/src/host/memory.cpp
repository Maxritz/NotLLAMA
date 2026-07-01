#include "rdna4.hpp"
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

namespace rdna4 {

class MemoryManager {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    std::vector<VkDeviceMemory> allocatedBuffers;
    std::vector<VkDeviceMemory> allocatedImages;

    MemoryManager(VkDevice dev, VkPhysicalDevice pdev) : device(dev), physicalDevice(pdev) {}

    VkBuffer allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceAddress* outAddr = nullptr) {
        VkBufferCreateInfo bufInfo = {};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = size;
        bufInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        vkCreateBuffer(device, &bufInfo, nullptr, &buffer);

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buffer, &memReq);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memTypeIndex = 0;
        bool found = false;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIndex = i;
                found = true;
                break;
            }
        }
        if (!found) {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((memReq.memoryTypeBits & (1 << i))) {
                    memTypeIndex = i;
                    break;
                }
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

        VkDeviceMemory memory;
        vkAllocateMemory(device, &allocInfo, nullptr, &memory);
        vkBindBufferMemory(device, buffer, memory, 0);
        allocatedBuffers.push_back(memory);

        if (outAddr) {
            VkBufferDeviceAddressInfo addrInfo = {};
            addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addrInfo.buffer = buffer;
            *outAddr = vkGetBufferDeviceAddress(device, &addrInfo);
        }

        return buffer;
    }

    VkImage allocateImage(uint32_t width, uint32_t height, VkFormat format) {
        VkImageCreateInfo imgInfo = {};
        imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = {width, height, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.format = format;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkImage image;
        VkResult r = vkCreateImage(device, &imgInfo, nullptr, &image);
        if (r != VK_SUCCESS) {
            std::cerr << "allocateImage: vkCreateImage failed: " << r << "\n";
            return VK_NULL_HANDLE;
        }

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memTypeIndex = 0;
        bool found = false;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIndex = i;
                found = true;
                break;
            }
        }
        if (!found) {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((memReq.memoryTypeBits & (1 << i))) {
                    memTypeIndex = i;
                    break;
                }
            }
        }

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memTypeIndex;

        VkDeviceMemory memory;
        vkAllocateMemory(device, &allocInfo, nullptr, &memory);
        vkBindImageMemory(device, image, memory, 0);
        allocatedImages.push_back(memory);

        return image;
    }

    void freeBuffer(VkBuffer buffer) {
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    }

    void freeImage(VkImage image) {
        if (image) vkDestroyImage(device, image, nullptr);
    }

    void freeAll() {
        for (auto mem : allocatedImages) vkFreeMemory(device, mem, nullptr);
        for (auto mem : allocatedBuffers) vkFreeMemory(device, mem, nullptr);
        allocatedImages.clear();
        allocatedBuffers.clear();
    }
};

} // namespace rdna4
