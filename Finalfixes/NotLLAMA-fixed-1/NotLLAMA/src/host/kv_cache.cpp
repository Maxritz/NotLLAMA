#include "rdna4_kv_cache.hpp"
#include <iostream>
#include <cstring>

namespace rdna4 {

KVCacheManager::KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                               uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd)
    : device(dev), physicalDevice(pdev),
      maxSeqLen(maxSeq), nLayers(nL), nKvHeads(nKV), headDim(hd) {
    layers.resize(nLayers);
}

bool KVCacheManager::allocate() {
    size_t layerBytes = static_cast<size_t>(maxSeqLen) * nKvHeads * headDim * sizeof(float);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    VkPhysicalDeviceProperties physProps;
    vkGetPhysicalDeviceProperties(physicalDevice, &physProps);
    VkDeviceSize atomSize = physProps.limits.nonCoherentAtomSize;
    VkDeviceSize atomMask = atomSize - 1;

    for (uint32_t i = 0; i < nLayers; ++i) {
        auto& layer = layers[i];

        for (int kv = 0; kv < 2; ++kv) {
            VkBuffer& buf = (kv == 0) ? layer.kBuffer : layer.vBuffer;
            VkDeviceMemory& mem = (kv == 0) ? layer.kMemory : layer.vMemory;
            VkDeviceAddress& addr = (kv == 0) ? layer.kAddress : layer.vAddress;

            VkBufferCreateInfo bufInfo = {};
            bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufInfo.size = layerBytes;
            bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &buf);
            if (r != VK_SUCCESS) {
                std::cerr << "KV cache: vkCreateBuffer failed: " << r << "\n";
                return false;
            }

            VkMemoryRequirements memReq;
            vkGetBufferMemoryRequirements(device, buf, &memReq);

            uint32_t memTypeIndex = UINT32_MAX;
            for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
                if ((memReq.memoryTypeBits & (1 << j)) &&
                    (memProps.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    memTypeIndex = j;
                    break;
                }
            }
            if (memTypeIndex == UINT32_MAX) {
                for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
                    if (memReq.memoryTypeBits & (1 << j)) {
                        memTypeIndex = j;
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

            r = vkAllocateMemory(device, &allocInfo, nullptr, &mem);
            if (r != VK_SUCCESS) {
                std::cerr << "KV cache: vkAllocateMemory failed: " << r << "\n";
                return false;
            }

            vkBindBufferMemory(device, buf, mem, 0);

            VkBufferDeviceAddressInfo addrInfo = {};
            addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addrInfo.buffer = buf;
            addr = vkGetBufferDeviceAddress(device, &addrInfo);
        }

        layer.currentSeqLen = 0;
    }

    fprintf(stderr, "[kv_cache] allocated: %u layers, %u max seq, %u KV heads, %u head dim\n",
            nLayers, maxSeqLen, nKvHeads, headDim);
    return true;
}

void KVCacheManager::free() {
    for (auto& layer : layers) {
        if (layer.kBuffer) vkDestroyBuffer(device, layer.kBuffer, nullptr);
        if (layer.vBuffer) vkDestroyBuffer(device, layer.vBuffer, nullptr);
        if (layer.kMemory) vkFreeMemory(device, layer.kMemory, nullptr);
        if (layer.vMemory) vkFreeMemory(device, layer.vMemory, nullptr);
        layer.kBuffer = VK_NULL_HANDLE;
        layer.vBuffer = VK_NULL_HANDLE;
        layer.kMemory = VK_NULL_HANDLE;
        layer.vMemory = VK_NULL_HANDLE;
    }
}

void KVCacheManager::append(uint32_t layer, const void* kData, const void* vData) {
    if (layer >= layers.size()) return;
    auto& l = layers[layer];

    size_t rowBytes = static_cast<size_t>(nKvHeads) * headDim * sizeof(float);
    size_t kOffset = static_cast<size_t>(l.currentSeqLen) * rowBytes;

    // Try direct map first (works if memory is host-visible)
    bool kDone = false, vDone = false;
    {
        void* mapped = nullptr;
        VkResult r = vkMapMemory(device, l.kMemory, kOffset, rowBytes, 0, &mapped);
        if (r == VK_SUCCESS && mapped) {
            std::memcpy(mapped, kData, rowBytes);
            vkFlushMappedMemoryRanges(device, 0, nullptr);
            vkUnmapMemory(device, l.kMemory);
            kDone = true;
        }
    }
    {
        void* mapped = nullptr;
        VkResult r = vkMapMemory(device, l.vMemory, kOffset, rowBytes, 0, &mapped);
        if (r == VK_SUCCESS && mapped) {
            std::memcpy(mapped, vData, rowBytes);
            vkFlushMappedMemoryRanges(device, 0, nullptr);
            vkUnmapMemory(device, l.vMemory);
            vDone = true;
        }
    }

    // If direct map failed, we need a staging buffer approach
    // Note: this requires a command pool and queue which we don't have in this context.
    // The shader-based KV_CACHE_WRITE path should be used instead for device-local memory.
    if (!kDone || !vDone) {
        fprintf(stderr, "[kv_cache] WARNING: Direct memory map failed for layer %u. "
                "Use shader-based KV cache write path instead.\n", layer);
    }

    l.currentSeqLen++;
}

} // namespace rdna4
