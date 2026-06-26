#include "rdna4_kv_cache.hpp"
#include <iostream>

namespace rdna4 {

KVCacheManager::KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                               uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd)
    : device(dev), physicalDevice(pdev),
      maxSeqLen(maxSeq), nLayers(nL), nKvHeads(nKV), headDim(hd) {
    layers.resize(nLayers);
    for (auto& layer : layers) {
        layer.currentSeqLen = 0;
        layer.kImage = VK_NULL_HANDLE;
        layer.vImage = VK_NULL_HANDLE;
    }
}

bool KVCacheManager::allocate() {
    // Total width: nKvHeads * headDim (e.g., 8 * 128 = 1024)
    // Height: maxSeqLen (e.g., 4096)
    // Format: R16G16B16A16_SFLOAT for 4x f16 per pixel, or R16_SFLOAT
    uint32_t width = nKvHeads * headDim / 4;  // 4 channels per pixel
    uint32_t height = maxSeqLen;

    VkImageCreateInfo imgInfo = {};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent = {width, height, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    for (uint32_t i = 0; i < nLayers; ++i) {
        // K image
        VkResult r = vkCreateImage(device, &imgInfo, nullptr, &layers[i].kImage);
        if (r != VK_SUCCESS) return false;

        // V image
        r = vkCreateImage(device, &imgInfo, nullptr, &layers[i].vImage);
        if (r != VK_SUCCESS) return false;

        // Allocate memory
        VkMemoryRequirements memReqK, memReqV;
        vkGetImageMemoryRequirements(device, layers[i].kImage, &memReqK);
        vkGetImageMemoryRequirements(device, layers[i].vImage, &memReqV);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memTypeIndex = 0;
        for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
            if ((memReqK.memoryTypeBits & (1 << j)) &&
                (memProps.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIndex = j;
                break;
            }
        }

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqK.size;
        allocInfo.memoryTypeIndex = memTypeIndex;

        vkAllocateMemory(device, &allocInfo, nullptr, &layers[i].kMemory);
        vkBindImageMemory(device, layers[i].kImage, layers[i].kMemory, 0);

        allocInfo.allocationSize = memReqV.size;
        vkAllocateMemory(device, &allocInfo, nullptr, &layers[i].vMemory);
        vkBindImageMemory(device, layers[i].vImage, layers[i].vMemory, 0);

        // Create views
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = layers[i].kImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(device, &viewInfo, nullptr, &layers[i].kView);

        viewInfo.image = layers[i].vImage;
        vkCreateImageView(device, &viewInfo, nullptr, &layers[i].vView);
    }

    std::cout << "KV cache allocated: " << nLayers << " layers, "
              << maxSeqLen << " max seq, " << nKvHeads << " KV heads, "
              << headDim << " head dim\n";
    return true;
}

void KVCacheManager::free() {
    for (auto& layer : layers) {
        if (layer.kView) vkDestroyImageView(device, layer.kView, nullptr);
        if (layer.vView) vkDestroyImageView(device, layer.vView, nullptr);
        if (layer.kImage) vkDestroyImage(device, layer.kImage, nullptr);
        if (layer.vImage) vkDestroyImage(device, layer.vImage, nullptr);
        if (layer.kMemory) vkFreeMemory(device, layer.kMemory, nullptr);
        if (layer.vMemory) vkFreeMemory(device, layer.vMemory, nullptr);
    }
}

void KVCacheManager::append(uint32_t layer, const void* kData, const void* vData) {
    // TODO: Use transfer queue to copy new K/V vectors into image at (currentSeqLen, 0)
    // For now, just increment
    layers[layer].currentSeqLen++;
}

} // namespace rdna4
