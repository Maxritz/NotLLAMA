#pragma once
#include "rdna4.hpp"
#include <vulkan/vulkan.h>
#include <vector>

namespace rdna4 {

// KV cache stored as DCC-compressed images for bandwidth efficiency.
// Each layer has K and V caches: [maxSeqLen, nKvHeads, headDim]
// Stored as 2D images where rows = seq positions, cols = headDim * nKvHeads

struct KVCacheLayer {
    VkImage kImage;
    VkImage vImage;
    VkDeviceMemory kMemory;
    VkDeviceMemory vMemory;
    VkImageView kView;
    VkImageView vView;
    uint32_t currentSeqLen;
};

class KVCacheManager {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    uint32_t maxSeqLen;
    uint32_t nLayers;
    uint32_t nKvHeads;
    uint32_t headDim;

    std::vector<KVCacheLayer> layers;

    KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                   uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd);

    bool allocate();
    void free();

    // Append new K/V vectors for the current token at position currentSeqLen
    void append(uint32_t layer, const void* kData, const void* vData);

    // Get image views for binding to compute shaders
    VkImageView getKView(uint32_t layer) const { return layers[layer].kView; }
    VkImageView getVView(uint32_t layer) const { return layers[layer].vView; }

    uint32_t getSeqLen(uint32_t layer) const { return layers[layer].currentSeqLen; }
    void incrementSeqLen(uint32_t layer) { layers[layer].currentSeqLen++; }
};

} // namespace rdna4
