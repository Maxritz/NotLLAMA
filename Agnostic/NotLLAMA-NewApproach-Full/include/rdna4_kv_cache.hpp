#pragma once
#include "rdna4.hpp"
#include <vulkan/vulkan.h>
#include <vector>

namespace rdna4 {

// KV cache as flat buffers with buffer device address for shader access.
// Each layer has K and V buffers: totalBytes = maxSeqLen * nKvHeads * headDim * sizeof(float16)
// Buffers use SHADER_DEVICE_ADDRESS_BIT so shaders access them via buffer_reference.

struct KVCacheBuffer {
    VkBuffer kBuffer = VK_NULL_HANDLE;
    VkBuffer vBuffer = VK_NULL_HANDLE;
    VkDeviceMemory kMemory = VK_NULL_HANDLE;
    VkDeviceMemory vMemory = VK_NULL_HANDLE;
    VkDeviceAddress kAddress = 0;
    VkDeviceAddress vAddress = 0;
    uint32_t currentSeqLen = 0;
};

class KVCacheManager {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    uint32_t maxSeqLen;
    uint32_t nLayers;
    uint32_t nKvHeads;
    uint32_t headDim;

    std::vector<KVCacheBuffer> layers;

    KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                   uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd);

    bool allocate();
    void free();

    // Append new K/V vectors for the current token at position currentSeqLen
    void append(uint32_t layer, const void* kData, const void* vData);

    // Get buffer addresses for binding to compute shaders via push constants
    VkDeviceAddress getKBufferAddress(uint32_t layer) const { return layers[layer].kAddress; }
    VkDeviceAddress getVBufferAddress(uint32_t layer) const { return layers[layer].vAddress; }

    uint32_t getSeqLen(uint32_t layer) const { return layers[layer].currentSeqLen; }
    void incrementSeqLen(uint32_t layer) { layers[layer].currentSeqLen++; }
};

} // namespace rdna4
