#pragma once
#include "rdna4_weights.hpp"
#include "engine/imemory_allocator.hpp"
#include "engine/imodel.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace notllama {

struct LayerResidency {
    uint32_t layerIndex;
    bool inVRAM;
    size_t vramOffset;
    size_t systemOffset;
    size_t sizeBytes;
};

class LayerSwapManager {
public:
    LayerSwapManager(size_t maxVramBytes, size_t maxSystemBytes);

    bool RegisterLayer(uint32_t layerIdx, const std::vector<rdna4::TensorDesc>& weights);

    bool EnsureLayerInVRAM(uint32_t layerIdx, VkDevice device, VkPhysicalDevice pdev,
                           uint32_t queueFamily, IMemoryAllocator* allocator);

    void FreeSystemCopies();

    VkDeviceAddress GetWeightAddress(uint32_t layerIdx, int slot) const;

    size_t GetVramUsed() const { return vram_used_; }
    size_t GetSystemUsed() const { return system_used_; }

private:
    size_t max_vram_;
    size_t max_system_;
    size_t vram_used_ = 0;
    size_t system_used_ = 0;

    std::vector<LayerResidency> layers_;
    std::vector<uint8_t> staging_buffer_;

    std::vector<uint32_t> lru_queue_;

    void Touch(uint32_t layerIdx);
    bool EvictOneLayer(VkDevice device, IMemoryAllocator* allocator);
};

} // namespace notllama
