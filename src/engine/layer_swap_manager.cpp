#include "layer_swap_manager.hpp"
#include "engine/types.hpp"
#include <algorithm>
#include <cstring>

namespace notllama {

LayerSwapManager::LayerSwapManager(size_t maxVramBytes, size_t maxSystemBytes)
    : max_vram_(maxVramBytes), max_system_(maxSystemBytes) {}

bool LayerSwapManager::RegisterLayer(uint32_t layerIdx,
                                     const std::vector<rdna4::TensorDesc>& weights) {
    size_t totalSize = 0;
    for (const auto& w : weights) {
        totalSize += w.sizeBytes;
    }

    LayerResidency lr;
    lr.layerIndex = layerIdx;
    lr.inVRAM = false;
    lr.vramOffset = 0;
    lr.systemOffset = staging_buffer_.size();
    lr.sizeBytes = totalSize;

    staging_buffer_.resize(staging_buffer_.size() + totalSize);

    layers_.push_back(lr);
    system_used_ += totalSize;
    return true;
}

bool LayerSwapManager::EnsureLayerInVRAM(uint32_t layerIdx, VkDevice device,
                                         VkPhysicalDevice pdev, uint32_t queueFamily,
                                         IMemoryAllocator* allocator) {
    (void)device; (void)pdev; (void)queueFamily;
    auto it = std::find_if(layers_.begin(), layers_.end(),
                           [layerIdx](const LayerResidency& lr) { return lr.layerIndex == layerIdx; });
    if (it == layers_.end()) return false;

    if (it->inVRAM) {
        Touch(layerIdx);
        return true;
    }

    while (vram_used_ + it->sizeBytes > max_vram_ && !lru_queue_.empty()) {
        if (!EvictOneLayer(device, allocator)) break;
    }

    if (vram_used_ + it->sizeBytes > max_vram_) {
        fprintf(stderr, "[LayerSwapManager] Cannot fit layer %u in VRAM (need %zu, have %zu)\n",
                layerIdx, it->sizeBytes, max_vram_ - vram_used_);
        return false;
    }

    GpuAllocation gpu_alloc = allocator->Allocate(MemoryType::WEIGHT, it->sizeBytes, 256);
    if (!gpu_alloc.buffer) {
        fprintf(stderr, "[LayerSwapManager] VRAM allocation failed for layer %u\n", layerIdx);
        return false;
    }

    it->vramOffset = gpu_alloc.device_address;
    it->inVRAM = true;
    vram_used_ += it->sizeBytes;
    Touch(layerIdx);
    return true;
}

void LayerSwapManager::FreeSystemCopies() {
    bool allInVram = true;
    for (auto& lr : layers_) {
        if (!lr.inVRAM) { allInVram = false; break; }
    }
    if (!allInVram) {
        fprintf(stderr, "[LayerSwapManager] Cannot free system copies: not all layers in VRAM\n");
        return;
    }
    staging_buffer_.clear();
    staging_buffer_.shrink_to_fit();
    system_used_ = 0;
    fprintf(stderr, "[LayerSwapManager] System RAM copies freed. VRAM-only mode active.\n");
}

VkDeviceAddress LayerSwapManager::GetWeightAddress(uint32_t layerIdx, int slot) const {
    (void)slot;
    auto it = std::find_if(layers_.begin(), layers_.end(),
                           [layerIdx](const LayerResidency& lr) { return lr.layerIndex == layerIdx; });
    if (it == layers_.end() || !it->inVRAM) return 0;
    return it->vramOffset;
}

void LayerSwapManager::Touch(uint32_t layerIdx) {
    auto it = std::find(lru_queue_.begin(), lru_queue_.end(), layerIdx);
    if (it != lru_queue_.end()) lru_queue_.erase(it);
    lru_queue_.push_back(layerIdx);
}

bool LayerSwapManager::EvictOneLayer(VkDevice device, IMemoryAllocator* allocator) {
    (void)device; (void)allocator;
    if (lru_queue_.empty()) return false;
    uint32_t victim = lru_queue_.front();
    lru_queue_.erase(lru_queue_.begin());

    auto it = std::find_if(layers_.begin(), layers_.end(),
                           [victim](const LayerResidency& lr) { return lr.layerIndex == victim; });
    if (it == layers_.end() || !it->inVRAM) return false;

    it->inVRAM = false;
    vram_used_ -= it->sizeBytes;
    fprintf(stderr, "[LayerSwapManager] Evicted layer %u from VRAM\n", victim);
    return true;
}

} // namespace notllama
