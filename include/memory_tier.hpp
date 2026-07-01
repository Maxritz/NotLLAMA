#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vulkan/vulkan.h>

namespace notllama {

// Where a tensor/layer resides
enum class MemoryTier {
    VRAM,       // GPU device-local memory
    SYSTEM_RAM, // Host memory (CPU-accessible)
    DISK,       // Paged out to disk/cache file
};

// A buffer that can live in VRAM or system RAM
struct TieredBuffer {
    std::string name;           // Tensor name (e.g., "blk.0.attn_q.weight")
    uint32_t layer_index = 0;   // Which layer (0 for global weights)
    size_t size_bytes = 0;
    MemoryTier current_tier = MemoryTier::DISK;

    // GPU side
    VkBuffer gpu_buffer = VK_NULL_HANDLE;
    VkDeviceMemory gpu_memory = VK_NULL_HANDLE;
    VkDeviceAddress gpu_address = 0;

    // CPU side
    void* cpu_data = nullptr;   // Aligned host memory
    size_t cpu_capacity = 0;

    // Quantization info
    uint32_t block_size = 0;
    uint32_t block_elements = 0;
    bool is_layer_weight = false;
};

// Configuration for memory tiering
struct TierConfig {
    size_t vram_budget_bytes = 0;        // 0 = auto-detect from GPU
    size_t system_ram_budget_bytes = 0;  // 0 = unlimited
    std::string spill_directory;          // Where to page out (empty = disable)
    int32_t gpu_layers = -1;             // -1 = all in VRAM, 0 = all in RAM, N = first N layers in VRAM
    std::string split_mode = "layer";     // "layer" | "row"
    bool kv_cache_in_vram = true;
    bool async_prefetch = true;
};

// Manages VRAM / System RAM / Disk tiers for model layers
class MemoryTierManager {
public:
    MemoryTierManager(VkDevice device, VkPhysicalDevice physical_device,
                      uint32_t queue_family_index);
    ~MemoryTierManager();

    void Configure(const TierConfig& cfg);
    const TierConfig& GetConfig() const { return config_; }

    void AutoDetectVRAMBudget();

    // Allocate a buffer (initially in system RAM)
    TieredBuffer* AllocateBuffer(const std::string& name, size_t size_bytes,
                                   uint32_t layer_index, bool is_layer);

    // Upload data into the CPU buffer
    bool WriteCPUData(TieredBuffer* buf, const void* data, size_t size);

    // Promote / demote between tiers
    bool PromoteToVRAM(TieredBuffer* buf);
    bool DemoteToRAM(TieredBuffer* buf);
    bool SpillToDisk(TieredBuffer* buf);
    bool RestoreFromDisk(TieredBuffer* buf);

    // Layer-level operations
    bool LoadLayerToVRAM(uint32_t layer_index);
    bool OffloadLayerToRAM(uint32_t layer_index);
    bool IsLayerInVRAM(uint32_t layer_index) const;

    // Get buffer (auto-promotes from lower tiers)
    TieredBuffer* GetLayerBuffer(const std::string& name, uint32_t layer_index);
    TieredBuffer* FindBuffer(const std::string& name);

    // Free
    void FreeBuffer(TieredBuffer* buf);

    // Stats
    size_t GetVRAMUsed() const { return vram_used_; }
    size_t GetVRAMBudget() const { return config_.vram_budget_bytes; }
    size_t GetVRAMAvailable() const;
    size_t GetRAMUsed() const { return ram_used_; }
    std::string GetStatsString() const;

    // Compute which layers fit in VRAM
    std::vector<uint32_t> ComputeVRAMLayout(size_t num_layers,
                                              const std::vector<size_t>& layer_sizes);

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    uint32_t queue_family_index_;
    TierConfig config_;

    std::vector<std::unique_ptr<TieredBuffer>> buffers_;
    std::unordered_map<std::string, TieredBuffer*> name_map_;
    std::unordered_map<uint32_t, std::vector<TieredBuffer*>> layer_map_;
    mutable std::mutex mutex_;

    size_t vram_used_ = 0;
    size_t ram_used_ = 0;
    size_t disk_used_ = 0;

    // Internal helpers
    bool UploadToGPU(TieredBuffer* buf);
    bool DownloadFromGPU(TieredBuffer* buf);
    void* AllocateAlignedCPU(size_t size);
    void FreeAlignedCPU(void* ptr);
    std::string GetSpillPath(const std::string& name);
};

} // namespace notllama
