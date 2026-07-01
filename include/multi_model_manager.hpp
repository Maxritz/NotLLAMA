#pragma once
#include "engine/model_adapter.hpp"
#include "engine/vulkan_compute_engine.hpp"
#include "engine/vulkan_shader_library.hpp"
#include "engine/vulkan_descriptor_manager.hpp"
#include "engine/ring_allocator_adapter.hpp"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <chrono>

namespace notllama {

// Info about a loaded model instance
struct ModelInstance {
    std::string id;                          // Unique model ID (user-defined)
    std::string name;                        // Human-readable name
    std::string path;                        // Path to model file
    std::string tags;                        // Comma-separated tags (e.g., "code,python")
    std::unique_ptr<notllama::ModelAdapter> adapter;
    std::unique_ptr<rdna4::KVCacheManager> kv_cache;
    uint32_t gpu_layers = 0;                 // Layers offloaded to GPU
    size_t vram_used_bytes = 0;              // VRAM consumed by this model
    size_t max_context = 2048;               // Max context length
    float temperature = 0.8f;                // Default sampling temp
    float top_p = 0.95f;                     // Default top-p
    int32_t top_k = 40;                      // Default top-k
    std::chrono::steady_clock::time_point last_used;
    uint64_t total_tokens_generated = 0;
    uint64_t total_prompts = 0;
    bool active = true;                      // Can be used for inference
};

// VRAM budget tracker
struct VRAMBudget {
    size_t total_bytes = 0;
    size_t used_bytes = 0;
    size_t reserved_bytes = 512 * 1024 * 1024; // 512MB reserve for OS/other
    size_t Available() const { return (total_bytes > used_bytes + reserved_bytes)
        ? (total_bytes - used_bytes - reserved_bytes) : 0; }
};

// Multi-model manager: loads, unloads, and manages multiple models
// Supports model swapping when VRAM budget exceeded
class MultiModelManager {
public:
    MultiModelManager(VkDevice device, VkPhysicalDevice physical_device,
                      uint32_t queue_family_index,
                      VulkanShaderLibrary* shader_lib,
                      VulkanDescriptorManager* desc_mgr,
                      RingAllocatorAdapter* allocator);
    ~MultiModelManager();

    // Load a model. Returns model ID or empty string on failure.
    std::string LoadModel(const std::string& path,
                          const std::string& id,
                          const std::string& name,
                          const std::string& tags,
                          uint32_t gpu_layers,
                          size_t max_context,
                          float temp, float top_p, int32_t top_k,
                          const std::string& load_mode = "mirror");

    // Unload a model by ID
    bool UnloadModel(const std::string& id);

    // Get model by ID
    ModelInstance* GetModel(const std::string& id);

    // Get model by tag (returns first match)
    ModelInstance* GetModelByTag(const std::string& tag);

    // List all loaded models
    std::vector<std::string> ListModels() const;

    // Check if a model is loaded
    bool IsLoaded(const std::string& id) const;

    // VRAM management
    size_t GetTotalVRAM() const { return budget_.total_bytes; }
    size_t GetUsedVRAM() const { return budget_.used_bytes; }
    size_t GetAvailableVRAM() const { return budget_.Available(); }
    void SetVRAMBudget(size_t total_bytes);

    // Swap least-recently-used model to CPU to free VRAM
    bool SwapOutLRU(size_t min_bytes_needed);

    // Swap a model back into VRAM
    bool SwapIn(const std::string& id);

    // Get count of loaded models
    size_t Count() const { return models_.size(); }

    // Lock for thread-safe access
    std::mutex& GetMutex() { return mutex_; }

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    uint32_t queue_family_index_;
    VulkanShaderLibrary* shader_lib_;
    VulkanDescriptorManager* desc_mgr_;
    RingAllocatorAdapter* allocator_;
    VRAMBudget budget_;
    std::unordered_map<std::string, std::unique_ptr<ModelInstance>> models_;
    mutable std::mutex mutex_;

    bool ProbeVRAM();
    bool UploadModelWeights(ModelInstance* inst);
    void FreeModelWeights(ModelInstance* inst);
};

} // namespace notllama
