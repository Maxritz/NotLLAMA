#include "multi_model_manager.hpp"
#include <cstring>
#include <algorithm>
#include <chrono>

namespace notllama {

MultiModelManager::MultiModelManager(VkDevice device, VkPhysicalDevice physical_device,
                                     uint32_t queue_family_index,
                                     VulkanShaderLibrary* shader_lib,
                                     VulkanDescriptorManager* desc_mgr,
                                     RingAllocatorAdapter* allocator)
    : device_(device), physical_device_(physical_device),
      queue_family_index_(queue_family_index),
      shader_lib_(shader_lib), desc_mgr_(desc_mgr), allocator_(allocator) {
    ProbeVRAM();
}

MultiModelManager::~MultiModelManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.clear(); // destroys all models and frees VRAM
}

bool MultiModelManager::ProbeVRAM() {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

    // Sum all device-local heaps
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += memProps.memoryHeaps[i].size;
        }
    }
    budget_.total_bytes = static_cast<size_t>(total);
    budget_.used_bytes = 0;
    fprintf(stderr, "[MultiModel] VRAM: total=%.1f GB, reserved=%.1f GB\n",
            total / (1024.0 * 1024.0 * 1024.0),
            budget_.reserved_bytes / (1024.0 * 1024.0 * 1024.0));
    return total > 0;
}

void MultiModelManager::SetVRAMBudget(size_t total_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_.total_bytes = total_bytes;
}

std::string MultiModelManager::LoadModel(const std::string& path,
                                           const std::string& id,
                                           const std::string& name,
                                           const std::string& tags,
                                           uint32_t gpu_layers,
                                           size_t max_context,
                                           float temp, float top_p, int32_t top_k) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (models_.count(id)) {
        fprintf(stderr, "[MultiModel] Model '%s' already loaded\n", id.c_str());
        return "";
    }

    fprintf(stderr, "[MultiModel] Loading model '%s' from %s...\n", id.c_str(), path.c_str());

    auto inst = std::make_unique<ModelInstance>();
    inst->id = id;
    inst->name = name.empty() ? id : name;
    inst->path = path;
    inst->tags = tags;
    inst->gpu_layers = gpu_layers;
    inst->max_context = max_context;
    inst->temperature = temp;
    inst->top_p = top_p;
    inst->top_k = top_k;
    inst->last_used = std::chrono::steady_clock::now();

    // Create adapter and load model
    inst->adapter = std::make_unique<notllama::ModelAdapter>(device_, physical_device_, queue_family_index_);

    bool is_gguf = (path.size() >= 5 && path.substr(path.size() - 5) == ".gguf");
    bool loaded = is_gguf ? inst->adapter->LoadFromGGUF(path)
                          : inst->adapter->LoadFromPath(path, nullptr);
    if (!loaded) {
        fprintf(stderr, "[MultiModel] FAILED to load model from %s\n", path.c_str());
        return "";
    }

    // Stream all layers to GPU
    uint32_t n_layers = static_cast<uint32_t>(inst->adapter->GetNumLayers());
    uint32_t layers_to_upload = (gpu_layers == (uint32_t)-1) ? n_layers
                                : std::min(gpu_layers, n_layers);

    for (uint32_t l = 0; l < layers_to_upload; l++) {
        if (!inst->adapter->StreamLayerWeights(l, nullptr)) {
            fprintf(stderr, "[MultiModel] FAILED to stream layer %u for '%s'\n", l, id.c_str());
            return "";
        }
    }

    if (!inst->adapter->UploadGlobalWeights()) {
        fprintf(stderr, "[MultiModel] FAILED to upload global weights for '%s'\n", id.c_str());
        return "";
    }

    // Create KV cache
    uint32_t n_kv_heads = static_cast<uint32_t>(inst->adapter->GetNumKVHeads());
    uint32_t head_dim = static_cast<uint32_t>(inst->adapter->GetHeadDim());
    inst->kv_cache = std::make_unique<rdna4::KVCacheManager>(
        device_, physical_device_,
        static_cast<uint32_t>(max_context), n_layers, n_kv_heads, head_dim);

    if (n_layers > 0) {
        if (!inst->kv_cache->allocate()) {
            fprintf(stderr, "[MultiModel] FAILED to allocate KV cache for '%s'\n", id.c_str());
            return "";
        }
    }

    // Estimate VRAM usage (rough: weights + KV cache + overhead)
    size_t weight_bytes = 0;
    for (const auto& t : inst->adapter->GetWeightTensors()) {
        weight_bytes += t.sizeBytes;
    }
    size_t kv_bytes = n_layers * max_context * n_kv_heads * head_dim * 2 * sizeof(float);
    inst->vram_used_bytes = weight_bytes + kv_bytes + 64 * 1024 * 1024; // 64MB overhead
    budget_.used_bytes += inst->vram_used_bytes;

    fprintf(stderr, "[MultiModel] Loaded '%s': %s | Layers: %zu | VRAM: %.1f MB\n",
            id.c_str(), inst->adapter->GetArchitecture().c_str(),
            inst->adapter->GetNumLayers(),
            inst->vram_used_bytes / (1024.0 * 1024.0));

    models_[id] = std::move(inst);
    return id;
}

bool MultiModelManager::UnloadModel(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id);
    if (it == models_.end()) return false;

    fprintf(stderr, "[MultiModel] Unloading model '%s'\n", id.c_str());
    if (budget_.used_bytes >= it->second->vram_used_bytes) {
        budget_.used_bytes -= it->second->vram_used_bytes;
    }
    models_.erase(it);
    return true;
}

ModelInstance* MultiModelManager::GetModel(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id);
    if (it != models_.end()) {
        it->second->last_used = std::chrono::steady_clock::now();
        return it->second.get();
    }
    return nullptr;
}

ModelInstance* MultiModelManager::GetModelByTag(const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, inst] : models_) {
        if (inst->tags.find(tag) != std::string::npos) {
            inst->last_used = std::chrono::steady_clock::now();
            return inst.get();
        }
    }
    return nullptr;
}

std::vector<std::string> MultiModelManager::ListModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& [id, inst] : models_) {
        result.push_back(id);
    }
    return result;
}

bool MultiModelManager::IsLoaded(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.count(id) > 0;
}

bool MultiModelManager::SwapOutLRU(size_t min_bytes_needed) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find least-recently-used active model
    ModelInstance* lru = nullptr;
    std::string lru_id;
    auto oldest = std::chrono::steady_clock::now();

    for (auto& [id, inst] : models_) {
        if (inst->active && inst->last_used < oldest) {
            oldest = inst->last_used;
            lru = inst.get();
            lru_id = id;
        }
    }

    if (!lru || lru->vram_used_bytes < min_bytes_needed) {
        fprintf(stderr, "[MultiModel] No model to swap out (need %zu MB)\n",
                min_bytes_needed / (1024 * 1024));
        return false;
    }

    fprintf(stderr, "[MultiModel] Swapping out '%s' to free %.1f MB\n",
            lru_id.c_str(), lru->vram_used_bytes / (1024.0 * 1024.0));

    // Free GPU weights (keep metadata)
    FreeModelWeights(lru);
    lru->active = false;
    budget_.used_bytes -= lru->vram_used_bytes;
    return true;
}

bool MultiModelManager::SwapIn(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id);
    if (it == models_.end()) return false;

    auto* inst = it->second.get();
    if (inst->active) return true; // Already active

    // Check VRAM availability
    if (inst->vram_used_bytes > budget_.Available()) {
        // Need to swap out other models
        size_t needed = inst->vram_used_bytes - budget_.Available();
        lock.unlock();
        if (!SwapOutLRU(needed)) {
            fprintf(stderr, "[MultiModel] Cannot swap in '%s': not enough VRAM\n", id.c_str());
            return false;
        }
        lock.lock();
    }

    // Re-upload weights
    if (!UploadModelWeights(inst)) {
        fprintf(stderr, "[MultiModel] Failed to swap in '%s'\n", id.c_str());
        return false;
    }

    inst->active = true;
    budget_.used_bytes += inst->vram_used_bytes;
    inst->last_used = std::chrono::steady_clock::now();
    fprintf(stderr, "[MultiModel] Swapped in '%s'\n", id.c_str());
    return true;
}

bool MultiModelManager::UploadModelWeights(ModelInstance* inst) {
    uint32_t n_layers = static_cast<uint32_t>(inst->adapter->GetNumLayers());
    uint32_t layers_to_upload = (inst->gpu_layers == (uint32_t)-1)
        ? n_layers : std::min(inst->gpu_layers, n_layers);

    for (uint32_t l = 0; l < layers_to_upload; l++) {
        if (!inst->adapter->StreamLayerWeights(l, nullptr)) return false;
    }
    return inst->adapter->UploadGlobalWeights();
}

void MultiModelManager::FreeModelWeights(ModelInstance* inst) {
    // GPU weights are freed when the adapter is destroyed or reset
    // For partial unload, we'd need a more granular approach
    (void)inst;
}

} // namespace notllama
