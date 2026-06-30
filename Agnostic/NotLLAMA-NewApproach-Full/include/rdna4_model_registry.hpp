#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>

namespace rdna4 {

struct ModelDesc;
struct ModelHParams;
struct VulkanContext;
struct Scheduler;
struct PipelineBuilder;
struct KVCacheManager;
struct InferenceEngine;
struct Tokenizer;
struct RingAllocator;

// ============================================================================
// Loaded model slot
// ============================================================================
struct ModelSlot {
    std::string id;               // user-defined model ID
    std::string name;             // friendly name
    std::string jsonPath;
    std::string binPath;

    ModelHParams hparams;
    std::unique_ptr<ModelDesc> model;

    // GPU resources
    std::unique_ptr<KVCacheManager> kvCache;
    std::unique_ptr<InferenceEngine> engine;

    // State
    bool loaded = false;
    uint64_t lastUsedUs = 0;
    uint64_t loadTimeUs = 0;
    size_t vramBytes = 0;
    size_t ramBytes = 0;

    // Usage counters
    uint64_t nTokensGenerated = 0;
    uint64_t nRequestsServed = 0;
};

// ============================================================================
// Model registry — multi-model loading, switching, VRAM budgeting
// ============================================================================
class ModelRegistry {
public:
    struct Config {
        size_t maxVramBytes = 0;          // 0 = auto-detect from GPU
        size_t maxRamBytes = 0;           // 0 = auto-detect
        size_t reserveVramBytes = 512ull * 1024 * 1024; // reserve 512MB headroom
        uint32_t maxLoadedModels = 4;
        bool autoUnloadOnPressure = true;
        float unloadThreshold = 0.95f;    // unload LRU model when VRAM > 95%
    };

    explicit ModelRegistry(VulkanContext* ctx, Scheduler* sched,
                           PipelineBuilder* pipes, Tokenizer* tok,
                           RingAllocator* alloc, const Config& cfg);

    // Register a model file pair (does not load yet)
    bool registerModel(const std::string& id,
                       const std::string& jsonPath,
                       const std::string& binPath,
                       const std::string& name = "");

    // Load a registered model into GPU memory
    bool loadModel(const std::string& id);

    // Unload a model (free GPU memory, keep registration)
    bool unloadModel(const std::string& id);

    // Switch active model (unloads others if VRAM pressure)
    bool activateModel(const std::string& id);

    // Get currently active model engine
    InferenceEngine* getActiveEngine() const;
    ModelSlot* getActiveSlot() const;

    // Lookup any registered model
    ModelSlot* getModel(const std::string& id);
    const ModelSlot* getModel(const std::string& id) const;

    // List registered / loaded models
    std::vector<std::string> listRegistered() const;
    std::vector<std::string> listLoaded() const;

    // VRAM budgeting
    size_t getTotalVram() const;
    size_t getUsedVram() const;
    size_t getAvailableVram() const;
    bool checkVramPressure() const;

    // Auto-unload LRU model under pressure
    bool evictLruIfNeeded();

    // Preload / warm models (load but don't activate)
    bool preloadModel(const std::string& id);

    // Hot-swap weights for LoRA (apply adapter without full reload)
    bool applyLora(const std::string& modelId,
                   const std::string& loraJsonPath,
                   const std::string& loraBinPath,
                   float scale);

    // Remove registration entirely
    void unregisterModel(const std::string& id);

    // Global stats
    uint64_t getTotalTokensGenerated() const;
    uint64_t getTotalRequestsServed() const;

private:
    VulkanContext* ctx_;
    Scheduler* sched_;
    PipelineBuilder* pipes_;
    Tokenizer* tok_;
    RingAllocator* alloc_;
    Config cfg_;

    std::unordered_map<std::string, std::unique_ptr<ModelSlot>> slots_;
    std::string activeId_;
    mutable std::mutex mutex_;

    bool doLoad(ModelSlot& slot);
    void doUnload(ModelSlot& slot);
};

} // namespace rdna4
