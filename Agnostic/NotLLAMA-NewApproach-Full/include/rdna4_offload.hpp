#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace rdna4 {

struct ModelHParams;
struct ModelDesc;
struct TensorDesc;

// ============================================================================
// Offload policy — which device executes which layers
// ============================================================================
enum class OffloadDevice : uint8_t {
    CPU = 0,
    GPU = 1,
    AUTO = 2,   // decide at runtime based on VRAM
};

struct LayerOffload {
    uint32_t layerIndex = 0;
    OffloadDevice device = OffloadDevice::GPU;
    // For CPU layers: pointer to CPU-allocated float32 weights
    std::vector<float> cpuWeights;  // contiguous dequantized buffer
    uint64_t cpuWeightBytes = 0;
};

// ============================================================================
// CPU execution backend interface
// ============================================================================
class CpuBackend {
public:
    virtual ~CpuBackend() = default;

    // Execute attention on CPU for offloaded layers
    virtual void attention(const float* q, const float* k, const float* v,
                           float* out, uint32_t seqLen, uint32_t headDim,
                           uint32_t nHeads, uint32_t nKvHeads) = 0;

    // Execute MLP on CPU for offloaded layers
    virtual void mlp(const float* in, const float* gate, const float* up,
                     const float* down, float* out,
                     uint32_t dim, uint32_t hiddenDim) = 0;

    // Execute RMS norm on CPU
    virtual void rmsNorm(const float* in, const float* weight,
                         float* out, uint32_t rowSize, float eps) = 0;

    // Execute RoPE on CPU
    virtual void rope(float* q, float* k, uint32_t seqPos,
                      uint32_t headDim, uint32_t nHeads, uint32_t nKvHeads,
                      float base, float scale) = 0;
};

// ============================================================================
// Offload manager — llama.cpp-style -ngl (n-gpu-layers) support
// ============================================================================
class OffloadManager {
public:
    struct Config {
        int32_t nGpuLayers = -1;       // -1 = all GPU, 0 = all CPU, N = first N on GPU
        size_t  maxGpuVramBytes = 0;   // auto-detect if 0
        bool    fallbackToCpu = true;  // if GPU OOM, offload remaining to CPU
        bool    asyncCpuCompute = true; // CPU layers run in background thread
        bool    pinCpuWeights = false; // mlock weights on CPU
    };

    explicit OffloadManager(const Config& cfg);

    // Plan offload split for a model
    bool planOffload(const ModelHParams& hparams,
                     const ModelDesc& model,
                     size_t availableVram);

    // Get offload decision for a specific layer
    OffloadDevice getLayerDevice(uint32_t layerIndex) const;

    // Query if layer is on GPU
    bool isLayerOnGpu(uint32_t layerIndex) const;

    // Get CPU backend (creates on first use)
    CpuBackend* getCpuBackend();

    // Prepare CPU buffers for offloaded layers
    bool uploadCpuLayer(uint32_t layerIndex,
                        const std::vector<TensorDesc>& tensors);

    // Execute a CPU layer (async if configured)
    void dispatchCpuLayer(uint32_t layerIndex,
                          const float* hiddenIn,
                          float* hiddenOut);

    // Synchronize CPU layer completion
    void syncCpuLayer(uint32_t layerIndex);

    // VRAM estimate for a given offload plan
    static size_t estimateVram(const ModelHParams& hparams,
                               const ModelDesc& model,
                               int32_t nGpuLayers);

    // Report split
    uint32_t getGpuLayerCount() const;
    uint32_t getCpuLayerCount() const;

    // Dynamic rebalancing (future)
    bool rebalance(int32_t newNGpuLayers);

private:
    Config cfg_;
    std::vector<LayerOffload> layerMap_;
    std::unique_ptr<CpuBackend> cpuBackend_;
    uint32_t nGpuLayers_ = 0;
    uint32_t nCpuLayers_ = 0;
};

} // namespace rdna4
