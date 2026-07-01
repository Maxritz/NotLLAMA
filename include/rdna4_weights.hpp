#pragma once
#include "rdna4.hpp"
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace notllama { class GGUFLoader; }

namespace rdna4 {

class Tokenizer;

struct TensorDesc {
    std::string name;
    QuantFormat format;
    std::vector<uint32_t> shape;
    uint32_t nDims;
    size_t sizeBytes;
    size_t binOffset;
    size_t binSize;
    uint32_t blockSize;
    uint32_t blockElements;
    VkDeviceAddress gpuAddress;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct ModelDesc {
    std::string architecture;
    uint32_t blockCount;
    uint32_t embeddingLength;
    uint32_t feedForwardLength;
    uint32_t headCount;
    uint32_t headCountKv;
    uint32_t headDim;       // actual head dim, derived from Q weight shape (not embeddingLength/headCount)
    uint32_t vocabSize;
    uint32_t contextLength;
    std::vector<TensorDesc> tensors;
};

// Weight loading strategy
enum class WeightLoadMode {
    VRAM,    // Upload to GPU, free system RAM copy immediately (fastest inference)
    MIRROR,  // Keep system RAM shadow copy for layer swapping (default)
    LAZY     // Upload on first use, lowest startup memory
};

class WeightUploader {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;

    WeightUploader(VkDevice dev, VkPhysicalDevice pdev, uint32_t qfi = 0);
    ~WeightUploader();

    ModelDesc load(const std::string& jsonPath, const std::string& binPath);
    ModelDesc loadMetadata(const std::string& jsonPath, const std::string& binPath);
    ModelDesc loadFromGGUF(const std::string& ggufPath, Tokenizer* tokenizer = nullptr);
    bool uploadTensor(TensorDesc& desc);
    bool uploadLayer(ModelDesc& model, uint32_t layerIndex);
    bool cpuDequantTensor(const TensorDesc& desc, std::vector<float>& out);
    void loadTokenizer(Tokenizer& tokenizer, const nlohmann::json& tokenizerJson);
    void freeTensor(const TensorDesc& desc);
    void freeAll(ModelDesc& model);

    // Set loading strategy (call before load/loadFromGGUF)
    void SetLoadMode(WeightLoadMode mode) { load_mode_ = mode; }
    WeightLoadMode GetLoadMode() const { return load_mode_; }

    // After all requested layers are uploaded, apply load-mode policy
    //   VRAM:  free system RAM copy
    //   MIRROR: keep system RAM copy (default)
    //   LAZY:   no-op (layers uploaded on demand)
    void OnAllLayersUploaded(ModelDesc& model);

    // Force free the system RAM copy (use when switching to VRAM-only mode)
    void FreeSystemRAMCopy();

    // Check if a tensor's data is still available in system RAM
    bool HasSystemRAMCopy() const;

private:
    std::vector<uint8_t> binData_;
    std::unique_ptr<notllama::GGUFLoader> gguf_loader_;
    WeightLoadMode load_mode_ = WeightLoadMode::MIRROR;

    const uint8_t* GetTensorDataPtr(const TensorDesc& desc) const;
    size_t GetTensorDataSize() const;

    VkBuffer createGpuBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem);
};

} // namespace rdna4
