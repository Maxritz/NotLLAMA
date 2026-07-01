#pragma once
#include "engine/imodel.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"

namespace notllama {

// Bridge adapter: maps IModel onto rdna4::ModelDesc and WeightUploader.
class ModelAdapter : public IModel {
public:
    ModelAdapter(VkDevice device, VkPhysicalDevice physical_device, uint32_t queue_family_index);
    ~ModelAdapter() override = default;

    std::string GetArchitecture() override;
    size_t GetNumLayers() override;
    size_t GetHeadDim() override;
    size_t GetNumKVHeads() override;
    size_t GetEmbeddingDim() override;
    float GetRoPEBase() override;
    float GetRoPEScale() override;

    std::string RemapTensorName(const std::string& gguf_name) override;
    std::vector<TensorMeta> GetWeightTensors() override;
    std::vector<rdna4::TensorDesc> GetRawTensors() override;

    bool LoadFromPath(const std::string& path, IMemoryAllocator* allocator) override;
    bool LoadFromGGUF(const std::string& path);
    bool StreamLayerWeights(uint32_t layer_index, IMemoryAllocator* allocator) override;
    bool UploadGlobalWeights();
    const void* GetWeightShadowCopy(const std::string& tensor_name) override;

    // Weight loading mode: vram (free RAM copy), mirror (keep shadow), lazy (on-demand)
    void SetLoadMode(rdna4::WeightLoadMode mode) { uploader_.SetLoadMode(mode); }
    rdna4::WeightLoadMode GetLoadMode() const { return uploader_.GetLoadMode(); }
    void OnAllLayersUploaded() { uploader_.OnAllLayersUploaded(model_); }

    rdna4::Tokenizer& GetTokenizer() { return tokenizer_; }

private:
    rdna4::Tokenizer tokenizer_;
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    uint32_t queue_family_index_;
    rdna4::WeightUploader uploader_;
    rdna4::ModelDesc model_;
    std::vector<bool> layers_loaded_;

    DataType ConvertFormat(rdna4::QuantFormat fmt) const;
};

} // namespace notllama
