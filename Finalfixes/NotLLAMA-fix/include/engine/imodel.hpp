#pragma once
#include "types.hpp"
#include "rdna4_weights.hpp"
#include <string>
#include <vector>

namespace notllama {

class IMemoryAllocator;

class IModel {
public:
    virtual ~IModel() = default;

    virtual std::string GetArchitecture() = 0;
    virtual size_t GetNumLayers() = 0;
    virtual size_t GetHeadDim() = 0;
    virtual size_t GetNumKVHeads() = 0;
    virtual size_t GetEmbeddingDim() = 0;
    virtual float GetRoPEBase() = 0;
    virtual float GetRoPEScale() = 0;

    virtual std::string RemapTensorName(const std::string& gguf_name) = 0;
    virtual std::vector<TensorMeta> GetWeightTensors() = 0;
    virtual std::vector<rdna4::TensorDesc> GetRawTensors() = 0;

    virtual bool LoadFromPath(const std::string& path, IMemoryAllocator* allocator) = 0;
    virtual bool StreamLayerWeights(uint32_t layer_index, IMemoryAllocator* allocator) = 0;
    virtual const void* GetWeightShadowCopy(const std::string& tensor_name) = 0;
};

} // namespace notllama
