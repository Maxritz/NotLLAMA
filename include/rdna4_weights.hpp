#pragma once
#include "rdna4.hpp"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

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
    uint32_t vocabSize;
    uint32_t contextLength;
    std::vector<TensorDesc> tensors;
};

class WeightUploader {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    WeightUploader(VkDevice dev, VkPhysicalDevice pdev) : device(dev), physicalDevice(pdev) {}

    ModelDesc load(const std::string& jsonPath, const std::string& binPath);
    void loadTokenizer(Tokenizer& tokenizer, const nlohmann::json& tokenizerJson);
    void uploadTensor(const TensorDesc& desc, const void* data);
    void freeTensor(const TensorDesc& desc);

private:
    VkBuffer createGpuBuffer(size_t size, VkDeviceAddress* outAddr);
};

} // namespace rdna4
