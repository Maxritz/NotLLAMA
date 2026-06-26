#pragma once
#include "rdna4_vulkan.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace notllama {

enum class GGUFType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

enum class GGUFTensorType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
    IQ4_XS = 23, IQ1_M = 24, BF16 = 30, F32_MAX = 31,
};

struct GGUFQuantMeta {
    GGUFTensorType type;
    uint32_t blockSize;
    uint32_t bytesPerBlock;
};

GGUFQuantMeta getQuantMeta(GGUFTensorType type);

struct GGUFMetadata {
    uint32_t nParams;
    uint32_t nLayers;
    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t nEmbd;
    uint32_t nFF;
    uint32_t nVocab;
    uint32_t nCtx;
    float ropeFreqBase;
    float ropeScaling;
    uint32_t headDim;
    std::string architecture;
};

struct GGUFFakeTensor {
    std::string name;
    GGUFTensorType type;
    std::vector<uint64_t> dims;
    uint64_t offset;
    uint64_t nbytes;
};

class GGUFLoader {
public:
    GGUFLoader() = default;
    ~GGUFLoader() = default;

    bool load(const std::string& path);
    void uploadToGPU(VkDevice device, VkPhysicalDevice physDev,
                     VkCommandPool pool, VkQueue queue);

    const GGUFMetadata& metadata() const { return meta_; }
    const std::vector<rdna4::GpuBuffer>& gpuBuffers() const { return gpuBuffers_; }
    rdna4::GpuBuffer getTensorBuffer(const std::string& name) const;
    int tensorIndex(const std::string& name) const;
    void printInfo() const;

private:
    bool readHeader(FILE* f);
    bool readTensorInfo(FILE* f);

    GGUFMetadata meta_{};
    std::vector<GGUFFakeTensor> tensors_;
    std::unordered_map<std::string, int> tensorMap_;
    std::vector<uint8_t> data_;
    std::vector<rdna4::GpuBuffer> gpuBuffers_;
    size_t dataSize_ = 0;
};

} // namespace notllama
