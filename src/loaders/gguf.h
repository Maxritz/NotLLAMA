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
    uint32_t nParams = 0;
    uint32_t nLayers = 0;
    uint32_t nHeads = 0;
    uint32_t nKVHeads = 0;
    uint32_t nEmbd = 0;
    uint32_t nFF = 0;
    uint32_t nVocab = 0;
    uint32_t nCtx = 0;
    float ropeFreqBase = 10000.0f;
    float ropeScaling = 1.0f;
    uint32_t headDim = 0;
    std::string architecture;

    // Tokenizer data (populated when present in GGUF metadata)
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    uint32_t bosId = 1;
    uint32_t eosId = 2;
    uint32_t padId = 0;
    uint32_t unkId = 3;
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

    const GGUFMetadata& metadata() const { return meta_; }
    const std::vector<GGUFFakeTensor>& tensors() const { return tensors_; }
    const std::vector<uint8_t>& data() const { return data_; }
    int tensorIndex(const std::string& name) const;
    void printInfo() const;

private:
    bool readHeader(FILE* f);
    bool readTensorInfo(FILE* f);

    GGUFMetadata meta_{};
    std::vector<GGUFFakeTensor> tensors_;
    std::unordered_map<std::string, int> tensorMap_;
    std::vector<uint8_t> data_;
    size_t dataSize_ = 0;
};

} // namespace notllama
