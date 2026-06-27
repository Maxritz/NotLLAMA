#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <utility>

namespace rdna4 {

struct CpuModelDesc {
    uint32_t blockCount;
    uint32_t embeddingLength;
    uint32_t feedForwardLength;
    uint32_t headCount;
    uint32_t headCountKv;
    uint32_t vocabSize;
};

struct CpuTensor {
    std::vector<float> data;
    std::vector<uint32_t> shape;
};

class CpuReference {
public:
    CpuModelDesc desc{};
    CpuTensor tokenEmbD;
    CpuTensor outputNorm;

    struct Layer {
        CpuTensor attnNorm;
        CpuTensor attnQ;
        CpuTensor attnK;
        CpuTensor attnV;
        CpuTensor attnOutput;
        CpuTensor ffnNorm;
        CpuTensor ffnGate;
        CpuTensor ffnUp;
        CpuTensor ffnDown;
    };
    std::vector<Layer> layers;

    bool load(const std::string& jsonPath, const std::string& binPath);
    std::vector<float> forward(uint32_t tokenId);

    static void printTopK(const std::vector<float>& logits, int k);
    static std::pair<uint32_t, float> argmax(const std::vector<float>& logits);
};

} // namespace rdna4
