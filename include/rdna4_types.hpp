#pragma once
#include <cstdint>

namespace rdna4 {

struct GemmPushConstants {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint32_t M, N, K;
    float alpha;
    uint32_t transB;
};

struct AttentionPushConstants {
    uint64_t addrQ, addrK, addrV, addrOut;
    uint32_t seqLen, headDim, nHeads, nKvHeads, headIndex;
    float invSqrtHeadDim;
};

struct FlashAttentionPushConstants {
    uint64_t addrQ;
    uint64_t addrKCache;
    uint64_t addrVCache;
    uint64_t addrOut;
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t qRowStart;
    uint32_t qRowCount;
    uint32_t kvTileSize;
    float invSqrtHeadDim;
};

struct MlpPushConstants {
    uint64_t addrIn;
    uint64_t addrUp;
    uint64_t addrGate;
    uint64_t addrDown;
    uint64_t addrOut;
    uint32_t dim, hiddenDim;
    uint32_t quantFormat;
    float scale;
};

struct RopePushConstants {
    uint64_t addrQ;
    uint64_t addrK;
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t nHeads;
    uint32_t nKvHeads;
    float ropeBase;
    float ropeScale;
};

struct TopKPushConstants {
    uint64_t addrLogits;
    uint64_t addrOutput;      // {tokenId, tokenProb} — 8 bytes
    uint64_t addrScratch;     // temp buffer for probabilities (vocabSize floats)
    uint32_t vocabSize;
    float temperature;
    uint32_t topK;
    float topP;
    uint32_t seed;
};

struct AddPushConstants {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint32_t nElements;
};

struct RmsNormPushConstants {
    uint64_t addrIn;
    uint64_t addrWeight;
    uint64_t addrOut;
    uint32_t rowSize;
    uint32_t nRows;
    float eps;
};

struct EmbedPushConstants {
    uint64_t addrEmbedTable;
    uint64_t addrHiddenState;
    uint32_t tokenId;
    uint32_t tokenPos;
    uint32_t dim;
};

struct KVCacheWritePushConstants {
    uint64_t addrKIn;
    uint64_t addrVIn;
    uint64_t addrKCache;
    uint64_t addrVCache;
    uint32_t seqPos;
    uint32_t headDim;
    uint32_t nKvHeads;
};

struct BdaTestPushConstants {
    uint64_t addrOut;
    uint32_t nElements;
};

struct DequantizePushConstants {
    uint64_t addrQuant;
    uint64_t addrOut;
    uint32_t nElements;
    uint32_t quantFormat;
    uint32_t blockSize;
    uint32_t blockElements;
    uint32_t totalThreads;
};

} // namespace rdna4
