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

// Fused gate+up+SiLU+down MLP — reads dequantized float32 weights.
// Replaces separate gate+up GEMMs + MLP with a single dispatch.
struct MlpFusedGateUpPushConstants {
    uint64_t addrIn;
    uint64_t addrGate;
    uint64_t addrUp;
    uint64_t addrDown;
    uint64_t addrOut;
    uint32_t dim;
    uint32_t hiddenDim;
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

struct SiluMulPushConstants {
    uint64_t addrGate;
    uint64_t addrUp;
    uint64_t addrOut;
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
    uint32_t totalThreads;
    uint32_t elementOffset;
};

// ============================================================================
// kernel_entry.comp push constants — must match GLSL layout(scalar) exactly.
// ============================================================================
struct KernelEntryPushConstants {
    uint64_t addrMailbox;
    uint64_t addrTokenEmbed;
    uint64_t addrOutputNorm;
    uint64_t addrLMHead;
    uint64_t addrHiddenState;
    uint64_t addrOutput;
    uint64_t addrLayerParams;
    uint64_t addrScratch;
    uint64_t addrLogits;         // host-readable logits output
    uint32_t vocabSize;
    uint32_t embeddingDim;
    uint32_t nLayers;
    uint32_t headDim;
    uint32_t nHeads;
    uint32_t nKvHeads;
    float    ropeBase;
    float    ropeScale;
};

// Per-layer weight addresses for kernel_entry.comp.
// Written once at init, read by the GPU kernel for every layer.
struct LayerParams {
    uint64_t addrAttnNorm;
    uint64_t addrQ;
    uint64_t addrK;
    uint64_t addrV;
    uint64_t addrAttnOut;
    uint64_t addrFfnNorm;
    uint64_t addrFfnUp;
    uint64_t addrFfnGate;
    uint64_t addrFfnDown;
    uint64_t addrKCache;
    uint64_t addrVCache;
};

// ── TurboQuant push constants ─────────────────────────────────────
// Matches layout in src/kernels/dequant_turbo.comp / gemm_turbo.comp.
// static_assert guarantees <= 128 bytes per Vulkan spec.
struct DequantTurboPushConstants {
    uint64_t addrSrc;        // source quantized weights (device address)
    uint64_t addrDst;        // destination f16 buffer (device address)
    uint32_t n;              // total elements to dequantize
    uint32_t blockSize;      // elements per block (64 or 128)
    uint32_t bits;           // bits per weight (3, 4, or 6)
    uint32_t scaleBits;      // scale width in bits (8 or 16)
    uint32_t zeroPoint;      // 0 = symmetric offset, 1 = unsigned
    float    scale;          // extra host-side scale multiplier
}; // 40 bytes under scalar layout
static_assert(sizeof(DequantTurboPushConstants) <= 128,
    "DequantTurboPushConstants must fit in 128 bytes");

struct GemmTurboPushConstants {
    uint64_t addrA;          // input activations (F16, M x K)
    uint64_t addrB;          // TurboQuant weights (K x N, packed)
    uint64_t addrC;          // output (F16, M x N)
    uint32_t M;              // rows
    uint32_t K;              // input features
    uint32_t N;              // output features
    uint32_t blockSize;      // elements per quant block
    uint32_t bits;           // bits per weight
    uint32_t scaleBits;      // scale width in bits
    uint32_t zeroPoint;      // 0 = symmetric offset
    float    alpha;          // output scale multiplier
}; // 52 bytes under scalar layout
static_assert(sizeof(GemmTurboPushConstants) <= 128,
    "GemmTurboPushConstants must fit in 128 bytes");

} // namespace rdna4
