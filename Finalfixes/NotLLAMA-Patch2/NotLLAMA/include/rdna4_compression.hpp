#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace rdna4 {

// Context compression configuration (matches docs/context_compression_schema.json)
// Byte layout: scale follows packed weights (d-last convention for TurboQuant).
struct ContextCompressionConfig {
    bool enabled = false;
    uint32_t targetLayers = 0;       // 0 = all layers
    uint32_t blockSize = 128;
    uint32_t bits = 4;               // 4, 5, 6, 8
    uint32_t scaleBits = 16;         // 8 or 16
    bool zeroPoint = true;
    float threshold = 0.85f;         // Compress when context exceeds this ratio of maxContext
    uint32_t strategy = 0;           // 0=uniform, 1=entropy, 2=importance
    std::vector<uint32_t> layerMask; // Per-layer override (empty = use targetLayers)
};

// KV cache compression configuration (matches docs/kv_compression_schema.json)
struct KVCompressionConfig {
    bool enabled = false;
    uint32_t blockSize = 64;
    uint32_t kBits = 4;
    uint32_t vBits = 4;
    uint32_t scaleBits = 16;
    bool zeroPoint = true;
    uint32_t minSeqLen = 256;        // Only compress sequences >= this length
    float tokenImportanceThreshold = 0.1f; // For importance-based quantization
    uint32_t quantStrategy = 0;      // 0=per-token, 1=per-head, 2=global
};

// Push constants for compress_context.comp.
// Matches GLSL layout(push_constant, scalar) in compress_context.comp.
struct CompressContextPushConstants {
    uint64_t addrSrc;
    uint64_t addrDst;
    uint64_t addrImportance; // Optional, 0 if not used
    uint32_t n;
    uint32_t blockSize;
    uint32_t bits;
    uint32_t scaleBits;
    uint32_t zeroPoint;
    uint32_t strategy;
    float    threshold;
};
static_assert(sizeof(CompressContextPushConstants) <= 128,
    "CompressContextPushConstants exceeds Vulkan 128-byte push constant limit");

// Push constants for kv_cache_quantize.comp.
// Matches GLSL layout(push_constant, scalar) in kv_cache_quantize.comp.
// Byte layout: packed weights first (uint[]), then scales in separate buffer (float16_t[]).
struct KVCacheQuantizePushConstants {
    uint64_t addrKSrc;
    uint64_t addrVSrc;
    uint64_t addrKDst;
    uint64_t addrVDst;
    uint64_t addrKScales;
    uint64_t addrVScales;
    uint32_t seqLen;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t blockSize;
};
static_assert(sizeof(KVCacheQuantizePushConstants) <= 128,
    "KVCacheQuantizePushConstants exceeds Vulkan 128-byte push constant limit");

// Push constants for kv_cache_dequant.comp.
// Matches GLSL layout(push_constant, scalar) in kv_cache_dequant.comp.
// Byte layout: per block = [packed_weights (bytes)] [scale (1B if int8, 2B if fp16 LE)] [zp (1B, optional)].
struct KVCacheDequantPushConstants {
    uint64_t addrSrc;     // Quantized data (uint8_t[])
    uint64_t addrDst;     // F16 output (float16_t[])
    uint32_t n;           // Total elements (tokens * nHeads * headDim)
    uint32_t blockSize;   // Elements per block (64 or 128)
    uint32_t bits;        // 4, 5, 6, or 8
    uint32_t scaleBits;   // 8 (int8) or 16 (float16)
    uint32_t zeroPoint;   // 0 = no zero point, 1 = int8 zero point after scale
};
static_assert(sizeof(KVCacheDequantPushConstants) <= 128,
    "KVCacheDequantPushConstants exceeds Vulkan 128-byte push constant limit");

} // namespace rdna4
