# llama.cpp Approach — Vulkan Backend Inference Pipeline

## Core Design

Pre-dequant ALL weights to F16 at load time. Single-submit per token with tiled GEMM and Flash Attention.

## Dequant Strategy

| Aspect | Detail |
|--------|--------|
| **Timing** | Once at load: all quantized weights → F16 GPU buffers |
| **Dequants per token** | 0 (all pre-dequantized) |
| **GPU memory** | ~2× weight size (quantized + F16 copies) |
| **Weight format** | GGML type system: `block_q8_0`, `block_q6_K`, etc. as typed structs |

## Submit Strategy

| Aspect | Detail |
|--------|--------|
| **Submits per token** | 1 |
| **Syncs per token** | 1 fence wait |
| **Technique** | Single cmd buffer, `vkCmdPipelineBarrier` between layers |

## Vulkan Device Setup

```cpp
// Feature chain: query ALL, pass ALL to vkCreateDevice
VkPhysicalDeviceFeatures2 features2 = {};
features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
features2.pNext = &feat14;  // full chain: 14→13→12→11→coop

vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);  // query

// Pass features2 directly to vkCreateDevice — ALL features preserved
// (NO zeroing of dynamicRendering or graphics features)
```

| Aspect | Detail |
|--------|--------|
| **Queue family** | Avoids `GRAPHICS_BIT`, picks ACE/Family 1 |
| **Env override** | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` → Family 0, **56% faster** on RDNA4 |
| **AMD routing** | Feature-rich → "modern app" → routes correctly on any queue |

## GEMM Shader (ggml-vulkan)

| Aspect | Detail |
|--------|--------|
| **Workgroup** | Tiled 16×16 (256 threads) with shared memory |
| **Algorithm** | `dotPacked4x8EXT` for sub-byte formats |
| **Format** | Reads from typed structs (`block_q8_0`, `block_q6_K`, etc.) |
| **Throughput** | ~64× naive implementation |

### Key Shader Difference

llama.cpp represents quantized weights as **typed structs**, not raw byte buffers:

```glsl
// llama.cpp style: typed struct per block
struct block_q8_0 {
    half d;        // float16 scale
    int8_t qs[32]; // signed int8 values
};
layout(set=0, binding=0, std430) readonly buffer WeightBuf {
    block_q8_0 data[];
};
```

This gives the compiler perfect type information — no manual byte extraction needed. The `half` type maps directly to Vulkan's 16-bit float support, and `int8_t[]` automatically sign-extends.

## Attention

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Flash Attention (tiled, memory-efficient) |
| **Optimization** | Years of tuning across GPU vendors |
| **Variant** | Uses GGML tensor ops, not custom Vulkan shaders for all paths |

## Per-Format Dequant

Each format has an exact byte layout (from GGML type system):

| Format | Struct | Block Size | Elements |
|--------|--------|-----------|----------|
| Q8_0 | `block_q8_0 { half d; int8_t qs[32]; }` | 34B | 32 |
| Q6_K | `block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; }` | 210B | 256 |
| Q4_0 | `block_q4_0 { half d; uint8_t qs[16]; }` | 18B | 32 |

Dequant is done ONCE on CPU during model load:
```cpp
// llama.cpp CPU dequant
for (int i = 0; i < nb; i++) {
    float d = GGML_FP16_TO_F32(x[i].d);
    for (int j = 0; j < qk/2; j++) {
        y[i*qk + j]        = d * (x[i].qs[j] & 0xF) - 8;
        y[i*qk + j + qk/2] = d * (x[i].qs[j] >> 4) - 8;
    }
}
```

## Strengths vs Weaknesses

| | Strength | Weakness |
|---|----------|----------|
| **llama.cpp** | Fastest time-to-token, mature tiled GEMM, single-submit, flash attention, all HW vendors | 2× VRAM for weights, complex codebase, GPU pre-dequant at init |

## Key Lessons for NotLLAMA

1. **Pre-dequant is simpler** — CPU-side dequant at load avoids all GPU byte-fiddling bugs
2. **Feature chain matters** — passing all features to `vkCreateDevice` is valid for AMD WDDM
3. **Graphics queue is faster** for compute on RDNA4 despite intuition
4. **Typed structs > raw bytes** — GLSL compiler handles sign extension and alignment
