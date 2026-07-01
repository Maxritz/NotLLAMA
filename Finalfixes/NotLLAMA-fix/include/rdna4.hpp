#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace rdna4 {

class Context;
class MemoryManager;
class Mailbox;
class Scheduler;

enum class QuantFormat : uint32_t {
    // Official ggml_type enum — exact match to ggml.org/ggml/include/ggml.h
    F32      = 0,
    F16      = 1,
    Q4_0     = 2,
    Q4_1     = 3,
    // 4, 5 removed from ggml
    Q5_0     = 6,
    Q5_1     = 7,
    Q8_0     = 8,
    Q8_1     = 9,
    // K-quants (256-element super-blocks)
    Q2_K     = 10,
    Q3_K     = 11,
    Q4_K     = 12,
    Q5_K     = 13,
    Q6_K     = 14,
    Q8_K     = 15,
    // IQ quants (importance-matrix, 256-element super-blocks)
    IQ2_XXS  = 16,
    IQ2_XS   = 17,
    IQ3_XXS  = 18,
    IQ1_S    = 19,
    IQ4_NL   = 20,   // NL = non-linear, 32-element blocks
    IQ3_S    = 21,
    IQ2_S    = 22,
    IQ4_XS   = 23,
    // Integer types
    I8       = 24,
    I16      = 25,
    I32      = 26,
    I64      = 27,
    F64      = 28,
    IQ1_M    = 29,
    BF16     = 30,
    // 31-33 removed (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8)
    TQ1_0    = 34,   // Ternary quantization
    TQ2_0    = 35,   // Ternary quantization
    // 36-38 removed (IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8)
    MXFP4    = 39,   // Microscaling FP4 (1 block)
    NVFP4    = 40,   // NV FP4 (4 blocks, E4M3 scale)
    Q1_0     = 41,   // 1-bit, fp16 block scale, 128-element blocks
    // ggml COUNT = 42

    // Custom extensions (NOT in ggml.h — engine-internal only)
    // IDs start at 100 to avoid collision with future ggml additions
    TQ4_128  = 100,  // TurboQuant 4-bit, 128-element blocks
    TQ3_128  = 101,  // TurboQuant 3-bit, 128-element blocks
    TQ6_64   = 102,  // TurboQuant 6-bit, 64-element blocks
    UNSUPPORTED = 255,
    COUNT    = 103,
};

struct BufferDesc {
    VkDeviceAddress gpuAddress;
    VkDeviceSize size;
    QuantFormat format;
    float scale;
};

} // namespace rdna4
