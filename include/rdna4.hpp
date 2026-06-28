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
    // Base formats (match GGML type IDs)
    F32      = 0,
    F16      = 1,
    Q4_0     = 2,
    Q4_1     = 3,
    Q5_0     = 6,
    Q5_1     = 7,
    Q8_0     = 8,
    Q8_1     = 9,
    // K-quants
    Q2_K     = 10,
    Q3_K     = 11,
    Q4_K     = 12,
    Q5_K     = 13,
    Q6_K     = 14,
    Q8_K     = 15,
    // IQ quants (importance-matrix quantized)
    IQ2_XXS  = 16,
    IQ2_XS   = 17,
    IQ3_XXS  = 18,
    IQ3_XS   = 19,
    IQ2_S    = 20,
    IQ3_S    = 21,
    IQ4_XS   = 22,
    IQ1_S    = 23,
    IQ4_NL   = 24,
    IQ3_M    = 25,
    IQ2_M    = 26,
    IQ1_M    = 27,
    IQ4_XXS  = 28,
    IQ5_XXS  = 29,
    IQ5_XS   = 30,
    IQ5_S    = 31,
    IQ5_M    = 32,
    IQ6_XXS  = 33,
    IQ6_K    = 34,
    IQ7_K    = 35,
    IQ8_K    = 36,
    // Additional base quants
    Q1_0     = 37,
    Q1_1     = 38,
    Q2_0     = 39,
    Q2_1     = 40,
    Q3_0     = 41,
    Q3_1     = 42,
    Q6_0     = 43,
    Q6_1     = 44,
    Q7_0     = 45,
    Q7_1     = 46,
    Q4_K_S   = 47,
    Q4_K_M   = 48,
    Q5_K_S   = 49,
    Q5_K_M   = 50,
    Q6_K_S   = 51,
    Q6_K_M   = 52,
    // Float variants
    BF16     = 53,
    // MoE / micro-scaling formats
    MXFP4    = 54,
    MXFP6    = 55,
    MXFP8    = 56,
    // TurboQuant formats (GPU-native, d-last layout)
    TQ4_128  = 57,  // TurboQuant 4-bit, 128-element blocks
    TQ3_128  = 58,  // TurboQuant 3-bit, 128-element blocks
    TQ6_64   = 59,  // TurboQuant 6-bit, 64-element blocks
    // Count sentinel
    COUNT    = 60,
};

struct BufferDesc {
    VkDeviceAddress gpuAddress;
    VkDeviceSize size;
    QuantFormat format;
    float scale;
};

} // namespace rdna4
