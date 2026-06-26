#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace rdna4 {

// =============================================================================
// DYNAMIC VGPR — RDNA4 ISA-level feature (NOT available in standard Vulkan).
// =============================================================================
// S_ALLOC_VGPR and S_FREE_VGPR are AMD GCN instructions that allow a shader to
// shrink/grow its VGPR allocation at runtime. This enables:
//   - Higher occupancy for memory-bound kernels (fewer VGPRs = more waves)
//   - Lower latency for compute-bound kernels (more VGPRs = better ILP)
//
// LIMITATION: These instructions are NOT exposed in SPIR-V or GLSL.
// They require one of:
//   1. AMD_shader_info + pre-compiled GCN binary (VK_AMD_shader_info)
//   2. Inline assembly in ROCm/HIP (not Vulkan)
//   3. Custom assembler emitting raw AMD ISA (rdna4_as.py future work)
//
// This header is a placeholder for when the custom assembler layer supports
// AMD GCN/RDNA4 binary emission.
// =============================================================================

class DynamicVGPRExtension {
public:
    VkDevice device;
    bool available = false;

    DynamicVGPRExtension(VkDevice dev);

    // Check if VK_AMD_shader_info is available
    bool checkSupport();

    // Load a pre-compiled AMD ISA binary as a shader module
    // The binary must contain S_ALLOC_VGPR instructions
    VkShaderModule loadAMDShader(const uint32_t* isaBinary, size_t wordCount);

    // Query current VGPR usage from a dispatched shader
    uint32_t queryVGPRUsage(VkPipeline pipeline);
};

} // namespace rdna4
