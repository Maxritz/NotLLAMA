# Purpose

Modular engine interface layer and adapter implementations for NotLLAMA's Vulkan inference engine.
This module decouples the inference pipeline into replaceable components behind pure-virtual interfaces,
matching the v6.0 production blueprint.

## Ownership

Owned by the inference architecture. Changes to interface contracts must be reviewed against both
`include/engine/` consumers and `src/engine/` adapters.

## Local Contracts

- Core interfaces in `include/engine/i<name>.hpp` are pure virtual. No implementation logic belongs there.
- Adapter headers in `include/engine/<adapter>.hpp` expose concrete implementations and may inline small constructors.
- `GpuAllocation` represents a sub-region within a Vulkan buffer, not a raw `VkDeviceMemory` allocation.
- Adapters in `src/engine/` wrap existing `rdna4::*` classes and are temporary bridges.
- The modular compute engine (`VulkanComputeEngine`) implements a real per-token inference
  pipeline (embed → per-layer attention/FFN → LM head → top-k).
  - Dense FFN uses GPU GEMM. MoE FFN routes on the GPU when `gate_inp` is F32, then falls
    back to a CPU top-1-expert FFN because most real MoE GGUFs use quantized experts.
  - Q6_K GEMM is a per-column matvec and validated on small models. Q2_K, Q3_K, and Q5_K
    GEMM shaders were added from `quantshaderpack/` with corrected llama.cpp block layouts.
  - Full prompt prefill is not batched; tokens are processed one at a time.
  - `WeightUploader::uploadTensor()` uses `GetTensorDataSize()` for the GGUF-mmap path so
    the empty `binData_` member no longer causes an OOB upload error.
- `ShaderCompiler` compiles GLSL source at runtime via the system `glslc`. If `VULKAN_SDK` is not set and `glslc` is not on `PATH`, runtime compilation is skipped and `VulkanShaderLibrary` falls back to precompiled `.spv` files.
- `VulkanShaderLibrary` supports both enum-based (`GetPipeline(KernelType, ...)`) and string-based (`CompileNamedShader(name)`, `GetNamedPipeline(name)`) access.
- `PrecompileAll(spec)` iterates every `.comp` file in the shader directory, compiles it with the current profile/spec, and caches the result. Called once on first use.

## Work Guidance

### Adding a New Interface

1. Define it in `include/engine/i<name>.hpp`.
2. Add the include to `include/engine/engine.hpp`.
3. Provide an adapter or stub in `src/engine/`.
4. Register the source file in `CMakeLists.txt`.
5. Update this AGENTS.md and the root Child DOX Index.

### Modifying an Existing Interface

1. Check all adapters in `src/engine/` for impacted call sites.
2. Keep methods pure virtual; do not add default implementations in interface headers.
3. Prefer small, single-purpose methods over large structs passed by value.

## Verification

- Build: `cd build && cmake --build . --config Release`
- The engine module compiles when added to `rdna4_llama`.
- Runtime verification: smoke-test on `stories260k.gguf` (generates tokens) and model validation on Q6_K MoE/Gemma models. Kernel unit tests (`test_rms_norm`, `test_rope`, etc.) should pass before relying on a new shader path.
- `quantshaderpack/` shaders (gemm_q2_k, gemm_q3_k, gemm_q5_k, moe_router, moe_experts) are integrated but Q2_K/Q3_K/Q5_K GEMM accuracy is unverified against reference outputs.

## Child DOX Index

- None (`include/engine/` headers are owned by this module; no separate child doc needed).

## Adapters in this module

- `vulkan_device.cpp` — `IDevice` adapter for `rdna4::VulkanContext`
- `ring_allocator_adapter.cpp` — `IMemoryAllocator` bridge using three `rdna4::RingAllocator` instances
- `tokenizer_adapter.cpp` — `ITokenizer` adapter for `rdna4::Tokenizer`
- `model_adapter.cpp` — `IModel` adapter for `rdna4::WeightUploader` + `rdna4::ModelDesc`
- `kv_cache_adapter.cpp` — `IAttentionScheduler` adapter for `rdna4::KVCacheManager`
- `vulkan_compute_engine.cpp` — real `IComputeEngine` implementation: per-token inference pipeline with lazy layer upload, Q2_K/Q3_K/Q5_K/Q6_K GEMM, GPU MoE router + CPU top-1-expert FFN fallback, and GGUF OOB upload fix
- `vulkan_shader_library.cpp` — `IShaderLibrary` implementation: enum-based + named pipelines, `PrecompileAll`, runtime GLSL compile with fallback. KernelType stream includes Q2_K/Q3_K/Q5_K GEMM and MOE_ROUTER/MOE_EXPERTS from `quantshaderpack/`.
- `vulkan_descriptor_manager.cpp` — `IDescriptorManager` implementation: bindless SSBO descriptor table + per-frame metadata ring buffer
- `vulkan_debug_context.cpp` — `IDebugContext` implementation using VK_EXT_debug_utils
- `shader_compiler.cpp` — runtime GLSL → SPIR-V compiler using the system `glslc`
