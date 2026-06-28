# src/kernels/ — Shader Conventions

## Purpose
GLSL compute shaders for LLM inference on RDNA4. Self-contained: dequantize → process → output.

## Ownership
Shaders are owned by the kernel pipeline. Host-side dispatch logic lives in `src/host/inference_engine.cpp` and `src/host/scheduler.cpp`.

## Local Contracts

### GLSL Requirements
- `#version 450` minimum. Use `#version 460` for scalar block layout support (matching existing shaders).
- `layout(local_size_x = ..., local_size_y = 1, local_size_z = 1) in;` unless fused GEMM uses 2D layout.
- All buffer references use `buffer_reference` with `buffer_reference_align = 16`.
- All push constants use `layout(push_constant, scalar) uniform Params { ... } pc;` naming.
- F16 I/O uses `float16_t`. Explicit conversion: `float16_t(x)` on write, `float(x)` on read.

### Data Layout
- Quantized weights: block-compressed. Block size varies by format.
- K-quant formats: **d-last layout**. Scales and deltas are at the END of each block, not the beginning.
- TurboQuant formats: **d-first layout** for standalone dequant, **d-last convention** for fused GEMM. Weights first, scales/zero_points last.
- Alignment: All buffers 128-bit (16-byte) aligned minimum.

### Push Constants
- All push constant structs must be declared in `include/rdna4_types.hpp` with matching GLSL layout.
- Host struct uses `uint32_t` / `uint64_t` / `float`. No padding assumptions — use explicit `alignas` or manual padding fields.
- Max push constant size: 128 bytes (Vulkan minimum). Stay under this.

### Performance Rules
- Each shader must be self-contained. No dependencies on host-staged F32 data.
- Use subgroup shuffle/broadcast for scale sharing within a warp.
- Avoid barriers where possible. If needed, use `memoryBarrierBuffer()` + `barrier()`.
- Tile size: 32×32 or 64×64 for GEMM. 128 threads for element-wise ops.

## Work Guidance

### Adding a New Shader
1. Write `.comp` file in `src/kernels/`.
2. Add to `CMakeLists.txt` under `set(SHADER_SOURCES ...)`.
3. Add push constant struct to `include/rdna4_types.hpp`.
4. Add `loadPipe(name, sizeof(...))` in `main.cpp` and `test_inference.cpp`.
5. Build and verify `glslc` compiles cleanly: `cmake --build . --config Release`
6. Copy `.spv` to `build/Release/shaders/`.

### Shader Checklist
- [ ] Compiles with `glslc --target-env=vulkan1.3 -fshader-stage=compute`
- [ ] No `sampler` or `image` types (compute-only device)
- [ ] Push constants match host struct exactly
- [ ] Buffer references aligned to 16 bytes
- [ ] F16 conversions explicit
- [ ] No shared memory > 16KB
- [ ] Subgroup extensions declared if used

## Verification
- Build: `cd build && cmake --build . --config Release`
- Shader copy: `Copy-Item build\shaders\*.spv build\Release\shaders\`
- Runtime: `test_inference.exe <model.json> <model.bin>` must not crash

## Compression Shaders

### compress_context.comp
- **Purpose**: Compress context embeddings using blockwise quantization
- **Input**: F32/F16 context buffer, optional importance scores
- **Output**: Quantized context buffer (packed weights + scales + optional zero points)
- **Block layout**: Packed weights first, then scale (1 or 2 bytes), then optional zero point (1 byte)
- **Strategies**: uniform (equal blocks), entropy (higher precision for high-entropy regions), importance (skip compression for important tokens)

### kv_cache_quantize.comp
- **Purpose**: Quantize K and V caches per-block (Q4_0) using blockwise symmetric quantization
- **Input**: F16 K and V caches
- **Output**: Quantized K and V caches (uint[] packed data) + separate F16 scale buffers per block
- **Block layout**: 32 elements per block, 2 values per byte (4-bit each), scales stored in separate float16_t[] buffer
- **Notes**: Only the older tier of hierarchical KV cache is quantized. Recent tokens stay F16.

### kv_cache_dequant.comp
- **Purpose**: Dequantize K and V caches back to F16 for attention computation
- **Input**: Quantized KV cache (uint[]) + F16 scales per block (float16_t[])
- **Output**: F16 KV cache
- **Block layout**: Matches kv_cache_quantize.comp exactly — each uint holds 4 bytes × 2 nibbles = 8 elements
- **Notes**: Must be dispatched before attention.comp if KV cache is quantized. One thread per element.

## TurboQuant Shaders

### dequant_turbo.comp
- **Purpose**: Standalone dequantization of TurboQuant weights to F16.
- **Input**: Packed TQ weights (uint[] bytes)
- **Output**: F16 dequantized weights
- **Block layout**: `[scale: uint16_t fp16][packed_weights]`. TQ4 = 66B/128 elements, TQ6 = 50B/64 elements.
- **Notes**: TQ4 and TQ6 implemented; TQ3 is stubbed. Push constants match `rdna4_types.hpp::DequantTurboPushConstants`.

### gemm_turbo.comp
- **Purpose**: Fused GEMM over TurboQuant weights.
- **Input**: F16 activations (M × K), packed TQ weights (K × N)
- **Output**: F16 result (M × N)
- **Block layout**: Same d-first TQ block layout as `dequant_turbo.comp`.
- **Notes**: Tile size 32×4, shared activation tile. Push constants match `rdna4_types.hpp::GemmTurboPushConstants`.

## Child DOX Index
- None (leaf directory)
