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

## Child DOX Index
- None (leaf directory)
