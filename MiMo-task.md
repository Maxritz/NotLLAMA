# MiMo Task Assignment — NotLLAMA TurboQuant + Shader Infrastructure

## Project Context

**NotLLAMA** is a Vulkan compute-only inference engine for LLMs, targeting AMD RX 9070 XT (RDNA4, 16GB VRAM). It is built on the philosophy: **"Many small units doing work together."** Weights stay quantized on GPU. Each layer dequantizes its own weights on-demand. No monolithic pre-dequantized buffers. Shaders are self-contained: dequantize → process → output.

### Architecture
- **Host**: C++17, CMake, MSVC 19.44
- **GPU**: 14 GLSL compute shaders compiled to SPIR-V via `glslc`
- **Scheduler**: Batch-mode Vulkan queue submission with FencePool (64 fences)
- **Inference**: Two paths — `forwardPartial()` (layer-by-layer, ring allocator) and `forwardKernelEntry()` (single-dispatch persistent mailbox kernel)
- **Quantization**: 57 GGUF formats supported (Q1–Q8, IQ1–IQ6, Q2_K–Q8_K, F16, BF16, F32, MXFP4/6/8)
- **Key rule**: No buffer > 1GB. No pre-dequantization at init time.

### Repository
- Path: `C:\Users\rr\Desktop\Notllama-loc`
- GitHub: https://github.com/Maxritz/NotLLAMA (master, LGPL-2.1)
- Build: `cd build && cmake --build . --config Release`
- Shader copy: `Copy-Item build\shaders\*.spv build\Release\shaders\`

## Your Assignment

You are responsible for **TurboQuant** — a new family of GPU-native quantization formats — and **shader infrastructure** (child AGENTS.md). You will write headers, shaders, and JSON schemas. You will NOT modify existing engine logic (that is Claude's job after you deliver).

### Deliverable 1: TurboQuant JSON Schema

**File**: `docs/turboquant_schema.json`

Define the configuration schema for TurboQuant formats. Must include:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TurboQuantConfig",
  "type": "object",
  "required": ["version", "block_size", "bits_per_weight", "scale_bits"],
  "properties": {
    "version": { "type": "string", "enum": ["1.0"] },
    "block_size": { "type": "integer", "enum": [64, 128, 256] },
    "bits_per_weight": { "type": "number", "enum": [3.0, 4.0, 4.5, 5.0, 6.0] },
    "scale_bits": { "type": "integer", "enum": [8, 16] },
    "zero_point": { "type": "boolean", "default": false },
    "group_size": { "type": "integer", "minimum": 1, "default": 1 },
    "activation_aware": { "type": "boolean", "default": false },
    "outlier_threshold": { "type": "number", "default": 0.0 },
    "outlier_bits": { "type": "integer", "enum": [8, 16], "default": 8 },
    "alignment": { "type": "integer", "enum": [16, 32, 64, 128], "default": 128 }
  }
}
```

Also create `docs/turboquant_formats.md` documenting each format variant:
- **TQ4_128**: 4-bit, 128-element blocks, fp16 scale, 128-bit aligned. 0.5 bytes/weight. Target: memory-constrained inference.
- **TQ3_128**: 3-bit, 128-element blocks, fp16 scale. 0.375 bytes/weight. Target: extreme compression.
- **TQ6_64**: 6-bit, 64-element blocks, fp16 scale. 0.75 bytes/weight. Target: near-F16 quality.

### Deliverable 2: QuantFormat Enum Extension

**File**: `include/rdna4.hpp`

Add three new entries to the `QuantFormat` enum. Find the existing enum (search for `enum class QuantFormat` or `QuantFormat`). Add at the end:

```cpp
    TQ4_128,   // TurboQuant 4-bit, 128-element blocks
    TQ3_128,   // TurboQuant 3-bit, 128-element blocks
    TQ6_64,    // TurboQuant 6-bit, 64-element blocks
```

Do NOT change existing enum values. Append only.

### Deliverable 3: TurboQuant Dequantization Shader

**File**: `src/kernels/dequant_turbo.comp`

Write a GLSL compute shader that dequantizes TurboQuant weights on-the-fly. Requirements:

- **Version**: `#version 450` with `#extension GL_KHR_shader_subgroup_arithmetic : require`
- **Layout**: `layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;`
- **Push constants** (must match host struct later):
  ```glsl
  layout(push_constant) uniform Params {
      uint64_t addrSrc;      // Quantized weights buffer
      uint64_t addrDst;      // Dequantized F16 output buffer
      uint32_t n;            // Total number of weights
      uint32_t blockSize;    // 64, 128, or 256
      uint32_t bits;         // 3, 4, 5, or 6
      uint32_t scaleBits;    // 8 or 16
      uint32_t zeroPoint;    // 0 or 1
      float    scale;        // Global scale factor (optional)
  } pc;
  ```
- **Algorithm**:
  1. Each workgroup processes `blockSize` weights.
  2. Thread `i` in workgroup reads the scale for its block (stored at end of block, like K-quants: weights first, scales last).
  3. Thread `i` reads its quantized weight (3/4/5/6 bits) from the block.
  4. Dequantize: `float w = (float(quant) - (zeroPoint ? zp : 0)) * scale`
  5. Write `float16_t(w)` to `addrDst`.
- **Bit packing**: Weights are packed tightly (no padding). For 3-bit: 3 weights per byte (with 2 bits waste every 8 weights). For 4-bit: 2 weights per byte. For 5-bit: 5 weights per 32 bits (complex packing). For 6-bit: 4 weights per 3 bytes.
  - **Simplification for now**: Support 4-bit and 6-bit only in v1. 3-bit and 5-bit can be stubbed (commented) for v2.
- **Alignment**: Ensure `addrDst` is aligned to 16 bytes. Use `align16` in buffer reference if needed.
- **Buffer references**: Use `layout(buffer_reference, std430, buffer_reference_align = 16) buffer Float16Buffer { float16_t data[]; };`

**Guardrails**:
- Do NOT allocate any shared memory > 16KB.
- Do NOT use `float[]` arrays in push constants.
- Do NOT pre-dequantize the entire tensor — this shader is per-block.
- Subgroup operations are optional but recommended for scale broadcast.

### Deliverable 4: Fused GEMM + Dequant Shader

**File**: `src/kernels/gemm_turbo.comp`

Write a GLSL compute shader that fuses TurboQuant weight dequantization with GEMM. This is the performance-critical kernel.

- **Version**: `#version 450` with `#extension GL_KHR_shader_subgroup_arithmetic : require`
- **Layout**: `layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;` (128 threads total, 4 rows per workgroup)
- **Push constants**:
  ```glsl
  layout(push_constant) uniform Params {
      uint64_t addrA;        // Input activations (F16, M×K)
      uint64_t addrB;        // TurboQuant weights (K×N)
      uint64_t addrC;        // Output (F16, M×N)
      uint32_t M, K, N;
      uint32_t blockSize;    // TurboQuant block size
      uint32_t bits;         // 4 or 6
      uint32_t scaleBits;    // 16
      uint32_t zeroPoint;    // 0 or 1
      float    alpha;        // GEMM scale
  } pc;
  ```
- **Algorithm**:
  1. Workgroup computes a 4×32 tile of C (4 rows, 32 cols).
  2. Each thread computes 1 output element.
  3. For each K step:
     - Load activation `a = float(A[...])` into register.
     - Load quantized weight block, extract relevant weight, dequantize to `b`.
     - Accumulate `acc += a * b`.
  4. Write `float16_t(acc * pc.alpha)` to C.
- **Weight loading**: Use buffer reference to read quantized weights. Each thread reads its column's weight from the appropriate block.
- **Tiling**: Process K in tiles of `blockSize`. Load `blockSize` activations into shared memory once per tile, then iterate over the block's weights.

**Guardrails**:
- Shared memory for activations only: `shared float shA[128];` (4 rows × 32 cols). That's 512 bytes. OK.
- Do NOT put dequantized weights in shared memory — dequantize on-the-fly.
- Do NOT use `image` types — this is pure compute.
- If `bits == 4`, use simple nibble extraction. If `bits == 6`, use 4-weights-per-3-bytes unpacking.

### Deliverable 5: Shader Conventions AGENTS.md

**File**: `src/kernels/AGENTS.md`

Write a child AGENTS.md documenting shader conventions for this project. Use this exact structure:

```markdown
# src/kernels/ — Shader Conventions

## Purpose
GLSL compute shaders for LLM inference on RDNA4. Self-contained: dequantize → process → output.

## Ownership
Shaders are owned by the kernel pipeline. Host-side dispatch logic lives in `src/host/inference_engine.cpp` and `src/host/scheduler.cpp`.

## Local Contracts

### GLSL Requirements
- `#version 450` minimum. Use `#extension GL_KHR_shader_subgroup_arithmetic : require` for RDNA4 optimizations.
- `layout(local_size_x = ..., local_size_y = 1, local_size_z = 1) in;` unless fused GEMM uses 2D layout.
- All buffer references use `buffer_reference` with `buffer_reference_align = 16`.
- All push constants use `layout(push_constant) uniform Params { ... } pc;` naming.
- F16 I/O uses `float16_t`. Explicit conversion: `float16_t(x)` on write, `float(x)` on read.

### Data Layout
- Quantized weights: block-compressed. Block size varies by format.
- K-quant formats: **d-last layout**. Scales and deltas are at the END of each block, not the beginning.
- TurboQuant formats: **d-last layout** (match K-quant convention). Weights first, scales/zero_points last.
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
```

## Guardrails (What You Must NOT Do)

1. **Do NOT modify `src/host/inference_engine.cpp`, `src/host/scheduler.cpp`, or `main.cpp` dispatch logic.**
   - You may add `loadPipe` calls in comments showing where they should go, but do not change the actual code.
   - Claude will wire your shaders into the engine after you deliver.

2. **Do NOT change existing enum values.**
   - When adding `TQ4_128`, `TQ3_128`, `TQ6_64` to `QuantFormat`, append only. Do not renumber existing entries.

3. **Do NOT allocate > 1GB for any single buffer in shaders or host logic.**
   - This is a core project philosophy. If your shader design requires large shared memory, tile smaller.

4. **Do NOT design shaders that require host to precompute and stage all data before dispatch.**
   - Shaders must be self-contained. Host writes push constants and dispatches. Shader does the rest.

5. **Do NOT assume float32 weights are available to shaders.**
   - Shaders receive quantized data and dequantize on-demand.

6. **Do NOT write inline implementations in headers.**
   - Headers (`include/*.hpp`) contain declarations only. Implementations go in `src/host/*.cpp`.

7. **Do NOT break the build.**
   - Every new file must be added to `CMakeLists.txt`.
   - Every new shader must compile with `glslc`.

## Verification Checklist

| Check | Status | Notes |
|-------|--------|-------|
| `docs/turboquant_schema.json` is valid JSON schema | ✅ PASS | Valid JSON. Includes TQ4_128, TQ3_128, TQ6_64 block configs |
| `docs/turboquant_formats.md` documents all 3 format variants | ✅ PASS | TQ4_128 (66B/block), TQ3_128 (50B/block), TQ6_64 (50B/block). Block layouts, packing, dequant formulas, comparison table, v2 future |
| `include/rdna4.hpp` has TQ4_128, TQ3_128, TQ6_64 appended to QuantFormat | ✅ PASS | TQ4_128=57, TQ3_128=58, TQ6_64=59, COUNT=60. Existing values unchanged |
| `src/kernels/dequant_turbo.comp` compiles | ✅ PASS | `glslc --target-env=vulkan1.3` → exit 0. TQ4+TQ6 implemented, TQ3 stubbed (v2) |
| `src/kernels/gemm_turbo.comp` compiles | ✅ PASS | `glslc --target-env=vulkan1.3` → exit 0. 32×4 workgroup, fused TQ4/TQ6 dequant |
| `src/kernels/AGENTS.md` follows correct section order | ✅ PASS | Purpose → Ownership → Local Contracts → Work Guidance → Verification → Child DOX Index |
| No existing source files modified except `include/rdna4.hpp` | ✅ PASS | Only rdna4.hpp enum append + new files created |

### Compilation fixes applied
- `uint32_t` → `uint` in push constants (GLSL uses `uint`, not `uint32_t`)
- Dropped `GL_KHR_shader_subgroup_arithmetic` (not recognized by glslc). Subgroup shuffle/broadcast already available via Vulkan 1.3 core.
- Aligned to `#version 460` matching existing shaders (enables `GL_EXT_scalar_block_layout` without explicit extension)

## Context Files You Should Read First

Read these files before starting to understand the codebase style:

1. `include/rdna4_types.hpp` — See how push constants are declared (e.g., `DequantizePushConstants`, `GemmPushConstants`).
2. `src/kernels/dequantize.comp` — See how existing K-quant dequantization works (note: it has layout bugs that Claude is fixing; follow the pattern but trust the AGENTS.md d-last rule).
3. `src/kernels/gemm.comp` — See how GEMM is structured.
4. `AGENTS.md` (root) — Understand DOX framework and architecture philosophy.
5. `include/rdna4.hpp` — See `QuantFormat` enum and surrounding types.

## Questions?

If you are unsure about:
- **GLSL syntax or Vulkan compute constraints**: Ask Kimi.
- **Project-specific conventions (DOX, AGENTS.md)**: Read the root `AGENTS.md` again.
- **Whether a change violates a guardrail**: Stop and ask Kimi before proceeding.

## Expected Time
- JSON schema + docs: 30 min
- Enum extension: 5 min
- `dequant_turbo.comp`: 2-3 hours
- `gemm_turbo.comp`: 3-4 hours
- `src/kernels/AGENTS.md`: 30 min
- Total: ~6-8 hours
