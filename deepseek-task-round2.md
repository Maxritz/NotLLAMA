# DeepSeek Task Assignment Round 2 — KV Dequant + Compression Host-Side + DOX Lint

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

## Your Assignment (Round 2)

You already delivered compression schemas, shaders, GraphifyClient stub, and tools AGENTS.md (Round 1 — COMPLETE). Now you will deliver the **matching dequant shader**, **host-side compression structs**, and a **DOX lint tool** so Claude can wire context/KV compression into the engine.

### Deliverable 1: KV Cache Dequantization Shader

**File**: `src/kernels/kv_cache_dequant.comp`

Write the matching dequantization shader for `kv_cache_quantize.comp`. If the KV cache is quantized per-token or per-head, this shader restores F16/F32 for attention computation.

**GLSL Requirements**:
```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16  : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint64_t addrSrc;   // Quantized KV cache
    uint64_t addrDst;   // F16 output buffer
    uint32_t n;         // Number of elements (tokens × heads × headDim)
    uint32_t blockSize; // Elements per block (64, 128)
    uint32_t bits;      // 4, 5, 6, 8
    uint32_t scaleBits; // 8 or 16
    uint32_t zeroPoint; // 0 or 1
} pc;

layout(set = 0, binding = 0) buffer BufferA { uint8_t data[]; } src;
layout(set = 0, binding = 1) buffer BufferB { float16_t data[]; } dst;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.n) return;

    // Read block index, offset within block
    uint blockIdx = idx / pc.blockSize;
    uint blockOff = idx % pc.blockSize;

    // Compute block base address in src
    // Layout per block: [packed_weights][scale (2 bytes if fp16, 1 if int8)][optional_zero_point]
    // You must define the exact byte layout and compute:
    //   scale = read_scale(blockBase)
    //   zp    = read_zero_point(blockBase) if zeroPoint==1 else 0.0
    //   q     = read_quantized_value(blockBase, blockOff, pc.bits)
    //   deq   = (q - zp) * scale
    // Write deq to dst[idx]

    // Implementation must support bits=4,5,6,8 with blockSize=64 or 128
}
```

**Requirements**:
- Support `bits = 4, 5, 6, 8` (unsigned blockwise quantization)
- Support `blockSize = 64, 128`
- Support `scaleBits = 8` (int8 scale, treated as `scale / 127.0`) and `16` (fp16 scale)
- Support `zeroPoint = 0` (no zero point) and `1` (int8 zero point stored after scale)
- If `scaleBits == 8`, scale is stored as 1 byte after packed weights
- If `scaleBits == 16`, scale is stored as 2 bytes (little-endian fp16) after packed weights
- If `zeroPoint == 1`, zero point is 1 byte stored immediately after scale
- Use `memoryBarrierBuffer()` if writing to dst is consumed by another shader in same dispatch (not needed here since it's a separate dispatch)
- Compile with `glslc --target-env=vulkan1.4 -O src/kernels/kv_cache_dequant.comp -o build/shaders/kv_cache_dequant.spv`

### Deliverable 2: Compression Host-Side Header

**File**: `include/rdna4_compression.hpp`

Write a C++17 header stub defining host-side structs for context compression and KV cache compression configurations.

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace rdna4 {

// Context compression configuration (matches docs/context_compression_schema.json)
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

// Push constants for compress_context.comp
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

// Push constants for kv_cache_quantize.comp
struct KVCacheQuantizePushConstants {
    uint64_t addrKSrc;
    uint64_t addrVSrc;
    uint64_t addrKDst;
    uint64_t addrVDst;
    uint32_t nTokens;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t kBits;
    uint32_t vBits;
    uint32_t scaleBits;
    uint32_t zeroPoint;
    uint32_t quantStrategy;
};

// Push constants for kv_cache_dequant.comp
struct KVCacheDequantPushConstants {
    uint64_t addrSrc;
    uint64_t addrDst;
    uint32_t n;
    uint32_t blockSize;
    uint32_t bits;
    uint32_t scaleBits;
    uint32_t zeroPoint;
};

} // namespace rdna4
```

**Requirements**:
- Use `#pragma once`
- Add `static_assert(sizeof(CompressContextPushConstants) <= 128, "");`
- Add `static_assert(sizeof(KVCacheQuantizePushConstants) <= 128, "");`
- Add `static_assert(sizeof(KVCacheDequantPushConstants) <= 128, "");`
- No Vulkan includes needed
- Document byte layout assumptions in comments (e.g., "scale follows packed weights")

### Deliverable 3: DOX Lint Tool

**File**: `tools/dox_lint.py`

Write a Python script that validates DOX compliance across the repository.

**Checks to implement**:

1. **AGENTS.md existence**: Every directory under `src/` and `include/` that contains `.cpp`, `.hpp`, or `.comp` files must have an `AGENTS.md` file OR be listed in a parent `AGENTS.md` Child DOX Index.

2. **AGENTS.md required sections**: Each `AGENTS.md` must contain (case-insensitive):
   - `Purpose`
   - `Ownership`
   - `Work Guidance` OR `Local Contracts`
   - `Verification`
   - `Child DOX Index` OR `## Child`

3. **Push constant size check**: For every `struct` in `include/rdna4_types.hpp`, `include/rdna4_turboquant.hpp`, and `include/rdna4_compression.hpp` whose name ends with `PushConstants` or `PushConstant`, verify `sizeof` would be <= 128 bytes. Since we can't compile, approximate by summing field sizes: `uint32_t=4`, `uint64_t=8`, `float=4`. Assume natural alignment (no padding for simplicity, but warn if padding might push it over).

4. **Shader compilation check**: Verify every `.comp` file in `src/kernels/` has a corresponding `.spv` file in `build/shaders/`. List missing ones.

5. **TODO/FIXME tracker**: Count `TODO` and `FIXME` occurrences in `src/` and `include/`. Report files with > 5 occurrences.

6. **File size check**: Report any single source file > 5000 lines.

**Output format**:
Print Markdown report to stdout:
```
# DOX Lint Report

## AGENTS.md Coverage
| Directory | Has AGENTS.md | Listed in Parent |
|-----------|---------------|------------------|
| src/host  | YES           | N/A              |
| src/kernels | YES         | N/A              |
...

## AGENTS.md Sections
| File | Purpose | Ownership | Work Guidance | Verification | Child DOX Index |
|------|---------|-----------|---------------|--------------|-----------------|
...

## Push Constant Size Check
| Struct | Estimated Size | Status |
|--------|---------------|--------|
| DequantizePushConstants | 32 | OK |
...

## Shader Compilation Check
| Shader | SPIR-V Exists |
|--------|---------------|
| dequantize.comp | YES |
| kv_cache_dequant.comp | NO |
...

## TODO/FIXME Summary
| File | TODO | FIXME |
|------|------|-------|
...

## Large Files
| File | Lines |
|------|-------|
...

## Exit Code
0 = all critical checks pass, 1 = any critical failure (missing AGENTS.md, push constant > 128, missing SPIR-V for committed shader)
```

**Requirements**:
- Pure Python, stdlib only (no external deps)
- Can run with `python tools/dox_lint.py`
- Exit code 0 on pass, 1 on any critical failure

### Deliverable 4: Update `src/kernels/AGENTS.md`

**File**: `src/kernels/AGENTS.md`

Read the existing file. Append a new section documenting the compression shaders.

**Section to append**:
```markdown
## Compression Shaders

### compress_context.comp
- **Purpose**: Compress context embeddings using blockwise quantization
- **Input**: F32/F16 context buffer, optional importance scores
- **Output**: Quantized context buffer (packed weights + scales + optional zero points)
- **Block layout**: Packed weights first, then scale (1 or 2 bytes), then optional zero point (1 byte)
- **Strategies**: uniform (equal blocks), entropy (higher precision for high-entropy regions), importance (skip compression for important tokens)

### kv_cache_quantize.comp
- **Purpose**: Quantize K and V caches per-token, per-head, or globally
- **Input**: F16 K and V caches
- **Output**: Quantized K and V caches
- **Block layout**: Same as compress_context.comp

### kv_cache_dequant.comp
- **Purpose**: Dequantize K and V caches back to F16 for attention computation
- **Input**: Quantized KV cache
- **Output**: F16 KV cache
- **Notes**: Must be dispatched before attention.comp if KV cache is quantized.
```

**Requirements**:
- Append after existing content
- Do not delete existing sections
- Update Child DOX Index if needed (add new shader docs)

## Guardrails (What You Must NOT Do)

1. **Do NOT modify `src/host/inference_engine.cpp`, `src/host/scheduler.cpp`, or `main.cpp`.**
   - You may add comments showing integration points, but do not change existing logic.
   - Claude will wire your code into the engine after delivery.

2. **Do NOT change existing enum values or struct layouts.**
   - Append new types only. Do not renumber.

3. **Do NOT allocate > 1GB for any single buffer in shaders or host logic.**
   - If your design needs more, tile it or document the limitation.

4. **Do NOT write inline implementations in headers (except trivial constexpr getters).**
   - `include/*.hpp` = declarations + constexpr data. `src/host/*.cpp` = implementations.

5. **Do NOT break the build.**
   - Every new file must be added to `CMakeLists.txt`.
   - Every new shader must compile with `glslc`.

6. **Do NOT introduce heavy Python dependencies.**
   - No `torch`, `tensorflow`, `transformers` unless absolutely necessary.
   - Prefer stdlib + optional `numpy`.

## Verification Checklist

Before declaring done, verify:

- [ ] `src/kernels/kv_cache_dequant.comp` compiles: `glslc --target-env=vulkan1.4 -O src/kernels/kv_cache_dequant.comp -o build/shaders/kv_cache_dequant.spv`
- [ ] `include/rdna4_compression.hpp` compiles as C++17
- [ ] All push constant structs in `rdna4_compression.hpp` pass `static_assert(sizeof <= 128)`
- [ ] `tools/dox_lint.py` runs and produces the Markdown report
- [ ] `tools/dox_lint.py` correctly identifies missing `AGENTS.md` files
- [ ] `tools/dox_lint.py` correctly checks push constant sizes against existing structs in `rdna4_types.hpp`
- [ ] `src/kernels/AGENTS.md` updated with compression shader docs
- [ ] No existing source files were modified except `src/kernels/AGENTS.md` (append only) and `include/rdna4_compression.hpp` (new file)

## Context Files You Should Read First

1. `src/kernels/kv_cache_quantize.comp` — Match dequant to quant layout
2. `docs/kv_compression_schema.json` — Understand KV compression config
3. `docs/context_compression_schema.json` — Understand context compression config
4. `include/rdna4_types.hpp` — See existing push constant style
5. `src/kernels/AGENTS.md` — Understand existing shader docs before appending
6. `AGENTS.md` (root) — Understand DOX framework

## Questions?

If you are unsure about:
- **GLSL bit-extraction / dequantization math**: Ask Kimi.
- **C++ struct layout / padding**: Ask Kimi.
- **Python AST walking for push constant size estimation**: Ask Kimi.
- **Project-specific conventions (DOX, AGENTS.md)**: Read the root `AGENTS.md` again.
- **Whether a change violates a guardrail**: Stop and ask Kimi before proceeding.

## Expected Time
- `kv_cache_dequant.comp`: 2-3 hours
- `rdna4_compression.hpp`: 45 min
- `dox_lint.py`: 2-3 hours
- `AGENTS.md` update: 30 min
- Total: ~5-6 hours
