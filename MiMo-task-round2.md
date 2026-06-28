# MiMo Task Assignment Round 2 — TurboQuant Host-Side + Validation

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

You already delivered TurboQuant shaders, schema, docs, and enum extensions (Round 1 — COMPLETE). Now you will deliver **host-side support** so Claude can wire TurboQuant into the engine without reverse-engineering block layouts.

### Deliverable 1: TurboQuant Host-Side Header Stub

**File**: `include/rdna4_turboquant.hpp`

Write a C++17 header stub that defines TurboQuant block layouts and host-side helpers. This is a declarations-only header — no `.cpp` implementation needed yet.

```cpp
#pragma once
#include <cstdint>
#include <cstddef>

namespace rdna4 {

// Block layout constants for each TurboQuant format
struct TurboQuantBlockLayout {
    uint32_t blockSize;      // Number of weights per block (64, 128, 256)
    uint32_t bitsPerWeight;  // 3, 4, 5, 6
    uint32_t scaleBits;      // 8 or 16
    uint32_t bytesPerBlock;  // Total block size in bytes
    uint32_t weightsOffset;  // Offset to packed weights (always 0)
    uint32_t scalesOffset;   // Offset to scales (end of weights)
    uint32_t zeroPointsOffset; // Offset to zero points (if enabled), or 0 if none
    uint32_t alignment;      // Required buffer alignment (16, 32, 64, 128)
};

// Predefined layouts
constexpr TurboQuantBlockLayout TQ4_128_LAYOUT = {
    128, 4, 16,
    66,   // 128*4/8 = 64 bytes weights + 2 bytes fp16 scale
    0, 64, 0, 128
};

constexpr TurboQuantBlockLayout TQ3_128_LAYOUT = {
    128, 3, 16,
    50,   // ceil(128*3/8)=48 bytes weights + 2 bytes fp16 scale
    0, 48, 0, 128
};

constexpr TurboQuantBlockLayout TQ6_64_LAYOUT = {
    64, 6, 16,
    50,   // 64*6/8 = 48 bytes weights + 2 bytes fp16 scale
    0, 48, 0, 128
};

// Helper: compute total bytes for N weights in a given format
size_t turboquantTensorBytes(uint64_t nWeights, const TurboQuantBlockLayout& layout);

// Helper: compute number of blocks for N weights
uint32_t turboquantBlockCount(uint64_t nWeights, uint32_t blockSize);

// Push constants for dequant_turbo.comp (must match GLSL exactly)
struct DequantTurboPushConstants {
    uint64_t addrSrc;
    uint64_t addrDst;
    uint32_t n;
    uint32_t blockSize;
    uint32_t bits;
    uint32_t scaleBits;
    uint32_t zeroPoint;
    float    scale;
};

// Push constants for gemm_turbo.comp (must match GLSL exactly)
struct GemmTurboPushConstants {
    uint64_t addrA;
    uint64_t addrB;
    uint64_t addrC;
    uint32_t M, K, N;
    uint32_t blockSize;
    uint32_t bits;
    uint32_t scaleBits;
    uint32_t zeroPoint;
    float    alpha;
};

} // namespace rdna4
```

**Requirements**:
- Use `#pragma once`
- No Vulkan includes needed
- Verify `sizeof(DequantTurboPushConstants) <= 128` bytes (Vulkan push constant minimum)
- Verify `sizeof(GemmTurboPushConstants) <= 128` bytes
- Add `static_assert(sizeof(...) <= 128, "Push constants too large");` for both structs

### Deliverable 2: Weight Converter Extension

**File**: `tools/weight_converter.py`

Extend the existing `weight_converter.py` to add TurboQuant conversion functions. You must read the existing file first to understand its structure, then append new functions.

**Functions to add** (append to the file):

```python
def convert_to_tq4(weights: list[float], block_size: int = 128) -> tuple[bytes, list[float]]:
    """Convert F32/F16 weights to TQ4_128 format.
    
    Returns:
        quantized_bytes: packed quantized weights + scales
        scales: list of per-block fp16 scales for validation
    """
    ...

def convert_to_tq3(weights: list[float], block_size: int = 128) -> tuple[bytes, list[float]]:
    """Convert F32/F16 weights to TQ3_128 format."""
    ...

def convert_to_tq6(weights: list[float], block_size: int = 64) -> tuple[bytes, list[float]]:
    """Convert F32/F16 weights to TQ6_64 format."""
    ...
```

**Algorithm for each format**:
1. Iterate weights in blocks of `block_size`
2. For each block:
   - Find `max_abs = max(|w|)`
   - Compute scale: `scale = max_abs / ((1 << bits) - 1)` (for unsigned) or `max_abs / ((1 << (bits-1)) - 1)` (for signed). Use **signed** quantization: `q = int(round(w / scale))` clamped to `[-(2^(bits-1)), 2^(bits-1)-1]`.
   - Write scale as little-endian fp16 (2 bytes) at end of block
   - Pack quantized values tightly into bytes
3. Return concatenated bytes + list of scales

**Packing rules**:
- TQ4: 2 weights per byte (low nibble first, then high nibble)
- TQ3: 8 weights per 3 bytes. Layout: bits[0-2] in byte0[0-2], bits[3-5] in byte0[3-5], bits[6-7] in byte0[6-7] + byte1[0] ... (document the exact bit layout in comments)
- TQ6: 4 weights per 3 bytes. Layout: bits[0-5] in byte0, bits[6-7] in byte0[6-7] + byte1[0-3], bits[8-13] in byte1[4-7] + byte2[0-1] ... (document exact layout)

**Guardrails**:
- Do NOT modify existing functions in `weight_converter.py` unless they have bugs.
- Append new functions at the end.
- Include a `__main__` test block that generates random weights, converts to TQ4/TQ6, and verifies round-trip error < 5%.

### Deliverable 3: TurboQuant Validation Script

**File**: `tools/validate_turboquant.py`

Write a standalone Python script that validates TurboQuant accuracy.

**Requirements**:
- Generate synthetic weights: random uniform, Gaussian, and a "spiky" distribution (99% small, 1% outliers)
- For each distribution, convert to TQ4_128, TQ3_128, TQ6_64 using the functions from `weight_converter.py`
- Dequantize back to F32 using a pure-Python reference dequant function
- Compute max absolute error and mean squared error
- Print a Markdown table:
  ```
  | Format | Distribution | Max Abs Error | MSE | Compression |
  |--------|-------------|---------------|-----|-------------|
  | TQ4_128 | uniform     | 0.0012        | ... | 8.00x       |
  ```
- Compression ratio = sizeof(float32) / bits_per_weight
- Exit code 0 if all MSE < 0.01, else exit 1

**Guardrails**:
- Pure Python, no numpy required (but use it if available)
- Self-contained: can run with `python tools/validate_turboquant.py`

### Deliverable 4: Update `include/rdna4_types.hpp` Push Constants

**File**: `include/rdna4_types.hpp`

Read the existing file. Add `DequantTurboPushConstants` and `GemmTurboPushConstants` structs in the same style as existing push constant structs (e.g., `DequantizePushConstants`, `GemmPushConstants`).

**Requirements**:
- Match the GLSL push constant layouts from `dequant_turbo.comp` and `gemm_turbo.comp`
- Use `uint32_t`, `uint64_t`, `float` only
- Add at the end of the file, before any closing namespace brace
- Do NOT modify existing structs

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
   - Exception: `turboquantTensorBytes` and `turboquantBlockCount` can be `inline` in the header since they're simple math.

5. **Do NOT break the build.**
   - Every new file must be added to `CMakeLists.txt`.
   - Every new shader must compile with `glslc`.

6. **Do NOT introduce heavy Python dependencies.**
   - No `torch`, `tensorflow`, `transformers` unless absolutely necessary.
   - Prefer stdlib + optional `numpy`.

## Verification Checklist

Before declaring done, verify:

- [ ] `include/rdna4_turboquant.hpp` compiles as C++17 (no errors with `cl /c /std:c++17 include/rdna4_turboquant.hpp` or equivalent)
- [ ] `sizeof(DequantTurboPushConstants) <= 128`
- [ ] `sizeof(GemmTurboPushConstants) <= 128`
- [ ] `tools/weight_converter.py` append works: `python -c "import weight_converter; w = [0.1]*256; b, s = weight_converter.convert_to_tq4(w); print(len(b), len(s))"`
- [ ] `tools/validate_turboquant.py` runs and prints the Markdown table
- [ ] `include/rdna4_types.hpp` still compiles (no broken existing structs)
- [ ] No existing source files were modified except `include/rdna4_types.hpp` (append only)

## Context Files You Should Read First

1. `include/rdna4_types.hpp` — See existing push constant structs
2. `tools/weight_converter.py` — Understand existing structure before appending
3. `src/kernels/dequant_turbo.comp` — Match push constants to GLSL
4. `src/kernels/gemm_turbo.comp` — Match push constants to GLSL
5. `AGENTS.md` (root) — Understand DOX framework

## Questions?

If you are unsure about:
- **C++ struct layout / padding**: Ask Kimi.
- **Python bit-packing logic**: Ask Kimi.
- **Project-specific conventions (DOX, AGENTS.md)**: Read the root `AGENTS.md` again.
- **Whether a change violates a guardrail**: Stop and ask Kimi before proceeding.

## Expected Time
- `rdna4_turboquant.hpp`: 30 min
- `weight_converter.py` extension: 2-3 hours
- `validate_turboquant.py`: 1 hour
- `rdna4_types.hpp` push constants: 30 min
- Total: ~4-5 hours
