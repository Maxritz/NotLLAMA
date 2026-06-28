# MiMo Task Assignment Round 3 — TurboQuant Benchmarking + Integration Guide + Test Stubs

## Project Context

**NotLLAMA** is a Vulkan compute-only inference engine for LLMs, targeting AMD RX 9070 XT (RDNA4, 16GB VRAM). It is built on the philosophy: **Many small units doing work together.** Weights stay quantized on GPU. Each layer dequantizes its own weights on-demand. No monolithic pre-dequantized buffers. Shaders are self-contained: dequantize → process → output.

### Architecture
- **Host**: C++17, CMake, MSVC 19.44
- **GPU**: 14 GLSL compute shaders compiled to SPIR-V via glslc
- **Scheduler**: Batch-mode Vulkan queue submission with FencePool (64 fences)
- **Inference**: Two paths — forwardPartial() (layer-by-layer, ring allocator) and forwardKernelEntry() (single-dispatch persistent mailbox kernel)
- **Quantization**: 57 GGUF formats supported (Q1–Q8, IQ1–IQ6, Q2_K–Q8_K, F16, BF16, F32, MXFP4/6/8)
- **Key rule**: No buffer > 1GB. No pre-dequantization at init time.

### Repository
- Path: C:\Users\rr\Desktop\Notllama-loc
- GitHub: https://github.com/Maxritz/NotLLAMA (master, LGPL-2.1)
- Build: cd build && cmake --build . --config Release
- Shader copy: Copy-Item build\shaders\*.spv build\Release\shaders\

## Your Assignment (Round 3)

You have delivered TurboQuant shaders, schema, docs, host-side structs, weight converter extension, and validation script (Rounds 1–2 — COMPLETE). Now you will deliver benchmarking tools, integration documentation, and C++ test stubs so Claude can validate TurboQuant end-to-end without guessing.

### Deliverable 1: TurboQuant Benchmarking Harness

**File**: tools/benchmark_turboquant.py

Write a Python script that benchmarks TurboQuant against existing GGUF formats on real weight data extracted from GGUF files.

**Requirements**:
- Read a GGUF file (use existing tools/gguf_loader.py — import it, do not copy its logic)
- For each weight tensor in the model:
  - Convert to TQ4_128, TQ3_128, TQ6_64 using weight_converter.py
  - Also keep the original GGUF quantization (Q4_0, Q8_0, etc.) if present
  - Dequantize all formats back to F32
  - Compute per-tensor metrics:
    - Max absolute error vs original F32
    - Mean squared error (MSE)
    - Signal-to-noise ratio (SNR) in dB: 10 * log10(sum(original^2) / sum((original - dequant)^2))
    - Compression ratio vs F32
- Print aggregate Markdown table:
  | Model | Tensor | Format | MSE | SNR (dB) | Compression | Speedup (theoretical) |
  |-------|--------|--------|-----|----------|-------------|----------------------|
- Theoretical speedup = (F32 bytes read) / (quantized bytes read + dequant overhead). Assume dequant overhead = 2x quant bytes for memory bandwidth bound.
- Save full results to benchmark_turboquant_results.json in the working directory

**Guardrails**:
- Do NOT require torch or transformers. Use stdlib + optional numpy.
- If numpy is not available, fall back to pure Python lists.
- Skip tensors > 100MB dequantized to avoid memory issues.
- Print progress every 10 tensors.

### Deliverable 2: GGUF-to-TurboQuant Standalone Converter

**File**: tools/convert_gguf_to_turboquant.py

Write a Python script that reads a GGUF file and outputs a TurboQuant-compatible model package (JSON metadata + binary weights).

**Requirements**:
- Input: path to .gguf file
- Output: two files in the same directory:
  - <model>_tq4.json — metadata with format: TQ4_128, tensor list with shapes, offsets, block sizes
  - <model>_tq4.bin — concatenated quantized tensor data
- For each F16/F32 tensor in the GGUF:
  - Convert to TQ4_128 (default). Allow --format tq3 or --format tq6 via argparse.
  - Write tensor entry to JSON with name, shape, format, blockSize, bits, offset, size
  - Append quantized bytes to .bin file
- For already-quantized tensors (Q4_0, Q8_0, etc.):
  - Dequantize to F32 first using gguf_loader.py dequant functions
  - Then re-quantize to TurboQuant format
- Print summary: total original size, total TurboQuant size, compression ratio

**Guardrails**:
- Use argparse for CLI: python convert_gguf_to_turboquant.py model.gguf --format tq4 --output-dir ./tq_models/
- Do NOT load entire model into RAM at once. Process tensors one at a time.
- Validate that output .bin size matches sum of all tensor size fields.

### Deliverable 3: TurboQuant C++ Test Stub

**File**: test/test_turboquant.cpp

Write a C++ test stub that validates include/rdna4_turboquant.hpp without needing a GPU.

**Requirements**:
```cpp
#include "rdna4_turboquant.hpp"
#include <cassert>
#include <cstdio>

int main() {
    using namespace rdna4;

    // Test 1: Block layout constants
    static_assert(TQ4_128_LAYOUT.blockSize == 128, "");
    static_assert(TQ4_128_LAYOUT.bitsPerWeight == 4, "");
    static_assert(TQ4_128_LAYOUT.bytesPerBlock == 66, "");

    // Test 2: Size helpers
    assert(turboquantBlockCount(256, 128) == 2);
    assert(turboquantBlockCount(300, 128) == 3);
    assert(turboquantTensorBytes(256, TQ4_128_LAYOUT) == 2 * 66);

    // Test 3: Push constant size limits
    static_assert(sizeof(DequantTurboPushConstants) <= 128, "");
    static_assert(sizeof(GemmTurboPushConstants) <= 128, "");

    // Test 4: Alignment
    assert(TQ4_128_LAYOUT.alignment == 128);
    assert(TQ6_64_LAYOUT.alignment == 128);

    printf("All TurboQuant tests passed.\n");
    return 0;
}
```

**Guardrails**:
- Do NOT include Vulkan headers.
- Do NOT modify existing test files (test_inference.cpp, test_cpu_ref.cpp).
- Must compile with: cl /std:c++17 /I include test/test_turboquant.cpp

### Deliverable 4: TurboQuant Integration Guide

**File**: docs/turboquant_integration.md

Write a Markdown guide for Claude explaining exactly how to wire TurboQuant into the engine.

**Required sections**:
1. **Overview**: What TurboQuant is and why it exists (faster than GGUF on RDNA4, blockwise uniform quantization)
2. **Block Layouts**: Byte-level diagrams for TQ4_128, TQ3_128, TQ6_64 showing weights offset, scale offset, alignment padding
3. **Shader Dispatch Guide**: For each shader (dequant_turbo.comp, gemm_turbo.comp):
   - Required workgroup size (local_size_x)
   - Push constant fields and their meanings
   - Buffer bindings (set/binding numbers)
   - Example dispatch call (number of workgroups)
4. **Host Wiring Checklist**: Step-by-step for Claude:
   - [ ] Add TQ4_128/TQ3_128/TQ6_64 cases to weight uploader
   - [ ] Allocate dequant staging buffer with turboquantTensorBytes()
   - [ ] Dispatch dequant_turbo.comp before GEMM
   - [ ] Dispatch gemm_turbo.comp with GemmTurboPushConstants
   - [ ] Add format string to model metadata JSON
5. **Performance Notes**: Expected memory bandwidth savings vs Q4_0, vs Q8_0. Expected dequant overhead.
6. **Fallback Strategy**: If TurboQuant shader fails to compile or produces wrong results, fall back to Q4_0 or Q8_0.

**Guardrails**:
- Keep under 200 lines.
- Use ASCII art for byte layout diagrams.
- Do not assume Claude knows TurboQuant internals — explain everything.

## Guardrails (What You Must NOT Do)

1. Do NOT modify src/host/inference_engine.cpp, src/host/scheduler.cpp, or main.cpp.
   - You may add comments showing integration points, but do not change existing logic.
   - Claude will wire your code into the engine after delivery.

2. Do NOT change existing enum values or struct layouts.
   - Append new types only. Do not renumber.

3. Do NOT allocate > 1GB for any single buffer in shaders or host logic.
   - If your design needs more, tile it or document the limitation.

4. Do NOT write inline implementations in headers (except trivial constexpr getters).
   - include/*.hpp = declarations + constexpr data. src/host/*.cpp = implementations.

5. Do NOT break the build.
   - Every new file must be added to CMakeLists.txt.
   - Every new shader must compile with glslc.

6. Do NOT introduce heavy Python dependencies.
   - No torch, tensorflow, transformers unless absolutely necessary.
   - Prefer stdlib + optional numpy.

## Verification Checklist

Before declaring done, verify:

- [ ] tools/benchmark_turboquant.py runs on a real GGUF file and produces benchmark_turboquant_results.json
- [ ] tools/convert_gguf_to_turboquant.py runs and produces valid .json + .bin files
- [ ] test/test_turboquant.cpp compiles and prints All TurboQuant tests passed.
- [ ] docs/turboquant_integration.md is under 200 lines and contains all 6 required sections
- [ ] No existing source files were modified except appends to weight_converter.py (already done in Round 2)

## Context Files You Should Read First

1. include/rdna4_turboquant.hpp — Your Round 2 deliverable
2. tools/weight_converter.py — Your Round 2 extension
3. tools/gguf_loader.py — For importing dequant functions
4. src/kernels/dequant_turbo.comp — Shader to document
5. src/kernels/gemm_turbo.comp — Shader to document
6. AGENTS.md (root) — Understand DOX framework

## Questions?

If you are unsure about:
- Python GGUF parsing / tensor iteration: Ask Kimi.
- C++ static_assert / constexpr math: Ask Kimi.
- Project-specific conventions (DOX, AGENTS.md): Read the root AGENTS.md again.
- Whether a change violates a guardrail: Stop and ask Kimi before proceeding.

## Expected Time
- benchmark_turboquant.py: 2-3 hours
- convert_gguf_to_turboquant.py: 2 hours
- test/test_turboquant.cpp: 30 min
- docs/turboquant_integration.md: 1 hour
- Total: ~5-6 hours
