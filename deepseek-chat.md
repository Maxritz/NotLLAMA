# DeepSeek Task Assignment — NotLLAMA Compression + GraphifyClient + Tooling Standards

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

You are responsible for **context compression**, **KV cache compression**, **GraphifyClient** header stub, and **Python tooling standards** (child AGENTS.md). You will write headers, shaders, JSON schemas, and documentation. You will NOT modify existing engine logic (that is Claude's job after you deliver).

### Deliverable 1: Context Compression JSON Schema

**File**: `docs/context_compression_schema.json`

Define the runtime configuration schema for context compression. Must include:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ContextCompressionConfig",
  "type": "object",
  "required": ["enabled", "trigger_threshold_percent", "target_percent", "strategy"],
  "properties": {
    "enabled": { "type": "boolean", "default": true },
    "trigger_threshold_percent": {
      "type": "number",
      "minimum": 50,
      "maximum": 100,
      "default": 85,
      "description": "When context usage exceeds this %, trigger compression"
    },
    "target_percent": {
      "type": "number",
      "minimum": 10,
      "maximum": 90,
      "default": 50,
      "description": "Compress down to this % of max context"
    },
    "strategy": {
      "type": "string",
      "enum": ["sliding_window", "half_slide", "fifo", "importance", "summary"],
      "default": "importance"
    },
    "strategies": {
      "type": "object",
      "properties": {
        "sliding_window": {
          "type": "object",
          "properties": {
            "window_size": { "type": "integer", "minimum": 64, "default": 1024 },
            "stride": { "type": "integer", "minimum": 1, "default": 512 }
          }
        },
        "half_slide": {
          "type": "object",
          "properties": {
            "preserve_recent": { "type": "integer", "minimum": 64, "default": 512 }
          }
        },
        "fifo": {
          "type": "object",
          "properties": {
            "drop_percent": { "type": "number", "minimum": 1, "maximum": 99, "default": 50 }
          }
        },
        "importance": {
          "type": "object",
          "properties": {
            "scorer": {
              "type": "string",
              "enum": ["attention_entropy", "gradient_magnitude", "token_frequency"],
              "default": "attention_entropy"
            },
            "keep_top_percent": { "type": "number", "minimum": 1, "maximum": 99, "default": 50 }
          }
        },
        "summary": {
          "type": "object",
          "properties": {
            "summarizer_model": { "type": "string", "enum": ["internal", "external"], "default": "internal" },
            "compress_ratio": { "type": "number", "minimum": 0.01, "maximum": 1.0, "default": 0.3 }
          }
        }
      }
    }
  }
}
```

Also create `docs/context_compression.md` explaining each strategy:
- **SLIDING_WINDOW**: Keep only the last `window_size` tokens. Fastest. Good for streaming.
- **HALF_SLIDE**: Drop first half of context, preserve recent half. Simple, no scoring needed.
- **FIFO**: Drop oldest tokens first. Predictable but may lose important early context.
- **IMPORTANCE**: Score each token position by attention entropy (how much each token attends to others). Keep highest-scoring positions. Best quality, requires attention scores.
- **SUMMARY**: Run a lightweight summarization on dropped tokens, insert summary tokens. Most complex, best retention of semantic content.

### Deliverable 2: Context Compression Shader

**File**: `src/kernels/compress_context.comp`

Write a GLSL compute shader that compacts the KV cache based on a `keep_mask`.

- **Version**: `#version 450` with `#extension GL_KHR_shader_subgroup_arithmetic : require`
- **Layout**: `layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;`
- **Push constants**:
  ```glsl
  layout(push_constant) uniform Params {
      uint64_t addrKSrc;      // Source K cache (F16, seqLen × nHeads × headDim)
      uint64_t addrVSrc;      // Source V cache (F16, seqLen × nHeads × headDim)
      uint64_t addrKDst;      // Destination K cache (F16)
      uint64_t addrVDst;      // Destination V cache (F16)
      uint64_t addrMask;      // keep_mask (uint8_t, 1 = keep, 0 = drop)
      uint32_t seqLen;        // Current sequence length
      uint32_t nHeads;        // Number of KV heads
      uint32_t headDim;       // Head dimension (e.g., 128)
      uint32_t targetLen;     // Expected output sequence length (sum of keep_mask)
  } pc;
  ```
- **Algorithm**:
  1. Each thread processes one head position (one element of K or V).
  2. Compute prefix sum of `keep_mask` to find destination index for each kept token.
     - Use subgroup scan for parallel prefix sum within workgroup.
     - For cross-workgroup prefix sum, use a two-pass approach (local scan + global offsets).
     - **Simplification for v1**: Assume `seqLen <= 2048` and use a single workgroup with shared memory for the prefix sum. Document this limitation.
  3. If `keep_mask[token] == 1`, copy K[token][head][dim] and V[token][head][dim] to `dst[prefix_sum(token)][head][dim]`.
- **Buffer references**:
  ```glsl
  layout(buffer_reference, std430, buffer_reference_align = 16) buffer Float16Buffer { float16_t data[]; };
  layout(buffer_reference, std430, buffer_reference_align = 16) buffer Uint8Buffer { uint8_t data[]; };
  ```

**Guardrails**:
- Shared memory for prefix sum: `shared uint shPrefix[2048];` — this is 8KB. OK.
- If `seqLen > 2048`, document that multi-workgroup prefix sum is needed in v2.
- Do NOT use `atomicAdd` for prefix sum — it is too slow for this. Use subgroup scan + shared memory.

### Deliverable 3: KV Cache Compression JSON Schema

**File**: `docs/kv_compression_schema.json`

Define the runtime configuration schema for KV cache compression:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KVCacheCompressionConfig",
  "type": "object",
  "required": ["enabled", "method"],
  "properties": {
    "enabled": { "type": "boolean", "default": false },
    "method": {
      "type": "string",
      "enum": ["quantize", "sparsify", "low_rank", "hierarchical"],
      "default": "quantize"
    },
    "methods": {
      "type": "object",
      "properties": {
        "quantize": {
          "type": "object",
          "properties": {
            "format": { "type": "string", "enum": ["Q4_0", "Q5_0", "Q8_0"], "default": "Q4_0" },
            "compress_every_n_layers": { "type": "integer", "minimum": 1, "default": 1 },
            "per_head_scales": { "type": "boolean", "default": true }
          }
        },
        "sparsify": {
          "type": "object",
          "properties": {
            "threshold": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.01 },
            "prune_heads": { "type": "array", "items": { "type": "integer" }, "default": [] }
          }
        },
        "low_rank": {
          "type": "object",
          "properties": {
            "rank": { "type": "integer", "minimum": 8, "default": 64 },
            "update_every_n_tokens": { "type": "integer", "minimum": 1, "default": 512 }
          }
        },
        "hierarchical": {
          "type": "object",
          "properties": {
            "recent_n": { "type": "integer", "minimum": 64, "default": 1024 },
            "recent_format": { "type": "string", "enum": ["F16", "F32"], "default": "F16" },
            "older_format": { "type": "string", "enum": ["Q4_0", "Q5_0", "Q8_0"], "default": "Q4_0" }
          }
        }
      }
    }
  }
}
```

Also create `docs/kv_compression.md` documenting each method:
- **Quantize**: Per-head block quantization. Simple, good compression (2-4×), small accuracy loss.
- **Sparsify**: Zero out small values. Only useful with sparse attention kernels (which we don't have yet). Mark as "future".
- **Low-Rank**: Factorize K and V into smaller matrices. Complex to update incrementally. Mark as "research".
- **Hierarchical**: Recent tokens in F16 (fast access), older tokens in Q4_0 (compressed). Best tradeoff. Recommended default once implemented.

### Deliverable 4: KV Cache Quantization Shader

**File**: `src/kernels/kv_cache_quantize.comp`

Write a GLSL compute shader that quantizes the KV cache in-place or to a separate buffer.

- **Version**: `#version 450`
- **Layout**: `layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;`
- **Push constants**:
  ```glsl
  layout(push_constant) uniform Params {
      uint64_t addrKSrc;      // Source K cache (F16)
      uint64_t addrVSrc;      // Source V cache (F16)
      uint64_t addrKDst;      // Destination quantized K
      uint64_t addrVDst;      // Destination quantized V
      uint64_t addrKScales;   // K scales (float16_t per head per block)
      uint64_t addrVScales;   // V scales
      uint32_t seqLen;
      uint32_t nHeads;
      uint32_t headDim;
      uint32_t blockSize;     // Quantization block size (e.g., 32)
  } pc;
  ```
- **Algorithm**:
  1. Each workgroup processes one head. Each thread processes one block of `blockSize` tokens.
  2. For the block, find `max_abs = max(|val|)` across all elements in the block.
  3. Compute scale: `scale = max_abs / 7.0` (for 4-bit) or `/ 15.0` (for 5-bit) or `/ 127.0` (for 8-bit).
  4. Write scale to `addrKScales[head * nBlocks + blockIdx]`.
  5. Quantize each element: `q = int8_t(round(val / scale))` (or `int4_t`, `int8_t`).
  6. Pack quantized values tightly and write to `addrKDst`.

**Guardrails**:
- Support Q4_0 only for v1. Q5_0 and Q8_0 can be stubbed.
- For Q4_0: 2 values per byte. Block size should be 32 (16 bytes of weights + 2 bytes scale = 18 bytes, align to 32 for simplicity).
- Do NOT quantize the most recent `recent_n` tokens if using hierarchical mode — this shader is for the "older" tier only. Document this.

### Deliverable 5: GraphifyClient Header Stub

**File**: `include/rdna4_graphify.hpp`

Write a C++ header stub for a client that queries the local graphify knowledge graph at runtime.

```cpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace rdna4 {

// Result of a graph query
struct GraphQueryResult {
    std::string question;
    std::string answer;
    std::vector<std::string> sourceNodes;      // Node IDs cited
    std::vector<std::string> sourceLocations;  // file:line references
    float confidence = 0.0f;
    uint64_t queryTimeMs = 0;
};

// Configuration for Graphify integration
struct GraphifyConfig {
    std::string graphPath = "graphify-out/graph.json";
    bool autoUpdate = true;
    bool queryBeforeRead = true;
    std::string preferredMode = "dfs";  // "bfs" or "dfs"
    uint32_t tokenBudget = 1500;
    bool budgetCap = true;
    uint32_t cacheSize = 32;  // LRU cache size
};

class GraphifyClient {
public:
    explicit GraphifyClient(const GraphifyConfig& cfg = {});
    ~GraphifyClient();

    // Query the knowledge graph. Returns cached result if available.
    GraphQueryResult query(const std::string& question);

    // Check if the graph is stale (files modified since last build)
    bool isStale() const;

    // Trigger incremental graph update (runs `graphify update`)
    bool updateGraph();

    // Get nodes related to a symbol (e.g., "dequantize_q6_k")
    std::vector<std::string> getRelatedNodes(const std::string& symbol, int depth = 2);

    // Clear the in-memory query cache
    void clearCache();

    // Static: check if graphify CLI is available in PATH
    static bool isAvailable();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace rdna4
```

**Notes**:
- This is a header-only stub. No `.cpp` implementation needed — Claude will write it later.
- Include `<memory>` for `std::unique_ptr`.
- Keep it simple. No Vulkan dependencies.

### Deliverable 6: Python Tooling Standards AGENTS.md

**File**: `tools/AGENTS.md`

Write a child AGENTS.md documenting Python tooling standards. Use this exact structure:

```markdown
# tools/ — Python Tooling Standards

## Purpose
Python scripts for model conversion, weight inspection, GGUF parsing, and RAG tools.

## Ownership
Python tools are owned by the build/CI pipeline. They are standalone scripts, not imported by the C++ runtime.

## Local Contracts

### Python Requirements
- Python 3.10+ required.
- Use `argparse` for CLI entry points.
- Use `typing` annotations (Python 3.9+ style: `list[int]`, `dict[str, float]`).
- No external dependencies unless strictly necessary. Prefer stdlib.
- If numpy is needed, guard the import and provide a pure-Python fallback for basic operations.

### Script Structure
```python
#!/usr/bin/env python3
"""One-line description."""
import argparse
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("-o", "--output", type=Path, default=Path("out.json"))
    args = parser.parse_args()
    # ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### GGUF / Quantization Rules
- All K-quant formats use **d-last layout**: quantized weights first, scales/deltas at the END of each block.
- Q2_K block size: 84 bytes (scales[16] + qs[64] + d + dmin)
- Q3_K block size: 110 bytes (hmask[32] + qs[64] + scales[12] + d)
- Q4_K block size: 144 bytes (qs[128] + scales[12] + d + dmin)
- Q5_K block size: 176 bytes (qs[128] + qh[32] + scales[12] + d + dmin)
- Q6_K block size: 210 bytes (ql[128] + qh[64] + scales[16] + d)
- Q8_K block size: 292 bytes (qs[256] + scales[32] + d + dmin)
- Nibble extraction in Q4 formats: alternate low/high nibble. `index 0 -> low, index 1 -> high, index 2 -> low, ...`
- Validation functions must check `d` (delta/scale) at the **correct end-of-block offset**, not offset 0.

### Error Handling
- Use `try/except` around file I/O. Print clear error messages to stderr.
- Validate file sizes before allocating buffers.
- Check `is_open()` on all streams before reading.
- Return non-zero exit codes on failure.

## Work Guidance

### Adding a New Tool
1. Create script in `tools/`.
2. Add to root `AGENTS.md` Child DOX Index under `tools/`.
3. Document usage in script docstring.
4. If the tool produces JSON output, validate against a schema in `docs/`.

### Tool Checklist
- [ ] Has `#!/usr/bin/env python3` shebang
- [ ] Has `if __name__ == "__main__"` guard
- [ ] Returns `int` from `main()`
- [ ] Uses `argparse`
- [ ] Handles `FileNotFoundError`, `PermissionError`
- [ ] Validates inputs before processing
- [ ] Prints to stdout, errors to stderr

## Verification
- Run script with `--help` to verify argparse works.
- Run on a small test file (e.g., a 1-tensor GGUF) to verify basic functionality.

## Child DOX Index
- None (leaf directory)
```

## Guardrails (What You Must NOT Do)

1. **Do NOT modify `src/host/inference_engine.cpp`, `src/host/scheduler.cpp`, or `main.cpp`.**
   - You may add comments showing integration points, but do not change existing logic.
   - Claude will wire your code into the engine after delivery.

2. **Do NOT change existing enum values or struct layouts.**
   - Append new types only. Do not renumber.

3. **Do NOT allocate > 1GB for any single buffer in shaders or host logic.**
   - If your shader design needs more, tile it or document the limitation.

4. **Do NOT design shaders that require host to precompute and stage all data before dispatch.**
   - Host writes push constants and masks. Shader compacts.

5. **Do NOT write inline implementations in headers.**
   - `include/*.hpp` = declarations only. `src/host/*.cpp` = implementations.
   - Exception: `GraphifyClient` is explicitly a stub header — no `.cpp` needed yet.

6. **Do NOT break the build.**
   - Every new file must be added to `CMakeLists.txt`.
   - Every new shader must compile with `glslc`.

7. **Do NOT introduce heavy Python dependencies.**
   - No `torch`, `tensorflow`, `transformers` unless absolutely necessary.
   - Prefer stdlib + optional `numpy`.

## Verification Checklist

Before declaring done, verify:

- [ ] `docs/context_compression_schema.json` is valid JSON schema
- [ ] `docs/context_compression.md` explains all 5 strategies
- [ ] `docs/kv_compression_schema.json` is valid JSON schema
- [ ] `docs/kv_compression.md` explains all 4 methods
- [ ] `src/kernels/compress_context.comp` compiles: `glslc --target-env=vulkan1.3 -fshader-stage=compute src/kernels/compress_context.comp -o /tmp/compress_context.spv`
- [ ] `src/kernels/kv_cache_quantize.comp` compiles: `glslc --target-env=vulkan1.3 -fshader-stage=compute src/kernels/kv_cache_quantize.comp -o /tmp/kv_cache_quantize.spv`
- [ ] `include/rdna4_graphify.hpp` is valid C++17 (no Vulkan deps, clean header)
- [ ] `tools/AGENTS.md` follows the exact section order: Purpose, Ownership, Local Contracts, Work Guidance, Verification, Child DOX Index
- [ ] No existing source files were modified

## Context Files You Should Read First

Read these files before starting to understand the codebase style:

1. `include/rdna4_types.hpp` — See how push constants are declared.
2. `include/rdna4_context_manager.hpp` — See `ContextManager` and `CompressionStrategy` enum.
3. `include/rdna4_kv_cache.hpp` — See `KVCacheManager` interface.
4. `src/kernels/dequantize.comp` — See buffer reference and push constant patterns.
5. `AGENTS.md` (root) — Understand DOX framework.
6. `tools/gguf_loader.py` — See Python style and GGUF parsing patterns.

## Questions?

If you are unsure about:
- **GLSL syntax or Vulkan compute constraints**: Ask Kimi.
- **Project-specific conventions (DOX, AGENTS.md)**: Read the root `AGENTS.md` again.
- **Whether a change violates a guardrail**: Stop and ask Kimi before proceeding.

## Expected Time
- JSON schemas + docs: 1 hour
- `compress_context.comp`: 3-4 hours
- `kv_cache_quantize.comp`: 2-3 hours
- `rdna4_graphify.hpp`: 30 min
- `tools/AGENTS.md`: 30 min
- Total: ~7-9 hours
