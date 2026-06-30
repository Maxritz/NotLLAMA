# DeepSeek Task Assignment Round 3 — GraphifyClient + Compression Benchmark + Scheduler Stub

## Project Context

**NotLLAMA** is a Vulkan compute-only inference engine for LLMs, targeting AMD RX 9070 XT (RDNA4, 16GB VRAM). It is built on the philosophy: **"Many small units doing work together."**

### Architecture
- **Host**: C++17, CMake, MSVC 19.44
- **GPU**: 14+ GLSL compute shaders compiled to SPIR-V via `glslc`
- **Scheduler**: Batch-mode Vulkan queue submission with FencePool (64 fences)
- **Inference**: `forwardPartial()` (layer-by-layer) and `forwardKernelEntry()` (persistent mailbox kernel)
- **Quantization**: 57 GGUF formats + TurboQuant (TQ4_128, TQ3_128, TQ6_64)
- **Key rule**: No buffer > 1GB. No pre-dequantization at init time.

### Repository
- Path: `C:\Users\rr\Desktop\Notllama-loc`
- GitHub: https://github.com/Maxritz/NotLLAMA (master, LGPL-2.1)
- Build: `cd build && cmake --build . --config Release`

### What Round 1 & 2 Delivered
- **Round 1**: Context/KV compression schemas, `compress_context.comp`, `kv_cache_quantize.comp`, `rdna4_graphify.hpp` stub, `tools/AGENTS.md`
- **Round 2**: `kv_cache_dequant.comp`, `rdna4_compression.hpp`, `tools/dox_lint.py`, `src/kernels/AGENTS.md` updated with compression docs

## Your Assignment (Round 3)

You will make the **GraphifyClient usable**, create a **compression benchmark**, design a **host-side compression scheduler**, write **compile-time tests**, and produce an **integration guide** so Claude can wire everything into the engine.

---

### Deliverable 1: GraphifyClient Python Implementation

**File**: `tools/graphify_client.py`

Implement the Python equivalent of `include/rdna4_graphify.hpp` as a working client that queries the local graphify knowledge graph.

**Requirements**:
- Pure Python, stdlib only (`subprocess`, `json`, `pathlib`, `re`, `collections`). No external dependencies.
- Class `GraphifyClient` with the same interface as the C++ stub:
  - `__init__(config: GraphifyConfig)` — config is a dataclass matching `rdna4_graphify.hpp`
  - `query(question: str) -> GraphQueryResult` — runs `graphify query` via subprocess
  - `is_stale() -> bool` — checks if `graph.json` mtime is older than any tracked source file
  - `update_graph() -> bool` — runs `graphify update` or `graphify build` via subprocess
  - `get_related_nodes(symbol: str, depth: int = 2) -> list[str]` — returns node IDs from graph
  - `clear_cache()` — clears in-memory LRU query cache
  - `is_available() -> bool` — static/class method; checks if `graphify` CLI is in PATH
- LRU cache for query results (size configurable via `GraphifyConfig.cacheSize`)
- `GraphifyConfig` dataclass with same defaults as C++ struct
- `GraphQueryResult` dataclass with same fields as C++ struct
- Query execution:
  ```python
  cmd = [
      sys.executable, "-m", "graphify", "query", question,
      "--graph", self.config.graphPath,
      "--budget", str(self.config.tokenBudget)
  ]
  if self.config.preferredMode == "dfs":
      cmd.append("--dfs")
  result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
  ```
- Parse JSON output from graphify. If graphify returns non-JSON, treat stdout as `answer` and set confidence=0.5.
- Cache key = `(question, config.graphPath)`
- All errors caught and returned as `GraphQueryResult` with empty answer and confidence=0.0

**Example usage**:
```python
from graphify_client import GraphifyClient, GraphifyConfig

cfg = GraphifyConfig(graphPath="graphify-out/graph.json", preferredMode="dfs")
client = GraphifyClient(cfg)
if client.is_available():
    result = client.query("How does dequantize_q6_k work?")
    print(result.answer)
    print(result.sourceLocations)
```

---

### Deliverable 2: Compression Benchmark Tool

**File**: `tools/benchmark_compression.py`

Benchmark KV cache compression and context compression to prove memory savings and acceptable accuracy loss.

**Requirements**:
- Pure Python, stdlib + optional `numpy`. Guard numpy import with pure-Python fallback.
- CLI via `argparse`:
  ```bash
  python benchmark_compression.py \
      --model model_q8_0.bin \
      --seq-len 1024 \
      --n-heads 32 \
      --head-dim 128 \
      --kv-formats q4_0 q5_0 q8_0 \
      --context-strategies importance sliding_window fifo
  ```
- Generate **synthetic KV caches** (random normal, structured sine, sparse) if no model provided.
- For each KV format:
  1. Simulate quantization: find max_abs per block, compute scale, quantize to N-bit, store packed
  2. Dequantize back to float32
  3. Compute MSE, max absolute error, cosine similarity vs original F16
  4. Compute compressed size and ratio vs F16 baseline
- For each context strategy:
  1. Generate synthetic context embeddings (seqLen × dim)
  2. Apply strategy (sliding_window, fifo, importance) to produce `keep_mask`
  3. Compute effective context length after compression
  4. Report memory saved
- Print two tables:

**KV Cache Compression:**
```
Format    | Strategy    | Orig MB | Compressed MB | Ratio | MSE       | MaxErr | CosSim
----------|-------------|---------|---------------|-------|-----------|--------|--------
F16       | baseline    | 16.00   | 16.00         | 1.00x | 0.00000   | 0.0000 | 1.0000
Q4_0      | per-block   | 16.00   | 4.50          | 0.28x | 0.00012   | 0.0034 | 0.9998
Q5_0      | per-block   | 16.00   | 5.50          | 0.34x | 0.00008   | 0.0021 | 0.9999
```

**Context Compression:**
```
Strategy       | SeqLen | Kept | Ratio | Memory Saved
---------------|--------|------|-------|-------------
sliding_window | 2048   | 1024 | 0.50x | 50.0%
fifo           | 2048   | 1024 | 0.50x | 50.0%
importance     | 2048   | 1200 | 0.59x | 41.0%
```

- `--output <file>`: writes JSON results for Claude to analyze
- Exit 0 if all MSE values < 0.01 (acceptable threshold)

---

### Deliverable 3: Compression Scheduler Header Stub

**File**: `include/rdna4_compression_scheduler.hpp`

Write a C++17 header stub that decides **when** to trigger context compression and KV cache compression. This is the host-side bridge between configuration (`rdna4_compression.hpp`) and the scheduler.

```cpp
#pragma once
#include "rdna4_compression.hpp"
#include <cstdint>
#include <vector>

namespace rdna4 {

// Decision returned by the scheduler each forward pass.
struct CompressionDecision {
    bool compressContext = false;
    bool compressKV = false;
    uint32_t contextTargetLen = 0;   // Desired output seqLen after context compression
    uint32_t kvQuantizeBits = 0;     // 0 = no change, 4/5/6/8 = re-quantize
    std::vector<uint8_t> keepMask;   // 1 = keep token, 0 = drop. Empty if compressContext=false.
};

// Host-side scheduler that monitors sequence state and decides when to compress.
// Does NOT execute compression — it only decides. Claude wires the decision into
// the inference loop (forwardPartial or forwardKernelEntry).
class CompressionScheduler {
public:
    explicit CompressionScheduler(const ContextCompressionConfig& ctxCfg,
                                   const KVCompressionConfig& kvCfg);

    // Call once per token generation step.
    // seqLen = current sequence length (including new token)
    // maxContext = model's max context length
    // importanceScores = optional per-token importance (empty if unavailable)
    // Returns a decision. If nothing needs compression, all fields are false/0.
    CompressionDecision step(uint32_t seqLen,
                              uint32_t maxContext,
                              const std::vector<float>& importanceScores = {});

    // Reset state (e.g., on new conversation)
    void reset();

    // Query current compression statistics
    uint32_t totalCompressedTokens() const;
    uint32_t totalCompressedLayers() const;

private:
    ContextCompressionConfig ctxCfg_;
    KVCompressionConfig kvCfg_;
    uint32_t lastSeqLen_ = 0;
    uint32_t totalCompressed_ = 0;
    bool contextCompressed_ = false;
    bool kvCompressed_ = false;
};

} // namespace rdna4
```

**Requirements**:
- Use `#pragma once`
- No Vulkan includes needed
- Include `<vector>`, `<cstdint>`
- Add `static_assert(sizeof(CompressionDecision) < 1024, "");` (it's host-only, but keep it small)
- Document in comments:
  - Context compression triggers when `seqLen > maxContext * ctxCfg_.threshold`
  - KV compression triggers when `seqLen >= kvCfg_.minSeqLen` AND `kvCfg_.enabled`
  - `keepMask` generation depends on `ctxCfg_.strategy`:
    - 0 (uniform): keep every Nth token to reach target length
    - 1 (entropy): placeholder — requires entropy scores from model
    - 2 (importance): use `importanceScores` to keep top-K tokens
- This is a stub — no `.cpp` implementation needed. Claude will implement `step()` after delivery.

---

### Deliverable 4: Compression Compile-Time Test

**File**: `test/test_compression.cpp`

Create a C++ compile-time test stub verifying compression structs and push constants.

```cpp
// DeepSeek Round 3 — Compression host-side compile test
// Compiles as: cl /EHsc /I ../include test_compression.cpp

#include "rdna4_compression.hpp"
#include "rdna4_compression_scheduler.hpp"
#include <cassert>
#include <cstdio>

int main() {
    printf("Compression compile-time tests...\n");

    // Push constant size checks
    static_assert(sizeof(rdna4::CompressContextPushConstants) <= 128,
        "CompressContextPushConstants must fit in 128 bytes");
    static_assert(sizeof(rdna4::KVCacheQuantizePushConstants) <= 128,
        "KVCacheQuantizePushConstants must fit in 128 bytes");
    static_assert(sizeof(rdna4::KVCacheDequantPushConstants) <= 128,
        "KVCacheDequantPushConstants must fit in 128 bytes");

    // Config default construction
    rdna4::ContextCompressionConfig ctxCfg;
    assert(ctxCfg.enabled == false);
    assert(ctxCfg.blockSize == 128);
    assert(ctxCfg.bits == 4);
    assert(ctxCfg.threshold > 0.8f && ctxCfg.threshold < 0.9f);

    rdna4::KVCompressionConfig kvCfg;
    assert(kvCfg.enabled == false);
    assert(kvCfg.blockSize == 64);
    assert(kvCfg.kBits == 4);
    assert(kvCfg.minSeqLen == 256);

    // Scheduler default construction
    rdna4::CompressionScheduler scheduler(ctxCfg, kvCfg);
    // step() with seqLen below threshold should return no-op decision
    auto decision = scheduler.step(100, 4096);
    assert(decision.compressContext == false);
    assert(decision.compressKV == false);

    printf("All compression compile-time tests PASSED\n");
    return 0;
}
```

**Requirements**:
- Must compile with MSVC: `cl /EHsc /I include test/test_compression.cpp`
- No Vulkan dependencies
- Include both `rdna4_compression.hpp` and `rdna4_compression_scheduler.hpp`

---

### Deliverable 5: Compression Integration Guide

**File**: `docs/compression_integration.md`

Write an integration guide for Claude/Kimi to wire compression into the engine.

**Structure**:
```markdown
# Compression Integration Guide

## Overview
NotLLAMA supports context compression (shortening the KV cache) and KV cache quantization
(reducing precision of K/V tensors). This doc explains how to wire both into the engine.

## Files Delivered by DeepSeek
- `src/kernels/compress_context.comp` — compacts KV cache based on keep_mask
- `src/kernels/kv_cache_quantize.comp` — quantizes KV cache to Q4_0/Q5_0/Q8_0
- `src/kernels/kv_cache_dequant.comp` — dequantizes KV cache back to F16
- `include/rdna4_compression.hpp` — config structs + push constants
- `include/rdna4_compression_scheduler.hpp` — decision logic stub
- `tools/dox_lint.py` — DOX compliance checker
- `tools/graphify_client.py` — GraphifyClient implementation
- `tools/benchmark_compression.py` — compression benchmarking

## Engine Wiring Checklist

### Step 1: Load Compression Configs
- [ ] Parse `context_compression_schema.json` and `kv_compression_schema.json` at startup
- [ ] Populate `ContextCompressionConfig` and `KVCompressionConfig`
- [ ] Pass configs to `CompressionScheduler`

### Step 2: KV Cache Quantization (Optional)
- [ ] After every `forwardPartial()` or `forwardKernelEntry()`, check `CompressionScheduler::step()`
- [ ] If `decision.compressKV == true`:
  - Allocate quantized KV buffers (size = compressed)
  - Dispatch `kv_cache_quantize.comp` with `KVCacheQuantizePushConstants`
  - Free original F16 KV buffers (or keep as fallback)
- [ ] Before attention dispatch, if KV is quantized:
  - Dispatch `kv_cache_dequant.comp` with `KVCacheDequantPushConstants`
  - Output goes to temporary F16 buffer consumed by `attention.comp`

### Step 3: Context Compression (Optional)
- [ ] If `decision.compressContext == true`:
  - Build `keepMask` vector on host (or use importance scores from model)
  - Upload `keepMask` to GPU
  - Dispatch `compress_context.comp` with `CompressContextPushConstants`
  - Update `seqLen` to `decision.contextTargetLen`

### Step 4: Scheduler Integration
- [ ] Instantiate `CompressionScheduler` in `InferenceEngine` or `ContextManager`
- [ ] Call `step(seqLen, maxContext)` every token
- [ ] If compression triggers, pause generation, run compression shader(s), resume

### Step 5: Fallback & Safety
- [ ] If compression shader fails (VK_ERROR_DEVICE_LOST), fall back to uncompressed path
- [ ] If `seqLen < kvCfg.minSeqLen`, never compress KV
- [ ] If `seqLen < 256`, never compress context

## Performance Targets
- KV Q4_0: 4× memory reduction, <0.1% MSE vs F16
- Context sliding_window: 2× reduction, zero quality loss for recent context
- Context importance: 1.5–2× reduction, <2% perplexity increase

## Graphify Integration
- Use `GraphifyClient` in `tools/graphify_client.py` to query the knowledge graph before
  reading source files during integration work.
- Example: `client.query("Where is KV cache allocated in inference_engine.cpp?")`
```

**Requirements**:
- Cover all 5 wiring steps
- Reference exact struct names and push constants
- Include fallback/safety rules
- Add Graphify integration section

---

## Guardrails (What You Must NOT Do)

1. **Do NOT modify `src/host/inference_engine.cpp`, `src/host/scheduler.cpp`, or `main.cpp`.**
   - You may add comments showing integration points, but do not change existing logic.
   - Claude will wire your code into the engine after delivery.

2. **Do NOT change existing enum values or struct layouts.**
   - Append new types only. Do not renumber.

3. **Do NOT allocate > 1GB for any single buffer in shaders or host logic.**
   - Benchmarks should use small synthetic data (seqLen <= 4096, heads <= 32).

4. **Do NOT write inline implementations in headers (except trivial constexpr getters).**
   - `CompressionScheduler` is explicitly a stub — no `.cpp` needed yet.
   - `GraphifyClient` is Python — full implementation required.

5. **Do NOT break the build.**
   - Every new shader must compile with `glslc`.
   - Every new header must compile as C++17 standalone.

6. **Do NOT introduce heavy Python dependencies.**
   - `graphify_client.py` and `benchmark_compression.py` must use stdlib only (optional numpy with fallback).

## Verification Checklist

Before declaring done, verify:

- [ ] `python tools/graphify_client.py` runs without import errors (can test with `--help` or a simple import)
- [ ] `python tools/benchmark_compression.py --seq-len 512` produces both tables
- [ ] `python tools/benchmark_compression.py --output results.json` writes valid JSON
- [ ] `include/rdna4_compression_scheduler.hpp` compiles as C++17 (test with `cl /EHsc /I include`)
- [ ] `test/test_compression.cpp` compiles and passes all asserts
- [ ] `docs/compression_integration.md` covers all 5 wiring steps + fallback rules
- [ ] `tools/dox_lint.py` still runs cleanly (no regressions from Round 2)
- [ ] No existing source files were modified

## Context Files You Should Read First

1. `include/rdna4_compression.hpp` — Configs and push constants from Round 2
2. `include/rdna4_graphify.hpp` — C++ stub to mirror in Python
3. `src/kernels/kv_cache_dequant.comp` — Dequant shader from Round 2
4. `src/kernels/compress_context.comp` — Context compression shader from Round 1
5. `docs/context_compression_schema.json` — Context config schema
6. `docs/kv_compression_schema.json` — KV config schema

## Questions?

If you are unsure about:
- **subprocess handling in Python**: Ask Kimi.
- **C++ stub design**: Ask Kimi.
- **Benchmark accuracy metrics**: Ask Kimi.
- **Project-specific conventions (DOX, AGENTS.md)**: Read the root `AGENTS.md` again.
- **Whether a change violates a guardrail**: Stop and ask Kimi before proceeding.

## Expected Time
- `graphify_client.py`: 2-3 hours
- `benchmark_compression.py`: 2-3 hours
- `rdna4_compression_scheduler.hpp`: 45 min
- `test_compression.cpp`: 30 min
- `compression_integration.md`: 1 hour
- Total: ~6-8 hours
