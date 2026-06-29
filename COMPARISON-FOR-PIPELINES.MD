# Comparison: NotLLAMA vs llama.cpp vs VindexLLM

Three-way comparison of the model handling and quant/dequant pipeline architectures.

---

## 0. Accuracy Architecture — Comparison of Dequant Math

This section compares how each project's GPU shaders interpret the raw quantized bytes. The root cause of the MaxAE=18.5 failure in `acrux-500m-q6k` is likely here.

### Q8_0 Block Layout

Each Q8_0 block is **34 bytes for 32 elements**:
| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0 | 2 | `d` | float16 scale |
| 2 | 32 | `qs[32]` | int8 quantized values |

Dequant: `value = d * qs[i]` (signed int8 multiply)

### Q6_K Block Layout

Each Q6_K block is **210 bytes for 256 elements** (d-last):
| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0 | 128 | `ql[128]` | low 4 bits, 2 per byte |
| 128 | 64 | `qh[64]` | high 2 bits, 4 per byte |
| 192 | 16 | `scales[16]` | int8 sub-block scales (16 elements each) |
| 208 | 2 | `d` | float16 super-block scale |

Dequant: `value = d * scales[subBlock] * (ql | (qh << 4) - 32)`

### Per-Project Math Verification

| Aspect | **llama.cpp** | **VindexLLM** | **NotLLAMA** |
|--------|---------------|---------------|--------------|
| **Buffer type** | Typed structs (`block_q8_0`, `block_q6_K`) with explicit signed fields | `int8_t[]` raw byte buffer (GLSL `int8_t` = auto sign-extend) | `uint[]` raw byte buffer (manual sign-extend via `if (q>127) q-=256`) |
| **Q8_0 read** | `float(data_a[ib].d) * data_a[ib].qs[...]` — signed int8 via struct | `float(weight_bytes[...]) * unpackHalf2x16(lo | hi<<8)` — signed via int8_t[] | `readF16(W, bs)` for d, `readByte(W, bs+2+eib)` + manual sign-extend |
| **Q6_K read** | `d * scales[is] * (int8_t((ql[...] & 0xF) | ((qh[...] >> shift) & 3) << 4) - 32)` | No Q6_K matvec — only uses Q8_0, Q4_0, F16 | `d * sc * ((q4 | (qh2 << 4)) - 32)` — same math, different addressing |
| **Q6_K d position** | `d` at end of struct (auto-calculated offset) | N/A | `bs + 208` — matches 210-byte block |
| **Accumulation** | 256 threads/row via `dotPacked4x8EXT` with shared memory reduction | 256 threads/row via shared scale cache + parallel reduction | 64 threads per workgroup, each thread computes one column independently, no shared memory |
| **Per-thread work** | 8/16/32 elements per inner loop iteration | Strided across `in_dim` elements | **Full sequential loop** over all `in_dim` elements per output column |

### Likely Root Cause of MaxAE=18.5

The fused matvec math for Q8_0 and Q6_K is **numerically identical** across all three projects — `d * qty * input[k]` is the same formula. The error is most likely in:

1. **Embedding lookup** — `embed_q8_0.comp` or `embed_q6_k.comp` may misinterpret the embedding table layout (row vs column indexing)
2. **Shape/dimension mismatch** — `dispatchFusedOrGemm` passes `inDim`/`outDim` that don't match the actual tensor shape
3. **Tensor naming** — the tensors found by `findTensorAddr()` at runtime differ from what the CPU reference loads (e.g., mismatched prefixes or missing fallback patterns)
4. **Workgroup count overflow** — if `(outDim + 63) / 64` exceeds a thread or block limit for a large output dimension, dispatches silently fail

The VindexLLM reference shows that Q8_0 matvec with shared memory scale caching compiles and runs correctly for the same GGUF format. The difference in NotLLAMA is that it uses **64 threads with no shared memory** and a sequential inner loop, which should give the same answer but might expose subtle indexing issues at tensor boundaries.

---

## 1. Dequant Strategy

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **Timing** | Pre-dequant ALL weights → float16 GPU buffers once at load | **Hybrid**: fused matvec for Q6_K/Q8_0/Q4_0 weights, on-demand staging dequant for F32/F16 | Fused inline dequant during matvec |
| **Dequants per token** | **0** (all pre-done) | ~36 (norm weights only) — was 324 before fused matvec | **0** (inline inside matvec) |
| **GPU memory** | ~2× weight size (quantized + float16 copies) | ~1 layer staging buffer (~1/36th of total) | Same as weights (no staging) |

NotLLAMA follows the **Dox philosophy** ("many small units, on-demand") — minimal VRAM. The new `dispatchFusedOrGemm()` (lines 760-788) eliminates 9 dequants/layer × 36 layers = 324 dequants/token for Q6_K/Q8_0/Q4_0 weights. Only RMS norm weights (which are F16) still go through staging dequant. `initWeightBuffer()` at line 467 is still **disabled**.

> **⚠️ Accuracy caveat**: Fused matvec removes the dispatch overhead, but accuracy on Q8_0/Q6_K models is **not yet validated**. Last test on `acrux-500m-q6k` (mostly Q8_0 weights, ffn_down as Q6_K) showed MaxAE=18.5 → FAIL. Until `test_inference.exe` passes, this bottleneck cannot be considered fully solved — the fused matvec shaders (`matvec_q8_0.comp`, `matvec_q6_k.comp`, `matvec_q4_0.comp`) likely have a correctness bug (block layout, sign extension, or indexing).

---

## 2. Submit Strategy (Vulkan Overhead)

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **Submits per token** | **1** | ~36 (1 batch per layer) | Unknown |
| **Syncs per token** | 1 fence wait | ~36 syncAllThrottled calls | Unknown |
| **Technique** | Single cmd buffer, `vkCmdPipelineBarrier` only | `beginBatch/dispatchInBatch/endBatch` per layer | Fused QKV + single dispatch |

Each layer is one batch → 36 submits/token. Each submit has ~50µs CPU overhead = 1.8ms before any GPU work. Scheduler now logs throttle stats:
```
[throttle] gpu=2.34ms sleep=0.59ms target=80%
```

---

## 3. GEMM Shader

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **Workgroup** | Tiled 16×16 (256 threads) with shared memory | **Dual-path**: fused matvec (64 threads) or naive gemm (32 threads) | 64 threads, fused matvec |
| **Throughput** | ~64× naive (shared memory tiling) | **Improved**: fused matvec matches VindexLLM on Q6_K/Q8_0/Q4_0 | ~64× (coalesced reads) |
| **Shared memory** | Yes | **None** | No staging needed |

**Crucial update**: `gemm.comp` (naive 32-thread scalar loop) is now the **fallback** — only runs for F32/F16 weights. The primary path is fused matvec (`matvec_q6_k.comp`, `matvec_q8_0.comp`, `matvec_q4_0.comp`) with 64 threads that dequantize inline and read quantized weights directly.

> **⚠️ Accuracy caveat**: Last test on `acrux-500m-q6k` (mostly Q8_0 weights) gave MaxAE=18.5. The fused matvec path (`matvec_q8_0.comp` in particular) has an unconfirmed correctness bug. This section describes the architectural design; the numbers assume correct implementation, which is not yet confirmed.

```glsl

```glsl
// matvec_q6_k.comp — 64 threads, each computes one output column
for (uint k = 0u; k < pc.inDim; ++k) {
    // Q6_K block: 210 bytes per 256 elements, d-last layout
    float d  = readF16(W, bs + 208);
    int   sc = sign_extend(readByte(W, bs + 192 + eib/16));
    uint  q4 = nibble(readByte(W, bs + qlOff));
    uint  q2 = nibble(readByte(W, bs + 128 + qhOff));
    int   val = int(q4 | (q2 << 4)) - 32;
    acc += d * float(sc) * float(val) * In.data[k];
}
```

This eliminates the separate `dequantize.comp` dispatch + float32 gemm for all Q6_K/Q8_0/Q4_0 weights.

### New Shaders in This Round

| Shader | Lines | Format | Threads | Purpose |
|--------|-------|--------|---------|---------|
| `matvec_q6_k.comp` | 93 | Q6_K (210B/256) | 64 | Fused matvec, d-last layout, int8 sub-block scales |
| `matvec_q8_0.comp` | 78 | Q8_0 (34B/32) | 64 | Fused matvec, int8 quants + fp16 delta |
| `matvec_q4_0.comp` | 79 | Q4_0 (18B/32) | 64 | Fused matvec, nibble unpack + fp16 delta |
| `embed_q6_k.comp` | 81 | Q6_K (210B/256) | 32 | Fused embedding lookup, same block layout |
| `embed_q8_0.comp` | — | Q8_0 | 32 | Existing fused embedding (from prior round) |

### dequantize.comp Rewrite
Complete rewrite from `uint[]` byte-addressing to proper `uint8_t[]` buffer references:
- Added `GL_EXT_shader_explicit_arithmetic_types_int8` and `GL_EXT_shader_8bit_storage`
- `fp16_to_fp32(uint8_t low, uint8_t high)` — cleaner f16 unpack
- `sign_extend_u8(uint8_t v)` — single instruction sign extension
- All 16 formats (F32 through Q8_K) use clean byte-level access
- Q6_K fixed: sub-block scale `sc` was missing (commit bf91ae7), now correctly reads from `bs + 192 + subBlock`

---

## 4. Attention

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **Algorithm** | Flash Attention (tiled, memory-efficient) | **Naive O(N²)** — full Q×K materialization | Subgroup reduction |
| **Per-head dispatch** | Tiled shared memory | **1 dispatch per head** in CPU loop | Subgroup broadcast in warp |
| **Optimization** | Years of tuning | SubgroupShuffleDown chain (Round 4) | Same subgroup pattern |

`flash_attention.comp` exists but remains **unwired**.

---

## 5. Fusion Level (QKV)

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **Q/K/V GEMMs** | 3 separate dispatches | 3 separate (now each uses fused matvec instead of dequant+gemm) | **Fused into 1 dispatch** with concatenated QKV weight |

`forwardPartial()` lines 873-875:
```cpp
dispatchFusedOrGemm("attn_q.weight", normHidden, qRowAddr, dim, dim, 0);
dispatchFusedOrGemm("attn_k.weight", normHidden, kRowAddr, dim, kvDim, 0);
dispatchFusedOrGemm("attn_v.weight", normHidden, vRowAddr, dim, kvDim, 0);
```

Three calls — same as before — but each now uses fused matvec (no dequant) instead of dequant+gemm for Q6_K/Q8_0/Q4_0. The dequant+gemm path only triggers for F32/F16 weight formats. Fused QKV (single dispatch reading [dim × 3*dim]) remains **not implemented** (Task A in `TASKS_DEEPSEEK_THINKER.md` — only subgroup sub-part done).

---

## 6. KV Cache Compression

| | **llama.cpp** | **NotLLAMA (ours)** | **VindexLLM** |
|---|---|---|---|
| **KV cache format** | FP16 (row-wise GGML quant avail) | FP16 via `kv_cache_write.comp` | **TQ3** (3-bit, Walsh-Hadamard transform) |
| **Compression** | None by default | None | **5.3×** over FP16 |
| **Quant shaders** | In GGML type system | `kv_cache_quantize.comp` / `kv_cache_dequant.comp` exist but **unwired** | Built into TQ3 pipeline |

Unchanged from prior analysis.

---

## 7. Windows/AMD WDDM Routing

| | **llama.cpp** | **NotLLAMA (ours)** |
|---|---|---|
| **Queue family** | Avoid `GRAPHICS_BIT`, picks ACE/Family 1 by default | Same avoidance pattern (`context.cpp:78-82`) |
| **Feature chain** | Passes **ALL** supported features to `vkCreateDevice` via full pNext chain | Compute-only: `dynamicRendering=VK_FALSE`, all others zeroed |
| **Env override** | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` → Family 0, **56% faster** on RDNA4 | No equivalent |
| **AMD driver routing** | Feature-rich → "modern app" → ACE or 3D engine both work | Compute-only → older driver path → stable but may be slower |

NotLLAMA's compute-only feature chain was a surgical fix for the 0% utilization bug (Round 6, commit 17e999e). llama.cpp's all-features approach is more permissive — the graphics queue is actually faster for compute on RDNA4.

---

## 8. The Big Picture

```
llama.cpp:     "Pre-dequant, one submit, all features enabled"
               → Fastest (6.8ms/tok), highest VRAM

VindexLLM:     "Fuse everything inline, no staging"
               → Moderate VRAM, moderate speed, Q6_K/Q8_0/Q4_0 matvec

NotLLAMA (ours): "Many small units, fused matvec for quantized, staging for F16"
                 → Low VRAM, stable AMD WDDM routing, fused matvec untested for accuracy
```

### Current NotLLAMA Bottlenecks (ranked, updated for fused matvec)

| Rank | Bottleneck | Est. Gap | Where | File |
|------|-----------|----------|-------|------|
| **1** | Naive scalar GEMM fallback for F32/F16 weights | **~50×** (only on F32/F16 path) | `gemm.comp` fallback | `src/kernels/gemm.comp` |
| **2** | 36 submits/token vs 1 | **~36×** submit overhead | `forwardPartial()` per-layer endBatch | `src/host/inference_engine.cpp` |
| **3** | Naive O(N²) attention with per-head CPU loop | **~10×** | `attention.comp` per head | `src/kernels/attention.comp` |
| **4** | No fused QKV — 3 dispatches/layer vs 1 | **~3×** | `dispatchFusedOrGemm` ×3 | `src/host/inference_engine.cpp` |
| **5** | Unwired KV cache quantization | Feature gap | `kv_cache_quantize.comp` unused | `src/kernels/` |
| **6** | No graphics-queue opt-in | **56%** possible speed on RDNA4 | `context.cpp` feature chain | `src/host/context.cpp` |

**Key change**: Bottleneck #3 from previous analysis ("324 on-demand dequants/token") is **structurally solved** — fused matvec removes the dispatch overhead — but **accuracy is not yet validated**. `test_inference.exe` on `acrux-500m-q6k` fails with MaxAE=18.5, indicating a correctness bug in `matvec_q8_0.comp` (the most-used path for this model). Until parity passes, the fused matvec path cannot be considered production-ready. The dispatch savings remain valid once the bug is fixed.

### What Each Project Does Well

| Project | Strength | Weakness |
|---------|----------|----------|
| **NotLLAMA** | Minimal VRAM, Dox philosophy, fused matvec for 3 formats, explicit barriers for correctness, Windows AMD routing stability | Naive GEMM fallback for F32/F16, too many submits, no fused QKV, flash attention unwired, **fused matvec accuracy unvalidated** |
| **llama.cpp** | Fastest time-to-token, mature tiled GEMM, single-submit pipeline, flash attention | High VRAM usage (pre-dequant), years of complexity |
| **VindexLLM** | Fused matvec dequant, TQ3 KV cache, fused QKV approach | Not a complete framework, limited format support (Q6_K/Q8_0/Q4_0) |
