# TRACE: First Token Forward Failure

## Session: 2026-06-29 — Round 2 fixes

## Bug
First forward (seqPos=0) returned token 0 with error 99.0%. Root cause: **three independent bugs** that all manifested simultaneously.

## Input State
| Variable | Value | Source |
|----------|-------|--------|
| Model | llamamobile (762 MB) | test_inference |
| Formats | Q6_K (embed, ffn_down), Q4_K (QKV, gate, up) | weight JSON |
| Layers | 16, dim=2048, heads=32, kv_heads=8, vocab=128256 | metadata |
| Ring alloc | 64 MB → OOM; later 1 GB | test_inference.cpp |
| Quantized data in staging | READ correctly from .bin | weight_uploader |

## Truth Table: Three Bugs

### Bug A: Dequant Workgroup Overflow (CRITICAL)

| Weight | nElements | Workgroups (64) | Limit | PASS? |
|--------|-----------|-----------------|-------|-------|
| attn_q | 4,194,304 | 65,536 | 65,535 | FAIL |
| attn_o | 4,194,304 | 65,536 | 65,535 | FAIL |
| ffn_gate | 16,777,216 | 262,144 | 65,535 | FAIL |
| ffn_up | 16,777,216 | 262,144 | 65,535 | FAIL |
| ffn_down | 24,460,373 | 382,194 | 65,535 | FAIL |

**Root cause**: `dequantIfNeeded()` dispatched all workgroups in one call. Vulkan `maxComputeWorkGroupCount[0]` = 65,535. Driver does not reject — it silently truncates or wraps, producing garbage dequant.

**Truth:**
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| wgCount ≤ 65535 | TRUE | FALSE (65,536) | FAIL |
| Dequant output = F32 | TRUE | GARBAGE | FAIL |
| GEMM reads correct F32 | TRUE | FALSE | FAIL |

### Bug B: Ring Allocator OOM (CRITICAL)

Per-layer dequant staging requirement: **~265 MB** for 7 weights (Q4_K, 256-element blocks).

Ring allocator was **64 MB**. Sequence of allocs:

1. Fixed allocs (hidden + ...): 1.4 MB → 62.6 MB remaining ✅
2. attn_q dequant (16 MB) → 46.6 MB remaining ✅
3. attn_k dequant (4 MB) → 42.6 MB ✅
4. attn_v dequant (6 MB) → 36.6 MB ✅
5. attn_o dequant (16 MB) → 20.6 MB ✅
6. ffn_gate dequant (64 MB) → **OOM** (need 64 MB, have 20.6 MB) ❌

**Root cause**: Ring too small. OOM returns 0 for ffn_gate staging address → GEMM reads quantized bytes as F32 → corrupt output.

### Bug C: Silent F32 Fallback (MAJOR)

Unknown GGML types returned `QuantFormat::F32` (0) by default in `ggmlToQuantFormat`. When a tensor's format wasn't in the switch, it was silently treated as F32, reading raw quantized bytes as 4-byte floats. This meant:

- If any tensor was unrecognized, its "dequant" produced garbage at full F32 size
- Took up 4× the staging space it should
- Amplified the ring OOM bug

**Fix**: Changed default to `UNSUPPORTED = 255`, caller skips tensor with stderr warning.

## Distillation
- Bug A (workgroup): Fixed via chunking loop, `DQ_CHUNK_ELEMS = 65535*64`
- Bug B (ring OOM): Fixed via 1 GB ring, checkpoint/restore per layer
- Bug C (silent F32): Fixed via `UNSUPPORTED` sentinel

## Debug Points
```cpp
// Before each dequant dispatch:
ASSERT(wgCount <= 65535);
LOG("[dequant] chunk=%u wg=%u offset=%u", chunk, wgCount, offset);
// At ring alloc:
ASSERT(offset + size <= ringSize);
LOG("[ring] allocated %u bytes at offset %u", size, offset);
```
