# TRACE: Q8_K Block Size Fix

## Session: 2026-06-29 — Round 2 cleanup

## Bug
Two bugs in Q8_K handling:
1. `gguf.cpp:getQuantMeta()` returned 256 bytes per block instead of 292
2. `gguf_loader.py:dequantize_q8_k()` used 290 bytes with wrong structure (fp16 d + 32 sub-block scales)

## Correct GGML Q8_K Structure
From `reference/llama.cpp/ggml-quants.c`:
```c
typedef struct {
    float  d;         // offset 0:  4 bytes, float32 block scale
    int8_t qs[256];   // offset 4: 256 bytes, 8-bit quants
    int16_t bsums[16];// offset 260: 32 bytes, block row sums (not used in dequant)
} block_q8_K;
static_assert(sizeof(block_q8_K) == 292);
```

Dequant: `y[i] = block.d * block.qs[i%256]` — single scale, no sub-blocks.

Q8_K is the **only K-quant that is d-first** (d at offset 0). All others are d-last.

## Bug 1: gguf.cpp

### Input State
| Variable | Value | Source |
|----------|-------|--------|
| gguf.cpp line 33 | `{type, 256, 256}` | getQuantMeta |
| Correct bytesPerBlock | 292 | ggml-quants.c |
| Callers | `t.nbytes` computation, staging buffer creation | gguf.cpp:161,238 |

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| bytesPerBlock == 292 | TRUE | 256 | FAIL |
| t.nbytes correct for Q8_K | TRUE | 256/292 ≈ 87.7% | FAIL |

### Impact
Q8_K tensors would be sized at 87.7% of actual size. Staging buffer truncates last 12.3% of data.

### Fix
```
case GGUFTensorType::Q8_K: return {type, 256, 256};
→ case GGUFTensorType::Q8_K: return {type, 256, 292};
```

## Bug 2: gguf_loader.py

### Before Fix (Wrong)
```python
BLOCK_SIZE = 290
d = struct.unpack("<e", raw[pos:pos+2])[0]   # fp16 (WRONG: should be float32)
scales = raw[pos+2:pos+34]                     # 32 int8 sub-block scales (WRONG: none exist)
qs = raw[pos+34:pos+290]                       # 256 quants
# ... applies sub-block scales ...
```

### After Fix (Correct)
```python
BLOCK_SIZE = 292
d = struct.unpack("<f", raw[pos:pos+4])[0]     # float32 (correct)
qs_data = raw[pos+4:pos+260]                    # 256 quants
pos += BLOCK_SIZE                                # skip bsums[16] (used in quantize only)
# result = d * q_signed per element, single scale
```

### Cross-Reference: All Q8_K Implementations
| File | Block Size | Structure | PASS? |
|------|-----------|-----------|-------|
| ggml-quants.c | 292 | float32 d + int8[256] + int16[16] | ✅ REFERENCE |
| dequantize.comp | 292 | Same | ✅ MATCHES |
| cpu_reference.cpp | 292 | Same | ✅ MATCHES |
| weight_uploader.cpp | 292 | Same | ✅ MATCHES |
| inference_engine.cpp | 292 | Same | ✅ MATCHES |
| gguf.cpp | **256→292 (FIXED)** | Same | ✅ FIXED |
| gguf_loader.py TYPE_BLOCK_SIZES | 292 | Same | ✅ CORRECT |
| gguf_loader.py dequantize_q8_k | **290→292 (FIXED)** | Same | ✅ FIXED |

## VERDICT: FIXED
- All 7 implementations now agree on 292-byte block size and correct structure
