# TRACE: Q4_K Scale Packing Fix

## Session: 2026-06-29 — Round 2 fixes

## Bug
GPU shader, CPU reference, and weight uploader all used wrong Q4_K scale extraction:
1. Treated `scales[]` as 8 × simple int8 (1 byte per sub-block)
2. Used 4-bit nibble packing for a pair of sub-blocks
3. Centered q4 value via `(nibble - 8)` (symmetric) instead of unsigned dequant

## GGML Standard (from `reference/llama.cpp/ggml-quants.c`)

### Block Layout (144 bytes)
```
offset 0:  d      (fp16)        — block-level float scale
offset 2:  dmin   (fp16)        — block-level float min
offset 4:  scales[12] (uint8)   — 8 × 6-bit scale + 8 × 6-bit min, packed
offset 16: qs[128]  (uint8)     — 4-bit nibbles, 256 elements
```

### Scales Packing (`get_scale_min_k4`)
Sub-blocks j=0..7, each with 6-bit scale (Ls[j]) and 6-bit min (Lm[j]):

| scales[] index | j < 4 (j=0..3) | j ≥ 4 (j=4..7) |
|----------------|----------------|-----------------|
| scales[j]      | Ls[j] (bits 0-5) | Ls[j] low 6 bits + Ls[j+4] high 2 bits |
| scales[j+4]    | Lm[j] (bits 0-5) | Lm[j] low 6 bits + Lm[j+4] high 2 bits |
| scales[j+8]    | —                | Ls[j] low 4 bits + Lm[j] low 4 bits |

Extraction (get_scale_min_k4):
```
if (j < 4):
    sc = scales[j] & 63          // low 6 bits
    sm = scales[j+4] & 63
else:
    sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    sm = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
```

### QS Layout
128 bytes for 256 elements. Sub-block pairs (0+1, 2+3, 4+5, 6+7) share 32 bytes:
- qs[(pair/2)*32 + l]: low nibble = element l of even sub-block, high nibble = element l of odd sub-block

### Dequant Formula
```
result = d_block * Ls[subBlock] * nibble - dmin_block * Lm[subBlock]
// nibble is UNSIGNED 0-15 (NOT centered)
```

## Before Fix (Wrong)
### GPU Shader (dequantize.comp)
```glsl
// WRONG: 4-bit scale packing, centered nibble
uint scaleByteIdx = sbOffset + 4u + (subBlock >> 1u);
uint scaleByte = readByte(src, scaleByteIdx);
subScale = d * float(scaleByte & 0xF);
subMin = dmin * float((scaleByte >> 4u) & 0xF);
int q = int(nibble) - 8;  // WRONG: centered
result = subScale * float(q) - subMin;  // WRONG: formula
```

### CPU Reference (cpu_reference.cpp) & Weight Uploader
```cpp
// WRONG: simple int8 scale per sub-block
uint8_t sc = data[bs + 4 + subBlock];  // WRONG: not get_scale_min_k4
result = d * (float)sc * ((float)nibble - 8.0f) + dmin;  // WRONG: formula
```

## After Fix (Correct)
### GPU Shader
```glsl
// Correct: get_scale_min_k4 extraction
if (subBlock < 4u) {
    sc = readByte(src, scalesBase + subBlock) & 63u;
    sm = readByte(src, scalesBase + subBlock + 4u) & 63u;
} else {
    uint bJp4 = readByte(src, scalesBase + subBlock + 4u);
    uint bJm4 = readByte(src, scalesBase + subBlock - 4u);
    uint bJ0  = readByte(src, scalesBase + subBlock);
    sc = (bJp4 & 0x0Fu) | ((bJm4 >> 6u) << 4u);
    sm = (bJp4 >> 4u)    | ((bJ0  >> 6u) << 4u);
}
// Correct: unsigned nibble, d*sc*q4 - dmin*sm
result = d * float(sc) * float(q4) - dmin * float(sm);
```

## Fixed Files
| File | Lines | Change |
|------|-------|--------|
| `src/kernels/dequantize.comp` | 230-260 | get_scale_min_k4 extraction + unsigned q4 |
| `cpu_reference.cpp` | 200-217 | Same fix |
| `src/host/weight_uploader.cpp` | 147-165 | Same fix |
| `include/rdna4.hpp` | 65 | Added UNSUPPORTED sentinel |

## VERDICT: FIXED
- Verified against `reference/llama.cpp/ggml-quants.c` dequantize_row_q4_K (lines 1471-1493)
