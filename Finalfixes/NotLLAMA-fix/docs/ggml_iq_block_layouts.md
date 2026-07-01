# GGML IQ Quantization Block Layouts — Complete Reference

Byte-level layouts extracted from llama.cpp source (`ggml-common.h`, `ggml-quants.c`).

---

## 1. Q1_0 (type 41) — 1.0078 bpw

- **Block size:** 128 weights (`QK1_0 = 128`)
- **Total bytes per block:** 18 bytes

```
struct block_q1_0 {
    ggml_half d;                // offset 0:  2 bytes, fp16 scale (mean absolute value)
    uint8_t   qs[QK1_0 / 8];   // offset 2: 16 bytes, 1 bit per weight (sign bits)
};
static_assert(sizeof(block_q1_0) == 18);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale = sum(|x[i]|) / 128 |
| 2–17 | 16B | `qs` — 128 sign bits, packed 8 per byte, LSB first |

**Packing:** Bit `j` of `qs[j/8]` is 1 if `x[j] >= 0`, else 0.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for j in 0..127:
    bit = (block.qs[j/8] >> (j%8)) & 1
    y[j] = bit ? d : -d
```

---

## 2. IQ4_NL (type 20) — 4.5 bpw

- **Block size:** 32 weights (`QK4_NL = 32`)
- **Total bytes per block:** 18 bytes

```
struct block_iq4_nl {
    ggml_half d;               // offset 0:  2 bytes, fp16 scale
    uint8_t   qs[QK4_NL / 2]; // offset 2: 16 bytes, 4-bit nibbles
};
static_assert(sizeof(block_iq4_nl) == 18);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–17 | 16B | `qs` — 32 × 4-bit indices into `kvalues_iq4nl[16]` grid, 2 per byte |

**Packing:** Low nibble = even index, high nibble = odd index.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for j in 0..15:
    idx0 = block.qs[j] & 0x0F
    idx1 = block.qs[j] >> 4
    y[j*2 + 0] = kvalues_iq4nl[idx0] * d
    y[j*2 + 1] = kvalues_iq4nl[idx1] * d
```

**Grid values** (`kvalues_iq4nl[16]`): `{ -127, -104, -83, -65, -49, -35, -22, -11, 0, 11, 22, 35, 49, 65, 83, 104 }`

---

## 3. IQ4_XS (type 23) — 4.328 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 146 bytes

```
struct block_iq4_xs {
    ggml_half d;                    // offset 0:   2 bytes, fp16 super-block scale
    uint16_t  scales_h;             // offset 2:   2 bytes, high bits of 6-bit sub-block scales
    uint8_t   scales_l[QK_K/64];   // offset 4:   4 bytes, low bits of 6-bit sub-block scales
    uint8_t   qs[QK_K/2];          // offset 8: 128 bytes, 4-bit quantized values
};
static_assert(sizeof(block_iq4_xs) == 146);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 super-block scale |
| 2–3 | 2B | `scales_h` — high 2 bits of each 6-bit scale (packed) |
| 4–7 | 4B | `scales_l` — low 4 bits of each 6-bit scale (1 per sub-block) |
| 8–135 | 128B | `qs` — 256 × 4-bit indices into `kvalues_iq4nl[16]`, 2 per byte |

**Sub-block structure:** 16 sub-blocks of 16 weights each.

**Scale reconstruction:** For sub-block `i` (0..15):
```
scale_i = (scales_h bits for sub-block i) | (scales_l[i] << 2)
```
Actually each sub-block scale is 6 bits: low 4 from `scales_l[i]`, high 2 from `scales_h`.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for i in 0..15:  // 16 sub-blocks
    sc = extract_6bit_scale(block.scales_h, block.scales_l, i)
    float d_sub = d * sc
    for j in 0..15:  // 16 weights per sub-block
        idx = (i*16 + j) / 2
        bit = (i*16 + j) % 2
        q = bit ? (block.qs[idx] >> 4) : (block.qs[idx] & 0x0F)
        y[i*16 + j] = kvalues_iq4nl[q] * d_sub
```

---

## 4. IQ2_XXS (type 16) — 2.0625 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 66 bytes

```
struct block_iq2_xxs {
    ggml_half d;               // offset 0:   2 bytes, fp16 scale
    uint16_t  qs[QK_K/8];     // offset 2:  64 bytes, grid indices + sign encoding
};
static_assert(sizeof(block_iq2_xxs) == 66);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–65 | 64B | `qs` — 32 × uint16, each encoding 8 weights |

**Packing (each uint16 at qs[i]):**
- Bits 0–8: index into `iq2xxs_grid[256]` (9-bit grid index)
- Bits 9–15: sign encoding for 8 weights (7 bits, XOR-packed)

Each grid entry is a `uint64_t` that holds 8 × 2-bit values (each 0x08, 0x19, or 0x2b = {-2, -1, +1} in 2-bit).

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for i in 0..31:  // 32 groups of 8
    grid_idx = block.qs[i] & 0x1FF     // 9-bit grid index
    sign_enc = block.qs[i] >> 9         // 7-bit sign encoding
    grid64 = iq2xxs_grid[grid_idx]      // uint64 with 8 × 2-bit values
    // Extract signs using ksigns_iq2xs table
    signs = ksigns_iq2xs[sign_enc]
    for j in 0..7:
        val_2bit = (grid64 >> (j*8)) & 0xFF  // extract 8-bit chunk
        y[i*8 + j] = d * sign_extend_2bit(val_2bit, signs[j])
```

**Key tables (in ggml-common.h):**
- `iq2xxs_grid[256]` — 256 uint64 entries, each encoding 8 × 2-bit absolute values
- `ksigns_iq2xs[128]` — 128 sign-mapping entries

---

## 5. IQ2_XS (type 17) — 2.3125 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 74 bytes

```
struct block_iq2_xs {
    ggml_half d;               // offset 0:   2 bytes, fp16 scale
    uint16_t  qs[QK_K/8];     // offset 2:  64 bytes, grid indices + sign encoding
    uint8_t   scales[QK_K/32]; // offset 66: 8 bytes, 4-bit sub-block scales
};
static_assert(sizeof(block_iq2_xs) == 74);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–65 | 64B | `qs` — 32 × uint16, grid indices + sign encoding (same as IQ2_XXS) |
| 66–73 | 8B | `scales` — 16 × 4-bit sub-block scales (2 per byte) |

**Sub-block structure:** 8 groups of 32 weights, each group has a 4-bit scale.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for g in 0..7:  // 8 groups of 32
    sc4 = (g%2 == 0) ? (block.scales[g/2] & 0x0F) : (block.scales[g/2] >> 4)
    float d_group = d * sc4
    for i in 0..3:  // 3 uint16 per group = 24 weights + implicit 8 more
        idx = g*4 + i
        grid_idx = block.qs[idx] & 0x1FF
        sign_enc = block.qs[idx] >> 9
        grid64 = iq2xs_grid[grid_idx]
        signs = ksigns_iq2xs[sign_enc]
        for j in 0..7:
            y[g*32 + i*8 + j] = d_group * extract_signed_2bit(grid64, j, signs)
```

**Note:** Uses `iq2xs_grid[512]` (512 entries, larger than IQ2_XXS's 256).

---

## 6. IQ2_S (type 22) — 2.5625 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 82 bytes

```
struct block_iq2_s {
    ggml_half d;               // offset 0:   2 bytes, fp16 scale
    uint8_t   qs[QK_K/4];     // offset 2:  64 bytes, 2-bit quants (4 per byte)
    uint8_t   qh[QK_K/32];    // offset 66: 8 bytes, high bits + grid shift
    uint8_t   scales[QK_K/32]; // offset 74: 8 bytes, 4-bit sub-block scales
};
static_assert(sizeof(block_iq2_s) == 82);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–65 | 64B | `qs` — 256 × 2-bit quantized values (4 per byte) |
| 66–73 | 8B | `qh` — high bits for grid selection (1 bit per weight pair) |
| 74–81 | 8B | `scales` — 16 × 4-bit sub-block scales |

**Packing:** Each byte of `qs` holds 4 × 2-bit values (bits 0–1, 2–3, 4–5, 6–7).

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for g in 0..7:  // 8 groups of 32
    sc4 = extract_4bit_scale(block.scales, g)
    float d_group = d * sc4
    for j in 0..31:
        // 2-bit value from qs, high bit from qh
        q2 = (block.qs[g*32/4 + j/4] >> ((j%4)*2)) & 3
        qh_bit = (block.qh[(g*32+j)/8] >> ((g*32+j)%8)) & 1
        // Use iq2s_grid: index = q2 | (qh_bit << 2)
        y[g*32 + j] = d_group * iq2s_grid[q2 | (qh_bit << 2)]
```

---

## 7. IQ3_XXS (type 18) — 3.0625 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 98 bytes

```
struct block_iq3_xxs {
    ggml_half d;               // offset 0:   2 bytes, fp16 scale
    uint8_t   qs[3*QK_K/8];   // offset 2:  96 bytes, 3-bit grid indices
};
static_assert(sizeof(block_iq3_xxs) == 98);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–97 | 96B | `qs` — 32 × 3-byte (24-bit) grid indices, each encoding 8 weights |

**Packing:** Each 24-bit value encodes 8 × 3-bit indices. The 24 bits are split as 8 groups of 3 bits each.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for i in 0..31:  // 32 groups of 8
    // Read 24-bit grid index from qs[i*3..i*3+2]
    grid_idx = qs[i*3] | (qs[i*3+1] << 8) | (qs[i*3+2] << 16)  // 24 bits
    // Look up in iq3xxs_grid (256 entries, each uint64 with 8 × 3-bit values)
    grid64 = iq3xxs_grid[grid_idx & 0xFF]  // low 8 bits = grid entry
    // High 16 bits = sign/magnitude encoding
    for j in 0..7:
        val_3bit = (grid64 >> (j*8)) & 0xFF
        y[i*8 + j] = d * val_3bit  // sign is embedded in the grid
```

**Key table:** `iq3xxs_grid[256]` — each uint64 encodes 8 signed 3-bit values.

---

## 8. IQ3_S (type 21) — 3.4375 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 110 bytes

```
#define IQ3S_N_SCALE (QK_K/64)  // = 4
struct block_iq3_s {
    ggml_half d;                    // offset 0:   2 bytes, fp16 scale
    uint8_t   qs[QK_K/4];          // offset 2:  64 bytes, low 2 bits of 3-bit quants
    uint8_t   qh[QK_K/32];         // offset 66: 8 bytes, high bit of each quant
    uint8_t   signs[QK_K/8];       // offset 74: 32 bytes, sign bits
    uint8_t   scales[IQ3S_N_SCALE]; // offset 106: 4 bytes, 2-bit sub-block scales
};
static_assert(sizeof(block_iq3_s) == 110);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–65 | 64B | `qs` — low 2 bits of each 3-bit quant (4 per byte) |
| 66–73 | 8B | `qh` — high bit of each quant (1 per weight) |
| 74–105 | 32B | `signs` — sign bit per weight (1 per weight, packed) |
| 106–109 | 4B | `scales` — 16 × 2-bit sub-block scales (8 per byte) |

**Sub-block structure:** 16 sub-blocks of 16 weights, each with a 2-bit scale.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for g in 0..15:  // 16 sub-blocks
    sc2 = extract_2bit_scale(block.scales, g)  // 0..3
    float d_sub = d * (sc2 - 2)  // offset binary
    for j in 0..15:
        idx = g*16 + j
        q2 = (block.qs[idx/4] >> ((idx%4)*2)) & 3
        q_high = (block.qh[idx/8] >> (idx%8)) & 1
        q3 = q2 | (q_high << 2)  // 0..7
        sign = (block.signs[idx/8] >> (idx%8)) & 1
        val = sign ? q3 : -q3
        y[idx] = d_sub * val
```

---

## 9. IQ1_S (type 19) — 1.5625 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 50 bytes

```
struct block_iq1_s {
    ggml_half d;               // offset 0:   2 bytes, fp16 scale
    uint8_t   qs[QK_K/8];     // offset 2:  32 bytes, grid indices (8 per byte)
    uint16_t  qh[QK_K/32];    // offset 34: 16 bytes, grid shift bits + high index bits
};
static_assert(sizeof(block_iq1_s) == 50);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–1 | 2B | `d` — fp16 scale |
| 2–33 | 32B | `qs` — low 8 bits of grid index per weight (1 byte per weight) |
| 34–49 | 16B | `qh` — per-weight: bit 0 = grid shift, bits 1–4 = high 4 bits of grid index |

**Grid:** Each weight selects from a 2-level grid (two possible values per index). The `qh` bits select which level.

**Dequant formula:**
```
d = fp16_to_fp32(block.d)
for i in 0..255:
    grid_idx = block.qs[i] | ((block.qh[i/16] >> ((i%16)*4 + 1)) & 0xF) << 8
    shift = (block.qh[i/16] >> ((i%16)*4)) & 1
    // iq1s_grid has 256 entries, each with 2 values
    val = iq1s_grid[grid_idx][shift]
    y[i] = d * val
```

**Key table:** `iq1s_grid[256]` — each entry has two possible values selected by the shift bit.

---

## 10. IQ1_M (type 29) — 1.75 bpw

- **Block size:** 256 weights (`QK_K = 256`)
- **Total bytes per block:** 56 bytes

```
struct block_iq1_m {
    uint8_t qs[QK_K/8];      // offset 0:  32 bytes, grid index low 8 bits
    uint8_t qh[QK_K/16];     // offset 32: 16 bytes, grid index high 3 bits + shift bit
    uint8_t scales[QK_K/32]; // offset 48:  8 bytes, 3-bit block scales
};
static_assert(sizeof(block_iq1_m) == 56);
```

**Byte layout:**
| Offset | Size | Content |
|--------|------|---------|
| 0–31 | 32B | `qs` — low 8 bits of grid index per weight |
| 32–47 | 16B | `qh` — high 3 bits of grid index + shift bit per weight (4 bits each) |
| 48–55 | 8B | `scales` — 16 × 3-bit sub-block scales |

**Note:** No global fp16 scale `d`. The block relies entirely on the 3-bit sub-block scales.

**Dequant formula:**
```
for g in 0..15:  // 16 sub-blocks of 16
    sc3 = extract_3bit_scale(block.scales, g)
    for j in 0..15:
        idx = g*16 + j
        grid_idx = block.qs[idx] | ((block.qh[idx/2] >> ((idx%2)*4)) & 0x7) << 8
        shift = (block.qh[idx/2] >> ((idx%2)*4 + 3)) & 1
        val = iq1m_grid[grid_idx][shift]
        y[idx] = sc3 * val
```

**Key table:** `iq1m_grid` — 512-entry grid (256 indices × 2 levels).

---

## Summary Table

| Format | Type ID | BPW | Block | Bytes/Block | Scale | Sub-blocks | Grid/Table |
|--------|---------|-----|-------|-------------|-------|------------|------------|
| Q1_0 | 41 | 1.008 | 128 | 18 | fp16 | 1 | sign bit |
| IQ1_S | 19 | 1.5625 | 256 | 50 | fp16 | 1 | iq1s_grid[256] |
| IQ1_M | 29 | 1.75 | 256 | 56 | 3-bit scales | 16×16 | iq1m_grid |
| IQ2_XXS | 16 | 2.0625 | 256 | 66 | fp16 | 1 | iq2xxs_grid[256] + ksigns |
| IQ2_XS | 17 | 2.3125 | 256 | 74 | fp16 + 4-bit | 8×32 | iq2xs_grid[512] + ksigns |
| IQ2_S | 22 | 2.5625 | 256 | 82 | fp16 + 4-bit | 8×32 | iq2s_grid + qs/qh |
| IQ3_XXS | 18 | 3.0625 | 256 | 98 | fp16 | 1 | iq3xxs_grid[256] |
| IQ3_S | 21 | 3.4375 | 256 | 110 | fp16 + 2-bit | 16×16 | qs/qh/signs/scales |
| IQ4_NL | 20 | 4.5 | 32 | 18 | fp16 | 1 | kvalues_iq4nl[16] |
| IQ4_XS | 23 | 4.328 | 256 | 146 | fp16 + 6-bit | 16×16 | kvalues_iq4nl[16] |

## GPU Shader Implementation Notes

1. **Lookup tables are essential.** All IQ formats use predefined grids (`iq2xxs_grid`, `iq2xs_grid`, `iq3xxs_grid`, `kvalues_iq4nl`). These must be declared as `const` arrays in the shader or passed as SSBOs.

2. **Sign handling varies.** IQ2_XXS/IQ2_XS encode signs in the `qs` uint16. IQ3_S stores signs separately. Q1_0 uses a single bit per weight.

3. **Sub-block scales.** IQ2_XS uses 4-bit scales. IQ3_S uses 2-bit scales. IQ4_XS uses 6-bit scales (split across `scales_h` and `scales_l`).

4. **No global scale for IQ1_M.** It uses only per-sub-block 3-bit scales, unlike IQ1_S which has a single fp16 `d`.

5. **Bit extraction patterns.** For IQ2 formats, the grid index is 9 bits (IQ2_XXS/IQ2_XS) or 10+ bits (IQ2_S). Signs are XOR-encoded with the grid pattern.
