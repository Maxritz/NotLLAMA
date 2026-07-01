# GGML Quantization Formats — GPU Dequantization Reference

Extracted from ggml master branch (2026-06-29). All formulas are from the reference C implementations in `ggml-quants.c` and struct definitions in `ggml-common.h`.

## Constants

```
QK_K = 256          // super-block size for K formats
QK4_NL = 32         // block size for IQ4_NL
QK_MXFP4 = 32       // block size for MXFP4
QK_NVFP4 = 64       // block size for NVFP4
QK_NVFP4_SUB = 16   // sub-block size for NVFP4 per-group scales
K_SCALE_SIZE = 12    // scale bytes for Q4_K, Q5_K
IQ3S_N_SCALE = 4     // QK_K/64
NGRID_IQ1S = 2048    // number of IQ1_S grid entries
IQ1S_DELTA = 0.125f
IQ1M_DELTA = 0.125f
```

## Lookup Tables

### kvalues_iq4nl[16] — Non-linear 4-bit value lookup
```c
{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
```

### kvalues_mxfp4[16] — E2M1 values (doubled)
```c
{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
```

### kmask_iq2xs[8]
```c
{1, 2, 4, 8, 16, 32, 64, 128}
```

### ksigns_iq2xs[128] — 7-bit index → 8-byte sign mask
```c
{0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,
 144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,
 160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,
 48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,
 192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,
 80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,
 96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,
 240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255}
```

### ksigns64[128] — 7-bit index → 64-bit sign mask (for 8 groups of 8)
```c
{0x0000000000000000, 0xff000000000000ff, 0xff0000000000ff00, 0x000000000000ffff,
 0xff00000000ff0000, 0x0000000000ff00ff, 0x0000000000ffff00, 0xff00000000ffffff,
 0xff000000ff000000, 0x00000000ff0000ff, 0x00000000ff00ff00, 0xff000000ff00ffff,
 0x00000000ffff0000, 0xff000000ffff00ff, 0xff000000ffffff00, 0x00000000ffffffff,
 0xff0000ff00000000, 0x000000ff000000ff, 0x000000ff0000ff00, 0xff0000ff0000ffff,
 0x000000ff00ff0000, 0xff0000ff00ff00ff, 0xff0000ff00ffff00, 0x000000ff00ffffff,
 0x000000ffff000000, 0xff0000ffff0000ff, 0xff0000ffff00ff00, 0x000000ffff00ffff,
 0xff0000ffffff0000, 0x000000ffffff00ff, 0x000000ffffffff00, 0xff0000ffffffffff,
 0xff00ff0000000000, 0x0000ff00000000ff, 0x0000ff000000ff00, 0xff00ff000000ffff,
 0x0000ff0000ff0000, 0xff00ff0000ff00ff, 0xff00ff0000ffff00, 0x0000ff0000ffffff,
 0x0000ff00ff000000, 0xff00ff00ff0000ff, 0xff00ff00ff00ff00, 0x0000ff00ff00ffff,
 0xff00ff00ffff0000, 0x0000ff00ffff00ff, 0x0000ff00ffffff00, 0xff00ff00ffffffff,
 0x0000ffff00000000, 0xff00ffff000000ff, 0xff00ffff0000ff00, 0x0000ffff0000ffff,
 0xff00ffff00ff0000, 0x0000ffff00ff00ff, 0x0000ffff00ffff00, 0xff00ffff00ffffff,
 0xff00ffffff000000, 0x0000ffffff0000ff, 0x0000ffffff00ff00, 0xff00ffffff00ffff,
 0x0000ffffffff0000, 0xff00ffffffff00ff, 0xff00ffffffffff00, 0x0000ffffffffffff,
 0xffff000000000000, 0x00ff0000000000ff, 0x00ff00000000ff00, 0xffff00000000ffff,
 0x00ff000000ff0000, 0xffff000000ff00ff, 0xffff000000ffff00, 0x00ff000000ffffff,
 0x00ff0000ff000000, 0xffff0000ff0000ff, 0xffff0000ff00ff00, 0x00ff0000ff00ffff,
 0xffff0000ffff0000, 0x00ff0000ffff00ff, 0x00ff0000ffffff00, 0xffff0000ffffffff,
 0x00ff00ff00000000, 0xffff00ff000000ff, 0xffff00ff0000ff00, 0x00ff00ff0000ffff,
 0xffff00ff00ff0000, 0x00ff00ff00ff00ff, 0x00ff00ff00ffff00, 0xffff00ff00ffffff,
 0xffff00ffff000000, 0x00ff00ffff0000ff, 0x00ff00ffff00ff00, 0xffff00ffff00ffff,
 0x00ff00ffffff0000, 0xffff00ffffff00ff, 0xffff00ffffffff00, 0x00ff00ffffffffff,
 0x00ffff0000000000, 0xffffff00000000ff, 0xffffff000000ff00, 0x00ffff000000ffff,
 0xffffff0000ff0000, 0x00ffff0000ff00ff, 0x00ffff0000ffff00, 0xffffff0000ffffff,
 0xffffff00ff000000, 0x00ffff00ff0000ff, 0x00ffff00ff00ff00, 0xffffff00ff00ffff,
 0x00ffff00ffff0000, 0xffffff00ffff00ff, 0xffffff00ffffff00, 0x00ffff00ffffffff,
 0xffffffff00000000, 0x00ffffff000000ff, 0x00ffffff0000ff00, 0xffffffff0000ffff,
 0x00ffffff00ff0000, 0xffffffff00ff00ff, 0xffffffff00ffff00, 0x00ffffff00ffffff,
 0x00ffffffff000000, 0xffffffffff0000ff, 0xffffffffff00ff00, 0x00ffffffff00ffff,
 0xffffffffffff0000, 0x00ffffffffff00ff, 0x00ffffffffffff00, 0xffffffffffffffff}
```

### Grid Tables

- **iq2xxs_grid[256]** — 256 × uint64 (8 bytes each). Each entry encodes 8 quantized 2-bit values.
- **iq2xs_grid[512]** — 512 × uint64. Each entry encodes 8 quantized 2-bit values.
- **iq2s_grid[1024]** — 1024 × uint64. Each entry encodes 8 quantized 2-bit values.
- **iq3xxs_grid[256]** — 256 × uint32. Each entry encodes 4 quantized 3-bit values.
- **iq3s_grid[512]** — 512 × uint32. Each entry encodes 4 quantized 3-bit values.
- **iq1s_grid[2048]** — 2048 × uint64. Each entry encodes 8 quantized 1-bit values (as int8: -1, 0, +1).

**Grid tables are too large to embed here.** They must be copied from `ggml-common.h` into GPU shared memory or constant buffers. For GLSL, declare them as `const uint64_t[]` or `const uint32_t[]` arrays.

---

## Format Specifications

### 1. BF16 (type ID 30)

| Property | Value |
|----------|-------|
| Block size | 1 element |
| Bytes/block | 2 |
| Elements/block | 1 |

**Byte layout:**
```
offset 0: uint16 raw bits
```

**Dequantization:**
```glsl
float dequant_bf16(uint16 raw) {
    return uintBitsToFloat(uint(raw) << 16);
}
```

---

### 2. Q1_0 (type ID 41)

| Property | Value |
|----------|-------|
| Block size (QK1_0) | 128 |
| Bytes/block | 2 + 16 = 18 |
| BPW | 1.016 |

**Byte layout:**
```
offset 0:   ggml_half d        (2 bytes, fp16 scale)
offset 2:   uint8_t qs[16]     (16 bytes, 1 bit per element)
```

**Dequantization:**
```glsl
float d = fp16_to_float(x.d);
float neg_d = -d;
for (int j = 0; j < 128; ++j) {
    int byte_index = j / 8;
    int bit_offset = j % 8;
    uint bit = (x.qs[byte_index] >> bit_offset) & 1;
    y[j] = bit != 0u ? d : neg_d;
}
```

---

### 3. IQ4_NL (type ID 20)

| Property | Value |
|----------|-------|
| Block size (QK4_NL) | 32 |
| Bytes/block | 2 + 16 = 18 |
| BPW | 4.5 |

**Byte layout:**
```
offset 0:  ggml_half d        (2 bytes, fp16 scale)
offset 2:  uint8_t qs[16]     (16 bytes, 2 nibbles per byte)
```

**Dequantization:**
```glsl
float d = fp16_to_float(x.d);
for (int j = 0; j < 16; ++j) {
    y[j       ] = d * kvalues_iq4nl[x.qs[j] & 0xf];
    y[j + 16  ] = d * kvalues_iq4nl[x.qs[j] >>  4];
}
```

---

### 4. IQ4_XS (type ID 23)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 2 + 4 + 128 = 136 |
| BPW | ~4.25 |

**Byte layout:**
```
offset 0:    ggml_half d           (2 bytes, fp16 super-block scale)
offset 2:    uint16_t scales_h     (2 bytes, high 2 bits of each 6-bit scale)
offset 4:    uint8_t scales_l[4]   (4 bytes, low 4 bits of each 6-bit scale)
offset 8:    uint8_t qs[128]       (128 bytes, 2 nibbles per byte)
```

**Dequantization:**
```glsl
float d = fp16_to_float(x.d);
for (int ib = 0; ib < 8; ++ib) {
    // Extract 6-bit scale for sub-block ib
    int ls = ((x.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x.scales_h >> 2*ib) & 3) << 4);
    float dl = d * float(ls - 32);
    for (int j = 0; j < 16; ++j) {
        y[j      ] = dl * kvalues_iq4nl[x.qs[ib*16 + j] & 0xf];
        y[j + 16 ] = dl * kvalues_iq4nl[x.qs[ib*16 + j] >>  4];
    }
}
```

---

### 5. IQ3_S (type ID 21)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 64 + 8 + 32 + 4 = 110 |
| BPW | 3.44 |

**Byte layout:**
```
offset 0:    ggml_half d           (2 bytes)
offset 2:    uint8_t qs[64]        (64 bytes, 2-bit quants)
offset 66:   uint8_t qh[8]         (8 bytes, high bit of grid index)
offset 74:   uint8_t signs[32]     (32 bytes, 7-bit sign index per sub-block)
offset 106:  uint8_t scales[4]     (4 bytes, 4-bit scale per pair of 32-blocks)
```

**Dequantization (per 32-block pair, ib32 = 0,2,4,6):**
```glsl
float d = fp16_to_float(x.d);
for (int ib32 = 0; ib32 < 8; ib32 += 2) {
    float db1 = d * float(1 + 2*(x.scales[ib32/2] & 0xf));
    float db2 = d * float(1 + 2*(x.scales[ib32/2] >> 4));
    for (int l = 0; l < 4; ++l) {
        // Grid index = qs[2*l] | ((qh[0] << (8-2*l)) & 0x100)
        uint idx1 = x.qs[ib32*4 + 2*l] | ((x.qh[ib32/2*4 + 0] << (8-2*l)) & 0x100);
        uint idx2 = x.qs[ib32*4 + 2*l+1] | ((x.qh[ib32/2*4 + 0] << (7-2*l)) & 0x100);
        const uint8_t* grid1 = iq3s_grid[idx1]; // 4 bytes
        const uint8_t* grid2 = iq3s_grid[idx2];
        uint8_t sign_byte = x.signs[ib32*4 + l];
        for (int j = 0; j < 4; ++j) {
            y[l*8 + j   ] = db1 * grid1[j] * ((sign_byte & kmask_iq2xs[j  ]) != 0 ? -1.0 : 1.0);
            y[l*8 + j+4 ] = db1 * grid2[j] * ((sign_byte & kmask_iq2xs[j+4]) != 0 ? -1.0 : 1.0);
        }
    }
    // Second half of pair uses db2, qh[1]
    // ... (same pattern with qh byte 1)
}
```

---

### 6. IQ3_XXS (type ID 18)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 96 = 98 |
| BPW | 3.06 |

**Byte layout:**
```
offset 0:   ggml_half d           (2 bytes)
offset 2:   uint8_t qs[96]        (96 bytes, 3-bit quants in 24-bit groups)
```

**Dequantization (per 32-block, ib32 = 0..7):**
```c
// From qs: first 64 bytes = quants, last 32 bytes = scales_and_signs
// scales_and_signs[4*ib32..4*ib32+3] is a uint32 with:
//   bits [28..31] = 4-bit scale
//   bits [0..27]  = 7-bit sign index per sub-block (4 sub-blocks × 7 bits)
const uint8_t * scales_and_signs = qs + QK_K/4;  // offset 66
memcpy(&aux32, scales_and_signs + 4*ib32, 4);
float db = d * (0.5 + (aux32 >> 28)) * 0.5;
for (int l = 0; l < 4; ++l) {
    uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
    const uint8_t* grid1 = iq3xxs_grid[qs[2*l+0]];  // 4 bytes
    const uint8_t* grid2 = iq3xxs_grid[qs[2*l+1]];
    for (int j = 0; j < 4; ++j) {
        y[j  ] = db * grid1[j] * ((signs & kmask_iq2xs[j  ]) != 0 ? -1.0 : 1.0);
        y[j+4] = db * grid2[j] * ((signs & kmask_iq2xs[j+4]) != 0 ? -1.0 : 1.0);
    }
    y += 8;
}
```

---

### 7. IQ2_S (type ID 22)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 64 + 8 + 8 + 8 = 90 |
| BPW | 2.56 |

**Byte layout:**
```
offset 0:    ggml_half d           (2 bytes)
offset 2:    uint8_t qs[64]        (64 bytes, 2-bit quants)
offset 66:   uint8_t qh[8]         (8 bytes, high 2 bits of grid index)
offset 74:   uint8_t signs[32]     (32 bytes, sign bytes)  [Note: qs+QK_K/8 in C]
offset 106:  (signs are at qs + QK_K/8 = offset 66)
```

Actually from the struct: `qs[QK_K/4]=64`, `qh[QK_K/32]=8`, `scales[QK_K/32]=8`. Total: 2+64+8+8=82. Let me recalculate:

```c
typedef struct {
    ggml_half d;           // 2 bytes
    uint8_t qs[QK_K/4];   // 64 bytes
    uint8_t qh[QK_K/32];  // 8 bytes  (high 2 bits of grid index)
    uint8_t scales[QK_K/32]; // 8 bytes (4-bit scale per sub-block)
} block_iq2_s;
// Total: 2 + 64 + 8 + 8 = 82 bytes
```

**Dequantization (per 32-block):**
```c
float d = fp16_to_float(x.d);
const uint8_t * qs = x.qs;       // offset 2
const uint8_t * qh = x.qh;       // offset 66
const uint8_t * signs = qs + QK_K/8;  // points into qs at offset 2+32=34
// Wait — in the C code: signs = qs + QK_K/8 = qs + 32
// But qs is 64 bytes, so signs = qs[32..63]

for (int ib32 = 0; ib32 < 8; ++ib32) {
    db[0] = d * (0.5 + (x.scales[ib32] & 0xf)) * 0.25;
    db[1] = d * (0.5 + (x.scales[ib32] >>  4)) * 0.25;
    for (int l = 0; l < 4; ++l) {
        float dl = db[l/2];
        // Grid index = qs[l] | (qh[ib32] << (8-2*l) & 0x300)
        uint idx = qs[l] | ((qh[ib32] << (8-2*l)) & 0x300);
        const uint8_t* grid = iq2s_grid[idx]; // 8 bytes
        for (int j = 0; j < 8; ++j) {
            y[j] = dl * grid[j] * ((signs[l] & kmask_iq2xs[j]) != 0 ? -1.0 : 1.0);
        }
        y += 8;
    }
    qs += 4;
    signs += 4;
}
```

---

### 8. IQ2_XS (type ID 17)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 64 + 8 = 74 |
| BPW | 2.31 |

**Byte layout:**
```
offset 0:   ggml_half d           (2 bytes)
offset 2:   uint16_t qs[32]       (64 bytes, 9-bit grid index + 7-bit sign packed)
offset 66:  uint8_t scales[8]     (8 bytes, 4-bit sub-scale per 32-block)
```

**Dequantization (per 32-block):**
```c
float d = fp16_to_float(x.d);
for (int ib32 = 0; ib32 < 8; ++ib32) {
    db[0] = d * (0.5 + (x.scales[ib32] & 0xf)) * 0.25;
    db[1] = d * (0.5 + (x.scales[ib32] >>  4)) * 0.25;
    for (int l = 0; l < 4; ++l) {
        // qs is uint16 array. Each element: low 9 bits = grid index, high 7 bits = sign index
        uint16_t val = x.qs[4*ib32 + l];
        const uint8_t* grid = iq2xs_grid[val & 511];  // 8 bytes
        uint8_t signs = ksigns_iq2xs[val >> 9];
        for (int j = 0; j < 8; ++j) {
            y[j] = db[l/2] * grid[j] * ((signs & kmask_iq2xs[j]) != 0 ? -1.0 : 1.0);
        }
        y += 8;
    }
}
```

---

### 9. IQ2_XXS (type ID 16)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 64 = 66 |
| BPW | 2.06 |

**Byte layout:**
```
offset 0:   ggml_half d           (2 bytes)
offset 2:   uint16_t qs[32]       (64 bytes, packed grid+scale+sign)
```

**Dequantization (per 32-block):**
```c
float d = fp16_to_float(x.d);
for (int ib32 = 0; ib32 < 8; ++ib32) {
    // Read 8 bytes (2 × uint32) from qs at offset 4*ib32
    uint32_t aux32[2];
    memcpy(aux32, x.qs + 4*ib32, 8);
    float db = d * (0.5 + (aux32[1] >> 28)) * 0.25;
    for (int l = 0; l < 4; ++l) {
        // Low byte of aux32[l] = grid index into iq2xxs_grid (256 entries)
        const uint8_t* grid = iq2xxs_grid[aux8[l]];  // 8 bytes
        uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
        for (int j = 0; j < 8; ++j) {
            y[j] = db * grid[j] * ((signs & kmask_iq2xs[j]) != 0 ? -1.0 : 1.0);
        }
        y += 8;
    }
}
```

---

### 10. IQ1_S (type ID 19)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 32 + 16 = 50 |
| BPW | 1.56 |

**Byte layout:**
```
offset 0:   ggml_half d           (2 bytes)
offset 2:   uint8_t qs[32]        (32 bytes, low 8 bits of grid index)
offset 34:  uint16_t qh[16]       (16 bytes, high 3 bits of grid index + scale + delta)
```

**Dequantization (per 32-block):**
```c
float d = fp16_to_float(x.d);
for (int ib = 0; ib < 8; ++ib) {
    float dl = d * (2*((x.qh[ib] >> 12) & 7) + 1);
    float delta = (x.qh[ib] & 0x8000) != 0 ? -0.125 : 0.125;
    for (int l = 0; l < 4; ++l) {
        // Grid index = qs[l] | (((qh[ib] >> 3*l) & 7) << 8)
        uint idx = x.qs[4*ib + l] | (((x.qh[ib] >> 3*l) & 7) << 8);
        const int8_t* grid = iq1s_grid[idx];  // 8 bytes (int8: -1, 0, +1)
        for (int j = 0; j < 8; ++j) {
            y[j] = dl * (float(grid[j]) + delta);
        }
        y += 8;
    }
    x.qs += 4;
}
```

---

### 11. IQ1_M (type ID 29)

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 0 (no fp16 d) + 32 + 16 + 8 = 56 |
| BPW | 1.75 |

**Byte layout:**
```
offset 0:   uint8_t qs[32]        (32 bytes, low 8 bits of grid index)
offset 32:  uint8_t qh[16]        (16 bytes, high 3 bits + delta bit per sub-block)
offset 48:  uint8_t scales[8]     (8 bytes, 3-bit block scale, packed into uint16)
```

**Scale reconstruction:**
```c
// Reconstruct fp16 scale from scales bytes:
const uint16_t * sc = (const uint16_t *)x.scales;
uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                     ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
float d = fp16_to_float(scale_u16);
```

**Dequantization (per 32-block):**
```c
for (int ib = 0; ib < 8; ++ib) {
    float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
    float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
    uint idx[4];
    idx[0] = x.qs[0] | ((x.qh[0] << 8) & 0x700);
    idx[1] = x.qs[1] | ((x.qh[0] << 4) & 0x700);
    idx[2] = x.qs[2] | ((x.qh[1] << 8) & 0x700);
    idx[3] = x.qs[3] | ((x.qh[1] << 4) & 0x700);
    float delta[4];
    delta[0] = (x.qh[0] & 0x08) != 0 ? -0.125 : 0.125;
    delta[1] = (x.qh[0] & 0x80) != 0 ? -0.125 : 0.125;
    delta[2] = (x.qh[1] & 0x08) != 0 ? -0.125 : 0.125;
    delta[3] = (x.qh[1] & 0x80) != 0 ? -0.125 : 0.125;
    for (int l = 0; l < 2; ++l) {
        const int8_t* grid = iq1s_grid[idx[l]];
        for (int j = 0; j < 8; ++j) y[j] = dl1 * (float(grid[j]) + delta[l]);
        y += 8;
    }
    for (int l = 2; l < 4; ++l) {
        const int8_t* grid = iq1s_grid[idx[l]];
        for (int j = 0; j < 8; ++j) y[j] = dl2 * (float(grid[j]) + delta[l]);
        y += 8;
    }
}
```

---

### 12. TQ1_0 (type ID 34) — Ternary 1-bit

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 4 + 48 = 54 |
| BPW | 1.6875 |

**Byte layout:**
```
offset 0:   uint8_t qs[48]        (48 bytes, 5 ternary elements per byte)
offset 48:  uint8_t qh[4]         (4 bytes, 4 extra elements per byte)
offset 52:  ggml_half d           (2 bytes)
```

Note: `qs` has `(256 - 4*256/64) / 5 = (256-16)/5 = 48` bytes. `qh` has `256/64 = 4` bytes.

**Dequantization:**
```c
float d = fp16_to_float(x.d);
const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

// Decode qs: 48 bytes × 5 elements = 240 elements
for (int j = 0; j < 48; ++j) {
    for (int n = 0; n < 5; ++n) {
        uint8_t q = x.qs[j] * pow3[n];
        int16_t xi = (uint16_t(q) * 3) >> 8;  // maps to {-1, 0, +1}
        *y++ = float(xi - 1) * d;
    }
}
// Decode qh: 4 bytes × 4 elements = 16 elements
for (int n = 0; n < 4; ++n) {
    for (int j = 0; j < 4; ++j) {
        uint8_t q = x.qh[j] * pow3[n];
        int16_t xi = (uint16_t(q) * 3) >> 8;
        *y++ = float(xi - 1) * d;
    }
}
```

**GLSL formula for ternary decode:**
```glsl
// Given a byte b and power index n (0-4), decode one ternary value:
// q = b * pow3[n]
// xi = (uint(q) * 3u) >> 8
// value = float(int(xi) - 1) * d
// This maps to values in {-1, 0, +1} * d
```

---

### 13. TQ2_0 (type ID 35) — Ternary 2-bit

| Property | Value |
|----------|-------|
| Block size (QK_K) | 256 |
| Bytes/block | 2 + 64 = 66 |
| BPW | 2.0625 |

**Byte layout:**
```
offset 0:   uint8_t qs[64]        (64 bytes, 4 two-bit elements per byte)
offset 64:  ggml_half d           (2 bytes)
```

**Dequantization:**
```c
float d = fp16_to_float(x.d);
for (int j = 0; j < 64; j += 32) {
    for (int l = 0; l < 4; ++l) {
        for (int m = 0; m < 32; ++m) {
            int8_t q = (x.qs[j + m] >> (l*2)) & 3;
            *y++ = float(q - 1) * d;
        }
    }
}
```

**GLSL:**
```glsl
float d = fp16_to_float(x.d);
for (int j = 0; j < 256; ++j) {
    int byte_idx = j / 4;         // 4 elements per byte
    int sub_idx  = j % 4;
    int q = (x.qs[byte_idx] >> (sub_idx * 2)) & 3;
    y[j] = float(q - 1) * d;
}
```

---

### 14. MXFP4 (type ID 39) — Microscaling FP4

| Property | Value |
|----------|-------|
| Block size (QK_MXFP4) | 32 |
| Bytes/block | 1 + 16 = 17 |
| BPW | 4.25 |

**Byte layout:**
```
offset 0:   uint8_t e             (1 byte, E8M0 shared exponent)
offset 1:   uint8_t qs[16]        (16 bytes, 2 E2M1 values per byte)
```

**Dequantization:**
```c
float d = GGML_E8M0_TO_FP32_HALF(x.e);  // 2^(e - 127)
for (int j = 0; j < 16; ++j) {
    int8_t x0 = kvalues_mxfp4[x.qs[j] & 0x0f];  // E2M1 value (doubled)
    int8_t x1 = kvalues_mxfp4[x.qs[j] >> 4];
    y[j      ] = float(x0) * d;
    y[j + 16 ] = float(x1) * d;
}
```

**E8M0 to float32:** `d = pow(2.0, e - 127)` where e is unsigned 8-bit.

---

### 15. NVFP4 (type ID 40) — NVIDIA FP4

| Property | Value |
|----------|-------|
| Block size (QK_NVFP4) | 64 |
| Sub-block size | 16 |
| Bytes/block | 4 + 32 = 36 |
| BPW | 4.5 |

**Byte layout:**
```
offset 0:   uint8_t d[4]          (4 bytes, UE4M3 scales, one per 16-element sub-block)
offset 4:   uint8_t qs[32]        (32 bytes, packed 4-bit E2M1 values)
```

**Dequantization:**
```c
for (int s = 0; s < 4; ++s) {
    float d = ggml_ue4m3_to_fp32(x.d[s]);  // UE4M3 → float
    float * yb = y + s*16;
    for (int j = 0; j < 8; ++j) {
        int8_t v0 = kvalues_mxfp4[x.qs[s*8 + j] & 0x0f];
        int8_t v1 = kvalues_mxfp4[x.qs[s*8 + j] >> 4];
        yb[j      ] = float(v0) * d;
        yb[j + 8  ] = float(v1) * d;
    }
}
```

**UE4M3 to float32:** Special conversion (8-bit: 1 sign + 4 exponent + 3 mantissa, bias=7). `ggml_ue4m3_to_fp32()` must be implemented as a lookup or bit manipulation.

---

## Summary Table

| Format | Type ID | Block | Bytes | BPW | Grid | Scale | Notes |
|--------|---------|-------|-------|-----|------|-------|-------|
| BF16 | 30 | 1 | 2 | 16 | — | — | `uint16 << 16` |
| Q1_0 | 41 | 128 | 18 | 1.02 | — | fp16 | 1-bit sign |
| IQ4_NL | 20 | 32 | 18 | 4.5 | kvalues_iq4nl | fp16 | NL lookup |
| IQ4_XS | 23 | 256 | 136 | 4.25 | kvalues_iq4nl | fp16 + 6-bit sub | 8 sub-blocks |
| IQ3_S | 21 | 256 | 110 | 3.44 | iq3s_grid[512] | 4-bit sub | signs array |
| IQ3_XXS | 18 | 256 | 98 | 3.06 | iq3xxs_grid[256] | 4-bit sub | packed 24-bit |
| IQ2_S | 22 | 256 | 82 | 2.56 | iq2s_grid[1024] | 4-bit sub | signs |
| IQ2_XS | 17 | 256 | 74 | 2.31 | iq2xs_grid[512] | 4-bit sub | uint16 packed |
| IQ2_XXS | 16 | 256 | 66 | 2.06 | iq2xxs_grid[256] | 4-bit sub | packed uint32 |
| IQ1_S | 19 | 256 | 50 | 1.56 | iq1s_grid[2048] | 3-bit + delta | int8 grid |
| IQ1_M | 29 | 256 | 56 | 1.75 | iq1s_grid[2048] | 3-bit + delta | no fp16 d |
| TQ1_0 | 34 | 256 | 54 | 1.69 | — | fp16 | ternary decode |
| TQ2_0 | 35 | 256 | 66 | 2.06 | — | fp16 | 2-bit ternary |
| MXFP4 | 39 | 32 | 17 | 4.25 | kvalues_mxfp4 | E8M0 shared | microscaling |
| NVFP4 | 40 | 64 | 36 | 4.5 | kvalues_mxfp4 | UE4M3 per-16 | NVIDIA format |

## GPU Implementation Notes

1. **Grid tables are large.** iq1s_grid alone is 2048×8 = 16KB. iq2s_grid is 1024×8 = 8KB. These must reside in shared memory or constant buffers.

2. **ksigns_iq2xs** is 128 bytes — small enough for shared memory.

3. **kvalues_iq4nl** and **kvalues_mxfp4** are 16 bytes each — trivial.

4. **IQ1_M scale reconstruction** requires bit gathering from 4 scale bytes into a uint16, then fp16→fp32. This is the most complex scale extraction.

5. **NVFP4 UE4M3 conversion** requires a dedicated function (8-bit float with bias 7, special handling for NaN/Inf).

6. **All IQ formats** use a common pattern: decode grid index → look up grid values → multiply by scale → apply signs. The grid lookup is the expensive part (shared memory recommended).
