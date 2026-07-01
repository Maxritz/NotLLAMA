# DARS IQ Dequantization Analysis

Source: C:\Users\rr\Desktop\llama.cpp-ROCM-Test\darsollama\some more\

## IQ1_S (1.5625 BPW)

### Block Structure (82 bytes per 256 weights)
```c
typedef struct {
    half     d;              // super-block scale (fp16)
    uint16_t qs[QK_K / 8];  // 32 entries: 11-bit grid idx + 1-bit sign delta
    uint16_t qh[QK_K / 32]; // 8 entries: sub-block scales (3 bits each, packed)
} block_iq1_s;
```

### Dequantization Algorithm
1. Extract 11-bit grid index from `qs[i] & 0x7FF`
2. Extract 1-bit sign delta from `qs[i] >> 11`
3. Look up `iq1s_grid_d[2048]` (16KB constant table) — each entry is 8 ternary values
4. Decode 3-bit sub-block deltas from `qh` entries (4 deltas per uint16_t)
5. Scale: `d * (delta + 1)`, apply sign flip

## IQ2_XXS (2.0625 BPW)

### Block Structure (66 bytes per 256 weights)
```c
typedef struct {
    half     d;              // super-block scale
    uint16_t qs[32];        // low byte = grid index (8 bits), high byte = signs
} block_iq2_xxs;
```

### Dequantization Algorithm
1. Extract 8-bit grid index from `qs[i] & 0xFF`
2. Extract 8-bit sign byte from `qs[i] >> 8`
3. Look up `kiq2xxs_grid[256]` (2KB constant table) — each entry provides 8 weight magnitudes
4. Magnitudes are from set {1, 3, 5, 7}
5. Apply signs: `d * sign(j) * magnitude`

## IQ2_XS (2.3125 BPW)

### Block Structure (74 bytes per 256 weights)
```c
typedef struct {
    half     d;              // super-block scale
    half     scales[4];      // per-sub-block fp16 scales (each covers 64 weights)
    uint16_t qs[32];        // 9-bit grid index + 7-bit signs
} block_iq2_xs;
```

### Dequantization Algorithm
1. Extract 9-bit grid index from `qs[i] & 0x1FF`
2. Extract 7-bit sign byte from `qs[i] >> 9`
3. Look up 512-entry grid (4KB constant table)
4. Apply two-level scale: `d_super * d_sub * magnitude`

## Q1_0 (1.5625 BPW)

### Block Structure (6 bytes per 32 weights)
```c
typedef struct {
    uint16_t d;         // delta (fp16)
    uint8_t qs[4];     // packed 1-bit quants
} block_q1_0;
```

### Dequantization Algorithm
1. Extract single bit from packed `qs` bytes
2. Map: bit 0 → -d, bit 1 → +d (symmetric binary quantization)

## Common GLSL Dequant Pattern

All IQ formats share this core algorithm:
```
1. Extract N-bit index from packed qs bits
2. Lookup grid table (256/512/2048 entries, stored as uint64_t)
3. Extract per-weight signs from remaining bits
4. Apply scale hierarchy (d_super * d_sub * d_delta)
5. FMA with input and reduce across workgroup
```

## Grid Table Sizes for Vulkan

| Format | Grid Entries | Table Size | Notes |
|--------|-------------|------------|-------|
| IQ2_XXS | 256 | 2 KB | 8-bit index, uint64_t entries |
| IQ2_XS | 512 | 4 KB | 9-bit index, uint64_t entries |
| IQ1_S | 2048 | 16 KB | 11-bit index, uint64_t entries |

All fit easily in a single Vulkan descriptor set buffer.

## RDNA4 Wavefront Alignment
- BLOCK_SIZE = 256 = 8 × Wave32 (optimal for RDNA4)
- One workgroup per output row
- Shared memory tree reduction: `shared float shmem[256]` + `barrier()`
- Grid tables in constant/readonly buffer memory
