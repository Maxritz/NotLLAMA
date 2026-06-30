# TurboQuant Format Variants

TurboQuant (TQ) is a family of GPU-native quantization formats designed for on-the-fly dequantization in Vulkan compute shaders. All formats follow the **d-last** convention: weights are packed first, then scale factors at the end of each block.

## General Properties

| Property | Value |
|----------|-------|
| Schema version | 1.0 |
| Scale storage | fp16 (16-bit) unless noted |
| Alignment | 128 bytes minimum |
| Block layout | Weights first, scales last (d-last) |
| Dequantization | Per-block, on GPU, fused into GEMM when possible |

## TQ4_128 — TurboQuant 4-bit, 128-element blocks

| Property | Value |
|----------|-------|
| Bits per weight | 4.0 |
| Block size | 128 elements |
| Block bytes | 64 (weights) + 2 (scale fp16) = **66 bytes** |
| Bytes per weight | 0.516 |
| Target | Memory-constrained inference |
| Quality | ~0.5% perplexity degradation vs F16 on typical LLMs |

**Block layout:**
```
Offset  Size   Description
0       2      scale (fp16)
2       64     packed nibbles (2 weights per byte, low nibble = even index)
66      end of block
```

**Packing:** Each byte holds two 4-bit weights. Low nibble = even index, high nibble = odd index. No padding.

**Dequantization:** `float w = (float(nibble) - 8.0) * scale`

## TQ3_128 — TurboQuant 3-bit, 128-element blocks

| Property | Value |
|----------|-------|
| Bits per weight | 3.0 |
| Block size | 128 elements |
| Block bytes | 48 (weights) + 2 (scale fp16) = **50 bytes** |
| Bytes per weight | 0.391 |
| Target | Extreme compression |
| Quality | ~2% perplexity degradation vs F16 |

**Block layout:**
```
Offset  Size   Description
0       2      scale (fp16)
2       48     packed 3-bit weights (see packing below)
50      end of block
```

**Packing:** 3-bit weights are packed tightly. 8 weights occupy 3 bytes (24 bits, using 24 of 24 bits — no waste). Alternatively, 32 weights occupy 12 bytes (96 bits = 32 × 3). Byte order is little-endian within each group.

**Dequantization:** `float w = (float(q3) - 4.0) * scale` (unsigned, range 0–7, midpoint 4)

**Status:** v2 (stubbed, not yet implemented in shader).

## TQ6_64 — TurboQuant 6-bit, 64-element blocks

| Property | Value |
|----------|-------|
| Bits per weight | 6.0 |
| Block size | 64 elements |
| Block bytes | 48 (weights) + 2 (scale fp16) = **50 bytes** |
| Bytes per weight | 0.781 |
| Target | Near-F16 quality |
| Quality | ~0.1% perplexity degradation vs F16 |

**Block layout:**
```
Offset  Size   Description
0       2      scale (fp16)
2       48     packed 6-bit weights (see packing below)
50      end of block
```

**Packing:** 6-bit weights are packed tightly. 4 weights occupy 3 bytes (24 bits = 4 × 6). Byte order is little-endian: weight 0 is in bits 0–5 of byte 0, weight 1 in bits 6–11 spanning bytes 0–1, weight 2 in bits 12–17 spanning bytes 1–2, weight 3 in bits 18–23 spanning bytes 2–3.

**Dequantization:** `float w = (float(q6) - 32.0) * scale` (unsigned, range 0–63, midpoint 32)

## Comparison Table

| Format | Bits | Block | Bytes/Block | Bytes/Weight | Use Case |
|--------|------|-------|-------------|--------------|----------|
| TQ3_128 | 3.0 | 128 | 50 | 0.391 | Extreme compression, draft models |
| TQ4_128 | 4.0 | 128 | 66 | 0.516 | Memory-constrained, large context |
| TQ6_64 | 6.0 | 64 | 50 | 0.781 | Near-lossless, production inference |

## Future Variants (v2)

- **TQ4_256**: 4-bit, 256-element blocks. Better scale amortization. 130 bytes/block.
- **TQ5_128**: 5-bit, 128-element blocks. 82 bytes/block. Midpoint between TQ4 and TQ6.
- **TQ4_AWQ**: Activation-aware TQ4. Sensitivity-weighted rounding for outlier preservation.
- **TQ4_ZP**: TQ4 with zero-point. Signed quantization for asymmetric distributions.
