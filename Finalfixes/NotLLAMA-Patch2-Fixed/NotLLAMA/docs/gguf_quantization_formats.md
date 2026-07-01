# GGUF Quantization Format Reference

Source: ggml-quants.h, ggml-common.h

## Simple Block Quantizations (block_size=32)

| Type | ID | Block Size | Bytes/Block | BPW | Structure |
|------|----|-----------|-------------|-----|-----------|
| Q4_0 | 2 | 32 | 18 | 4.5 | [fp16 d][uint8 qs[16]] |
| Q4_1 | 3 | 32 | 24 | 5.5 | [fp16 d][fp16 m][uint8 qs[16]] |
| Q5_0 | 6 | 32 | 26 | 6.5 | [fp16 d][uint8 qh[4]][uint8 qs[16]] |
| Q5_1 | 7 | 32 | 32 | 8.5 | [fp16 d][fp16 m][uint8 qh[4]][uint8 qs[16]] |
| Q8_0 | 8 | 32 | 34 | 10.5 | [fp16 d][int8 qs[32]] |
| Q8_1 | 9 | 32 | 36 | 11.5 | [float d][float s][int8 qs[32]] |

## K-Quant Block Quantizations (block_size=256, super-blocks)

| Type | ID | BPW | Structure |
|------|----|-----|-----------|
| Q2_K | 10 | ~2.56 | [uint8 scales[16]][uint8 qs[64]][fp16 d][fp16 dmin] |
| Q3_K | 11 | ~3.44 | [uint8 hmask[32]][uint8 qs[64]][uint8 scales[12]][fp16 d] |
| Q4_K | 12 | ~4.5 | [fp16 d][fp16 dmin][uint8 scales[12]][uint8 qs[128]] |
| Q5_K | 13 | ~5.5 | [fp16 d][fp16 dmin][uint8 scales[12]][uint8 qh[32]][uint8 qs[128]] |
| Q6_K | 14 | ~6.56 | [uint8 ql[128]][uint8 qh[64]][int8 scales[16]][fp16 d] |
| Q8_K | 15 | ~10.8 | [float d][int8 qs[256]][int16 bsums[16]] |

## IQ Quantizations (Importance-Matrix)

| Type | ID | BPW | Block Size | Grid | Structure |
|------|----|-----|-----------|------|-----------|
| IQ2_XXS | 16 | ~2.06 | 256 | 256 entries | [fp16 d][uint16 qs[32]] — 8-bit idx + 8-bit signs |
| IQ2_XS | 17 | ~2.31 | 256 | 512 entries | [fp16 d][fp16 scales[4]][uint16 qs[32]] — 9-bit idx + 7-bit signs |
| IQ3_XXS | 18 | ~3.06 | 256 | — | Similar to IQ2_XXS with 3-bit |
| IQ1_S | 19 | ~1.56 | 256 | 2048 entries (16KB) | [fp16 d][uint16 qs[32]][uint16 qh[8]] — 11-bit idx + 1-bit sign + 3-bit delta |
| IQ4_NL | 20 | ~4.5 | 32 | — | Non-linear 4-bit |
| IQ3_S | 21 | ~3.44 | 256 | — | 3-bit with sub-block scales |
| IQ2_S | 22 | ~2.5 | 256 | — | 2-bit with sub-block scales |
| IQ4_XS | 23 | ~4.3 | 256 | — | 4-bit with sub-block scales |
| IQ1_M | 29 | ~1.75 | 256 | — | 1-bit mixed |

## Other Types

| Type | ID | Description |
|------|----|-------------|
| F32 | 0 | 32-bit float, 4 bytes/value |
| F16 | 1 | 16-bit float, 2 bytes/value |
| BF16 | 30 | Brain float 16, 2 bytes/value |
| TQ1_0 | 34 | TurboQuant 1-bit |
| TQ2_0 | 35 | TurboQuant 2-bit |
| MXFP4 | 39 | MX format 4-bit (1 block) |
| I8/I16/I32/I64 | 24-27 | Integer types |
| F64 | 28 | 64-bit double |

## Block Size Reference

| Block Size | Types |
|-----------|-------|
| 1 | F32, F16, BF16, I8, I16, I32, I64, F64 |
| 32 | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, IQ4_NL |
| 256 | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS |

## IQ Dequantization Pattern (from DARS HIP kernels)

All IQ formats share: extract N-bit index → lookup grid table → extract signs → apply scale hierarchy → FMA
- IQ2_XXS: 8-bit index → 256-entry grid (2KB), 8 signs per group
- IQ2_XS: 9-bit index → 512-entry grid (4KB), 7 signs, 4 sub-block fp16 scales
- IQ1_S: 11-bit index → 2048-entry grid (16KB), 1 sign, 3-bit sub-block delta

## IQ Safe Floor (from Ollama/DARS)

For sensitive layers when using IQ quantization:
- IQ1/IQ2 requested → floor is IQ3_S
- IQ3 requested → floor is IQ4_XS
- IQ4 requested → floor is Q4_K
- output.weight, output_norm.weight, token_embd.weight → always minimum Q6_K
- attn_v.weight, attn_qkv.weight → apply safe floor
- FFN layers (ffn_gate, ffn_down, ffn_up) → accept requested type
