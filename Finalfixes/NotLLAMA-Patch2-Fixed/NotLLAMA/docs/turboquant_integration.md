# TurboQuant Integration Guide

## Overview

TurboQuant (TQ) is a family of GPU-native block quantization formats designed for the RDNA4 Vulkan compute engine. Unlike GGUF K-quants, TQ blocks use a simple `[scale][packed_weights]` layout and are small enough to be dequantized on-demand inside shaders, keeping weights quantized in GPU memory until they are consumed. This reduces memory traffic and allows larger models/contexts without pre-dequantizing entire tensors.

Three variants are implemented:

| Format  | Bits | Block | Bytes/Block | Compression | Quality target |
|---------|------|-------|-------------|-------------|----------------|
| TQ4_128 | 4    | 128   | 66          | 4x vs F32   | ~0.5% ppl degradation |
| TQ3_128 | 3    | 128   | 50          | ~5.3x vs F32 | ~2% ppl degradation |
| TQ6_64  | 6    | 64    | 50          | ~5.3x vs F32 | near-lossless |

## Block Layouts (for weight_converter.py)

All TQ blocks store a 2-byte little-endian `float16` scale first, followed by tightly packed weights. Weights are quantized to unsigned integers and re-centered in the shader.

### TQ4_128 — 66 bytes, 128 elements

```
Offset  Size   Description
0       2      scale (fp16)
2       64     packed 4-bit weights
66      end of block
```

**Nibble layout (low nibble first):**

```
byte[0] = [elem0:4bits][elem1:4bits]
byte[1] = [elem2:4bits][elem3:4bits]
...
byte[63] = [elem126:4bits][elem127:4bits]
```

**Dequantization:** `float w = (float(nibble) - 8.0) * scale`

### TQ3_128 — 50 bytes, 128 elements

```
Offset  Size   Description
0       2      scale (fp16)
2       48     packed 3-bit weights
50      end of block
```

**Sequential bit packing:** weights are stored as a little-endian bit stream. 8 values occupy exactly 24 bits = 3 bytes.

```
bits[0:2]   = elem0
bits[3:5]   = elem1
bits[6:8]   = elem2
...
bits[381:383] = elem127
```

**Dequantization:** `float w = (float(q3) - 4.0) * scale`

### TQ6_64 — 50 bytes, 64 elements

```
Offset  Size   Description
0       2      scale (fp16)
2       48     packed 6-bit weights
50      end of block
```

**Sequential bit packing:** weights are stored as a little-endian bit stream. 64 values occupy exactly 384 bits = 48 bytes.

```
bits[0:5]   = elem0
bits[6:11]  = elem1
bits[12:17] = elem2
...
bits[378:383] = elem63
```

**Dequantization:** `float w = (float(q6) - 32.0) * scale`

## Engine Wiring Checklist

### Step 1: Weight Loading (inference_engine.cpp)

- [ ] Add case for `QuantFormat::TQ4_128`, `TQ3_128`, `TQ6_64` in weight loading.
- [ ] Use `tq4_packed_size` / `tq3_packed_size` / `tq6_packed_size` to size GPU buffers.
- [ ] Store `block_size` and `format` in weight metadata for dispatch.
- [ ] Do NOT pre-dequantize entire tensors; upload packed TQ bytes directly.

### Step 2: Dequant Shader (dequant_turbo.comp)

- [ ] Already implemented by MiMo (supports TQ4 and TQ6; TQ3 is stubbed).
- [ ] Wire `DequantTurboPushConstants` in the scheduler.
- [ ] Test with embed weights first.

### Step 3: Fused GEMM (gemm_turbo.comp)

- [ ] Already implemented by MiMo.
- [ ] Wire `GemmTurboPushConstants` in the scheduler.
- [ ] Test MLP `gate_up` first.

### Step 4: Scheduler Integration

- [ ] Add `dequant_turbo` and `gemm_turbo` pipelines to `PipelineBuilder`.
- [ ] Push constant sizes are guarded by `static_assert(sizeof(...) <= 128)`.
- [ ] Match GLSL `layout(push_constant, scalar)` exactly.

### Step 5: Fallback

- [ ] If tensor format is not TQ4/TQ3/TQ6, use the existing `dequantize.comp` path.
- [ ] If tensor format is TQ, use `dequant_turbo.comp` or `gemm_turbo.comp`.

## Performance Targets

| Format  | MSE vs Q8_0 | Compression vs F32 | Notes |
|---------|-------------|--------------------|-------|
| TQ4_128 | < 0.1%      | 4x                 | balanced quality/size |
| TQ3_128 | < 0.5%      | ~5.3x              | extreme compression |
| TQ6_64  | < 0.01%     | ~5.3x              | near-lossless |

## Files Added / Modified by MiMo

- `include/rdna4_turboquant.hpp` — host block layouts + size helpers (new).
- `include/rdna4_types.hpp` — `DequantTurboPushConstants` + `GemmTurboPushConstants` (append).
- `src/kernels/dequant_turbo.comp` — standalone dequant shader (existing).
- `src/kernels/gemm_turbo.comp` — fused GEMM shader (existing).
- `tools/weight_converter.py` — `convert_to_tq4/tq3/tq6` helpers (append).
- `tools/validate_turboquant.py` — accuracy validation (new).
- `tools/benchmark_turboquant.py` — model benchmarking (new).
- `tools/convert_gguf_to_turboquant.py` — standalone GGUF→TQ converter (new).
- `test/test_turboquant.cpp` — compile-time size checks (new).
- `docs/turboquant_integration.md` — this guide (new).
