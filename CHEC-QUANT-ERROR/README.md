# NotLLAMA Quant Fix — All Formats

## Problem
All quantized formats (Q8_0, Q6_K, Q4_0, etc.) produce wrong results. The root cause is
inconsistent and buggy byte-level access across shaders.

## Root Cause
The existing shaders use `uint data[]` with manual byte extraction (`readByte()` functions).
While this CAN work, it is error-prone and several shaders had subtle bugs:
- Missing sign-extension on int8 values (negative scales read as 251 instead of -5)
- Wrong FP16 unpacking (`unpackFloat2x16` vs `unpackHalf2x16` — different functions!)
- `uintBitsToFloat` used on 16-bit values (reads garbage)
- Manual byte extraction from `uint[]` can have alignment issues if the buffer address
  is not a multiple of 4

## Fix
ALL shaders now use `uint8_t data[]` with `buffer_reference`:
```glsl
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer ByteRef {
    uint8_t data[];
};
```

This requires:
- `#extension GL_EXT_shader_8bit_storage : require`
- `#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require`

These are supported on AMD RDNA4 with Vulkan 1.2+.

## Files Changed

| File | Description |
|------|-------------|
| `matvec_all.comp` | Unified matvec for ALL quant formats. Replaces matvec_q8_0, matvec_q6_k, etc. |
| `embed_q8_0.comp` | Fixed embedding lookup. Uses `uint8_t data[]` instead of `uint data[]` + manual extraction. |
| `dequantize.comp` | Fixed dequantize for ALL formats. Same `uint8_t` approach. |

## Build

```powershell
glslangValidator -V matvec_all.comp -o matvec_all.spv --target-env vulkan1.2
glslangValidator -V embed_q8_0.comp -o embed_q8_0.spv --target-env vulkan1.2
glslangValidator -V dequantize.comp -o dequantize.spv --target-env vulkan1.2
```

## Host Code Changes Needed

### 1. Register the new pipeline
In your PipelineBuilder init, add:
```cpp
// matvec_all.comp handles all quant formats via DType push constant
pipes->registerPipeline("matvec_all", "shaders/matvec_all.spv", sizeof(MatvecPushConstants), 0, nullptr);
```

### 2. Update the MatvecPushConstants struct
```cpp
struct MatvecPushConstants {
    uint64_t addrWeight;
    uint64_t addrInput;
    uint64_t addrOutput;
    uint32_t inDim;
    uint32_t outDim;
    uint32_t transB;
    uint32_t DType;  // quant format id
};
```

### 3. Replace all matvec dispatch calls
Wherever you dispatch `matvec_q8_0`, `matvec_q6_k`, etc., use `matvec_all` with the
appropriate `DType`:
```cpp
MatvecPushConstants pc = {};
pc.addrWeight = weightAddr;
pc.addrInput = inputAddr;
pc.addrOutput = outputAddr;
pc.inDim = K;
pc.outDim = N;
pc.transB = 0;
pc.DType = 2;  // Q8_0 = 2, Q6_K = 15, Q4_0 = 3, etc.
scheduler->dispatchInBatch(pipes->getPipeline("matvec_all"), pipes->getLayout("matvec_all"),
    &pc, sizeof(pc), (N + 63) / 64, 1, 1);
```

### 4. DType mapping (must match your CPU reference)
| Format | DType Value |
|--------|-------------|
| F32 | 0 |
| F16 | 1 |
| Q8_0 | 2 |
| Q4_0 | 3 |
| Q4_1 | 4 |
| Q5_0 | 5 |
| Q5_1 | 6 |
| Q8_1 | 7 |
| Q2_K | 8 |
| Q3_K | 9 |
| Q4_K | 10 |
| Q5_K | 11 |
| Q8_K | 14 |
| Q6_K | 15 |

## Verification
After applying, run:
```powershell
.\test_inference.exe ..\..\model\acrux-500m-q6k.weights.json ..\..\model\acrux-500m-q6k.weights.bin
```
Expected: `nan=0`, `Max absolute error < 0.01`, `PASS`.
