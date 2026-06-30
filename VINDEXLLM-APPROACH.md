# VindexLLM Approach — Fused Quantized Matvec (from CHEC-QUANT-ERROR)

## Core Design

A single unified `matvec_all.comp` shader handling ALL quant formats via a `DType` push constant. One GLSL function `dequant_weight(addr, idx, DType)` selects the format at runtime.

## Shader Structure

| Aspect | Detail |
|--------|--------|
| **local_size_x** | 64 threads per workgroup |
| **Each thread** | One output column (independent, no reduction) |
| **Dispatch** | `(outDim + 63) / 64` workgroups |
| **Input tile** | 256 F32 elements loaded cooperatively into shared memory |
| **Dot product** | Sequential loop over tile, calling `dequant_weight()` per element |
| **Write** | Direct: `output.data[n] = acc` (no reduction barrier) |

## Key Implementation Details

### Buffer Types
```glsl
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer ByteRef {
    uint8_t data[];    // quantized weights — unsigned byte for universal reads
};
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer FloatInRef {
    float data[];      // input vector
};
layout(buffer_reference, scalar, buffer_reference_align = 8) writeonly buffer FloatOutRef {
    float data[];      // output vector
};
```

### Weight Indexing (transB-aware)
```glsl
uint weightIdx;
if (pc.transB == 0) {
    weightIdx = globalK * pc.outDim + n;  // row-major: [k][col]
} else {
    weightIdx = n * pc.inDim + globalK;   // col-major: [col][k]
}
float w = dequant_weight(pc.addrWeight, weightIdx, pc.DType);
```

### Sign Extension
```glsl
int sign_extend_u8(uint8_t v) {
    return int(int8_t(v));  // explicit int8_t cast = sign extension
}
```

### F16 Read
```glsl
float fp16_to_fp32(uint8_t low, uint8_t high) {
    uint raw = uint(low) | (uint(high) << 8);
    return unpackHalf2x16(raw).x;
}
```

### Shared Memory Input Tile
```glsl
#define TILE_K 256
shared float vecShared[TILE_K];

// Cooperative load
for (uint i = gl_LocalInvocationID.x; i < tileSize; i += 64) {
    vecShared[i] = input.data[tileStart + i];
}
barrier();

// Sequential dot product over tile
for (uint k = 0; k < tileSize; ++k) {
    acc += vecShared[k] * dequant_weight(addr, weightIdx, DType);
}
barrier();
```

## Per-Format Dequant (from `dequant_weight()`)

| DType | Format | Block | Formula |
|-------|--------|-------|---------|
| 2 | Q8_0 | 32 elems, 34B | `d * q` (q=int8_t) |
| 3 | Q4_0 | 32 elems, 18B | `d * (nibble - 8)` |
| 15 | Q6_K | 256 elems, 210B | `d * sc * (val - 32)` (d-last) |

## Dispatch Pattern

```glsl
layout(push_constant, scalar) uniform PushConstants {
    uint64_t addrWeight;
    uint64_t addrInput;
    uint64_t addrOutput;
    uint inDim;
    uint outDim;
    uint transB;
    uint DType;           // NOT in our MatVecPushConstants
} pc;
```

## Verification Status
- All formats tested in isolation (CHEC-QUANT-ERROR directory)
- Q8_0 verification: embed[0..3] matches CPU reference
- Q6_K verification: produces finite (non-NaN) output
