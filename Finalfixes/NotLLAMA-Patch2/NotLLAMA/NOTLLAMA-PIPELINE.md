# NotLLAMA Pipeline — Current State (2026-06-29)

## Philosophy

"Many small units doing work together." Weights stay quantized on GPU. Each layer is a self-contained work unit that dequantizes its own weights on-demand. No monolithic pre-dequantized weight buffers.

## Pipeline Flow (single token)

```
Host CPU                   GPU
───────                    ──────
forwardPartial()
  ├─ beginBatch(0)
  │   ├─ embed_q8_0/embed_q6_k/embed  ──► hidden[dim]
  │   └─ barrier()                    
  ├─ endBatch() + syncAllThrottled()  ◄── fence
  │
  └─ for each layer:
      ├─ beginBatch(0)
      │   ├─ rms_norm                  ──► normed[dim]
      │   ├─ barrier()
      │   ├─ fused matvec ×3           ──► Q[headDim*nHeads], K[headDim*nKv], V[headDim*nKv]
      │   │    (matvec_q8_0/q6_k/q4_0   or dequant+gemm fallback)
      │   ├─ barrier()
      │   ├─ rope                      ──► apply RoPE to Q, K
      │   ├─ kv_cache_write            ──► K, V into cache
      │   ├─ attention × nHeads        ──► attnOut[dim]
      │   │    (loop over heads, each head = 1 dispatch)
      │   ├─ barrier()
      │   ├─ fused matvec              ──► attnProjection[dim]
      │   ├─ barrier()
      │   ├─ add (residual)            ──► hidden += attnOut
      │   ├─ barrier()
      │   ├─ rms_norm                  ──► normed[dim]
      │   ├─ barrier()
      │   ├─ fused matvec ×2           ──► gate[hiddenDim], up[hiddenDim]
      │   ├─ barrier()
      │   ├─ silu_mul                  ──► gate*SiLU(gate) * up
      │   ├─ barrier()
      │   ├─ fused matvec              ──► down[dim]
      │   ├─ barrier()
      │   ├─ add (residual)            ──► hidden += ffnOut
      │   └─ endBatch() + syncAllThrottled() ◄── fence
      │
      ├─ final norm + lm_head gemm     ──► logits[vocabSize]
      └─ cpu sampleArgmax              ──► next token
```

## Dispatches per Token

| Stage | Dispatches | Layers | Total |
|-------|-----------|--------|-------|
| Embed | 1 | 1 | 1 |
| Per-layer: attention | 6 (norm, 3×matvec, rope, kv_write) + nHeads attn | 36 | 216 + 36×nHeads |
| Per-layer: ffn | 4 (norm, 2×matvec, silu_mul, matvec, add) + 2 barriers | 36 | ~180 |
| Final | 2 (norm + lm_head gemm) | 1 | 2 |
| **Barriers** | ~7 per layer | 36 | ~252 |
| **Syncs** | 1 per layer + final | 37 | ~37 |

Total dispatches: ~400+ for a 36-layer model (dominated by per-head attention).

## Matvec Shader Structure

### Fused Matvec (quantized formats only)

```
local_size_x = 64, each thread = 1 output column
Dispatch: (outDim + 63) / 64 workgroups

For each tile of 256 input elements:
  1. Cooperative load tile into inputTile[256] shared memory
  2. For each element in tile:
     a. Compute flat index based on transB
     b. Find quant block (divide by block size)
     c. Read F16 scale from block header
     d. Read quantized value + sign extend
     e. acc += inputTile[k] * d * q
  3. barrier()
Write: output[col] = acc (no reduction needed)
```

Three separate shaders (no unified `DType` dispatch):
- `matvec_q8_0.comp` — Q8_0 (34B/32)
- `matvec_q6_k.comp` — Q6_K (210B/256, d-last) 
- `matvec_q4_0.comp` — Q4_0 (18B/32)

### Fallback Path (F32/F16 weights)

```
Dequant → F32 staging buffer → gemm.comp (32-thread F32 GEMM)
```

## Shader Details

### RmsNorm
```glsl
local_size_x = 256, workgroups = 1
// All threads reduce sumSq across dim elements (strided)
// Single thread computes invRms, broadcasts
// All threads apply: out[d] = in[d] * invRms * weight[d]
```

### Attention (naive O(N²), per-head dispatch)
```glsl
local_size_x = 32 (one warp)
// seqLen=1 so attention is trivial: Q*K^T = dot(Q_head, K_head)
// softmax over 1 element = 1.0
// V weighted sum = V_head * 1.0 = V_head
```

### Rope
```glsl
local_size_x = 32
// Applies rotary position embedding to Q and K
```

### SiluMul
```glsl
local_size_x = 256
// out[i] = gate[i] * (gate[i] / (1 + exp(-gate[i]))) * up[i]
```

### Add (residual)
```glsl
local_size_x = 256
// out[i] = a[i] + b[i]
```

## Embed Shaders

Three embed shaders selected by quantization format:
- `embed_q8_0` — Q8_0 embedding table, `int8_t[]` buffer, 256 threads
- `embed_q6_k` — Q6_K embedding table (d-last), `uint8_t[]` buffer, 256 threads
- `embed` — F32 embedding table, `float[]` buffer, 32 threads

## Known Issues

### Current Diagnosis (2026-06-29)

**Symptom**: GPU/CPU logit mismatch on ALL quantized models. MaxAE=10-24 even with the fallback dequant+gemm path.

**Embed verified correct**: GPU embed[0..7] = CPU embed[0..7] (per diagnostic readback):
```
GPU embed[0..7]: -0.031412 -0.000365 -0.027394 -0.004748 -0.002192 0.011323 0.046388 -0.009497
CPU embed[0..3]: -0.031412 -0.000365 -0.027394 -0.004748
```

**Conclusion**: Embed is correct (bit-exact with CPU). The bug is in the attention, FFN, or downstream pipeline stages.

### Suspected Areas
1. **RMS norm** — incorrect `sumSq` reduction or broadcast
2. **Attention** — off-by-one in K/V cache indexing or head offset
3. **Pipeline barriers** — missing or incorrect staging between dispatches
4. **Residual add** — accumulating at wrong offset
5. **LM head** — weight tying transpose indexing

### Dispatch Count Bug (now fixed)
Original matvec shaders used `local_size_x=256` with shared memory reduction, but dispatch was `(outDim+63)/64` (64-thread pattern). Fixed to use 64 threads per workgroup, each handling one output column independently (no reduction needed).

## Context.cpp Device Setup

```
Feature chain: query ALL features → ZERO graphics features → enable only compute-relevant
```

| Feature | Status |
|---------|--------|
| `shaderInt64` (BDA) | VK_TRUE |
| `shaderInt16` | VK_TRUE |
| `bufferDeviceAddress` | VK_TRUE |
| `shaderFloat16` | VK_TRUE |
| `shaderInt8` | VK_TRUE |
| `storageBuffer8BitAccess` | VK_TRUE |
| `scalarBlockLayout` | VK_TRUE |
| `timelineSemaphore` | VK_TRUE |
| `synchronization2` | VK_TRUE |
| `maintenance4` | VK_TRUE |
| `dynamicRendering` | VK_FALSE (compute-only) |

Queue family: prefers compute-only (no graphics bit), falls back to graphics+compute.
