# LOGIC TRACE — Before Build Validation
# Walk through every condition for a Q4_0 model

## SCENARIO: Qwen2.5-3B Q4_0 model loaded

### Step 1: Weight Loading (weight_uploader.cpp)
```
Tensor "blk.0.attn_q.weight":
  format = Q4_0 (enum value 2)          → TRUE
  sizeBytes = 524288 (example)          → TRUE
  binSize = 278528 (compressed)         → TRUE
  gpuAddress = 0x123400000000 (BDA)     → TRUE
  Uploaded as raw Q4_0 bytes            → TRUE
```

### Step 2: forwardPartial() Entry
```
allocator->reset()                       → TRUE (resets ring buffer)
dim = 2048                               → TRUE (from model desc)
headDim = 128                            → TRUE (2048/16)
hiddenDim = 5504                         → TRUE (from model desc)
seqLen = 1                               → TRUE (seqPos=0 + 1)

Ring buffer allocations:
  hiddenAddr = allocator->alloc(8192)    → TRUE (2048 * 4 bytes)
  qAddr = allocator->alloc(32768)        → TRUE (1 * 128 * 16 * 4)
  kAddr = allocator->alloc(4096)         → TRUE (1 * 128 * 2 * 4)
  vAddr = allocator->alloc(4096)         → TRUE
  attnOutAddr = allocator->alloc(8192)   → TRUE
  mlpOutAddr = allocator->alloc(8192)    → TRUE
  logitsAddr = allocator->alloc(512000)  → TRUE (128000 * 4)
  sampleOutAddr = allocator->alloc(16)   → TRUE
```

### Step 3: Embedding
```
findTensor("token_embd.weight")          → TRUE (tensor exists)
tensor->format = F32 (for embedding)     → TRUE (embeddings are F32)
addrW = tensor->gpuAddress               → TRUE
isF32(F32) = TRUE                        → TRUE
dequantIfNeeded returns addrW directly   → TRUE (no dequant for F32)
dispatchInBatch(embed, ...)              → TRUE
```

### Step 4: Layer 0 — Attention RMS Norm
```
findTensor("blk.0.attn_norm.weight")     → TRUE
addrAttnNorm = tensor->gpuAddress        → TRUE
RmsNormPushConstants filled correctly    → TRUE
dispatchInBatch(rms_norm, ...)           → TRUE
barrierBetweenGroups()                   → TRUE
```

### Step 5: Layer 0 — QKV GEMM (THE CRITICAL PATH)
```
findTensor("blk.0.attn_q.weight")        → TRUE (tensor exists)
tensor->format = Q4_0                    → TRUE (enum value 2)
tensor->gpuAddress = 0x123400000000      → TRUE (BDA address)
tensor->sizeBytes = 524288               → TRUE

getBytesPerElement(Q4_0)                 → returns 1
nElem = 524288 / 1 = 524288             → TRUE (number of elements)

dequantIfNeeded called:
  isF32(Q4_0) = FALSE                   → TRUE (Q4_0 is not F32)
  outSize = 524288 * 4 = 2097152        → TRUE (2MB for F32 output)
  stagingAddr = allocator->alloc(2097152) → TRUE (if ring buffer has space)

  DequantizePushConstants:
    addrQuant = 0x123400000000           → TRUE (Q4_0 data on GPU)
    addrOut = stagingAddr                → TRUE (F32 staging buffer)
    nElements = 524288                   → TRUE
    quantFormat = 2                      → TRUE (Q4_0 = 2)
    totalThreads = 524288                → TRUE
    elementOffset = 0                    → TRUE

  dispatchInBatch(dequantize, ...):
    gx = (524288 + 63) / 64 = 8192      → TRUE (workgroup count)
    gy = 1, gz = 1                       → TRUE

  barrierBetweenGroups()                 → TRUE

  Returns stagingAddr                    → TRUE

GEMM uses stagingAddr as addrB:
  GemmPushConstants:
    addrA = normHidden                   → TRUE (F32 activations)
    addrB = stagingAddr                  → TRUE (F32 dequantized weights)
    addrC = qRowAddr                     → TRUE (output)
    M = 1, N = dim, K = dim             → TRUE
    alpha = 1.0, transB = 0             → TRUE

  dispatchInBatch(gemm, ...)             → TRUE
```

### Step 6: Layer 0 — Q4_0 Dequant Shader Execution
```
gid = gl_GlobalInvocationID.x            → TRUE (0..524287)
globalIdx = gid + 0                      → TRUE (elementOffset=0)
globalIdx < nElements                    → TRUE (524288 < 524288? NO, 524287 < 524288)

src = QuantRef(addrQuant)                → TRUE (points to Q4_0 data)
dst = FloatOutRef(addrOut)               → TRUE (points to F32 staging)

quantFormat == 2u (Q4_0):
  blockIdx = globalIdx / 32              → TRUE (which 32-element block)
  elemInBlock = globalIdx % 32           → TRUE (position in block)
  blockStart = blockIdx * 18             → TRUE (18 bytes per Q4_0 block)

  scale = readF16(src, blockStart)       → TRUE (2-byte fp16 scale)
    byteOffset = blockStart              → TRUE
    lo = uint(src.data[blockStart])      → TRUE
    hi = uint(src.data[blockStart + 1])  → TRUE
    h = uint16_t(lo | (hi << 8))        → TRUE
    fp16ToFloat(h)                       → TRUE (converts to float)

  byteIdx = blockStart + 2 + (elemInBlock / 2) → TRUE
  rawByte = uint(src.data[byteIdx])      → TRUE (read packed nibbles)

  elemInBlock is even:
    q = int(rawByte & 0x0F) - 8         → TRUE (low nibble, offset by -8)
  elemInBlock is odd:
    q = int((rawByte >> 4) & 0x0F) - 8  → TRUE (high nibble, offset by -8)

  dst.data[globalIdx] = scale * float(q) → TRUE (F32 output)
```

### Step 7: GEMM Reads F32 Weights
```
gemm.comp:
  col = gl_GlobalInvocationID.x          → TRUE (0..dim-1)
  if (col >= N) return                   → TRUE (bounds check)

  A = ARef(addrA)                        → TRUE (F32 activations)
  B = BRef(addrB)                        → TRUE (F32 dequantized weights)
  C = CRef(addrC)                        → TRUE (output buffer)

  for k = 0 to K-1:
    a = A.data[k]                        → TRUE (F32 activation)
    b = B.data[col * K + k]             → TRUE (F32 weight, NOT Q4_0 bytes!)
    acc += a * b                         → TRUE (correct F32 multiply)

  C.data[col] = alpha * acc             → TRUE (F32 output)
```

## TRUTH TABLE — Every Condition

| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| Tensor exists in model | TRUE | TRUE | ✅ |
| Tensor format is Q4_0 | TRUE | TRUE | ✅ |
| isF32(Q4_0) returns FALSE | TRUE | TRUE | ✅ |
| stagingAddr allocation succeeds | TRUE | TRUE* | ✅* |
| DequantizePushConstants filled | TRUE | TRUE | ✅ |
| dequantize pipeline loaded | TRUE | TRUE** | ✅** |
| dequantize shader compiles | TRUE | TRUE*** | ✅*** |
| Q4_0 block layout matches shader | TRUE | TRUE**** | ✅**** |
| GEMM receives F32 stagingAddr | TRUE | TRUE | ✅ |
| GEMM reads F32 data | TRUE | TRUE | ✅ |
| barrierAfter dequant | TRUE | TRUE | ✅ |
| barrierAfter GEMM | TRUE | TRUE | ✅ |

* = if ring buffer has space (64MB default, 2MB needed)
** = if loadPipe succeeds (we added it)
*** = if glslc compiles without errors
**** = if ggml Q4_0 layout matches our shader

## POTENTIAL FAILURE POINTS

### F1: Ring Buffer Overflow
```
Ring buffer size: 64MB (default)
Per-layer dequant staging: ~2MB * 7 weights = ~14MB
Per-layer activations: ~200KB
Total per layer: ~14.2MB
For 36 layers: ~511MB needed
64MB ring buffer: NOT ENOUGH
```
**VERDICT: WILL FAIL on models with many layers**

### F2: Q4_0 Block Layout Mismatch
```
ggml Q4_0 block: [fp16_scale (2B)] [32 nibbles (16B)] = 18 bytes/32 elements
Our shader: blockStart = blockIdx * 18 → TRUE
Scale at blockStart → TRUE
Nibbles at blockStart + 2 → TRUE
Low nibble: rawByte & 0x0F - 8 → TRUE
High nibble: (rawByte >> 4) & 0x0F - 8 → TRUE
```
**VERDICT: CORRECT (matches ggml spec)**

### F3: Shader Compilation Error
```
GLSL types: uint data[] for QuantRef → needs byte-level access
F32 case uses: src.data[byteOff] where byteOff = globalIdx * 4
But src.data[] is uint[], so src.data[byteOff] reads uint at index byteOff
This reads 4 bytes at index byteOff, not at byte offset byteOff
```
**VERDICT: BUG — F32 case reads wrong bytes**

### F4: nElements Calculation
```
tensor->sizeBytes = total compressed bytes
getBytesPerElement(Q4_0) = 1
nElem = sizeBytes / 1 = sizeBytes

But Q4_0 has 2 elements per byte!
nElem should be sizeBytes * 2 for Q4_0
```
**VERDICT: BUG — nElements is wrong for Q4_0**

## CRITICAL BUGS FOUND

### BUG 1: F32 Case in Shader
```glsl
// WRONG: src.data is uint[], byteOff is byte offset
uint byteOff = globalIdx * 4u;
uint raw = uint(src.data[byteOff]) | ...;

// RIGHT: need to read 4 bytes individually
uint byteOff = globalIdx * 4u;
uint raw = uint(src.data[byteOff / 4u]) & 0xFFFFFFFF; // reads full uint
```

### BUG 2: nElements Calculation
```cpp
// WRONG: sizeBytes / 1 = sizeBytes (too many elements for Q4_0)
uint32_t nElem = tensor->sizeBytes / getBytesPerElement(tensor->format);

// RIGHT: need actual element count, not compressed byte count
// For Q4_0: sizeBytes * 2 (2 elements per byte)
// For Q8_0: sizeBytes * 1 (1 element per byte)
// For F32: sizeBytes / 4 (4 bytes per element)
```

### BUG 3: readF16 Byte Indexing
```glsl
// WRONG: src.data is uint[], so indexing is by uint, not bytes
uint lo = uint(src.data[byteOffset]);

// RIGHT: need to convert byte offset to uint index
uint lo = uint(src.data[byteOffset / 4u]);
```

## VERDICT: WILL NOT WORK AS-IS

The shader has 3 critical bugs:
1. F32 case reads wrong bytes (uint[] vs byte[] confusion)
2. nElements calculation wrong for quantized formats
3. readF16 byte indexing wrong

**DO NOT BUILD YET**
