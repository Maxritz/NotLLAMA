# RoPE Data Flow Map

## Overview
Rotary Position Embedding (RoPE) applies position-dependent rotation to Q and K vectors
in multi-head attention. This map traces the complete data path from allocation to use.

## Flow: Token Generation (seqPos = N)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. ALLOCATION (Ring Buffer)                                        │
│                                                                     │
│   qAddr = allocator->alloc(qSize)        ← ring base for Q         │
│   kAddr = allocator->alloc(kvSize)       ← ring base for K         │
│   vAddr = allocator->alloc(kvSize)       ← ring base for V         │
│                                                                     │
│   qRowAddr = qAddr + N * dim * 4         ← row N in Q buffer       │
│   kRowAddr = kAddr + N * kvDim * 4       ← row N in K buffer       │
│   vRowAddr = vAddr + N * kvDim * 4       ← row N in V buffer       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. GEMM PROJECTIONS (writes to qRowAddr, kRowAddr, vRowAddr)      │
│                                                                     │
│   GEMM(attn_q.weight, normHidden) → writes to qRowAddr             │
│   GEMM(attn_k.weight, normHidden) → writes to kRowAddr             │
│   GEMM(attn_v.weight, normHidden) → writes to vRowAddr             │
│                                                                     │
│   After: Q[N,:], K[N,:], V[N,:] are in the ring buffer             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. RoPE (FIXED: uses qAddr/kAddr as base)                         │
│                                                                     │
│   Push constants:                                                   │
│     addrQ = qAddr          ← BASE address (NOT qRowAddr)           │
│     addrK = kAddr          ← BASE address (NOT kRowAddr)           │
│     seqLen = N+1           ← 1-based sequence length               │
│                                                                     │
│   Shader computes:                                                  │
│     qIdx = (seqLen-1) * nHeads * headDim + head * headDim + d      │
│          = N * dim + head * headDim + d                            │
│          = offset into row N                                        │
│                                                                     │
│   Actual GPU address:                                               │
│     qAddr + qIdx * sizeof(float)                                   │
│     = qAddr + (N*dim + head*headDim + d) * 4                       │
│     = qRowAddr + (head*headDim + d) * 4                            │
│     = correct location of Q[N, head, d]                            │
│                                                                     │
│   RoPE rotation applied IN-PLACE on Q[N,:] and K[N,:]              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. KV CACHE WRITE (uses kRowAddr/vRowAddr as source)              │
│                                                                     │
│   Push constants:                                                   │
│     addrKIn = kRowAddr     ← row N of K (source for this token)    │
│     addrVIn = vRowAddr     ← row N of V (source for this token)    │
│     seqPos = N              ← position in cache to write            │
│                                                                     │
│   Shader computes:                                                  │
│     cacheOffset = seqPos * nKvHeads * headDim                      │
│     KCache[cacheOffset + i] = KIn[i]  for i in 0..kvDim-1         │
│                                                                     │
│   After: K[N,:] and V[N,:] are in the KV cache                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. ATTENTION (reads from KV cache, Q from ring buffer)            │
│                                                                     │
│   Q matrix: qAddr (base), reads Q[0..N,:] rows                    │
│   K matrix: KV cache K[0..N,:]                                     │
│   V matrix: KV cache V[0..N,:]                                     │
│                                                                     │
│   Output: attention result → attnOutAddr                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Bug That Was Fixed

### Before (BROKEN)
```
RoPE push constants:
  addrQ = qRowAddr = qAddr + N * dim * 4
  addrK = kRowAddr = kAddr + N * kvDim * 4

Shader address calculation:
  qIdx = (N) * dim + head * headDim + d
  actual_addr = qRowAddr + qIdx * 4
              = (qAddr + N*dim*4) + (N*dim + head*headDim + d) * 4
              = qAddr + N*dim*4 + N*dim*4 + (head*headDim + d) * 4
              = qAddr + 2*N*dim*4 + (head*headDim + d) * 4
              ↑ DOUBLE OFFSET: reads row 2N instead of row N
```

### After (FIXED)
```
RoPE push constants:
  addrQ = qAddr              ← base address
  addrK = kAddr              ← base address

Shader address calculation:
  qIdx = N * dim + head * headDim + d
  actual_addr = qAddr + qIdx * 4
              = qAddr + (N*dim + head*headDim + d) * 4
              = qRowAddr + (head*headDim + d) * 4
              ↑ CORRECT: reads row N, element (head, d)
```

## Edge Cases

| seqPos | Before Fix | After Fix | Status |
|--------|-----------|-----------|--------|
| 0 | addrQ = qAddr, offset = -1*dim → UNDERFLOW | addrQ = qAddr, offset = 0 → reads qAddr[0] | FIXED |
| 1 | addrQ = qAddr+dim, offset = 0 → correct by accident | addrQ = qAddr, offset = dim → reads qAddr[dim] | FIXED |
| 2 | addrQ = qAddr+2dim, offset = dim → reads row 3 | addrQ = qAddr, offset = 2dim → reads row 2 | FIXED |
| N | addrQ = qAddr+N*dim, offset = (N-1)dim → reads row 2N-1 | addrQ = qAddr, offset = N*dim → reads row N | FIXED |

## Key Insight

The shader was designed to take the **base address** and compute its own offset.
The host was passing the **row address** (base + seqPos offset), causing a double-offset.
The fix is to match the shader's contract: pass base addresses only.
