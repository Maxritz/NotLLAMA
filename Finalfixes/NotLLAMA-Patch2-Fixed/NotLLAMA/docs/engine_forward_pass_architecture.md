# Engine Forward Pass Architecture — Complete Analysis

Source: `src/host/inference_engine.cpp`, `include/rdna4_engine.hpp`, `include/rdna4_model_arch.hpp`

---

## Forward Pass Dispatch Sequence

The engine's `forwardPartial()` function executes a **single batched command buffer** per token forward pass. All dispatches are recorded via `scheduler->beginBatch()` / `endBatch()` and submitted in one `vkQueueSubmit`.

### Complete Dispatch Sequence Per Layer

For each transformer layer N (0 to `blockCount-1`):

```
1. EMBEDDING (one-time, before layers)
   └─ embed shader: token_embd.weight[tokenId] → hiddenAddr

2. ATTENTION BLOCK (per layer N)
   ├─ 2a. RMSNorm (attention pre-norm)
   │       hiddenAddr → attnOutAddr
   │       weight: blk.N.attn_norm.weight
   │       epsilon: 1e-6
   │
   ├─ 2b. [BARRIER]
   │
   ├─ 2c. Q Projection GEMM
   │       normHidden (attnOutAddr) × blk.N.attn_q.weight → qRowAddr
   │       dims: [1, dim, dim] (row vector × matrix → row vector)
   │
   ├─ 2d. K Projection GEMM
   │       normHidden × blk.N.attn_k.weight → kRowAddr
   │       dims: [1, kvDim, dim] where kvDim = headDim × headCountKv
   │
   ├─ 2e. V Projection GEMM
   │       normHidden × blk.N.attn_v.weight → vRowAddr
   │       dims: [1, kvDim, dim]
   │
   ├─ 2f. [BARRIER]
   │
   ├─ 2g. RoPE
   │       qRowAddr, kRowAddr in-place rotation
   │       params: seqLen, headDim, headCount, headCountKv
   │       base frequency: 10000.0f (hardcoded)
   │
   ├─ 2h. KV Cache Write
   │       kRowAddr → kvCacheK[layer][seqPos]
   │       vRowAddr → kvCacheV[layer][seqPos]
   │
   ├─ 2i. [BARRIER]
   │
   ├─ 2j. Attention (per head, headCount dispatches)
   │       For each head h in 0..headCount-1:
   │         qAddr[h], kvCacheK[layer], kvCacheV[layer] → attnOutAddr
   │         softmax scale: 1.0 / sqrt(headDim)
   │
   ├─ 2k. [BARRIER]
   │
   ├─ 2l. Output Projection GEMM
   │       attnOutAddr × blk.N.attn_output.weight → mlpOutAddr
   │       dims: [1, dim, dim]
   │
   ├─ 2m. [BARRIER]
   │
   ├─ 2n. Residual Add (attention)
   │       hiddenAddr + mlpOutAddr → hiddenAddr
   │
   └─ 2o. [BARRIER]

3. FFN BLOCK (per layer N)
   ├─ 3a. RMSNorm (FFN pre-norm)
   │       hiddenAddr → attnOutAddr
   │       weight: blk.N.ffn_norm.weight
   │       epsilon: 1e-6
   │
   ├─ 3b. [BARRIER]
   │
   ├─ 3c. Gate GEMM
   │       attnOutAddr × blk.N.ffn_gate.weight → gateScratchAddr
   │       dims: [1, hiddenDim, dim]
   │
   ├─ 3d. Up GEMM
   │       attnOutAddr × blk.N.ffn_up.weight → upScratchAddr
   │       dims: [1, hiddenDim, dim]
   │
   ├─ 3e. [BARRIER]
   │
   ├─ 3f. SiLU + Mul
   │       SiLU(gateScratchAddr) × upScratchAddr → interScratchAddr
   │
   ├─ 3g. [BARRIER]
   │
   ├─ 3h. Down GEMM
   │       interScratchAddr × blk.N.ffn_down.weight → mlpOutAddr
   │       dims: [1, dim, hiddenDim]
   │
   ├─ 3i. [BARRIER]
   │
   ├─ 3j. Residual Add (FFN)
   │       hiddenAddr + mlpOutAddr → hiddenAddr
   │
   └─ 3k. [BARRIER]

4. FINAL (after all layers)
   ├─ 4a. Output Norm
   │       hiddenAddr → attnOutAddr
   │       weight: output_norm.weight (fallback: norm.weight)
   │       epsilon: 1e-6
   │
   ├─ 4b. [BARRIER]
   │
   └─ 4c. LM Head GEMM
           attnOutAddr × output.weight → logitsAddr
           dims: [1, vocabSize, dim]
           (fallback: token_embd.weight if output.weight missing)
```

---

## Hardcoded Tensor Name Strings

All tensor lookups are string-based via `findTensorAddr()`:

| Line | Tensor Name | Purpose |
|------|-------------|---------|
| 83 | `"token_embd.weight"` | Token embedding |
| 100 | `"blk.{i}.attn_norm.weight"` | Attention pre-norm |
| 112 | `"blk.{i}.attn_q.weight"` | Q projection |
| 118 | `"blk.{i}.attn_k.weight"` | K projection |
| 125 | `"blk.{i}.attn_v.weight"` | V projection |
| 169 | `"blk.{i}.attn_output.weight"` | Attention output projection |
| 190 | `"blk.{i}.ffn_norm.weight"` | FFN pre-norm |
| 204 | `"blk.{i}.ffn_gate.weight"` | FFN gate (SwiGLU) |
| 210 | `"blk.{i}.ffn_up.weight"` | FFN up projection |
| 231 | `"blk.{i}.ffn_down.weight"` | FFN down projection |
| 249 | `"output_norm.weight"` | Final norm |
| 250 | `"norm.weight"` | Fallback final norm |
| 261 | `"output.weight"` | LM head |
| 263 | `"token_embd.weight"` | Fallback LM head (weight-tied) |

---

## Hardcoded Assumptions

### Normalization

| Assumption | Value | Hardcoded at |
|------------|-------|--------------|
| Normalization type | RMSNorm only | `rms_norm` pipeline selection |
| Epsilon | 1e-6 | Lines 101, 191, 251 |
| Pre-norm only | Always pre-norm (not post-norm) | Norm before attention/FFN |
| Single attention norm | No dual attention norm support | Only `attn_norm` used |

### Activation

| Assumption | Value | Hardcoded at |
|------------|-------|--------------|
| FFN activation | SiLU only | `silu_mul` pipeline |
| Gated FFN | Always present (SwiGLU) | `ffn_gate.weight` always looked up |
| No GELU/ReLU/etc. | No fallback | `actFn` field ignored |

### Positional Encoding

| Assumption | Value | Hardcoded at |
|------------|-------|--------------|
| Position type | RoPE only | `rope` pipeline |
| RoPE base frequency | 10000.0f | Line 138 |
| RoPE scaling | None (factor=1.0) | Line 138 |
| No ALiBi | Not implemented | No ALiBi pipeline |
| No learned positions | Not implemented | No `pos_embd` used |

### Attention

| Assumption | Value | Hardcoded at |
|------------|-------|--------------|
| Separate Q/K/V | Always (no fused attn_qkv) | Lines 112-130 |
| GQA | Supported via headCountKv | Lines 119-129 |
| No QK norm | Not implemented | No QK norm before RoPE |
| No MLA | Not implemented | No `attn_kv_b` |
| No sliding window | Not implemented | Full context attention |
| Softmax scale | 1.0/sqrt(headDim) | Line 158 |
| No soft-capping | Not implemented | Standard softmax |

### MoE

| Assumption | Value | Hardcoded at |
|------------|-------|--------------|
| MoE | Not implemented | No expert routing |
| No ffn_gate_inp | Not looked up | Single FFN path |
| No expert weights | Not supported | Dense FFN only |

---

## Buffer Allocation

The ring allocator assigns buffers at the start of each forward pass:

```
Buffer        Size                    Purpose
──────────────────────────────────────────────────────────────
hiddenAddr    dim × sizeof(float)     Hidden state (residual stream)
qAddr         seqLen × headDim ×      Q projections (all heads)
              headCount × sizeof(float)
kAddr         seqLen × headDim ×      K cache rows (GQA)
              headCountKv × sizeof(float)
vAddr         seqLen × headDim ×      V cache rows (GQA)
              headCountKv × sizeof(float)
attnOutAddr   dim × sizeof(float)     Norm output / attention output
mlpOutAddr    dim × sizeof(float)     Attention proj output / FFN output
logitsAddr    vocabSize × sizeof(float)  FFN scratch (gate, up, inter)
              + (3 × hiddenDim × sizeof(float))  ALSO used as FFN scratch
sampleOutAddr 16 bytes                Sample output (unused in forwardPartial)
```

### Memory Overlap

- `logitsAddr` is reused as FFN scratch during transformer layers:
  - `gateScratchAddr = logitsAddr`
  - `upScratchAddr = logitsAddr + hiddenDim × sizeof(float)`
  - `interScratchAddr = logitsAddr + 2 × hiddenDim × sizeof(float)`
- This only works because FFN scratch is consumed before LM head writes to `logitsAddr`
- **Critical constraint**: `vocabSize ≥ 3 × hiddenDim` (always true for real models)

---

## findTensorAddr Linear Scan Pattern

```cpp
static uint64_t findTensorAddr(const ModelDesc& model, const std::string& name) {
    for (const auto& t : model.tensors) {
        if (t.name == name) return t.gpuAddress;
    }
    return 0;  // not found → null address
}
```

**Performance concern**: Linear scan over all tensors (~1000+ for 36-layer model) for each lookup. Called ~14 times per layer + 4 times for base/final = **~508 scans per forward pass**.

**Mitigation**: All lookups happen at `forwardPartial()` entry (not per-dispatch), and tensor names are constructed via `std::to_string()` concatenation. No hash map or index.

---

## ModelHParams — Fields That Exist But Are Not Used

The `ModelHParams` struct (`include/rdna4_model_arch.hpp:60`) contains fields parsed from GGUF metadata that are **not consumed** by the forward pass:

| Field | Type | Default | Used? |
|-------|------|---------|-------|
| `actFn` | `ActFn::SILU` | SILU | ❌ Hardcoded to SiLU |
| `ropeBase` | `float` | 10000.0f | ❌ Hardcoded in push constants |
| `ropeScale` | `float` | 1.0f | ❌ Not used |
| `ropeScaled` | `bool` | false | ❌ Not used |
| `ropeFreqScale` | `float` | 1.0f | ❌ Not used |
| `ropeFreqBase` | `float` | 10000.0f | ❌ Not used |
| `ropeNTimes` | `uint32_t` | 0 | ❌ Not used |
| `useRmsNorm` | `bool` | true | ❌ Hardcoded to RMSNorm |
| `useLayerNorm` | `bool` | false | ❌ Not implemented |
| `useGqa` | `bool` | true | ✅ Used (headCountKv) |
| `useMla` | `bool` | false | ❌ Not implemented |
| `useSlidingWindow` | `bool` | false | ❌ Not implemented |
| `slidingWindow` | `uint32_t` | 0 | ❌ Not implemented |
| `useAlibi` | `bool` | false | ❌ Not implemented |
| `useClampedKqv` | `bool` | false | ❌ Not implemented |
| `attnSoftcap` | `float` | 0.0f | ❌ Not implemented |
| `useQKNorm` | `bool` | false | ❌ Not implemented |
| `isMoe` | `bool` | false | ❌ Not implemented |
| `nExperts` | `uint32_t` | 0 | ❌ Not implemented |
| `nExpertsUsed` | `uint32_t` | 0 | ❌ Not implemented |
| `moeFreq` | `uint32_t` | 0 | ❌ Not implemented |
| `moeSharedExpert` | `bool` | false | ❌ Not implemented |
| `nSharedExperts` | `uint32_t` | 0 | ❌ Not implemented |
| `tieWordEmbeddings` | `bool` | false | ⚠️ Partial (fallback lookup) |
| `tieLMHead` | `bool` | false | ⚠️ Partial (fallback lookup) |
| `useImatrix` | `bool` | false | ❌ Not used in forward |
| `defaultWeightFormat` | `QuantFormat::F16` | F16 | ❌ Not used at runtime |

### What IS used from ModelHParams

| Field | Used at |
|-------|---------|
| `dim` (embeddingLength) | Buffer sizes, GEMM dims |
| `nLayers` (blockCount) | Layer loop bound |
| `nHeads` (headCount) | Q/GEMM dims, attention loop |
| `nHeadsKv` (headCountKv) | K/V dims, GQA |
| `headDim` | RoPE, attention |
| `ffnDim` (feedForwardLength) | FFN GEMM dims, scratch sizing |
| `vocabSize` | LM head output dim |
| `contextLength` | KV cache sizing (not in forward pass directly) |
| `rmsNormEps` | ⚠️ Parsed but forward pass uses hardcoded 1e-6 |
| `ropeBase` | ⚠️ Parsed but forward pass uses hardcoded 10000.0f |

---

## Summary of Gaps

The engine currently supports a **narrow slice** of GGUF architectures:

1. **LLaMA-like only** — separate Q/K/V projections, SwiGLU FFN, RMSNorm, pre-norm, RoPE
2. **No fused QKV** — GPT-like architectures with `attn_qkv` are unsupported
3. **No MoE** — Mixtral, Qwen-MoE, DeepSeek-V3, OLMoE all unsupported
4. **No MLA** — DeepSeek-V2/V3 multi-head latent attention unsupported
5. **No alternative activations** — GELU, ReLU, GEGLU unsupported
6. **No ALiBi** — BLOOM, MPT, StarCoder unsupported
7. **No sliding window** — Mistral's sliding window attention unsupported
8. **No QK norm** — Qwen-2.5, Command-R, Nemotron unsupported
9. **No post-norm** — OLMo, Nemotron, GPT-2 post-norm unsupported
10. **No parallel residual** — GPT-NeoX, GPT-J, StableLM unsupported
11. **No encoder-decoder** — T5 family unsupported
12. **No SSM** — Mamba family unsupported
13. **No RWKV** — Different architecture entirely
14. **Hardcoded epsilon** — Model metadata `rmsNormEps` parsed but ignored
15. **Hardcoded RoPE base** — Model metadata `ropeBase` parsed but ignored
