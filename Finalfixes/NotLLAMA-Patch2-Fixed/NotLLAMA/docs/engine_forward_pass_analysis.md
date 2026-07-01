# Engine Forward Pass Architecture Analysis

Source: src/host/inference_engine.cpp, include/rdna4_engine.hpp, include/rdna4_types.hpp

## Current Dispatch Sequence (forwardPartial)

### Per-Layer Pipeline
1. Embed: embed.comp — copies token_embd[tokenId] to hiddenAddr
2. Attn Norm: rms_norm — hiddenAddr → attnOutAddr, weight=blk.N.attn_norm
3. Q GEMM: gemm — attnOutAddr → qAddr[row], weight=blk.N.attn_q
4. K GEMM: gemm — attnOutAddr → kAddr[row], weight=blk.N.attn_k
5. V GEMM: gemm — attnOutAddr → vAddr[row], weight=blk.N.attn_v
6. RoPE: rope — in-place on qAddr[row], kAddr[row]
7. KV Cache Write: kv_cache_write — kAddr/vAddr → cache[layer]
8. Attention: attention — per-head loop, 1 dispatch per head
9. Output Proj GEMM: gemm — attnOutAddr → mlpOutAddr, weight=blk.N.attn_output
10. Residual Add: add — hiddenAddr + mlpOutAddr → hiddenAddr
11. FFN Norm: rms_norm — hiddenAddr → attnOutAddr, weight=blk.N.ffn_norm
12. Gate GEMM: gemm — attnOutAddr → logitsAddr, weight=blk.N.ffn_gate
13. Up GEMM: gemm — attnOutAddr → logitsAddr+hiddenDim, weight=blk.N.ffn_up
14. SiLU+Mul: silu_mul — gate * SiLU(up) → logitsAddr+2*hiddenDim
15. Down GEMM: gemm — logitsAddr+2*hiddenDim → mlpOutAddr, weight=blk.N.ffn_down
16. Residual Add: add — hiddenAddr + mlpOutAddr → hiddenAddr

### Post-Layer
17. Final Norm: rms_norm — hiddenAddr → attnOutAddr, weight=output_norm (or norm)
18. LM Head GEMM: gemm — attnOutAddr → logitsAddr, weight=output (or token_embd, transB=1)

## Hardcoded Tensor Names
```
"token_embd.weight"
"blk.{layer}.attn_norm.weight"
"blk.{layer}.attn_q.weight"
"blk.{layer}.attn_k.weight"
"blk.{layer}.attn_v.weight"
"blk.{layer}.attn_output.weight"
"blk.{layer}.ffn_norm.weight"
"blk.{layer}.ffn_gate.weight"
"blk.{layer}.ffn_up.weight"
"blk.{layer}.ffn_down.weight"
"output_norm.weight" / "norm.weight"
"output.weight" / "token_embd.weight"
```

## Hardcoded Assumptions

| Element | Hardcoded Value | What It Should Be |
|---------|----------------|-------------------|
| Tensor prefix | "blk.{i}." | Architecture-dependent (gpt2 uses different format) |
| Norm type | RMSNorm only | Some architectures use LayerNorm (gpt2, bloom, bert) |
| Norm epsilon | 1e-6f | Read from metadata: [arch].attention.layer_norm_rms_epsilon |
| Activation | SiLU only | Some use GELU (gpt2, falcon) or ReLU (old architectures) |
| RoPE base | 10000.0f | Read from metadata: [arch].rope.freq_base |
| RoPE scale | 1.0f | Read from metadata: [arch].rope.scaling.factor |
| QKV layout | Separate Q/K/V | Some use fused attn_qkv (gpt2, falcon, bloom) |
| FFN gate | Always present | Some architectures have no ffn_gate (gpt2, bloom) |
| Attention | Always standard | No MLA (deepseek2), no sliding window, no ALiBi |
| headDim | dim/headCount | Should read from metadata if available |
| MoE | Not supported | Should check expert_count/expert_used_count |
| QK norm | Not supported | Some architectures (gemma) need post-Q/K norm |
| KV cache dtype | float32 | Should use float16 for efficiency |

## Buffer Allocation (ring allocator)
- hiddenAddr: dim * sizeof(float) — persistent hidden state
- qAddr: seqLen * headDim * nHeads * sizeof(float)
- kAddr: seqLen * headDim * nKvHeads * sizeof(float)
- vAddr: seqLen * headDim * nKvHeads * sizeof(float)
- attnOutAddr: dim * sizeof(float) — scratch for norm/attention output
- mlpOutAddr: dim * sizeof(float) — scratch for proj/FFN output
- logitsAddr: vocabSize * sizeof(float) — reused as FFN scratch during layers
- sampleOutAddr: 16 bytes

## findTensorAddr Pattern
Linear scan O(n) over model.tensors vector, called ~12 times per layer per forward pass.
Should be replaced with precomputed address lookup table indexed by architecture config.

## ModelHParams (rdna4_model_arch.hpp) — Available But Unused
- actFn (SILU, GELU, RELU, etc.)
- ropeBase, ropeScale
- useQKNorm
- useSlidingWindow
- useAlibi
- isMoe, nExperts
- useParallelResidual
- attnSoftcap
- tieWordEmbeddings, tieLMHead
- LayerConfig per-layer overrides
