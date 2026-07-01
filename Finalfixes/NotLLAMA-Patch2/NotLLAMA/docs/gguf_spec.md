# GGUF Specification Summary

Source: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## File Structure
1. Header: magic (0x47475546="GGUF"), version (3), tensor_count, metadata_kv_count
2. Metadata KV pairs (key-value store, extensible)
3. Tensor infos (name, n_dims, dimensions, type, offset)
4. Padding to alignment
5. Tensor data (aligned to general.alignment, default 32)

## Official ggml_type Enum

```
GGML_TYPE_F32     = 0
GGML_TYPE_F16     = 1
GGML_TYPE_Q4_0    = 2
GGML_TYPE_Q4_1    = 3
GGML_TYPE_Q5_0    = 6
GGML_TYPE_Q5_1    = 7
GGML_TYPE_Q8_0    = 8
GGML_TYPE_Q8_1    = 9
GGML_TYPE_Q2_K    = 10
GGML_TYPE_Q3_K    = 11
GGML_TYPE_Q4_K    = 12
GGML_TYPE_Q5_K    = 13
GGML_TYPE_Q6_K    = 14
GGML_TYPE_Q8_K    = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_I8      = 24
GGML_TYPE_I16     = 25
GGML_TYPE_I32     = 26
GGML_TYPE_I64     = 27
GGML_TYPE_F64     = 28
GGML_TYPE_IQ1_M   = 29
GGML_TYPE_BF16    = 30
GGML_TYPE_TQ1_0   = 34
GGML_TYPE_TQ2_0   = 35
GGML_TYPE_MXFP4   = 39
GGML_TYPE_COUNT   = 40
```

## Standardized Metadata Keys

### Required
- `general.architecture: string` — architecture name (llama, gemma, qwen2, etc.)
- `general.quantization_version: uint32`
- `general.alignment: uint32` — default 32

### Per-Architecture [llm]
- `[llm].context_length: uint64`
- `[llm].embedding_length: uint64`
- `[llm].block_count: uint64`
- `[llm].feed_forward_length: uint64`
- `[llm].use_parallel_residual: bool`
- `[llm].expert_count: uint32`
- `[llm].expert_used_count: uint32`

### Attention
- `[llm].attention.head_count: uint64`
- `[llm].attention.head_count_kv: uint64`
- `[llm].attention.layer_norm_rms_epsilon: float32`
- `[llm].attention.layer_norm_epsilon: float32`
- `[llm].attention.key_length: uint32` — optional, default n_embd/n_head
- `[llm].attention.value_length: uint32` — optional

### RoPE
- `[llm].rope.dimension_count: uint64`
- `[llm].rope.freq_base: float32`
- `[llm].rope.scaling.type: string` — none/linear/yarn
- `[llm].rope.scaling.factor: float32`

### SSM (Mamba)
- `[llm].ssm.conv_kernel: uint32`
- `[llm].ssm.inner_size: uint32`
- `[llm].ssm.state_size: uint32`
- `[llm].ssm.time_step_rank: uint32`

## Standardized Tensor Names

### Base layers
- `token_embd.weight` — Token embedding
- `pos_embd.weight` — Position embedding
- `output_norm.weight` — Output normalization
- `output.weight` — Output/LM head

### Per-layer blocks (blk.N.*)
- `blk.N.attn_norm.weight` — Attention normalization
- `blk.N.attn_norm_2.weight` — Second attention norm
- `blk.N.attn_qkv.weight` — Fused QKV projection
- `blk.N.attn_q.weight` — Query projection
- `blk.N.attn_k.weight` — Key projection
- `blk.N.attn_v.weight` — Value projection
- `blk.N.attn_output.weight` — Attention output projection
- `blk.N.ffn_norm.weight` — FFN normalization
- `blk.N.ffn_up.weight` — FFN up projection
- `blk.N.ffn_gate.weight` — FFN gate projection
- `blk.N.ffn_down.weight` — FFN down projection
- `blk.N.ffn_gate_inp.weight` — MoE expert routing
- `blk.N.ffn_gate_exp.weight` — MoE expert gate
- `blk.N.ffn_down_exp.weight` — MoE expert down
- `blk.N.ffn_up_exp.weight` — MoE expert up

### SSM layers
- `blk.N.ssm_in.weight`, `blk.N.ssm_conv1d.weight`, `blk.N.ssm_x.weight`
- `blk.N.ssm_a.weight`, `blk.N.ssm_d.weight`, `blk.N.ssm_dt.weight`
- `blk.N.ssm_out.weight`

## Type Traits

| Type | Block Size | Bytes/Block |
|------|-----------|-------------|
| F32 | 1 | 4 |
| F16 | 1 | 2 |
| Q4_0 | 32 | 18 |
| Q4_1 | 32 | 24 |
| Q5_0 | 32 | 26 |
| Q5_1 | 32 | 32 |
| Q8_0 | 32 | 34 |
| Q8_1 | 32 | 36 |
| Q2_K | 256 | 84 |
| Q3_K | 256 | 110 |
| Q4_K | 256 | 144 |
| Q5_K | 256 | 176 |
| Q6_K | 256 | 210 |
| Q8_K | 256 | 272 |
