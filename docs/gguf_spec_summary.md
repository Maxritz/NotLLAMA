# GGUF Specification Summary

Source: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## File Structure

GGUF files use a global alignment specified in `general.alignment` (default 32). All fields are little-endian unless the file is marked big-endian. The binary layout:

```
┌─────────────────────────────────┐
│ Header                          │
│   magic: 0x47475546 ("GGUF")    │
│   version: uint32 (must be 3)   │
│   tensor_count: uint64          │
│   metadata_kv_count: uint64     │
│   metadata_kv[count]            │
├─────────────────────────────────┤
│ Tensor Infos                    │
│   tensor_info[tensor_count]     │
│   each: name (string)           │
│         n_dimensions (uint32)   │
│         dimensions[] (uint64)   │
│         type (ggml_type uint32) │
│         offset (uint64)         │
├─────────────────────────────────┤
│ Padding (ALIGNMENT-aligned)     │
├─────────────────────────────────┤
│ Tensor Data                     │
│   raw binary weight data        │
│   each tensor at aligned offset │
└─────────────────────────────────┘
```

### String type

```c
struct gguf_string_t {
    uint64_t len;
    char     string[len]; // UTF-8, NOT null-terminated
};
```

### Metadata value type

```c
enum gguf_metadata_value_type: uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,  // 0=false, 1=true
    STRING   = 8,
    ARRAY    = 9,  // nested, with type+len prefix
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};
```

---

## Official ggml_type Enum

| Name | Value | Description |
|------|-------|-------------|
| `GGML_TYPE_F32` | 0 | 32-bit IEEE754 float |
| `GGML_TYPE_F16` | 1 | 16-bit IEEE754 float |
| `GGML_TYPE_Q4_0` | 2 | 4-bit quantization (block size 32, scale per block) |
| `GGML_TYPE_Q4_1` | 3 | 4-bit quantization (block size 32, scale+min per block) |
| `GGML_TYPE_Q5_0` | 6 | 5-bit quantization (block size 32) |
| `GGML_TYPE_Q5_1` | 7 | 5-bit quantization (block size 32, with min) |
| `GGML_TYPE_Q8_0` | 8 | 8-bit quantization (block size 32) |
| `GGML_TYPE_Q8_1` | 9 | 8-bit quantization (block size 32, with min) |
| `GGML_TYPE_Q2_K` | 10 | K-quant 2-bit (super-blocks of 256, with scales) |
| `GGML_TYPE_Q3_K` | 11 | K-quant 3-bit |
| `GGML_TYPE_Q4_K` | 12 | K-quant 4-bit |
| `GGML_TYPE_Q5_K` | 13 | K-quant 5-bit |
| `GGML_TYPE_Q6_K` | 14 | K-quant 6-bit |
| `GGML_TYPE_Q8_K` | 15 | K-quant 8-bit (block 256) |
| `GGML_TYPE_IQ2_XXS` | 16 | Importance quantization 2-bit XXS (66B/256) |
| `GGML_TYPE_IQ2_XS` | 17 | Importance quantization 2-bit XS (74B/256) |
| `GGML_TYPE_IQ3_XXS` | 18 | Importance quantization 3-bit XXS |
| `GGML_TYPE_IQ1_S` | 19 | Importance quantization 1-bit S (82B/256) |
| `GGML_TYPE_IQ4_NL` | 20 | Importance quantization 4-bit non-linear |
| `GGML_TYPE_IQ3_S` | 21 | Importance quantization 3-bit S |
| `GGML_TYPE_IQ2_S` | 22 | Importance quantization 2-bit S |
| `GGML_TYPE_IQ4_XS` | 23 | Importance quantization 4-bit XS |
| `GGML_TYPE_I8` | 24 | 8-bit signed integer |
| `GGML_TYPE_I16` | 25 | 16-bit signed integer |
| `GGML_TYPE_I32` | 26 | 32-bit signed integer |
| `GGML_TYPE_I64` | 27 | 64-bit signed integer |
| `GGML_TYPE_F64` | 28 | 64-bit IEEE754 float |
| `GGML_TYPE_IQ1_M` | 29 | Importance quantization 1-bit M |
| `GGML_TYPE_BF16` | 30 | Brain float 16 |
| `GGML_TYPE_TQ1_0` | 34 | TurboQuant 1-bit |
| `GGML_TYPE_TQ2_0` | 35 | TurboQuant 2-bit |
| `GGML_TYPE_MXFP4` | 39 | Microscaling FP4 (1 block) |
| `GGML_TYPE_COUNT` | 40 | Sentinel |

Deprecated/removed: Q4_2=4, Q4_3=5, Q4_0_4_4=31, Q4_0_4_8=32, Q4_0_8_8=33, IQ4_NL_4_4=36, IQ4_NL_4_8=37, IQ4_NL_8_8=38.

---

## Standardized Metadata Keys

### Required

| Key | Type | Description |
|-----|------|-------------|
| `general.architecture` | string | Model architecture (`llama`, `gpt2`, `mpt`, etc.) — lowercase `[a-z0-9]+` only |
| `general.quantization_version` | uint32 | Version of quantization format (required if any tensors are quantized) |
| `general.alignment` | uint32 | Global alignment, must be multiple of 8 (default 32) |

### General metadata (optional)

| Key | Type |
|-----|------|
| `general.name` | string |
| `general.author` | string |
| `general.version` | string |
| `general.organization` | string |
| `general.basename` | string |
| `general.finetune` | string |
| `general.description` | string |
| `general.quantized_by` | string |
| `general.size_label` | string |
| `general.license` | string |
| `general.file_type` | uint32 |
| `general.tags` | string[] |
| `general.languages` | string[] |

### Architecture-specific keys (`[llm]` = architecture prefix, e.g. `llama`, `qwen2`)

| Key | Type | Description |
|-----|------|-------------|
| **`[llm].block_count`** | uint64 | Number of transformer layers |
| **`[llm].embedding_length`** | uint64 | Hidden size |
| **`[llm].feed_forward_length`** | uint64 | FFN intermediate size |
| **`[llm].attention.head_count`** | uint64 | Number of attention heads |
| `[llm].context_length` | uint64 | Max context length |
| `[llm].attention.head_count_kv` | uint64 | KV heads (GQA) |
| `[llm].rope.freq_base` | float32 | RoPE base frequency |
| `[llm].rope.dimension_count` | uint64 | Partial rotary dimensions |
| `[llm].attention.layer_norm_rms_epsilon` | float32 | RMSNorm epsilon |
| `[llm].attention.layer_norm_epsilon` | float32 | LayerNorm epsilon |
| `[llm].attention.max_alibi_bias` | float32 | ALiBi max bias |
| `[llm].attention.clamp_kqv` | float32 | QKV clamp value |
| `[llm].expert_count` | uint32 | MoE expert count |
| `[llm].expert_used_count` | uint32 | MoE top-k experts per token |
| `[llm].use_parallel_residual` | bool | Parallel residual (GPT-NeoX, GPT-J) |
| `[llm].tensor_data_layout` | string | Tensor layout variant |
| `[llm].attention.key_length` | uint32 | Custom key head dim |
| `[llm].attention.value_length` | uint32 | Custom value head dim |

### RoPE scaling keys

| Key | Type |
|-----|------|
| `[llm].rope.scaling.type` | string (`none`, `linear`, `yarn`) |
| `[llm].rope.scaling.factor` | float32 |
| `[llm].rope.scaling.original_context_length` | uint32 |
| `[llm].rope.scaling.finetuned` | bool |

### SSM keys (Mamba)

| Key | Type |
|-----|------|
| `[llm].ssm.conv_kernel` | uint32 |
| `[llm].ssm.inner_size` | uint32 |
| `[llm].ssm.state_size` | uint32 |
| `[llm].ssm.time_step_rank` | uint32 |

---

## Standardized Tensor Names

### Base layers

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `token_embd.weight` | [vocabSize, dim] | Token embedding |
| `pos_embd.weight` | [maxSeqLen, dim] | Position embedding (if used) |
| `output_norm.weight` | [dim] | Final RMSNorm/LayerNorm |
| `output.weight` | [vocabSize, dim] | LM head (output projection) |

Fallback names: `norm.weight` for output_norm, `token_embd.weight` for output (weight-tied models).

### Per-layer tensors (`blk.N.*` where N = layer index)

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| `blk.N.attn_norm.weight` | [dim] | Attention pre-norm |
| `blk.N.attn_norm_2.weight` | [dim] | Second attention norm (dual-norm archs) |
| `blk.N.attn_qkv.weight` | [3*dim, dim] or per-head | Fused QKV projection |
| `blk.N.attn_q.weight` | [nHeads*headDim, dim] | Separate Q projection |
| `blk.N.attn_k.weight` | [nHeadsKv*headDim, dim] | Separate K projection |
| `blk.N.attn_v.weight` | [nHeadsKv*headDim, dim] | Separate V projection |
| `blk.N.attn_output.weight` | [dim, nHeads*headDim] | Output projection |
| `blk.N.ffn_norm.weight` | [dim] | FFN pre-norm |
| `blk.N.ffn_up.weight` | [ffnDim, dim] | FFN up projection |
| `blk.N.ffn_gate.weight` | [ffnDim, dim] | FFN gate (SwiGLU) |
| `blk.N.ffn_down.weight` | [dim, ffnDim] | FFN down projection |

### MoE per-layer tensors

| Tensor Name | Description |
|-------------|-------------|
| `blk.N.ffn_gate_inp.weight` | Expert routing / gating |
| `blk.N.ffn_gate_exp.weight` | Per-expert gate |
| `blk.N.ffn_down_exp.weight` | Per-expert down |
| `blk.N.ffn_up_exp.weight` | Per-expert up |

### SSM tensors (Mamba)

| Tensor Name | Description |
|-------------|-------------|
| `ssm_in.weight` | State space model input projection |
| `ssm_conv1d.weight` | Rolling/shift layer |
| `ssm_x.weight` | Selective parametrization |
| `ssm_a.weight` | State compression |
| `ssm_d.weight` | Skip connection |
| `ssm_dt.weight` | Time step |
| `ssm_out.weight` | Output projection |

---

## GGUF Naming Convention

Format: `[<Sidecar>]<BaseName><SizeLabel><FineTune><Version><Encoding><Type><Shard>.gguf`

- **Sidecar** (optional): `mmproj` (vision/audio), `mtp` (multi-token prediction draft)
- **BaseName**: Model name from `general.basename`
- **SizeLabel**: `<expertCount>x<count><scale>` — scale: Q=quadrillion, T=trillion, B=billion, M=million, K=thousand
- **FineTune**: Chat, Instruct, etc.
- **Version**: `v<Major>.<Minor>` (default v1.0)
- **Encoding**: Weight encoding (F16, Q4_0, Q4_K_M, etc.)
- **Type**: `LoRA` or `vocab` (optional)
- **Shard**: `<00001>-of-<00009>` (5-digit zero-padded)

### File type values (general.file_type)

```
ALL_F32 = 0, MOSTLY_F16 = 1, MOSTLY_Q4_0 = 2, MOSTLY_Q4_1 = 3,
MOSTLY_Q8_0 = 7, MOSTLY_Q5_0 = 8, MOSTLY_Q5_1 = 9,
MOSTLY_Q2_K = 10, MOSTLY_Q3_K_S = 11, MOSTLY_Q3_K_M = 12, MOSTLY_Q3_K_L = 13,
MOSTLY_Q4_K_S = 14, MOSTLY_Q4_K_M = 15, MOSTLY_Q5_K_S = 16, MOSTLY_Q5_K_M = 17,
MOSTLY_Q6_K = 18
```
