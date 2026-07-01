# GGUF Architecture Tensor Name Mappings

Complete mapping of all GGUF-supported architectures to their tensor names and characteristics. Source: llama.cpp `src/llama-arch.cpp`.

Each architecture overrides certain LLaMA-default tensor names. Below, **default** = standard LLaMA names; anything different is noted.

---

## Category 1 — LLaMA-like Architectures

These all use `blk.N.*` naming with minor variations.

### LLaMA / LLaMA-2 / LLaMA-3 / LLaMA-3.1 / LLaMA-3.2 / LLaMA-3.3 / LLaMA-4

**Base tensor map** (the reference for all others):

| Role | Tensor Name |
|------|-------------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn norm | `blk.N.attn_norm.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN norm | `blk.N.ffn_norm.weight` |
| FFN gate | `blk.N.ffn_gate.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes (`attn_qkv`)
- **FFN gate**: Yes (SwiGLU)
- **Dual attn norm**: No
- **ALiBi**: No
- **LayerNorm**: No (RMSNorm)
- **Pre-norm**: Yes
- **MoE**: LLaMA-4 only (Mixtral-style MoE variant)

**LLaMA-3/3.1/3.2/3.3 specifics**: Uses separate `attn_q`, `attn_k`, `attn_v` instead of fused `attn_qkv` in some conversions. GQA with `head_count_kv < head_count`.

**LLaMA-4**: MoE layers with `expert_count` and `expert_used_count`.

---

### Mistral / Mixtral

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |
| MoE (Mixtral) | `blk.N.ffn_gate_inp.weight`, `blk.N.ffn_gate_exp.weight`, `blk.N.ffn_down_exp.weight`, `blk.N.ffn_up_exp.weight` |

- **Fused QKV**: Yes (`attn_qkv`)
- **FFN gate**: Yes
- **Sliding window**: Mistral uses `sliding_window`
- **MoE**: Mixtral uses MoE (8 experts, top-2)
- **ALiBi**: No
- **RMSNorm**: Yes

---

### Phi-2

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |
| Uses `attn_q`, `attn_k`, `attn_v` | Separate Q/K/V |

- **Fused QKV**: No (separate Q/K/V)
- **FFN gate**: Yes
- **LayerNorm**: Yes (`layer_norm_epsilon` instead of RMSNorm)
- **Parallel residual**: Yes (`use_parallel_residual`)
- **RoPE**: Yes

---

### Phi-3 / Phi-3.5 / Phi-4 / PhiMoE

| Override | Notes |
|----------|-------|
| `blk.N.attn_norm.weight` | Same |
| `blk.N.attn_qkv.weight` | Fused QKV |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | Same |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

- **Fused QKV**: Yes
- **FFN gate**: No (no `ffn_gate` — simple FFN, not SwiGLU)
- **RMSNorm**: Yes
- **Sliding window**: Some Phi-3 variants
- **MoE**: PhiMoE uses MoE

---

### Qwen / Qwen-2 / Qwen-2.5 / Qwen-3 / Qwen-3.5

| Override | Notes |
|----------|-------|
| `token_embd.weight` | Same |
| `blk.N.attn_norm.weight` | Same |
| `blk.N.attn_q.weight` | Separate Q |
| `blk.N.attn_k.weight` | Separate K |
| `blk.N.attn_v.weight` | Separate V |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | Same |
| `blk.N.ffn_gate.weight` | Same |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

- **Fused QKV**: No (separate Q/K/V)
- **FFN gate**: Yes (SwiGLU)
- **QK norm**: Qwen-2.5+ uses `useQKNorm = true` (RMSNorm on Q and K before RoPE)
- **RMSNorm**: Yes
- **MoE**: Qwen-MoE uses `expert_count`
- **Dual attn norm**: No

---

### Command-R / Command-R-Plus (Cohere)

| Override | Notes |
|----------|-------|
| `token_embd.weight` | Same |
| `blk.N.attn_norm.weight` | Same |
| `blk.N.attn_q.weight` | Separate Q |
| `blk.N.attn_k.weight` | Separate K |
| `blk.N.attn_v.weight` | Separate V |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | Same |
| `blk.N.ffn_gate.weight` | Same |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

- **Fused QKV**: No (separate Q/K/V)
- **FFN gate**: Yes
- **ALiBi**: Yes (`max_alibi_bias`)
- **RMSNorm**: No (`layer_norm_epsilon`)
- **QK norm**: Yes (Q and K are RMS-normed before projection)
- **Logit soft-capping**: Command-R uses `attn_logit_softcapping`

---

### InternLM2

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |
| Uses `attn_qkv.weight` | Fused QKV |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **Dual attn norm**: Yes (`attn_norm` + `attn_norm_2` — both applied)
- **RMSNorm**: Yes

---

### MiniCPM

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **RMSNorm**: Yes
- **Dual attn norm**: No

---

### Orion

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **RMSNorm**: Yes

---

### StableLM

| Override | Notes |
|----------|-------|
| Same as LLaMA (with `attn_qkv`) | Fused QKV |
| `blk.N.ffn_norm.weight` | Yes |
| `use_parallel_residual = true` | Parallel residual |

- **Fused QKV**: Yes
- **FFN gate**: Yes (SwiGLU)
- **Parallel residual**: Yes
- **LayerNorm**: Yes (not RMSNorm)
- **RMSNorm**: No

---

### OLMo / OLMoE

| Override | Notes |
|----------|-------|
| `token_embd.weight` | Same |
| `output_norm.weight` | Same |
| `blk.N.attn_norm.weight` | Pre-norm |
| `blk.N.attn_qkv.weight` | Fused QKV |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | FFN norm |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

- **Fused QKV**: Yes
- **FFN gate**: No (simple FFN, not SwiGLU)
- **LayerNorm**: Yes (not RMSNorm)
- **Post-norm**: OLMo uses post-norm (norm after attention output, not before)
- **MoE**: OLMoE uses MoE with `expert_count`

---

### Granite (IBM)

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **RMSNorm**: Yes
- **Residual scale**: Granite scales residuals by `residual_scale`

---

### Nemotron

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **RMSNorm**: Yes
- **QK norm**: Yes (RMSNorm on Q and K)
- **Post-norm**: Nemotron uses post-norm (norm after attention/FFN, not before)

---

### Arctic (Snowflake)

| Override | Notes |
|----------|-------|
| Same as LLaMA | Standard `blk.N.*` |
| MoE | Uses MoE in FFN layers |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **MoE**: Yes (dense-sparse hybrid)
- **RMSNorm**: Yes

---

### ChatGLM

| Override | Notes |
|----------|-------|
| `token_embd.weight` | Same |
| `output_norm.weight` | Same |
| `blk.N.attn_norm.weight` | Same |
| `blk.N.attn_qkv.weight` | Fused QKV |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | Same |
| `blk.N.ffn_gate.weight` | Same |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

- **Fused QKV**: Yes
- **FFN gate**: Yes
- **RMSNorm**: Yes
- **Dual attention norm**: No
- **Unique**: GLM uses a specific attention mask pattern (prefix lm)

---

### DeepSeek-V2 / DeepSeek-V3

| Override | Notes |
|----------|-------|
| `token_embd.weight` | Same |
| `output_norm.weight` | Same |
| `blk.N.attn_norm.weight` | Same |
| `blk.N.attn_q.weight` | Separate Q (for MLA) |
| `blk.N.attn_kv_b.weight` | MLA compressed KV (unique) |
| `blk.N.attn_output.weight` | Same |
| `blk.N.ffn_norm.weight` | Same |
| `blk.N.ffn_up.weight` | Same |
| `blk.N.ffn_down.weight` | Same |

Unique tensors:
- `blk.N.attn_kv_b.weight` — MLA (Multi-head Latent Attention) compressed KV-b projection
- `blk.N.ffn_gate_inp.weight` — MoE routing (DeepSeek-V3)

- **Fused QKV**: No — uses MLA instead (unique compressed KV representation)
- **FFN gate**: Yes (SwiGLU)
- **MLA**: Yes (`useMla = true`)
- **MoE**: DeepSeek-V3 uses MoE with `expert_count`
- **RMSNorm**: Yes
- **QK norm**: No
- **Unique**: `attn_kv_b` replaces separate K/V projections; MLA stores compressed latent KV

---

## Category 2 — GPT-like Architectures

These use different tensor naming conventions.

### GPT-2

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Position embedding | `pos_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No (simple FFN)
- **ALiBi**: No
- **LayerNorm**: Yes (not RMSNorm)
- **Post-norm**: Yes
- **Positional**: Learned position embedding (`pos_embd`)
- **Unique**: `pos_embd.weight` — no RoPE

---

### GPT-NeoX

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN norm | `blk.N.ffn_norm.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **Parallel residual**: Yes (`use_parallel_residual`)
- **LayerNorm**: Yes
- **RoPE**: Yes

---

### GPT-J

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **Parallel residual**: Yes
- **LayerNorm**: Yes
- **RoPE**: Yes
- **Unique**: No `ffn_norm` — LayerNorm is applied before attention only

---

### BLOOM

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **ALiBi**: Yes (`max_alibi_bias`)
- **LayerNorm**: Yes
- **Post-norm**: Yes
- **Unique**: No `ffn_norm`, no RoPE, ALiBi positional encoding

---

### MPT

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **ALiBi**: Yes (`max_alibi_bias`)
- **LayerNorm**: Yes (`layer_norm_epsilon`)
- **Unique**: No `ffn_norm`, no RoPE, uses `clip_kqv`

---

### Baichuan

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **RMSNorm**: Yes
- **ALiBi**: Baichuan-1 uses ALiBi; Baichuan-2 uses RoPE
- **Unique**: No `ffn_norm`

---

### Refact

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **LayerNorm**: Yes
- **Unique**: Simple architecture, no `ffn_norm`

---

### Falcon

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes (rearranged: Q [n_heads], K [n_heads_kv], V [n_heads_kv])
- **FFN gate**: No
- **LayerNorm**: Yes (`layer_norm_epsilon`)
- **GQA**: Yes (`head_count_kv < head_count`)
- **Parallel residual**: Some variants
- **Unique**: No `ffn_norm`, QKV rearranged during conversion

---

### StarCoder / StarCoder2

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **ALiBi**: Yes
- **LayerNorm**: Yes
- **Unique**: No `ffn_norm`, no RoPE

---

### GPT-BigCode

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: No
- **LayerNorm**: Yes
- **Unique**: No `ffn_norm`

---

## Category 3 — Special Architectures

### Mamba / Mamba-2

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| SSM in | `blk.N.ssm_in.weight` |
| SSM conv1d | `blk.N.ssm_conv1d.weight` |
| SSM x | `blk.N.ssm_x.weight` |
| SSM a | `blk.N.ssm_a.weight` |
| SSM d | `blk.N.ssm_d.weight` |
| SSM dt | `blk.N.ssm_dt.weight` |
| SSM out | `blk.N.ssm_out.weight` |

- **No attention**: SSM (State Space Model) — no Q/K/V/attention tensors
- **RMSNorm**: Yes
- **Unique**: `ssm_in`, `ssm_conv1d`, `ssm_x`, `ssm_a`, `ssm_d`, `ssm_dt`, `ssm_out`
- **Mamba-2**: Adds `ssm_d_conv` and `ssm_dt_bias`

---

### RWKV-5 / RWKV-6

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn time mix | `blk.N.att.time_mix_k.weight` |
| Attn time mix | `blk.N.att.time_mix_v.weight` |
| Attn time mix | `blk.N.att.time_mix_r.weight` |
| Attn key | `blk.N.att.key.weight` |
| Attn value | `blk.N.att.value.weight` |
| Attn receptance | `blk.N.att.receptance.weight` |
| Attn output | `blk.N.att.output.weight` |
| FFN time mix | `blk.N.ffn.time_mix_k.weight` |
| FFN time mix | `blk.N.ffn.time_mix_r.weight` |
| FFN key | `blk.N.ffn.key.weight` |
| FFN receptance | `blk.N.ffn.receptance.weight` |
| FFN value | `blk.N.ffn.value.weight` |

- **No attention**: WKV (linear attention variant)
- **Unique naming**: `att.*` and `ffn.*` with `time_mix_*` tensors
- **No RoPE, no ALiBi**: WKV mechanism is its own positional encoding
- **LayerNorm**: Yes (not RMSNorm)

---

### BERT / ModernBERT

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Position embedding | `pos_embd.weight` |
| Token type embedding | `token_embd_type.weight` |
| Norm | `output_norm.weight` |
| Pooler | `output_norm.bias` |
| Attn Q | `blk.N.attn_q.weight` |
| Attn K | `blk.N.attn_k.weight` |
| Attn V | `blk.N.attn_v.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Separate Q/K/V**: Yes
- **FFN gate**: No
- **LayerNorm**: Yes
- **Positional**: Learned position embedding (`pos_embd`)
- **Unique**: `token_embd_type.weight` (segment embeddings), pooler output
- **ModernBERT**: Uses RoPE instead of learned positions, adds `attn_norm` (pre-norm), `rotary_emb`

---

### T5 / Flan-T5 / UL2

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Decoder norm | `output_norm.weight` |
| LM head | `output.weight` |
| Encoder block `blk.N.*` | Standard attention + FFN |
| Decoder block `blk.N.*` | Standard attention + FFN + cross-attention |

Unique tensors:
- `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight` (separate)
- `blk.N.attn_output.weight`
- `blk.N.ffn_norm.weight`
- `blk.N.ffn_up.weight` (Wi)
- `blk.N.ffn_down.weight` (wo)
- Relative position bias: `blk.N.attn_rel_b.weight`

- **Encoder-decoder**: Unique architecture (not decoder-only)
- **Relative position bias**: Yes (not RoPE, not ALiBi)
- **LayerNorm**: Yes (`layer_norm_epsilon`)
- **FFN gate**: No (uses `GEGLU`: `ffn_up` + `ffn_gate` both present, GEGLU activation)

---

### Chameleon

| Tensor | Name |
|--------|------|
| Token embedding | `token_embd.weight` |
| Output norm | `output_norm.weight` |
| LM head | `output.weight` |
| Attn norm | `blk.N.attn_norm.weight` |
| Attn QKV (fused) | `blk.N.attn_qkv.weight` |
| Attn output | `blk.N.attn_output.weight` |
| FFN norm | `blk.N.ffn_norm.weight` |
| FFN gate | `blk.N.ffn_gate.weight` |
| FFN up | `blk.N.ffn_up.weight` |
| FFN down | `blk.N.ffn_down.weight` |

- **Fused QKV**: Yes
- **FFN gate**: Yes (SwiGLU)
- **RMSNorm**: Yes
- **Unique**: Codebook-based discrete visual tokens; uses VQ-VAE for images
- **Architecture**: LLaMA-like but with multi-modal (text + image) support

---

## Summary Comparison Table

| Architecture | Fused QKV | FFN Gate | Dual AttnNorm | Position | Norm | MoE | MLA | Unique Features |
|-------------|-----------|----------|---------------|----------|------|-----|-----|-----------------|
| LLaMA | Yes | Yes | No | RoPE | RMS | No | No | — |
| LLaMA-4 | Yes | Yes | No | RoPE | RMS | Yes | No | MoE |
| Mistral | Yes | Yes | No | RoPE | RMS | No | No | Sliding window |
| Mixtral | Yes | Yes | No | RoPE | RMS | Yes | No | MoE, top-2 |
| Phi-2 | No | Yes | No | RoPE | LN | No | No | Parallel residual |
| Phi-3 | Yes | No | No | RoPE | RMS | No | No | Simple FFN |
| Qwen2 | No | Yes | No | RoPE | RMS | No | No | Separate Q/K/V |
| Qwen2.5 | No | Yes | No | RoPE | RMS | No | No | QK norm |
| Command-R | No | Yes | No | ALiBi | LN | No | No | QK norm, softcap |
| InternLM2 | Yes | Yes | Yes | RoPE | RMS | No | No | Dual attn norm |
| OLMo | Yes | No | No | RoPE | LN | Optional | No | Post-norm |
| DeepSeek-V2 | No* | Yes | No | RoPE | RMS | No | Yes | MLA, attn_kv_b |
| DeepSeek-V3 | No* | Yes | No | RoPE | RMS | Yes | Yes | MLA + MoE |
| GPT-2 | Yes | No | No | Learned | LN | No | No | Post-norm |
| GPT-NeoX | Yes | No | No | RoPE | LN | No | No | Parallel residual |
| GPT-J | Yes | No | No | RoPE | LN | No | No | Parallel residual |
| BLOOM | Yes | No | No | ALiBi | LN | No | No | Post-norm |
| MPT | Yes | No | No | ALiBi | LN | No | No | clip_kqv |
| Falcon | Yes | No | No | — | LN | No | No | GQA |
| StarCoder | Yes | No | No | ALiBi | LN | No | No | — |
| Mamba | N/A | N/A | N/A | N/A | RMS | No | No | SSM tensors |
| RWKV | N/A | N/A | N/A | WKV | LN | No | No | time_mix_* |
| BERT | No | No | No | Learned | LN | No | No | token_embd_type |
| T5 | No | Yes(GEGLU) | No | RelPos | LN | No | No | Encoder-decoder |
| Chameleon | Yes | Yes | No | RoPE | RMS | No | No | VQ-VAE tokens |
