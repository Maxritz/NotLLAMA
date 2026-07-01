# GGUF Architecture Tensor Name Mappings

Source: llama.cpp llama-arch.h / llama-arch.cpp
All architectures use `blk.N.` prefix for per-layer tensors.
Default names are LLaMA. Only deviations listed.

## Category 1: LLaMA-family (separate Q/K/V, SwiGLU, pre-norm RMSNorm, RoPE)

### llama (default)
- token_embd, output_norm, output
- blk.N: attn_norm, attn_q, attn_k, attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down

### llama2, llama3, llama3.1, llama3.2, llama3.3, llama4
Same as llama. llama4 adds MoE: ffn_gate_inp, ffn_gate_exp, ffn_down_exp, ffn_up_exp

### mistral
Same as llama. Optional sliding window attention.

### mixtral
MoE: ffn_gate_inp, ffn_gate_exp, ffn_down_exp, ffn_up_exp (8 experts, 2 active)

### qwen2, qwen2.5
Same as llama tensor names. Qwen2.5 has per-layer QK norm (attn_norm_2 used as post-QK norm in some configs).

### qwen3, qwen3.5
Same as llama. qwen3.5 = Qwen 3.5 variant.

### phi2
Same as llama. Uses parallel residual (use_parallel_residual=true).

### phi3, phimoe
Same as llama. phimoe adds MoE.

### gemma, gemma2, gemma3, gemma4
Same tensor names as llama (attn_q, attn_k, attn_v, ffn_gate, ffn_up, ffn_down).
Key differences: Q/K norm (attn_norm_2 used), V has no norm, attention scale=1.0, GELU in gate, 3 residual adds per layer.

### command-r, command-r-plus
Uses attn_qkv (fused QKV), ffn_down, ffn_up. No ffn_gate.

### internlm2
Same as llama. Uses ffn_up, ffn_down. Some variants have ffn_gate.

### minicpm, minicpm3
Same as llama.

### orion
Same as llama.

### stablelm
Same as llama. Uses parallel residual.

### olmo, olmoe
Same as llama. olmoe adds MoE.

### granite
Same as llama.

### nemotron, nemotron4, nemotron-340b, nemotron-51b
Same as llama. nemotron uses attn_qkv (fused).

### arctic
MoE with dense layers.

### chatglm4
Same as llama tensor names.

### deepseek2, deepseek3
DeepSeek MLA (multi-head latent attention): uses attn_qkv for compressed latent, plus per-head Q/K/V.
deepseek3 = DeepSeek V3 (MoE).

## Category 2: GPT-family (different naming, no RoPE or different positional)

### gpt2
- wte (token_embd), wpe (pos_embd)
- blk.N: ln_1 (attn_norm), c_attn (attn_qkv), c_proj (attn_output), ln_2 (ffn_norm), c_fc (ffn_up), c_proj (ffn_down)
- No ffn_gate. Uses GELU. Post-norm (LayerNorm after attention/FFN, not before).

### gptneox
- blk.N: input_layernorm (attn_norm), attention.query_key_value (attn_qkv), attention.dense (attn_output), post_attention_layernorm (ffn_norm), mlp.dense_h_to_4h (ffn_up), mlp.dense_4h_to_h (ffn_down)
- Uses LayerNorm, RoPE. Parallel residual option.

### gptj
Same as gptneox naming but no parallel residual.

### bloom
- word_embeddings (token_embd), word_embeddings_layernorm (attn_norm), lm_head (output)
- blk.N: input_layernorm, self_attention.query_key_value (attn_qkv), self_attention.dense (attn_output), post_attention_layernorm, mlp.dense_h_to_4h, mlp.dense_4h_to_h
- Uses ALiBi (not RoPE), LayerNorm.

### mpt
- wte (token_embd)
- blk.N: norm_1 (attn_norm), Wqkv (attn_qkv), out_proj (attn_output), norm_2 (ffn_norm), up_proj (ffn_up), down_proj (ffn_down)
- Uses ALiBi, LayerNorm.

### baichuan
- wte (token_embd)
- blk.N: input_layernorm, W_pack (attn_qkv), o_proj (attn_output), post_attention_layernorm (ffn_norm), gate_proj (ffn_up), up_proj (ffn_gate), down_proj (ffn_down)

### refact
- blk.N: norm1 (attn_norm), norm2 (ffn_norm), attn_qkv, attn_output, ffn_up, ffn_down

### falcon
- blk.N: input_layernorm (attn_norm), attention.query_key_value (attn_qkv), dense (attn_output), post_attention_layernorm (ffn_norm), mlp.dense_h_to_4h (ffn_up), mlp.dense_4h_to_h (ffn_down)
- Optional: attention_norm (attn_norm_2)

### starcoder, starcoder2
- wte (token_embd)
- blk.N: ln_1 (attn_norm), c_attn (attn_qkv), c_proj (attn_output), ln_2 (ffn_norm), c_fc (ffn_up), c_proj (ffn_down)

### gptbigcode
Same as starcoder.

## Category 3: Special Architectures

### mamba
- blk.N: norm (attn_norm), in_proj (ssm_in), conv1d (ssm_conv1d), x_proj (ssm_x), dt_proj (ssm_dt), out_proj (ssm_out)
- SSM (state space model), no attention.

### mamba2
Same as mamba with additional SSM tensors.

### rwkv5, rwkv6
- blk.N: ln1 (attn_norm), ln2 (ffn_norm), time_mix_k, time_mix_v, time_mix_r, key, value, receptance, output, ffn_up, ffn_down
- RNN-based, no attention.

### bert, modernbert
- word_embeddings (token_embd), token_type_embeddings, position_embeddings, ln_f (output_norm)
- blk.N: attention.output.query, attention.output.key, attention.output.value (separate Q/K/V), attention.output.dense (attn_output), layernorm_output (attn_norm), intermediate (ffn_up), output (ffn_down)
- No ffn_gate. Uses GELU.

### t5, t5encoder
- encoder.embed_tokens, decoder.embed_tokens
- decoder.block.N.layer.0.SelfAttention.q/k/v/o (separate Q/K/V)
- decoder.block.N.layer.1.DenseReluDense.wi/wo (FFN)

### chameleon
- Same as llama but with image token embedding tokens.

## Key Architectural Differences Matrix

| Feature | LLaMA | GPT-2 | Bloom | Mamba | Falcon |
|---------|-------|-------|-------|-------|--------|
| Norm | RMSNorm | LayerNorm | LayerNorm | RMSNorm | LayerNorm |
| Norm position | Pre-norm | Post-norm | Pre-norm | Pre-norm | Pre-norm |
| Attention QKV | Separate | Fused | Fused | N/A | Fused |
| Position | RoPE | Learned | ALiBi | N/A | ALiBi |
| FFN gate | Yes (SwiGLU) | No | No | N/A | No |
| FFN activation | SiLU | GELU | GELU | N/A | GELU |
| Parallel residual | No | No | No | N/A | Optional |
| MoE | Optional | No | No | No | No |
