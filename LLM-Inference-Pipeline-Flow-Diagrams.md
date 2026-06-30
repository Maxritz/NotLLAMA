# LLM Inference Pipeline — Four Implementations Compared
## Function-Level Flow Diagrams & API Reference

**Date:** 2026-06-29  
**Scope:** NotLLAMA (Vulkan), llama.cpp (Vulkan), VindexLLM (Vulkan), vLLM (CUDA/ROCm)  
**Focus:** Function-level logic, NOT graphics API calls. API table provided at the end.

---

## Table of Contents
1. [Universal LLM Inference Pipeline](#1-universal-llm-inference-pipeline)
2. [NotLLAMA Pipeline](#2-notllama-pipeline)
3. [llama.cpp Pipeline](#3-llamacpp-pipeline)
4. [VindexLLM Pipeline](#4-vindexllm-pipeline)
5. [vLLM Pipeline](#5-vllm-pipeline)
6. [Consolidated API Comparison Table](#6-consolidated-api-comparison-table)
7. [Key Architectural Differences](#7-key-architectural-differences)

---

## 1. Universal LLM Inference Pipeline

Every LLM inference engine follows this conceptual flow, regardless of backend:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MODEL LOAD                                                        │
│  ─────────────                                                              │
│  read_model_file(path)                                                      │
│    ├─ parse_header() → architecture, dims, layers, heads, vocab_size       │
│    ├─ read_tensor_metadata() → name, shape, dtype, offset for each tensor │
│    ├─ read_metadata_kv() → hyperparameters, tokenizer config               │
│    ├─ decide_dequant_strategy() → when/how to convert quantized weights    │
│    └─ upload_weights_to_accelerator() → GPU/TPU/NPU memory                  │
│                                                                              │
│  initialize_compute_backend()                                               │
│    ├─ detect_accelerator() → CUDA, ROCm, Vulkan, Metal, CPU                │
│    ├─ load_kernels/shaders() → compile or JIT-load compute programs        │
│    ├─ allocate_kv_cache() → n_layers × n_kv_heads × max_seq × head_dim    │
│    └─ create_command_graph() → define execution order                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TOKENIZATION                                                      │
│  ─────────────────                                                          │
│  tokenize(prompt_string)                                                    │
│    ├─ load_vocabulary() → token↔id mapping, merge rules, special tokens   │
│    ├─ subword_segment() → BPE or SentencePiece or TikToken                 │
│    │   └─ greedy longest-match tokenization                                 │
│    └─ return token_id[] array                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: PREFILL (Process Prompt)                                          │
│  ────────────────────────────────                                           │
│  FOR each token in prompt_tokens:                                           │
│    ├─ embed(token_id) → hidden[model_dim]                                  │
│    │   └─ lookup embedding_table[token_id] → float vector                   │
│    │                                                                        │
│    ├─ FOR layer = 0 to n_layers-1:                                          │
│    │   ├─ rms_norm(hidden) → normed[dim]                                    │
│    │   │   └─ sum_of_squares = Σ(hidden[i]²) / dim                         │
│    │   │   └─ inv_rms = 1 / √(sum_of_squares + ε)                         │
│    │   │   └─ normed[i] = hidden[i] × inv_rms × weight[i]                │
│    │   │                                                                    │
│    │   ├─ qkv_projection(normed) → Q[heads×head_dim], K[heads×head_dim], V[heads×head_dim]│
│    │   │   └─ matvec(normed, w_q) → Q                                       │
│    │   │   └─ matvec(normed, w_k) → K                                       │
│    │   │   └─ matvec(normed, w_v) → V                                       │
│    │   │                                                                    │
│    │   ├─ apply_rope(Q, K, position) → rotated Q, K                       │
│    │   │   └─ for each pair (i, i+1):                                      │
│    │   │       q[i]   = q[i]   × cos(mθ) - q[i+1] × sin(mθ)               │
│    │   │       q[i+1] = q[i+1] × cos(mθ) + q[i]   × sin(mθ)               │
│    │   │       (same for K)                                                 │
│    │   │                                                                    │
│    │   ├─ kv_cache_write(layer, position, K, V)                            │
│    │   │   └─ K_cache[layer][head][position][0..head_dim-1] = K            │
│    │   │   └─ V_cache[layer][head][position][0..head_dim-1] = V            │
│    │   │                                                                    │
│    │   ├─ attention(Q, K_cache, V_cache, seq_len) → attn_out[dim]          │
│    │   │   ├─ FOR each head:                                               │
│    │   │   │   ├─ scores[seq_len] = dot(Q_head, K_cache[head][0..seq_len-1])│
│    │   │   │   ├─ scores = scores / √(head_dim)                            │
│    │   │   │   ├─ softmax(scores) → weights[seq_len]                        │
│    │   │   │   └─ attn_head = Σ(weights[t] × V_cache[head][t]) for t=0..seq_len-1│
│    │   │   └─ concat all attn_head → attn_out[dim]                         │
│    │   │                                                                    │
│    │   ├─ attention_projection(attn_out) → attn_proj[dim]                   │
│    │   │   └─ matvec(attn_out, w_o)                                        │
│    │   │                                                                    │
│    │   ├─ add_residual(hidden, attn_proj) → hidden = hidden + attn_proj   │
│    │   │                                                                    │
│    │   ├─ rms_norm(hidden) → normed[dim]  // pre-FFN                       │
│    │   │                                                                    │
│    │   ├─ ffn(normed) → ffn_out[dim]                                        │
│    │   │   ├─ gate = matvec(normed, w_gate) → gate[hidden_dim]             │
│    │   │   ├─ up   = matvec(normed, w_up)   → up[hidden_dim]               │
│    │   │   ├─ activated = silu_mul(gate, up)                               │
│    │   │   │   └─ SiLU(x) = x × sigmoid(x)                                 │
│    │   │   │   └─ activated[i] = SiLU(gate[i]) × up[i]                    │
│    │   │   └─ down = matvec(activated, w_down) → ffn_out[dim]              │
│    │   │                                                                    │
│    │   └─ add_residual(hidden, ffn_out) → hidden = hidden + ffn_out       │
│    │                                                                        │
│    ├─ final_rms_norm(hidden) → normed[dim]                                  │
│    ├─ lm_head(normed) → logits[vocab_size]                                   │
│    │   └─ matvec(normed, w_lm_head)                                          │
│    └─ sample(logits) → next_token_id                                        │
│        ├─ apply_temperature(logits, temp)                                    │
│        ├─ apply_top_k(logits, k)                                             │
│        ├─ apply_top_p(logits, p)                                             │
│        └─ multinomial_sample(probs) → token_id                               │
│  ENDFOR                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: GENERATION (Auto-Regressive)                                      │
│  ─────────────────────────────────────                                      │
│  WHILE next_token != EOS AND len < max_length:                             │
│    ├─ embed(last_token_id) → hidden[dim]                                     │
│    ├─ FOR layer = 0 to n_layers-1:                                         │
│    │   ├─ rms_norm(hidden) → normed                                         │
│    │   ├─ qkv_projection(normed) → Q, K, V                                  │
│    │   ├─ apply_rope(Q, K, current_position)                               │
│    │   ├─ kv_cache_write(layer, current_position, K, V)                    │
│    │   ├─ attention(Q, K_cache[0..current_position], V_cache[0..current_position])│
│    │   │   └─ NOW: K/V cache has >1 entry → non-trivial softmax            │
│    │   ├─ attention_projection → attn_proj                                 │
│    │   ├─ add_residual → hidden                                             │
│    │   ├─ rms_norm → normed                                                │
│    │   ├─ ffn(normed) → ffn_out                                            │
│    │   └─ add_residual → hidden                                            │
│    ├─ final_rms_norm → normed                                              │
│    ├─ lm_head → logits[vocab_size]                                           │
│    ├─ sample(logits) → next_token_id                                         │
│    ├─ append next_token_id to output                                         │
│    └─ current_position += 1                                                  │
│  END                                                                         │
│                                                                              │
│  detokenize(output_token_ids) → output_string                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. NotLLAMA Pipeline

**Philosophy:** "Many small units doing work together." Weights stay quantized on GPU. Each layer is a self-contained work unit. No pre-dequantized buffers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MODEL LOAD                                                        │
│  ─────────────                                                              │
│  read_gguf_file(path)                                                         │
│    ├─ parse_header() → magic, version, n_tensors, n_kv                        │
│    ├─ read_tensor_infos() → name, shape, type, offset for each tensor         │
│    │   └─ types: Q4_0, Q8_0, Q6_K, F32, F16                                 │
│    ├─ read_metadata_kv() → arch, vocab_size, n_layers, dim, n_heads, etc.      │
│    ├─ allocate_gpu_buffers()                                                  │
│    │   ├─ for each tensor: allocate_device_buffer() + upload_quantized_bytes()│
│    │   │   └─ weights uploaded AS-IS (quantized), no conversion               │
│    │   └─ embedding table → GPU buffer (format-preserved)                     │
│    └─ compile_shaders()                                                         │
│        ├─ embed_q8_0.comp, embed_q6_k.comp, embed.comp                        │
│        ├─ matvec_q8_0.comp, matvec_q6_k.comp, matvec_q4_0.comp                │
│        ├─ gemm_fallback.comp (F32 GEMM for non-quant weights)                │
│        ├─ rms_norm.comp, rope.comp, attention.comp                            │
│        ├─ kv_cache_write.comp, silu_mul.comp, add_residual.comp               │
│        └─ create_pipeline_objects() → one VkPipeline per shader variant         │
│                                                                              │
│  create_context()                                                             │
│    ├─ create_vulkan_device() → compute-only queue preferred                   │
│    ├─ create_command_pool() + allocate_command_buffers()                      │
│    ├─ create_fence_pool() → one fence per layer + final                       │
│    └─ allocate_kv_cache() → n_layers × n_kv_heads × max_seq × head_dim       │
│         └─ per-layer VkBuffer, GPU-only memory                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TOKENIZATION                                                      │
│  ─────────────────                                                          │
│  tokenize_prompt(string prompt)                                               │
│    ├─ load_vocab_from_gguf() → vocab_size tokens, scores, type               │
│    ├─ bpe_encode() or spm_encode() depending on model type                   │
│    │   └─ greedy longest-match tokenization                                   │
│    └─ return vector<int32_t> token_ids                                        │
│                                                                              │
│  allocate_token_buffer() → GPU buffer for token IDs (uint32_t[])             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: PREFILL / GENERATION — forwardPartial(token_id, position)          │
│  ────────────────────────────────────────────────────────────────            │
│                                                                              │
│  beginBatch(0)                                                                │
│    ├─ dispatch_embed_shader()                                                 │
│    │   ├─ select_embed_shader_by_format() → embed_q8_0 / q6_k / f32           │
│    │   ├─ local_size = 256 threads                                            │
│    │   ├─ read embedding_table[token_id] from quantized GPU buffer           │
│    │   ├─ on-the-fly_dequant() → d × q → float                               │
│    │   └─ write hidden[0..dim-1] → output buffer                             │
│    └─ barrier() // workgroup sync                                             │
│                                                                              │
│  endBatch() + syncAllThrottled() // CPU waits for GPU fence                   │
│                                                                              │
│  FOR layer = 0 to n_layers-1:                                                 │
│    beginBatch(0)                                                              │
│      ├─ dispatch_rms_norm()                                                   │
│      │   ├─ local_size = 256, workgroups = 1                                 │
│      │   ├─ strided_sum_of_squares() across dim elements                      │
│      │   ├─ thread_0_computes_inv_rms() = 1 / sqrt(sumSq / dim + eps)        │
│      │   └─ all threads: out[d] = in[d] × inv_rms × weight[d]                │
│      │       ⚠️ BUG: 256 threads, RDNA Wave32 → reduction may be incomplete   │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_matvec_qkv() × 3 (one dispatch per projection)             │
│      │   ├─ select_matvec_by_quant_format() → q8_0 / q6_k / q4_0 / fallback │
│      │   ├─ local_size = 64, dispatch = (outDim + 63) / 64 workgroups        │
│      │   ├─ cooperative_load_input_tile(256) → shared_memory                  │
│      │   ├─ for each tile element:                                            │
│      │   │   ├─ compute_flat_weight_index(transB-aware)                       │
│      │   │   ├─ find_quant_block = weightIdx / blockSize                      │
│      │   │   ├─ read_f16_scale_d_from_block_header()                          │
│      │   │   │   └─ little_endian_unpack: uint(low) | (uint(high) << 8)      │
│      │   │   ├─ read_quantized_byte() + sign_extend_u8()                      │
│      │   │   │   └─ int(int8_t(v)) // explicit sign extension                 │
│      │   │   └─ accumulate: acc += inputTile[k] × d × q                       │
│      │   ├─ barrier() per tile                                              │
│      │   └─ write: Q[headDim×nHeads], K[headDim×nKv], V[headDim×nKv]        │
│      │       ⚠️ BUG: no bounds check if outDim not multiple of 64            │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_rope()                                                         │
│      │   ├─ local_size = 32                                                   │
│      │   └─ apply_rotary_position_embedding(Q, K, position)                   │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_kv_cache_write()                                             │
│      │   └─ write K, V into kv_cache[layer][head][position]                 │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ FOR each head: dispatch_attention()                                   │
│      │   ├─ local_size = 32 (one warp per head)                               │
│      │   ├─ Q_head = Q[head_offset .. head_offset+headDim]                   │
│      │   ├─ K_head = read_from_kv_cache(layer, head, 0..seqLen)             │
│      │   ├─ scores = dot(Q_head, K_head[seqPos]) for each seqPos            │
│      │   ├─ softmax(scores) // seqLen=1 → trivial (weight=1.0)               │
│      │   └─ V_head = read_from_kv_cache weighted by softmax → attnOut         │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_matvec_attn_proj()                                           │
│      │   └─ matvec on attention output → attnProjection[dim]                  │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_add_residual()                                               │
│      │   └─ hidden += attnOut (element-wise)                                 │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_rms_norm() // pre-FFN                                        │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_matvec_gate_up() × 2                                         │
│      │   └─ gate[hiddenDim], up[hiddenDim]                                     │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_silu_mul()                                                   │
│      │   └─ out[i] = gate[i] × SiLU(gate[i]) × up[i]                         │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_matvec_down()                                                │
│      │   └─ ffnOut[dim]                                                       │
│      ├─ barrier()                                                             │
│      │                                                                        │
│      ├─ dispatch_add_residual()                                               │
│      │   └─ hidden += ffnOut                                                  │
│      └─ endBatch() + syncAllThrottled() // fence wait per layer              │
│          ⚠️ BUG: 37 fence waits per token → TDR/hang on Windows              │
│                                                                              │
│  final_rms_norm()                                                             │
│  barrier()                                                                    │
│  dispatch_lm_head_gemm()                                                      │
│    └─ hidden[dim] × lm_head[dim × vocabSize] → logits[vocabSize]              │
│  syncAllThrottled() // final fence wait                                       │
│                                                                              │
│  read_back_logits() → CPU buffer                                               │
│  sampleArgmax(logits) OR temperature_sample(logits, temp, top_k, top_p)      │
│    → next_token_id                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Known Issues (Current State):**
- **RMS Norm:** 256-thread reduction may fail on Wave32 (RDNA). `sumSq` incomplete → `invRms` wrong → NaN.
- **Fallback GEMM:** `gemm.comp` may have wrong transpose/stride assumptions. MaxAE=10–24 even with F32 weights.
- **Bounds:** No `if (n >= outDim) return;` guard → out-of-bounds writes when dim not multiple of 64.
- **Sync:** 37 fence waits per token causes Windows TDR (>2s GPU stall).
- **Sign Extension:** Quantized shaders may read `uint8_t` without `int8_t()` cast → unsigned values.
- **Q6_K Layout:** `d-last` at offset 208 may be misread if indexing formula is wrong.

---

## 3. llama.cpp Pipeline

**Philosophy:** Pre-dequant ALL weights to F16 at load time. Single-submit per token with tiled GEMM and Flash Attention. Mature, battle-tested across all GPU vendors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MODEL LOAD                                                        │
│  ─────────────                                                              │
│  llama_model_load(path)                                                       │
│    ├─ gguf_init_from_file() → parse header, tensor infos, metadata         │
│    ├─ llama_model_loader::load_all_data()                                   │
│    │   ├─ FOR each tensor:                                                  │
│    │   │   ├─ read_quantized_bytes_from_gguf()                              │
│    │   │   ├─ llama_tensor_get_type() → GGML_TYPE_Q8_0, Q6_K, etc.         │
│    │   │   ├─ CPU-side_dequantization() → F32 staging buffer               │
│    │   │   │   ├─ block_q8_0: d × int8_t(qs[j])                             │
│    │   │   │   ├─ block_q6_K: d × sc × (val - 32)  // d-last layout        │
│    │   │   │   ├─ block_q4_0: d × ((nibble & 0xF) - 8)                      │
│    │   │   │   └─ → F16 GPU buffer                                          │
│    │   │   └─ vkCreateBuffer() + vkMapMemory() + memcpy() + vkUnmapMemory()│
│    │   └─ ALL weights now in F16 on GPU (~2× VRAM usage)                     │
│    ├─ build_tensor_graph() → ggml compute graph                              │
│    └─ init_vulkan_backend()                                                   │
│        ├─ ggml_vk_init()                                                        │
│        │   ├─ pick_physical_device() → prefers ACE/Family 1 (no graphics)    │
│        │   ├─ query_feature_chain() → ALL 14 features (NO zeroing)          │
│        │   │   └─ shaderInt64, shaderInt16, bufferDeviceAddress,             │
│        │   │       shaderFloat16, shaderInt8, storageBuffer8BitAccess,       │
│        │   │       scalarBlockLayout, timelineSemaphore, sync2, maint4,     │
│        │   │       cooperativeMatrix, dynamicRendering, etc.                │
│        │   └─ vkCreateDevice() → PASS ALL features                           │
│        │       └─ env: GGML_VK_ALLOW_GRAPHICS_QUEUE=1 → Family 0, +56% perf  │
│        ├─ compile_shaders() → SPIR-V from GLSL source                        │
│        │   ├─ dequant shaders (compile-time only, not runtime)               │
│        │   ├─ gemm.comp → tiled 16×16 F16 GEMM                               │
│        │   │   └─ dotPacked4x8EXT for sub-byte packed ops                    │
│        │   ├─ flash_attn.comp → tiled Flash Attention                         │
│        │   ├─ rms_norm.comp, rope.comp, silu.comp, add.comp                 │
│        │   └─ all shaders use TYPED STRUCTS:                                 │
│        │       struct block_q8_0 { half d; int8_t qs[32]; };                │
│        │       └─ compiler handles sign extension, alignment automatically  │
│        └─ create_pipelines() → VkPipeline cache                              │
│                                                                              │
│  allocate_kv_cache()                                                          │
│    ├─ ggml_new_tensor_4d() → n_layers × n_kv_heads × max_seq × head_dim     │
│    └─ vk_buffer_allocate() → GPU-only memory                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TOKENIZATION                                                      │
│  ─────────────────                                                          │
│  llama_tokenize(model, prompt, add_bos)                                       │
│    ├─ llama_vocab::tokenize()                                                 │
│    ├─ spm_tokenize() or bpe_tokenize() depending on model family             │
│    └─ return std::vector<llama_token> (int32_t ids)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: CONTEXT CREATION                                                    │
│  ──────────────────                                                           │
│  llama_new_context_with_model()                                               │
│    ├─ llama_kv_cache_init()                                                   │
│    │   └─ allocate per-layer KV buffers (ggml_tensor → VkBuffer)             │
│    ├─ llama_batch_init() → input batch structure                              │
│    └─ llama_synchronize() → initial sync                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: INFERENCE — llama_decode(ctx, batch) (Single Submit Per Token)   │
│  ────────────────────────────────────────────────────────────────────────    │
│  llama_decode(ctx, batch)                                                     │
│    ├─ build_compute_graph() → ggml_cgraph                                     │
│    │   └─ one graph for ALL layers, single command buffer                    │
│    │                                                                        │
│    ├─ ggml_backend_graph_compute()                                            │
│    │   └─ ggml_vk_graph_compute()                                             │
│    │       ├─ begin_command_buffer()                                         │
│    │       ├─ FOR each node in graph:                                        │
│    │       │   ├─ case GGML_OP_GET_EMBEDDINGS:                              │
│    │       │   │   └─ vkCmdDispatch(embed_shader, 1, 1, 1)                 │
│    │       │   │       └─ direct F16 lookup (NO dequant, already F16)       │
│    │       │   ├─ case GGML_OP_NORM:                                        │
│    │       │   │   └─ vkCmdDispatch(rms_norm, (dim+255)/256, 1, 1)          │
│    │       │   │       └─ tiled reduction across workgroups                 │
│    │       │   ├─ case GGML_OP_MUL_MAT:                                     │
│    │       │   │   └─ vkCmdDispatch(gemm, M, N, 1)                          │
│    │       │   │       └─ tiled 16×16 GEMM on F16 buffers                   │
│    │       │   │       └─ dotPacked4x8EXT for packed ops                    │
│    │       │   ├─ case GGML_OP_ROPE:                                        │
│    │       │   │   └─ vkCmdDispatch(rope, n_heads, 1, 1)                    │
│    │       │   ├─ case GGML_OP_FLASH_ATTN:                                  │
│    │       │   │   └─ vkCmdDispatch(flash_attn, tiles_x, tiles_y, 1)        │
│    │       │   │       └─ tiled, memory-efficient, NO full Q×K materialization│
│    │       │   │       └─ years of cross-vendor tuning                      │
│    │       │   ├─ case GGML_OP_SILU:                                        │
│    │       │   │   └─ vkCmdDispatch(silu, (hiddenDim+255)/256, 1, 1)       │
│    │       │   ├─ case GGML_OP_ADD:                                         │
│    │       │   │   └─ vkCmdDispatch(add, (dim+255)/256, 1, 1)               │
│    │       │   └─ vkCmdPipelineBarrier() between each node // GPU sync       │
│    │       ├─ end_command_buffer()                                           │
│    │       ├─ vkQueueSubmit() → ONE submit for ALL layers                  │
│    │       └─ vkWaitForFences() → ONE fence wait per token                   │
│    │                                                                        │
│    ├─ extract logits from output tensor                                       │
│    └─ return logits pointer (CPU or mapped GPU memory)                        │
│                                                                              │
│  llama_sample()                                                               │
│    ├─ llama_sample_temperature()                                              │
│    ├─ llama_sample_top_k()                                                  │
│    ├─ llama_sample_top_p()                                                  │
│    ├─ llama_sample_softmax()                                                  │
│    └─ llama_sample_token() → next token ID                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: GENERATION LOOP                                                     │
│  ──────────────────                                                           │
│  WHILE token != EOS:                                                          │
│    llama_decode(ctx, batch_with_last_token)                                   │
│    llama_sample()                                                             │
│    llama_kv_cache_update() → append K,V to cache (no reallocation)          │
│    output_tokens.push_back(next_token)                                        │
│  END                                                                          │
│                                                                              │
│  llama_token_to_piece() → detokenize each ID to string piece                 │
│  concatenate_pieces() → final output string                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Strengths:**
- Pre-dequant eliminates ALL GPU-side quantization bugs.
- Single submit + 1 fence = minimal CPU-GPU round-trip overhead.
- Flash Attention = memory-efficient, fast attention on all vendors.
- Typed structs in GLSL = compiler handles sign extension, alignment.
- All features passed to device creation = driver takes optimized path.

**Key Weakness:**
- ~2× VRAM usage (quantized + F16 copies).

---

## 4. VindexLLM Pipeline

**Philosophy:** Single unified shader handling ALL quant formats via runtime `DType` dispatch. One GLSL function `dequant_weight(addr, idx, DType)` selects format at runtime. Similar to NotLLAMA but with better shader discipline.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MODEL LOAD                                                        │
│  ─────────────                                                              │
│  load_gguf_model(path)                                                        │
│    ├─ parse_gguf_header() → metadata, tensor count, architecture             │
│    ├─ FOR each tensor:                                                        │
│    │   ├─ read_tensor_info() → name, dims, dtype, byte_offset              │
│    │   ├─ dtype = Q8_0 | Q6_K | Q4_0 | F32 | F16                            │
│    │   ├─ upload_raw_bytes_to_gpu() → VkBuffer (quantized, as-is)            │
│    │   │   └─ uses buffer_device_address (BDA) for direct access             │
│    │   └─ NO pre-dequantization (same as NotLLAMA)                            │
│    └─ build_shader_cache()                                                    │
│        └─ ONE unified shader: matvec_all.comp                                  │
│            ├─ handles ALL formats via DType push constant                      │
│            ├─ dequant_weight(addr, idx, DType) → float                         │
│            │   ├─ DType=2 (Q8_0): d × int8_t(q)                              │
│            │   ├─ DType=3 (Q4_0): d × (nibble - 8)                            │
│            │   └─ DType=15 (Q6_K): d × sc × (val - 32)  // d-last            │
│            └─ compile to SPIR-V once, dispatch with DType parameter          │
│                                                                              │
│  allocate_kv_cache() → VkBuffer per layer, GPU-only                          │
│  allocate_staging_buffers() → for CPU readback of logits                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TOKENIZATION                                                      │
│  ─────────────────                                                          │
│  bpe_tokenize(prompt) or sentencepiece_tokenize(prompt)                     │
│  → vector<uint32_t> token_ids                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: INFERENCE — process_token()                                         │
│  ──────────────────────────────────                                           │
│  FOR each token in sequence:                                                  │
│    run_inference_step(token_id, position)                                      │
│      ├─ dispatch_embed()                                                        │
│      │   ├─ select shader by embedding format (embed_q8_0 / q6_k / f32)         │
│      │   ├─ local_size = 256                                                  │
│      │   └─ write hidden[dim] to output buffer                                │
│      │                                                                        │
│      ├─ FOR layer = 0 to n_layers-1:                                          │
│      │   ├─ dispatch_matvec_all(                                             │
│      │   │   addrWeight = w_norm[layer],                                       │
│      │   │   addrInput = hidden,                                              │
│      │   │   addrOutput = normed,                                             │
│      │   │   DType = dtype_of(w_norm),                                        │
│      │   │   transB = 0 or 1                                                  │
│      │   │ ) // RMS norm weight matvec (treated as 1×dim × dim×1)            │
│      │   │   ├─ local_size = 64                                               │
│      │   │   ├─ dispatch = (outDim + 63) / 64 workgroups                     │
│      │   │   ├─ cooperative load: vecShared[256] from input                   │
│      │   │   ├─ for k in 0..tileSize-1:                                      │
│      │   │   │   ├─ weightIdx = transB ? n×inDim+globalK : globalK×outDim+n  │
│      │   │   │   ├─ w = dequant_weight(addrWeight, weightIdx, DType)        │
│      │   │   │   └─ acc += vecShared[k] × w                                   │
│      │   │   ├─ barrier() per tile                                            │
│      │   │   └─ write output.data[n] = acc                                    │
│      │   │       └─ includes bounds check: if (n >= outDim) return;          │
│      │   │                                                                    │
│      │   ├─ dispatch_matvec_all() for Q projection                           │
│      │   ├─ dispatch_matvec_all() for K projection                           │
│      │   ├─ dispatch_matvec_all() for V projection                           │
│      │   │   └─ all use SAME matvec_all shader, different DType/weights      │
│      │   │                                                                    │
│      │   ├─ dispatch_rope() → apply rotary to Q, K                         │
│      │   │   └─ local_size = 32, position-aware theta                        │
│      │   │                                                                    │
│      │   ├─ dispatch_kv_cache_write()                                         │
│      │   │   └─ K, V → kv_cache[layer][head][position]                        │
│      │   │                                                                    │
│      │   ├─ dispatch_attention() // per-head or tiled                         │
│      │   │   ├─ naive: one dispatch per head (like NotLLAMA)                 │
│      │   │   OR                                                               │
│      │   │   ├─ tiled: single dispatch, shared mem for Q×K scores            │
│      │   │   └─ softmax + weighted V sum → attnOut                            │
│      │   │                                                                    │
│      │   ├─ dispatch_matvec_all() for attention output projection             │
│      │   ├─ dispatch_add() → hidden += attnOut                              │
│      │   │                                                                    │
│      │   ├─ dispatch_matvec_all() for FFN gate                              │
│      │   ├─ dispatch_matvec_all() for FFN up                                │
│      │   ├─ dispatch_silu_mul() → gate × SiLU(gate) × up                    │
│      │   ├─ dispatch_matvec_all() for FFN down                                │
│      │   └─ dispatch_add() → hidden += ffnOut                               │
│      │                                                                        │
│      ├─ dispatch_rms_norm() → final norm                                      │
│      ├─ dispatch_matvec_all() → lm_head gemm                                │
│      │   └─ logits[vocabSize]                                                 │
│      └─ read_back_logits() → CPU                                              │
│                                                                              │
│    sample_token(logits, temperature, top_k, top_p)                            │
│    → next_token_id                                                            │
│  ENDFOR                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: GENERATION                                                          │
│  ─────────────────────                                                        │
│  Same loop as above, auto-regressive.                                         │
│  Key difference: unified matvec_all shader means fewer shader variants,       │
│  but same dispatch count and sync pattern as NotLLAMA.                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Difference from NotLLAMA:**
- One shader handles all quant formats (cleaner code, fewer variants).
- Explicit `int8_t()` sign extension in `dequant_weight()`.
- Explicit bounds check in shader.
- Same sync overhead as NotLLAMA (still ~37 fence waits if not batched).

---

## 5. vLLM Pipeline

**Philosophy:** Production-grade serving engine. Continuous batching, PagedAttention, paged KV cache, multi-GPU tensor parallelism. Optimized for throughput, not single-request latency.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: MODEL LOAD & ENGINE INITIALIZATION (Host CPU)                       │
│  ───────────────────────────────────────────────────────                      │
│  LLMEngine.from_engine_args(engine_args)                                      │
│    ├─ load_model_weights()                                                    │
│    │   ├─ supports: FP16, BF16, GPTQ, AWQ, GGUF (via llama-cpp-python)       │
│    │   ├─ if GGUF: uses llama.cpp loader OR custom GGUF loader               │
│    │   ├─ weights → GPU VRAM (CUDA/ROCm device tensors)                     │
│    │   └─ quantization: GPTQ (4-bit), AWQ (4-bit), GGUF (various)            │
│    │                                                                        │
│    ├─ init_cache_engine()                                                     │
│    │   ├─ allocate_gpu_cache()                                                │
│    │   │   ├─ block_size = 16 (tokens per KV block)                         │
│    │   │   ├─ num_gpu_blocks = gpu_memory / (block_size × head_dim × layers × 2 × dtype_size)│
│    │   │   └─ KV cache as BLOCKS, not continuous tensor                       │
│    │   │       └─ torch.Tensor shape: [num_blocks, block_size, n_heads, head_dim]│
│    │   └─ allocate_cpu_cache() (for CPU offloading)                           │
│    │                                                                        │
│    ├─ init_worker() → Ray worker per GPU (multi-GPU)                         │
│    │   └─ each worker loads model shards                                      │
│    │                                                                        │
│    └─ _init_tokenizer() → HuggingFace AutoTokenizer or SentencePiece          │
│                                                                              │
│  CUDA/ROCm Kernel Compilation (JIT)                                           │
│    ├─ if using custom CUDA kernels: compile at first run                      │
│    ├─ paged_attention_v1/v2 kernels (CUDA only, Triton for ROCm)            │
│    ├─ activation kernels (silu_and_mul, gelu_and_mul)                        │
│    ├─ quantization kernels (marlin, awq, gptq)                                │
│    └─ custom all-reduce for tensor parallelism                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: REQUEST SCHEDULING & BATCHING                                       │
│  ────────────────────────────────────────                                     │
│  add_request(prompt, sampling_params)                                       │
│    └─ Request placed in waiting queue                                       │
│                                                                              │
│  step() // called repeatedly by API server                                   │
│    ├─ Scheduler.schedule()                                                    │
│    │   ├─ _schedule_prefills() → new requests to prefill                     │
│    │   ├─ _schedule_decodes() → ongoing generation requests                  │
│    │   ├─ _schedule_swapped() → requests swapped to CPU back to GPU          │
│    │   └─ output: SchedulerOutputs (batch of sequences to run)               │
│    │                                                                        │
│    └─ _process_model_outputs()                                                │
│        ├─ execute_model(batch)                                                │
│        └─ update sequence states                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: TOKENIZATION (Per Request)                                        │
│  ───────────────────────────────────                                          │
│  tokenizer.encode(prompt)                                                     │
│    ├─ HuggingFace: AutoTokenizer.encode()                                     │
│    ├─ SentencePiece: sp_model.encode()                                        │
│    └─ return List[int] token_ids                                              │
│                                                                              │
│  allocate_blocks_for_sequence()                                               │
│    ├─ BlockAllocator.allocate() → assign physical GPU blocks                  │
│    │   └─ logical block 0 → physical block 47, 12, 89...                    │
│    └─ block_table[seq_id] = [physical_block_ids]                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: MODEL EXECUTION — execute_model() (CUDA/ROCm)                      │
│  ───────────────────────────────────────────────────────                      │
│  model_runner.execute_model(seq_group_metadata_list)                          │
│    ├─ prepare_input_tensors()                                                 │
│    │   ├─ input_ids: torch.Tensor [batch_size, seq_len]                     │
│    │   ├─ positions: torch.Tensor [batch_size, seq_len] (absolute positions) │
│    │   ├─ kv_cache: List[torch.Tensor] per layer                             │
│    │   ├─ attn_metadata:                                                      │
│    │   │   ├─ block_tables: torch.IntTensor [batch_size, max_blocks]        │
│    │   │   ├─ context_lens: torch.IntTensor [batch_size]                    │
│    │   │   ├─ slot_mapping: torch.IntTensor [batch_size, seq_len]          │
│    │   │   └─ num_prefill_tokens, num_decode_tokens                          │
│    │   └─ _prepare_prompt() or _prepare_decode()                             │
│    │                                                                        │
│    ├─ model.forward(input_ids, positions, kv_cache, attn_metadata)            │
│    │   └─ LlamaModel.forward()                                                │
│    │       ├─ embed_tokens(input_ids) → hidden_states [B, S, hidden_size]    │
│    │       │   └─ nn.Embedding lookup (CUDA kernel)                          │
│    │       │                                                                │
│    │       ├─ FOR layer in layers:                                            │
│    │       │   ├─ LlamaDecoderLayer.forward()                                │
│    │       │   │   ├─ input_layernorm(hidden) → normed                       │
│    │       │   │   │   └─ RMSNorm CUDA kernel                                │
│    │       │   │   │                                                        │
│    │       │   │   ├─ self_attn(normed, positions, kv_cache, attn_metadata) │
│    │       │   │   │   ├─ q_proj, k_proj, v_proj → nn.Linear (GEMM)         │
│    │       │   │   │   │   └─ torch.matmul() → cuBLAS/cuBLASLt/rocBLAS       │
│    │       │   │   │   │       └─ if quantized: Marlin/GPTQ/AWQ GEMM kernel │
│    │       │   │   │   │                                                    │
│    │       │   │   │   ├─ apply_rotary_pos_emb(q, k, positions)             │
│    │       │   │   │   │   └─ fused rope CUDA kernel                         │
│    │       │   │   │   │                                                    │
│    │       │   │   │   ├─ PagedAttention.write_to_cache()                    │
│    │       │   │   │   │   ├─ reshape_and_cache_kernel()                   │
│    │       │   │   │   │   │   └─ scatter K,V into block_table blocks        │
│    │       │   │   │   │   └─ slot_mapping determines exact KV cache slots  │
│    │       │   │   │   │                                                    │
│    │       │   │   │   ├─ PagedAttention.forward()                            │
│    │       │   │   │   │   ├─ IF prefill (prompt processing):                │
│    │       │   │   │   │   │   └─ flash_attn_varlen_func()                   │
│    │       │   │   │   │   │       └─ FlashAttention-2 (Triton/CUDA)         │
│    │       │   │   │   │   │       └─ computes attention over full sequence │
│    │       │   │   │   │   ├─ IF decode (single token):                      │
│    │       │   │   │   │   │   └─ paged_attention_v1/v2()                    │
│    │       │   │   │   │   │       ├─ kernel: paged_attention_v2_kernel()   │
│    │       │   │   │   │   │       ├─ each thread handles one query token  │
│    │       │   │   │   │   │       ├─ loads K from block_table blocks       │
│    │       │   │   │   │   │       ├─ computes Q×K dot products              │
│    │       │   │   │   │   │       ├─ online softmax (numerically stable)   │
│    │       │   │   │   │   │       ├─ loads V from blocks, accumulates      │
│    │       │   │   │   │   │       └→ attn_output [B, num_heads, head_dim]  │
│    │       │   │   │   │                                                    │
│    │       │   │   │   └─ o_proj(attn_output) → nn.Linear GEMM               │
│    │       │   │   │                                                        │
│    │       │   │   ├─ residual = hidden + attn_output                        │
│    │       │   │   │                                                        │
│    │       │   │   ├─ post_attention_layernorm(residual) → normed            │
│    │       │   │   │                                                        │
│    │       │   │   ├─ mlp(normed)                                             │
│    │       │   │   │   ├─ gate_proj(normed) → gate [B, S, intermediate_dim]  │
│    │       │   │   │   ├─ up_proj(normed) → up [B, S, intermediate_dim]      │
│    │       │   │   │   ├─ silu_and_mul(gate, up) → activated                  │
│    │       │   │   │   │   └─ fused SiLU + elementwise_mul CUDA kernel      │
│    │       │   │   │   └─ down_proj(activated) → ffn_output [B, S, hidden]   │
│    │       │   │   │       └─ nn.Linear GEMM (quantized if applicable)       │
│    │       │   │   │                                                        │
│    │       │   │   └─ hidden = residual + ffn_output                          │
│    │       │   │                                                            │
│    │       ├─ norm(hidden) → final normed                                    │
│    │       │   └─ RMSNorm CUDA kernel                                        │
│    │       │                                                                │
│    │       └─ lm_head(normed) → logits [B, S, vocab_size]                    │
│    │           └─ nn.Linear GEMM (can be quantized)                          │
│    │                                                                        │
│    ├─ _sample_from_logits()                                                   │
│    │   ├─ apply_logits_processors() → penalties, frequency, presence        │
│    │   ├─ _apply_temperature()                                              │
│    │   ├─ _apply_top_k_top_p()                                                │
│    │   │   └─ torch.topk() + softmax + cumulative prob filter                 │
│    │   └─ multinomial_sample() → next_token_ids [B]                            │
│    │                                                                        │
│    └─ update_sequences(next_token_ids)                                        │
│        ├─ append token to each sequence                                       │
│        ├─ check_stop_conditions() → EOS, max_length, stop strings            │
│        ├─ maybe_swap_blocks_to_cpu() if GPU memory pressure                  │
│        └─ free_finished_sequences() → release blocks                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: OUTPUT STREAMING                                                    │
│  ─────────────────────                                                        │
│  tokenizer.decode(token_ids) → output string                                  │
│  stream_response_to_client() → SSE/HTTP chunk                                  │
│                                                                              │
│  CONTINUOUS BATCHING:                                                         │
│  ├─ New requests can join between steps                                       │
│  ├─ Finished requests are removed                                             │
│  ├─ KV blocks are reused via block allocator                                │
│  └─ All running requests share one model.forward() call                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Consolidated API Comparison Table

### 6.1 Model Load & Setup

| Stage | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|-------|----------|-----------|-----------|------|
| **File Format** | GGUF | GGUF | GGUF | GGUF, HF Safetensors, PyTorch bin |
| **Parser** | `read_gguf_file()` → custom | `gguf_init_from_file()` → `ggml` parser | `parse_gguf_header()` → custom | `load_model_weights()` → HF/llama.cpp loader |
| **Weight Upload** | `vkCreateBuffer()` + `vkAllocateMemory()` + `memcpy` to mapped memory | `vkMapMemory()` + `memcpy` + `vkUnmapMemory()` | `upload_raw_bytes_to_gpu()` → VkBuffer | `torch.Tensor` → `.cuda()` / `.to('cuda')` |
| **Dequant Strategy** | **On-the-fly in shaders** (per token) | **Pre-dequant at load** (CPU → F16 GPU) | **On-the-fly in shaders** (per token) | **Pre-dequant at load** (GPU tensors, or Marlin/GPTQ kernels) |
| **VRAM Overhead** | ~1× weight size (quantized) | ~2× weight size (quantized + F16) | ~1× weight size (quantized) | ~1× (quantized) to ~2× (FP16/BF16) |
| **Shader Compilation** | `compile_glsl_to_spv()` per shader variant | `ggml_vk_init()` compiles all shaders | Single `matvec_all.comp` + variants | JIT compilation of Triton/CUDA kernels at runtime |
| **Quant Kernel** | Manual byte extraction in GLSL | Typed structs (`block_q8_0 { half d; int8_t qs[32]; }`) | `dequant_weight(addr, idx, DType)` | Marlin/GPTQ/AWQ CUDA kernels |

### 6.2 Device & Backend Setup

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Backend** | Vulkan Compute | Vulkan Compute | Vulkan Compute | CUDA / ROCm |
| **Physical Device** | `pick_physical_device()` → compute-only preferred | `ggml_vk_init()` → prefers ACE/Family 1 | `pick_physical_device()` → compute | `torch.cuda.get_device_properties()` |
| **Feature Chain** | Query ALL → **ZERO graphics features** | Query ALL → **PASS ALL features** | Query ALL → compute-relevant | N/A (PyTorch abstracts this) |
| **Queue Family** | Compute-only preferred | Family 1 (ACE), `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` → Family 0 | Compute-only | N/A |
| **Memory Model** | BDA + scalar buffers | BDA + typed structs (`block_q8_0`) | BDA + scalar buffers | PyTorch CUDA allocator |
| **Multi-GPU** | No | No | No | Yes (Ray workers, tensor parallelism) |

### 6.3 Tokenization

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Tokenizer** | Custom BPE/SPM from GGUF metadata | `llama_vocab::tokenize()` → SPM/BPE | Custom BPE/SPM | HuggingFace `AutoTokenizer` or `SentencePiece` |
| **API** | Host CPU, C++ string ops | Host CPU, C++ string ops | Host CPU, C++ string ops | Python `tokenizer.encode()` |
| **Special Tokens** | Read from GGUF KV | Read from GGUF KV | Read from GGUF KV | From tokenizer config JSON |

### 6.4 Context / KV Cache

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Cache Init** | `allocate_kv_cache()` → per-layer VkBuffer | `llama_kv_cache_init()` → ggml tensors → VkBuffer | `allocate_kv_cache()` → VkBuffer per layer | `init_cache_engine()` → BLOCK-based torch tensors |
| **Layout** | `[layer][head][seq][head_dim]` continuous | `[layer][head][seq][head_dim]` continuous | `[layer][head][seq][head_dim]` continuous | `[num_blocks][block_size][head][head_dim]` paged |
| **Block Size** | N/A (continuous) | N/A (continuous) | N/A (continuous) | 16 tokens per block |
| **Allocator** | None (static allocation) | None (static allocation) | None (static allocation) | `BlockAllocator` (dynamic, reuse, swap to CPU) |
| **Growth** | Append at `current_position` | Append at `current_position` | Append at `current_position` | Append via `slot_mapping` into physical blocks |

### 6.5 Embed Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `embed_q8_0.comp`, `embed_q6_k.comp`, `embed.comp` | `get_embeddings` kernel (F16 lookup) | `embed_q8_0`, `embed_q6_k`, `embed` | `nn.Embedding` (CUDA kernel) |
| **Dispatch** | 1 dispatch, 256 threads | 1 dispatch, part of single graph | 1 dispatch, 256 threads | Fused in `model.forward()` |
| **Dequant** | On-the-fly in shader | None (already F16) | On-the-fly in shader | None (already F16/BF16) |
| **Output** | `hidden[dim]` | `hidden_states[B, S, hidden_size]` | `hidden[dim]` | `hidden_states[B, S, hidden_size]` |

### 6.6 RMS Norm Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `rms_norm.comp` → local_size=256, wg=1 | `rms_norm.comp` → `(dim+255)/256` wgs | `rms_norm.comp` | `RMSNorm` CUDA kernel |
| **Reduction** | Strided sumSq (⚠️ possible Wave32 issue) | Tiled reduction across workgroups | Strided sumSq | `torch.rms_norm()` or custom CUDA |
| **Epsilon** | Should be 1e-6 | 1e-6 (default) | Should be 1e-6 | 1e-6 (default) |
| **Broadcast** | Thread 0 computes invRms, all threads apply | Workgroup-local then broadcast | Thread 0 computes invRms | Fused in kernel |

### 6.7 QKV Projection Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `matvec_q8_0`, `matvec_q6_k`, `matvec_q4_0`, `gemm_fallback` | `gemm.comp` → tiled 16×16 F16 GEMM | `matvec_all.comp` → unified, DType dispatch | `nn.Linear` → `torch.matmul()` → cuBLAS/rocBLAS |
| **Workgroup** | 64 threads, `(outDim+63)/64` dispatch | 16×16 = 256 threads, tiled | 64 threads, `(outDim+63)/64` dispatch | cuBLAS auto-tuned tile sizes |
| **Quant Read** | Manual byte extraction: blockIdx → scale → sign_extend | Typed structs: `block_q8_0 { half d; int8_t qs[32]; }` | `dequant_weight(addr, idx, DType)` | Marlin/GPTQ/AWQ kernels for quantized |
| **Shared Memory** | `inputTile[256]` cooperative load | `tileA[16][16]`, `tileB[16][16]` for GEMM | `vecShared[256]` cooperative load | Managed by cuBLAS/rocBLAS |
| **Transpose** | `transB` push constant | Handled by ggml tensor layout | `transB` push constant | PyTorch handles internally |
| **Bounds Check** | ⚠️ MISSING | Handled by ggml dispatch math | Explicit `if (n >= outDim) return;` | PyTorch handles internally |

### 6.8 RoPE Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `rope.comp` → local_size=32 | `rope.comp` → `n_heads` workgroups | `rope.comp` → local_size=32 | `apply_rotary_pos_emb()` fused CUDA kernel |
| **Position** | seq position passed as uniform | seq position from ggml tensor | seq position passed as uniform | `positions` tensor |
| **Theta Base** | 10000.0 (default) | 10000.0 (default) | 10000.0 (default) | 10000.0 (configurable) |

### 6.9 KV Cache Write Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `kv_cache_write.comp` | Fused in Flash Attention or explicit write | `kv_cache_write.comp` | `reshape_and_cache_kernel()` (CUDA) |
| **Indexing** | `layerOffset + headOffset + seqOffset` | ggml tensor indexing | `layerOffset + headOffset + seqOffset` | `slot_mapping` + `block_tables` |
| **Format** | Direct write to continuous buffer | Direct write to continuous buffer | Direct write to continuous buffer | Scatter into paged blocks |

### 6.10 Attention Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Algorithm** | Naive O(N²) per-head dispatch | **Flash Attention** (tiled, memory-efficient) | Naive or tiled | **PagedAttention v2** (decode) / FlashAttention-2 (prefill) |
| **Shader/Kernel** | `attention.comp` → 1 dispatch per head | `flash_attn.comp` → tiled dispatch | `attention.comp` | `paged_attention_v2_kernel()` (CUDA) |
| **Workgroup** | 32 threads (one warp) per head | Tile-based (e.g., 128×128 threads) | 32 or 64 threads | One thread per query token |
| **KV Read** | Direct from continuous cache | Direct from continuous cache | Direct from continuous cache | From `block_tables` → physical blocks |
| **Softmax** | Online over seqLen | Tiled, numerically stable | Online over seqLen | Online, numerically stable |
| **Materialization** | Full scores[seqLen] per head | NO full Q×K materialization | Full scores[seqLen] per head | NO full Q×K materialization |
| **Dispatches** | n_heads per layer | 1 per layer | n_heads or 1 per layer | 1 per batch (fused) |

### 6.11 FFN Stage (Gate / Up / Down)

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Gate/Up** | `matvec_*` ×2 + `silu_mul.comp` | `gemm.comp` ×2 + `silu.comp` | `matvec_all` ×2 + `silu_mul` | `nn.Linear` ×2 + `silu_and_mul` CUDA kernel |
| **Down** | `matvec_*` | `gemm.comp` | `matvec_all` | `nn.Linear` |
| **Fusion** | Separate dispatches (6+ barriers) | Single graph, pipeline barriers | Separate dispatches | Fused in PyTorch autograd |
| **Activation** | `SiLU(x) * up` in shader | `SiLU(x) * up` in shader | `SiLU(x) * up` in shader | `silu_and_mul` fused CUDA kernel |

### 6.12 Residual Add Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `add.comp` → local_size=256 | `add.comp` → `(dim+255)/256` | `add.comp` | Fused CUDA kernel or `torch.add()` |
| **Operation** | `hidden += attnOut` (element-wise) | `hidden += attnOut` | `hidden += attnOut` | `hidden = residual + ffn_output` |

### 6.13 LM Head Stage

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Shader/Kernel** | `gemm.comp` (fallback) or matvec | `gemm.comp` | `matvec_all` | `nn.Linear` → `torch.matmul()` |
| **Output** | `logits[vocabSize]` | `logits[vocabSize]` | `logits[vocabSize]` | `logits[B, S, vocab_size]` |

### 6.14 Submit / Sync Strategy

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Command Buffer** | Per-layer `beginBatch()`/`endBatch()` + `syncAllThrottled()` | **Single command buffer, ONE submit per token** | Per-layer batches with sync | **Single `model.forward()` call** (PyTorch handles graph) |
| **Barriers** | `barrier()` in GLSL (workgroup only) + 37 fence waits | `vkCmdPipelineBarrier()` between nodes + 1 fence | `barrier()` + fence waits | `torch.cuda.synchronize()` implicit |
| **Sync Points** | ~37 per token (fence waits) | 1 per token | ~37 per token | 0 explicit (PyTorch async) |
| **Fence Waits** | `vkWaitForFences()` per layer | `vkWaitForFences()` once per token | `vkWaitForFences()` per layer | Implicit in PyTorch |

### 6.15 Sampling

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Method** | `sampleArgmax()` or `temperature_sample()` | `llama_sample_*()` → temp, top_k, top_p, repetition | `sample_token()` | `multinomial_sample()` + logits processors |
| **Temperature** | Yes | Yes | Yes | Yes |
| **Top-K** | Yes | Yes | Yes | Yes |
| **Top-P** | Yes | Yes | Yes | Yes |
| **Repetition Penalty** | No | Yes | No | Yes |
| **API** | Host CPU, C++ | Host CPU, C++ | Host CPU, C++ | Python (PyTorch) |

### 6.16 Generation Loop

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Loop** | `forwardPartial()` + `sample()` per token | `llama_decode()` + `llama_sample()` per token | `process_token()` + `sample()` per token | `step()` → `execute_model()` + `_sample_from_logits()` |
| **Batching** | Single sequence only | Single sequence only | Single sequence only | **Continuous batching** (multiple sequences) |
| **KV Management** | Grow continuous buffer | Grow continuous buffer | Grow continuous buffer | **Paged block allocator** (reuse, swap to CPU) |
| **Detokenization** | `detokenize()` → C++ string concat | `llama_token_to_piece()` → piece concat | `detokenize()` | `tokenizer.decode()` |

### 6.17 Key Libraries & Dependencies

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Core** | Vulkan SDK, GLSL, SPIR-V | Vulkan SDK, GGML, GLSL, SPIR-V | Vulkan SDK, GLSL, SPIR-V | PyTorch, CUDA/ROCm, Triton, cuBLAS |
| **Math** | Manual GLSL | GGML tensor ops | Manual GLSL | cuBLAS, cuBLASLt, rocBLAS |
| **Parallelism** | None | None | None | Ray, tensor parallelism, pipeline parallelism |
| **Serving** | None (embedded) | None (embedded) | None (embedded) | FastAPI, OpenAI-compatible API |
| **Quantization** | GGUF (Q4_0, Q8_0, Q6_K) | GGUF (all formats) | GGUF (Q4_0, Q8_0, Q6_K) | GPTQ, AWQ, Marlin, GGUF, FP8 |

---

## 7. Key Architectural Differences

| Aspect | NotLLAMA | llama.cpp | VindexLLM | vLLM |
|--------|----------|-----------|-----------|------|
| **Dequant When** | Every token, GPU shader | Once at load, CPU | Every token, GPU shader | Once at load, GPU (or quantized GEMM) |
| **Submit Granularity** | Per-dispatch barriers, 37 syncs/token | Single submit, 1 sync/token | Per-dispatch barriers, ~37 syncs/token | Single PyTorch forward, async |
| **Attention** | Naive per-head | Flash Attention | Naive or tiled | PagedAttention + FlashAttention |
| **Cache** | Continuous per-layer | Continuous per-layer | Continuous per-layer | Paged blocks (16 tokens) |
| **Batching** | None | None | None | Continuous (production) |
| **Quant Handling** | Raw byte buffers + manual extract | Typed structs (compiler-friendly) | Unified function + DType dispatch | Marlin/GPTQ/AWQ kernels |
| **Platform** | Windows, Vulkan | Cross-platform, Vulkan | Vulkan | Linux, CUDA/ROCm |
| **Speed Bottleneck** | 37 fence waits, per-head dispatches | Pre-dequant VRAM, but fast inference | Same sync overhead as NotLLAMA | Best throughput via continuous batching |
| **Primary Use Case** | Learning / custom Vulkan engine | Local inference, all platforms | Learning / unified shader design | Production serving, high throughput |

---

## 8. NotLLAMA Bug Map (Cross-Referenced)

Based on the pipeline analysis above, here is where each known bug lives in your code:

| Bug | Location in Pipeline | Root Cause | Fix |
|-----|---------------------|------------|-----|
| **NaN in logits** | `rms_norm.comp` (Layer 0) | 256-thread reduction on Wave32 RDNA → incomplete sumSq → division by zero | Two-stage shared memory reduction (256→128→64→32→16→8→4→2→1). Do NOT use `subgroupAdd`. |
| **MaxAE=10–24 (fallback)** | `gemm.comp` or dispatch math | Wrong transpose assumption, wrong stride, or missing bounds check | Verify `transB` matches weight layout. Add `if (n >= outDim) return;`. |
| **MaxAE=10–24 (quantized)** | `matvec_q8_0.comp`, `matvec_q6_k.comp` | Missing `int8_t()` sign extension, wrong `d-last` offset, or wrong block indexing | Add `int(int8_t(v))`. Verify Q6_K `d` at offset 208. |
| **GPU Hang / TDR** | `syncAllThrottled()` after every layer | 37 fence waits × ~1-5ms = >2s GPU stall on Windows | Batch all layer dispatches into ONE command buffer. ONE fence wait per token. |
| **Device Lost** | Any out-of-bounds thread | Dispatch `(outDim+63)/64` × 64 threads → last thread writes past buffer | Add `if (global_id >= outDim) return;` to ALL shaders. |
| **Embed correct but downstream wrong** | Verified: embed is bit-exact | Corruption starts at RMS norm or GEMM fallback | Fix RMS norm first. Then verify GEMM fallback with known-good weights. |

---

*Document generated 2026-06-29. Covers NotLLAMA, llama.cpp, VindexLLM, and vLLM function-level pipelines.*
