# NotLLAMA Feature Roadmap

## Overview
This document outlines all planned features for NotLLAMA. **Current priority: fix existing bugs before writing new feature code.** The sections marked **[NOW]** are what we will approach immediately after bugs are fixed. The rest are future work.

---

## 1. TurboQuant — Ultra-Fast Quantization Pipeline

### 1.1 Purpose
A family of GPU-native quantization formats that dequantize faster than GGUF K-quants while maintaining competitive accuracy. Designed for RDNA4 subgroup-optimized dequantization.

### 1.2 What Needs to Be Done

#### A. TurboQuant Format Specification (JSON Schema)
- Define `TurboQuantConfig` schema with fields:
  - `version`: string (e.g., "1.0")
  - `block_size`: int (64, 128, 256)
  - `bits_per_weight`: float (3.0, 4.0, 4.5, 5.0, 6.0)
  - `scale_bits`: int (8, 16)
  - `zero_point`: bool
  - `group_size`: int (for finer-grained scales)
  - `activation_aware`: bool (use activation outliers)
  - `outlier_threshold`: float
  - `outlier_bits`: int (8 for outliers)
- Document layout: `weights[] | scales[] | zero_points[] | outliers[] | outlier_indices[]`
- Document alignment requirements (128-bit aligned for RDNA4)

#### B. Weight Converter Extension
- Extend `tools/weight_converter.py` to emit TurboQuant from F16/F32:
  - `convert_to_tq4()` — 4-bit, 128-element blocks, fp16 scales
  - `convert_to_tq3()` — 3-bit, 128-element blocks, for memory-constrained models
  - `convert_to_tq6()` — 6-bit, 64-element blocks, near-F16 quality
- Calibration: use a small representative dataset to find optimal scales
- Outlier detection: identify top-N outlier channels and store at fp8/fp16

#### C. GLSL Dequantization Shaders
- `dequant_turbo.comp` — generic TurboQuant dequantizer
  - Subgroup-optimized: each subgroup processes one block
  - Use `subgroupBroadcast` for scale broadcast
  - Support `block_size` 64/128/256 as push constant
- `gemm_turbo.comp` — fused dequant+GEMM for TurboQuant weights
  - On-the-fly dequantization during matmul (no separate dequant pass)
  - Each thread loads quantized weights, dequantizes to registers, accumulates
  - Tile size: 32×32 or 64×64 workgroups

#### D. Host-Side Integration
- Add `TQ4_128`, `TQ3_128`, `TQ6_64` to `QuantFormat` enum
- Update `initWeightBuffer()` to recognize TurboQuant formats
- Update `dequantWeight()` dispatch logic

#### E. Validation
- Accuracy test: TurboQuant vs F16 on perplexity (WikiText-2)
- Speed test: dequant time vs Q4_K, Q6_K, Q8_0
- Compression ratio: bits per weight vs quality

### 1.3 Priority: **[NOW]** — Start with JSON schema + shader stubs

---

## 2. DOX Framework Enhancements

### 2.1 Purpose
Extend the AGENTS.md hierarchy to cover the new feature domains (TurboQuant, context compression, etc.) and add automation.

### 2.2 What Needs to Be Done

#### A. Child AGENTS.md for New Domains
- `src/kernels/AGENTS.md` — shader conventions, GLSL style, push constant rules
- `tools/AGENTS.md` — Python tooling standards, converter scripts
- `include/AGENTS.md` — header design philosophy, no inline implementation rule

#### B. Automated DOX Validation
- Python script `tools/dox_lint.py`:
  - Verify every `.cpp`/`.hpp` has a matching AGENTS.md within its parent chain
  - Verify AGENTS.md files have required sections (Purpose, Ownership, etc.)
  - Flag files modified more recently than their nearest AGENTS.md
- Run in CI / pre-commit hook

#### C. DOX Feature: Work Template Integration
- Add `WorkTemplate` serialization to AGENTS.md
- Document how keep-alive triggers map to DOX child docs
- Add verification section for each new header domain

### 2.3 Priority: **[NOW]** — Create child AGENTS.md stubs for kernels/ and tools/

---

## 3. Claude.md Integration

### 3.1 Purpose
Make the user's `~/.claude/CLAUDE.md` preferences available as runtime-configurable options in NotLLAMA, not just static agent instructions.

### 3.2 What Needs to Be Done

#### A. JSON Config Schema Extension
- Add `claude` section to runtime config:
  ```json
  {
    "claude": {
      "graphify_auto_update": true,
      "graphify_query_before_read": true,
      "knowledge_graph_path": "graphify-out/graph.json",
      "preferred_exploration_mode": "dfs",
      "token_budget": 1500,
      "query_budget_cap": true
    }
  }
  ```

#### B. Runtime Graphify Client
- `GraphifyClient` class in `rdna4_graphify.hpp`:
  - Wraps `graphify query` CLI calls
  - Caches query results in memory (LRU)
  - Auto-updates graph on model load / file change
- Integration points:
  - `InferenceEngine` queries graph before loading unknown quant formats
  - `ContextManager` queries graph for context compaction strategy suggestions

#### C. Context-Aware Error Messages
- When a crash occurs, `GraphifyClient` queries the graph for related nodes
- Embeds relevant `source_location` from graph into crash handler output

### 3.3 Priority: **[NOW]** — JSON schema + `GraphifyClient` header stub

---

## 4. Context Compression

### 4.1 Purpose
Reduce context length when approaching the model's context limit, with configurable strategy and threshold.

### 4.2 What Needs to Be Done

#### A. JSON Config Schema
```json
{
  "context_compression": {
    "enabled": true,
    "trigger_threshold_percent": 85,
    "target_percent": 50,
    "strategy": "importance",
    "strategies": {
      "sliding_window": { "window_size": 1024, "stride": 512 },
      "half_slide": { "preserve_recent": 512 },
      "fifo": { "drop_percent": 50 },
      "importance": { "scorer": "attention_entropy", "keep_top_percent": 50 },
      "summary": { "summarizer_model": "internal", "compress_ratio": 0.3 }
    }
  }
}
```

#### B. Compression Strategies (already stubbed in `rdna4_context_manager.hpp`)
1. **SLIDING_WINDOW** — keep only last N tokens, drop the rest
2. **HALF_SLIDE** — drop first half, keep recent half
3. **FIFO** — oldest tokens dropped first
4. **IMPORTANCE** — score each token position by attention entropy or gradient magnitude, keep highest-scoring
5. **SUMMARY** — run a lightweight summarization pass on dropped text, replace with summary tokens

#### C. GPU Implementation
- `compress_context.comp` — kernel that rewrites KV cache based on `CompressionPlan`
  - Input: `keep_mask[]` (bool per token position)
  - Output: compacted K/V tensors
  - Use subgroup shuffle to compact in parallel
- Host: `ContextManager::compact()` generates `keep_mask` then dispatches kernel

#### D. Integration
- `InferenceEngine::forwardPartial()` checks `seqPos / maxContext` against threshold
- If exceeded, calls `contextManager.compact()` before proceeding

### 4.3 Priority: **[NOW]** — JSON schema + `compress_context.comp` stub

---

## 5. Memory Compression (KV Cache Compression)

### 5.1 Purpose
Compress the KV cache itself to fit longer contexts in the same VRAM, or batch more sequences.

### 5.2 What Needs to Be Done

#### A. JSON Config Schema
```json
{
  "kv_cache_compression": {
    "enabled": true,
    "method": "quantize",
    "methods": {
      "quantize": { "format": "Q4_0", "compress_every_n_layers": 1 },
      "sparsify": { "threshold": 0.01, "prune_heads": [] },
      "low_rank": { "rank": 64, "update_every_n_tokens": 512 },
      "hierarchical": { "recent_n": 1024, "format": "F16", "older_format": "Q4_0" }
    }
  }
}
```

#### B. Compression Methods
1. **Quantize** — store KV cache in Q4_0/Q5_0/Q8_0 instead of F16
   - `kv_cache_quantize.comp` — per-head quantization with per-channel scales
   - `kv_cache_dequant.comp` — on-the-fly dequant in attention kernel
2. **Sparsify** — zero out small values, use sparse attention mask
3. **Low-Rank** — factorize KV cache into `K = A × B` where A is small
4. **Hierarchical** — recent tokens in F16, older tokens in Q4_0

#### C. Attention Kernel Modifications
- `flash_attention.comp` modified to read quantized K/V directly
- Scale application inside the online softmax loop
- No separate dequant pass (fused)

#### D. Host Integration
- `KVCacheManager` allocates compressed cache based on config
- `InferenceEngine` passes compression params to attention dispatch

### 5.3 Priority: **[NEXT]** — After context compression is working

---

## 6. Vision / Multimodal (Last Priority)

### 6.1 Purpose
Support image inputs via CLIP/SigLIP/DINOv2 encoders, fused into text embeddings.

### 6.2 What Needs to Be Done
- `rdna4_vision.hpp` already has stubs
- `vision_encoder.comp` — patch embedding + transformer encoder
- `vision_projector.comp` — MLP projector from vision hidden to text hidden
- Image tokenizer: convert image to patch embeddings on GPU
- Model registry: add vision tower configs (LLaVA, Qwen-VL, etc.)

### 6.3 Priority: **[LAST]** — After all other features are stable

---

## Parallel Task Assignment

### Claude (Me) — Core Fixes
1. Fix K-quant layout bug in `dequantize.comp`, `cpu_reference.cpp`, `gguf_loader.py`
2. Fix Python nibble alternation bug in `dequantize_q6_k`
3. Fix weight_converter.py validation offsets
4. Fix crash 0xE06D7363 in `weight_uploader.cpp`
5. Fix scheduler fence tracking bug (`endBatch` pushing non-owned fences)
6. Build and verify with Q6_K model

### MiMo (Parallel)
1. Write `TurboQuantConfig` JSON schema (`docs/turboquant_schema.json`)
2. Write `dequant_turbo.comp` shader stub with push constants
3. Write `gemm_turbo.comp` shader stub
4. Create `src/kernels/AGENTS.md` with shader conventions
5. Add `TQ4_128`, `TQ3_128`, `TQ6_64` to `QuantFormat` enum in `rdna4.hpp`

### DeepSeek (Parallel)
1. Write context compression JSON schema (`docs/context_compression_schema.json`)
2. Write `compress_context.comp` shader stub
3. Write KV cache compression JSON schema (`docs/kv_compression_schema.json`)
4. Write `GraphifyClient` header stub (`include/rdna4_graphify.hpp`)
5. Write `tools/AGENTS.md` with Python tooling standards

---

## Execution Order

### Phase 1: Bug Fixes (Claude only)
- [ ] K-quant layout swap
- [ ] Python nibble fix
- [ ] Validation offset fix
- [ ] Crash handler fix
- [ ] Fence tracking fix
- [ ] Build + test Q6_K

### Phase 2: Feature Stubs (All 3 systems in parallel)
- [ ] MiMo: TurboQuant schema + shaders + AGENTS.md
- [ ] DeepSeek: Compression schemas + shaders + GraphifyClient + AGENTS.md
- [ ] Claude: Integrate stubs, update CMake, write .cpp implementations

### Phase 3: Integration (Claude)
- [ ] Wire TurboQuant into weight uploader + inference engine
- [ ] Wire context compression into `InferenceEngine::forwardPartial()`
- [ ] Wire KV compression into `KVCacheManager`
- [ ] Wire GraphifyClient into crash handler + model load
- [ ] Update root AGENTS.md with new child indexes

### Phase 4: Validation
- [ ] TurboQuant accuracy vs F16
- [ ] Context compression quality (perplexity retention)
- [ ] KV compression memory savings
- [ ] End-to-end inference test

### Phase 5: Vision (Future)
- [ ] Vision encoder shader
- [ ] Multimodal prompt parser
- [ ] LLaVA/Qwen-VL model support
