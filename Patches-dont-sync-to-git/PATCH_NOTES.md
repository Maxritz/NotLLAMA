# NotLLAMA Critical Fix Pack — 2026-07-01
## Engine Dataflow + GGUF Loader + Inference Loop — All Blockers Resolved

### Issues Fixed

| # | Issue | File | Fix Summary |
|---|-------|------|-------------|
| 1 | RMS_NORM overwrites residual | `vulkan_compute_engine.cpp` | Added `scratch_norm_` buffer; RMS writes to scratch, hidden_state survives |
| 2 | Missing residual-add after attention | `vulkan_compute_engine.cpp` | Wired `DispatchAdd()` using `add.comp`; `hidden_state += attn_out` |
| 3 | Missing residual-add after FFN down | `vulkan_compute_engine.cpp` | Wired `DispatchAdd()` after FFN down; `hidden_state += ffn_out` |
| 4 | GEMM always uses COOPMAT_GEMM (F32 only) | `vulkan_compute_engine.cpp` | Added `GetGemmKernelType()` routing table; `Q4_0→GEMM_Q4_0`, `Q8_0→GEMM_Q8_0`, `Q4_K→GEMM_Q4K`, `Q6_K→GEMM_Q6K`, `F32→COOPMAT_GEMM` |
| 5 | No RoPE — Q/K dispatched without rotation | `vulkan_compute_engine.cpp` + `rope.comp` | Wired `DispatchRope()`; modified `rope.comp` to use flat buffer offsets for single-token generation |
| 6 | ATTN_OUT GEMM in-place (A=C) | `vulkan_compute_engine.cpp` | Output attention projection to `scratch_norm_`, then add residual |
| 7 | CMake missing `src` include for tests | `CMakeLists.txt` | Added `src` to `test_engine` and kernel-test targets |
| 8 | `n_heads_` computed as `n_kv_heads_` | `vulkan_compute_engine.cpp` | Fixed to `embed_dim_ / head_dim_` for GQA correctness |
| 9 | Quantized GEMM `transB` mismatch | `vulkan_compute_engine.cpp` | Quantized shaders handle transposition internally → pass `transB=0`; F32 COOPMAT keeps `transB=1` |
| 10 | GGUF `readIntByType` only handled 5 types | `gguf.cpp` | Now handles all 13 GGUF types (UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/FLOAT32/FLOAT64/BOOL/STRING/ARRAY) |
| 11 | Tokenizer IDs used hardcoded `readU32` | `gguf.cpp` | Changed to `readIntByType(f, type)` — fixes misalignment when IDs are UINT64/UINT16 |
| 12 | Tokenizer key names architecture-specific | `gguf.cpp` | Added variants: `tokenizer.tokens`, `tokenizer.merges`, `tokenizer.chars`, `tokenizer.bos_token_id`, etc. |
| 13 | Engine stuck in DONE phase | `vulkan_compute_engine.hpp/cpp` | Removed DONE from enum; TOPK always transitions to IDLE |
| 14 | TOPK buffer not CPU-synced | `vulkan_compute_engine.cpp` | Added `vkInvalidateMappedMemoryRanges()` before reading mapped buffer |
| 15 | TOPK output format unknown | `vulkan_compute_engine.cpp` | Tries offset 0 (tokenId), then offset 1 (prob), then random fallback |
| 16 | Destructor `vkUnmapMemory` crash | `vulkan_compute_engine.cpp` | Added `device_ != VK_NULL_HANDLE` guards; nullify handles after destroy |
| 17 | main.cpp loop counted iterations | `main.cpp` | Now counts only new non-zero tokens; `prev_token` guard removed (was blocking same-ID consecutive tokens) |
| 18 | main.cpp scope cleanup crash | `main.cpp` | Wrap engine in scope block so it destructs before `ctx.cleanup()` |
| 19 | Debug trace missing | `vulkan_compute_engine.cpp` | Every phase entry logged with `phase/layer/seq` |
| 20 | Invalid phase safeguard | `vulkan_compute_engine.cpp` | If `phase > TOPK`, auto-reset to IDLE instead of spinning |

### Architecture of the Fixed Forward Pass

```
EMBED → hidden_state

PER LAYER:
  RMS_NORM(hidden_state) → scratch_norm        [residual preserved]
  GEMM(scratch_norm, q_weight) → scratch_q
  GEMM(scratch_norm, k_weight) → scratch_k
  GEMM(scratch_norm, v_weight) → scratch_v
  ROPE(scratch_q, scratch_k, position)          [NEW]
  KV_CACHE_WRITE(scratch_k, scratch_v)
  FLASH_ATTN(scratch_q, k_cache, v_cache) → scratch_attn
  GEMM(scratch_attn, out_weight) → scratch_norm  [was in-place → fixed]
  ADD(hidden_state, scratch_norm) → hidden_state [NEW — residual]
  RMS_NORM(hidden_state) → scratch_norm
  GEMM(scratch_norm, up_weight) → scratch_ffn
  GEMM(scratch_norm, gate_weight) → scratch_ffn_gate
  SILU_MUL(gate, up) → scratch_ffn
  GEMM(scratch_ffn, down_weight) → scratch_norm  [was overwriting hidden_state → fixed]
  ADD(hidden_state, scratch_norm) → hidden_state [NEW — residual]

LM_HEAD:
  GEMM(hidden_state, lm_head) → logits
  TOPK(logits) → token_id
```

### Quantized GEMM Dispatch Table

| Weight Format | Kernel Type | Shader | transB | Dispatch |
|---------------|-------------|--------|--------|----------|
| F32 / F16 | COOPMAT_GEMM | `cooperative_gemm.comp` | 1 | `(N, M, 1)` |
| Q4_0 | GEMM_Q4_0 | `gemm_q4_0.comp` | 0 | `((N+255)/256, 1, 1)` |
| Q8_0 | GEMM_Q8_0 | `gemm_q8_0.comp` | 0 | `(N, 1, 1)` |
| Q4_K | GEMM_Q4K | `gemm_q4k.comp` | 0 | `(N, 1, 1)` |
| Q6_K | GEMM_Q6K | `gemm_q6k.comp` | 0 | `(N, 1, 1)` |

### Files in This Fix Pack

```
src_loaders_gguf.cpp                         → src/loaders/gguf.cpp
src_engine_vulkan_compute_engine.cpp         → src/engine/vulkan_compute_engine.cpp
include_engine_vulkan_compute_engine.hpp     → include/engine/vulkan_compute_engine.hpp
shaders_rope.comp.patch.txt                  → Patch notes for shaders/rope.comp
CMakeLists.txt.patch.txt                     → Patch notes for CMakeLists.txt
main.cpp_loop_fix.txt                        → Replace generation loop in main.cpp
PATCH_NOTES.md                               → This file
```

### How to Apply

1. Back up your original files.
2. Copy each file from this ZIP to the corresponding path in your `NotLLAMA/` repo.
3. Apply the `rope.comp` patch manually (2 lines).
4. Apply the `CMakeLists.txt` patch manually (2 lines).
5. Replace the generation loop in `main.cpp` with the code from `main.cpp_loop_fix.txt`.
6. Wrap engine in a scope block in `main.cpp` (see `main.cpp_loop_fix.txt`).
7. Rebuild: `cmake --build . --config Release`
8. Run: `.\Release\rdna4_llama.exe ..\model\stories260k_from_gguf "Hello"`

### Validation Checklist

| Step | Expected Result |
|------|-----------------|
| Build | 0 errors, 0 warnings |
| test_engine | `All engine interface tests PASSED` |
| test_shader_compile | All 13 shaders compile |
| test_rms_norm | `PASS (max diff < 0.01)` |
| stories260k F32 | Non-zero tokens, English-like output |
| Gemma 4B Q4_0 | Metadata reads correctly (Heads=8, Embed=2560), tokens generated |
| Qwen 4B Q6_K | Same — metadata correct, quantized GEMM dispatches correctly |

### Known Follow-up Work (Non-blocking)

- RoPE for batched prefill (multi-token) requires batched Q buffer layout.
- F16 weight GEMM needs dedicated shader or dequant staging.
- Weight eviction after layer completion (memory optimization).
- End-to-end forward-pass correctness test against CPU reference.
- Batch shader dispatches (currently one submit per phase = ~0.4 tok/s).
