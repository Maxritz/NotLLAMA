# NotLLAMA Single-Command-Buffer Fix

## What This Fixes

**Problem:** ~361 vkQueueSubmit + ~72 vkWaitForFences per token → massive CPU overhead.
**Solution:** ONE VkCommandBuffer for entire forward pass → ONE vkQueueSubmit → ONE sync.

## Patches

| Patch | File | What It Does |
|-------|------|-------------|
| 12 | src/host/inference_engine.cpp | Adds `dequantWeightInBatch()` helper — records dequant dispatches into active batch cmd buffer, NO separate submits, NO syncs |
| 13 | src/host/inference_engine.cpp | Rewrites `forwardPartial()` — single `beginBatch(0)` at top, all dequants + compute + sampling recorded into ONE cmd buffer, `endBatch` + `syncAll()` ONCE at end |

## Apply

```bash
cd /path/to/NotLLAMA
git apply 12_dequantWeightInBatch.patch
git apply 13_single_cmdbuffer_forwardPartial.patch
```

## Architecture (AGENTS.md Compliant)

- **No pre-dequantization:** Each layer still dequantizes its own weights on-demand inside the single cmd buffer.
- **No monolithic buffers:** `dequantBuffer` is the same reusable staging buffer (sized for max per-layer weight set).
- **Buffer reuse via barriers:** `barrierBetweenGroups` ensures GPU finishes reading dequantized weights before next layer overwrites them.
- **Per-layer sequence:**
  ```
  Dequant attention weights → BARRIER → Attention compute → BARRIER →
  Dequant FFN weights (overwrite) → BARRIER → FFN compute → BARRIER →
  [Next layer...]
  ```

## Key Changes in forwardPartial()

1. `scheduler->beginBatch(0)` — once at the top
2. Embedding dequant → `dispatchInBatch("embed")` → barrier
3. Per layer:
   - `dequantWeightInBatch()` for all 5 attention weights
   - `barrierBetweenGroups(WRITE → READ)` — dequant writes visible to compute
   - Attention compute dispatches (norm, gemm×3, rope, kv_write, attention×heads, gemm, add)
   - `barrierBetweenGroups(WRITE|READ → WRITE|READ)` — attention done, FFN can overwrite
   - `dequantWeightInBatch()` for all 4 FFN weights
   - `barrierBetweenGroups(WRITE → READ)` — FFN dequant visible to compute
   - FFN compute dispatches (norm, mlp_fused_gateup, add)
   - `barrierBetweenGroups(WRITE|READ → WRITE|READ)` — FFN done, next layer can overwrite
4. After loop:
   - `dequantWeightInBatch("output_norm")` → barrier → `dispatchInBatch("rms_norm")`
   - barrier → LM head gemm → barrier → `dispatchInBatch("topk")`
5. `scheduler->endBatch(VK_NULL_HANDLE)`
6. `scheduler->syncAll()` — ONE sync for entire token
7. `kvCache->incrementSeqLen(layer)` for all layers (after GPU done)
8. Read back token from `sampleOutAddr` via mapped memory

## What Was Removed

- `scheduler->createLayerFence()` call (dead code)
- `scheduler->syncAllThrottled(0.8)` from hot path (all 4 locations)
- `scheduler->syncLayer(0)` calls (replaced by single `syncAll()`)
- Per-layer `beginBatch()`/`endBatch()` pairs (merged into one)
- Diagnostic readbacks inside forward pass (moved after final sync)
- `sampleGpu()` call from `forwardPartial()` (topk dispatch inlined into batch)

## Expected Performance

| Metric | Before | After |
|--------|--------|-------|
| vkQueueSubmit / token | ~361 | **1** |
| vkWaitForFences / token | ~72 | **1** |
| CPU overhead | ~36ms | **~0.5ms** |
| Expected forward pass | ~500-1000ms | **~50-100ms** |

The remaining ~50-100ms is pure GPU shader time (naive scalar-loop GEMM). The next phase is rewriting `gemm.comp` with tiled shared memory.
