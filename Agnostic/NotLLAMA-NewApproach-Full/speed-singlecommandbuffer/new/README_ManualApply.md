# NotLLAMA Single-Command-Buffer Fix — Manual Application Guide

## Problem
~361 vkQueueSubmit + ~72 vkWaitForFences per token = massive CPU overhead.

## Solution
ONE VkCommandBuffer for entire forward pass → ONE vkQueueSubmit → ONE sync.

## Files in this ZIP

| File | What to do with it |
|------|-------------------|
| `dequantWeightInBatch.cpp` | Paste this function into `src/host/inference_engine.cpp`, after the existing `dequantWeight()` function (around line 79) |
| `forwardPartial_single_cmdbuffer.cpp` | Replace the entire `forwardPartial()` function in `src/host/inference_engine.cpp` (lines 655-996) with this code |

## Step-by-Step Application

### Step 1: Add `dequantWeightInBatch()`

Open `src/host/inference_engine.cpp`. Find the `dequantWeight()` function (ends around line 79). After its closing `}`, paste the contents of `dequantWeightInBatch.cpp`.

### Step 2: Replace `forwardPartial()`

Find `uint32_t InferenceEngine::forwardPartial(uint32_t tokenId, ...)` (starts around line 655). Delete from that line to the closing `}` of the function (around line 996). Paste the contents of `forwardPartial_single_cmdbuffer.cpp` in its place.

### Step 3: Adapt FFN path (if needed)

If your local tree replaced `mlp_fused_gateup` with separate `gate`/`up`/`silu_mul`/`down` shaders, replace the FFN compute block in `forwardPartial_single_cmdbuffer.cpp` (marked with `NOTE` comment) with your current separate-shader sequence.

### Step 4: Build and test

```bash
cmake --build . --config Release
./test_inference.exe
```

## What Changed

| Before | After |
|--------|-------|
| `dequantWeight()` → separate cmd buffer + submit per tensor | `dequantWeightInBatch()` → records into active batch cmd buffer, no submit |
| `scheduler->syncAllThrottled(0.8)` after every dequant batch | `scheduler->barrierBetweenGroups()` between phases |
| `scheduler->beginBatch()`/`endBatch()` per layer (2×) | ONE `beginBatch(0)` at top, ONE `endBatch()` at end |
| `scheduler->syncLayer(0)` per layer | Removed — single `syncAll()` at end |
| `kvCache->incrementSeqLen()` inside loop | Moved after final `syncAll()` |
| Diagnostic readbacks inside hot path | Removed (moved to after sync if needed) |
| `sampleGpu()` called from `forwardPartial()` | Inlined `topk` dispatch into batch, manual readback after sync |

## Barrier Sequence Per Layer

```
[Dequant attention weights] → BARRIER(WRITE→READ) →
[Attention compute] → BARRIER(WRITE→READ) →
[Dequant FFN weights] → BARRIER(WRITE→READ) →
[FFN compute] → BARRIER(WRITE|READ→WRITE|READ) →
[Next layer...]
```

The `WRITE|READ → WRITE|READ` barrier at the end of each layer ensures the GPU is done with the dequant buffer before the next layer overwrites it.

## Expected Performance

| Metric | Before | After |
|--------|--------|-------|
| vkQueueSubmit / token | ~361 | **1** |
| vkWaitForFences / token | ~72 | **1** |
| Forward pass | ~500-1000ms/token | **~50-100ms/token** |

The remaining ~50-100ms is pure GPU shader time (naive scalar-loop GEMM). Next target: rewrite `gemm.comp` with tiled shared memory.
