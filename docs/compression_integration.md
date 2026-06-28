# Compression Integration Guide

## Overview
NotLLAMA supports context compression (shortening the KV cache) and KV cache quantization
(reducing precision of K/V tensors). This doc explains how to wire both into the engine.

## Files Delivered by DeepSeek
- `src/kernels/compress_context.comp` — compacts KV cache based on keep_mask
- `src/kernels/kv_cache_quantize.comp` — quantizes KV cache to Q4_0/Q5_0/Q8_0
- `src/kernels/kv_cache_dequant.comp` — dequantizes KV cache back to F16
- `include/rdna4_compression.hpp` — config structs + push constants
- `include/rdna4_compression_scheduler.hpp` — decision logic stub
- `tools/dox_lint.py` — DOX compliance checker
- `tools/graphify_client.py` — GraphifyClient implementation
- `tools/benchmark_compression.py` — compression benchmarking

## Engine Wiring Checklist

### Step 1: Load Compression Configs
- [ ] Parse `context_compression_schema.json` and `kv_compression_schema.json` at startup
- [ ] Populate `ContextCompressionConfig` and `KVCompressionConfig`
- [ ] Pass configs to `CompressionScheduler`

### Step 2: KV Cache Quantization (Optional)
- [ ] After every `forwardPartial()` or `forwardKernelEntry()`, check `CompressionScheduler::step()`
- [ ] If `decision.compressKV == true`:
  - Allocate quantized KV buffers (size = compressed)
  - Dispatch `kv_cache_quantize.comp` with `KVCacheQuantizePushConstants`
  - Free original F16 KV buffers (or keep as fallback)
- [ ] Before attention dispatch, if KV is quantized:
  - Dispatch `kv_cache_dequant.comp` with `KVCacheDequantPushConstants`
  - Output goes to temporary F16 buffer consumed by `attention.comp`

### Step 3: Context Compression (Optional)
- [ ] If `decision.compressContext == true`:
  - Build `keepMask` vector on host (or use importance scores from model)
  - Upload `keepMask` to GPU
  - Dispatch `compress_context.comp` with `CompressContextPushConstants`
  - Update `seqLen` to `decision.contextTargetLen`

### Step 4: Scheduler Integration
- [ ] Instantiate `CompressionScheduler` in `InferenceEngine` or `ContextManager`
- [ ] Call `step(seqLen, maxContext)` every token
- [ ] If compression triggers, pause generation, run compression shader(s), resume

### Step 5: Fallback & Safety
- [ ] If compression shader fails (VK_ERROR_DEVICE_LOST), fall back to uncompressed path
- [ ] If `seqLen < kvCfg.minSeqLen`, never compress KV
- [ ] If `seqLen < 256`, never compress context

## Performance Targets
- KV Q4_0: 4x memory reduction, <0.1% MSE vs F16
- Context sliding_window: 2x reduction, zero quality loss for recent context
- Context importance: 1.5-2x reduction, <2% perplexity increase

## Graphify Integration
- Use `GraphifyClient` in `tools/graphify_client.py` to query the knowledge graph before
  reading source files during integration work.
- Example: `client.query("Where is KV cache allocated in inference_engine.cpp?")`
