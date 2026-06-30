# Critical Bugs (Will Produce Wrong Results)

## 1. KV Cache Type Mismatch

**Files**: kv_cache_write.comp, attention.comp

**Issue**: Float32↔Float16 type mismatch. GEMM writes K/V as float32 to ring buffer. KV cache write shader reads that buffer as float16_t[] — reads 2 bytes instead of 4. Corrupted K/V cache. Attention attends over garbage.

**Impact**: Entire attention mechanism fails due to corrupted KV cache.

## 2. Top-K Sampling Performance

**File**: topk.comp (deepseek rewrite)

**Issue**: O(K×N) iterative top-k. 40 iterations × 151936 elements = 6M ops per workgroup. Will take seconds per token. Needs single-pass radix select.

**Impact**: Extremely slow sampling that makes the engine unusable.

## 3. Top-K Float Equality Masking

**File**: topk.comp

**Issue**: Float equality masking. scratch.data[i] == globalMaxProb after softmax is unreliable — will sometimes fail to mask selected token, causing duplicate outputs.

**Impact**: Duplicate tokens in output sequence.

## 4. RoPE Data Race

**File**: rope.comp

**Issue**: Data race on paired dimensions. Thread d=0 reads Q[0]+Q[1], writes Q[0]. Thread d=1 reads Q[0]+Q[1], writes Q[1]. Concurrent wavefront execution = stale reads.

**Impact**: Incorrect RoPE embeddings that break positional awareness.

# High Severity Bugs

## 5. Flash Attention Register Pressure

**File**: flash_attention.comp

**Issue**: 288 floats in VGPRs (qVec[128]+acc[128]+scores[32]). 36KB register file needed per SIMD — massive spilling. May crash driver.

**Impact**: Potential driver crashes due to excessive register usage.

## 6. Kernel Entry Spin-Wait

**File**: kernel_entry.comp

**Issue**: Spin-wait without memory barrier. while(M.tokenReady == 0) with no memoryBarrierBuffer(). GPU compiler may cache read in register forever.

**Impact**: Potential infinite loop due to register caching.

## 7. Ring Allocator Overflow

**File**: inference_engine.cpp

**Issue**: Ring allocator overflow. Q alone = 32MB for seqLen=2048. Ring is 64MB. No allocation failure check.

**Impact**: Silent memory corruption when ring buffer overflows.

## 8. Sample Argmax Fallback

**File**: inference_engine.cpp

**Issue**: sampleArgmax(nullptr,...) fallback. Always returns 0. Broken error handling for GPU sampling failure.

**Impact**: Always returns token 0 when GPU sampling fails.

# Medium Severity Issues

## 9. Excessive Synchronization

**File**: inference_engine.cpp

**Issue**: Sync-after-every-dispatch. ~200 sync points per token. Throughput 10-100x lower than batched submission.

**Impact**: Significant performance degradation.

## 10. Per-Weight Dequant Dispatch

**File**: inference_engine.cpp

**Issue**: Each of ~20 dequantized tensors gets its own dispatch+sync. Dominated by overhead.

**Impact**: High dispatch overhead reducing throughput.

## 11. Speculative Decode No-Op

**File**: inference_engine.cpp

**Issue**: Speculative decode is no-op. draftForward() just calls forward() — ignores nLayers parameter.

**Impact**: Speculative decoding doesn't actually decode partial layers.

# Working Components

## ✅ BDA Test

**Status**: Passes

## ✅ Embedding (f16 weights)

**Status**: Correct

## ✅ RMS Norm (f16 weights)

**Status**: Correct

## ✅ GEMM (single token, f16)

**Status**: Works by accident (always reads row 0)

## ✅ Dequantize Q6_K

**Status**: Fixed (float32 output)

## ⚠️ Dequantize Other Formats

**Status**: Untested, format enums may not match

## ❌ KV Cache Write

**Status**: Broken (float32→float16 reinterpret)

## ❌ Attention

**Status**: Reads garbage KV cache

## ❌ Top-K Sampling

**Status**: Too slow + duplicates

## ❌ RoPE

**Status**: Data race

## ❌ Flash Attention

**Status**: Register overflow

## ⚠️ MLP

**Status**: Works for f16, fragile for other dims

## ⚠️ LM Head

**Status**: GEMM transB works, but reads row 0

# Remaining Work

1. Fix KV cache type mismatch — either make cache float32, or add explicit f32→f16 conversion in kv_cache_write kernel
2. Rewrite topk.comp — single-pass radix select, not iterative max-find
3. Fix RoPE race — two-pass approach (all read, then all write)
4. Fix flash_attention register pressure — reduce headDim tile, use shared memory for scores
5. Fix kernel_entry spin-wait — add memoryBarrierBuffer()
6. Batch dequant dispatches — submit multiple tensors per sync point
7. Batch inference dispatches — pipeline kernels instead of sync-after-each
8. Fix ring allocator overflow check — detect and fail gracefully
9. Implement draftForward — partial layer execution for speculative decoding
10. Build CLI tools — llama-cli, llama-server, llama-benchmark equivalents
11. Fix exit crash — test deepseek's new cleanup order

# Action Plan

1. Prioritize critical bugs (KV cache, top-k, RoPE)
2. Address high severity issues (flash attention, spin-wait, ring allocator)
3. Implement performance optimizations (batching, pipelining)
4. Build CLI tools for usability
5. Test and validate fixes thoroughly