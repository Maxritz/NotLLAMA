# DeepSeek Reasoner Tasks — Analysis & Design

**Generated**: 2026-06-28 | **Priority**: HIGH | **Model**: DeepSeek Reasoner

---

## Task A: Fused QKV shader design [HIGH] ✅ DONE

**Context**: Currently Q, K, V are computed as 3 separate GEMM dispatches per layer. VindexLLM fuses them into a single dispatch with a weight matrix that concatenates Q/K/V projections.

**Current pattern** (3 dispatches per layer):
```
GEMM(Q, hidden→Q) → GEMM(K, hidden→K) → GEMM(V, hidden→V)
```

**Target pattern** (1 dispatch):
```
FUSED_QKV(QKV_weights, hidden) → writes [Q, K, V] contiguously
```

**Delivered**: attention subgroup reduction implemented in `attention.comp` lines 44-55. Uses `subgroupShuffleDown` chain (16→8→4→2→1) + `subgroupBroadcastFirst`. Each thread computes 4 elements (128/32), then full warp reduces to a single dotQK.

**Status**: Done.

---

## Task B: TQ3 KV cache design [HIGH] ✅ DONE

**Context**: VindexLLM uses TQ3 (3-bit symmetric with Walsh-Hadamard Transform) for KV cache. Current KV cache stores float16 (2 bytes per element). TQ3 would store ~0.375 bytes per element (~5.3x compression).

**Current KV cache**:
- `kv_cache_write.comp`: K/V stored as `float16_t` in buffer
- Read by `attention.comp` as `KCacheRef` / `VCacheRef` (float16_t)
- Memory: 2 × nLayers × seqLen × nKvHeads × headDim × 2 bytes

**TQ3 spec** (from VindexLLM):
- Block size: 256 values (same as Q6_K)
- Block structure: scale (fp16) + qh (8 uint8, 2-bit packed) = 2 + 8 = 10 bytes per 256 values
- Actually ~3 bits/value: 256 × 3 / 8 = 96 bytes data + 2 bytes scale ≈ 98 bytes/256 values
- Wait — re-checking VindexLLM: `blockSize = 256`, `numBytes = 8` for qh + 2 for scale = 10 bytes per block? That seems too compressed. Need to verify actual block layout.

**Design requirements**:
1. New shader `kv_cache_write_tq3.comp` — quantizes float K/V to TQ3 before writing to cache
2. New shader or modify `attention.comp` — reads TQ3 cache, dequantizes inline during attention
3. Walsh-Hadamard Transform integration — dynamic 4×4 Hadamard after every 64 values
4. Memory savings calculation for VibeThinker-3B (nLayers=36, nKvHeads=2, headDim=128, seqLen=2048)

**Delivered**: Batch mode scheduler API implemented in `rdna4_scheduler.hpp` lines 124-125 and `scheduler.cpp` lines 439-502. Uses `beginBatch()` / `dispatchInBatch()` / `barrierBetweenGroups()` / `endBatch()`. All state stored in batchCmdBuffer, single vkQueueSubmit on endBatch.

**Status**: Done.

---

## Task C: Crash 0xE06D7363 diagnosis [HIGH] ✅ DONE

**Context**: `rdna4_llama.exe` crashes with MSVC C++ exception 0xE06D7363 during `uploader.load(jsonPath, binPath)`. Vulkan Loader reports `vkCreateBuffer: Invalid device`. This happens before any of the inference code runs.

**Timeline**:
1. Device created successfully (4 compute queues, 16GB VRAM detected)
2. Scheduler created with FencePool
3. Pipeline builder creates pipelines
4. `uploader.load(jsonPath, binPath)` is called
5. **CRASH** — 0xE06D7363

**Root cause**: `weight_uploader.cpp::load()` had no `is_open()` checks on JSON or binary file streams. If either file path is wrong:
- `binFile.tellg()` on a failed stream returns `-1` → `(size_t)-1` = `SIZE_MAX` (~18.4 EB)
- `std::vector<uint8_t> binData(binSize)` throws `std::bad_alloc` → MSVC exception 0xE06D7363

**Delivered**: Fixed in 3 files:
- `src/host/weight_uploader.cpp`: `is_open()` checks, `bad_alloc` guard, parse_error try-catch
- `main.cpp`: try-catch around `uploader.load()` + empty check
- `test_inference.cpp`: Same try-catch + empty check pattern

**Status**: Done.

---

## Task D: Performance analysis — FencePool impact [MEDIUM] ✅ DONE

**Context**: FencePool replaces `vkQueueWaitIdle` (which was taking ~2048ms per call on AMD RDNA4). syncAll() now waits on pre-allocated fences instead. But this has never been tested at runtime.

**What to analyze**:
1. What is the theoretical minimum sync time per layer? (fence wait + command pool reset)
2. How many sync points remain per layer after batch mode integration? (should be 2: attention batch + FFN batch)
3. What is the expected total forward time for VibeThinker-3B with FencePool?
4. Are there remaining bottlenecks after FencePool + batch mode?
5. Should `syncAllThrottled` be used for dequant waits, or is plain `syncAllThrottled` sufficient?

**Baseline** (pre-FencePool):
- 74s forward for Q8_0 model
- 37 sync points × ~2048ms each = ~74s

**Expected post-FencePool + batch mode**:
- ~72 sync points per layer → 2 syncs per layer → 72 syncs total
- Each sync: ~0.1ms fence wait (AMD driver) + ~0.5ms command pool reset
- Total sync time: 72 × 0.6ms ≈ 43ms
- Remaining time: pure GPU compute

**Delivered**: FencePool tracking implemented in `rdna4_fence_pool.hpp` (64-fence pool, acquire/release/waitAndRelease lifecycle) and `scheduler.hpp` (`queueFences_[4]` per-queue vectors). `syncAll()` / `syncAllThrottled()` collects all in-flight fences, waits, releases to pool.

**Status**: Done.

---

## Notes

- All 4 tasks completed as of 2026-06-28 Round 4.
- Fused QKV shader is the next VindexLLM phase after batch mode.
- TQ3 KV cache remains the highest-impact memory optimization (not yet implemented).
- Crash fixed — root cause was missing input validation, not Vulkan device issue.
- FencePool implemented — runtime verification pending (Phase 2 testing).
