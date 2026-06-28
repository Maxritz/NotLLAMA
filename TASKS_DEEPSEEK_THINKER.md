# DeepSeek Reasoner Tasks — Analysis & Design

**Generated**: 2026-06-28 | **Priority**: HIGH | **Model**: DeepSeek Reasoner

---

## Task A: Fused QKV shader design [HIGH]

**Context**: Currently Q, K, V are computed as 3 separate GEMM dispatches per layer. VindexLLM fuses them into a single dispatch with a weight matrix that concatenates Q/K/V projections.

**Current pattern** (3 dispatches per layer):
```
GEMM(Q, hidden→Q) → GEMM(K, hidden→K) → GEMM(V, hidden→V)
```

**Target pattern** (1 dispatch):
```
FUSED_QKV(QKV_weights, hidden) → writes [Q, K, V] contiguously
```

**Design requirements**:
1. New shader `matmul_fused_qkv.comp` — single dispatch computes Q, K, V
2. Weight layout: [3 × nHeads × headDim, hiddenDim] concatenated, row-major
3. Output layout: [3, nHeads, headDim] written contiguously to buffer
4. Push constant struct: `{addrIn, addrQKVWeights, addrOut, dim, nHeads, headDim}` (32 bytes)
5. Each thread computes one output element (same outer-product pattern as gemm.comp)
6. Dispatch size: `(nHeads * headDim * 3 + 31) / 32` workgroups in X

**Questions to answer**:
- Should the output be [3, seq, nHeads, headDim] or [seq, 3, nHeads, headDim]?
- How does this interact with the existing RopePushConstants which expect separate Q and K addresses?
- Does this save a sync point, or just reduce dispatch count?

**Deliverable**: GLSL shader code + push constant struct + notes on Rope integration

---

## Task B: TQ3 KV cache design [HIGH]

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

**Questions to answer**:
1. What is the exact TQ3 block layout? (byte-level structure)
2. What are the WHT parameters? (H matrix size, stride, number of pre-computed sets)
3. Can the WHT be applied during dequant rather than during attention?
4. What is the dequant quality vs Q8_0/Q16?

**Deliverable**: TQ3 block layout spec, shader design, memory savings analysis

---

## Task C: Crash 0xE06D7363 diagnosis [HIGH]

**Context**: `rdna4_llama.exe` crashes with MSVC C++ exception 0xE06D7363 during `uploader.load(jsonPath, binPath)`. Vulkan Loader reports `vkCreateBuffer: Invalid device`. This happens before any of the inference code runs.

**Timeline**:
1. Device created successfully (4 compute queues, 16GB VRAM detected)
2. Scheduler created with FencePool
3. Pipeline builder creates pipelines
4. `uploader.load(jsonPath, binPath)` is called
5. **CRASH** — 0xE06D7363

**Questions to answer**:
1. Is the VkDevice valid when WeightUploader uses it? Could there be a device loss during pipeline creation that goes unnoticed?
2. Is WeightUploader creating its own VkCommandBuffer/VkFence? If so, does it use the same command pool or its own?
3. Could the FencePool initialization be interfering? (e.g., pre-allocated fences consuming device resources before WeightUploader runs)
4. Is there a validation layer that can catch the exact error before the crash?
5. Could it be a stack overflow from large model JSON parsing?

**Approach**:
- Add `VK_LAYER_KHRONOS_validation` to check for errors before the crash
- Add `fprintf(stderr, ...)` before and after each step in main.cpp to pinpoint exactly where the crash occurs
- Check if WeightUploader creates its own resources that conflict with the existing setup

**Deliverable**: Root cause analysis + fix recommendation

---

## Task D: Performance analysis — FencePool impact [MEDIUM]

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

**Deliverable**: Performance model + expected improvement + bottleneck analysis

---

## Notes

- Task A (Fused QKV) is the highest-impact VindexLLM phase after batch mode
- Task B (TQ3) is the highest-impact memory optimization
- Task C (Crash) blocks all runtime testing — must be resolved first
- Task D (Performance) helps prioritize what to optimize next
- All tasks require analysis, not code changes. Write findings to this file or a new `ANALYSIS.md`.
