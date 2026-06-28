# DeepSeek Chat Tasks — Batch Mode Integration + Cleanup

**Generated**: 2026-06-28 | **Priority**: HIGH | **Model**: DeepSeek Chat (Laguna M.1)

---

## Context

The batch mode API (`beginBatch`/`dispatchInBatch`/`barrierBetweenGroups`/`endBatch`) is fully implemented in `scheduler.cpp` but **never called** from `inference_engine.cpp`. The forward pass still uses the old `dispatchBatchBarriers` path which allocates a `vector<DispatchDesc>` per layer — unnecessary overhead. Additionally, there are minor cleanup items (dead code, missing error checks).

**Goal**: Rewrite `forwardPartial()` to use batch mode, clean up dead code, add missing error checks.

---

## Task 1: Integrate batch mode into forwardPartial() [HIGH]

**File**: `src/host/inference_engine.cpp`, `forwardPartial()` method (~lines 660-904)

**Current pattern** (per layer):
```cpp
std::vector<Scheduler::DispatchDesc> dispatches;
std::vector<uint32_t> groupEnds;
std::vector<Scheduler::PipelineBarrier> barriers;
// ... emplace_back dispatches ...
scheduler->dispatchBatchBarriers(dispatches, groupEnds, barriers, 0, scheduler->layerFence);
scheduler->syncLayer(0);
```

**Target pattern** (per layer):
```cpp
scheduler->beginBatch(0);
scheduler->dispatchInBatch(pipeline, layout, &pc, sizeof(pc), gx, gy, gz);
scheduler->barrierBetweenGroups(
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
scheduler->dispatchInBatch(pipeline2, layout2, &pc2, sizeof(pc2), gx2, gy2, gz2);
scheduler->endBatch(scheduler->layerFence);
scheduler->syncLayer(0);
```

**What to change**:
1. **Attention batch** (~lines 710-762): Replace `dispatchBatchBarriers` with `beginBatch`/`dispatchInBatch`/`barrierBetweenGroups`/`endBatch`. The barrier is between norm+QKV GEMMs and RoPE+KV write+attention+output GEMM.
2. **FFN batch** (~lines 860-901): Same replacement. Barrier is between norm and fused_gateup MLP.
3. Remove the `dispatches`, `groupEnds`, `barriers` vectors entirely.
4. Each batch ends with `endBatch(scheduler->layerFence)` followed by `syncLayer(0)`.

**Key constraints**:
- Push constants must be passed as `&pc, sizeof(pc)` to `dispatchInBatch` — they are copied inline into the command buffer.
- Barriers use the same compute→compute + write→read pattern already in the code.
- `syncLayer(0)` stays after each `endBatch` — this is the per-layer sync point.

**Verify**: Build with `cmake --build . --config Release`

---

## Task 2: Remove dead MlpPushConstants [MEDIUM]

**Files**: `include/rdna4_types.hpp`, `src/kernels/mlp.comp`, `CMakeLists.txt`

`MlpPushConstants` (lines 34-41 in rdna4_types.hpp) is never used — `forwardPartial()` uses `MlpFusedGateUpPushConstants` instead. The `mlp.comp` shader is also dead code (replaced by `mlp_fused_gateup.comp`).

**Steps**:
1. Remove `MlpPushConstants` struct from `rdna4_types.hpp` (lines 34-41)
2. Remove `src/kernels/mlp.comp` from CMakeLists.txt KERNELS list (line 55)
3. Optionally delete `src/kernels/mlp.comp` entirely

**Verify**: Build succeeds, `mlp.spv` no longer generated

---

## Task 3: Add vkEndCommandBuffer error checks [LOW]

**File**: `src/host/scheduler.cpp`

`vkEndCommandBuffer` is called 5 times (lines 78, 132, 218, 409, 475) without checking the return value. On failure, the command buffer is submitted in an invalid state.

**Fix**: Add `VkResult endResult = vkBeginCommandBuffer(cmd, &beginInfo); if (endResult != VK_SUCCESS) { fprintf(stderr, "..."); return; }` after each `vkEndCommandBuffer` call. Note: `vkEndCommandBuffer` can fail if the command buffer is in a bad state — log and skip submission.

**Affected functions**:
- `dispatch()` (line 78)
- `dispatchTimed()` (line 132)
- `dispatchBatch()` (line 218)
- `dispatchBatchBarriers()` (line 409)
- `endBatch()` (line 475)

**Verify**: Build

---

## Verification

```powershell
cd C:\Users\rr\Desktop\Notllama-loc\build
cmake --build . --config Release 2>&1 | Select-String "error C"
# Should have NO errors (only C4267 warnings)
```

---

## Notes

- The batch mode API is already implemented and working in scheduler.cpp. Task 1 is purely a call-site change in inference_engine.cpp.
- `dispatchBatchBarriers` can stay in scheduler.cpp — it's still useful for other callers. Don't delete it.
- `MlpPushConstants` removal (Task 2) will break any code that references it — grep first to confirm it's truly unused.
- `vkEndCommandBuffer` failure is rare but possible. The fix is defensive — log and return without submitting.
