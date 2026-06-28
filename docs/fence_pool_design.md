# Fence Pool Design — Task C

## Problem

```
dequantWeight (layer N):
  dispatch(dequant Q)    → no fence
  dispatch(dequant K)    → no fence
  dispatch(dequant V)    → no fence
  dispatch(dequant O)    → no fence
  syncAllThrottled()     → vkQueueWaitIdle × 4  // 2000ms

  dispatchBatchBarriers(attention batch) → layerFence
  syncLayer()                              // single fence, fast

dequantWeight (layer N):
  dispatch(dequant up)   → no fence
  dispatch(dequant gate) → no fence
  dispatch(dequant down) → no fence
  syncAllThrottled()     → vkQueueWaitIdle × 4  // 2000ms

  dispatchBatchBarriers(ffn batch) → layerFence
  syncLayer()
```

Four+ `vkQueueWaitIdle` calls per layer (36 layers → 144+ GPU-draining stalls). Each drains the entire GPU pipeline instead of just waiting on the specific submissions.

## Solution: FencePool

### Struct

```cpp
struct FencePool {
    static constexpr uint32_t MAX_OUTSTANDING = 64;

    VkDevice device;
    VkFence fences[MAX_OUTSTANDING];
    uint64_t submitTags[MAX_OUTSTANDING];  // monotonic tag per submission
    uint32_t head;       // next free slot (round-robin)
    uint32_t pending;    // count of outstanding (= in-flight) submissions
    uint64_t nextTag;    // monotonic counter, never wraps

    // Track which tags are pending for syncAll
    // pendingTags is implicit: fences in slots [head-pending, head) are active
};
```

### Allocation

```
acquire() → VkFence
  if pending == MAX_OUTSTANDING:
      // All fences in flight — must reap oldest
      waitAndReap(head % MAX_OUTSTANDING)
  slot = head % MAX_OUTSTANDING
  if fences[slot] == VK_NULL_HANDLE:
      create VkFence (initially SIGNALED — already complete)
      if first use: need to wait+reset before first submit
  else if submitTags[slot] != 0 && fence not signaled:
      // Still in flight from previous round — forced wait
      vkWaitForFences(device, 1, &fences[slot], VK_TRUE, UINT64_MAX)
  vkResetFences(device, 1, &fences[slot])
  submitTags[slot] = nextTag++
  pending++
  head++
  return fences[slot]
```

### Wait strategies

```
syncAll():
  if pending == 0: return
  count = 0
  for i in 0..MAX_OUTSTANDING:
      if submitTags[i] != 0:
          waitFences.Add(fences[i])
          count++
  vkWaitForFences(device, count, waitFences, VK_TRUE, UINT64_MAX)
  for each signaled fence:
      vkResetFences(device, 1, &fence)
      submitTags[i] = 0
  pending = 0

syncAllThrottled(double util = 0.8):
  auto t0 = steady_clock::now()
  syncAll()  // now a fence wait, not vkQueueWaitIdle
  auto t1 = steady_clock::now()
  // Reset command pools
  for i in 0..3: vkResetCommandPool(device, cmdPools[i], 0)
  // Throttle
  double gpuMs = (t1 - t0).count_ms()
  if gpuMs > 0.5 && util in (0,1):
      sleep for gpuMs * (1/util - 1) ms
```

### Integration: dequantWeight

```cpp
static uint64_t dequantWeight(...) {
    // ... chunk loop ...
    for each chunk:
        VkFence f = sched->fencePool.acquire();
        sched->dispatch(..., f);
    if (sync):
        sched->fencePool.syncAllThrottled();
}
```

### Integration: Scheduler::dispatch

```cpp
void Scheduler::dispatch(..., VkFence fence) {
    // ... existing code ...
    vkQueueSubmit(queues[aceIndex], 1, &submitInfo, fence);
    // If fence is VK_NULL_HANDLE, submission is untracked
    // (caller must sync externally — e.g. layerFence path)
}
```

No change to `dispatchBatch` or `dispatchBatchBarriers` — they already use `layerFence`.

### cleanup()

```cpp
void Scheduler::cleanup() {
    fencePool.syncAll();       // Wait for all outstanding submissions
    fencePool.destroy();       // Destroy all fences
    // ... existing cleanup ...
}
```

### Edge Cases

| Case | Handling |
|------|----------|
| 64 fences all in-flight | `acquire()` blocks on oldest fence |
| fence still signaled from previous use | `vkResetFences` before re-submit |
| `syncAll()` called with 0 pending | No-op |
| interleaved `dispatch()` + `dispatchBatch()` | Separate tracking — dequants use pool, batches use layerFence |

### Expected Speedup

- Current: 4× `vkQueueWaitIdle` per sync point. Each drains ALL queues fully.
- New: 1× `vkWaitForFences` per sync point. Only waits on the N dequant submissions.
- Layer with 4 dequant dispatches: wait on 4 fences instead of draining 4 queues.
- Over 36 layers: ~144 `vkQueueWaitIdle` calls eliminated → **~1800ms saved** (from the 2048ms total).
- Remaining `vkQueueWaitIdle` calls: zero (removed from `syncAll`/`syncAllThrottled`).

### Migration Path

1. Add `FencePool` class to `rdna4_scheduler.hpp` as a public nested type
2. Add `FencePool fencePool` member to `Scheduler`
3. Modify `syncAll()` to use `fencePool.syncAll()` instead of `vkQueueWaitIdle`
4. Modify `syncAllThrottled()` similarly
5. No changes to `dequantWeight` signature — fence acquisition is internal to `dispatch()` or injected at the call site
6. Test with single layer first, verify no regression, then full 36 layers
