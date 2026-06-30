# NotLLAMA NaN Fix — Root Cause Analysis

## Problem
Q6_K (and likely other quantized formats on large models) produces all-NaN hidden states and logits on AMD RDNA4 (RX 9070 XT). F32 works fine. 260K model passes.

## Root Causes Found

### 1. CRITICAL: Workgroup count exceeds Vulkan device limit (→ NaN)
**File:** `src/host/inference_engine.cpp`  
**Bug:** `MAX_WG_PER_DISPATCH = 256 * 1024 = 262,144` exceeds the AMD RDNA4 per-dimension workgroup limit of **65,535**.

For a 3B model, `ffn_up` has ~11M elements. The dequant shader uses `local_size_x = 64`, so it dispatches `(11,272,192 + 63) / 64 = 176,128` workgroups. This exceeds 65,535, causing **undefined behavior** — the driver silently drops or wraps the dispatch. The dequant buffer is never written, leaving uninitialized garbage that downstream kernels read as NaN.

**Fix:** Changed `MAX_WG_PER_DISPATCH` to `65535` (the Vulkan minimum guaranteed limit). Large tensors are now split into multiple chunks, each within the limit.

### 2. CRITICAL: Return address bug in dequantWeight (→ wrong memory read)
**File:** `src/host/inference_engine.cpp`  
**Bug:** Both `dequantWeight()` and `dequantWeightInBatch()` return:
```cpp
return dequantBufAddr + outOffset + (uint64_t)offset * sizeof(float);
```
After the while-loop, `offset == nElements`, so this returns the address **past the end of the tensor** instead of the start.

**Fix:** Changed to:
```cpp
return dequantBufAddr + outOffset;
```

### 3. BUG: Q6_K dequantization missing scale multiplication (→ incorrect values, not NaN)
**File:** `src/kernels/dequantize.comp`  
**Bug:** The `sc` (per-subblock scale) is read but never multiplied:
```glsl
// WRONG:
Out.data[i] = d * float(val);

// CORRECT:
Out.data[i] = d * float(sc) * float(val);
```
This alone would produce ~1/10th magnitude values (not NaN), but combined with the workgroup limit bug, it compounds the failure.

**Fix:** Added `* float(sc)` to the Q6_K path.

### 4. BUG: Command pool memory leak
**File:** `src/host/scheduler.cpp`  
**Bug:** `vkResetCommandPool(device, cmdPools[i], 0)` resets command buffers but **never frees them**. They accumulate until the pool is exhausted.

**Fix:** Changed to `VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT`, which frees the underlying command buffer memory.

### 5. Added: Workgroup count validation
**File:** `src/host/scheduler.cpp`  
Added explicit checks in `dispatch()` and `dispatchInBatch()` that log a FATAL error if any dimension exceeds 65,535. This prevents silent failures in the future.

## Files Changed
- `src/kernels/dequantize.comp` — Q6_K scale fix
- `src/host/inference_engine.cpp` — workgroup limit + return address fix
- `src/host/scheduler.cpp` — command pool reset + validation

## How to apply
Copy the fixed files from this archive over your local repo files, rebuild shaders and the project.

## Answers to Your 5 Questions

1. **Why would vkCmdDispatch silently not execute in a batch?**  
   Because `gx` exceeded `maxComputeWorkGroupCount[0]` (65,535 on AMD). Vulkan validation layers would catch this, but without them the driver silently drops or wraps the dispatch.

2. **Are there known command buffer recording limits on AMD?**  
   Yes — 65,535 workgroups per dimension is the hard limit. Also, command pool buffer exhaustion if you never free them.

3. **How should push constants and dispatch be sequenced with barriers?**  
   Your current sequencing (bind pipeline → push constants → dispatch → barrier) is correct. The issue was the workgroup count, not the sequencing.

4. **Could vkResetCommandPool cause race conditions?**  
   Not in your current code because you wait for fences first. But using flag `0` causes a memory leak. Use `VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT`.

5. **What tools detect silently dropped dispatches?**  
   - `VK_LAYER_KHRONOS_validation` — catches `maxComputeWorkGroupCount` violations immediately.  
   - RenderDoc / Radeon GPU Profiler — shows which dispatches actually executed.  
   - The `dequantize_test.comp` shader (writes 1234.0) — read back and verify.
