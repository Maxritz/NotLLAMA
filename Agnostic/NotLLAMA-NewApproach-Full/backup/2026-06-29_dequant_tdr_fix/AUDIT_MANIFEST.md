# Audit Manifest — Dequant Accuracy & TDR Fix
**Date**: 2026-06-29 00:40
**Last commit**: `45672fb` (`NotLLAMA inference fixes: embed_q8_0, CPU sampling, scheduler fence, CPU diagnostics`)
**Machine**: Windows x64, AMD RX 9070 XT (RDNA4), WDDM driver, Vulkan SDK 1.4.350.0

## Problem Statement

1. **VK_ERROR_DEVICE_LOST (-4)** on 36-layer models (vibethinker-3b-q6k, qwen2.5-coder-3b-q8_0)
   - Root cause: forwardPartial() submitted ALL layers in a single Vulkan command buffer
   - GPU works >2 seconds → exceeds AMD WDDM TDR timeout → driver kills the device
   - 24-layer model (acrux-500m) completes batch without crash BUT dequant dispatches silently dropped

2. **Q6_K/Q8_0 dequant accuracy issues**
   - Standalone `dispatch()` works correctly (dequantize_test writes 1234.0 to ffn_down offset → confirmed)
   - `dispatchInBatch()` at large buffer offsets (~34 MB into dequant buffer) silently drops dispatches
   - Q8_0 max error vs CPU = 59.5, mean error = 2.85 (when not crashing)

3. **dispatchInBatch() vs dispatch() discrepancy**
   - Same shader, same pipeline, same GPU address works via `scheduler->dispatch()` but not `scheduler->dispatchInBatch()`
   - Suggests either batch command buffer overflow, barrier misordering, or the single-batch approach is fundamentally incompatible with AMD WDDM

## Changes Made

### 1. `src/host/inference_engine.cpp` — Per-layer batch refactoring
**Location**: `forwardPartial()` function (~line 706)
**What changed**:
- **Before**: Everything in a single `beginBatch(0)` / `endBatch(VK_NULL_HANDLE)` / `syncAll()`
- **After**: Three separate batches (embedding, per-layer loop, final norm + LM head)
  - Embedding: `beginBatch(0)` → embed dispatch → `endBatch` → `syncAll`
  - Per layer loop: `beginBatch(0)` → all attention+FFN dispatches for ONE layer → `endBatch` → `syncAll` (repeats for each layer)
  - Final: `beginBatch(0)` → output_norm dequant + rms_norm + LM head GEMM → `endBatch` → `syncAll`
- **Key diff**: Each layer gets its own Vulkan command buffer and submit, keeping GPU work <100ms per submit (below ~2s TDR threshold)

### 2. `src/kernels/dequantize.comp` — Reverted debug to production
**What changed**: The debug modification that wrote `999.0 * d` was reverted back to the correct `d * float(val)`
**Line 72** (Q6_K): `d * float(val)` (was temporarily `999.0 * d` for debug)

### 3. `src/kernels/dequantize_test.comp` — NEW diagnostic shader
**Purpose**: Unconditionally writes `1234.0` to all output elements, regardless of weight buffer content
**Uses**: `DequantizePushConstants` (same struct as dequantize.comp for drop-in pipeline swap)
**Findings**:
- `scheduler->dispatch()` via this shader writes 1234.0 correctly to ffn_down offset → 1,048,576 instances found
- `scheduler->dispatchInBatch()` via this shader writes 0 instances → batch dispatches at large offsets are silently dropped

### 4. `CMakeLists.txt` — Added dequantize_test.comp to KERNELS
**Line 63**: Added `src/kernels/dequantize_test.comp` to the shader build list

### 5. `test_inference.cpp` — Added dequantize_test pipeline loading
**Line 132**: Added `loadPipe("dequantize_test", sizeof(DequantizePushConstants));`

## Reversal Instructions

To revert all changes:

```powershell
# From C:\Users\rr\Desktop\Notllama-loc
$backup = "backup\2026-06-29_dequant_tdr_fix"

# Restore inference_engine.cpp (CRITICAL — per-layer batching)
Copy-Item "$backup\inference_engine.cpp" "src\host\inference_engine.cpp" -Force

# Restore scheduler.cpp (if needed)
Copy-Item "$backup\scheduler.cpp" "src\host\scheduler.cpp" -Force

# Restore dequantize.comp (reverted to production)
Copy-Item "$backup\dequantize.comp" "src\kernels\dequantize.comp" -Force

# Restore test_inference.cpp
Copy-Item "$backup\test_inference.cpp" "test_inference.cpp" -Force

# Restore CMakeLists.txt
Copy-Item "$backup\CMakeLists.txt" "CMakeLists.txt" -Force
```

To remove dequantize_test diagnostics:
```powershell
Remove-Item "src\kernels\dequantize_test.comp" -Force
Remove-Item "build\shaders\dequantize_test.spv" -Force
Remove-Item "build\Release\shaders\dequantize_test.spv" -Force
# Then remove dequantize_test.comp from CMakeLists.txt KERNELS list
# Then remove loadPipe("dequantize_test", ...) from test_inference.cpp
```

## Verification

After reversal, rebuild:
```powershell
cd build; cmake --build . --config Release; Copy-Item build\shaders\*.spv build\Release\shaders\ -Force
```

## Files in Backup

| File | Size | Role |
|------|------|------|
| `inference_engine.cpp` | 60,787 | Main inference loop with per-layer batching |
| `scheduler.cpp` | 21,213 | Scheduler (for reference, may be unchanged) |
| `dequantize.comp` | 9,541 | Production dequant shader (reverted from debug) |
| `dequantize_test.comp` | 847 | Diagnostic shader (new, writes 1234.0) |
| `dequantize_test.spv` | 1,120 | Compiled diagnostic SPIR-V |
| `dequantize.spv` | 24,580 | Compiled production SPIR-V |
| `test_inference.cpp` | 11,564 | Test harness with dequantize_test pipeline |
| `CMakeLists.txt` | 5,108 | Build system with dequantize_test.comp |
| `rdna4_types.hpp` | 5,602 | Push constant structs (for reference) |
| `AUDIT_MANIFEST.md` | this file | Audit trail and reversal guide |

## Known Issues (after this change set)

- **dispatchInBatch() drops dispatches at large dequant buffer offsets** (>~30 MB). Root cause not yet identified. Possible causes:
  - Push constant data going out of scope between `dispatchInBatch()` call and `endBatch()` submit (stack lifetime issue?)
  - Command buffer recording silently failing without error return
  - AMD WDDM driver dropping commands in long-running buffers
- **Q6_K accuracy not yet verified** with per-layer batching
- **acrux-500m-o1-q6k model files deleted** from `model/` directory (accident)
