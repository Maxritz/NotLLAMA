# Kimi — Strict Validation Protocol

## Current Status: READY FOR VALIDATION
- Crash fixed ✅ (vkDeviceWaitIdle + _exit)
- LM head fixed ✅ (output.weight)
- GPU forward working ✅ (10ms, token 99, no NaN/Inf)
- GPU/CPU mismatch expected ⚠️ (algorithmic differences, max error 13.0)
- Ready to validate DeepSeek's Option D implementation

## Core Philosophy (NON-NEGOTIABLE)
**Many small units doing work together.** Weights stay quantized. GPU dequantizes on-demand per layer in small chunks. Each layer is a self-contained work unit. Not one monolithic buffer. Not precomputing everything upfront. Like many small cops patrolling — not one army.

If any code or design violates this philosophy, STOP and report it.

## Pre-condition
GLM (Laguna XS.2) has completed architecture review. **Option D selected.** You may start validation now.

## Steps (follow EXACTLY in order)

### Step 1: Read the architecture decision
```powershell
Get-Content "C:\Users\rr\Desktop\Notllama-loc\tasker\GLM_TASK.md" | Select-String -Pattern "RECOMMENDATION"
```
Record which option was recommended (A/B/C/D). This determines what you validate.

### Step 2: Build
```powershell
cd C:\Users\rr\Desktop\Notllama-loc\build
cmake --build . --config Release 2>&1 | Select-String -Pattern "error C|error LNK|fatal error"
```
- If output is NOT empty: STOP. Copy-paste the errors.
- If output is empty: proceed to Step 3.

### Step 3: Copy SPIR-V
```powershell
Copy-Item "C:\Users\rr\Desktop\Notllama-loc\build\shaders\*.spv" "C:\Users\rr\Desktop\Notllama-loc\build\Release\shaders\" -Force
```

### Step 4: Run test
```powershell
cd "C:\Users\rr\Desktop\Notllama-loc\build\Release"
.\test_inference.exe ..\model\VibeThinker-3B.Q6_K.weights.json ..\model\VibeThinker-3B.Q6_K.weights.bin 2> stderr.txt
```
- Timeout: 120 seconds. If hangs, kill and report "HUNG".
- If crashes: report exit code + crash handler output.

### Step 5: Check stderr for philosophy compliance
```powershell
Select-String -Path "C:\Users\rr\Desktop\Notllama-loc\build\Release\stderr.txt" -Pattern "weight-buffer.*11771|FAILED to allocate.*11771"
```
If you see `11771 MB` or `FAILED to allocate` — the monolithic weight buffer approach is still in use. This VIOLATES the philosophy. Report it as a BLOCKER.

### Step 6: Record results
Look for these in stderr and record YES/NO:
1. `[kernel-entry] dispatch:` — kernel_entry path used?
2. `[kernel-entry] output token:` — completed?
3. `[NAN/INF DETECTED!]` — NaN/Inf in logits?
4. `[PASS]` or `[FAIL]` — comparison result
5. `Max error:` — numeric value
6. Any `FAILED to allocate` — OOM?

### Step 7: Report
```
ARCH_OPTION: [which option GLM recommended]
BUILD: [PASS/FAIL]
PHILOSOPHY_COMPLIANT: [YES/NO — no monolithic 11GB buffer]
KERNEL_ENTRY_USED: [YES/NO]
COMPLETED: [YES/NO]
HUNG: [YES/NO]
NAN_INF: [YES/NO]
RESULT: [PASS/FAIL]
MAX_ERROR: [number]
```

## Forbidden actions
- Do NOT modify any source file
- Do NOT run cmake configure
- Do NOT recompile shaders
- Do NOT skip steps
- Do NOT guess — report raw output only
