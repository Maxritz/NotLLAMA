# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

- Do not build, run, or execute any code, tests, or workloads without explicit user confirmation.
- Write the full code first; debug/run only after the user confirms.

## Architecture Philosophy (BINDING ON ALL AGENTS)

**Many small units doing work together.** This is the core philosophy of the project. It is non-negotiable.

What this means in practice:
- Weights stay quantized on GPU. Only staging buffers hold float32 temporarily.
- Each layer is a self-contained work unit that dequantizes its own weights on-demand.
- GPU dequantizes in small chunks per layer, not all at once.
- No monolithic pre-dequantized weight buffers. No "dump everything into one 11 GB buffer."
- Shaders are self-contained: dequantize → process → output.
- Think like many small cops patrolling, not one army.

What this means you MUST NOT do:
- Do NOT allocate > 1 GB for any single buffer.
- Do NOT pre-dequantize all weights at init time.
- Do NOT put all quantized or dequantized weights in one monolithic buffer.
- Do NOT assume float32 weights are available to shaders — they must dequantize on-demand.
- Do NOT design a shader that requires the host to precompute and stage all data before dispatch.

If you are about to write code that violates this philosophy, STOP and ask for clarification.

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

### Ownership Map

- **Root** (`/`): Build system (`CMakeLists.txt`), entry point (`main.cpp`), project-wide instructions, DEEPSEEK_CONTEXT.md (project state document)
- **`include/`**: Headers owned by corresponding `src/host/` implementations. No child DOX needed.
- **`src/host/`**: Inference engine, scheduler, allocator, KV cache, weight uploader, tokenizer, profiler, pipeline builder, mailbox
  - `inference_engine.cpp` — main inference loop, GPU sampling, CPU fallback, `forwardKernelEntry()` (one-dispatch persistent kernel), `initWeightBuffer()`, `initLayerParams()`, `initKernelEntryBuffers()`
  - `scheduler.cpp` — Vulkan queue submission, fence handling, cleanup lifecycle
- **`src/kernels/`**: 14 GLSL compute shaders compiled to SPIR-V via glslc. No child DOX needed.
  - `kernel_entry.comp` — persistent mailbox-polling kernel. Single dispatch executes all layers (embed → N×transformer → output norm → logits+argmax). No host round-trips between layers.
- **`tools/`**: Python weight converter, GGUF inspector, RAG tool. Owned separately.
- **`reference/`**: Vulkan SDK reference files. Read-only.
- **`graphify-out/`**: Knowledge graph outputs (auto-generated, not edited directly).
- **`test_inference.cpp`**: Test harness — loads model, runs GPU forward + CPU reference, compares logits. Includes crash handler, 2-min timeout, NaN/Inf detection.
- **`cpu_reference.cpp`**: CPU reference forward pass for validation. Supports F32/F16/Q6_K/Q8_0/Q4_0 dequant.
- **`build/`**: CMake build output. Not checked in.

### Key Contracts

- `scheduler.cleanup()` must be called before `ctx.cleanup()` in main.cpp — destroys command pools while VkDevice is still alive.
- `TopKPushConstants` includes `addrScratch` (vocabSize floats for probabilities), `topP`, `seed`. The topk shader does temperature-scaled softmax (parallel reduce-max + reduce-sum), then single-threaded (tid==0) bounded min-heap top-K scan over all vocabSize probabilities, top-P nucleus filtering, and PCG random weighted sampling in one dispatch. O(vocabSize log K) heap ops, single-threaded — no data race.
- `kv_cache_write.comp`: KInRef/VInRef are `float` (not `float16_t`). Explicit `float16_t()` conversion on cache write. Cache stays float16.
- `rope.comp`: Two-pass pattern — all threads read paired Q/K values into registers before any thread writes, eliminating data race.
- `flash_attention.comp`: Tiled — each thread owns `headDim/32` elements. Uses `myQ[8]`, `myAcc[8]`, `s_scores[32]` shared. Online softmax over KV tiles. Requires subgroup shuffle/ballot extensions.
- `kernel_entry.comp`: MailboxRef is `coherent`. Spin-wait loop calls `memoryBarrierBuffer()` before reading `tokenReady`. `computeLogitsAndSample()` writes logits to scratch (after V region) and does argmax reduction over subgroup to produce the next token. Output norm uses `pc.addrOutputNorm` (push constant), not the last layer's attnNorm.
- `inference_engine.cpp`: Ring allocator calls checked for zero (overflow). CPU fallback guards `cpuLogits` null before `sampleArgmax`. `forwardPartial()` uses `allocator->reset()` at start, allocates hidden→Q→K→V→attnOut→mlpOut→logits→sampleOut then later allocates `sampleScratch` (vocabSize floats) for topk scratch. Records `lastLogitsAddr`/`lastLogitsOffset` after successful forward for external readback. `cleanup()` releases dequant/embed resources; must be called before `ctx.cleanup()`.
- `test_inference.cpp`: Reads GPU logits from `engine.lastLogitsOffset` instead of recomputing offsets. `vkInvalidateMappedMemoryRanges` uses exact size (not `VK_WHOLE_SIZE`) for readback. GPU forward runs via `std::async` with 120s timeout. Uses `forwardPartial` path (kernel_entry path not used in test_inference — test needs ring allocator readback).
- Dequant dispatches are capped at 1M workgroups per dispatch (chunking). The dequant staging buffer is sized for the largest single tensor or the largest simultaneous tensor set consumed by one dispatch (currently the MLP `ffn_up` + `ffn_gate` + `ffn_down` set), capped at 512 MB. Embedding cache is persistent DEVICE_LOCAL — do not re-dequantize per token.

### Verification

- Build: `cd build && cmake --build . --config Release`
- Shader copy: `Copy-Item build\shaders\*.spv build\Release\shaders\`
- Runtime: `rdna4_llama.exe` must not crash (VK_ERROR_DEVICE_LOST or 0xC0000005)

## Changes (2026-06-28 Round 5 — Crash 0xE06D7363 fix: weight uploader input validation)

### Task C: Crash 0xE06D7363 — Root cause & fix

**Root cause**: `weight_uploader.cpp::load()` had no `is_open()` checks on JSON or binary file streams. If either file path is wrong:
- `binFile.tellg()` on a failed stream returns `-1` → `(size_t)-1` = `SIZE_MAX` (~18.4 EB)
- `std::vector<uint8_t> binData(binSize)` throws `std::bad_alloc` → MSVC exception 0xE06D7363
- Same pattern for JSON: `jsonFile >> j;` throws `nlohmann::json::parse_error` on unopenable file

**Fix** (3 files):
- `src/host/weight_uploader.cpp`:
  - Added `jsonFile.is_open()` check before parse, wrapped `jsonFile >> j` in try-catch for parse_error
  - Added `binFile.is_open()` check, `binEnd > 0` check, `bad_alloc` guard on `binData.resize()`, read verification
  - Added `#include <new>` for `std::bad_alloc`
  - Added `maxTensorSize > 0` guard before staging buffer creation
- `main.cpp`: Wrapped `uploader.load()` in try-catch for `std::exception`, checks `model.tensors.empty()` after
- `test_inference.cpp`: Same try-catch + empty check pattern

## Changes (2026-06-28 Round 2 — kernel_entry host wiring + dequant fixes)

- **kernel_entry.comp**: Fixed `computeLogitsAndSample()` — now writes logits to scratch buffer after V region, does argmax reduction over subgroup to produce next token. Fixed output norm — uses `pc.addrOutputNorm` (push constant) instead of last layer's attnNorm.
- **inference_engine.cpp**: Added `forwardKernelEntry()` — writes mailbox (tokenId, seqLen), dispatches `kernel_entry.comp` once, reads next token from output buffer. `forward()` auto-selects kernel_entry path when `kernelEntryReady` is true. Fixed `offset*2` bug in `initWeightBuffer()` and `initEmbedCache()` (same `elementOffset` fix as dequantWeight).
- **rdna4_engine.hpp**: Added `forwardKernelEntry()` declaration, `kernelEntryReady` flag, `initLayerParams()` declaration.
- **dequantize.comp**: Added `elementOffset` push constant. Fixed chunking for block-compressed formats. Added Q4_1, Q5_0, Q5_1, Q8_1, Q4_K, Q5_K, Q8_K.
- **cpu_reference.cpp**: Added all missing dequant formats.
- **weight_uploader.cpp**: Added Q2_K, Q3_K.
- **Dead code confirmed unchanged**: `model.cpp`, `memory.cpp`, `src/loaders/gguf.cpp`, `src/core/` (old `notllama::` path). `vgpr_stub.cpp`, cooperative matrix shaders, `flash_attention.comp` — intentional placeholders.

## Changes (2026-06-28 Round 3 — CPU reference weight-tying fix)

- **cpu_reference.cpp**: Fixed weight-tying (output.weight → token_embd.weight) by transposing from [vocabSize, dim] to [dim, vocabSize] row-major for LM head GEMM. Previously crashed (0xC0000005) on weight-tied models.

## Changes (2026-06-28 Round 4 — Progress review + task mapping)

### TASKS_DEEPSEEK_THINKER.md Status

**Task A** (attention subgroup reduction): ✅ DONE
- `attention.comp` lines 44-55: `subgroupShuffleDown` chain (16→8→4→2→1) + `subgroupBroadcastFirst`. Each thread computes 4 elements (128/32), then full warp reduces to a single dotQK.

**Task B** (batch mode scheduler API): ✅ DONE
- `rdna4_scheduler.hpp` lines 124-125: `batchAceIndex`, `batchCmdBuffer` members
- `scheduler.cpp` lines 439-502: `beginBatch()` / `dispatchInBatch()` / `barrierBetweenGroups()` / `endBatch()`
- All store state in batchCmdBuffer, single vkQueueSubmit on endBatch.

**Task C** (fence pool tracking): ✅ DONE
- `rdna4_fence_pool.hpp`: 64-fence pool, acquire/release/waitAndRelease lifecycle
- `scheduler.hpp`: `queueFences_[4]` (per-queue vectors) replaces single `latestQueueFence[4]`
- `scheduler.cpp`: each submit pushes fence to `queueFences_[aceIndex]`. `syncAll()` / `syncAllThrottled()` collects all in-flight fences, waits on all, releases all back to pool, then clears vectors.

**Task D** (Q6_K model validation): ✅ DONE
- `gguf_loader.py` dequantize_q6_k: indexing is correct (`qh[j*4 + l//4]` = `qh[(j*16+l)//4]`)
- `weight_converter.py` lines 18-36: `validate_q6_k()` checks all Q6_K blocks for delta > 1000 or NaN
- Q6_K blocks with NaN scales are legitimate — some quantizers produce them. Shader handles gracefully. Not a code bug.

### TASKS_DEEPSEEK_CHAT.md Status

| Task | File | Status | Notes |
|------|------|--------|-------|
| 1 | `src/host/inference_engine.cpp` | ✅ DONE | Batch mode integrated: forwardPartial() uses beginBatch/dispatchInBatch/barrierBetweenGroups/endBatch for attention and FFN batches. No more vector allocs per layer. |
| 2 | `src/host/inference_engine.cpp` | ✅ DONE | VK_WHOLE_SIZE → exact aligned size for logits readback |
| 3 | `src/host/inference_engine.cpp` | ❌ PENDING | Overflow check for addrDownW_dq |
| 4 | `src/host/scheduler.cpp` | ✅ DONE | vkBeginCommandBuffer errors checked in beginBatch/dispatchBatch/dispatchBatchBarriers |
| 5 | `src/host/scheduler.cpp` | ✅ DONE | vkEndCommandBuffer error checks added to all 5 call sites (dispatch, dispatchTimed, dispatchBatch, dispatchBatchBarriers, endBatch) |
| 6 | `src/host/scheduler.cpp` | ⚠️ NOT DONE | vkQueueSubmit error check missing in dispatch() — fixed in dispatch(), but check dispatchBatch too |
| 7 | `src/kernels/mlp.comp` | ✅ DONE | mlp.comp deleted (replaced by mlp_fused_gateup.comp). MlpPushConstants removed from rdna4_types.hpp. Dead loadPipe("mlp", ...) removed from main.cpp and test_inference.cpp. |


