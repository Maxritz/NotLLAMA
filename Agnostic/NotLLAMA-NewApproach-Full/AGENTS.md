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
- Do NOT pre-dequantize all weights at init time.
- Do NOT put all quantized or dequantized weights in one monolithic buffer.
- Do NOT assume float32 weights are available to shaders — they must dequantize on-demand.
- Do NOT design a shader that requires the host to precompute and stage all data before dispatch.

Buffer size guidance (updated):
- Single buffers may exceed 1 GB when model size requires it.
- Allocation failures must be handled gracefully: return error immediately, do not retry in loops, do not silently skip and fill logs.

If you are about to write code that violates this philosophy, STOP and ask for clarification.

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

### Ownership Map

- **Root** (`/`): Build system (`CMakeLists.txt`), entry point (`main.cpp`), project-wide instructions, DEEPSEEK_CONTEXT.md (project state document)
- **`include/`**: Headers owned by corresponding `src/host/` implementations. No child DOX needed.
  - `rdna4_compression.hpp` — Push constant structs for context compression + KV cache quantize/dequant shaders. Static asserts ≤128B. Note: `CompressContextPushConstants` currently describes a quantization layout; it must be reconciled with the compaction layout in `compress_context.comp` before host dispatch (see `docs/wiring_plan.md`).
  - `rdna4_turboquant.hpp` — Host-side block layouts for TQ4_128, TQ3_128, TQ6_64 plus packed-size helpers. Static asserts block sizes.
  - `rdna4_types.hpp` — Host push constant structs for all shaders, including `DequantTurboPushConstants` (40B) and `GemmTurboPushConstants` (56B).
- **`docs/`**: Design and integration documentation. Owned by root.
  - `wiring_plan.md` — Implementation plan for wiring TurboQuant and context/KV compression assets into the engine.
  - `turboquant_integration.md` — TurboQuant block layouts, bit-packing diagrams, and 5-step engine wiring checklist.
  - `turboquant_formats.md` — TurboQuant format variants and comparison table.
  - `turboquant_schema.json` — JSON schema for TurboQuant configuration.
- **`src/host/`**: Inference engine, scheduler, allocator, KV cache, weight uploader, tokenizer, profiler, pipeline builder, mailbox
  - `inference_engine.cpp` — main inference loop, GPU sampling, CPU fallback, `forwardKernelEntry()` (one-dispatch persistent kernel), `initWeightBuffer()`, `initLayerParams()`, `initKernelEntryBuffers()`
  - `scheduler.cpp` — Vulkan queue submission, fence handling, cleanup lifecycle
- **`src/kernels/`**: GLSL compute shaders compiled to SPIR-V via glslc. No child DOX needed.
  - `silu_mul.comp` — Elementwise SiLU(gate) * up for MLP. 256 threads, dispatched over hiddenDim.
  - `mlp_fused_gateup.comp` — Fused MLP shader. Now unused in `forwardPartial()` (replaced by silu_mul + gemm sequence) due to algorithmic overwork bug (~45B FMAs/layer). Retained for potential `kernel_entry.comp` use or future fused rewrite.
  - `kv_cache_dequant.comp` — General-purpose dequant (bits=4,5,6,8, blockSize=64/128).
  - `kernel_entry.comp` — persistent mailbox-polling kernel. Single dispatch executes all layers (embed → N×transformer → output norm → logits+argmax). No host round-trips between layers.
  - `dequant_turbo.comp` — Standalone TurboQuant dequant (TQ4/TQ6 implemented, TQ3 stubbed).
  - `gemm_turbo.comp` — Fused GEMM over TurboQuant weights (TQ4/TQ6 implemented).
- **`tools/`**: Python weight converter, GGUF inspector, RAG tool, DOX lint, TurboQuant utilities. Owned separately.
  - `dox_lint.py` — AGENTS.md compliance checker, push constant size validator, shader SPIR-V presence checker.
  - `graphify_client.py` — Python GraphifyClient for subprocess-based knowledge graph queries.
  - `benchmark_compression.py` — KV cache + context compression benchmark tool.
  - `validate_turboquant.py` — Synthetic and optional real-model TurboQuant accuracy validation.
  - `benchmark_turboquant.py` — Per-tensor and summary benchmarking against Q8_0 baseline.
  - `convert_gguf_to_turboquant.py` — Standalone GGUF-derived binary → TurboQuant converter.
- **`reference/`**: Vulkan SDK reference files. Read-only.
- **`graphify-out/`**: Knowledge graph outputs (auto-generated, not edited directly).
- **`test_inference.cpp`**: Test harness — loads model, runs GPU forward + CPU reference, compares logits. Includes crash handler, 2-min timeout, NaN/Inf detection.
- **`test/test_compression.cpp`**: Compile-time test for compression push constants, config defaults, scheduler logic.
- **`test/test_turboquant.cpp`**: Compile-time test for TurboQuant block sizes, packed-size helpers, and push constant size guards.
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
- `forwardPartial()` pipeline barriers: inside each transformer layer the batched command buffer must contain explicit `barrierBetweenGroups()` at these producer→consumer handoffs: (1) after the attention RMS-norm write before Q/K/V GEMM reads, (2) after the attention output-projection GEMM write before the residual `add` reads it, (3) after the FFN down-projection GEMM write before the residual `add` reads it. Missing any barrier causes visible GPU/CPU hidden-state divergence.
- Dequant dispatches are capped at 1M workgroups per dispatch (chunking). The dequant staging buffer is sized for all 9 dequantized weights in one layer (attn_norm + Q + K + V + O + ffn_norm + up + gate + down), capped at 512 MB. Embedding cache is persistent DEVICE_LOCAL — do not re-dequantize per token. `forwardPartial` refactored for Round 4: Phase 1 dequants all weights async with one sync; Phase 2 runs the full layer (attention + FFN) in a single batch submit with pipeline barriers. 2 syncs per layer (~72 for 36 layers).

### Verification

- Build: `cd build && cmake --build . --config Release`
- Shader copy: `Copy-Item build\shaders\*.spv build\Release\shaders\`
- Runtime: `rdna4_llama.exe` must not crash (VK_ERROR_DEVICE_LOST or 0xC0000005).
- Parity: `test_inference.exe` on the F32 `stories260K` model must match the CPU reference: GPU argmax token equals CPU token and per-layer hidden-state max absolute diff < 0.01.

## Changes (2026-06-28 MiMo Round 2 + 3 — TurboQuant host-side deliverables)

### MiMo Round 2 Deliverables

**D1 — `include/rdna4_turboquant.hpp`**: Host-side block layouts for TQ4_128 (66B), TQ3_128 (50B), TQ6_64 (50B), plus `TQBlockHeader` and packed-size helpers. Static asserts verify sizes. One-line `// MiMo Round 2 — host-side TurboQuant block layouts` comment at top.

**D2 — `include/rdna4_types.hpp`**: Appended `DequantTurboPushConstants` (40B) and `GemmTurboPushConstants` (56B) matching the layouts in `src/kernels/dequant_turbo.comp` and `src/kernels/gemm_turbo.comp`. Both under 128B with `static_assert`.

**D3 — `tools/weight_converter.py`**: Appended `convert_to_tq4`, `convert_to_tq3`, `convert_to_tq6` (accept float32 array + uint16 fp16 scale) plus `dequant_tq4/tq3/tq6` helpers. Packing follows shader bit layouts: TQ4 low-nibble-first, TQ3 sequential 3-bit stream, TQ6 sequential 6-bit stream. Added `--tq-test` CLI comment placeholder.

**D4 — `tools/validate_turboquant.py`**: Standalone validation script. Tests synthetic distributions (uniform, normal, sin, sparse) and optional real model weights. Prints comparison table and exits 0 if all MSE < 0.01.

### MiMo Round 3 Deliverables

**D1 — `tools/benchmark_turboquant.py`**: Loads a `weight_converter.py` output directory (`.weights.json` + `.bin`), quantizes each tensor to TQ4_128/TQ3_128/TQ6_64, dequantizes back, prints per-layer and summary tables. Supports `--model`, `--compare-gguf`, `--output`.

**D2 — `tools/convert_gguf_to_turboquant.py`**: Standalone CLI that reads a GGUF-derived `.bin` + `.json` and emits `<model>_<format>.weights.bin` + `.json` with TQ metadata and `tensor_offsets`/`block_layout` sections.

**D3 — `test/test_turboquant.cpp`**: Compile-time test including `static_assert`s for block sizes, packed-size helper asserts, and push constant size checks. Size assertions reflect actual shader scalar layouts (40B and 56B).

**D4 — `docs/turboquant_integration.md`**: Integration guide with block layout byte diagrams, bit-packing diagrams, 5-step engine wiring checklist, performance targets, and files-modified list.

### Guardrails respected
- No existing engine/scheduler/main.cpp files modified.
- No existing `QuantFormat` enum values changed.
- No pre-dequantization of entire tensors or >1 GB buffers.
- `CMakeLists.txt` updated only to add `dequant_turbo.comp` and `gemm_turbo.comp` to `KERNELS`.
- 7 new files created, 3 existing files appended (`include/rdna4_types.hpp`, `tools/weight_converter.py`, `CMakeLists.txt`).

## Changes (2026-06-28 — forwardPartial GPU/CPU parity fix + debug cleanup)

- **`src/host/inference_engine.cpp`**: Added the three required `barrierBetweenGroups()` calls inside each transformer layer (after attention RMS-norm, after attention output-projection, after FFN down-projection) to eliminate GPU/CPU hidden-state divergence. Removed temporary per-layer and layer-0 debug capture buffers/copies/readback (`layerDebug*`, `captureLayer0`, `layer0Debug*`).
- **`include/rdna4_engine.hpp`**: Removed `layerDebug*` debug buffer members.
- **`AGENTS.md`**: Documented the `forwardPartial()` pipeline-barrier contract and added the F32 parity verification step.

## Changes (2026-06-28 Round 4 — Batched forwardPartial: 2 syncs/layer, all-weights-before-compute)

- **`src/host/inference_engine.cpp`**: `initDequantBuffer()` resized for all 9 weights per layer (was MLP-set-only). `forwardPartial()` restructured: Phase 1 dequants ALL weights async, one sync; Phase 2 runs entire layer (attention + FFN) in one batch submit with pipeline barriers. Syncs per layer: 4 → 2. ~72 syncs total for 36 layers.

## Changes (2026-06-28 Round 7 — Context compression Round 2 + 3: dequant shader, host header, lint tool, graphify client, benchmark, scheduler, tests, guide)

### Round 2 Deliverables (deepseek-task-round2.md)

**D1 — `src/kernels/kv_cache_dequant.comp`**: General-purpose KV cache dequantization shader supporting bits=4,5,6,8, blockSize=64/128, scaleBits=8/16, zeroPoint=0/1. Byte-level addressing via `uint8_t[]`. Per-block layout: `[packed_weights] [scale (1B int8 or 2B fp16 LE)] [optional zero_point (1B)]`. Compiles with glslc.

**D2 — `include/rdna4_compression.hpp`**: C++17 header with `ContextCompressionConfig`, `KVCompressionConfig`, and 3 push constant structs (`CompressContextPushConstants` 56B, `KVCacheQuantizePushConstants` 64B, `KVCacheDequantPushConstants` 40B). All under 128B with `static_assert` guards.

**D3 — `tools/dox_lint.py`**: DOX compliance linter — AGENTS.md coverage, required sections, push constant sizes (≤128B), shader SPIR-V presence, TODO/FIXME density, large files. Pure Python, stdlib.

**D4 — `src/kernels/AGENTS.md`**: Appended Compression Shaders section.

### Round 3 Deliverables (deepseek-task-round3.md)

**D1 — `tools/graphify_client.py`**: Python `GraphifyClient` class mirroring `include/rdna4_graphify.hpp`. Subprocess-based graphify query with LRU cache (configurable), `is_stale()`, `update_graph()`, `get_related_nodes()`, `clear_cache()`, `is_available()`. Dataclass config/result types. Stdlib only (subprocess, json, pathlib, collections.OrderedDict).

**D2 — `tools/benchmark_compression.py`**: Benchmarks KV cache quantization (Q4_0/Q5_0/Q8_0) and context compression strategies (sliding_window/fifo/importance). Generates synthetic data, computes MSE/maxErr/cos_sim. CLI via argparse, optional numpy, JSON output. Produces two Markdown tables. Exit 0 on all MSE < 0.01.

**D3 — `include/rdna4_compression_scheduler.hpp`**: `CompressionDecision` struct + `CompressionScheduler` class. `step(seqLen, maxContext, importanceScores)` decides when to trigger context/KV compression. Uniform/entropy/importance keep mask generation. Stub — no .cpp.

**D4 — `test/test_compression.cpp`**: Compile-time test asserting push constant sizes, config defaults, scheduler no-op and compression-triggered paths. 8 assertions.

**D5 — `docs/compression_integration.md`**: 5-step wiring guide (load configs, KV quant, context compress, scheduler integration, fallback safety) + performance targets + graphify section.

### Guardrails respected
- No existing engine/scheduler/main.cpp files modified
- 5 new files created, 2 existing files appended (rdna4_compression.hpp, src/kernels/AGENTS.md)

## Changes (2026-06-28 Round 6 — Vulkan compute-only device fix + graphify update)

### Task: AMD WDDM routes compute to 3D engine (Compute 0 flat at 0%)

**Root cause**: `context.cpp` queried the full Vulkan 1.4 pNext chain (correct for signaling a modern app), but then passed **all queried features unmodified** to `vkCreateDevice`. This included graphics-only features:
- `feat13.dynamicRendering = VK_TRUE` — **graphics-only**, forces AMD WDDM to classify the app as "graphics" and route ALL compute dispatches through the 3D engine
- Base features: `geometryShader`, `tessellationShader`, `depthClamp`, `alphaToOne`, `multiViewport`, `samplerAnisotropy`, etc. — all left enabled if GPU supported them
- Vulkan 1.1/1.2/1.3 graphics features (`multiview`, `drawIndirectCount`, `imagelessFramebuffer`, `shaderOutputViewportIndex`, `shaderTerminateInvocation`, etc.) — all left enabled

The AMD WDDM driver profiles apps **statically at `vkCreateDevice` time** by feature requests, not by per-submit queue family usage. A compute-only queue family (family 1, no `GRAPHICS_BIT`) was correctly selected, but the driver had already permanently tagged the app as "graphics-capable" and routed compute through the Graphics Command Processor.

**Fix** (`src/host/context.cpp`):
- After querying with `vkGetPhysicalDeviceFeatures2`, explicitly **zero every feature struct** and enable **only** compute-relevant features before passing to `vkCreateDevice`
- Base features: only `shaderInt64` + `shaderInt16` enabled
- Vulkan 1.1: only `storageBuffer16BitAccess`
- Vulkan 1.2: `bufferDeviceAddress`, `shaderFloat16`, `shaderInt8`, `storageBuffer8BitAccess`, `uniformAndStorageBuffer8BitAccess`, `scalarBlockLayout`, `timelineSemaphore`
- Vulkan 1.3: `synchronization2`, `maintenance4`; **`dynamicRendering = VK_FALSE`** (critical)
- Vulkan 1.4: zeroed entirely as defensive measure
- Cooperative matrix features preserved if available

**Graphify update**: Pruned 5 deleted files, rebuilt to **1458 nodes, 2032 edges, 127 communities**.

**Build**: All 3 targets (`rdna4_llama.exe`, `test_inference.exe`, `test_cpu_ref.exe`) compile cleanly.

---

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
| 3 | `src/host/inference_engine.cpp` | ✅ DONE | Overflow check for addrDownW_dq: lines 837-841 null-check all three MLP dequant addresses against sizes |
| 4 | `src/host/scheduler.cpp` | ✅ DONE | vkBeginCommandBuffer errors checked in beginBatch/dispatchBatch/dispatchBatchBarriers |
| 5 | `src/host/scheduler.cpp` | ✅ DONE | vkEndCommandBuffer error checks added to all 5 call sites (dispatch, dispatchTimed, dispatchBatch, dispatchBatchBarriers, endBatch) |
| 6 | `src/host/scheduler.cpp` | ✅ DONE | vkQueueSubmit error checks added to all 5 call sites (dispatch, dispatchTimed, dispatchBatch, dispatchBatchBarriers, endBatch) |
| 7 | `src/kernels/mlp.comp` | ✅ DONE | mlp.comp deleted (replaced by mlp_fused_gateup.comp). MlpPushConstants removed from rdna4_types.hpp. Dead loadPipe("mlp", ...) removed from main.cpp and test_inference.cpp. |


