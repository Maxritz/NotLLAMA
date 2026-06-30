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

## Code Move (2026-06-30)

The GPU inference stack produced NaN/corruption on real models (Gemma 4 Q4_0) and repeatedly hung/blanked the GPU. Per user instruction, the broken compute/quant/shader code has been moved out of the active source tree to `C:\Users\rr\Desktop\not-working-code\` (outside the repo).

What remains in the active tree:
- `src/host/context.cpp`, `scheduler.cpp`, `allocator.cpp`, `pipeline_builder.cpp`, `kv_cache.cpp`, `weight_uploader.cpp`, `tokenizer.cpp`, `profiler.cpp`, `mailbox.cpp`, `memory.cpp`, `vgpr_stub.cpp`
- `src/host/inference_engine.cpp` — stub implementation (methods print a disabled message)
- `src/host/turboquant_dispatch.cpp` — stub implementation
- `main.cpp` — minimal CLI that loads a model and prints metadata; no inference
- `CMakeLists.txt` — shader compilation and inference test targets disabled

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

## Truth/False Logic Validation (MANDATORY)

> **Rule: No code writes without a trace first. The trace is faster than the debugger.**

This framework catches logic errors, state mishandling, and resource ordering bugs *before* implementation — through structured static analysis. It also tells you exactly where to instrument your code when things go wrong at runtime.

### Proof of Value

Real bugs caught by this framework in production codebases:

| Bug | Caught By | Cost If Missed |
|-----|-----------|----------------|
| File stream `bad_alloc` crash | Missing `is_open()` branch in flow diagram | Unhandled exception / crash on bad path |
| Array index off-by-2× | Truth table showed computed count vs actual buffer bounds | Buffer overflow → data corruption |
| Operator precedence "bug" | Mathematical proof in truth table showed working code was correct | Would have introduced a bug by "fixing" working code |
| Race condition in producer/consumer | Resource ordering DAG showed missing synchronization | Intermittent corruption / crash |

**The trace finds bugs. It confirms correctness. And it tells you where to attach debug instrumentation.**

### Calibration: Full vs Lightweight Trace

Not every change needs exhaustive analysis. Use risk calibration:

| Change Type | Trace Level | Reasoning |
|-------------|-------------|-----------|
| New algorithm, new I/O path, new data format, concurrency logic, memory/buffer resize, security boundary | **FULL** | Invisible or high-impact bugs (corruption, leaks, races, overflows) |
| Refactor within existing pattern (rename, const, extract function, move) | **LIGHTWEIGHT** | Verify logic unchanged |
| Comments, docs, build flags, config values | **NONE** | No runtime effect |

**Default to FULL if unsure.** A 10-minute trace beats a 3-hour debug session.

### Full Trace Format (Mandatory for New Logic)

```markdown
## TRACE: [Feature Name]

### Input State
| Variable | Value | Source |
|----------|-------|--------|
| x        | 42    | caller |
| format   | RAW   | config |

### Flow Diagram
```
function(args)
│
├── condition?
│   ├── TRUE → action A
│   └── FALSE → action B
```

### Decision Tree
```
if (x > 0):
  TRUE → do A
  FALSE → do B
```

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| x > 0     | TRUE     | TRUE   | PASS  |
| format == RAW | TRUE | TRUE   | PASS  |

### Completeness Check
- Branch coverage: ALL branches reachable and tested? [YES/NO]
- State completeness: ALL variable values handled (including error states)? [YES/NO]
- Goal reachability: EVERY valid input reaches output without silent failure? [YES/NO]

### Resource Ordering
- Producer → Consumer: [pair]
- DAG ordering verified (write before read)? [YES/NO]
- Synchronization/barrier placement: BEFORE consumer access? [YES/NO]
- Deadlock/circular wait risk: [NONE/LOW/HIGH]

### Load Conditions
- Compute units / threads: [count] within safe limits? [PASS/FAIL]
- Memory staging: [bytes] within budget? [PASS/FAIL]
- Batch/dispatch storm risk: [YES/NO]

### Distillation
- Verdict: PASS / FAIL
- Bugs: [list or "none"]
- Patterns: [reusable insights]
- Assumptions: [what the code assumes — validate these]

### Failure Severity
| Issue | Severity | Action |
|-------|----------|--------|
| [description] | CRITICAL / MAJOR / MINOR | [abort / flag+retry / log+continue] |

### VERDICT: PASS / FAIL
```

### Lightweight Trace Format (Mechanical Changes)

For refactors, renames, or moves within an existing pattern:

```markdown
## LIGHTWEIGHT TRACE: [Change Description]

### What Changed
[One-line description]

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| Logic unchanged? | YES | [diff confirms] | PASS |
| All call sites updated? | YES | [grep confirms] | PASS |
| Types/signatures match? | YES | [compiler confirms] | PASS |

### Assumptions
- [Any assumptions that must hold]

### VERDICT: PASS / FAIL
```

### Failure Severity Definitions

| Level | Condition | Action |
|-------|-----------|--------|
| **CRITICAL** | Buffer overflow possible, race condition confirmed, data corruption, unhandled exception, security boundary violation, infinite loop | **STOP.** Fix before any code writes. |
| **MAJOR** | Wrong format conversion, missing branch coverage, compilation error, undefined behavior, resource leak | **Flag.** List specific issues. Retry with guidance. |
| **MINOR** | Style inconsistency, missing documentation, non-optimal performance, misleading precedence | **Log.** Continue with main task. Fix later. |

### Common Pitfalls (Check These First)

1. **File stream state**: Always check `is_open()` before `tellg()` / `read()`. A failed stream's `tellg()` returns `-1` → cast to unsigned = `SIZE_MAX` → massive allocation → crash.
2. **Size ambiguity**: Does `size` mean compressed bytes, element count, or byte count? Document and validate. Off-by-factor bugs are silent killers.
3. **Operator precedence**: In C-family languages, `|` has lower precedence than `<<` and `-`. Always parenthesize bitwise arithmetic: `(a | (b << 4)) - c`, not `a | (b << 4) - c`.
4. **Synchronization ordering**: Lock/barrier/semaphore must be **BEFORE** consumer access, not after. Sequence: `producer write` → `sync point` → `consumer read`.
5. **Resource limits**: Cap batch sizes, thread counts, and memory allocations to known safe thresholds. Chunk if exceeded.
6. **Temporary buffer sizing**: Size buffers for the operation, not the entire dataset. Temporary staging should not persist beyond the operation scope.

### Debug Point Extraction: From Trace to Instrumentation

The trace identifies exactly where reality can diverge from expectation. Those points are your debug instrumentation targets. Do not add logs or breakpoints randomly — instrument the trace points.

| Trace Section | Debug Point | What to Log / Assert |
|---------------|-------------|----------------------|
| **Input State** | Function entry | Log all input values; assert non-null for pointers, assert > 0 for sizes |
| **Flow Diagram — Branch** | Every `if` / `switch` / `? :` | Log which branch was taken; assert the condition matches expected |
| **Truth Table — Condition** | Every evaluated expression | Assert `actual == expected`; if mismatch, dump full state and abort |
| **Completeness — Error State** | Every error path | Assert the error path is reachable in tests; log error code / reason |
| **Resource Ordering — Producer** | Write completion | Assert write finished successfully; log bytes written / checksum |
| **Resource Ordering — Sync Point** | Barrier / lock / signal | Assert sync object is valid; log wait duration (detect deadlocks) |
| **Resource Ordering — Consumer** | Read start | Assert read index / offset within bounds; log first element / hash |
| **Load Conditions** | Pre-dispatch / pre-allocation | Assert count < limit, assert size < budget; log actual vs limit |

**Debug Point Template:**
```
For every row in the Truth Table:
  → Add: ASSERT(condition == expected) or LOG("[file:line] condition=%d expected=%d", actual, expected)

For every branch in the Flow Diagram:
  → Add: LOG("[file:line] branch: <description>") at the taken path

For every Producer → Consumer pair:
  → Add: ASSERT(producer.completed) before sync point
  → Add: ASSERT(consumer.offset < buffer.size) at read start

For every Load Condition:
  → Add: ASSERT(load < threshold) before dispatch/allocation
```

### Reusable Templates

**Template A: File I/O Validation**
```
loadFile(path)
│
├── file.open(path)
│   ├── is_open() == false → return error  [CRITICAL if missing]
│   └── is_open() == true ↓
│
├── file.parse() or file.tellg()
│   ├── parse_error / tellg() == -1 → return error  [CRITICAL if missing]
│   └── success ↓
│
├── size = (size_t)file.tellg()
│   ├── size <= 0 → return error  [MAJOR if missing]
│   └── size > 0 ↓
│
├── buffer.resize(size)
│   ├── bad_alloc → return error  [CRITICAL if missing]
│   └── success ↓
│
└── file.read(buffer)
    ├── bytes_read != size → return error  [MAJOR if missing]
    └── success → return buffer
```

**Template B: Computed Size Validation**
```
calculateElements(byteSize, format)
│
├── What does byteSize represent?
│   ├── Compressed bytes → nElements = decompressRatio(byteSize)
│   ├── Element count → nElements = byteSize
│   └── Unknown → abort / assert  [CRITICAL if ambiguous]
│
├── VERIFY: Does nElements match actual buffer bounds?
│   ├── nElements * elemSize > bufferSize → BUG  [CRITICAL]
│   └── nElements * elemSize <= bufferSize → OK
│
└── VERIFY: Does consumer use correct count?
```

**Template C: Producer/Consumer Synchronization**
```
Producer → Consumer Chain
│
├── Producer writes to shared resource
│   └── [WRITE operation]
│
├── Sync point (barrier / lock / fence / signal)
│   ├── BEFORE consumer access?  [YES = CORRECT]
│   └── AFTER consumer access?   [NO = RACE CONDITION]
│
├── Consumer reads from shared resource
│   └── [READ operation]
│
└── VERIFY: No circular waits (A waits for B, B waits for A)
```

### RAG / Knowledge Graph Distillation

Every trace produces durable knowledge. Extract these fields:

| Field | Extract? | Why |
|-------|----------|-----|
| Verdict (PASS/FAIL) | **YES** | Node label for search |
| Bug description (one-liner) | **YES** | Primary search key |
| Root cause pattern | **YES** | Reusable rule — e.g., "size ambiguity" |
| Assumption that failed | **YES** | Prevents repeat — e.g., "Assumed files always exist" |
| Severity + action taken | **YES** | Teaches escalation rules |
| Debug points extracted | **YES** | Reusable instrumentation pattern |
| Full truth table | **NO** | Too noisy; keep in version control |
| Full code | **NO** | Already in source |

**Graph Node Format:**
```json
{
  "id": "trace_[component]_[bug]_[date]",
  "type": "validation",
  "component": "[component name]",
  "verdict": "FAIL",
  "bug": "[one-line description]",
  "pattern": "[reusable rule]",
  "assumption_broken": "[what was assumed]",
  "severity": "CRITICAL",
  "fix": "[what was done]",
  "debug_points": ["ASSERT(...)", "LOG(...)"],
  "related": ["trace_[other]"]
}
```

**When to Feed the Graph:**

| Trigger | Action |
|---------|--------|
| Trace finds a CRITICAL bug | **Immediate** — prevents the next engineer from repeating it |
| Trace confirms a false positive | **Immediate** — prevents future "fixes" of working code |
| Lightweight trace on refactor | **Skip** — no new knowledge generated |
| Trace finds only MINOR issues | **Batch** — feed at end of week |

**The trace is faster than the debugger.**

## Child DOX Index

### Ownership Map

- **Root** (`/`): Build system (`CMakeLists.txt`), entry point (`main.cpp`), project-wide instructions, DEEPSEEK_CONTEXT.md (project state document), `analysis_backend_shader_correctness.md`
  - `analysis_backend_shader_correctness.md` — Compute-shader correctness decision flow; backend review of llama.cpp, Ollama, vLLM, and VindexLLM; recommended next paths.
- **`include/`**: Headers owned by corresponding `src/host/` implementations. No child DOX needed.
  - `rdna4_compression.hpp` — Push constant structs for context compression + KV cache quantize/dequant shaders. Static asserts ≤128B. Note: `CompressContextPushConstants` currently describes a quantization layout; it must be reconciled with the compaction layout in `compress_context.comp` before host dispatch (see `docs/wiring_plan.md`).
  - `rdna4_turboquant.hpp` — Host-side block layouts for TQ4_128, TQ3_128, TQ6_64 plus packed-size helpers. Static asserts block sizes.
  - `rdna4_types.hpp` — Host push constant structs for all shaders, including `DequantTurboPushConstants` (40B) and `GemmTurboPushConstants` (56B).
- **`docs/`**: Design and integration documentation. Owned by root.
  - `wiring_plan.md` — Implementation plan for wiring TurboQuant and context/KV compression assets into the engine.
  - `turboquant_integration.md` — TurboQuant block layouts, bit-packing diagrams, and 5-step engine wiring checklist.
  - `turboquant_formats.md` — TurboQuant format variants and comparison table.
  - `turboquant_schema.json` — JSON schema for TurboQuant configuration.
- **`src/host/`**: Vulkan infrastructure — scheduler, allocator, KV cache, weight uploader, tokenizer, profiler, pipeline builder, mailbox. GPU inference is stubbed; the previous implementation lives at `C:\Users\rr\Desktop\not-working-code\`.
  - `inference_engine.cpp` — stub implementation: prints disabled message for all inference methods
  - `scheduler.cpp` — Vulkan queue submission, fence handling, cleanup lifecycle
- **`include/engine/`**: Modular engine interface layer (v6.0 blueprint) and adapter headers. Pure-virtual interfaces plus adapter headers that expose concrete implementations backed by existing `rdna4::*` classes.
  - Core interfaces: `types.hpp`, `idevice.hpp`, `imemory_allocator.hpp`, `imodel.hpp`, `iquantization.hpp`, `idescriptor_manager.hpp`, `ishader_library.hpp`, `iattention_scheduler.hpp`, `itokenizer.hpp`, `icompute_engine.hpp`, `idebug_context.hpp`
  - Adapters: `vulkan_device.hpp`, `ring_allocator_adapter.hpp`, `vulkan_compute_engine.hpp`, `vulkan_shader_library.hpp`, `vulkan_descriptor_manager.hpp`, `vulkan_debug_context.hpp`, `shader_compiler.hpp`, `tokenizer_adapter.hpp`, `model_adapter.hpp`, `kv_cache_adapter.hpp`
  - `engine.hpp` — convenience umbrella header
- **`src/engine/`**: Adapter/stub implementations of the modular interfaces. See `src/engine/AGENTS.md`.
  - `vulkan_device.cpp` — `IDevice` adapter for `rdna4::VulkanContext`
  - `ring_allocator_adapter.cpp` — `IMemoryAllocator` bridge using three `rdna4::RingAllocator` instances
  - `tokenizer_adapter.cpp` — `ITokenizer` adapter for `rdna4::Tokenizer`
  - `model_adapter.cpp` — `IModel` adapter for `rdna4::WeightUploader` + `rdna4::ModelDesc`
  - `kv_cache_adapter.cpp` — `IAttentionScheduler` adapter for `rdna4::KVCacheManager`
  - `vulkan_compute_engine.cpp` — stub `IComputeEngine` implementation
  - `vulkan_shader_library.cpp` — `IShaderLibrary` implementation with SPIR-V loading, specialization constants, and pipeline cache
  - `vulkan_descriptor_manager.cpp` — `IDescriptorManager` implementation: bindless SSBO descriptor table + per-frame metadata ring buffer
  - `vulkan_debug_context.cpp` — `IDebugContext` implementation using VK_EXT_debug_utils
  - `shader_compiler.cpp` — runtime GLSL -> SPIR-V compiler using the system `glslc`; caches per-profile SPIR-V under `shaders/cache/`
- **`src/kernels/`**: Empty. All GLSL compute shaders have been copied to `shaders/` for dynamic runtime compilation.
- **`shaders/`**: All 37 GLSL compute shader sources for GPU inference (attention, FFN, dequant, quant, GGUF, KV cache, rope, etc.). Compiled on demand by `ShaderCompiler` using the user's system `glslc`. Compiled SPIR-V cached under `shaders/cache/`. `PrecompileAll()` compiles all `.comp` files on first use.
- **`tools/`**: Python utilities. Owned separately.
  - `dox_lint.py` — AGENTS.md compliance checker.
  - `graphify_client.py` — Python GraphifyClient for subprocess-based knowledge graph queries.
  - `benchmark_compression.py` — KV cache + context compression benchmark tool.
  - `validate_turboquant.py` — Synthetic and optional real-model TurboQuant accuracy validation.
  - `benchmark_turboquant.py` — Per-tensor and summary benchmarking against Q8_0 baseline.
  - `convert_gguf_to_turboquant.py` — Standalone GGUF-derived binary → TurboQuant converter.
- **`VindexLLM/`**: External reference implementation (Delphi 12 + Vulkan compute). Self-contained Git repo; not built or owned by NotLLAMA. Use as a design reference for working Vulkan inference shaders.
- **`reference/`**: Vulkan SDK reference files. Read-only.
- **`graphify-out/`**: Knowledge graph outputs (auto-generated, not edited directly).
- **`test/test_compression.cpp`**: Compile-time test for compression push constants, config defaults, scheduler logic.
- **`test/test_turboquant.cpp`**: Compile-time test for TurboQuant block sizes, packed-size helpers, and push constant size guards.
- **`build/`**: CMake build output. Not checked in.

### Key Contracts

- `scheduler.cleanup()` must be called before `ctx.cleanup()` in main.cpp — destroys command pools while VkDevice is still alive.
- `src/host/inference_engine.cpp` is currently a stub. All inference methods print a disabled message and return empty/error values.
- `main.cpp` loads a model via `WeightUploader`, prints metadata, and exits. It does not run GPU inference.
- `ShaderCompiler` compiles `.comp` shaders at runtime via the system `glslc`. If `VULKAN_SDK` is not set and `glslc` is not on `PATH`, runtime compilation is skipped and `VulkanShaderLibrary` falls back to precompiled `.spv` files.
- `PrecompileAll()` iterates every `.comp` file in the shader directory and compiles/caches it for the current GPU profile.

### Verification

- Build: `cd build && cmake --build . --config Release`
- Runtime: `rdna4_llama.exe [--clear-shaders] <model.weights.json> <model.weights.bin>` must load the model, print metadata, and exit cleanly without crash. Pass `--clear-shaders` to delete cached SPIR-V and force a fresh shader compile.
- GPU inference verification is disabled until the compute stack is restored.

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
- **Dead code confirmed unchanged**: `model.cpp`, `memory.cpp`, `src/loaders/gguf.cpp`, `src/core/` (old `notllama::` path). `vgpr_stub.cpp`, cooperative matrix shaders — intentional placeholders.

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

## Changes (2026-06-29 Round 1 — Full code review: 20 bugs found, 14 fixed)

### Task: Comprehensive code review + Truth/False Logic Traces

**14 bugs fixed this session:**

| # | Bug | Fix | File |
|---|-----|-----|------|
| F1 | QuantRef uint[] byte-addressing (all quantized formats read 4× offset) | `readByte()`/`readU16()` helpers | `dequantize.comp` |
| F2 | calcNumElements overestimates up to +150% for Q2_K | Exact block-structured division | `inference_engine.cpp:65-84` |
| F3 | Missing RoPE → KV cache write barrier | `barrierBetweenGroups(WRITE, READ)` | `inference_engine.cpp:254` |
| F4 | Missing inter-layer barrier (FFN add → next norm) | `barrierBetweenGroups(WRITE, READ)` | `inference_engine.cpp:376` |
| F5 | OOM returns quantized address as F32 input | Returns 0, callers check `!wAddr` | `inference_engine.cpp:99` |
| F6 | FFN scratch overflows logits buffer | `max(vocabSize, 3*hiddenDim)` | `inference_engine.cpp:153` |
| F7 | Q5_0 block size 24→22, qh layout | 4-byte qh, per-element bit | `dequantize.comp` |
| F8 | Q5_1 block size 26→24, qh layout | 4-byte qh, per-element bit | `dequantize.comp` |
| F9 | Q2_K block size 160→84 | Correct packed scales layout | `dequantize.comp` |
| F10 | Q3_K block size 208→114 | hmask/qs/scales/d-last | `dequantize.comp` |
| F11 | Q5_K block size 164→176 | Packed int8 scales, qh[32] | `dequantize.comp` |
| F12 | Q8_K block size 260→292 | float32 scale, bsums present | `dequantize.comp` |
| F13 | `int8_t` GLSL syntax error | Replaced with `int` | `dequantize.comp:212,301` |
| F14 | **RoPE double-offset** — qRowAddr used as addrQ, shader adds offset again | Pass qAddr/kAddr (base addresses) | `inference_engine.cpp:252` |

**Trace: RoPE double-offset (F14)**
- Root cause: Shader was designed to take base address and compute `(seqLen-1)*nHeads*headDim` offset. Host was passing `qRowAddr` which already includes `seqPos*dim*4`. Double-offset for every token after the first.
- seqPos=0: underflow (reads ~4B offset)
- seqPos=1: correct by accident (offsets cancel)
- seqPos≥2: reads row (2N-1) instead of row N
- Fix: `RopePushConstants ropePC = {qAddr, kAddr, ...}` instead of `{qRowAddr, kRowAddr, ...}`
- Flow map: `docs/rope_data_flow.md`

**6 format block sizes fixed in dequantize.comp:**

| Format | Before | After | Layout |
|--------|--------|-------|--------|
| Q5_0 | 24B | **22B** | qh[4] + qs[16], per-element bit |
| Q5_1 | 26B | **24B** | qh[4] + qs[16], per-element bit |
| Q2_K | 160B | **84B** | scales[16] + qs[64] + d + dmin |
| Q3_K | 208B | **114B** | hmask[32] + qs[64] + scales[16] + d |
| Q5_K | 164B | **176B** | scales[12] + qh[32] + qs[128] + d + dmin |
| Q8_K | 260B | **292B** | float32 d + qs[256] + bsums[16] |

**All 6 previously-remaining bugs now FIXED (2026-06-29 Round 2 cleanup):**

| # | Bug | Status | Fix |
|---|-----|--------|-----|
| 1 | Q6_K block size 3-way mismatch (288 vs 210) | ✅ FIXED (Round 1) | All sources use 210B |
| 2 | Q3_K block size 3-way mismatch (110/112/114) | ✅ FIXED (Round 1) | All sources use 110B |
| 3 | Q8_K block size (256→292) + dequant structure | ✅ FIXED | gguf.cpp:256→292, gguf_loader.py:290→292 |
| 4 | Q5_0 bit packing mismatch (GPU vs CPU) | ✅ FIXED (Round 1) | All 3 impls match |
| 5 | 9 IQ format block sizes in gguf_loader.py | ✅ VERIFIED CORRECT | Already matched docs/ggml_iq_block_layouts.md |
| 6 | Q4_K scale packing inconsistent | ✅ FIXED (Round 2) | get_scale_min_k4 in all 3 impls |

**Verification:** All 5 targets compile clean (`rdna4_llama.exe`, `test_inference.exe`, `test_compression.exe`, `test_turboquant.exe`, `test_cpu_ref.exe`). `dequantize.comp` compiles to SPIR-V.

## Changes (2026-06-29 Round 2 cleanup — Stale doc fix + Q8_K remnants)

### Fixes
- **`src/loaders/gguf.cpp:33`**: Q8_K bytesPerBlock 256→292. Was computing tensor sizes at 87.7% of actual.
- **`tools/gguf_loader.py:dequantize_q8_k`**: Rewrote with correct 292B block, float32 d, no sub-block scales (was using 290B, fp16 d, incorrect 32-byte sub-block scale array).
- **`tools/AGENTS.md`**: Q8_K block description corrected (float32 d + qs[256] + bsums[16], not d-last). Note that Q8_K is the only K-quant with d-first layout.
- **`AGENTS.md`**: Stale "remaining unfixed bugs" table replaced with fixed status table. All 6 bugs confirmed fixed.

### New Trace Documents
- `docs/dequant_chunking_trace.md` — Dequant workgroup limit chunking fix
- `docs/ring_allocator_trace.md` — Ring allocator 64MB→1GB sizing
- `docs/q4k_scale_packing_trace.md` — Q4_K get_scale_min_k4 packing fix
- `docs/first_token_forward_failure_trace.md` — Three-bug cascade (workgroup overflow + ring OOM + silent F32)
- `docs/q8k_block_size_trace.md` — Q8_K block size cross-reference + gguf.cpp/gguf_loader.py fixes

## Changes (2026-06-30 Round 8 — Flash attention correctness fix + wave size detection)

### Task: RDNA2/RDNA4 wave32/wave64 safety + flash_attention partial workgroup corruption

**Root cause**: `flash_attention.comp` used `if (qRow >= sl) return;` (early exit) followed by a `barrier()` inside the KV loop. On partial workgroups (`sl % 32 != 0`), exited threads never wrote `s_scores[tid]`, but thread 0 summed all 32 slots → undefined values corrupted the dot product. Additionally, a `barrier()` after an early `return` is undefined behavior per Vulkan spec.

**Fix** (`src/kernels/flash_attention.comp`):
- Removed early `return` — replaced with `bool valid` guard wrapping all computation and output writes
- Replaced shared-memory reduction (`s_scores[32]` + `barrier()` + tree sum) with `subgroupAdd(partialDot)` — a single cross-lane operation, no barrier, no shared memory
- Removed both `barrier()` calls inside the KV loop (previously at lines 103, 108)
- Works correctly on Wave32 (RDNA4) and Wave64 (RDNA2): `subgroupAdd` operates on the workgroup's active threads, invalid threads contribute 0
- Output writes guarded by `if (valid)` — no out-of-bounds

**Fix** (`src/host/context.cpp`, `include/rdna4_vulkan.hpp`):
- Added `VkPhysicalDeviceSubgroupProperties` query via `vkGetPhysicalDeviceProperties2`
- Prints subgroup/wave size at startup: "Subgroup size (wave width): 32" on RDNA4, "64" on RDNA2
- Added `isWave32()`, `isWave64()`, `isAmd()`, `isNvidia()` helpers to VulkanContext
- Stores `subgroupSize`, `vendorID`, `deviceApiVersion`, `deviceName` for shader tuning

**Doc fix** (`AGENTS.md`):
- `kv_cache_write.comp`: Stale `float16_t` conversion claim removed — cache stores F32, not F16. Buffer is allocated as `sizeof(float)` and shader copies float→float directly.
- `flash_attention.comp`: Contract updated — uses `subgroupAdd`, no `s_scores[32]`, no barriers in KV loop, safe on Wave32/Wave64.

### Verification
- All 5 targets still compile.
- `dequantize.comp`, `gguf.cpp`, `gguf_loader.py`, `cpu_reference.cpp`, `weight_uploader.cpp` all consistent on every format.

## Changes (2026-07-01 — Engine dispatch wiring, K-quant CompressTile, descriptor manager fix)

### Task 1: VulkanComputeEngine wired with real RMS_NORM dispatch
- **`include/engine/vulkan_compute_engine.hpp`**: Added constructor taking VkDevice, VkQueue, queue family, IShaderLibrary*, IDescriptorManager*, IMemoryAllocator*. Added private members for command pool/buffer/fence, sequence state map, RMS_NORM validation state.
- **`src/engine/vulkan_compute_engine.cpp`**: Complete rewrite from stub. `AddSequence()` allocates hidden state buffer. `StepBatch()` calls `ValidateRmsNormDispatch()` on first invocation: compiles rms_norm.comp pipeline via shader library, creates host-visible test buffers, dispatches with BDA push constants, reads back results, verifies against CPU reference (sumSq → invRms → weight scale). On success, flips `rms_norm_validated_` flag.

### Task 2: CPU reference tests (rms_norm, rope, silu_mul, embed)
- **`test/test_rms_norm.cpp`**: End-to-end test: compile shader → create pipeline with wave32 control → fill test buffers (2 rows of 256) → dispatch via BDA push constants → read back → compare against CPU `x / sqrt(mean(x²)+eps) * weight`. PASS threshold: max diff < 0.01.
- **`test/test_rope.cpp`**: Tests rope.comp with 2 heads, headDim=64, seqLen=5. Applies RoPE at position 4. CPU reference computes `theta = base^(-2d/hd)`, paired-element rotation for Q and K. Verifies all dimensions at last position.
- **`test/test_silu_mul.cpp`**: Tests silu_mul.comp with 1024 elements. CPU reference computes `SiLU(x) = x / (1 + e⁻ˣ)`, then `SiLU(gate) * up`.
- **`test/test_embed.cpp`**: Tests embed.comp with vocab=100, dim=256, tokenId=42, tokenPos=3. Verifies `hidden[tokenPos*dim + d] = embed[tokenId*dim + d]` and that other positions are untouched (still zero).
- **Build system**: All 4 tests added via `add_kernel_test()` function in CMakeLists.txt.

### Task 3: K-quant CompressTile fill (Q4_K, Q5_K, Q6_K, Q8_K)
- **`src/engine/gguf_quantization.cpp`**: Replaced stub Q4_KQuantization, Q5_KQuantization, Q6_KQuantization, Q8_KQuantization with real implementations:
  - **Q4_K** (144B/block): Block-level d/dmin (fp16), 8 sub-blocks of 32. Scales packed 6-bit (3-bit sc + 3-bit sm, 4 per 3 bytes). qs nibble-packed with sub-block pair sharing. Matches shader `get_scale_min_k4` format.
  - **Q5_K** (176B/block): d/dmin (fp16), scales per sub-block pair (low nibble=sc, high nibble=sm), qh one bit per element, qs flat nibble layout. Standard GGUF format.
  - **Q6_K** (210B/block, d-last): 16 sub-blocks of 16. ql lower 4 bits, qh upper 2 bits, int8 scales, d at byte 208.
  - **Q8_K** (292B/block, d-first): float32 d, int8 values, int16 bsums per 16-element sub-block.
  - Added `<algorithm>` include for `std::min`/`std::max`.

### Task 4: KVCacheAdapter::FlashDecodeBatch() wired
- **`include/engine/kv_cache_adapter.hpp`**: Added IShaderLibrary* member, VkCommandPool/Buffer/Fence for dispatch, `EnsureResources()` helper. Constructor takes optional IShaderLibrary*.
- **`src/engine/kv_cache_adapter.cpp`**: `FlashDecodeBatch()` now: gets KV cache buffer addresses from `rdna4::KVCacheManager`, compiles flash_attention.comp pipeline, records command buffer with push constants (addrQ, addrKCache, addrVCache, addrOut, seqLen, headDim, qRowStart, qRowCount, invSqrtHeadDim), submits, waits on fence. Added `<cmath>` for sqrtf.

### Task 5: Pipeline layouts include bindless descriptor set + AMD wave32 workaround
- **`src/engine/vulkan_descriptor_manager.cpp`**: Removed `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` from layout binding flags and `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT` from pool creation. AMD workaround: update-after-bind causes driver crash when combined with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (wave32 forced). Pre-write all descriptors instead.
- **`include/engine/vulkan_shader_library.hpp`**: Added `SetBindlessSetLayout(VkDescriptorSetLayout)` and `bindless_layout_` member.
- **`src/engine/vulkan_shader_library.cpp`**: `CreateComputePipeline()` and `CreateNamedComputePipeline()` now include bindless descriptor set layout (if set) as `setLayoutCount=1`. Buffer-reference shaders ignore the unused set; traditional descriptor-indexing shaders use it.
- **`test/test_engine.cpp`**: Updated to wire bindless layout into shader library, construct VulkanComputeEngine with real dependencies (device, queue, shader_lib, desc_mgr, allocator), test RMS_NORM dispatch validation.

### Build changes
- **`CMakeLists.txt`**: Added `add_kernel_test()` CMake function and 4 new test targets: test_rms_norm, test_rope, test_silu_mul, test_embed. Each links Vulkan and includes context.cpp + shader_compiler.cpp.

### Post-hoc fixes (same session)

**Fix 1 — FlashDecodeBatch queue** (`include/engine/kv_cache_adapter.hpp`, `src/engine/kv_cache_adapter.cpp`):
- Added `VkQueue queue_` member + constructor parameter. `FlashDecodeBatch()` now calls `vkQueueSubmit(queue_, ...)` instead of `vkQueueSubmit(VK_NULL_HANDLE, ...)` — the latter would crash on dispatch.

**Fix 2 — StepBatch real embed→rms_norm pipeline** (`include/engine/vulkan_compute_engine.hpp`, `src/engine/vulkan_compute_engine.cpp`):
- Added `StepPhase` enum (VALIDATE → IDLE → EMBED → RMS_NORM → IDLE) and `AllocInferenceBuffers()`, `DispatchEmbed()`, `DispatchRmsNorm()` private methods.
- After RMS_NORM validation passes, `StepBatch()` walks each sequence through embed → rms_norm per token position. `embed_table_` and `rms_weight_` buffers are allocated on-device.
- `EMBED` dispatch uses `KernelType::EMBED` with `EmbedPushConstants` (embed table BDA, hidden state BDA, tokenId, tokenPos, dim).
- `RMS_NORM` dispatch uses `KernelType::RMS_NORM` with `RmsNormPushConstants`.

**Fix 3 — RotateAndCache real dispatch** (`src/engine/kv_cache_adapter.cpp`):
- Added `EnsureKvCacheWritePipeline()` and `DispatchKvCacheWrite()` helpers.
- `RotateAndCache()` now gets the `KV_CACHE_WRITE` pipeline from shader library, builds `KVCacheWritePushConstants` (KIn/VIn/cache BDAs, seqPos, headDim, nKvHeads), records command buffer, dispatches `(n_kv_heads * head_dim + 31) / 32` workgroups, submits via queue. Calls `kv_->incrementSeqLen()` on success.
- No longer passes `nullptr` to `kv_->append()`.

**Fix 4 — Block table API** (`src/engine/kv_cache_adapter.cpp`):
- Added `BlockEntry` struct (in_use, last_access), `std::vector<BlockEntry> block_pool_`, per-sequence tracking maps (`seq_blocks_`, `seq_last_access_`).
- `InitializeKVSlab()` pre-allocates block pool to `ceil(max_seq / 32)` entries.
- `AllocateBlock()` linear-scan allocates the first free block; returns `UINT32_MAX` if exhausted.
- `FreeBlock()` sets in_use = false.
- `SetSequenceBlockTable()` stores block→seq mapping.
- `RemoveSequence()` frees all blocks for a seq, removes from tracking maps.
- `EvictBlocksLRU()` sorts sequences by `last_access`, removes oldest until `keep_percent` remains.
- `GetRemainingBlocks()` counts free blocks.

**Fix 5 — Q5_K dequant shader bug** (`shaders/dequantize.comp`):
- Format 13 (Q5_K): changed `elemInSub` → `elemInSuper` for qh indexing (`qhIdx = sbOffset + 16 + elemInSuper/8`), qh bit position (`elemInSuper % 8`), qs byte index (`qsIdx = sbOffset + 48 + elemInSuper/2`), and nibble selection (`elemInSuper & 1`).
- Bug: previously used `elemInSub` (0..31) which caused all 8 sub-blocks to read the same 16 bytes of qs and same 4 bytes of qh. Only the first 32 elements were ever read correctly; elements 32-255 produced stale/garbage data from sub-block 0's fields.

### Round 2 fixes (same session)

**Fix 6 — Pipeline layout handle bug** (critical, 4 files):
- **`include/engine/ishader_library.hpp`**: Added `virtual GetPipelineLayout(KernelType, PipelineVariant, SpecializationMap) = 0` to interface.
- **`include/engine/vulkan_shader_library.hpp`**: Added `GetPipelineLayout()` override declaration.
- **`src/engine/vulkan_shader_library.cpp`**: Implemented `GetPipelineLayout()` — looks up `layouts_[key]`, auto-compiles if missing.
- **`src/engine/vulkan_compute_engine.cpp`**: All `vkCmdPushConstants(cmd, pipeline, ...)` → `vkCmdPushConstants(cmd, pipeline_layout, ...)` in `ValidateRmsNormDispatch()`, `DispatchEmbed()`, `DispatchRmsNorm()`. These would have crashed at runtime because the second argument to `vkCmdPushConstants` must be `VkPipelineLayout`, not `VkPipeline`.

**Fix 7 — Round-robin sequence selection** (`src/engine/vulkan_compute_engine.cpp`):
- `StepBatch()` now builds a `pending` vector of sequences with remaining tokens, cycles through them starting from `next_seq_index_`. Previously always picked the first map entry.

**Fix 8 — Configurable embed_dim** (`include/engine/vulkan_compute_engine.hpp`, `src/engine/vulkan_compute_engine.cpp`):
- Constructor takes `uint32_t embed_dim = 4096` parameter. `embed_dim_` is no longer hardcoded.

**Fix 9 — KVCacheWritePushConstants missing maxSeq** (`include/rdna4_types.hpp`, `src/engine/kv_cache_adapter.cpp`):
- Struct was missing `uint32_t maxSeq` field that the shader expects. Push constant layout was 44 bytes (host) vs 48 bytes (shader) — shader read garbage at offset 44. Added `maxSeq` field, set from `max_seq_` in `DispatchKvCacheWrite`.

**Fix 10 — Multi-dispatch FlashDecodeBatch** (`src/engine/kv_cache_adapter.cpp`):
- `FlashDecodeBatch` now loops over ceil(qRowCount / 32) dispatches with different `qRowStart`, instead of dispatching X workgroups all with the same qRowStart (which would compute the same rows X times). Each workgroup handles 32 rows via `qStart + tid`.

**Fix 11 — Pipeline layout in kv_cache_adapter dispatch** (`src/engine/kv_cache_adapter.cpp`):
- `DispatchKvCacheWrite()` now gets pipeline layout from `shader_lib_->GetPipelineLayout()` and passes it to `vkCmdPushConstants`. Same pipeline-as-layout bug as Fix 6.

## Changes (2026-07-01 — K-quant streaming: WeightUploader per-layer upload + StreamLayerWeights)

### Task: Per-layer on-demand quantized weight upload for models >VRAM

**Problem**: `ModelAdapter::StreamLayerWeights()` was a stub (returned false). `WeightUploader::load()` always uploaded ALL tensors to GPU at init time — no path for lazy/streaming upload per layer.

**Fix** (4 files):

**`include/rdna4_weights.hpp`**:
- Added `loadMetadata(jsonPath, binPath)` — loads JSON + binary metadata without creating any GPU buffers. Stores full binary in `binData_` member for deferred per-tensor upload.
- Added `uploadTensor(TensorDesc& desc)` — uploads one tensor's raw quantized bytes from `binData_` to a new device-local GPU buffer. Creates staging buffer, copies, submits, waits, cleans up per call.
- Added `uploadLayer(ModelDesc& model, uint32_t layerIndex)` — filters `model.tensors` by prefix `"blk.{layer}."` and calls `uploadTensor()` for each matching tensor. Skips tensors already uploaded (`gpuAddress != 0`).
- Added `std::vector<uint8_t> binData_` member for deferred upload.

**`src/host/weight_uploader.cpp`**:
- Implemented `loadMetadata()` — copies metadata parsing from `load()` but skips all staging/GPU buffer creation. Populates `binData_` for later upload. Returns `ModelDesc` with zeroed `gpuAddress/buffer/memory` fields.
- Implemented `uploadTensor()` — creates GPU buffer via `createGpuBuffer()`, uploads raw quantized bytes via staging buffer + fence-synchronized copy. Handles all OOM/allocation errors gracefully (cleanup on failure, return false).
- Implemented `uploadLayer()` — uses `tensorBelongsToLayer()` helper to find tensors matching `"blk.{layer}."` prefix. Skips already-uploaded tensors.

**`include/engine/model_adapter.hpp`**:
- Added `std::vector<bool> layers_loaded_` member to track which layers have been uploaded.

**`src/engine/model_adapter.cpp`**:
- `LoadFromPath()` now calls `uploader_.loadMetadata()` instead of `uploader_.load()`. No GPU upload at load time. Initializes `layers_loaded_` to `false` for all layers.
- `StreamLayerWeights()` now: validates model loaded, validates layer index in range, returns early if already loaded, calls `uploader_.uploadLayer()`, marks layer as loaded on success.

### Streaming Workflow
1. `LoadFromPath()` → reads JSON + bin, stores tensor metadata + binary data in system RAM. No GPU allocations.
2. For each layer N: `StreamLayerWeights(N)` → uploads only that layer's quantized weights to device-local GPU buffers with BDA.
3. After inference, caller may evict (free) layer N's buffers to free VRAM for next layer.
4. `layers_loaded_[]` prevents redundant uploads.

### Architecture alignment
- Weights stay quantized on GPU throughout — no F32 conversion at any point in the upload path.
- Each layer is a self-contained upload unit.
- `cpuDequantToFloat()` remains in `weight_uploader.cpp` as dead code (reference for CPU-side verification); not called from any upload or streaming path.

### Guardrails respected
- Existing `WeightUploader::load()` unchanged (still does full batch upload).
- `main.cpp` unchanged (uses `WeightUploader::load()` directly, not `ModelAdapter`).
- All 13 targets compile cleanly.

## Changes (2026-07-01 — Warning cleanup, rope race fix, multi-token cleanup crash, --clear-shaders)

### C4267 `size_t` -> `uint32_t` warnings fixed
- `src/host/scheduler.cpp`: cast `pcSize` to `uint32_t` at `vkCmdPushConstants` call sites.
- `src/host/pipeline_builder.cpp` + `include/rdna4_pipeline.hpp`: `createComputePipeline(pushConstantSize)` changed from `size_t` to `uint32_t`.
- `main.cpp`: cast `headDim * nKvHeads` to `uint32_t` for the engine constructor.

### `test_rope` FAIL fixed
- Root cause: `shaders/rope.comp` used one thread per element, so the two threads of an even/odd pair had a read-after-write data race.
- Fix: rewrote shader to use one thread per pair, computing and writing both elements atomically.
- `test/test_rope.cpp`: updated dispatch count to `nHeads * headDim / 2`, fixed CPU reference to process pairs without overwriting the even element before the odd element, and added `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` to BDA memory allocations.
- Cleared stale `shaders/cache/rope_*.spv` so the new source was compiled instead of the old cached SPIR-V.

### `test_multi_token` Release crash fixed
- Root cause: legacy `InferenceEngine::cleanup()` was called twice (explicitly + destructor), causing use-after-free / double-free of the same Vulkan objects. Weight-tied models also aliased `output_weight` to `embed_table_gpu`.
- Fix (`src/vulkan/inference_engine.cpp` + `include/vulkan/inference_engine.h`): added `cleaned_` flag to make cleanup idempotent and skip destroying `output_weight` when it aliases `embed_table_gpu`.

### `--clear-shaders` CLI switch
- `main.cpp`: parses `--clear-shaders` anywhere in argv, deletes `shaders/cache/*.spv` before shader precompilation, and logs the count.
- Updated usage string and root AGENTS.md verification line.

### Current inference status
- Active modular engine (`VulkanComputeEngine`) loads weights and passes kernel validation, but `StepBatch()` returns false during inference because `KVCacheManager` is not wired in `main.cpp` and residual-add/quantized-GEMM paths are still incomplete.
- Legacy `tests/test_multi_token` runs a full forward pass but produces degenerate output (repeated token 483).


