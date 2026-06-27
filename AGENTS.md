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

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

### Ownership Map

- **Root** (`/`): Build system (`CMakeLists.txt`), entry point (`main.cpp`), project-wide instructions, DEEPSEEK_CONTEXT.md (project state document)
- **`include/`**: Headers owned by corresponding `src/host/` implementations. No child DOX needed.
- **`src/host/`**: Inference engine, scheduler, allocator, KV cache, weight uploader, tokenizer, profiler, pipeline builder, mailbox
  - `inference_engine.cpp` — main inference loop, GPU sampling, CPU fallback
  - `scheduler.cpp` — Vulkan queue submission, fence handling, cleanup lifecycle
- **`src/kernels/`**: 13 GLSL compute shaders compiled to SPIR-V via glslc. No child DOX needed.
- **`tools/`**: Python weight converter, GGUF inspector, RAG tool. Owned separately.
- **`reference/`**: Vulkan SDK reference files. Read-only.
- **`graphify-out/`**: Knowledge graph outputs (auto-generated, not edited directly).
- **`build/`**: CMake build output. Not checked in.

### Key Contracts

- `scheduler.cleanup()` must be called before `ctx.cleanup()` in main.cpp — destroys command pools while VkDevice is still alive.
- `TopKPushConstants` includes `addrScratch` (vocabSize floats for probabilities), `topP`, `seed`. The topk shader does temperature-scaled softmax (parallel reduce-max + reduce-sum), then single-threaded (tid==0) bounded min-heap top-K scan over all vocabSize probabilities, top-P nucleus filtering, and PCG random weighted sampling in one dispatch. O(vocabSize log K) heap ops, single-threaded — no data race.
- `kv_cache_write.comp`: KInRef/VInRef are `float` (not `float16_t`). Explicit `float16_t()` conversion on cache write. Cache stays float16.
- `rope.comp`: Two-pass pattern — all threads read paired Q/K values into registers before any thread writes, eliminating data race.
- `flash_attention.comp`: Tiled — each thread owns `headDim/32` elements. Uses `myQ[8]`, `myAcc[8]`, `s_scores[32]` shared. Online softmax over KV tiles. Requires subgroup shuffle/ballot extensions.
- `kernel_entry.comp`: MailboxRef is `coherent`. Spin-wait loop calls `memoryBarrierBuffer()` before reading `tokenReady`. `computeLogits()` writes all logits unconditionally.
- `inference_engine.cpp`: Ring allocator calls checked for zero (overflow). CPU fallback guards `cpuLogits` null before `sampleArgmax`.
- Dequant dispatches are capped at 1M workgroups per dispatch (chunking). Embedding cache is persistent DEVICE_LOCAL — do not re-dequantize per token.

### Verification

- Build: `cd build && cmake --build . --config Release`
- Shader copy: `Copy-Item build\shaders\*.spv build\Release\shaders\`
- Runtime: `rdna4_llama.exe` must not crash (VK_ERROR_DEVICE_LOST or 0xC0000005)