original idea:

es. In fact, I'd structure it similarly to how highly optimised inference libraries are organised, but with Vulkan as the backend and AMD RDNA as the primary target.

FastVK-LLM
│
├── runtime/
│   ├── device.cpp
│   ├── queues.cpp
│   ├── scheduler.cpp
│   ├── command_pool.cpp
│   ├── pipeline_cache.cpp
│   ├── descriptor_cache.cpp
│   ├── barrier_optimizer.cpp
│   ├── profiler.cpp
│   └── memory_allocator.cpp
│
├── kernels/
│   ├── gemm/
│   │      gemm_16x16.comp
│   │      gemm_32x32.comp
│   │      gemm_wave32.comp
│   │      gemm_int4.comp
│   │      gemm_fp16.comp
│   │      gemm_fp8.comp
│   │
│   ├── attention/
│   │      flash_attention.comp
│   │      paged_attention.comp
│   │      rope.comp
│   │      kv_cache.comp
│   │
│   ├── norm/
│   │      rmsnorm.comp
│   │      layernorm.comp
│   │
│   ├── activation/
│   │      silu.comp
│   │      gelu.comp
│   │      swiglu.comp
│   │
│   ├── moe/
│   │      router.comp
│   │      expert_dispatch.comp
│   │      merge.comp
│   │
│   └── quant/
│          q2.comp
│          q3.comp
│          q4.comp
│          q5.comp
│          q6.comp
│          q8.comp
│
├── compiler/
│   ├── spirv_generator.cpp
│   ├── specialization.cpp
│   ├── pipeline_builder.cpp
│   └── shader_cache.cpp
│
├── tuner/
│   ├── benchmark.cpp
│   ├── occupancy.cpp
│   ├── register_pressure.cpp
│   ├── lds_usage.cpp
│   ├── autotuner.cpp
│   └── profile_database.cpp
│
├── models/
│   ├── llama.cpp
│   ├── qwen.cpp
│   ├── kimi.cpp
│   ├── deepseek.cpp
│   └── gemma.cpp
│
├── rdna/
│   ├── rdna2.cpp
│   ├── rdna3.cpp
│   ├── rdna4.cpp
│   ├── isa_parser.cpp
│   ├── occupancy_model.cpp
│   ├── cache_model.cpp
│   ├── lds_model.cpp
│   └── wave_scheduler.cpp
│
├── graph/
│   ├── graph_optimizer.cpp
│   ├── fusion.cpp
│   ├── execution_plan.cpp
│   └── memory_planner.cpp
│
└── api/
       fastvkllm.h
       context.cpp
       inference.cpp

Runtime flow

Load Model

↓

Detect GPU

↓

RDNA Detector

↓

Architecture Database

↓

Kernel Selection

↓

Autotune

↓

Compile SPIR-V

↓

Pipeline Cache

↓

Inference

Architecture-aware runtime

On startup, detect:

GPU Name

RDNA generation

Compute Units

Wave32 support

LDS per CU

VGPR count

SGPR count

Cache sizes

FP16 support

FP8 support

Cooperative Matrix support

Available Vulkan extensions

Generate a hardware profile such as:

{
  "gpu":"RX9070XT",
  "architecture":"RDNA4",
  "wave":"32",
  "lds":65536,
  "vgpr":512,
  "sgpr":128,
  "cu":64,
  "matrix":true
}

The scheduler then selects kernels based on this profile.

Autotuner

For each operation, benchmark combinations of:

Tile size (8×8, 16×16, 32×32, 64×64)

Workgroup size

Subgroup size

LDS usage

Prefetch distance

Vector width

Pipeline specialisation constants

Quantisation format

Kernel fusion level


For example:

GEMM

Variant 1
16×16
Wave32
48 registers

13.2 TFLOPS

Variant 2
32×16
Wave32
56 registers

14.7 TFLOPS

Variant 3
64×8
Wave32
64 registers

13.8 TFLOPS

The fastest variant is cached for future runs.

RDNA optimisation database

Encode architecture knowledge rather than hard-coding it:

RDNA2

Preferred LDS tile

Preferred subgroup width

Maximum VGPR before occupancy drops

Optimal workgroup size

Cache line alignment

Register bank behaviour

Instruction latency

Similar profiles would exist for RDNA3 and RDNA4.

Kernel fusion engine

Automatically fuse common sequences such as:

MatMul
↓

Bias
↓

RMSNorm
↓

SiLU

into a single compute dispatch to minimise global memory traffic.

Model-specific tuning

Maintain tuned defaults per family:

Llama 3

Qwen 3

DeepSeek

Kimi

Gemma

Mistral

Mixtral

Each profile would include preferred kernel variants, fusion patterns, KV-cache layout, and quantisation strategies.

Long-term vision

The end result would resemble a Vulkan-native analogue to vendor-optimised compute libraries:

GGUF Model

↓

Graph Optimiser

↓

Kernel Fusion

↓

Autotuner

↓

RDNA Scheduler

↓

Vulkan Compute Runtime

↓

AMD Driver

↓

RDNA GPU

A further extension would be to incorporate your earlier interest in physics-inspired scheduling by allowing the runtime to adapt kernel selection and expert routing based on live profiling data, occupancy, memory pressure, and queue utilisation, rather than relying solely on static heuristics. This could make the runtime increasingly efficient over repeated inference runs on a given GPU and model.

=========================
deepseek revision


1. Scaffold Generator (Bash)
Save this as create_scaffold.sh in your project root and run chmod +x create_scaffold.sh && ./create_scaffold.sh.

bash
#!/bin/bash
# FastVK-LLM Directory Scaffold Generator

PROJECT_ROOT="FastVK-LLM"

mkdir -p $PROJECT_ROOT

# Top-level files
touch $PROJECT_ROOT/CMakeLists.txt
touch $PROJECT_ROOT/README.md

# Docs
mkdir -p $PROJECT_ROOT/docs
touch $PROJECT_ROOT/docs/DESIGN.md

# Public API
mkdir -p $PROJECT_ROOT/include
touch $PROJECT_ROOT/include/fastvkllm.h

# Source subdirectories
SRC_DIRS=(
  "src/runtime"
  "src/compiler"
  "src/tuner"
  "src/graph"
  "src/models"
  "src/arch"
  "src/api"
  "src/utils"
)

for dir in "${SRC_DIRS[@]}"; do
  mkdir -p "$PROJECT_ROOT/$dir"
done

# Runtime files
RUNTIME_FILES=(
  "device.cpp" "device.hpp"
  "queues.cpp" "queues.hpp"
  "scheduler.cpp" "scheduler.hpp"
  "command_pool.cpp" "command_pool.hpp"
  "pipeline_cache.cpp" "pipeline_cache.hpp"
  "descriptor_cache.cpp" "descriptor_cache.hpp"
  "barrier_optimizer.cpp" "barrier_optimizer.hpp"
  "profiler.cpp" "profiler.hpp"
  "memory_allocator.cpp" "memory_allocator.hpp"
)
for f in "${RUNTIME_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/runtime/$f"
done

# Compiler files
COMPILER_FILES=(
  "spirv_generator.cpp" "spirv_generator.hpp"
  "specialization.cpp" "specialization.hpp"
  "pipeline_builder.cpp" "pipeline_builder.hpp"
  "shader_cache.cpp" "shader_cache.hpp"
)
for f in "${COMPILER_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/compiler/$f"
done

# Tuner files
TUNER_FILES=(
  "benchmark.cpp" "benchmark.hpp"
  "occupancy.cpp" "occupancy.hpp"
  "register_pressure.cpp" "register_pressure.hpp"
  "lds_usage.cpp" "lds_usage.hpp"
  "autotuner.cpp" "autotuner.hpp"
  "profile_database.cpp" "profile_database.hpp"
)
for f in "${TUNER_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/tuner/$f"
done

# Graph files
GRAPH_FILES=(
  "graph_optimizer.cpp" "graph_optimizer.hpp"
  "fusion.cpp" "fusion.hpp"
  "execution_plan.cpp" "execution_plan.hpp"
  "memory_planner.cpp" "memory_planner.hpp"
)
for f in "${GRAPH_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/graph/$f"
done

# Model files
MODEL_FILES=(
  "llama.cpp" "llama.hpp"
  "qwen.cpp" "qwen.hpp"
  "kimi.cpp" "kimi.hpp"
  "deepseek.cpp" "deepseek.hpp"
  "gemma.cpp" "gemma.hpp"
)
for f in "${MODEL_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/models/$f"
done

# Architecture (HAL) files
ARCH_FILES=(
  "detector.cpp" "detector.hpp"
  "caps.hpp"
  "gcn_polaris.cpp" "gcn_polaris.hpp"   # RX 580
  "rdna2.cpp" "rdna2.hpp"               # 6700 XT
  "rdna4.cpp" "rdna4.hpp"               # 9070 XT
)
for f in "${ARCH_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/arch/$f"
done

# API files
API_FILES=(
  "context.cpp" "context.hpp"
  "inference.cpp" "inference.hpp"
)
for f in "${API_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/api/$f"
done

# Utility files
UTIL_FILES=(
  "logger.cpp" "logger.hpp"
  "hash.cpp" "hash.hpp"
)
for f in "${UTIL_FILES[@]}"; do
  touch "$PROJECT_ROOT/src/utils/$f"
done

# Kernel (GLSL/SPIR-V) directories and placeholders
KERNEL_DIRS=(
  "gemm"
  "attention"
  "norm"
  "activation"
  "moe"
  "quant"
)
mkdir -p "$PROJECT_ROOT/kernels"
for sub in "${KERNEL_DIRS[@]}"; do
  mkdir -p "$PROJECT_ROOT/kernels/$sub"
done

KERNEL_FILES=(
  "kernels/gemm/gemm_16x16.comp"
  "kernels/gemm/gemm_32x32.comp"
  "kernels/gemm/gemm_wave32.comp"
  "kernels/gemm/gemm_int4.comp"
  "kernels/gemm/gemm_fp16.comp"
  "kernels/gemm/gemm_fp8.comp"
  "kernels/attention/flash_attention.comp"
  "kernels/attention/paged_attention.comp"
  "kernels/attention/rope.comp"
  "kernels/attention/kv_cache.comp"
  "kernels/norm/rmsnorm.comp"
  "kernels/norm/layernorm.comp"
  "kernels/activation/silu.comp"
  "kernels/activation/gelu.comp"
  "kernels/activation/swiglu.comp"
  "kernels/moe/router.comp"
  "kernels/moe/expert_dispatch.comp"
  "kernels/moe/merge.comp"
  "kernels/quant/q2.comp"
  "kernels/quant/q3.comp"
  "kernels/quant/q4.comp"
  "kernels/quant/q5.comp"
  "kernels/quant/q6.comp"
  "kernels/quant/q8.comp"
)
for f in "${KERNEL_FILES[@]}"; do
  touch "$PROJECT_ROOT/$f"
done

# Tests
mkdir -p "$PROJECT_ROOT/tests"
touch "$PROJECT_ROOT/tests/test_autotuner.cpp"
touch "$PROJECT_ROOT/tests/test_memory.cpp"
touch "$PROJECT_ROOT/tests/test_barriers.cpp"

echo "✅ FastVK-LLM scaffold created successfully at ./$PROJECT_ROOT"
tree ./$PROJECT_ROOT
2. Directional Design Document (docs/DESIGN.md)
Paste this into FastVK-LLM/docs/DESIGN.md.

markdown
# FastVK-LLM: Directional Design Document

## 1. Vision

Build a **production‑grade, Vulkan‑native LLM inference runtime** that extracts maximum performance from AMD GPUs by treating each micro‑architecture as a first‑class citizen. The runtime is **not** a generic compute wrapper—it is an **architecture‑aware compiler + scheduler** that tunes itself to the specific silicon under the hood.

**Primary Hardware Targets (in‑house test bench):**
- **GCN 4th gen (RX 580)** – legacy, Wave64, no matrix cores. Used for correctness validation and low‑end fallback.
- **RDNA 2 (RX 6700 XT)** – Wave32, 40 CUs, high VGPR count. The realistic daily‑driver baseline.
- **RDNA 4 (RX 9070 XT)** – Wave32, cooperative matrix (`VK_KHR_cooperative_matrix`), FP8, massive LDS. The performance king.

## 2. Core Principles

| Principle | Implementation |
| :--- | :--- |
| **Architecture‑first** | Runtime detects GCN vs. RDNA2 vs. RDNA4 at startup and loads specialised kernel variants. |
| **Autotuning by default** | No one‑size‑fits‑all. Benchmarks tile sizes, workgroup dims, and prefetch distances on first launch; persists results to disk. |
| **Kernel Fusion** | Graph optimiser fuses common sequences (e.g., MatMul + Bias + RMSNorm + SiLU) into single dispatches to minimise global VRAM traffic. |
| **Low‑latency scheduling** | Command pools, descriptor caches, and barriers are pre‑computed and reused. Pipeline creation is asynchronous. |
| **Explicit memory control** | Buddy‑allocator with sub‑pooling for KV‑caches, activation buffers, and weights. Zero‑copy staging where possible. |

## 3. High‑Level Component Breakdown
FastVK-LLM/
├── include/ → Public C-API (fastvkllm.h)
├── src/
│ ├── arch/ → Hardware abstraction layer (detection + capability structs)
│ │ ├── detector.cpp (Vulkan physical device queries)
│ │ ├── gcn_polaris.cpp (RX 580 specific occupancy/cache models)
│ │ ├── rdna2.cpp (6700 XT wave32/vector tuning)
│ │ └── rdna4.cpp (9070 XT matrix core + FP8 tuning)
│ ├── runtime/ → Vulkan object management & execution engine
│ │ ├── device/queues/scheduler
│ │ ├── memory_allocator (pool + sub‑allocation)
│ │ ├── barrier_optimizer (merges adjacent barriers)
│ │ └── profiler (GPU timestamps + occupancy counters)
│ ├── compiler/ → SPIR-V generation & pipeline caching
│ │ ├── spirv_generator (builds shaders from .comp with #defines)
│ │ └── pipeline_cache (disk‑persistent, includes autotune results)
│ ├── tuner/ → Autotuning engine
│ │ ├── occupancy / register_pressure / lds_usage (heuristics)
│ │ └── autotuner (exhaustive/guided search over kernel variants)
│ ├── graph/ → Model graph representation & optimisation
│ │ ├── fusion (pattern matching for matmul+norm+act)
│ │ └── memory_planner (allocates tensors to minimise fragmentation)
│ ├── models/ → Model‑specific frontends (Llama, Qwen, DeepSeek…)
│ └── api/ → Context, session, and inference entry points
├── kernels/ → GLSL / SPIR‑V source (one .comp per operation)
│ ├── gemm/ (16x16, 32x32, int4, fp16, fp8 variants)
│ ├── attention/ (Flash, Paged, RoPE)
│ └── ... (norm, act, moe, quant)
└── tests/ → Unit + integration tests (especially for barrier sync)

text

## 4. Hardware‑Specific Strategy

### 4.1 GCN (RX 580 – Polaris)
- **Subgroup size**: forced **Wave64**.
- **Matrix cores**: absent → all GEMMs use vector `v_fma` instructions.
- **LDS**: 64 KB per CU. Tile sizes capped at **16×64** to avoid bank conflicts.
- **Quantisation**: Q4_0 / Q6_K only (no FP8/INT4 hardware acceleration).
- **Memory**: extremely sensitive to random access. KV‑cache layout must be **linear** (not page‑striped).

### 4.2 RDNA 2 (RX 6700 XT – Navi 22)
- **Subgroup size**: **Wave32** (enables higher occupancy).
- **Matrix cores**: **No** hardware acceleration (`VK_KHR_cooperative_matrix` absent).
- **VGPRs**: 256 per CU → register pressure is the primary bottleneck.
  - Autotuner explores: 32×16, 32×32 tiles.
  - Prefetch distance tuned to keep VGPR usage ≤ 200.
- **LDS**: 64 KB, but lower latency than GCN. Use LDS for matrix multiplication tiling.

### 4.3 RDNA 4 (RX 9070 XT – Navi 4x)
- **Subgroup size**: **Wave32**.
- **Matrix cores**: Full `VK_KHR_cooperative_matrix` support (maps to `v_mfma` instructions).
  - FP16, BF16, and **FP8** dispatch all use MFMA paths.
- **VGPRs**: 512 → allows massive 64×64 tiles with high prefetch.
- **LDS**: 64 KB but with **higher bandwidth**. LDS bank conflicts are rare; use large tiles to saturate.
- **Quantisation**: FP8 and INT4 are first‑class.

## 5. The Autotuner Pipeline

1. **Startup** – detect GPU, load `profile_database.sqlite` for cached results.
2. **On cache miss** – run a mini‑benchmark suite for each operation:
   - Vary: tile size, workgroup size, LDS usage, prefetch distance, vector width.
   - Measure: throughput (TFLOPS) and latency (µs).
3. **Selection** – pick fastest variant **for that specific model layer shape** (e.g., M=4096, K=4096, N=11008).
4. **Persist** – store result keyed by (GPU model, operation, shape hash).

> The tuner is **guided** by the `arch/` occupancy models to prune obviously invalid configs (e.g., tile size requiring > 64KB LDS).

## 6. Kernel Fusion Engine

The graph optimiser applies **pattern matching** to replace:
[MatMul] → [BiasAdd] → [RMSNorm] → [SiLU]

text
with a single fused kernel that streams weights through LDS once, keeping intermediate activations in VGPRs/LDS.

**Fusion examples per architecture:**
- **GCN**: fuse only small sequences (limited VGPRs).
- **RDNA2**: fuse medium sequences (balanced).
- **RDNA4**: fuse large sequences (use matrix cores for the MatMul part, scalar for norm/act).

## 7. Build & Dependencies

```cmake
# CMakeLists.txt (snippet)
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

add_executable(fastvkllm ...)
target_link_libraries(fastvkllm Vulkan::Vulkan Threads::Threads)
Required Vulkan extensions (queried at runtime):

VK_KHR_cooperative_matrix – optional (RDNA4 only).

VK_EXT_subgroup_size_control – to force Wave32/Wave64.

VK_KHR_shader_non_semantic_info – for debug annotations.

VK_KHR_timeline_semaphore – for asynchronous scheduling.

8. Roadmap (MVP → Production)
Phase	Milestone	Key Deliverables
P0	Scaffold + Device Detection	Detect RX 580 / 6700 XT / 9070 XT; print capabilities.
P1	Memory Allocator + Command Pool	Sub‑allocate 1GB VRAM; submit no‑op dispatches.
P2	Single GEMM (FP16) + Autotuner	Run 4096×4096 matmul; tune tile sizes; persist cache.
P3	Model Graph (Llama 3 8B) + Kernels	Implement attention, RoPE, RMSNorm; fuse MatMul+Norm.
P4	Quantisation (Q4_0, Q8_0, FP8)	Load GGUF; run inference with paged KV‑cache.
P5	Multi‑GPU / Async Scheduling	Timeline semaphores; overlap compute + copy.
P6	Physics‑Inspired Scheduler (Stretch)	Online feedback loop adjusts expert routing based on occupancy/utilization live metrics.
9. Testing Strategy
Correctness: Compare FP32 CPU reference outputs for each kernel.

Sync Stress: Run heavy batch sizes on the RX 580 to catch missing VK_ACCESS_SHADER_WRITE_BIT barriers.

Memory Leaks: Long‑running inference sessions (10k+ tokens) with heap tracking.

Performance Regression: GitHub Actions with a headless 6700 XT (if available) running the autotuner suite.

10. Contribution & Future Work
This design intentionally leaves space for:

Wave64 optimisation for GCN – separate kernel compilation pass.

Dynamic shape support – autotuner will re‑tune when sequence length changes significantly.

ROCm interop – optional backend to access HIP for fine‑grained GPU metrics.

Start building: The create_scaffold.sh script above generates all empty source files. Begin with src/arch/detector.cpp—query vkGetPhysicalDeviceProperties2 and vkGetPhysicalDeviceSubgroupProperties to fill the HardwareProfile struct.

text

---

## 3. Quick Start Text for `README.md`

(Place this in `FastVK-LLM/README.md`)

```markdown
# FastVK-LLM

**Vulkan‑native LLM inference for AMD GPUs (GCN, RDNA2, RDNA4).**

- 🚀 **Architecture‑aware** – Wave32/Wave64, Cooperative Matrix (RDNA4), FP8.
- ⚙️ **Self‑tuning** – Autotuner caches optimal kernel variants per GPU/model.
- 🔗 **Kernel Fusion** – reduces VRAM traffic by combining MatMul + Norm + Activations.
- 🧪 **Tested on**: RX 580, RX 6700 XT, RX 9070 XT.

## Build & Run

```bash
git clone <your-repo>
cd FastVK-LLM
mkdir build && cd build
cmake .. -DVULKAN_SDK=<path> -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
Next Steps
See docs/DESIGN.md for the full architectural breakdown and roadmap. Start implementing the hardware detector in src/arch/detector.cpp.

text

---

Run the bash script first to create the empty tree, then drop the `DESIGN.md` and `README.md` content into their respective files. You now have a crystal‑clear map to start coding the **detector** and **memory allocator** immediately. Let me know which component you want to tackle first—I can write the full `detector.cpp` implementation for you.