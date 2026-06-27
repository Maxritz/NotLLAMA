Executive Summary
NotLLAMA is a nascent, from-scratch Vulkan inference engine for Llama-architecture models, architected specifically for AMD RDNA4 GPUs. It is not a fork of llama.cpp — it is an ideological attempt to build a "graphics-native" inference runtime that treats RDNA4 as a graphics/compute hybrid rather than a "broken CUDA chip."
The repository was created on June 26, 2026 (yesterday) and is currently at the scaffolding stage: the host dispatch code, GLSL kernel shells, and weight conversion pipeline exist, but the project has never been run on real hardware and contains critical synchronization bugs that would prevent it from working today.
1. Repository Metadata
Table
Attribute	Detail
Owner	Maxritz
Created	2026-06-26
License	LGPL-2.1
Language	C++20 (host), GLSL 460 (kernels), Python (tools)
Build	CMake 3.25+, Vulkan SDK, glslc
Size	Effectively ~30 KB of source; empty commit history
Status	Pre-alpha; scaffolding complete, hardware validation pending
2. Architecture & Design Philosophy
The Core Premise
The project is built around "The RDNA4 Manifesto" (included in the repo), which argues that porting CUDA-centric kernels to AMD GPUs is fundamentally wrong. Instead, it advocates treating RDNA4 as a graphics predator that happens to be good at math, leveraging:
Wave32 (32-thread waves) as the native execution unit, not 128-thread CUDA blocks
Register-to-register data movement (DPP / subgroup shuffle) instead of shared memory (LDS) for reductions
Persistent kernel execution (one kernel loops over all layers, eliminating launch overhead)
Multi-ACE speculative decoding (four asynchronous compute engines running draft/target/cache/verify in parallel)
VBUFFER format conversion (hardware dequantization during load)
Dynamic VGPR (S_ALLOC_VGPR) to oversubscribe the register file
The Reality Gap
None of the low-level ISA features advocated in the manifesto (S_WAITCNT, S_CLAUSE, S_DELAY_ALU, VOPD, OMOD, dynamic VGPR) are actually present in the code. The kernels are standard GLSL 460 compute shaders compiled to SPIR-V via glslc. This is a significant philosophical/implementation gap: the manifesto describes a custom ISA assembler (rdna4_as.py), but the actual build system uses unmodified glslc --target-env=vulkan1.3.
3. Code Structure
Host Layer (src/host/)
Table
Component	Purpose	Assessment
context.cpp	Vulkan 1.3 init, BDA extension, 4-queue setup	Clean but minimal; no validation layers
inference_engine.cpp	Transformer forward pass, generate loop, speculative decode	Critical bug: syncAll() after every kernel dispatch; completely synchronous
scheduler.cpp	4-queue dispatch, multi-ACE submission	Dispatches exist but no real async overlap; dispatchMulti() just loops
allocator.cpp	Ring allocator for transient GPU memory	Tries for host-visible+device-local (ReBAR), falls back to host-visible
weight_uploader.cpp	GGUF → GPU upload via staging buffers	Critical bug: staging buffer destroyed before command buffer submission
kv_cache.cpp	KV cache allocation (buffers, not images)	Functional; image/DCC path deferred
pipeline_builder.cpp	VkPipeline creation from SPIR-V	Standard
tokenizer.cpp	BPE tokenizer from JSON vocab	Basic but functional
profiler.cpp	CPU/GPU timing via Vulkan queries	Good instrumentation
vgpr_stub.cpp	Dynamic VGPR extension probe	Stub; checks for VK_AMD_shader_info, does nothing
Kernel Layer (src/kernels/)
Table
Kernel	Workgroup	Notes
gemm.comp	32x8x1	Naive triple-loop GEMM. No tiling, no LDS, no vectorization. Reads FP16, accumulates FP32. This is ~100x slower than an optimized WMMA/SDWA kernel.
attention.comp	32x1x1	Causal attention with subgroup shuffle (subgroupShuffleDown) for softmax reduction. Correct algorithm for wave32, but uses large static arrays (float qVec[32], float scores[128]) that may cause register pressure.
flash_attention.comp	32x1x1	Tiled flash attention with online softmax. Loads Q row into float qVec[128] — very high register pressure for large head dims. No LDS tiling.
mlp.comp	—	SiLU-gated MLP with fused up/gate/down projections
rope.comp	—	Rotary position embedding
rms_norm.comp	—	Root-mean-square layer norm
topk.comp	256x1x1	Argmax sampling kernel
kv_cache_write.comp	—	Cache update for K/V tensors
dequantize.comp	—	Software dequantization (GGUF Q4_0, etc.)
Tools (tools/)
weight_converter.py: Converts GGUF → flat binary + JSON metadata. Pads to 256-byte alignment for GPU upload.
gguf_loader.py: Full GGUF parser (supports all GGML quantization types).
rdna4_as.py: AMD GCN/RDNA4 assembler stub. Intended to emit raw ISA binaries for VK_AMD_shader_info, but unimplemented.
benchmark.py / validate_build.py: Build/perf validation scripts.
rag.py: Retrieval-augmented generation script (unrelated to core engine).
4. Critical Blocking Issues
From STATUS-NOTLLAMA.md and code review:
1. Staging Buffer Lifetime (CRITICAL)
uploadTensor() creates a staging buffer, records vkCmdCopyBuffer, then destroys the staging buffer before submitting the command buffer. The GPU will read from freed memory. Fix requires persistent staging or batching all uploads into one submit.
2. Host-Visible Readback (CRITICAL)
forward() attempts to read the sampled token directly from allocator->mappedPtr without a GPU→host copy or memory flush. If the ring allocator is not host-visible (or if writes aren't flushed), this returns garbage. Needs a dedicated readback staging buffer.
3. Excessive Synchronization
The inference engine calls scheduler->syncAll() after every single kernel dispatch (embedding, norm, QKV, RoPE, attention, MLP, residual, etc.). This means:
Zero kernel overlap
CPU is bottlenecked on vkQueueWaitIdle
Multi-ACE queues are pointless because everything is serialized
4. Naive GEMM
The GEMM kernel is a textbook naive implementation. For RDNA4, this should at minimum:
Tile into 16×16 or 32×32 tiles
Use f16 dot products or matrix accumulation
Consider WMMA/SDWA instructions (though these require ISA-level control, not GLSL)
5. No Real Multi-ACE Parallelism
The scheduler has 4 queues, but InferenceEngine::forward() is a single-threaded synchronous loop. Speculative decode dispatches draft verification to queues 1-3, but immediately waits on them. There is no true concurrent draft model execution.
5. Strengths
Clean Abstraction: The separation into VulkanContext, Scheduler, RingAllocator, PipelineBuilder, and InferenceEngine is well-structured.
GGUF Native: Full GGUF parser and weight converter means it can load real models (TinyLlama, Qwen, etc.) without ONNX/PyTorch dependencies.
Buffer Device Address: Uses VK_KHR_buffer_device_address with 64-bit buffer references in GLSL, eliminating descriptor set overhead per kernel — this is genuinely good for RDNA4.
Push Constants: All kernel parameters are passed via push constants, avoiding uniform buffer updates.
Flash Attention: The online softmax flash attention kernel is algorithmically correct and avoids the O(N²) memory blowup of standard attention.
Profiler: Built-in CPU/GPU timing infrastructure is present from day one.
6. Comparison to llama.cpp Vulkan
Table
Feature	NotLLAMA	llama.cpp Vulkan
Maturity	Pre-alpha, never run	Production-ready, thousands of users
GEMM	Naive GLSL	Tiled, optimized, multiple variants
Attention	Flash attn (registers)	Multiple backends (CUDA, Vulkan, Metal, CPU)
Quantization	Software dequant	Hand-optimized per-format kernels
Multi-GPU	Not implemented	Supported
Model support	Llama-like only	Llama, GPT-2, Gemma, Qwen, Mistral, etc.
RDNA4-specific	Claims native, but standard SPIR-V	Generic Vulkan, but Mesa/RADV optimizations benefit it
Speculative decode	Skeleton only	Fully implemented
Recent community benchmarks show llama.cpp's Vulkan backend on RDNA4 (RX 9070 XT) can outperform ROCm in some cases, especially on Linux with Mesa 25.x + RADV. NotLLAMA is starting from zero against a moving target.
7. Verdict & Trajectory
NotLLAMA is an interesting research/experimental project, not a practical inference engine today. It has the scaffolding of something ambitious — a native RDNA4 inference runtime with multi-ACE scheduling and flash attention — but the implementation is still standard Vulkan compute with critical bugs preventing execution.
To reach viability, the author needs to:
Fix staging buffer lifetime and host readback (Phase 1 — this weekend)
Run on real hardware with TinyLlama Q4_0 and verify end-to-end token generation
Replace naive GEMM with a tiled, vectorized implementation (or call into a mature matrix library)
Eliminate syncAll() between independent kernels (QKV projections can dispatch in parallel)
Decide on the ISA strategy: either commit to SPIR-V + GLSL and accept the limitations, or actually build the rdna4_as.py assembler and use VK_AMD_shader_info to load raw GCN binaries
The manifesto's philosophy — "Fuck CUDA. Build for the kitchen, not the factory" — is compelling, but the code has not yet caught up to the rhetoric. If the synchronization bugs are fixed and the project runs on real RDNA4 hardware, it could become a valuable testbed for RDNA4-specific inference optimizations. Until then, it remains a promising skeleton.