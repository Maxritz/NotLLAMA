# Context for deepseek-reasoner: RDNA4 LLM Inference Engine

## Project Overview
We're building a full-scale RDNA4-native LLM inference engine using Vulkan compute shaders. No CUDA patterns. Target: AMD RX 9070 XT (16GB VRAM). Model: VibeThinker-3B.Q6_K (2.4GB GGUF, qwen2 architecture).

## What's Already Working
- Full build system (CMake + MSVC + glslc), all 13 shaders compile to SPIR-V
- `rdna4_llama.exe` runs, detects RX 9070 XT, initializes 4 ACE queues
- Full inference pipeline: 434 tensors loaded, 36-layer forward pass completes VK_ERROR_DEVICE_LOST-free
- BDA test passes (256 uint32 values verified)
- Weight converter: GGUF → JSON + BIN
- Dequantize shader: Q6_K, Q8_0, Q4_0, F16, F32
- All shaders: kernel_entry, gemm, attention, flash_attention, mlp, rope, topk, add, rms_norm, embed, kv_cache_write, dequant, bda_test
- Embedding cache: dequantized once at init, reused across tokens (prevents 9.7M workgroup GPU hang)
- Chunked dequant dispatch (caps at 1M workgroups per dispatch)

## CRITICAL: What Just Got Fixed (DO NOT REDO)
- `dequantWeight()` in `inference_engine.cpp` now chunks large dispatches (max 1M workgroups)
- Embedding table is cached persistently in a DEVICE_LOCAL buffer (`embedCacheAddr`)
- The 9.7M workgroup hang is fixed — do not remove the chunking or the cache

## What Needs To Be Done

### Task 1: Implement Proper Top-K / Top-P Sampling
**Current state**: `forward()` uses a basic top-k shader that writes to `sampleOutAddr`, then reads back with `memcpy`. The `sampleArgmax()` function is a CPU fallback.

**What's needed**:
- The `topk.comp` shader currently does a basic scan. It needs proper top-k with sorting.
- Add temperature scaling before top-k
- Add top-p (nucleus) sampling after top-k
- The sampling should happen on GPU (read logits from ring allocator, apply temperature, sort, sample)
- Fallback to CPU `sampleArgmax()` if GPU sampling fails

**Key files**:
- `src/kernels/topk.comp` — current top-k shader (needs rewrite)
- `src/host/inference_engine.cpp` lines 499-522 — sampling section in `forward()`
- `include/rdna4_types.hpp` — `TopKPushConstants` struct

**Constraints**:
- Keep using buffer device address (BDA) for all buffer reads/writes
- Use float32 for logits (Q6_K dequant outputs float32)
- GPU sampling preferred, CPU fallback acceptable
- Must handle vocab sizes up to 151936 (Qwen2)

### Task 2: Fix Exit Crash (0xC0000005)
**Current state**: Process crashes at teardown with access violation.

**What to check**:
- `main.cpp` cleanup order: `pipelines.cleanup()`, `kvCache.free()`, `engine.cleanupEmbedCache()`, `engine.cleanupDequantBuffer()`, `uploader.freeAll(model)`, `profiler.cleanup()`, `ctx.cleanup()`
- Possible double-free of VkBuffer/VkDeviceMemory
- Possible use-after-free if scheduler references cleaned-up resources
- Check if any VkFence/VkSemaphore are not being waited on before destroy

**Key files**:
- `main.cpp` lines 306-314 — cleanup sequence
- `src/host/scheduler.cpp` — queue submission and fence handling
- `include/rdna4_scheduler.hpp` — Scheduler class definition

### Task 3: Build CLI Tools
**Current state**: Only `rdna4_llama.exe` exists with hardcoded prompt.

**What's needed** (modeled after llama.cpp):
- `rdna4-cli.exe` — interactive chat, batch inference, perplexity
- `rdna4-server.exe` — HTTP server with OpenAI-compatible API
- `rdna4-bench.exe` — benchmarking (tokens/sec, latency, VRAM usage)

**Key files**:
- `main.cpp` — current entry point (reference for argument parsing)
- `include/rdna4_engine.hpp` — InferenceEngine class API
- `include/rdna4_tokenizer.hpp` — Tokenizer API

**Constraints**:
- Use simple argument parsing (no external deps beyond STL)
- Server needs basic HTTP (use platform socket API, no boost/beast)
- All tools share the same InferenceEngine backend

### Task 4: Implement draftForward Properly
**Current state**: `draftForward()` just calls `forward()` with all layers (stub).

**What's needed**:
- Skip to layer `nLayers` instead of running all 36 layers
- Use fewer layers for faster draft generation
- Properly handle KV cache for partial layers

**Key file**: `src/host/inference_engine.cpp` line 611-614

## Architecture Notes
- **Shaders**: GLSL 460 with buffer_reference, compiled with glslc
- **Dispatch**: `scheduler->dispatch(pipeline, layout, pushConstants, size, gx, gy, gz)`
- **Memory**: Ring allocator for activations, separate buffers for weights/KV cache
- **Quantization**: Weights stay quantized on GPU, dequantized to float32 staging buffer for compute
- **KV cache**: float16_t, stored as VkBuffer with BDA
- **Push constants**: All shader parameters passed via push constants (max 256 bytes)

## File Structure
```
C:\Users\rr\Desktop\Notllama-loc\
├── CMakeLists.txt
├── main.cpp                          # Entry point
├── include/
│   ├── rdna4.hpp                     # Core types
│   ├── rdna4_engine.hpp              # InferenceEngine class
│   ├── rdna4_types.hpp               # Push constant structs, enums
│   ├── rdna4_vulkan.hpp              # VulkanContext
│   ├── rdna4_weights.hpp             # Weight uploader, ModelDesc, TensorDesc
│   ├── rdna4_pipeline.hpp            # PipelineBuilder
│   ├── rdna4_scheduler.hpp           # Scheduler (queue submission)
│   ├── rdna4_allocator.hpp           # RingAllocator
│   ├── rdna4_kv_cache.hpp            # KVCacheManager
│   └── rdna4_tokenizer.hpp           # Tokenizer
├── src/
│   ├── host/
│   │   ├── inference_engine.cpp      # Main inference logic (629 lines)
│   │   ├── scheduler.cpp             # Queue submission
│   │   ├── pipeline_builder.cpp      # Pipeline/layout creation
│   │   ├── ring_allocator.cpp        # Ring buffer allocator
│   │   ├── kv_cache.cpp              # KV cache management
│   │   ├── weight_uploader.cpp       # GGUF → GPU upload
│   │   ├── gguf_loader.cpp           # GGUF parser
│   │   ├── memory_manager.cpp        # VkBuffer/VkMemory helpers
│   │   ├── tokenizer.cpp             # BPE tokenizer
│   │   ├── mailbox.cpp               # GPU->CPU readback
│   │   └── profiler.cpp              # Timestamp queries
│   └── kernels/
│       ├── dequantize.comp           # Quantized → float32
│       ├── gemm.comp                 # Matrix multiply (with transB)
│       ├── attention.comp            # Softmax attention
│       ├── flash_attention.comp      # Flash attention variant
│       ├── mlp.comp                  # Feed-forward network
│       ├── rope.comp                 # Rotary position embedding
│       ├── topk.comp                 # Top-K sampling
│       ├── add.comp                  # Element-wise add
│       ├── rms_norm.comp             # RMS normalization
│       ├── embed.comp                # Token embedding lookup
│       ├── kv_cache_write.comp       # Write K/V to cache
│       ├── kernel_entry.comp         # Kernel entry point
│       └── bda_test.comp             # BDA verification
├── build/                            # Build output
├── model/                            # Model files (JSON + BIN)
├── tools/
│   ├── weight_converter.py           # GGUF → JSON + BIN
│   ├── gguf_loader.py                # GGUF inspection
│   └── rag.py                        # RAG tool
├── graphify-out/                     # Knowledge graph
└── reference/                        # Vulkan SDK files
```

## Build & Run
```bash
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd Release
rdna4_llama.exe ..\model\VibeThinker-3B.Q6_K.weights.json ..\model\VibeThinker-3B.Q6_K.weights.bin "Hello"
```

After shader recompile: `Copy-Item build\shaders\*.spv build\Release\shaders\`

## Testing
- Build verification: `cmake --build . --config Release` must succeed
- Runtime: `rdna4_llama.exe` must not crash (VK_ERROR_DEVICE_LOST or 0xC0000005)
- GPU hang prevention: dequant dispatch must stay under 1M workgroups per dispatch
