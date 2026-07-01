# NotLLAMA

RDNA4-native LLM inference engine. Vulkan compute shaders, not a CUDA port.

## What Is This

A from-scratch Vulkan inference runtime built specifically for AMD RDNA4 GPUs. Treats RDNA4 as a graphics predator that happens to be good at math — Wave32 native, DPP register shuffles, persistent kernels, multi-ACE scheduling.

**Not a fork of llama.cpp.** Different architecture, different approach, different philosophy.

## Current Status

| Milestone | Status |
|-----------|--------|
| Build system (CMake + MSVC + glslc) | Done |
| 13 SPIR-V compute shaders | Done |
| BDA pipeline (zero descriptor overhead) | Done |
| Weight upload (434 tensors, ~2.9 GB) | Done |
| Buffer-based KV cache with BDA | Done |
| Q6_K dequantization (correct nibble/scale handling) | Done |
| Dequant dispatch overflow fix (totalThreads push constant) | Done |
| Compute queue routing (family 1, not 3D engine) | Done |
| VK_ERROR_DEVICE_LOST resolution | Done |
| Full 36-layer forward pass | Done |
| First non-zero token generation | **In progress** |
| Top-k sampling | Stub |
| Multi-ACE parallel dispatch | Stub |
| Cooperative matrix GEMM | Placeholder |
| Speculative decoding | Stub |

## Architecture

```
┌─────────────────────────────────────────────┐
│  Persistent Kernel (one launch, zero overhead) │
├──────────┬──────────┬──────────┬────────────┤
│  ACE 0   │  ACE 1   │  ACE 2   │  ACE 3    │
│  GEMM    │  Attn    │  MLP     │  KV/Wrap  │
└──────────┴──────────┴──────────┴────────────┘
         ↓ BDA (no descriptors) ↓
┌─────────────────────────────────────────────┐
│  Weight Buffer (quantized on GPU)            │
│  Dequantize on-the-fly before each GEMM     │
└─────────────────────────────────────────────┘
```

**Key design decisions:**
- Weights stay quantized on GPU, dequantized to staging buffer before each GEMM
- All activations are float; weights dequant to float16
- KV cache is flat VkBuffer with buffer device address
- Push constants for all kernel parameters (no uniform buffers)
- Queue family 1 (compute-only) on AMD RX 9070 XT — avoids 3D engine routing

## Supported Quantizations

Q1–Q8, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, F16, BF16, F32, mxfp4_moe

## Build

```bash
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Run

```bash
cd build\Release
rdna4_llama.exe ..\model\<weights.json> ..\model\<weights.bin> "Hello"
```

## Tools

| Tool | Purpose |
|------|---------|
| `weight_converter.py` | GGUF → flat binary + JSON metadata |
| `gguf_loader.py` | Full GGUF parser (all quant types) |
| `rdna4_as.py` | AMD GCN/RDNA4 assembler (WIP) |
| `rag.py` | Retrieval-augmented generation |
| `benchmark.py` | Performance benchmarks |

## Hardware

- **Target**: AMD RX 9070 XT (RDNA4, 16 GB VRAM)
- **Queue families**: Family 1 (compute-only, 8 queues) — confirmed routing to Compute 0 engine
- **Vulkan SDK**: 1.4.350.0
- **Extensions**: VK_KHR_cooperative_matrix, VK_EXT_shader_float8, VK_AMD_shader_core_properties

## Roadmap

1. First non-zero token generation (in progress)
2. Full autoregressive decode loop
3. Top-k/top-p sampling
4. Fused dequant+GEMM kernel (2x bandwidth savings)
5. Flash attention with LDS tiling
6. Cooperative matrix GEMM
7. Multi-ACE true parallel dispatch
8. GPU-driven dispatch (VK_EXT_device_generated_commands)
9. Speculative decoding
10. Multi-computer agent framework

## License

LGPL-2.1

## Links

- [GitHub](https://github.com/Maxritz/NotLLAMA)
