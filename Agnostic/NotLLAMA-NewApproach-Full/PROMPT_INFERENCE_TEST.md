# Task: Prove the Forward Pass Produces Correct Token Output

## Context

You are working on an RDNA4 LLM inference engine at `C:\Users\rr\Desktop\Notllama-loc`. The engine runs VibeThinker-3B.Q6_K (qwen2 architecture) end-to-end on Vulkan compute shaders. The forward pass completes without crashes (no VK_ERROR_DEVICE_LOST), but we have NEVER verified the output tokens are correct.

## Goal

Create a test harness (`test_inference.cpp`) that proves the model generates correct tokens. If output is wrong, add debug readback to identify which kernel is broken.

## What We Know

- Model: VibeThinker-3B.Q6_K (36 layers, 2048 embedding, 32 heads, 151936 vocab)
- Forward pass: embed → 36 transformer blocks → final norm → LM head → sample
- All weights dequantized to float32 before compute
- KV cache: float16, stored as VkBuffer with BDA
- Sampling: topk shader with temperature=1.0, topK=40, topP=0.9
- Build: `cd build && cmake --build . --config Release`
- Run: `cd build\Release && rdna4_llama.exe ..\model\VibeThinker-3B.Q6_K.weights.json ..\model\VibeThinker-3B.Q6_K.weights.bin "prompt"`

## Task 1: Create `test_inference.cpp`

Create a new file `test_inference.cpp` that:

1. Loads the model (same as `main.cpp`)
2. Runs a SINGLE token through `forward(tokenId, seqPos=0)`
3. Reads back the logits tensor from GPU (use `vkMapMemory` on the ring allocator)
4. Prints the top-5 token IDs and their logit values
5. Compares against a CPU reference implementation (see Task 2)
6. Reports PASS/FAIL with error metrics (max absolute error, mean absolute error)

The test should:
- Use a known input token (e.g., token ID 1 = "Hello" in Qwen2 tokenizer)
- Print the first 10 logits (indices 0-9) for manual inspection
- Print the argmax token ID and its logit value
- Exit with code 0 on PASS, 1 on FAIL

## Task 2: Implement CPU Reference Forward Pass

Create a simplified CPU reference implementation that:

1. Loads the same weights (read from the .bin file directly)
2. Implements the forward pass in pure C++:
   - Embedding lookup (token_embd.weight[tokenId])
   - For each layer:
     - RMS norm (attn_norm.weight)
     - Q/K/V GEMM (attn_q.weight, attn_k.weight, attn_v.weight)
     - RoPE
     - Attention (Q @ K^T / sqrt(headDim), softmax, @ V)
     - Output projection (attn_output.weight)
     - Residual add
     - RMS norm (ffn_norm.weight)
     - MLP (gate GEMM, up GEMM, SiLU activation, down GEMM)
     - Residual add
   - Final RMS norm (output_norm.weight)
   - LM head GEMM (token_embd.weight transposed, weight-tied)
3. Returns the logits vector (vocabSize floats)
4. Prints the top-5 token IDs and their logit values

**Important**: The CPU reference only needs to work for F16/F32 weights. For Q6_K weights, either:
- Convert Q6_K → F32 in Python first, OR
- Implement Q6_K dequant in C++ (block size=210, scale + min + 6-bit values)

The CPU reference is for VALIDATION ONLY — it doesn't need to be fast.

## Task 3: Debug Readback (If Output Is Wrong)

If the GPU output doesn't match the CPU reference, add debug readback to identify the broken kernel:

1. After each kernel dispatch in `forwardPartial()`, read back the output buffer and print:
   - First 4 float values
   - Min/max/mean values
   - Whether values are all-zero, all-NaN, or all-inf

2. Create a table comparing GPU vs CPU at each stage:
   ```
   Stage              | GPU[0..3]              | CPU[0..3]              | MaxErr
   --------------------|------------------------|------------------------|--------
   After embed         | 0.123 0.456 0.789 0.012| 0.123 0.456 0.789 0.012| 0.001
   After layer 0 attn  | ...                    | ...                    | ...
   After layer 0 mlp   | ...                    | ...                    | ...
   After final norm    | ...                    | ...                    | ...
   After LM head       | ...                    | ...                    | ...
   ```

3. The FIRST stage with MaxErr > 0.01 identifies the broken kernel.

## Task 4: Build and Run Instructions

Add to `CMakeLists.txt`:
```cmake
add_executable(test_inference test_inference.cpp)
target_link_libraries(test_inference PRIVATE rdna4_core)
```

Build and run:
```bash
cd build
cmake --build . --config Release
cd Release
test_inference.exe ..\model\VibeThinker-3B.Q6_K.weights.json ..\model\VibeThinker-3B.Q6_K.weights.bin
```

## Constraints

- Do NOT modify existing files (inference_engine.cpp, shaders, etc.) unless adding debug readback
- The test harness is a NEW file: `test_inference.cpp`
- CPU reference is a NEW file: `cpu_reference.cpp` + `cpu_reference.h`
- All memory reads use `vkMapMemory` + `vkInvalidateMappedMemoryRanges`
- Use the existing `ModelDesc`, `TensorDesc`, `WeightUploader` classes to load weights
- The ring allocator's mapped pointer is available via `allocator->mappedPtr`

## Expected Output

If everything works:
```
=== Inference Test ===
Loading model...
Model loaded: 434 tensors
Running forward pass for token 1 ("Hello")...
GPU logits[0..9]: -12.345 -8.765 -6.543 -5.432 -4.321 -3.210 -2.109 -1.098 -0.987 -0.876
GPU argmax: token 42 = -0.123
CPU logits[0..9]: -12.344 -8.764 -6.542 -5.431 -4.320 -3.209 -2.108 -1.097 -0.986 -0.875
CPU argmax: token 42 = -0.122
Max absolute error: 0.001
PASS
```

If something is broken:
```
=== Inference Test ===
Loading model...
Model loaded: 434 tensors
Running forward pass for token 1 ("Hello")...
After embed:        GPU[0..3] = 0.123 0.456 0.789 0.012 | CPU[0..3] = 0.123 0.456 0.789 0.012 | MaxErr = 0.000
After layer 0 norm: GPU[0..3] = 0.234 0.567 0.890 0.123 | CPU[0..3] = 0.234 0.567 0.890 0.123 | MaxErr = 0.000
After layer 0 Q:    GPU[0..3] = 0.000 0.000 0.000 0.000 | CPU[0..3] = 0.345 0.678 0.901 0.234 | MaxErr = 0.901
BROKEN: Layer 0 Q GEMM produces all zeros
```

## Key Files

- `src/host/inference_engine.cpp` — forwardPartial() at line 293
- `include/rdna4_engine.hpp` — InferenceEngine class
- `include/rdna4_types.hpp` — Push constant structs
- `src/kernels/gemm.comp` — GEMM shader
- `src/kernels/embed.comp` — Embedding shader
- `src/kernels/rms_norm.comp` — RMS norm shader
- `src/kernels/attention.comp` — Attention shader
- `src/kernels/mlp.comp` — MLP shader
- `src/kernels/rope.comp` — RoPE shader
- `src/kernels/dequantize.comp` — Dequantization shader
- `main.cpp` — Reference for model loading and engine setup
