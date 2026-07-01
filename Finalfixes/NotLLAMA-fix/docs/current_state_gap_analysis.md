# Current State Gap Analysis — 2026-06-29

## What We Have (Verified Actual Code)

### Build System
- CMakeLists.txt: 8 shaders in KERNELS (gemm, attention, rope, add, silu_mul, rms_norm, embed, kv_cache_write)
- 5 targets compile: rdna4_llama, test_inference, test_cpu_ref, test_turboquant, test_compression
- 21 .comp files in src/kernels/ (but only 8 are compiled)

### Pipelines Loaded (main.cpp lines 167-174)
1. gemm — F32 GEMM (reads float data[])
2. attention — per-head softmax attention
3. rope — rotary position embedding
4. add — elementwise residual add
5. silu_mul — SiLU(gate) * up
6. rms_norm — RMS normalization
7. embed — token embedding lookup
8. kv_cache_write — write K/V to cache

### Pipelines NOT Loaded (exist in src/kernels/ but not compiled/loaded)
- kernel_entry.comp — persistent mailbox kernel
- flash_attention.comp — tiled flash attention
- topk.comp — top-K sampling
- mlp_fused_gateup.comp — fused MLP
- bda_test.comp — BDA verification
- cooperative_gemm.comp — cooperative matrix GEMM
- gemm_coopmat.comp — coopmat GEMM
- compress_context.comp — context compression
- kv_cache_quantize.comp — KV cache quantization
- kv_cache_dequant.comp — KV cache dequantization
- dequantize_test.comp — debug dequant (writes 1234.0)

### Weight Loading (weight_uploader.cpp)
- Uploads RAW quantized bytes to GPU (no dequant at load time)
- Creates individual VkBuffer per tensor with BDA
- Stores format, blockSize, blockElements in TensorDesc
- Supports 42 quant formats via ggmlToQuantFormat()
- CPU dequant function exists (cpuDequantToFloat) but NOT called during upload

### Forward Pass (inference_engine.cpp)
- Single beginBatch/endBatch wraps ALL dispatches
- Ring allocator: hidden, Q, K, V, attnOut, mlpOut, logits, sampleOut
- Per-layer: embed → N×(norm → QKV GEMM → RoPE → KV write → attention → output GEMM → residual → FFN norm → gate GEMM → up GEMM → silu+mul → down GEMM → residual) → final norm → LM head GEMM
- CPU argmax sampling
- vkInvalidateMappedMemoryRanges for logits readback

### GEMM Shader (gemm.comp)
- Reads float data[] (F32 only)
- No dequant inside shader
- Single-threaded per column (32 threads per workgroup)
- Supports transB flag for weight transposition

### Memory Model
- RingAllocator: bump allocator, reset() per token, 64MB default
- Weights: individual VkBuffers with BDA (one per tensor)
- KV cache: VkBuffer per layer, float16
- No staging buffers for dequant

---

## What We Need (To Run Non-F32 Models)

### Critical Path: GPU Dequant Before GEMM

The GEMM shader reads F32 weights. Quantized models (Q4_0, Q6_K, etc.) have quantized bytes on GPU. Without dequant, GEMM reads garbage.

**Required:**
1. dequantize.comp shader (real implementation, not test)
2. Dequant pipeline loaded in main.cpp
3. Per-weight dequant dispatch in forwardPartial()
4. Staging buffer for dequant output (one per weight, reused)

### Decision Flow (Per Weight Tensor)

```
findTensorAddr("blk.X.attn_q.weight") → addrW
│
├── tensor.format == F32?
│   └── YES → Pass addrW directly to GEMM ✓
│
├── tensor.format == F16?
│   └── YES → Dequant F16→F32 in staging, pass staging to GEMM
│
└── tensor.format == Q4_0/Q6_K/etc?
    └── YES → Dequant to F32 in staging, pass staging to GEMM
```

### Staging Buffer Strategy

- Allocate ONE staging buffer per weight (largest weight per layer)
- Reuse across layers (reset offset at layer start)
- Size: max(tensor.size_bytes for all weights in layer) * 4 (for F32 output)
- For LLaMA 7B: ~22 MB per weight, ~154 MB total staging

### Dequant Dispatch Pattern

```cpp
// For each weight tensor:
size_t nElements = tensor.sizeBytes / bytesPerElement(tensor.format);
uint64_t stagingAddr = allocator->alloc(nElements * sizeof(float));
DequantizePushConstants dqPC = {tensor.gpuAddress, stagingAddr, nElements, tensor.format, nElements, 0};
scheduler->dispatchInBatch(dequantPipeline, dequantLayout, &dqPC, sizeof(dqPC), (nElements+63)/64, 1, 1);
scheduler->barrierBetweenGroups(CS, CS, SHADER_WRITE, SHADER_READ);
// Then use stagingAddr instead of tensor.gpuAddress for GEMM
```

---

## What's Missing (Complete List)

### Tier 1: Critical (Blocks All Non-F32 Inference)
| Item | Status | Impact |
|------|--------|--------|
| dequantize.comp shader | DELETED | No GPU dequant possible |
| Dequant pipeline load | NOT IN main.cpp | Pipeline unavailable |
| Per-weight dequant dispatch | NOT IN forwardPartial() | GEMM reads garbage |
| Staging buffer allocation | NOT IN forwardPartial() | No output buffer for dequant |

### Tier 2: Required (Blocks Correctness)
| Item | Status | Impact |
|------|--------|--------|
| F16 dequant support | N/A (no dequant shader) | F16 models fail |
| Q4_0 dequant | N/A | Q4_0 models fail |
| Q6_K dequant | N/A | Q6_K models fail |
| Q8_0 dequant | N/A | Q8_0 models fail |
| Q4_K/Q5_K/Q8_K dequant | N/A | K-quant models fail |
| Q2_K/Q3_K dequant | N/A | K-quant models fail |

### Tier 3: Important (Blocks Performance)
| Item | Status | Impact |
|------|--------|--------|
| Embedding cache | NOT IMPLEMENTED | Re-dequants per token |
| Chunked dequant dispatch | NOT IMPLEMENTED | Large weight hangs |
| GPU sampling (topk) | NOT LOADED | CPU argmax only |
| Flash attention | NOT LOADED | Basic attention only |
| kernel_entry path | NOT LOADED | No persistent kernel |

### Tier 4: Future (Architecture Dispatch)
| Item | Status | Impact |
|------|--------|--------|
| ArchConfig system | NOT IMPLEMENTED | LLaMA-only tensor names |
| Gemma-4 support | NOT IMPLEMENTED | All-zero logits |
| Qwen/Phi/Mistral | NOT IMPLEMENTED | Wrong tensor mapping |
| BF16 support | NOT IMPLEMENTED | BF16 models fail |
| IQ format support | NOT IMPLEMENTED | IQ models fail |

---

## Validation: What Works vs What Doesn't

| Model | Format | Weight Type | GEMM Expects | Result |
|-------|--------|-------------|--------------|--------|
| stories260K F32 | F32 | F32 on GPU | F32 | ✅ PASS |
| acrux-500m Q6_K | Q6_K | Q6_K bytes on GPU | F32 | ❌ GARBAGE |
| gemma-4 Q4_0 | Q4_0 | Q4_0 bytes on GPU | F32 | ❌ ALL-ZERO |
| qwen2.5 Q8_0 | Q8_0 | Q8_0 bytes on GPU | F32 | ❌ GARBAGE |
| vibethinker Q6_K | Q6_K | Q6_K bytes on GPU | F32 | ❌ NaN |

---

## Minimum Viable Fix (Step by Step)

### Step 1: Create dequantize.comp
- Support Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16
- Push constants: addrQuant, addrOut, nElements, quantFormat, totalThreads, elementOffset
- One thread per element, stride loop

### Step 2: Add to CMakeLists.txt
- Add src/kernels/dequantize.comp to KERNELS list

### Step 3: Load pipeline in main.cpp and test_inference.cpp
- loadPipe("dequantize", sizeof(DequantizePushConstants));

### Step 4: Add dequant to forwardPartial()
- Before each GEMM, check tensor.format
- If not F32, dispatch dequant to staging buffer
- Use staging buffer address for GEMM

### Step 5: Test
- stories260K F32: must still pass
- Q6_K model: must produce non-NaN logits
- Q4_0 model: must produce non-NaN logits
