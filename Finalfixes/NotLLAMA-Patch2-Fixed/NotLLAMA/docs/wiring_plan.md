# NotLLAMA Engine Wiring Plan — TurboQuant + Context/KV Compression

> **Scope:** This is an implementation-ready design document. It does **not** modify source logic; it tells the next developer exactly where and how to wire the already-delivered TurboQuant and compression assets into the existing engine.
>
> **Assumed files (already in repo):**
> - TurboQuant: `src/kernels/dequant_turbo.comp`, `src/kernels/gemm_turbo.comp`
> - Compression: `src/kernels/compress_context.comp`, `src/kernels/kv_cache_quantize.comp`, `src/kernels/kv_cache_dequant.comp`
> - Host headers: `include/rdna4_types.hpp`, `include/rdna4_compression.hpp`, `include/rdna4_compression_scheduler.hpp`
> - Engine wiring points: `src/host/inference_engine.cpp`, `src/host/kv_cache.cpp`, `main.cpp`, `test_inference.cpp`, `CMakeLists.txt`

---

## 1. TurboQuant Engine Wiring

### 1.1 Add host push-constant structs (`include/rdna4_types.hpp`)

The delivered shaders expose push constants that are **not** in `rdna4_types.hpp`. Add two structs that match the GLSL `layout(push_constant, scalar)` layouts byte-for-byte.

```cpp
// Matches src/kernels/dequant_turbo.comp lines 17-26
struct TurboQuantPushConstants {
    uint64_t addrSrc;      // Raw TurboQuant weight bytes
    uint64_t addrDst;      // F16 output buffer
    uint32_t n;            // Total elements to dequantize
    uint32_t blockSize;    // 128 for TQ4_128, 64 for TQ6_64
    uint32_t bits;         // 4 or 6
    uint32_t scaleBits;    // 16 (fp16 scale)
    uint32_t zeroPoint;    // 0 for TQ (symmetric)
    float    scale;        // Host-side scale multiplier (usually 1.0f)
};
static_assert(sizeof(TurboQuantPushConstants) <= 128,
    "TurboQuantPushConstants exceeds Vulkan push-constant limit");

// Matches src/kernels/gemm_turbo.comp lines 20-32
struct GemmTurboPushConstants {
    uint64_t addrA;        // Activations (F16, M x K)
    uint64_t addrB;        // TurboQuant weights (K x N, packed)
    uint64_t addrC;        // Output (F16, M x N)
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t blockSize;    // 128 for TQ4_128, 64 for TQ6_64
    uint32_t bits;         // 4 or 6
    uint32_t scaleBits;    // 16
    uint32_t zeroPoint;    // 0
    float    alpha;        // GEMM scaling (usually 1.0f)
};
static_assert(sizeof(GemmTurboPushConstants) <= 128,
    "GemmTurboPushConstants exceeds Vulkan push-constant limit");
```

> **Placement:** Insert before `KernelEntryPushConstants` so the existing `static_assert` block for kernel_entry is not disturbed.

### 1.2 Build the SPIR-V (`CMakeLists.txt`)

Add the two new `.comp` files to the `KERNELS` list (currently `CMakeLists.txt` lines 50-66):

```cmake
set(KERNELS
    src/kernels/kernel_entry.comp
    src/kernels/gemm.comp
    src/kernels/attention.comp
    ...
    src/kernels/dequant_turbo.comp      # <-- add
    src/kernels/gemm_turbo.comp         # <-- add
    ...
)
```

The existing `add_custom_command` will compile them to `${CMAKE_BINARY_DIR}/shaders/*.spv`. The same copy step used for other shaders applies:

```powershell
Copy-Item build\shaders\dequant_turbo.spv build\Release\shaders\
Copy-Item build\shaders\gemm_turbo.spv    build\Release\shaders\
```

### 1.3 Register the pipelines (`main.cpp` / `test_inference.cpp`)

The current registrations use the **wrong** push-constant sizes:

- `main.cpp` line 184: `loadPipe("dequant_turbo", sizeof(DequantizePushConstants));`  
  `main.cpp` line 185: `loadPipe("gemm_turbo",    sizeof(GemmPushConstants));`
- `test_inference.cpp` does not load either pipeline at all.

Replace / add:

```cpp
// main.cpp, inside the loadPipe lambda block (lines ~172-186)
loadPipe("dequant_turbo", sizeof(TurboQuantPushConstants));
loadPipe("gemm_turbo",    sizeof(GemmTurboPushConstants));

// test_inference.cpp, inside the loadPipe lambda block (lines ~119-132)
loadPipe("dequant_turbo", sizeof(TurboQuantPushConstants));
loadPipe("gemm_turbo",    sizeof(GemmTurboPushConstants));
```

`PipelineBuilder::createComputePipeline` will then create layouts with the correct push-constant range sizes.

### 1.4 Tensor metadata needed from the weight loader

`WeightUploader::load` already reads `quant_block_size` (`src/host/weight_uploader.cpp:180`). Ensure that for TurboQuant tensors the JSON also carries:

- `format`: integer matching `QuantFormat::TQ4_128` (57) or `QuantFormat::TQ6_64` (59)
- `blockSize`: 128 for TQ4_128, 64 for TQ6_64
- `scaleBits`: 16
- `zeroPoint`: false

`TensorDesc::format`, `TensorDesc::blockSize`, and `TensorDesc::gpuAddress` are the only fields the wiring code needs at dispatch time.

### 1.5 Where to dispatch in `forwardPartial()`

`forwardPartial()` is one big batched command buffer (`src/host/inference_engine.cpp:743`). The TurboQuant assets can be used in two ways:

| Strategy | Shader | Use when |
|----------|--------|----------|
| Standalone dequant | `dequant_turbo.comp` | Weight tensor is TQ and you want F16 weights to feed into another shader (e.g. an F16-aware embed lookup). |
| Fused GEMM | `gemm_turbo.comp` | Weight tensor is TQ and activations are already F16. |

**Critical prerequisite:** `gemm_turbo.comp` reads `float16_t` activations and writes `float16_t` outputs. The current engine allocates `hiddenAddr`, `attnOutAddr`, `mlpOutAddr`, `qAddr`/`kAddr`/`vAddr`, etc. as **float32**. Two practical paths are described below; pick one before editing.

#### Path A — Minimal wiring (add F16⇄F32 conversion helpers)
Keep the existing float32 pipeline and add small conversion shaders (`f16_to_f32.comp`, `f32_to_f16.comp`). This lets `dequant_turbo.comp` and `gemm_turbo.comp` participate without rewriting `rms_norm.comp`, `add.comp`, or `silu_mul.comp`.

1. Before a `gemm_turbo` dispatch, convert the float32 activation buffer to a temporary F16 buffer with `f32_to_f16.comp`.
2. Run `gemm_turbo.comp`.
3. Convert the F16 output back to float32 with `f16_to_f32.comp` before the residual `add.comp` or `silu_mul.comp` reads it.

This path is slower but isolated.

#### Path B — Full F16 activation pipeline (recommended for performance)
If the model contains TurboQuant weights, allocate the forward scratch buffers as F16 and provide F16 elementwise shaders (`rms_norm_f16`, `add_f16`, `silu_mul_f16`). Then `gemm_turbo.comp` drops in directly. This is the path the delivered shaders were designed for.

The rest of this section assumes Path B; if Path A is chosen, insert the conversion dispatches at each producer→consumer boundary.

#### 1.5.1 Embedding table (`token_embd.weight`)

Current code: `src/host/inference_engine.cpp:746-768`.

If `token_embd.weight` is TQ:

```cpp
const TensorDesc* embedTensor = findTensor(*model, "token_embd.weight");
if (embedTensor && (embedTensor->format == QuantFormat::TQ4_128 ||
                    embedTensor->format == QuantFormat::TQ6_64)) {
    // Allocate a temporary F16 buffer from the ring (or use a persistent F16 embed cache)
    size_t embedF16Bytes = (size_t)model->vocabSize * dim * sizeof(uint16_t);
    uint64_t embedF16Addr = allocator->alloc(embedF16Bytes);

    TurboQuantPushConstants dqPC = {};
    dqPC.addrSrc    = embedTensor->gpuAddress;
    dqPC.addrDst    = embedF16Addr;
    dqPC.n          = (uint32_t)model->vocabSize * dim;
    dqPC.blockSize  = embedTensor->blockSize;   // 128 or 64
    dqPC.bits       = (embedTensor->format == QuantFormat::TQ4_128) ? 4u : 6u;
    dqPC.scaleBits  = 16;
    dqPC.zeroPoint  = 0;
    dqPC.scale      = 1.0f;

    uint32_t wg = (dqPC.n + 127) / 128;  // dequant_turbo uses local_size_x=128
    scheduler->dispatchInBatch(pipelines->getPipeline("dequant_turbo"),
                               pipelines->getLayout("dequant_turbo"),
                               &dqPC, sizeof(dqPC), wg, 1, 1);
    scheduler->barrierBetweenGroups(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    // Feed the F16 table into an F16-aware embed lookup shader.
    // (The existing embed.comp reads float[], so it cannot consume this directly.)
}
```

> **Note:** `embed.comp` reads `float data[]`, so a new `embed_f16.comp` (or `embed_turbo.comp`) is required unless the engine converts the F16 table back to F32 first.

#### 1.5.2 Attention weights

Current dequant location: `src/host/inference_engine.cpp:780-784`.

For each attention weight (`attn_q`, `attn_k`, `attn_v`, `attn_output`), if the tensor is TQ you can either:

- **Fused path:** Skip standalone dequant and bind the raw TQ tensor address directly to `gemm_turbo.comp`.
- **Standalone path:** Dispatch `dequant_turbo.comp` into a per-weight F16 slot in the staging buffer, then use `gemm_turbo.comp` or an F16 GEMM.

The recommended fused dispatch replaces the existing `gemm` dispatch at lines 805-813:

```cpp
// Example: gate projection in the FFN, but the same pattern applies to attn_q/k/v/output.
// Tensor is TQ4_128, shape [dim, hiddenDim] => M=1, K=dim, N=hiddenDim.

GemmTurboPushConstants gatePC = {};
gatePC.addrA      = f16NormHidden;          // F16 RMS-norm output
                                            // For attention Q: addrA = f16NormHidden
                                            // For attention output: addrA = f16AttnOut
                                            // For FFN gate: addrA = f16FfnNormHidden
                                            // For FFN down: addrA = f16SiLuMulOut

gatePC.addrB      = gateTensor.gpuAddress;  // raw TQ4_128 bytes
gatePC.addrC      = gateScratchF16;         // F16 output buffer
gatePC.M          = 1;
gatePC.K          = dim;
gatePC.N          = hiddenDim;
gatePC.blockSize  = 128;                    // TQ4_128
gatePC.bits       = 4;
gatePC.scaleBits  = 16;
gatePC.zeroPoint  = 0;
gatePC.alpha      = 1.0f;

uint32_t wgX = (hiddenDim + 31) / 32;       // gemm_turbo local_size_x=32
uint32_t wgY = 1;                           // M=1 -> local_size_y=4 covers it
scheduler->dispatchInBatch(pipelines->getPipeline("gemm_turbo"),
                           pipelines->getLayout("gemm_turbo"),
                           &gatePC, sizeof(gatePC), wgX, wgY, 1);
```

Workgroup sizing for common cases:

| Projection | M | K | N | Workgroups (x, y) |
|------------|---|---|---|-------------------|
| Q / O / down | 1 | dim | dim | `((dim+31)/32, 1, 1)` |
| K / V | 1 | dim | kvDim | `((kvDim+31)/32, 1, 1)` |
| gate / up | 1 | dim | hiddenDim | `((hiddenDim+31)/32, 1, 1)` |
| down | 1 | hiddenDim | dim | `((dim+31)/32, 1, 1)` |

#### 1.5.3 FFN weights

Current dequant location: `src/host/inference_engine.cpp:862-869`; compute at lines 875-942.

Same pattern as attention. The gate projection example above is exactly the FFN gate case. Key exact values for a TQ4_128 gate projection:

```cpp
addrA     = f16FfnNormOutput;      // output of rms_norm on hidden
addrB     = ffn_gate_weight.gpuAddress;
addrC     = gateScratchF16;        // must be F16
M         = 1;
K         = dim;
N         = hiddenDim;
blockSize = 128;
bits      = 4;
scaleBits = 16;
zeroPoint = 0;
alpha     = 1.0f;
```

For the **down projection** (`ffn_down.weight`), `K = hiddenDim`, `N = dim`.

#### 1.5.4 Required barriers

Insert `barrierBetweenGroups` at the same logical points already used in `forwardPartial()`:

1. After any `dequant_turbo.comp` write before the consumer reads it.
2. After `rms_norm` writes the F16 normalized hidden state before `gemm_turbo` reads it.
3. After `gemm_turbo` writes Q/K/V before `rope.comp` reads them.
4. After `gemm_turbo` writes the attention-output projection before the residual `add.comp` reads it.
5. After `gemm_turbo` gate/up writes before `silu_mul.comp` reads them.
6. After `silu_mul.comp` writes before the down `gemm_turbo` reads it.
7. After the down `gemm_turbo` writes before the residual `add.comp` reads it.

Use the same barrier parameters the file already uses:

```cpp
scheduler->barrierBetweenGroups(
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
```

---

## 2. Context + KV Compression Engine Wiring

### 2.1 Fix the context push-constant mismatch

`include/rdna4_compression.hpp` declares `CompressContextPushConstants` (lines 38-49) with fields for **quantization** (`addrSrc`, `addrDst`, `blockSize`, `bits`, etc.). However, the delivered shader `src/kernels/compress_context.comp` (lines 28-38) expects **compaction** push constants (`addrKSrc`, `addrVSrc`, `addrKDst`, `addrVDst`, `addrMask`, `seqLen`, `nHeads`, `headDim`, `targetLen`).

Because the shader must not be modified, update the header to match the shader. Replace / add:

```cpp
// Matches src/kernels/compress_context.comp lines 28-38
struct ContextCompactPushConstants {
    uint64_t addrKSrc;      // Source K cache (F16)
    uint64_t addrVSrc;      // Source V cache (F16)
    uint64_t addrKDst;      // Destination K cache (F16)
    uint64_t addrVDst;      // Destination V cache (F16)
    uint64_t addrMask;      // keep_mask uint8 buffer
    uint32_t seqLen;        // Current sequence length
    uint32_t nHeads;        // nKvHeads
    uint32_t headDim;
    uint32_t targetLen;     // decision.contextTargetLen
};
static_assert(sizeof(ContextCompactPushConstants) <= 128,
    "ContextCompactPushConstants exceeds Vulkan push-constant limit");
```

`KVCacheQuantizePushConstants` and `KVCacheDequantPushConstants` already match their shaders.

### 2.2 Register compression pipelines

Add to both `main.cpp` (after the existing `loadPipe` calls, ~line 186) and `test_inference.cpp` (after line 132):

```cpp
loadPipe("compress_context",  sizeof(ContextCompactPushConstants));
loadPipe("kv_cache_quantize", sizeof(KVCacheQuantizePushConstants));
loadPipe("kv_cache_dequant",  sizeof(KVCacheDequantPushConstants));
```

Add the three `.comp` files to `CMakeLists.txt` `KERNELS`:

```cmake
src/kernels/compress_context.comp
src/kernels/kv_cache_quantize.comp
src/kernels/kv_cache_dequant.comp
```

### 2.3 Add quantized KV-cache storage

The existing `KVCacheManager` (`include/rdna4_kv_cache.hpp`) only holds F16 K/V buffers. Extend it (or add an engine-side companion) with per-layer quantized storage.

Suggested additions to `KVCacheManager`:

```cpp
struct QuantizedKVBuffer {
    VkBuffer kQuant = VK_NULL_HANDLE;     // uint[] packed weights
    VkBuffer vQuant = VK_NULL_HANDLE;
    VkBuffer kScales = VK_NULL_HANDLE;    // float16_t[] per block
    VkBuffer vScales = VK_NULL_HANDLE;
    VkDeviceMemory kQuantMem = VK_NULL_HANDLE;
    VkDeviceMemory vQuantMem = VK_NULL_HANDLE;
    VkDeviceMemory kScalesMem = VK_NULL_HANDLE;
    VkDeviceMemory vScalesMem = VK_NULL_HANDLE;
    VkDeviceAddress kQuantAddr = 0;
    VkDeviceAddress vQuantAddr = 0;
    VkDeviceAddress kScalesAddr = 0;
    VkDeviceAddress vScalesAddr = 0;
    bool isQuantized = false;
};
std::vector<QuantizedKVBuffer> quantLayers;

// Allocate using the same DEVICE_LOCAL + DEVICE_ADDRESS pattern as allocate().
// Sizes for blockSize=32:
//   nBlocks   = (maxSeqLen + 31) / 32
//   kQuantBytes = nKvHeads * nBlocks * (32 * headDim / 8) * sizeof(uint32_t)
//   scalesBytes = nKvHeads * nBlocks * sizeof(uint16_t)
```

### 2.4 Instantiate `CompressionScheduler`

Add to `InferenceEngine` (`include/rdna4_engine.hpp`):

```cpp
#include "rdna4_compression_scheduler.hpp"

class InferenceEngine {
    ...
    std::unique_ptr<CompressionScheduler> compressor_;
    uint32_t maxContext_ = 0;

public:
    bool initCompression(const ContextCompressionConfig& ctxCfg,
                         const KVCompressionConfig& kvCfg,
                         uint32_t maxContext);
private:
    void maybeCompress(uint32_t seqLen);
    ...
};
```

Call from `main.cpp` after KV cache allocation (around line 152):

```cpp
ContextCompressionConfig ctxCfg;   // parse from fullJson["context_compression"] if present
KVCompressionConfig kvCfg;         // parse from fullJson["kv_compression"] if present
engine.initCompression(ctxCfg, kvCfg, maxContext);
```

If the JSON does not contain compression sections, pass default-constructed configs (`enabled=false`). The scheduler will then always return a no-op decision.

### 2.5 Call `step()` every token

Add `maybeCompress()` and call it from `forward()` (`src/host/inference_engine.cpp:1148-1153`):

```cpp
uint32_t InferenceEngine::forward(uint32_t tokenId, uint32_t seqPos) {
    uint32_t nextToken = 0;
    if (kernelEntryReady) {
        nextToken = forwardKernelEntry(tokenId, seqPos);
    } else {
        nextToken = forwardPartial(tokenId, seqPos, model->blockCount);
    }
    maybeCompress(seqPos + 1);
    return nextToken;
}
```

This keeps the decision logic outside the layer hot path and ensures KV cache sequence lengths have already been incremented by `forwardPartial()` (`inference_engine.cpp:995-997`).

### 2.6 `maybeCompress()` dispatcher

```cpp
void InferenceEngine::maybeCompress(uint32_t seqLen) {
    if (!compressor_) return;

    CompressionDecision decision = compressor_->step(seqLen, maxContext_);
    if (!decision.compressContext && !decision.compressKV) return;

    // It is safest to end the current batch and issue compression in a fresh
    // command buffer, because compression is rare and correctness matters more
    // than batching it into the token forward pass.
    scheduler->beginBatch(0);

    if (decision.compressContext) {
        dispatchContextCompression(decision, seqLen);
    }
    if (decision.compressKV) {
        dispatchKVQuantization(decision, seqLen);
    }

    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAll();
}
```

### 2.7 Context compression dispatch sequence (`decision.compressContext`)

Inputs:

- `decision.keepMask`: `std::vector<uint8_t>` of length `seqLen`, `1 = keep`, `0 = drop`.
- `decision.contextTargetLen`: desired length after compaction.

Steps:

1. **Upload keep mask.** The shader reads masks as bytes packed into `uint` words (`compress_context.comp:59-61`). Allocate a small host-visible GPU buffer of `((seqLen + 3) / 4) * 4` bytes and `memcpy` the vector directly.

2. **Allocate compacted KV buffers.** For each layer, allocate F16 buffers sized for `targetLen * nKvHeads * headDim` elements. Reuse pre-allocated temp buffers if possible.

3. **Dispatch `compress_context.comp` per layer.**

```cpp
ContextCompactPushConstants pc = {};
pc.addrKSrc    = kvCache->getKBufferAddress(layer);
pc.addrVSrc    = kvCache->getVBufferAddress(layer);
pc.addrKDst    = compactedKAddr;
pc.addrVDst    = compactedVAddr;
pc.addrMask    = maskAddr;
pc.seqLen      = seqLen;
pc.nHeads      = model->headCountKv;
pc.headDim     = headDim;
pc.targetLen   = decision.contextTargetLen;

scheduler->dispatchInBatch(pipelines->getPipeline("compress_context"),
                           pipelines->getLayout("compress_context"),
                           &pc, sizeof(pc), 1, 1, 1);  // shader uses one workgroup
```

4. **Swap / update state.** After `syncAll()`, either:
   - Copy the compacted buffers back over the original KV cache, or
   - Update `KVCacheManager` to use the compacted buffers as the new canonical buffers and free the originals.

   In either case, set `kvCache->layers[layer].currentSeqLen = decision.contextTargetLen`.

**Fallback:**

- If `seqLen > 2048`, do **not** dispatch `compress_context.comp`; the v1 shader uses a fixed `shared uint shPrefix[2048]` and will fail. Fall back to no context compression for this token.
- If any allocation or dispatch fails, log and continue with the uncompressed KV cache.

### 2.8 KV compression dispatch sequence (`decision.compressKV`)

Inputs:

- `decision.kvQuantizeBits`: bits to use (the delivered `kv_cache_quantize.comp` v1 is hard-coded for Q4_0, so this must be 4).
- Current `seqLen`.

Steps:

1. **Ensure quantized buffers exist** (size computed in §2.3).

2. **Dispatch `kv_cache_quantize.comp` per layer.**

```cpp
KVCacheQuantizePushConstants pc = {};
pc.addrKSrc    = kvCache->getKBufferAddress(layer);      // F16 source
pc.addrVSrc    = kvCache->getVBufferAddress(layer);
pc.addrKDst    = kvCache->quantLayers[layer].kQuantAddr; // uint[] packed
pc.addrVDst    = kvCache->quantLayers[layer].vQuantAddr;
pc.addrKScales = kvCache->quantLayers[layer].kScalesAddr;
pc.addrVScales = kvCache->quantLayers[layer].vScalesAddr;
pc.seqLen      = seqLen;
pc.nHeads      = model->headCountKv;
pc.headDim     = headDim;
pc.blockSize   = 32;  // matches kv_cache_quantize.comp v1

uint32_t wg = (model->headCountKv + 63) / 64;  // local_size_x=64, one thread per head
scheduler->dispatchInBatch(pipelines->getPipeline("kv_cache_quantize"),
                           pipelines->getLayout("kv_cache_quantize"),
                           &pc, sizeof(pc), wg, 1, 1);
```

3. **Mark the layer as quantized.** `kvCache->quantLayers[layer].isQuantized = true`.

4. **Before the next token's attention**, if the layer is quantized, dequantize into a temporary F16 buffer and pass that to `attention.comp`:

```cpp
uint32_t n = seqLen * model->headCountKv * headDim;
KVCacheDequantPushConstants kpc = {};
kpc.addrSrc    = kvCache->quantLayers[layer].kQuantAddr;
kpc.addrDst    = tempKCacheF16Addr;
kpc.n          = n;
kpc.blockSize  = 32;
kpc.bits       = 4;
kpc.scaleBits  = 16;
kpc.zeroPoint  = 0;
scheduler->dispatchInBatch(pipelines->getPipeline("kv_cache_dequant"),
                           pipelines->getLayout("kv_cache_dequant"),
                           &kpc, sizeof(kpc), (n + 255) / 256, 1, 1);  // local_size_x=256

// Repeat for V, then barrier before attention.comp reads tempKCacheF16Addr / tempVCacheF16Addr.
```

**Fallback:**

- If quantization buffer allocation fails, leave `isQuantized = false` and continue using the F16 cache.
- The v1 quantize shader quantizes the **entire** cache. A production v2 should add a `startPos`/`endPos` so recent tokens stay F16. Until then, the simplest safe behavior is to quantize the whole cache when triggered.

### 2.9 Keep the uncompressed path untouched

- Default configs have `enabled = false`.
- `maybeCompress()` returns immediately if `compressor_` is null or both decision flags are false.
- No existing `forwardPartial()` layer logic changes unless a Tensor is detected as TurboQuant.
- Context compression only runs when `seqLen > maxContext * ctxCfg.threshold`.
- KV compression only runs when `kvCfg.enabled && seqLen >= kvCfg.minSeqLen`.

---

## 3. Verification Checklist

- [ ] `docs/wiring_plan.md` exists and contains both sections.
- [ ] `TurboQuantPushConstants` and `GemmTurboPushConstants` are added to `include/rdna4_types.hpp` and match the GLSL layouts.
- [ ] `dequant_turbo.comp`, `gemm_turbo.comp`, `compress_context.comp`, `kv_cache_quantize.comp`, `kv_cache_dequant.comp` are in `CMakeLists.txt` `KERNELS`.
- [ ] `main.cpp` and `test_inference.cpp` load all five new pipelines with the correct push-constant sizes.
- [ ] `ContextCompactPushConstants` matches `compress_context.comp` (the existing `CompressContextPushConstants` does not).
- [ ] Build succeeds: `cd build && cmake --build . --config Release`.
- [ ] SPIR-V files are copied to `build/Release/shaders/`.

---

## 4. Open Questions / Risks

1. **F16 activation mismatch.** `gemm_turbo.comp` and `dequant_turbo.comp` use `float16_t` I/O. The current engine uses `float32` activations and elementwise shaders (`rms_norm.comp`, `add.comp`, `silu_mul.comp`, `embed.comp`). A clean integration requires either F16⇄F32 conversion helpers or a full F16 activation pipeline.
2. **`CompressContextPushConstants` mismatch.** The header struct describes quantization, but `compress_context.comp` performs keep-mask compaction. The header must be reconciled with the shader before the host can safely dispatch it.
3. **`compress_context.comp` seqLen limit.** The shader declares `shared uint shPrefix[2048]` and is documented for `seqLen <= 2048`. The host must clamp or fall back when `seqLen > 2048`.
4. **`kv_cache_quantize.comp` v1 quantizes everything.** There is no host-controlled carve-out for recent tokens. Future work should add `startPos`/`endPos` push constants or a two-tier buffer layout.
5. **Mixed precision attention.** `attention.comp` reads K/V cache as `float16_t` but the engine currently passes float32 Q/K/V row addresses. The wiring plan assumes the attention shader inputs are eventually made consistent; otherwise the dequant-to-F16 path for KV compression will still be misinterpreted.
6. **No conversion shader delivered.** If Path A is chosen, `f16_to_f32.comp` and `f32_to_f16.comp` must be written.

---

## 5. Recommended Next Concrete Step for the Implementation Agent

**Add the missing host push-constant structs and fix the pipeline registrations.** Specifically:

1. Insert `TurboQuantPushConstants` and `GemmTurboPushConstants` into `include/rdna4_types.hpp`.
2. Add the five new `.comp` files to `CMakeLists.txt`.
3. Update `main.cpp` and `test_inference.cpp` `loadPipe` calls with the correct sizes.
4. Replace `CompressContextPushConstants` in `include/rdna4_compression.hpp` with `ContextCompactPushConstants` matching `compress_context.comp`.
5. Build and confirm all SPIR-V files compile cleanly.

Only after the build is clean should the agent proceed to wire the dispatch logic in `forwardPartial()` and `maybeCompress()`.
