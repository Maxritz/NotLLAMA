# NotLLAMA Phase 2: Architectural Rewrite to Match llama.cpp Performance

## Current State vs llama.cpp

| Metric | NotLLAMA (after Fix 1) | llama.cpp | Gap |
|--------|----------------------|-----------|-----|
| Forward pass | ~5,000 ms/token | ~6.8 ms/token | **735× slower** |
| vkQueueSubmit per token | ~720 (20/layer × 36) | ~1 | **720× more** |
| vkWaitForFences per token | ~72 (2/layer × 36) | ~1 | **72× more** |
| Weight dequant per token | ALL weights, EVERY token | ONCE at load | **36× more** |
| GEMM shader | Scalar loop, 1 element/thread | Tiled shared memory, 64× throughput | **~50× slower** |
| Attention | Full O(N²) materialization | Flash Attention (memory-efficient) | **~10× slower** |

## Root Cause: Architecture Mismatch

llama.cpp's Vulkan backend uses **ONE command buffer per forward pass** with:
1. All dispatches recorded into a single `VkCommandBuffer`
2. `vkCmdPipelineBarrier` between dependent operations (no CPU sync)
3. **ONE** `vkQueueSubmit` per token
4. **ONE** `vkWaitForFences` per token (after sampling)
5. Pre-dequantized weights stored in GPU memory at load time

NotLLAMA uses **individual submit per dispatch**:
1. `vkBeginCommandBuffer` → `vkCmdDispatch` → `vkEndCommandBuffer` → `vkQueueSubmit` → `vkWaitForFences`
2. Repeated 720 times per token
3. Each submit has ~50μs CPU overhead → 720 × 50μs = **36ms of pure CPU overhead**
4. Plus GPU idle time between submits → **~5s total**

## Phase 2A: Single Command Buffer Per Forward Pass

**Impact: ~5s → ~500ms/token (10×)**

Rewrite `forwardPartial()` to record the ENTIRE forward pass into one `VkCommandBuffer`:

```cpp
uint32_t InferenceEngine::forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers) {
    // ... setup unchanged ...

    // === SINGLE COMMAND BUFFER FOR ENTIRE FORWARD PASS ===
    VkCommandBuffer cmd = scheduler->allocateCmd(0);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Embedding (recorded into cmd, NOT submitted)
    if (embedAddr) {
        EmbedPushConstants embedPC = {embedAddr_dq, hiddenAddr, tokenId, seqPos, dim};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines->getPipeline("embed"));
        vkCmdPushConstants(cmd, pipelines->getLayout("embed"), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(embedPC), &embedPC);
        vkCmdDispatch(cmd, (dim + 31) / 32, 1, 1);
    }

    for (uint32_t layer = 0; layer < layersToRun; ++layer) {
        // All attention dispatches recorded into SAME cmd buffer
        // NO vkQueueSubmit between them — only vkCmdPipelineBarrier

        // RMS Norm
        vkCmdPipelineBarrier(cmd, 
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 0, nullptr);
        // ... record norm dispatch ...

        // Q GEMM
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 0, nullptr);
        // ... record Q GEMM ...

        // K GEMM
        // ... etc ...

        // NO SYNC. Just barriers. The GPU executes everything in order.
    }

    // Final norm + LM head + sampling
    // ... record ...

    vkEndCommandBuffer(cmd);

    // === ONE SUBMIT FOR ENTIRE FORWARD PASS ===
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VkFence fence = fencePool->acquire();
    vkQueueSubmit(queues[0], 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    fencePool->release(fence);

    // Read back token from logits
    // ...
}
```

This eliminates 719 of 720 `vkQueueSubmit` calls and 71 of 72 `vkWaitForFences` calls.

## Phase 2B: Pre-dequantize All Weights at Load Time

**Impact: ~500ms → ~100ms/token (5×)**

Currently, every token dequantizes:
- 4 attention weights per layer (Q, K, V, O) × 36 layers = 144 dequants
- 3 FFN weights per layer (Up, Gate, Down) × 36 layers = 108 dequants
- Total: **252 dequant dispatches per token**

With pre-dequantization:
- All weights dequantized ONCE after `load()`
- Stored in persistent GPU buffers
- Forward pass reads pre-dequantized float32 directly
- **252 dequant dispatches → 0 per token**

Implement `initWeightBuffer()`:

```cpp
bool InferenceEngine::initWeightBuffer() {
    // Calculate total size of all dequantized weights
    size_t totalBytes = 0;
    for (auto& t : model->tensors) {
        if (t.format == QuantFormat::F16 || t.format == QuantFormat::F32) continue;
        uint32_t nElements = 1;
        for (auto d : t.shape) nElements *= d;
        totalBytes += (size_t)nElements * sizeof(float);
    }

    // Allocate persistent GPU buffer
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = totalBytes;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    // ... allocate, bind, get address ...

    // Dequantize all weights in ONE batch
    scheduler->beginBatch(0);
    size_t offset = 0;
    for (auto& t : model->tensors) {
        if (t.format == QuantFormat::F16 || t.format == QuantFormat::F32) {
            weightAddrs[t.name] = t.gpuAddress;  // Already float
            continue;
        }
        uint32_t nElements = 1;
        for (auto d : t.shape) nElements *= d;
        // ... dispatch dequant for this tensor ...
        weightAddrs[t.name] = weightBufferAddr + offset;
        offset += (size_t)nElements * sizeof(float);
    }
    scheduler->endBatch(VK_NULL_HANDLE);
    scheduler->syncAll();

    weightBufferSize = totalBytes;
    return true;
}
```

Then in `forwardPartial()`, replace all `dequantWeight()` calls with `weightAddrs[name]` lookups.

## Phase 2C: Optimize GEMM Shader (Tiled Shared Memory)

**Impact: ~100ms → ~20ms/token (5×)**

Current `gemm.comp` is a naive scalar loop. Replace with a tiled GEMM using `shared` memory:

```glsl
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Tile size: 16×16 = 256 threads per workgroup
// Each thread computes one 16×16 tile of output
// Uses shared memory for A and B tiles

shared float tileA[16][16];
shared float tileB[16][16];

void main() {
    uint row = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint col = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;

    float acc = 0.0;
    for (uint t = 0; t < pc.K; t += 16) {
        // Load A tile into shared memory
        tileA[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
            (row < pc.M && t + gl_LocalInvocationID.x < pc.K) 
            ? A.data[row * pc.K + t + gl_LocalInvocationID.x] : 0.0;

        // Load B tile into shared memory
        tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
            (col < pc.N && t + gl_LocalInvocationID.y < pc.K)
            ? B.data[(t + gl_LocalInvocationID.y) * pc.N + col] : 0.0;

        barrier();

        for (uint k = 0; k < 16; ++k) {
            acc += tileA[gl_LocalInvocationID.y][k] * tileB[k][gl_LocalInvocationID.x];
        }

        barrier();
    }

    if (row < pc.M && col < pc.N) {
        C.data[row * pc.N + col] = pc.alpha * acc;
    }
}
```

For M=1 (matvec), this is still not optimal. A better approach for M=1 is to use **cooperative subgroup operations**:

```glsl
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Each workgroup (256 threads) computes a block of N outputs
// Thread i reads A[i] and B[i*N + col] for its assigned columns
// Uses subgroup shuffle for reduction

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint wg = gl_WorkGroupID.x;
    uint colsPerWg = 256;  // Each workgroup handles 256 output columns
    uint colStart = wg * colsPerWg;

    float myA[8];  // Each thread caches 8 elements of A
    for (uint k = tid; k < pc.K; k += 256) {
        myA[k / 256] = A.data[k];  // Simplified
    }

    // Each thread computes partial sum for its assigned columns
    // Then subgroup-reduce across threads
    // ...
}
```

Actually, the simplest high-performance matvec for RDNA4 is:
- Workgroup size: 64 (1 wave)
- Each thread computes a contiguous block of N/64 outputs
- Read A once (broadcast via L1), read B coalesced

```glsl
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint nThreads = 64;
    uint nCols = pc.N;

    // Each thread handles a strided subset of columns
    for (uint col = tid; col < nCols; col += nThreads) {
        float acc = 0.0;
        for (uint k = 0; k < pc.K; ++k) {
            acc += A.data[k] * B.data[k * nCols + col];
        }
        C.data[col] = pc.alpha * acc;
    }
}
```

This is the same as the current shader but with 64 threads instead of 32. The real win is the **single command buffer** and **pre-dequantization**.

## Phase 2D: Flash Attention

**Impact: ~20ms → ~10ms/token (2×)**

The `flash_attention.comp` shader exists but is not wired. Flash attention:
1. Does NOT materialize the full Q×K attention matrix
2. Processes attention in blocks that fit in shared memory / cache
3. Uses online softmax to avoid numerical instability

For a 2048-dim model with 2048 context length:
- Naive attention: Q×K = 2048 × 2048 = 4M elements = 16MB
- Flash attention: Processes in 128×128 tiles = ~0.5MB per tile

Wiring flash attention requires:
1. Dispatch `flash_attention.comp` instead of `attention.comp`
2. Pass correct tile sizes (e.g., 128 for RDNA4)
3. Ensure KV cache is in the right layout

## Expected Timeline

| Phase | Change | Expected Speed | Effort |
|-------|--------|----------------|--------|
| Current (fixed) | Per-dispatch submit + per-token dequant | ~5s/token | — |
| 2A | Single command buffer per forward pass | ~500ms/token | Medium |
| 2B | Pre-dequantize all weights | ~100ms/token | Medium |
| 2C | Tiled GEMM shader | ~20ms/token | High |
| 2D | Flash Attention | ~10ms/token | Medium |
| 2E | Multi-command buffer pipelining | ~7ms/token | High |
| **llama.cpp** | All of above + years of tuning | **~6.8ms/token** | — |

## Recommendation

Apply **Phase 2A + 2B** first. These are the biggest wins (5s → 100ms, **50× improvement**) and are architectural changes that don't require shader rewriting. They get you to **10% of llama.cpp speed** with moderate effort.

Phase 2C (tiled GEMM) and 2D (flash attention) require shader expertise and extensive testing. They get you from 10% to 100% of llama.cpp.
