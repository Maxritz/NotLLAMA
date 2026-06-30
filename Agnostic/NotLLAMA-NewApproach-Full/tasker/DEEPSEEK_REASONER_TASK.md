# DeepSeek-Reasoner — Architecture Analysis

## InferenceEngine God Class Analysis

### Size and Scope
- **1166 lines** total (inference_engine.cpp + 127 lines header)
- **20+ public methods**, 5 private methods
- **15 VkBuffer + VkDeviceMemory + VkDeviceAddress triplets** stored as flat members
- **7 raw pointers** to other subsystems (ctx, model, kvCache, pipelines, tokenizer, scheduler, allocator)
- **forwardPartial() alone**: 350+ lines

### Responsibility Map

| Responsibility | Lines | Members | Methods | Severity |
|---|---|---|---|---|
| **Buffer creation boilerplate** | 110 (9.4%) | — | `createBufferBDA()`, `createBufferHostVisible()` | DUPLICATED — each *init* function rewrites the same pattern |
| **Dequant staging buffer** | 95 | `dequantBuffer/Memory/Addr/Capacity` | `initDequantBuffer()`, `cleanupDequantBuffer()` | ISOLATED — easy to extract |
| **Embed cache** | 110 | `embedCacheBuffer/Memory/Addr/Size/Ready` | `initEmbedCache()`, `cleanupEmbedCache()` | ISOLATED but has dequant dispatch coupling |
| **Weight buffer (disabled)** | 20 | `weightBuffer/Memory/Addr/Size` | `initWeightBuffer()`(no-op), `cleanupWeightBuffer()` | DISABLED — no-op, easy to extract |
| **Layer params** | 115 | `layerParamsBuffer/Memory/Addr` | `initLayerParams()` | COUPLED — walks weight buffer + KV cache + model tensors |
| **Kernel entry buffers** | 100 | `mailbox/outputToken/scratch/hiddenState/logits` (15 members) | `initKernelEntryBuffers()`, `cleanupKernelEntryBuffers()` | ISOLATED — discrete set of buffers |
| **Dequant dispatch helper** | 50 (static) | — | `dequantWeight()`, `findTensorAddr()`, `findTensor()`, `getF16OutputSize()` | CROSS-CUTTING — called from forwardPartial + initEmbedCache |
| **forwardPartial** | 350+ | `lastLogitsAddr/Offset` | `forwardPartial()` | THE CORE — embeds knowledge of: tensor names, allocator, pipelines, scheduler, KV cache, dequant |
| **forwardKernelEntry** | 80 | — | `forwardKernelEntry()` | BROKEN — depends on disabled weight buffer |
| **Sampling** | 65 | — | `sampleGpu()`, `sampleArgmax()` | ISOLATED — depends only on pipelines + scheduler + allocator |
| **Generation loops** | 130 | — | `generate()`, `generateSpeculative()`, `forwardSpeculative()`, `draftForward()`, `verifyDraftToken()` | ISOLATED — wraps forward() |
| **Engine state / dispatch** | 10 | `kernelEntryReady` | `forward()`, `cleanup()` | THIN — trivial |

### Duplicated Patterns (Technical Debt)

1. **Buffer creation** — `initDequantBuffer()` (lines 96-186) and `initEmbedCache()` (196-299) both inline the same VkBufferCreateInfo → memory type search → VkMemoryAllocateFlagsInfo → vkAllocateMemory → vkBindBufferMemory → device address query pattern. `initKernelEntryBuffers()` uses the `createBufferBDA()` helper but `initDequantBuffer()` and `initEmbedCache()` do not.

2. **Memory type search** — DEVICE_LOCAL vs HOST_VISIBLE+DEVICE_LOCAR preference logic duplicated in 4 places.

3. **Cleanup pattern** — Each `cleanup*()` method follows the same destroy-null pattern. The `cleanup()` method is a manual registry.

### Coupling Analysis (What Talks to What)

```
forwardPartial() reads/writes:
  → allocator (ring buffer alloc/free)
  → scheduler (dispatch + sync)
  → pipelines (getPipeline + getLayout)
  → model (tensor addresses, blockCount, dims)
  → kvCache (KV buffer addresses, seqLen)
  → dequantWeight() static helper (tensor names, quant formats)
  → sampleGpu() (pipelines, allocator)

initLayerParams() reads:
  → weightBufferAddr (0 when disabled — BROKEN)
  → model tensors (all 11 per layer)
  → kvCache (buffer addresses)

initEmbedCache() reads/writes:
  → model (token_embd tensor)
  → scheduler + pipelines (dequant dispatch)
```

### Proposed Split: 4 Classes

#### 1. BufferFactory (NEW — extracted from helpers)
**Purpose:** Single source of truth for Vulkan buffer creation with device address.
**Extracted from:** `createBufferBDA()` (line 313), `createBufferHostVisible()` (line 359), memory type search boilerplate.
**Contents:**
```
class BufferFactory {
    VkDevice device;
    VkPhysicalDevice physDev;

    VkBuffer createStorageBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem,
                                  bool hostVisible = false, void** mapped = nullptr);
    void destroyBuffer(VkBuffer buf, VkDeviceMemory mem);
};
```
**Impact:** Removes ~100 lines of duplicated Vulkan boilerplate from InferenceEngine. All `init*()` methods switch to BufferFactory calls.
**File:** `src/host/buffer_factory.cpp`, `include/rdna4_buffer_factory.hpp`
**Effort:** S (1-2 hours, pure extraction, no logic change)

#### 2. WeightManager (NEW — extracted from InferenceEngine)
**Purpose:** Manages quantized weight metadata, dequant staging buffer, embed cache, layer params. Knows tensor names and format conversions, but NOT inference.
**Extracted from:** `initDequantBuffer()` (line 96), `initEmbedCache()` (196), `cleanupDequantBuffer()` (188), `cleanupEmbedCache()` (301), `initWeightBuffer()` (413 — disabled), `cleanupWeightBuffer()` (421), `initLayerParams()` (433), all static helpers: `findTensorAddr()`, `findTensor()`, `getF16OutputSize()`, `dequantWeight()`.
**Contents:**
```
class WeightManager {
    VulkanContext* ctx;
    ModelDesc* model;
    Scheduler* scheduler;
    PipelineBuilder* pipelines;
    KVCacheManager* kvCache;
    BufferFactory* bufFactory;

    // Dequant staging buffer
    VkBuffer dequantBuffer; VkDeviceMemory dequantMemory;
    VkDeviceAddress dequantAddr; size_t dequantCapacity;

    // Embed cache
    VkBuffer embedCacheBuffer; VkDeviceMemory embedCacheMemory;
    VkDeviceAddress embedCacheAddr; size_t embedCacheSize; bool embedCacheReady;

    // Weight buffer (DISABLED — kept for kernel_entry)
    VkBuffer weightBuffer; VkDeviceMemory weightMemory;
    VkDeviceAddress weightBufferAddr; size_t weightBufferSize;

    // Layer params (kept for kernel_entry)
    VkBuffer layerParamsBuffer; VkDeviceMemory layerParamsMemory;
    VkDeviceAddress layerParamsAddr;

    bool initDequantBuffer();
    void cleanupDequantBuffer();
    bool initEmbedCache();
    void cleanupEmbedCache();
    bool initWeightBuffer();     // DISABLED
    void cleanupWeightBuffer();
    bool initLayerParams();      // depends on kvCache addresses
    void cleanupLayerParams();
    void cleanup();

    // Dequant single tensor → staging buffer, returns address
    uint64_t dequantTensor(const std::string& name, size_t outOffset = 0);
    // Find tensor address (original quantized or cached dequantized)
    uint64_t getTensorAddr(const std::string& name);
    // Lookup helpers
    const TensorDesc* findTensor(const std::string& name);
    size_t getF16OutputSize(const std::string& name);
};
```
**Impact:** Removes ~400 lines from InferenceEngine. forwardPartial() calls `weightManager.dequantTensor()` instead of inline `dequantWeight()`. WeightManager is testable standalone with CPU reference.
**File:** `src/host/weight_manager.cpp`, `include/rdna4_weight_manager.hpp`
**Effort:** M (3-4 hours — needs careful interface design for forwardPartial callers)

#### 3. GpuResourceManager (NEW — owns kernel_entry buffers)
**Purpose:** Owns the kernel_entry.comp persistent buffers (mailbox, output, scratch, hidden state, logits). Gatekeeps the kernel_entry dispatch path.
**Extracted from:** `initKernelEntryBuffers()` (line 551), `cleanupKernelEntryBuffers()` (606), mailbox/output/scratch/hidden/logits members.
**Contents:**
```
class GpuResourceManager {
    VulkanContext* ctx;
    BufferFactory* bufFactory;
    ModelDesc* model;

    // kernel_entry buffers
    VkBuffer mailboxBuffer; VkDeviceMemory mailboxMemory; VkDeviceAddress mailboxAddr;
    VkBuffer outputTokenBuffer; VkDeviceMemory outputTokenMemory; VkDeviceAddress outputTokenAddr;
    VkBuffer scratchBuffer; VkDeviceMemory scratchMemory; VkDeviceAddress scratchAddr;
    VkBuffer hiddenStateBuffer; VkDeviceMemory hiddenStateMemory; VkDeviceAddress hiddenStateAddr;
    VkBuffer logitsBuffer; VkDeviceMemory logitsMemory; VkDeviceAddress logitsAddr;
    float* logitsMapped;

    bool init(ModelDesc* model);   // allocates all kernel_entry buffers
    void cleanup();
    bool isReady() const;
    uint64_t logitsAddress() const;
    float* logitsMappedPtr() const;
};
```
**Impact:** Removes ~100 lines + 15 flat members from InferenceEngine. Kernel-entry path uses `gpuResources->logitsAddress()` etc.
**File:** `src/host/gpu_resource_manager.cpp`, `include/rdna4_gpu_resource_manager.hpp`
**Effort:** S (1-2 hours)

#### 4. InferenceEngine (CORE — slimmed down)
**Purpose:** Pure inference orchestration. Forward pass, sampling, generation loops. No Vulkan boilerplate, no buffer management, no weight format knowledge.
**Retains:**
```
class InferenceEngine {
    // Dependencies (unchanged)
    VulkanContext* ctx;
    ModelDesc* model;
    KVCacheManager* kvCache;
    PipelineBuilder* pipelines;
    Tokenizer* tokenizer;
    Scheduler* scheduler;
    RingAllocator* allocator;

    // NEW: extracted managers
    WeightManager* weights;           // created before engine
    GpuResourceManager* gpuRes;       // created before engine (optional)
    BufferFactory* bufFactory;        // shared utility

    bool kernelEntryReady;            // true if gpuRes + layerParams ready

    // METHODS — trimmed
    uint32_t forward(uint32_t tokenId, uint32_t seqPos);
    std::vector<uint32_t> generate(const std::string& prompt, uint32_t maxTokens);
    std::vector<uint32_t> generateSpeculative(..., uint32_t nDraft);
    uint32_t sampleGpu(uint64_t logitsAddr, uint32_t vocabSize, ...);
    uint32_t sampleArgmax(const float* logits, uint32_t vocabSize);

    uint64_t lastLogitsAddr;
    size_t lastLogitsOffset;

private:
    uint32_t forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers);
    uint32_t forwardKernelEntry(uint32_t tokenId, uint32_t seqPos);
    uint32_t draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers);
    uint32_t verifyDraftToken(uint32_t draftToken, uint32_t seqPos);
    std::vector<uint32_t> forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft);
};
```
**`forwardPartial()` after extraction** (~250 lines instead of 350):
- No `dequantWeight()` calls → `weights->dequantTensor(name)`
- No `createBufferBDA` helpers
- No `findTensorAddr()` calls → `weights->getTensorAddr(name)` or `weights->dequantTensor(name)`
- No `getF16OutputSize()` → `weights->getF16OutputSize(name)`
- `allocator->reset()` + allocs + shader dispatches + syncs → pure compute orchestration

### Migration Plan (Minimal Disruption)

**Phase 1 — BufferFactory (S, 1-2h)**
1. Create `buffer_factory.cpp/hpp`
2. Move `createBufferBDA` and `createBufferHostVisible` there
3. Refactor `initDequantBuffer()` and `initEmbedCache()` to use BufferFactory
4. Verify: build + test_inference passes

**Phase 2 — WeightManager (M, 3-4h)**
1. Create `weight_manager.cpp/hpp`
2. Move: `dequantWeight()`, `findTensorAddr()`, `findTensor()`, `getF16OutputSize()`
3. Move: `initDequantBuffer/cleanupDequantBuffer`, `initEmbedCache/cleanupEmbedCache`, `initWeightBuffer/cleanupWeightBuffer`
4. Move: `initLayerParams()` (needs KV cache reference — pass as dependency)
5. Create `WeightManager::dequantTensor()` as public API
6. Update `forwardPartial()` to call `weights->dequantTensor()` and `weights->getTensorAddr()`
7. Update `forwardKernelEntry()` to use `weights->layerParamsAddr`
8. Verify: build + test_inference passes, GPU forward still 10ms

**Phase 3 — GpuResourceManager (S, 1-2h)**
1. Create `gpu_resource_manager.cpp/hpp`
2. Move kernel_entry buffer members + init/cleanup
3. Wire into InferenceEngine constructor
4. Verify: build passes, no functional change (kernel_entry is disabled)

**Phase 4 — Cleanup (S, 1h)**
1. Remove all moved members from InferenceEngine header
2. Update `cleanup()` to delegate to WeightManager::cleanup() + GpuResourceManager::cleanup()
3. Remove `createBufferBDA` and `createBufferHostVisible` from inference_engine.cpp
4. Document new dependency graph in AGENTS.md

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| forwardPartial breaks during refactor | MEDIUM | HIGH | Phase 2 changes are mechanical (replace inline calls with equivalent weightManager calls). Verify after each change. |
| WeightManager needs KV cache pointer for LayerParams | LOW | MEDIUM | Pass KVCacheManager* to WeightManager constructor or as late init param |
| kernel_entry path gets harder to fix after split | LOW | LOW | kernel_entry is already broken; split doesn't change that. Easy to fix when weight buffer is re-enabled. |
| Circular dependency: WeightManager → Scheduler/Pipelines | LOW | LOW | WeightManager takes scheduler+pipelines as constructor deps (same as before) |

### Summary

| Class | Current Lines | After Split | New File |
|---|---|---|---|
| InferenceEngine | 1166 | ~400 | `inference_engine.cpp` (trimmed) |
| BufferFactory | 0 | ~80 | `buffer_factory.cpp` (NEW) |
| WeightManager | 0 | ~400 | `weight_manager.cpp` (NEW) |
| GpuResourceManager | 0 | ~120 | `gpu_resource_manager.cpp` (NEW) |

**Net change:** +600 lines of new files, −700 lines from InferenceEngine. No new functionality — pure decomposition.
