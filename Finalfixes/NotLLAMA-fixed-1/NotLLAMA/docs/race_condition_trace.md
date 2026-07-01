# RACE CONDITION & LOAD TRACE

## ISSUE 1: Ring Buffer Overflow (CRITICAL)

### Current Flow
```
allocator->reset()                              # offset = 0
hiddenAddr = allocator->alloc(dim * 4)          # ~16KB
qAddr = allocator->alloc(qkvSize)               # ~128KB
kAddr = allocator->alloc(kvSize)                # ~16KB
vAddr = allocator->alloc(kvSize)                # ~16KB
attnOutAddr = allocator->alloc(dim * 4)         # ~16KB
mlpOutAddr = allocator->alloc(dim * 4)          # ~16KB
logitsAddr = allocator->alloc(vocabSize * 4)    # ~2MB
sampleOutAddr = allocator->alloc(16)            # 16 bytes

FOR EACH LAYER (36 layers):
  # Staging buffers allocated IN the loop, NEVER freed
  Q staging:   dim * dim * 4 = 16MB            # ALLOCATED
  K staging:   kvDim * dim * 4 = 2MB           # ALLOCATED
  V staging:   kvDim * dim * 4 = 2MB           # ALLOCATED
  O staging:   dim * dim * 4 = 16MB            # ALLOCATED
  gate staging: hiddenDim * dim * 4 = 44MB     # ALLOCATED
  up staging:  hiddenDim * dim * 4 = 44MB      # ALLOCATED
  down staging: dim * hiddenDim * 4 = 44MB     # ALLOCATED

  Total per layer: ~168MB
  Total for 36 layers: ~6GB
  Ring buffer: 64MB
  RESULT: OVERFLOW AFTER LAYER 0
```

### Fix: Checkpoint/Restore Pattern
```
allocator->reset()
# Allocate activations ONCE
hiddenAddr = allocator->alloc(...)
qAddr = allocator->alloc(...)
...
logitsAddr = allocator->alloc(...)

FOR EACH LAYER:
  size_t checkpoint = allocator->offset    # SAVE

  # Allocate staging for this layer
  qStaging = allocator->alloc(dim * dim * 4)
  kStaging = allocator->alloc(...)
  ...

  # Dequant + GEMM
  dequantIfNeeded(... qStaging)
  gemm(... qStaging)

  # Restore checkpoint (reuse space for next layer)
  allocator->offset = checkpoint           # RESTORE
```

## ISSUE 2: Barrier Placement (OK)

### Current Flow
```
dequantIfNeeded():
  dispatch(dequant)                         # WRITE to staging
  barrierBetweenGroups(WRITE, READ)         # BARRIER

return stagingAddr

dispatch(gemm)                              # READ from staging
```

### Analysis
- Barrier is INSIDE dequantIfNeeded
- Barrier completes BEFORE GEMM dispatch
- Same command buffer, same queue
- **VERDICT: CORRECT** - no race condition

## ISSUE 3: GPU Load (OK)

### Dequant Dispatch
```
nElements = 524288 (for Q4_0 2048x2048 weight)
workgroups = (524288 + 63) / 64 = 8192
threads = 8192 * 64 = 524288

GPU: RX 9070 XT
Compute units: 64
Wavefronts per CU: 32
Max concurrent wavefronts: 64 * 32 = 2048

8192 workgroups / 64 CU = 128 waves per CU
Each wave = 64 threads
Wave execution time: ~10 microseconds

Total dequant time: ~1.3 ms per weight
7 weights per layer: ~9 ms
36 layers: ~324 ms

GPU load: ~80% (acceptable)
```

## ISSUE 4: VRAM Usage (OK)

### Weights (on GPU)
```
Q4_0 model: 3B params * 0.5 bytes = 1.5GB
Activations: ~100KB (ring buffer)
KV cache: ~50MB
Total: ~1.6GB
VRAM: 16GB
Usage: 10% (OK)
```

## VERDICT

| Issue | Status | Fix Needed |
|-------|--------|------------|
| Race condition | OK | No |
| Barrier placement | OK | No |
| GPU load | OK | No |
| VRAM usage | OK | No |
| Ring buffer overflow | CRITICAL | Yes |

## REQUIRED FIX

Add checkpoint/restore to RingAllocator:
```cpp
size_t RingAllocator::checkpoint() {
    return offset;
}

void RingAllocator::restore(size_t cp) {
    offset = cp;
}
```

Then in forwardPartial:
```cpp
for (uint32_t layer = 0; layer < layersToRun; ++layer) {
    size_t cp = allocator->checkpoint();
    
    // ... dequant + GEMM ...
    
    allocator->restore(cp);
}
```
