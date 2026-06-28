# DeepSeek (Code Implementation) — Option D

## Current Task: Implement Option D (Batched forwardPartial)

### Goal
Refactor `forwardPartial()` to batch dequant dispatches per layer. Reduce sync count from ~12/layer to ~8/layer.

### Strategy
```
// Per layer: batch dequant for all weights (independent, no data deps)
dequantWeight(attn_norm); dequantWeight(q); dequantWeight(k); dequantWeight(v); dequantWeight(attn_output);
syncAll(); // 1 sync instead of 5

// Then run compute path
rms_norm → gemm_q → gemm_k → gemm_v → sync → rope → kv_write → attention → gemm_out → add → sync
ffn_norm → dequant ffn_up/gate → sync → mlp → dequant ffn_down → sync → add → sync
```

### Target
- ~8 syncs per layer × 36 layers = ~288 syncs (down from ~432)
- 33% reduction in sync overhead

### Files to Modify
- `src/host/inference_engine.cpp` — refactor `forwardPartial()`

### Constraints
- Weights stay quantized on GPU
- No monolithic pre-dequantized buffers
- Each layer is a self-contained work unit
- Dequant staging buffer capped at 512 MB

### After Completion
- Build and test
- Verify token generation works
- Report sync count improvement
