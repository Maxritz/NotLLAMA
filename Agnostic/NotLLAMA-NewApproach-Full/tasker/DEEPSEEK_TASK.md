# DeepSeek (Code Implementation) — Option D (DONE)

## Completed: Round 4 — Batched forwardPartial

### What was done
- `initDequantBuffer()`: resized from MLP-set to full-layer set (9 weights)
- `forwardPartial()`: two-phase per layer — dequant all weights async (1 sync), then single batch submit with pipeline barriers for the full layer
- Sync count: 4 → 2 per layer, ~72 total for 36 layers

### Files Modified
- `src/host/inference_engine.cpp`

### Next Available Tasks
- Debug GPU output token = 0 (logit readback vs sampling)
- Increase dequant chunk size (1M WG cap reduces per-dispatch throughput)
- Task 1: Proper Top-K/Top-P GPU sampling
- Task 2: Fix exit crash (0xC0000005)
