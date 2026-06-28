# KV Cache Compression Methods

## Overview

KV cache compression reduces the memory footprint of the key-value cache by trading precision for capacity. This enables longer context windows without increasing VRAM usage. Compression can be applied per-layer, per-head, or hierarchically based on token recency.

## Methods

### Quantize
Per-head block quantization of K and V cache values.

- **How it works**: Each block of tokens is quantized using a per-block scale factor. F16 values are converted to Q4_0, Q5_0, or Q8_0 format.
- **Compression ratio**: 2-4× depending on format (Q4_0 = 4×, Q5_0 = 3.2×, Q8_0 = 2×).
- **Accuracy impact**: Small. Q8_0 is nearly lossless. Q4_0 introduces minor noise but does not degrade generation quality noticeably for most models.
- **Implementation**: `kv_cache_quantize.comp` shader handles F16→quantized conversion.
- **Config**: `format` (Q4_0 default), `compress_every_n_layers` (1 = all layers), `per_head_scales` (true).

### Sparsify
Zero out small values in the KV cache.

- **Status**: ⚠️ **Future**. Only useful when paired with sparse attention kernels.
- **How it works**: Threshold-based pruning. Values below `threshold * max(abs(value))` are set to zero. Compressed sparse row (CSR) format not yet implemented in attention kernels.
- **Compression ratio**: Variable, depends on sparsity threshold. Aggressive thresholds (0.1) can achieve 5-10× but degrade quality.
- **Accuracy impact**: Medium to high depending on threshold. Not recommended without sparse attention.

### Low-Rank
Factorize K and V cache into smaller low-rank matrices.

- **Status**: 🔬 **Research**. Complex to update incrementally as new tokens arrive.
- **How it works**: Maintain a rank-r approximation of the KV cache. When new tokens arrive, update the factorization via incremental SVD or similar.
- **Compression ratio**: r × (nHeads × headDim + seqLen) / (seqLen × nHeads × headDim). E.g., rank=64 for seqLen=4096, headDim=128, nHeads=8 → ~1.5× compression.
- **Accuracy impact**: High quality at sufficient rank. Low-rank assumption may not hold for all attention patterns.
- **Not recommended for initial implementation** due to update complexity.

### Hierarchical
Recent tokens kept in high precision, older tokens quantized to lower precision.

- **How it works**: The most recent `recent_n` tokens remain in F16 for fast attention access. Older tokens are quantized (default Q4_0). When a token becomes older than `recent_n`, it is quantized in the background.
- **Compression ratio**: ~2-4× overall, depending on ratio of recent to old tokens. For 4096 context with recent_n=1024 and Q4_0: (1024×F16 + 3072×Q4_0) / 4096×F16 ≈ 2.5×.
- **Accuracy impact**: Minimal. The model primarily attends to recent tokens. Older tokens still contribute semantic context even at lower precision.
- **Recommended default** once implemented: Best tradeoff between memory savings and accuracy.

## Integration

KV cache compression is applied after the KV write for each new token, or in a background compaction pass. The `kv_cache_quantize.comp` shader handles F16→quantized conversion for the "older" tier. The most recent `recent_n` tokens are never quantized — this is enforced by the host-side logic.

### Lifecycle
1. New token's K and V are written in F16 format.
2. If token position > `recent_n`, the token at position `pos - recent_n` is eligible for quantization.
3. The host dispatches `kv_cache_quantize.comp` on a range of tokens in the "older" tier.
4. Attention shaders must handle mixed-precision KV cache (F16 recent + quantized old).
