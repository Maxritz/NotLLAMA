# TRACE: Ring Allocator Sizing

## Session: 2026-06-29 — Round 2 fixes

## Bug
Ring allocator was 64 MB, insufficient for per-layer dequant staging.

## Input State
| Variable | Value | Source |
|----------|-------|--------|
| ringSize (before) | 64 MB (67,108,864) | `test_inference.cpp:109` |
| ringSize (after) | 1 GB (1,073,741,824) | fix |
| memory type | HOST_VISIBLE + HOST_COHERENT | `allocator.cpp` |
| GPU | RX 9070 XT, 16GB VRAM, resizable BAR | vulkaninfo |

## Per-Layer Staging Requirements (worst layer, Q4_K)

All weights dequantized from quantized → F32 simultaneously in ring buffer.

| Weight | Quantized Bytes | nElements | F32 Staging | Format |
|--------|----------------|-----------|-------------|--------|
| attn_q | 2,359,296 | 4,194,304 | 16 MB | Q4_K |
| attn_k | 589,824 | 1,048,576 | 4 MB | Q4_K |
| attn_v | 860,160 | 1,529,173 | 6 MB | Q4_K |
| attn_o | 2,359,296 | 4,194,304 | 16 MB | Q4_K |
| ffn_gate | 9,437,184 | 16,777,216 | 64 MB | Q4_K |
| ffn_up | 9,437,184 | 16,777,216 | 64 MB | Q4_K |
| ffn_down | 13,762,560 | 24,460,373 | 93 MB | Q4_K |
| **Subtotal** | | | **263 MB** | |
| Fixed allocs | hidden+Q+K+V+attnOut+mlpOut+logits+scratch | | **~1.4 MB** | |
| **Layer total** | | | **~265 MB** | |

## LM Head Staging
| Weight | Bytes | nElements | F32 Staging |
|--------|-------|-----------|-------------|
| token_embd (Q6_K) | 215,470,080 | 262,668,288 | **1,002 MB** |

## Decision Tree
```
forwardPartial()
│
├── allocator->reset() → full ring available
│
├── fixed allocs (1.4 MB)
│   └── 64MB ring: 62.6MB remaining → OK
│   └── 1GB ring:  1022.6MB remaining → OK
│
├── Layer N:
│   ├── 64MB ring: need 265MB > 62.6MB → OOM at ffn_gate (64MB)
│   ├── 256MB ring: need 265MB > 254.6MB → OOM at ffn_down (93MB)
│   └── 1GB ring: need 265MB < 1022MB → OK
│
├── LM head dequant:
│   ├── 1GB ring: need 1002MB < 1022MB → OK
│   └── 512MB ring: need 1002MB > 510MB → OOM
│
└── checkpoint/restore at layer end → ring freed
```

## Truth Table
| Ring Size | Fixed | 1 Layer Dequant | LM Head | Fits Worst? |
|-----------|-------|-----------------|---------|-------------|
| 64 MB | 1.4 MB | 265 MB | 1 GB | FAIL (OOM at ffn_gate) |
| 256 MB | 1.4 MB | 265 MB | 1 GB | FAIL (OOM at ffn_down) |
| 512 MB | 1.4 MB | 265 MB | 1 GB | FAIL (OOM at LM head) |
| **1 GB** | 1.4 MB | 265 MB | 1 GB | **PASS** |

## Fix
Increased from 64 MB → 1 GB in `test_inference.cpp:109`.

```cpp
size_t ringSize = 1024 * 1024 * 1024; // 1 GB
```

## VRAM Budget (RX 9070 XT, 16 GB)
| Usage | Size |
|-------|------|
| Model weights (DEVICE_LOCAL) | 762 MB |
| Ring buffer (HOST_VISIBLE + DEVICE_LOCAL) | 1,024 MB |
| KV cache (DEVICE_LOCAL) | 43 MB |
| Pipeline/descriptor overhead | ~50 MB |
| **Total** | **~1.88 GB** |
| **Available** | **16 GB** |
| **Headroom** | **88%** |

## Future Optimization
- Pre-dequant token_embd.weight → persistent F32 buffer at init time
- Removes 1 GB spike from ring
- Reduces ring requirement back to ~265 MB
- Reduces per-token dequant work by 262M elements
