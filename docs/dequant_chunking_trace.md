# TRACE: Dequant Dispatch Workgroup Limit

## Session: 2026-06-29 — Round 2 fixes

## Bug
`dequantIfNeeded()` dispatched a single `vkCmdDispatch` for the entire weight dequant. Weights larger than 4,194,240 elements (65,535 workgroups × 64 threads) exceeded the Vulkan `maxComputeWorkGroupCount[0]` = 65,535 limit.

## Input State
| Variable | Value | Source |
|----------|-------|--------|
| local_size_x | 64 | `dequantize.comp:9` |
| maxWorkgroupCount | 65535 | VkPhysicalDeviceLimits |
| Q4_K block bytes | 144 | ggml spec |
| Q4_K elements/block | 256 | ggml spec |
| Q6_K block bytes | 210 | ggml spec |
| Q6_K elements/block | 256 | ggml spec |

## Affected Weights (llamamobile, Q4_K/Q6_K mixed)

| Weight | Bytes | Format | nElements | Workgroups (64) | Limit | PASS? |
|--------|-------|--------|-----------|-----------------|-------|-------|
| attn_q | 2,359,296 | Q4_K | 4,194,304 | 65,536 | 65,535 | FAIL |
| attn_output | 2,359,296 | Q4_K | 4,194,304 | 65,536 | 65,535 | FAIL |
| ffn_gate | 9,437,184 | Q4_K | 16,777,216 | 262,144 | 65,535 | FAIL |
| ffn_up | 9,437,184 | Q4_K | 16,777,216 | 262,144 | 65,535 | FAIL |
| ffn_down | 13,762,560 | Q4_K | 24,460,373 | 382,194 | 65,535 | FAIL |
| token_embd | 215,470,080 | Q6_K | 262,668,288 | 4,104,192 | 65,535 | FAIL |

## Root Cause
`dequantIfNeeded()` in `inference_engine.cpp` had no chunking loop. It dispatched `(nElements + 63) / 64` workgroups in one call. The Vulkan spec limits this to 65,535.

The driver does NOT reject the dispatch — it silently wraps or executes a truncated number of workgroups, producing garbage dequant output.

## Decision Tree
```
dequantIfNeeded(nElements)
│
├── wgCount = (nElements + 63) / 64
│   ├── wgCount > 65535?
│   │   ├── BEFORE FIX: dispatch anyway → GPU produces garbage
│   │   └── AFTER FIX: chunk into (nElements / DQ_CHUNK_ELEMS) dispatches
│   └── wgCount ≤ 65535? → single dispatch
```

## Truth Table (After Fix)
| Condition | nElements | Chunk | Workgroups | Limit | PASS? |
|-----------|-----------|-------|-----------|-------|-------|
| attn_q (4,194,304) | ≤ DQ_CHUNK_ELEMS=4,194,240? NO | 2 chunks of 2,097,152 | 32,768 | 65,535 | PASS |
| token_embd (262,668,288) | need 63 chunks | 4,168,544/chunk | 65,133 | 65,535 | PASS |

## Fix
```cpp
static const uint32_t DQ_CHUNK_ELEMS = 65535u * 64u; // 4,194,240

uint32_t remaining = nElements;
uint32_t offset = 0;
while (remaining > 0) {
    uint32_t chunk = min(remaining, DQ_CHUNK_ELEMS);
    uint32_t wgCount = (chunk + 63u) / 64u;
    dqPC.totalThreads = chunk;
    dqPC.elementOffset = offset;
    scheduler->dispatchInBatch(..., wgCount, 1, 1);
    offset += chunk;
    remaining -= chunk;
}
```

## Debug Points
- ASSERT before each dispatch: `wgCount <= 65535`
- LOG: `"[dequant] chunked: %u elements, %u workgroups, offset=%u"` per chunk
- ASSERT after loop: `offset == nElements`

## VERDICT: FIXED
- Infinite loop risk: NO (remaining strictly decreases by at least 64 per iteration)
- Barrier placement: ONE barrier after ALL chunk dispatches (all write to same staging buffer, last write is safe before barrier)
