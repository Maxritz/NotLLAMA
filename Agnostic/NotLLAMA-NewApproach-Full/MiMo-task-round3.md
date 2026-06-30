# MiMo-task-round3.md — TurboQuant Round 3: Benchmarking, Conversion, and Docs

## Why
Round 1: Shaders + schema + enum. Round 2: Host types + Python validators. Round 3 bridges everything to real models:
- Benchmark on actual model weights (not synthetic) to prove TurboQuant is competitive with GGUF
- Standalone converter so users can convert any GGUF → TurboQuant without touching the engine
- C++ test stub so future host-side code has a compile-time safety net
- Integration doc so Claude (Kimi, Laguna M.1) can wire TurboQuant into the engine without re-reading all MiMo files

## Deadline
Until Claude/Kimi finishes Round 2 + Kimi's architecture scaffolding

## Task 1: tools/benchmark_turboquant.py
Create a benchmarking script that:
1. Loads GGUF model weights from disk (binary format: num_tensors × [name_len, name, dtype, ndim, shape, data])
2. For each weight tensor:
   a. Record original size (bytes)
   b. Quantize to TQ4_128, TQ3_128, TQ6_64 using convert_to_tq4/tq3/tq6 from weight_converter
   c. Dequantize back to float32
   d. Compute MSE, max absolute error, cosine similarity, L1 norm
3. Prints two tables:

**Per-layer table:**
```
Layer              | Format  | Orig KB  | TQ KB  | Ratio | MSE       | CosSim
token_embd.weight  | TQ4_128 | 768.0    | 192.0  | 0.25x | 0.00123   | 0.9987
blk.0.attn_q.weight| TQ6_64 | 512.0    | 128.0  | 0.25x | 0.00045   | 0.9998
...
```

**Summary table:**
```
Format    | Total MB | Compressed MB | Ratio | Avg MSE  | Avg CosSim | Max MSE
TQ4_128   | 3610.0   | 902.5         | 0.25x | 0.00123  | 0.9987     | 0.00456
TQ3_128   | 3610.0   | 686.0         | 0.19x | 0.00345  | 0.9945     | 0.01234
TQ6_64    | 3610.0   | 686.0         | 0.19x | 0.00045  | 0.9998     | 0.00123
Q8_0      | 3610.0   | 3610.0        | 1.00x | 0.00000  | 1.0000     | 0.00000
```

4. Also prints timing: quantize_time, dequantize_time, throughput (MB/s)

Requirements:
- Uses weight_converter functions for quantization/dequantization
- `--model <dir>` flag: reads Q8_0 .bin + .json from directory
- `--compare-gguf` flag: also loads original GGUF (via gguf python library if available) for reference
- Prints memory savings vs Q8_0 baseline
- `--output <file>` flag: writes results to JSON for Claude to analyze

File: `tools/benchmark_turboquant.py`

## Task 2: tools/convert_gguf_to_turboquant.py
Create a standalone GGUF → TurboQuant converter:
1. Reads a GGUF model (binary format from weight_converter.py)
2. Converts ALL weight tensors to a specified TurboQuant format
3. Outputs:
   - `<model_name>_tq4_128.weights.bin` — packed TurboQuant binary
   - `<model_name>_tq4_128.json` — model metadata (same as GGUF JSON but with format="TQ4_128")
4. Also supports TQ3_128 and TQ6_64

CLI:
```bash
python convert_gguf_to_turboquant.py \
    --input model_q8_0.bin \
    --format tq4_128 \
    --output-dir ./converted/
```

Output JSON format (for engine consumption):
```json
{
    "model": {
        "name": "model_tq4_128",
        "architecture": "qwen2",
        "format": "TQ4_128",
        "num_layers": 36,
        "hidden_dim": 2048,
        "num_heads": 32,
        "head_dim": 64,
        "vocab_size": 151936,
        "max_seq_len": 4096
    },
    "tensor_offsets": {
        "token_embd.weight": {"offset": 0, "size": 12582912, "dtype": "TQ4_128"},
        "blk.0.attn_q.weight": {"offset": 12582912, "size": 33554432, "dtype": "TQ4_128"},
        ...
    },
    "block_layout": {
        "format": "TQ4_128",
        "block_bytes": 66,
        "elements_per_block": 128,
        "scale_bytes": 2,
        "data_bytes": 64
    }
}
```

Requirements:
- Uses weight_converter functions for actual quantization
- Validates all converted tensors: check packed size matches expected
- Prints per-tensor progress and final summary
- Handles all QuantFormat types (Q6_K, Q8_0, F16, etc.) — but only TQ4/TQ3/TQ6 are new
- Fallback: if tensor dtype is already TQ format, skip conversion

File: `tools/convert_gguf_to_turboquant.py`

## Task 3: test/test_turboquant.cpp
Create a C++ compile-time test stub:
```cpp
// MiMo Round 3 — TurboQuant host-side compile test
// Compiles as: cl /EHsc /I ../include test_turboquant.cpp

#include "rdna4_turboquant.hpp"
#include "rdna4_types.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    printf("TurboQuant compile-time tests...\n");

    // Test block sizes
    static_assert(sizeof(TQ4Block128) == 66, "TQ4Block128 must be 66 bytes");
    static_assert(sizeof(TQ3Block128) == 50, "TQ3Block128 must be 50 bytes");
    static_assert(sizeof(TQ6Block64) == 50, "TQ6Block64 must be 50 bytes");

    // Test packed size helpers
    assert(tq4_packed_size(128) == 66);    // exactly 1 block
    assert(tq4_packed_size(129) == 132);   // 2 blocks
    assert(tq3_packed_size(128) == 50);    // exactly 1 block
    assert(tq3_packed_size(256) == 100);   // 2 blocks
    assert(tq6_packed_size(64) == 50);     // exactly 1 block
    assert(tq6_packed_size(65) == 100);    // 2 blocks

    // Test push constant sizes
    static_assert(sizeof(DequantTurboPushConstants) <= 128, "DequantTurboPushConstants too large");
    static_assert(sizeof(GemmTurboPushConstants) <= 128, "GemmTurboPushConstants too large");
    assert(sizeof(DequantTurboPushConstants) == 48);
    assert(sizeof(GemmTurboPushConstants) == 48);

    // Test TQ4 block packing (manual)
    TQ4Block128 block;
    memset(&block, 0, sizeof(block));
    block.scale = 0x3C00; // 1.0 in float16
    // Pack element 0 = 5: data[0] |= 5 (low nibble)
    // Pack element 1 = -3: data[0] |= (0x0D << 4) (high nibble, -3 & 0xF = 0xD)
    block.data[0] = 0xD5;
    assert((block.data[0] & 0x0F) == 5);      // element 0
    assert(((block.data[0] >> 4) & 0x0F) == 0xD); // element 1 (0xD = -3 mod 16)

    printf("All compile-time tests PASSED\n");
    return 0;
}
```

Requirements:
- This is a STUB — just compile-time checks, no runtime GPU tests
- Must compile with both MSVC and GCC/Clang
- Include rdna4_turboquant.hpp and rdna4_types.hpp

File: `test/test_turboquant.cpp`

## Task 4: docs/turboquant_integration.md
Create an integration guide for Claude (Kimi, Laguna M.1) to wire TurboQuant into the engine:

Structure:
```
# TurboQuant Integration Guide

## Overview
Brief explanation of TurboQuant and why it matters.

## Block Layouts (for weight_converter.py)
Byte diagrams for TQ4_128, TQ3_128, TQ6_64:
- TQ4_128: [scale:2][data:64] = 66 bytes/block, 128 elements
- TQ3_128: [scale:2][data:48] = 50 bytes/block, 128 elements  
- TQ6_64:  [scale:2][data:48] = 50 bytes/block, 64 elements

Include bit-packing diagrams:
```
TQ4 nibble layout:
byte[0] = [elem0:4bits][elem1:4bits]
byte[1] = [elem2:4bits][elem3:4bits]
...

TQ6 packing (8 elements → 6 bytes):
byte[0] = [elem0:6][elem1低2位:2]
byte[1] = [elem1高4位:4][elem2:6低2位:2]  -- NO, too complex

Simpler: Use sequential bit packing
TQ4: bit[0:4]=elem0, bit[4:8]=elem1, ...
TQ3: bit[0:3]=elem0, bit[3:6]=elem1, ...
TQ6: bit[0:6]=elem0, bit[6:12]=elem1, ...
```

## Engine Wiring Checklist

### Step 1: Weight Loading (inference_engine.cpp)
- [ ] Add case for TQ4_128, TQ3_128, TQ6_64 in weight loading
- [ ] Use tq4_packed_size/tq3_packed_size/tq6_64_packed_size for buffer allocation
- [ ] Store block_size and format in weight metadata

### Step 2: Dequant Shader (dequant_turbo.comp)
- [ ] Already implemented by MiMo
- [ ] Wire DequantTurboPushConstants in scheduler
- [ ] Test with embed weights first

### Step 3: Fused GEMM (gemm_turbo.comp)
- [ ] Already implemented by MiMo
- [ ] Wire GemmTurboPushConstants in scheduler
- [ ] Test MLP gate_up first

### Step 4: Scheduler Integration
- [ ] Add dequant_turbo and gemm_turbo to PipelineBuilder
- [ ] Push constants: sizeof must be <= 128 bytes (verified by static_assert)

### Step 5: Fallback
- [ ] If tensor format is not TQ4/TQ3/TQ6, use existing dequantize.comp path
- [ ] If tensor format is TQ, use dequant_turbo.comp or gemm_turbo.comp

## Performance Targets
- TQ4_128: <0.1% MSE vs Q8_0, 4x compression
- TQ3_128: <0.5% MSE vs Q8_0, 5x compression
- TQ6_64: <0.01% MSE vs Q8_0, 5x compression

## Files Modified by MiMo
- include/rdna4.hpp (enum)
- include/rdna4_turboquant.hpp (new)
- include/rdna4_types.hpp (append)
- src/kernels/dequant_turbo.comp (new)
- src/kernels/gemm_turbo.comp (new)
- src/kernels/AGENTS.md (new)
- docs/turboquant_schema.json (new)
- docs/turboquant_formats.md (new)
```

File: `docs/turboquant_integration.md`

## Acceptance Criteria
1. `python tools/benchmark_turboquant.py --model model/` produces summary table
2. `python tools/convert_gguf_to_turboquant.py --input model/q8_0.bin --format tq4_128` produces .bin + .json
3. `test/test_turboquant.cpp` compiles with MSVC: `cl /EHsc /I include test/test_turboquant.cpp`
4. `docs/turboquant_integration.md` covers all 5 wiring steps
5. No existing files modified except new files created
6. Each deliverable has a one-line comment at the top: `// MiMo Round 3 — <purpose>`

## Verification
```powershell
# Benchmark works
python tools/benchmark_turboquant.py --model model/

# Converter works
python tools/convert_gguf_to_turboquant.py --input model/q8_0.bin --format tq4_128 --output-dir ./converted/

# C++ test compiles
cl /EHsc /I include test/test_turboquant.cpp /Fe:test_turboquant.exe
```
