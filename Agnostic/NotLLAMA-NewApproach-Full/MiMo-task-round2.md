# MiMo-task-round2.md — TurboQuant Round 2: Host Integration

## Why
Round 1 delivered shaders + JSON schema + formats doc + enum. But nothing on the host side knows how to actually pack TurboQuant blocks or how the shaders receive their parameters. We need:
1. Host-side block layout structs (so weight converter can emit valid bins)
2. Push constant structs with compile-time size guards (so shaders can read them)
3. Actual Python conversion functions that produce TurboQuant binaries
4. A validation script that proves quantized accuracy is acceptable

## Deadline
Until Kimi finishes architecture scaffolding (est. 1 hour)

## Task 1: include/rdna4_turboquant.hpp
Create a new header with host-side TurboQuant block layouts.

Requirements:
- `TQ4Block128` — 66 bytes: `uint16_t scale; uint8_t data[64];` — packed nibbles, 128 elements
- `TQ3Block128` — 50 bytes: `uint16_t scale; uint8_t data[48];` — packed 3-bit values, 128 elements
- `TQ6Block64` — 50 bytes: `uint16_t scale; uint8_t data[48];` — packed 6-bit values, 64 elements
- `TQBlockHeader` — common header: `uint16_t scale; uint16_t reserved; uint32_t num_elements;`
- `static_assert(sizeof(TQ4Block128) == 66, "...")` and same for others
- Helper functions: `tq4_packed_size(int n)`, `tq3_packed_size(int n)`, `tq6_packed_size(int n)` — returns byte count for n elements
- `tq4_num_blocks(int n)` — ceil(n / 128)
- Comment: "For shader layouts, see dequant_turbo.comp / gemm_turbo.comp. This header is for host-side packing/unpacking only."

File: `include/rdna4_turboquant.hpp`

## Task 2: include/rdna4_types.hpp append
Append to the end of the existing file (before the last #endif guard):

```cpp
// ── TurboQuant push constants ─────────────────────────────────────
// Matches layout in dequant_turbo.comp / gemm_turbo.comp.
// static_assert guarantees <= 128 bytes per Vulkan spec.
struct DequantTurboPushConstants {
    uint32_t  inputOffset;       // byte offset into input buffer
    uint32_t  outputOffset;      // float16 offset into output buffer
    uint32_t  elementOffset;     // first global element index
    uint32_t  elementCount;      // total elements to dequantize
    uint32_t  format;            // QuantFormat enum value
    uint32_t  groupSize;         // elements per scale (1, 64, or 128)
    uint32_t  reserved[6];       // future use
}; // 12 × 4 = 48 bytes — well within 128
static_assert(sizeof(DequantTurboPushConstants) <= 128,
    "DequantTurboPushConstants must fit in 128 bytes");

struct GemmTurboPushConstants {
    uint32_t M;                  // rows (always 1 for LLM)
    uint32_t N;                  // cols (output features)
    uint32_t K;                  // input features (hidden dim)
    uint32_t inputOffset;        // byte offset into activation buffer
    uint32_t weightOffset;       // byte offset into quantized weight buffer
    uint32_t outputOffset;       // float16 offset into output buffer
    uint32_t alpha;              // float16 scale (packed as uint)
    uint32_t format;             // QuantFormat enum value
    uint32_t KPerBlock;          // elements per quant block
    uint32_t reserved[3];        // future use
}; // 12 × 4 = 48 bytes
static_assert(sizeof(GemmTurboPushConstants) <= 128,
    "GemmTurboPushConstants must fit in 128 bytes");
```

Important: Check the current content of rdna4_types.hpp first. If the last #endif is preceded by the DequantPushConstants block, insert right before #endif.

File: `include/rdna4_types.hpp` — append only

## Task 3: tools/weight_converter.py extension
Add three conversion functions to the existing weight_converter.py. Do NOT rewrite the file — append these functions at the end (after the `if __name__ == "__main__":` block).

### convert_to_tq4(elements, scale)
- Input: numpy float32 array, uint16 float16 scale
- Output: bytes (66 bytes per block)
- Block size: 128 elements
- Process: divide each element by scale, clamp [-8,7], cast to int8, pack two int4 per byte (little-endian: low nibble first)
- Pad last block with zeros if n % 128 != 0
- Returns: raw bytes

### convert_to_tq3(elements, scale)
- Input: numpy float32 array, uint16 float16 scale
- Output: bytes (50 bytes per block)
- Block size: 128 elements
- Process: divide by scale, clamp [-4,3], cast to int8, pack four int3 per byte
- 3-bit packing: byte[i] = v0 | (v1 << 3) | (v2 << 6) | (v3 << 1) -- wait, that's not clean 3-bit boundaries
- Correct 3-bit packing: Use 42 values per 128-byte group (128 × 3 = 384 bits = 48 bytes exactly). Pack sequentially: bits [0:2] = val0, [3:5] = val1, etc. Every 8 elements = 3 bytes (8 × 3 = 24 bits = 3 bytes).
- Returns: raw bytes

### convert_to_tq6(elements, scale)
- Input: numpy float32 array, uint16 float16 scale
- Output: bytes (50 bytes per block)
- Block size: 64 elements
- Process: divide by scale, clamp [-32,31], cast to int8, pack two int6 per byte
- 6-bit packing: byte[i] = v0 | (v1 << 6) -- wait, that's not clean either
- Correct: 64 × 6 = 384 bits = 48 bytes. Pack: byte[i] = v0 | (v1 << 6) for even pairs. Actually: every 4 elements = 3 bytes (4 × 6 = 24 bits). Or simpler: 8 elements per 6 bytes (8 × 6 = 48 bits = 6 bytes).
- Returns: raw bytes

Also add a CLI entry:
```python
if __name__ == "__main__":
    # ... existing code ...
    # TurboQuant conversion test
    # python weight_converter.py --tq-test <input.bin> <format: tq4|tq3|tq6>
```

File: `tools/weight_converter.py` — append functions

## Task 4: tools/validate_turboquant.py
Create a standalone validation script that:
1. Loads a GGUF model (any format)
2. Extracts weight tensors
3. For each TurboQuant format (TQ4_128, TQ3_128, TQ6_64):
   a. Quantizes each weight tensor using the conversion functions
   b. Dequantizes back to float32
   c. Computes MSE, max absolute error, cosine similarity vs original
4. Prints a comparison table:
   ```
   Format    | MSE        | MaxErr     | CosSim  | Size   | Ratio
   TQ4_128   | 0.00123    | 0.0456     | 0.9987  | 66 B   | 0.25x
   TQ3_128   | 0.00345    | 0.0891     | 0.9945  | 50 B   | 0.19x
   TQ6_64    | 0.00045    | 0.0234     | 0.9998  | 50 B   | 0.19x
   ```
5. Tests on synthetic distributions: random uniform, random normal, structured (sin wave), sparse (mostly zeros)
6. Exits 0 if all formats within acceptable thresholds (MSE < 0.01)

Requirements:
- Uses numpy, struct (no gguf dependency — just reads raw binary)
- Import weight_converter functions: `from weight_converter import convert_to_tq4, convert_to_tq3, convert_to_tq6`
- Must be runnable standalone: `python validate_turboquant.py`
- Add a `--model <path>` flag to test on real model weights

File: `tools/validate_turboquant.py`

## Acceptance Criteria
1. `glslc --target-env=vulkan1.3 src/kernels/dequant_turbo.comp` still compiles
2. `glslc --target-env=vulkan1.3 src/kernels/gemm_turbo.comp` still compiles
3. `python tools/validate_turboquant.py` exits 0 with table output
4. `sizeof(DequantTurboPushConstants) == 48` (verifiable via static_assert)
5. No existing files modified except rdna4_types.hpp (append only) and weight_converter.py (append only)
6. Each deliverable has a one-line comment at the top: `// MiMo Round 2 — <purpose>`

## Verification
```powershell
# Shaders still compile
glslc --target-env=vulkan1.3 src/kernels/dequant_turbo.comp
glslc --target-env=vulkan1.3 src/kernels/gemm_turbo.comp

# Validation script works
python tools/validate_turboquant.py

# Types header compiles (quick smoke test with a test .cpp)
# Include rdna4_turboquant.hpp and rdna4_types.hpp in a test file
```
