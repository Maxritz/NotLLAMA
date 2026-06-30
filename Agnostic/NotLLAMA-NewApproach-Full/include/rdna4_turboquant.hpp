// MiMo Round 2 — host-side TurboQuant block layouts
// For shader layouts, see dequant_turbo.comp / gemm_turbo.comp. This header is for host-side packing/unpacking only.

#pragma once
#include <cstdint>

namespace rdna4 {

// TurboQuant 4-bit, 128-element blocks.
// Layout: [scale: uint16_t][data: 64 bytes] = 66 bytes.
// Packing: 2 weights per byte, low nibble = even index.
struct TQ4Block128 {
    uint16_t scale;
    uint8_t  data[64];
};

// TurboQuant 3-bit, 128-element blocks.
// Layout: [scale: uint16_t][data: 48 bytes] = 50 bytes.
// Packing: sequential 3-bit stream, 8 weights per 3 bytes (24 bits, no waste).
struct TQ3Block128 {
    uint16_t scale;
    uint8_t  data[48];
};

// TurboQuant 6-bit, 64-element blocks.
// Layout: [scale: uint16_t][data: 48 bytes] = 50 bytes.
// Packing: sequential 6-bit stream, 64 weights = 384 bits = 48 bytes.
struct TQ6Block64 {
    uint16_t scale;
    uint8_t  data[48];
};

// Common header for a TurboQuant tensor slice (optional, host metadata).
struct TQBlockHeader {
    uint16_t scale;
    uint16_t reserved;
    uint32_t num_elements;
};

static_assert(sizeof(TQ4Block128) == 66, "TQ4Block128 must be 66 bytes");
static_assert(sizeof(TQ3Block128) == 50, "TQ3Block128 must be 50 bytes");
static_assert(sizeof(TQ6Block64)  == 50, "TQ6Block64 must be 50 bytes");
static_assert(sizeof(TQBlockHeader) == 8, "TQBlockHeader must be 8 bytes");

// Number of TQ4_128 blocks required for n elements (ceil(n / 128)).
inline int tq4_num_blocks(int n) {
    return (n + 127) / 128;
}

// Packed byte size for n TQ4_128 elements.
inline int tq4_packed_size(int n) {
    return tq4_num_blocks(n) * 66;
}

// Packed byte size for n TQ3_128 elements.
inline int tq3_packed_size(int n) {
    return ((n + 127) / 128) * 50;
}

// Packed byte size for n TQ6_64 elements.
inline int tq6_packed_size(int n) {
    return ((n + 63) / 64) * 50;
}

} // namespace rdna4
