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
    static_assert(sizeof(rdna4::TQ4Block128) == 66, "TQ4Block128 must be 66 bytes");
    static_assert(sizeof(rdna4::TQ3Block128) == 50, "TQ3Block128 must be 50 bytes");
    static_assert(sizeof(rdna4::TQ6Block64) == 50, "TQ6Block64 must be 50 bytes");

    // Test packed size helpers
    assert(rdna4::tq4_packed_size(128) == 66);    // exactly 1 block
    assert(rdna4::tq4_packed_size(129) == 132);   // 2 blocks
    assert(rdna4::tq3_packed_size(128) == 50);    // exactly 1 block
    assert(rdna4::tq3_packed_size(256) == 100);   // 2 blocks
    assert(rdna4::tq6_packed_size(64) == 50);     // exactly 1 block
    assert(rdna4::tq6_packed_size(65) == 100);    // 2 blocks

    // Test push constant sizes (match shader layouts in dequant_turbo.comp / gemm_turbo.comp)
    static_assert(sizeof(rdna4::DequantTurboPushConstants) <= 128, "DequantTurboPushConstants too large");
    static_assert(sizeof(rdna4::GemmTurboPushConstants) <= 128, "GemmTurboPushConstants too large");
    assert(sizeof(rdna4::DequantTurboPushConstants) == 40); // scalar layout
    assert(sizeof(rdna4::GemmTurboPushConstants) == 56);    // scalar layout

    // Test TQ4 block packing (manual)
    rdna4::TQ4Block128 block;
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
