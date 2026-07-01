// DeepSeek Round 3 — Compression host-side compile test
// Compiles as: cl /EHsc /I ../include test_compression.cpp

#include "rdna4_compression.hpp"
#include "rdna4_compression_scheduler.hpp"
#include <cassert>
#include <cstdio>

int main() {
    printf("Compression compile-time tests...\n");

    // Push constant size checks
    static_assert(sizeof(rdna4::CompressContextPushConstants) <= 128,
        "CompressContextPushConstants must fit in 128 bytes");
    static_assert(sizeof(rdna4::KVCacheQuantizePushConstants) <= 128,
        "KVCacheQuantizePushConstants must fit in 128 bytes");
    static_assert(sizeof(rdna4::KVCacheDequantPushConstants) <= 128,
        "KVCacheDequantPushConstants must fit in 128 bytes");

    // Config default construction
    rdna4::ContextCompressionConfig ctxCfg;
    assert(ctxCfg.enabled == false);
    assert(ctxCfg.blockSize == 128);
    assert(ctxCfg.bits == 4);
    assert(ctxCfg.threshold > 0.8f && ctxCfg.threshold < 0.9f);

    rdna4::KVCompressionConfig kvCfg;
    assert(kvCfg.enabled == false);
    assert(kvCfg.blockSize == 64);
    assert(kvCfg.kBits == 4);
    assert(kvCfg.minSeqLen == 256);

    // Scheduler default construction
    rdna4::CompressionScheduler scheduler(ctxCfg, kvCfg);
    // step() with seqLen below threshold should return no-op decision
    auto decision = scheduler.step(100, 4096);
    assert(decision.compressContext == false);
    assert(decision.compressKV == false);

    // Scheduler with compression enabled
    rdna4::ContextCompressionConfig ctxCfg2;
    ctxCfg2.enabled = true;
    ctxCfg2.threshold = 0.5f; // trigger at >50%
    rdna4::KVCompressionConfig kvCfg2;
    kvCfg2.enabled = true;
    kvCfg2.minSeqLen = 50;
    rdna4::CompressionScheduler scheduler2(ctxCfg2, kvCfg2);
    auto decision2 = scheduler2.step(100, 128);
    assert(decision2.compressContext == true);
    assert(decision2.compressKV == true);
    assert(decision2.contextTargetLen >= 64);
    assert(decision2.kvQuantizeBits == 4);

    printf("All compression compile-time tests PASSED\n");
    return 0;
}
