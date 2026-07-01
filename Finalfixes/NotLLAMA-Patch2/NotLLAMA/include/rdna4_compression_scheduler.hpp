#pragma once
#include "rdna4_compression.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>

namespace rdna4 {

// Decision returned by the scheduler each forward pass.
// All flags false/0 means no compression needed.
struct CompressionDecision {
    bool compressContext = false;
    bool compressKV = false;
    uint32_t contextTargetLen = 0;   // Desired output seqLen after context compression
    uint32_t kvQuantizeBits = 0;     // 0 = no change, 4/5/6/8 = re-quantize
    std::vector<uint8_t> keepMask;   // 1 = keep token, 0 = drop. Empty if compressContext=false.
};
static_assert(sizeof(CompressionDecision) < 1024,
    "CompressionDecision should stay small (host-only but keep it tight)");

// Host-side scheduler that monitors sequence state and decides when to compress.
// Does NOT execute compression — it only decides.
// Context compression triggers when seqLen > maxContext * ctxCfg_.threshold.
// KV compression triggers when seqLen >= kvCfg_.minSeqLen AND kvCfg_.enabled.
// keepMask generation depends on ctxCfg_.strategy:
//   0 (uniform): keep every Nth token to reach target length
//   1 (entropy): placeholder — requires entropy scores from model
//   2 (importance): use importanceScores to keep top-K tokens
class CompressionScheduler {
public:
    explicit CompressionScheduler(const ContextCompressionConfig& ctxCfg,
                                   const KVCompressionConfig& kvCfg)
        : ctxCfg_(ctxCfg), kvCfg_(kvCfg) {}

    // Call once per token generation step.
    // seqLen = current sequence length (including new token)
    // maxContext = model's max context length
    // importanceScores = optional per-token importance (empty if unavailable)
    CompressionDecision step(uint32_t seqLen,
                              uint32_t maxContext,
                              const std::vector<float>& importanceScores = {}) {
        CompressionDecision d;

        // Context compression trigger
        if (ctxCfg_.enabled && seqLen > static_cast<uint32_t>(maxContext * ctxCfg_.threshold)) {
            d.compressContext = true;
            uint32_t targetLen = static_cast<uint32_t>(maxContext * ctxCfg_.threshold * 0.6f);
            if (targetLen < 64) targetLen = 64;
            d.contextTargetLen = targetLen;

            // Build keep mask
            d.keepMask.resize(seqLen, 1);
            if (ctxCfg_.strategy == 0) {
                // uniform: keep every Nth token
                uint32_t step = seqLen / targetLen;
                if (step < 1) step = 1;
                for (uint32_t i = 0; i < seqLen; ++i) {
                    if (i % step != 0) d.keepMask[i] = 0;
                }
            } else if (ctxCfg_.strategy == 2 && !importanceScores.empty()) {
                // importance: keep top-K tokens
                // Build index array and partial sort by importance
                std::vector<uint32_t> idx(seqLen);
                for (uint32_t i = 0; i < seqLen; ++i) idx[i] = i;
                // Simple O(n log n) — acceptable for seqLen < 4096
                std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
                    return importanceScores[a] > importanceScores[b];
                });
                std::fill(d.keepMask.begin(), d.keepMask.end(), 0);
                for (uint32_t i = 0; i < targetLen && i < seqLen; ++i) {
                    d.keepMask[idx[i]] = 1;
                }
            }
            // strategy == 1 (entropy): placeholder — keep last targetLen tokens
            if (ctxCfg_.strategy == 1) {
                std::fill(d.keepMask.begin(), d.keepMask.end(), 0);
                uint32_t start = (seqLen > targetLen) ? seqLen - targetLen : 0;
                for (uint32_t i = start; i < seqLen; ++i) d.keepMask[i] = 1;
            }
        }

        // KV compression trigger
        if (kvCfg_.enabled && seqLen >= kvCfg_.minSeqLen) {
            d.compressKV = true;
            d.kvQuantizeBits = kvCfg_.kBits;
        }

        if (!contextCompressed_) {
            totalCompressed_ += seqLen;
            contextCompressed_ = d.compressContext;
        }
        if (!kvCompressed_) {
            kvCompressed_ = d.compressKV;
        }

        return d;
    }

    void reset() {
        lastSeqLen_ = 0;
        contextCompressed_ = false;
        kvCompressed_ = false;
    }

    uint32_t totalCompressedTokens() const { return totalCompressed_; }
    uint32_t totalCompressedLayers() const { return contextCompressed_ ? 1 : 0; }

private:
    ContextCompressionConfig ctxCfg_;
    KVCompressionConfig kvCfg_;
    uint32_t lastSeqLen_ = 0;
    uint32_t totalCompressed_ = 0;
    bool contextCompressed_ = false;
    bool kvCompressed_ = false;
};

} // namespace rdna4
