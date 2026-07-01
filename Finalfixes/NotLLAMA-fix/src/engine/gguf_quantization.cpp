#include "engine/gguf_quantization.hpp"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace notllama {

namespace {

inline uint16_t floatToHalf(float f) {
    // Minimal fp32 -> fp16 conversion (round-to-nearest, no denormals/inf/nan handling).
    uint32_t x;
    static_assert(sizeof(x) == sizeof(f), "float size mismatch");
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1u;
    uint32_t expn = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (expn == 0xFF) {
        // inf/nan
        return static_cast<uint16_t>((sign << 15) | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (expn == 0) {
        // zero / denormal
        return static_cast<uint16_t>(sign << 15);
    }

    int32_t new_exp = static_cast<int32_t>(expn) - 127 + 15;
    if (new_exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00u); // inf
    }
    if (new_exp <= 0) {
        return static_cast<uint16_t>(sign << 15); // flush to zero
    }

    uint32_t new_mant = mant >> 13;
    uint32_t round_bit = (mant >> 12) & 1u;
    if (round_bit) {
        new_mant += 1;
        if (new_mant >= 0x400u) {
            new_mant = 0;
            new_exp += 1;
            if (new_exp >= 31) {
                return static_cast<uint16_t>((sign << 15) | 0x7C00u);
            }
        }
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | new_mant);
}

class F32Quantization : public IQuantization {
public:
    DataType GetType() override { return DataType::F32; }
    size_t PackedBytesPerElement() override { return 4; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        std::memcpy(dst, src, tile_size * 4);
    }
    uint32_t GetShaderSpecializationID() override { return 0; }
    size_t GetBlockAlignment() override { return 4; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 3u) == 0;
    }
};

class F16Quantization : public IQuantization {
public:
    DataType GetType() override { return DataType::F16; }
    size_t PackedBytesPerElement() override { return 2; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        uint16_t* out = reinterpret_cast<uint16_t*>(dst);
        for (size_t i = 0; i < tile_size; ++i) out[i] = floatToHalf(src[i]);
    }
    uint32_t GetShaderSpecializationID() override { return 1; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

class Q4_0Quantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q4_0; }
    size_t PackedBytesPerElement() override { return 18; } // per 32 elements
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        const size_t block_elems = 32;
        size_t num_blocks = tile_size / block_elems;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* blk = src + b * block_elems;
            uint8_t* out = dst + b * 18;
            float amax = 0.0f;
            for (size_t i = 0; i < block_elems; ++i) {
                float a = std::abs(blk[i]);
                if (a > amax) amax = a;
            }
            float delta = amax / 8.0f;
            uint16_t d16 = floatToHalf(delta);
            std::memcpy(out, &d16, 2);
            for (size_t i = 0; i < block_elems; i += 2) {
                float v0 = blk[i];
                float v1 = blk[i + 1];
                int q0 = delta > 0.0f ? static_cast<int>(std::round(v0 / delta)) + 8 : 8;
                int q1 = delta > 0.0f ? static_cast<int>(std::round(v1 / delta)) + 8 : 8;
                if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                out[2 + i / 2] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    }
    uint32_t GetShaderSpecializationID() override { return 2; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

class Q8_0Quantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q8_0; }
    size_t PackedBytesPerElement() override { return 34; } // per 32 elements
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        const size_t block_elems = 32;
        size_t num_blocks = tile_size / block_elems;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* blk = src + b * block_elems;
            uint8_t* out = dst + b * 34;
            float amax = 0.0f;
            for (size_t i = 0; i < block_elems; ++i) {
                float a = std::abs(blk[i]);
                if (a > amax) amax = a;
            }
            float delta = amax / 127.0f;
            uint16_t d16 = floatToHalf(delta);
            std::memcpy(out, &d16, 2);
            for (size_t i = 0; i < block_elems; ++i) {
                int q = delta > 0.0f ? static_cast<int>(std::round(blk[i] / delta)) : 0;
                if (q < -127) q = -127; if (q > 127) q = 127;
                out[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
            }
        }
    }
    uint32_t GetShaderSpecializationID() override { return 8; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

// ── Q4_1, Q5_0, Q5_1 stubs (simple block formats, not priority) ──
template <DataType Type, uint32_t SpecID, size_t BytesPerBlock, size_t ElemsPerBlock = 32>
class StubQuantization : public IQuantization {
public:
    DataType GetType() override { return Type; }
    size_t PackedBytesPerElement() override { return BytesPerBlock; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        size_t num_blocks = tile_size / ElemsPerBlock;
        std::memset(dst, 0, num_blocks * BytesPerBlock);
        (void)src;
    }
    uint32_t GetShaderSpecializationID() override { return SpecID; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

using Q4_1Quantization = StubQuantization<DataType::Q4_1, 3, 20, 32>;
using Q5_0Quantization = StubQuantization<DataType::Q5_0, 5, 22, 32>;
using Q5_1Quantization = StubQuantization<DataType::Q5_1, 6, 24, 32>;

// ── Q4_K: 256-element super-block, 8 sub-blocks of 32, 144 bytes ──
// Layout: [d(fp16,2)] [dmin(fp16,2)] [scales(12)] [qs(128)]
// Scales packing: 12 bytes encode 8×6-bit sc + 8×6-bit sm
//   For sc[sb] (sb=0..3): scales[sb] bits 5-0
//   For sm[sb] (sb=0..3): scales[4+sb] bits 5-0
//   For sc[sb] (sb=4..7): (scales[8+sb-4] & 0x0F) | ((scales[sb-4] >> 6) << 4)
//   For sm[sb] (sb=4..7): (scales[8+sb-4] >> 4) | ((scales[4+sb-4] >> 6) << 4)
// qs packing: (sb, elem) pairs share bytes.
//   byte = (sb>>1)*32 + elem, low nibble for even sb, high nibble for odd sb
class Q4_KQuantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q4_K; }
    size_t PackedBytesPerElement() override { return 144; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        constexpr size_t BLOCK = 256, SUB = 32, SUB_PER_BLOCK = 8;
        size_t num_blocks = tile_size / BLOCK;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* block = src + b * BLOCK;
            uint8_t* out = dst + b * 144;
            // Block-level min/max
            float fmin = block[0], fmax = block[0];
            for (size_t i = 1; i < BLOCK; ++i) {
                fmin = std::min(fmin, block[i]);
                fmax = std::max(fmax, block[i]);
            }
            float range = fmax - fmin;
            float d_val = range > 0 ? range / (15.0f * 63.0f) : 1e-10f;
            float dmin_val = fmin < 0 ? -fmin / 63.0f : 1e-10f;
            if (fmin >= 0) dmin_val = 1e-10f;
            uint16_t d_f16 = floatToHalf(d_val);
            uint16_t dmin_f16 = floatToHalf(dmin_val);
            std::memcpy(out, &d_f16, 2);
            std::memcpy(out + 2, &dmin_f16, 2);

            // sc/sm values for all 8 sub-blocks (each 0..63)
            int sc_vals[8] = {}, sm_vals[8] = {};
            uint8_t qs[128] = {};
            for (size_t sb = 0; sb < SUB_PER_BLOCK; ++sb) {
                size_t sstart = sb * SUB;
                float sb_min = block[sstart], sb_max = block[sstart];
                for (size_t i = 1; i < SUB; ++i) {
                    float v = block[sstart + i];
                    sb_min = std::min(sb_min, v);
                    sb_max = std::max(sb_max, v);
                }
                int sc_val = 1, sm_val = 1;
                if (d_val > 0) {
                    sc_val = (int)std::round((sb_max - sb_min) / (d_val * 15.0f));
                    if (sc_val < 1) sc_val = 1; if (sc_val > 63) sc_val = 63;
                }
                if (dmin_val > 0) {
                    sm_val = (int)std::round(-sb_min / dmin_val);
                    if (sm_val < 1) sm_val = 1; if (sm_val > 63) sm_val = 63;
                }
                sc_vals[sb] = sc_val;
                sm_vals[sb] = sm_val;
                // Quantize 32 elements to 4-bit nibbles
                float inv_scale = 1.0f / (d_val * static_cast<float>(sc_val));
                for (size_t i = 0; i < SUB; ++i) {
                    float qf = (block[sstart + i] + dmin_val * static_cast<float>(sm_val)) * inv_scale;
                    int qi = (int)std::round(qf);
                    if (qi < 0) qi = 0; if (qi > 15) qi = 15;
                    size_t byte_idx = (sb >> 1) * 32 + i;
                    if ((sb & 1) == 0)
                        qs[byte_idx] = (qs[byte_idx] & 0xF0) | (static_cast<uint8_t>(qi) & 0x0F);
                    else
                        qs[byte_idx] = (qs[byte_idx] & 0x0F) | ((static_cast<uint8_t>(qi) & 0x0F) << 4);
                }
            }
            // Pack scales into 12 bytes matching shader get_scale_min_k4 format:
            // For sb=0..3: sc scales[0..3] low 6 bits, sm scales[4..7] low 6 bits
            // For sb=4..7: sc split between scales[0..3] bits 7-6 and scales[8..11] low nibble
            //              sm split between scales[4..7] bits 7-6 and scales[8..11] high nibble
            for (size_t i = 0; i < 4; ++i) {
                int sc0 = sc_vals[i], sm0 = sm_vals[i];
                int sc4 = sc_vals[4 + i], sm4 = sm_vals[4 + i];
                out[4 + i] = static_cast<uint8_t>((sc0 & 0x3F) | ((sc4 >> 4) << 6));
                out[8 + i] = static_cast<uint8_t>((sm0 & 0x3F) | ((sm4 >> 4) << 6));
                out[12 + i] = static_cast<uint8_t>(((sm4 & 0x0F) << 4) | (sc4 & 0x0F));
            }
            std::memcpy(out + 16, qs, 128);
        }
    }
    uint32_t GetShaderSpecializationID() override { return 12; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

// ── Q5_K: 256-element block, 8 sub-blocks of 32, 176 bytes ──
// Layout: [d(fp16,2)] [dmin(fp16,2)] [scales(12)] [qh(32)] [qs(128)]
// scales: per sub-block pair, low nibble=scale, high nibble=min
// qh: one bit per element (5th bit)
// qs: nibbles for lower 4 bits
// Dequant: result = subScale * q5 - subMin, q5 = (q4 | (highBit<<4)) - 16
class Q5_KQuantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q5_K; }
    size_t PackedBytesPerElement() override { return 176; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        constexpr size_t BLOCK = 256, SUB = 32, SUB_PER_BLOCK = 8;
        size_t num_blocks = tile_size / BLOCK;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* block = src + b * BLOCK;
            uint8_t* out = dst + b * 176;
            float fmin = block[0], fmax = block[0];
            for (size_t i = 1; i < BLOCK; ++i) {
                fmin = std::min(fmin, block[i]);
                fmax = std::max(fmax, block[i]);
            }
            float d_val = (fmax - fmin) / (31.0f * 15.0f);
            float dmin_val = fmin < 0 ? -(fmin + 16.0f * d_val * 15.0f) / 15.0f : 1e-10f;
            if (d_val < 1e-10f) d_val = 1e-10f;
            if (dmin_val < 1e-10f) dmin_val = 1e-10f;
            uint16_t d_f16 = floatToHalf(d_val);
            uint16_t dmin_f16 = floatToHalf(dmin_val);
            std::memcpy(out, &d_f16, 2);
            std::memcpy(out + 2, &dmin_f16, 2);
            uint8_t scales[12] = {};
            uint8_t qh[32] = {};
            uint8_t qs[128] = {};
            for (size_t sb = 0; sb < SUB_PER_BLOCK; ++sb) {
                size_t sstart = sb * SUB;
                float sb_min = block[sstart], sb_max = block[sstart];
                for (size_t i = 1; i < SUB; ++i) {
                    float v = block[sstart + i];
                    sb_min = std::min(sb_min, v);
                    sb_max = std::max(sb_max, v);
                }
                float subScaleVal = (sb_max - sb_min) / 31.0f;
                float subMinVal = sb_min + 16.0f * subScaleVal;
                int sc = (int)std::round(subScaleVal / d_val);
                if (sc < 1) sc = 1; if (sc > 15) sc = 15;
                int sm = (int)std::round(-subMinVal / dmin_val);
                if (sm < 0) sm = 0; if (sm > 15) sm = 15;
                size_t sc_idx = 4 + (sb >> 1);
                uint8_t old = scales[sc_idx];
                if ((sb & 1) == 0)
                    old = (old & 0xF0) | (static_cast<uint8_t>(sc) & 0x0F);
                else
                    old = (old & 0x0F) | ((static_cast<uint8_t>(sm) & 0x0F) << 4);
                scales[sc_idx] = old;
                float inv_scale = 1.0f / (d_val * static_cast<float>(sc));
                for (size_t i = 0; i < SUB; ++i) {
                    size_t e = sb * SUB + i;
                    float qf = (block[e] + dmin_val * static_cast<float>(sm)) * inv_scale;
                    int qi = (int)std::round(qf);
                    if (qi < -16) qi = -16; if (qi > 15) qi = 15;
                    int q4 = qi & 0x0F;
                    int bit5 = (qi >> 4) & 1;
                    // qs: flat per-element layout, 2 elements per byte
                    if ((e & 1) == 0)
                        qs[e / 2] = (qs[e / 2] & 0xF0) | (q4 & 0x0F);
                    else
                        qs[e / 2] = (qs[e / 2] & 0x0F) | ((q4 & 0x0F) << 4);
                    // qh: flat per-element, one bit per element
                    if (bit5) qh[e / 8] |= (1u << (e % 8));
                }
            }
            std::memcpy(out + 4, scales, 12);
            std::memcpy(out + 16, qh, 32);
            std::memcpy(out + 48, qs, 128);
        }
    }
    uint32_t GetShaderSpecializationID() override { return 13; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

// ── Q6_K: 256-element block, 16 sub-blocks of 16, 210 bytes (d-last) ──
// Layout: [ql(128)] [qh(64)] [scales(16)] [d(fp16,2)]
// ql: nibbles, lower 4 bits per element
// qh: 2 bits per element, upper bits
// scales: int8 per sub-block
// d at byte 208
class Q6_KQuantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q6_K; }
    size_t PackedBytesPerElement() override { return 210; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        constexpr size_t BLOCK = 256, SUB = 16, SUB_PER_BLOCK = 16;
        size_t num_blocks = tile_size / BLOCK;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* block = src + b * BLOCK;
            uint8_t* out = dst + b * 210;
            float fmin = block[0], fmax = block[0];
            for (size_t i = 1; i < BLOCK; ++i) {
                fmin = std::min(fmin, block[i]);
                fmax = std::max(fmax, block[i]);
            }
            // Q6_K: val = d * sc * q, q = (hi2<<4 | lo2) - 32, signed 6-bit, -32..31
            // Range per sub-block = d * sc * 63 (max variation)
            // Want: d * sc * 63 ≈ sb_max - sb_min
            float d_val = (fmax - fmin) / (63.0f * 127.0f);
            if (d_val < 1e-10f) d_val = 1e-10f;
            uint16_t d_f16 = floatToHalf(d_val);
            std::memcpy(out + 208, &d_f16, 2);
            uint8_t ql[128] = {}, qh[64] = {};
            int8_t scales[16] = {};
            for (size_t sb = 0; sb < SUB_PER_BLOCK; ++sb) {
                size_t sstart = sb * SUB;
                float sb_min = block[sstart], sb_max = block[sstart];
                for (size_t i = 1; i < SUB; ++i) {
                    float v = block[sstart + i];
                    sb_min = std::min(sb_min, v);
                    sb_max = std::max(sb_max, v);
                }
                int sc_val = (int)std::round((sb_max - sb_min) / (d_val * 63.0f));
                if (sc_val < 1) sc_val = 1; if (sc_val > 127) sc_val = 127;
                scales[sb] = (int8_t)sc_val;
                float inv = 1.0f / (d_val * (float)sc_val);
                for (size_t i = 0; i < SUB; ++i) {
                    float qf = block[sstart + i] * inv;
                    int qi = (int)std::round(qf);
                    if (qi < -32) qi = -32; if (qi > 31) qi = 31;
                    int lo4 = qi & 0xF;
                    int hi2 = (qi >> 4) & 0x3;
                    size_t lo_idx = (sstart + i) / 2;
                    size_t hi_idx = (sstart + i) / 4;
                    if (((sstart + i) & 1) == 0)
                        ql[lo_idx] = (ql[lo_idx] & 0xF0) | (lo4 & 0x0F);
                    else
                        ql[lo_idx] = (ql[lo_idx] & 0x0F) | ((lo4 & 0x0F) << 4);
                    qh[hi_idx] |= (hi2 & 0x3) << (((sstart + i) & 3) * 2);
                }
            }
            std::memcpy(out, ql, 128);
            std::memcpy(out + 128, qh, 64);
            std::memcpy(out + 192, scales, 16);
        }
    }
    uint32_t GetShaderSpecializationID() override { return 14; }
    size_t GetBlockAlignment() override { return 2; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 1u) == 0;
    }
};

// ── Q8_K: 256-element block, 292 bytes (d float32 first) ──
// Layout: [d(float32,4)] [qs(int8,256)] [bsums(int16,32)]
// Dequant: result = d * q
class Q8_KQuantization : public IQuantization {
public:
    DataType GetType() override { return DataType::Q8_K; }
    size_t PackedBytesPerElement() override { return 292; }
    void CompressTile(const float* src, uint8_t* dst, size_t tile_size) override {
        constexpr size_t BLOCK = 256;
        size_t num_blocks = tile_size / BLOCK;
        for (size_t b = 0; b < num_blocks; ++b) {
            const float* block = src + b * BLOCK;
            uint8_t* out = dst + b * 292;
            float amax = 0.0f;
            for (size_t i = 0; i < BLOCK; ++i) {
                float a = std::abs(block[i]);
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            if (d < 1e-10f) d = 1e-10f;
            std::memcpy(out, &d, 4);
            int32_t bsums[16] = {};
            for (size_t i = 0; i < BLOCK; ++i) {
                int q = (int)std::round(block[i] / d);
                if (q < -128) q = -128; if (q > 127) q = 127;
                out[4 + i] = (uint8_t)(int8_t)q;
                bsums[i / 16] += q;
            }
            for (size_t i = 0; i < 16; ++i)
                std::memcpy(out + 260 + i * 2, &bsums[i], 2);
        }
    }
    uint32_t GetShaderSpecializationID() override { return 15; }
    size_t GetBlockAlignment() override { return 4; }
    bool ValidateBlockAlignment(const void* data) override {
        return (reinterpret_cast<uintptr_t>(data) & 3u) == 0;
    }
};

} // namespace

std::unique_ptr<IQuantization> CreateQuantization(DataType type) {
    switch (type) {
        case DataType::F32: return std::make_unique<F32Quantization>();
        case DataType::F16: return std::make_unique<F16Quantization>();
        case DataType::Q4_0: return std::make_unique<Q4_0Quantization>();
        case DataType::Q4_1: return std::make_unique<Q4_1Quantization>();
        case DataType::Q5_0: return std::make_unique<Q5_0Quantization>();
        case DataType::Q5_1: return std::make_unique<Q5_1Quantization>();
        case DataType::Q8_0: return std::make_unique<Q8_0Quantization>();
        case DataType::Q4_K: return std::make_unique<Q4_KQuantization>();
        case DataType::Q5_K: return std::make_unique<Q5_KQuantization>();
        case DataType::Q6_K: return std::make_unique<Q6_KQuantization>();
        case DataType::Q8_K: return std::make_unique<Q8_KQuantization>();
        default: return nullptr;
    }
}

} // namespace notllama
