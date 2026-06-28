#include "cpu_reference.h"
#include "rdna4.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdio>

namespace rdna4 {

static float fp16ToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= 0x3FF;
    } else if (exponent == 31) {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    exponent += 127 - 15;
    uint32_t raw = (sign << 31) | (exponent << 23) | (mantissa << 13);
    float result;
    memcpy(&result, &raw, sizeof(result));
    return result;
}

static float readF16(const uint8_t* data, size_t offset) {
    uint16_t raw = data[offset] | ((uint16_t)data[offset + 1] << 8);
    return fp16ToFloat(raw);
}

static std::vector<float> dequantize(const uint8_t* data, size_t dataSize,
                                      QuantFormat format, uint32_t blockSize,
                                      uint32_t blockElements, uint32_t nElements) {
    std::vector<float> result(nElements, 0.0f);

    if (format == QuantFormat::F32) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t raw;
            memcpy(&raw, data + i * 4, 4);
            memcpy(&result[i], &raw, 4);
        }
    } else if (format == QuantFormat::F16) {
        for (uint32_t i = 0; i < nElements; ++i) {
            result[i] = readF16(data, i * 2);
        }
    } else if (format == QuantFormat::Q6_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 210;

            float d = readF16(data, bs + 0);

            uint32_t subBlock = eleInBlock / 16;
            int sc = (int8_t)data[bs + 2 + subBlock];

            uint32_t qlByteIdx = eleInBlock / 2;
            uint8_t q4raw = data[bs + 18 + qlByteIdx];
            uint32_t q4 = (eleInBlock & 1u) == 0u ? (q4raw & 0xFu) : (q4raw >> 4);

            uint32_t qhByteIdx = eleInBlock / 4;
            uint8_t qhByte = data[bs + 146 + qhByteIdx];
            uint32_t qhShift = (eleInBlock & 3u) * 2;

            int val = (int)(q4 | (((qhByte >> qhShift) & 3) << 4)) - 32;

            result[i] = d * (float)sc * (float)val;
        }
    } else if (format == QuantFormat::Q8_0) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 34;

            float delta = readF16(data, bs);
            int q = (int8_t)data[bs + 2 + eleInBlock];

            result[i] = delta * (float)q;
        }
    } else if (format == QuantFormat::Q4_0) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 18;

            float delta = readF16(data, bs);
            uint8_t bval = data[bs + 2 + (eleInBlock / 2)];
            uint32_t nibble = (eleInBlock % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);

            result[i] = delta * ((float)nibble - 8.0f);
        }
    } else if (format == QuantFormat::Q4_1) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 20;
            float delta = readF16(data, bs);
            float deltaMin = readF16(data, bs + 2);
            uint8_t bval = data[bs + 4 + (eleInBlock / 2)];
            uint32_t nibble = (eleInBlock % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            result[i] = delta * (float)nibble + deltaMin;
        }
    } else if (format == QuantFormat::Q5_0) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 22;
            float delta = readF16(data, bs);
            uint32_t byteIdx = bs + 2 + (eleInBlock * 5 / 8);
            uint8_t bval = data[byteIdx];
            uint32_t shift = (eleInBlock * 5) % 8;
            uint32_t q5 = (bval >> shift) & 0x1F;
            if (q5 > 15) q5 |= 0xE0;
            result[i] = delta * ((float)(int)q5 - 16.0f);
        }
    } else if (format == QuantFormat::Q5_1) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 24;
            float delta = readF16(data, bs);
            float deltaMin = readF16(data, bs + 2);
            uint32_t byteIdx = bs + 4 + (eleInBlock * 5 / 8);
            uint8_t bval = data[byteIdx];
            uint32_t shift = (eleInBlock * 5) % 8;
            uint32_t q5 = (bval >> shift) & 0x1F;
            result[i] = delta * (float)q5 + deltaMin;
        }
    } else if (format == QuantFormat::Q8_1) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 36;
            float delta = readF16(data, bs);
            float deltaMin = readF16(data, bs + 2);
            int q = (int8_t)data[bs + 4 + eleInBlock];
            result[i] = delta * (float)q + deltaMin;
        }
    } else if (format == QuantFormat::Q2_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 80;
            float d = readF16(data, bs);
            float dmin = readF16(data, bs + 2);
            uint32_t subBlock = eleInBlock / 64;
            uint32_t subEle = eleInBlock % 64;
            uint8_t sc = data[bs + 4 + subBlock];
            uint32_t byteIdx = bs + 8 + subBlock * 16 + subEle / 4;
            uint8_t bval = data[byteIdx];
            uint32_t shift = (subEle % 4) * 2;
            uint32_t q2 = (bval >> shift) & 3;
            result[i] = d * (float)sc * (float)(int)q2 + dmin;
        }
    } else if (format == QuantFormat::Q3_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 112;
            float d = readF16(data, bs);
            float dmin = readF16(data, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint32_t scIdx = bs + 4 + subBlock;
            uint8_t sc = data[scIdx];
            uint32_t byteIdx = bs + 12 + subBlock * 24 + subEle / 2;
            uint8_t bval = data[byteIdx];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            result[i] = d * (float)(int8_t)sc * (float)(int)(nibble - 8) - dmin;
        }
    } else if (format == QuantFormat::Q4_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 144;
            float d = readF16(data, bs);
            float dmin = fp16ToFloat(*(uint16_t*)(data + bs + 2));
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint32_t scOffset = bs + 132 + subBlock * 2;
            float sc = readF16(data, scOffset);
            uint32_t byteIdx = bs + 4 + subBlock * 16 + subEle / 2;
            uint8_t bval = data[byteIdx];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            uint32_t hiByte = 0;
            if (subBlock < 4) {
                uint32_t hiByteIdx = bs + 4 + 64 + subBlock * 8 + subEle / 2;
                hiByte = data[hiByteIdx];
                uint32_t hiNibble = (subEle % 2 == 0) ? (hiByte & 0x0F) : ((hiByte >> 4) & 0x0F);
                nibble |= hiNibble << 4;
            }
            result[i] = sc * ((float)(int)(nibble - 16)) * d - dmin;
        }
    } else if (format == QuantFormat::Q5_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 176;
            float d = readF16(data, bs);
            float dmin = readF16(data, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            float sc = readF16(data, bs + 164 + subBlock * 2);
            uint32_t nibble = 0;
            if (subBlock < 4) {
                uint32_t byteIdx = bs + 4 + subBlock * 20 + subEle / 2;
                uint8_t bval = data[byteIdx];
                uint32_t lo4 = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
                uint32_t hiByteIdx = bs + 4 + 80 + subBlock * 8 + subEle / 2;
                uint8_t bvalHi = data[hiByteIdx];
                uint32_t hi4 = (subEle % 2 == 0) ? (bvalHi & 0x0F) : ((bvalHi >> 4) & 0x0F);
                nibble = lo4 | (hi4 << 4);
            } else {
                uint32_t byteIdx = bs + 4 + 112 + subEle / 2;
                uint8_t bval = data[byteIdx];
                nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            }
            if (nibble >= 16) nibble |= 0xFFF0;
            result[i] = sc * (float)(int)nibble * d - dmin;
        }
    } else if (format == QuantFormat::Q8_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 290;
            float d = readF16(data, bs);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            int sc = (int8_t)data[bs + 2 + subBlock];
            int q = (int8_t)data[bs + 10 + subBlock * 32 + subEle];
            result[i] = d * (float)sc * (float)q;
        }
    }

    return result;
}

bool CpuReference::load(const std::string& jsonPath, const std::string& binPath) {
    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) return false;
    nlohmann::json j;
    jsonFile >> j;

    auto& m = j["model"];
    desc.blockCount = m.value("block_count", 0);
    desc.embeddingLength = m.value("embedding_length", 0);
    desc.feedForwardLength = m.value("feed_forward_length", 0);
    desc.headCount = m.value("attention.head_count", 0);
    desc.headCountKv = m.value("attention.head_count_kv", 0);
    desc.vocabSize = m.value("vocab_size", 0);

    fprintf(stderr, "[cpu-ref] model: blocks=%u dim=%u heads=%u/%u vocab=%u\n",
            desc.blockCount, desc.embeddingLength, desc.headCount,
            desc.headCountKv, desc.vocabSize);

    std::ifstream binFile(binPath, std::ios::binary | std::ios::ate);
    if (!binFile.is_open()) return false;
    size_t binSize = binFile.tellg();
    binFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> binData(binSize);
    binFile.read(reinterpret_cast<char*>(binData.data()), binSize);
    fprintf(stderr, "[cpu-ref] binary loaded: %zu MB\n", binSize / 1024 / 1024);

    auto findTensor = [&](const std::string& name) -> nlohmann::json* {
        for (auto& t : j["tensors"]) {
            if (t.value("name", "") == name) return &t;
        }
        return nullptr;
    };

    auto loadTensor = [&](CpuTensor& tensor, const std::string& name) {
        auto* t = findTensor(name);
        if (!t) {
            fprintf(stderr, "[cpu-ref] tensor not found: %s\n", name.c_str());
            return;
        }

        tensor.shape = t->value("shape", std::vector<uint32_t>{});
        uint32_t nElements = 1;
        for (auto d : tensor.shape) nElements *= d;

        QuantFormat format = (QuantFormat)t->value("dtype_id", 0);
        uint32_t blockSize = t->value("quant_block_size", 1);
        uint32_t blockElements = t->value("quant_block_elements", 1);
        size_t binOffset = t->value<size_t>("bin_offset", 0);
        size_t binSz = t->value<size_t>("bin_size", 0);

        tensor.data = dequantize(binData.data() + binOffset, binSz,
                                  format, blockSize, blockElements, nElements);
        fprintf(stderr, "[cpu-ref] loaded %s: %u elements, format=%d\n",
                name.c_str(), nElements, (int)format);
    };

    loadTensor(tokenEmbD, "token_embd.weight");
    loadTensor(outputWeight, "output.weight");
    if (outputWeight.data.empty()) {
        fprintf(stderr, "[cpu-ref] output.weight not found, falling back to token_embd.weight (weight tying)\n");
        // token_embd.weight is shape [vocabSize, dim] row-major.
        // The LM head expects shape [dim, vocabSize] row-major.
        // Transpose: outputWeight[k * vocabSize + col] = tokenEmbD[col * dim + k]
        uint32_t vs = desc.vocabSize;
        uint32_t d = desc.embeddingLength;
        outputWeight.data.resize((size_t)d * vs);
        for (uint32_t k = 0; k < d; ++k) {
            for (uint32_t col = 0; col < vs; ++col) {
                outputWeight.data[(size_t)k * vs + col] = tokenEmbD.data[(size_t)col * d + k];
            }
        }
    }
    loadTensor(outputNorm, "output_norm.weight");
    if (outputNorm.data.empty()) {
        loadTensor(outputNorm, "norm.weight");
    }

    layers.resize(desc.blockCount);
    for (uint32_t i = 0; i < desc.blockCount; ++i) {
        std::string prefix = "blk." + std::to_string(i);
        loadTensor(layers[i].attnNorm, prefix + ".attn_norm.weight");
        loadTensor(layers[i].attnQ, prefix + ".attn_q.weight");
        loadTensor(layers[i].attnK, prefix + ".attn_k.weight");
        loadTensor(layers[i].attnV, prefix + ".attn_v.weight");
        loadTensor(layers[i].attnOutput, prefix + ".attn_output.weight");
        loadTensor(layers[i].ffnNorm, prefix + ".ffn_norm.weight");
        loadTensor(layers[i].ffnGate, prefix + ".ffn_gate.weight");
        loadTensor(layers[i].ffnUp, prefix + ".ffn_up.weight");
        loadTensor(layers[i].ffnDown, prefix + ".ffn_down.weight");
    }

    fprintf(stderr, "[cpu-ref] model loaded successfully\n");
    return true;
}

std::vector<float> CpuReference::forward(uint32_t tokenId) {
    uint32_t dim = desc.embeddingLength;
    uint32_t hiddenDim = desc.feedForwardLength;
    uint32_t nHeads = desc.headCount;
    uint32_t nKvHeads = desc.headCountKv;
    uint32_t headDim = dim / nHeads;
    uint32_t vocabSize = desc.vocabSize;
    float eps = 1e-6f;
    float ropeBase = 10000.0f;

    std::vector<float> hidden(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        hidden[d] = tokenEmbD.data[tokenId * dim + d];
    }

    std::vector<float> kCache(nKvHeads * headDim);
    std::vector<float> vCache(nKvHeads * headDim);

    for (uint32_t layer = 0; layer < desc.blockCount; ++layer) {
        const auto& l = layers[layer];

        // Pre-attention RMS norm
        float sumSq = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            sumSq += hidden[d] * hidden[d];
        }
        float invRms = 1.0f / std::sqrt(sumSq / dim + eps);
        std::vector<float> normed(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            normed[d] = hidden[d] * invRms * l.attnNorm.data[d];
        }

        // Q GEMM: out[col] = sum_k normed[k] * W[k*dim + col]
        std::vector<float> q(dim);
        for (uint32_t col = 0; col < dim; ++col) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < dim; ++k) {
                acc += normed[k] * l.attnQ.data[k * dim + col];
            }
            q[col] = acc;
        }

        // K GEMM: out[col] = sum_k normed[k] * W[k*kvDim + col]
        uint32_t kvDim = nKvHeads * headDim;
        std::vector<float> k(kvDim);
        for (uint32_t col = 0; col < kvDim; ++col) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < dim; ++kk) {
                acc += normed[kk] * l.attnK.data[kk * kvDim + col];
            }
            k[col] = acc;
        }

        // V GEMM
        std::vector<float> v(kvDim);
        for (uint32_t col = 0; col < kvDim; ++col) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < dim; ++kk) {
                acc += normed[kk] * l.attnV.data[kk * kvDim + col];
            }
            v[col] = acc;
        }

        // RoPE (position = 0, so angle = 0 → no rotation)
        for (uint32_t h = 0; h < nHeads; ++h) {
            for (uint32_t d = 0; d < headDim; d += 2) {
                float theta = std::pow(ropeBase, -(float)d / (float)headDim);
                float angle = 0.0f * theta;
                float cosA = std::cos(angle);
                float sinA = std::sin(angle);
                uint32_t idx = h * headDim + d;
                float q0 = q[idx];
                float q1 = q[idx + 1];
                q[idx]     = q0 * cosA - q1 * sinA;
                q[idx + 1] = q1 * cosA + q0 * sinA;
            }
        }
        for (uint32_t kv = 0; kv < nKvHeads; ++kv) {
            for (uint32_t d = 0; d < headDim; d += 2) {
                float theta = std::pow(ropeBase, -(float)d / (float)headDim);
                float angle = 0.0f * theta;
                float cosA = std::cos(angle);
                float sinA = std::sin(angle);
                uint32_t idx = kv * headDim + d;
                float k0 = k[idx];
                float k1 = k[idx + 1];
                k[idx]     = k0 * cosA - k1 * sinA;
                k[idx + 1] = k1 * cosA + k0 * sinA;
            }
        }

        // KV cache write (seqPos=0)
        kCache = k;
        vCache = v;

        // Attention
        float invSqrtHeadDim = 1.0f / std::sqrt((float)headDim);
        std::vector<float> attnOut(dim, 0.0f);

        for (uint32_t h = 0; h < nHeads; ++h) {
            uint32_t kvHead = h / (nHeads / nKvHeads);

            // Compute score for seqLen=1: only one position
            float score = 0.0f;
            for (uint32_t d = 0; d < headDim; ++d) {
                score += q[h * headDim + d] * kCache[kvHead * headDim + d];
            }
            score *= invSqrtHeadDim;

            // Softmax with single element: weight = 1
            // Output = V[kvHead]
            for (uint32_t d = 0; d < headDim; ++d) {
                attnOut[h * headDim + d] = vCache[kvHead * headDim + d];
            }
        }

        // Output projection GEMM
        std::vector<float> mlpOut(dim);
        for (uint32_t col = 0; col < dim; ++col) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < dim; ++kk) {
                acc += attnOut[kk] * l.attnOutput.data[kk * dim + col];
            }
            mlpOut[col] = acc;
        }

        // Residual add
        for (uint32_t d = 0; d < dim; ++d) {
            hidden[d] += mlpOut[d];
        }

        // Pre-FFN RMS norm
        sumSq = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            sumSq += hidden[d] * hidden[d];
        }
        invRms = 1.0f / std::sqrt(sumSq / dim + eps);
        for (uint32_t d = 0; d < dim; ++d) {
            normed[d] = hidden[d] * invRms * l.ffnNorm.data[d];
        }

        // MLP: gate + up → SiLU(gate) * up → down
        std::vector<float> gate(hiddenDim);
        std::vector<float> up(hiddenDim);
        for (uint32_t h = 0; h < hiddenDim; ++h) {
            float gAcc = 0.0f;
            float uAcc = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                gAcc += normed[d] * l.ffnGate.data[d * hiddenDim + h];
                uAcc += normed[d] * l.ffnUp.data[d * hiddenDim + h];
            }
            float g = gAcc;
            gate[h] = g / (1.0f + std::exp(-g)) * uAcc;
        }

        // Down projection
        std::vector<float> mlpOut2(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (uint32_t h = 0; h < hiddenDim; ++h) {
                acc += gate[h] * l.ffnDown.data[h * dim + d];
            }
            mlpOut2[d] = acc;
        }

        // Residual add
        for (uint32_t d = 0; d < dim; ++d) {
            hidden[d] += mlpOut2[d];
        }
    }

    // Final RMS norm
    float sumSq = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        sumSq += hidden[d] * hidden[d];
    }
    float invRms = 1.0f / std::sqrt(sumSq / dim + eps);
    std::vector<float> normed(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        normed[d] = hidden[d] * invRms * outputNorm.data[d];
    }

    // LM head GEMM: output.weight is shape [dim, vocabSize], row-major
    // logits[col] = sum_k normed[k] * W[k * vocabSize + col]
    std::vector<float> logits(vocabSize);
    for (uint32_t col = 0; col < vocabSize; ++col) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < dim; ++k) {
            acc += normed[k] * outputWeight.data[k * vocabSize + col];
        }
        logits[col] = acc;
    }

    return logits;
}

void CpuReference::printTopK(const std::vector<float>& logits, int k) {
    std::vector<std::pair<float, uint32_t>> indexed(logits.size());
    for (uint32_t i = 0; i < (uint32_t)logits.size(); ++i) {
        indexed[i] = {logits[i], i};
    }
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; ++i) {
        printf("  token %u = %.6f\n", indexed[i].second, indexed[i].first);
    }
}

std::pair<uint32_t, float> CpuReference::argmax(const std::vector<float>& logits) {
    uint32_t best = 0;
    float bestVal = logits[0];
    for (uint32_t i = 1; i < (uint32_t)logits.size(); ++i) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            best = i;
        }
    }
    return {best, bestVal};
}

} // namespace rdna4
