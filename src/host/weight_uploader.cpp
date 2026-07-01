#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"
#include "loaders/gguf.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <new>
#include <cmath>
#include <vector>
#include <cstdint>

namespace rdna4 {

static QuantFormat ggmlToQuantFormat(int ggmlType) {
    switch (ggmlType) {
        case  0: return QuantFormat::F32;
        case  1: return QuantFormat::F16;
        case  2: return QuantFormat::Q4_0;
        case  3: return QuantFormat::Q4_1;
        case  6: return QuantFormat::Q5_0;
        case  7: return QuantFormat::Q5_1;
        case  8: return QuantFormat::Q8_0;
        case  9: return QuantFormat::Q8_1;
        case 10: return QuantFormat::Q2_K;
        case 11: return QuantFormat::Q3_K;
        case 12: return QuantFormat::Q4_K;
        case 13: return QuantFormat::Q5_K;
        case 14: return QuantFormat::Q6_K;
        case 15: return QuantFormat::Q8_K;
        case 16: return QuantFormat::IQ2_XXS;
        case 17: return QuantFormat::IQ2_XS;
        case 18: return QuantFormat::IQ3_XXS;
        case 19: return QuantFormat::IQ1_S;
        case 20: return QuantFormat::IQ4_NL;
        case 21: return QuantFormat::IQ3_S;
        case 22: return QuantFormat::IQ2_S;
        case 23: return QuantFormat::IQ4_XS;
        case 24: return QuantFormat::I8;
        case 25: return QuantFormat::I16;
        case 26: return QuantFormat::I32;
        case 27: return QuantFormat::I64;
        case 28: return QuantFormat::F64;
        case 29: return QuantFormat::IQ1_M;
        case 30: return QuantFormat::BF16;
        case 34: return QuantFormat::TQ1_0;
        case 35: return QuantFormat::TQ2_0;
        case 39: return QuantFormat::MXFP4;
        case 40: return QuantFormat::NVFP4;
        case 41: return QuantFormat::Q1_0;
        default:
            fprintf(stderr, "[FATAL] Unsupported GGML tensor type %d\n", ggmlType);
            return QuantFormat::UNSUPPORTED;
    }
}

static bool isQuantized(QuantFormat fmt) {
    return fmt != QuantFormat::F32 && fmt != QuantFormat::F16;
}

// Block size lookup for each quant format
static uint32_t getBlockElements(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::Q4_0: return 32;
        case QuantFormat::Q4_1: return 32;
        case QuantFormat::Q5_0: return 32;
        case QuantFormat::Q5_1: return 32;
        case QuantFormat::Q8_0: return 32;
        case QuantFormat::Q8_1: return 32;
        case QuantFormat::Q2_K: return 256;
        case QuantFormat::Q3_K: return 256;
        case QuantFormat::Q4_K: return 256;
        case QuantFormat::Q5_K: return 256;
        case QuantFormat::Q6_K: return 256;
        case QuantFormat::Q8_K: return 256;
        default: return 1;
    }
}

static uint32_t getBlockSize(QuantFormat fmt) {
    switch (fmt) {
        case QuantFormat::Q4_0: return 18;
        case QuantFormat::Q4_1: return 20;
        case QuantFormat::Q5_0: return 22;
        case QuantFormat::Q5_1: return 24;
        case QuantFormat::Q8_0: return 34;
        case QuantFormat::Q8_1: return 36;
        case QuantFormat::Q2_K: return 84;
        case QuantFormat::Q3_K: return 110;
        case QuantFormat::Q4_K: return 144;
        case QuantFormat::Q5_K: return 176;
        case QuantFormat::Q6_K: return 210;
        case QuantFormat::Q8_K: return 292;
        default: return (fmt == QuantFormat::F16 || fmt == QuantFormat::BF16) ? 2 : 4;
    }
}

static float fp16ToFloatCpu(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
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

static float readF16Cpu(const uint8_t* data, size_t offset) {
    uint16_t raw = data[offset] | ((uint16_t)data[offset + 1] << 8);
    return fp16ToFloatCpu(raw);
}

static bool cpuDequantToFloat(const uint8_t* srcData, size_t srcSize,
                               QuantFormat format, uint32_t blockSize,
                               uint32_t blockElements, uint32_t nElements,
                               std::vector<float>& out) {
    out.resize(nElements, 0.0f);

    if (format == QuantFormat::F32) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t raw;
            memcpy(&raw, srcData + i * 4, 4);
            memcpy(&out[i], &raw, 4);
        }
        return true;
    }
    if (format == QuantFormat::F16) {
        for (uint32_t i = 0; i < nElements; ++i) {
            out[i] = readF16Cpu(srcData, i * 2);
        }
        return true;
    }
    if (format == QuantFormat::Q8_0) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 34;
            if (bs + 2 + eleInBlock >= srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            int q = (int8_t)srcData[bs + 2 + eleInBlock];
            out[i] = delta * (float)q;
        }
        return true;
    }
    if (format == QuantFormat::Q4_0) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 18;
            if (bs + 2 + (eleInBlock / 2) >= srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            uint8_t bval = srcData[bs + 2 + (eleInBlock / 2)];
            uint32_t nibble = (eleInBlock % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            out[i] = delta * ((float)nibble - 8.0f);
        }
        return true;
    }
    if (format == QuantFormat::Q6_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 210;
            if (bs + 210 > srcSize) continue;
            float d = readF16Cpu(srcData, bs + 208);
            uint32_t subBlock = eleInBlock / 16;
            int sc = (int8_t)srcData[bs + 192 + subBlock];
            uint32_t qlByteIdx = eleInBlock / 2;
            uint8_t q4raw = srcData[bs + 0 + qlByteIdx];
            uint32_t q4 = (eleInBlock & 1u) == 0u ? (q4raw & 0xFu) : (q4raw >> 4);
            uint32_t qhByteIdx = eleInBlock / 4;
            uint8_t qhByte = srcData[bs + 128 + qhByteIdx];
            uint32_t qhShift = (eleInBlock & 3u) * 2;
            int val = (int)(q4 | (((qhByte >> qhShift) & 3) << 4)) - 32;
            out[i] = d * (float)sc * (float)val;
        }
        return true;
    }
    if (format == QuantFormat::Q4_K) {
        // Layout: d(fp16)@0, dmin(fp16)@2, scales(12)@4, qs(128)@16
        // scales: 12 bytes packing 8 × 6-bit scale + 8 × 6-bit min (get_scale_min_k4)
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 144;
            if (bs + 144 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            float dmin = readF16Cpu(srcData, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;

            uint8_t sc, sm;
            if (subBlock < 4) {
                sc = srcData[bs + 4 + subBlock] & 63;
                sm = srcData[bs + 4 + subBlock + 4] & 63;
            } else {
                uint8_t bJp4 = srcData[bs + 4 + subBlock + 4];
                uint8_t bJm4 = srcData[bs + 4 + subBlock - 4];
                uint8_t bJ0  = srcData[bs + 4 + subBlock];
                sc = (bJp4 & 0x0F) | ((bJm4 >> 6) << 4);
                sm = (bJp4 >> 4)    | ((bJ0  >> 6) << 4);
            }

            uint32_t qsChunk = (subBlock / 2) * 32;
            uint32_t qsOffset = bs + 16 + qsChunk + subEle;
            if (qsOffset >= srcSize) continue;
            uint8_t bval = srcData[qsOffset];
            uint32_t nibble = (subBlock & 1) == 0 ? (bval & 0x0F) : ((bval >> 4) & 0x0F);

            out[i] = (float)(int)sc * d * (float)nibble - (float)(int)sm * dmin;
        }
        return true;
    }
    if (format == QuantFormat::Q5_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 176;
            if (bs + 176 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            float dmin = readF16Cpu(srcData, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint32_t scOffset = bs + 132 + subBlock * 2;
            float sc = readF16Cpu(srcData, scOffset);
            uint32_t byteIdx = bs + 4 + subBlock * 16 + subEle / 2;
            if (byteIdx >= srcSize) continue;
            uint8_t bval = srcData[byteIdx];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            uint32_t hiByteIdx = bs + 4 + 64 + subBlock * 8 + subEle / 2;
            uint32_t hi = 0;
            if (hiByteIdx < srcSize) {
                uint8_t hb = srcData[hiByteIdx];
                hi = ((subEle % 2 == 0) ? (hb & 0x0F) : ((hb >> 4) & 0x0F)) << 4;
            }
            uint32_t qhIdx = bs + 4 + 64 + 32 + subBlock * 4 + subEle / 8;
            uint32_t qh = 0;
            if (qhIdx < srcSize) qh = (srcData[qhIdx] >> (subEle % 8)) & 1;
            int val = (int)(nibble | hi | (qh << 5)) - 16;
            out[i] = sc * (float)val + dmin;
        }
        return true;
    }
    if (format == QuantFormat::Q8_K) {
        // block_q8_k: { float d; int8_t qs[256]; int16_t bsums[16]; } = 292 bytes
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 292;
            if (bs + 292 > srcSize) continue;
            float d;
            memcpy(&d, &srcData[bs], sizeof(float));
            int q = (int8_t)srcData[bs + 4 + eleInBlock];
            out[i] = d * (float)q;
        }
        return true;
    }
    if (format == QuantFormat::Q3_K) {
        // block_q3_k: { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; fp16_t d; } = 110 bytes
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 110;
            if (bs + 110 > srcSize) continue;
            float d = readF16Cpu(srcData, bs + 108);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint8_t sc = srcData[bs + 96 + subBlock];
            uint32_t qsIdx = bs + 32 + (subEle / 2);
            uint8_t bval = srcData[qsIdx];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            uint32_t hmaskIdx = bs + (subEle / 8);
            uint32_t hmaskBit = (srcData[hmaskIdx] >> (subEle % 8)) & 1;
            int q3 = (nibble & 0x3) | (hmaskBit << 2);
            if (nibble & 0x8) q3 -= 4;
            out[i] = d * (float)(int8_t)sc * (float)q3;
        }
        return true;
    }
    if (format == QuantFormat::Q2_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 80;
            if (bs + 80 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            float dmin = readF16Cpu(srcData, bs + 2);
            uint32_t subBlock = eleInBlock / 64;
            uint32_t subEle = eleInBlock % 64;
            uint8_t sc = srcData[bs + 4 + subBlock];
            uint32_t byteIdx = bs + 8 + subBlock * 16 + subEle / 4;
            if (byteIdx >= srcSize) continue;
            uint8_t bval = srcData[byteIdx];
            uint32_t shift = (subEle % 4) * 2;
            uint32_t q2 = (bval >> shift) & 3;
            out[i] = d * (float)sc * (float)(int)q2 + dmin;
        }
        return true;
    }
    if (format == QuantFormat::Q5_0) {
        // block_q5_0: { fp16_t d; uint8_t qh[4]; uint8_t qs[16]; } = 22 bytes
        // qs pairs: element j (low nibble) + element j+16 (high nibble) for j=0..15
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 22;
            if (bs + 22 > srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            uint8_t qh_byte = srcData[bs + 2 + (eleInBlock / 8)];
            uint8_t qs_val = srcData[bs + 6 + (eleInBlock & 0xF)];
            uint32_t lo = (eleInBlock < 16) ? (qs_val & 0x0F) : ((qs_val >> 4) & 0x0F);
            uint32_t hi = (qh_byte >> (eleInBlock % 8)) & 1;
            int q5 = (int)(lo | (hi << 4)) - 16;
            out[i] = delta * (float)q5;
        }
        return true;
    }
    if (format == QuantFormat::Q5_1) {
        // block_q5_1: { fp16_t d; fp16_t m; uint8_t qh[4]; uint8_t qs[16]; } = 24 bytes
        // qs pairs: element j (low nibble) + element j+16 (high nibble) for j=0..15
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 24;
            if (bs + 24 > srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            float deltaMin = readF16Cpu(srcData, bs + 2);
            uint8_t qh_byte = srcData[bs + 4 + (eleInBlock / 8)];
            uint8_t qs_val = srcData[bs + 8 + (eleInBlock & 0xF)];
            uint32_t lo = (eleInBlock < 16) ? (qs_val & 0x0F) : ((qs_val >> 4) & 0x0F);
            uint32_t hi = (qh_byte >> (eleInBlock % 8)) & 1;
            int q5 = (int)(lo | (hi << 4));
            out[i] = delta * (float)q5 + deltaMin;
        }
        return true;
    }
    // Unknown format — output zeros
    fprintf(stderr, "[cpu-dequant] WARNING: unsupported format %u, outputting zeros\n", (uint32_t)format);
    return false;
}

static std::string layerPrefix(uint32_t layerIndex) {
    return "blk." + std::to_string(layerIndex) + ".";
}

static bool tensorBelongsToLayer(const std::string& name, uint32_t layerIndex) {
    std::string prefix = layerPrefix(layerIndex);
    return name.compare(0, prefix.size(), prefix) == 0;
}

ModelDesc WeightUploader::loadMetadata(const std::string& jsonPath, const std::string& binPath) {
    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        fprintf(stderr, "[uploader] Failed to open JSON: %s\n", jsonPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    nlohmann::json j;
    try {
        jsonFile >> j;
    } catch (const nlohmann::json::parse_error& e) {
        fprintf(stderr, "[uploader] JSON parse error: %s\n", e.what()); fflush(stderr);
        return ModelDesc();
    }

    ModelDesc model;
    auto& m = j["model"];
    model.architecture = m.value("architecture", "unknown");
    model.blockCount = m.value("block_count", 0);
    model.embeddingLength = m.value("embedding_length", 0);
    model.feedForwardLength = m.value("feed_forward_length", 0);
    model.headCount = m.value("attention.head_count", 0);
    model.headCountKv = m.value("attention.head_count_kv", 0);
    model.vocabSize = m.value("vocab_size", 0);
    model.contextLength = m.value("context_length", 0);

    fprintf(stderr, "[uploader] JSON parsed: %s | blocks=%u dim=%u heads=%u\n",
            model.architecture.c_str(), model.blockCount, model.embeddingLength, model.headCount);
    fflush(stderr);

    std::ifstream binFile(binPath, std::ios::binary | std::ios::ate);
    if (!binFile.is_open()) {
        fprintf(stderr, "[uploader] Failed to open binary: %s\n", binPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    std::streampos binEnd = binFile.tellg();
    if (binEnd <= 0) {
        fprintf(stderr, "[uploader] Binary file empty: %s\n", binPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    size_t binSize = static_cast<size_t>(binEnd);
    binFile.seekg(0, std::ios::beg);
    try {
        binData_.resize(binSize);
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "[uploader] OOM: cannot allocate %zu MB for binary\n",
                binSize / 1024 / 1024); fflush(stderr);
        return ModelDesc();
    }
    binFile.read(reinterpret_cast<char*>(binData_.data()), binSize);
    if (!binFile) {
        fprintf(stderr, "[uploader] Failed to read %zu bytes from binary\n", binSize); fflush(stderr);
        binData_.clear();
        return ModelDesc();
    }
    fprintf(stderr, "[uploader] Binary loaded: %zu MB (deferred upload)\n", binSize / 1024 / 1024);
    fflush(stderr);

    size_t tensorIdx = 0;
    size_t totalTensors = j["tensors"].size();
    for (auto& t : j["tensors"]) {
        TensorDesc desc;
        desc.name = t.value("name", "");
        desc.format = ggmlToQuantFormat(t.value("dtype_id", 0));
        if (desc.format == QuantFormat::UNSUPPORTED) {
            fprintf(stderr, "  [!] UNSUPPORTED format, skipping tensor\n"); fflush(stderr);
            tensorIdx++;
            continue;
        }
        desc.shape = t.value("shape", std::vector<uint32_t>{});
        desc.nDims = t.value("n_dims", 0);
        desc.sizeBytes = t.value<size_t>("size_bytes", 0);
        desc.binOffset = t.value<size_t>("bin_offset", 0);
        desc.binSize = t.value<size_t>("bin_size", 0);
        // Set correct block size and elements for the format
        if (isQuantized(desc.format)) {
            desc.blockSize = getBlockSize(desc.format);
            desc.blockElements = getBlockElements(desc.format);
        } else {
            desc.blockSize = (desc.format == QuantFormat::F16 || desc.format == QuantFormat::BF16) ? 2 : 4;
            desc.blockElements = 1;
        }
        desc.gpuAddress = 0;
        desc.buffer = VK_NULL_HANDLE;
        desc.memory = VK_NULL_HANDLE;

        if (desc.binOffset + desc.binSize > binSize) {
            fprintf(stderr, "  [!] OUT OF BOUNDS: offset %zu + size %zu > binSize %zu\n",
                    desc.binOffset, desc.binSize, binSize); fflush(stderr);
            tensorIdx++;
            continue;
        }

        model.tensors.push_back(desc);
        tensorIdx++;
    }

    fprintf(stderr, "[uploader] Metadata loaded: %zu tensors (none uploaded yet)\n", model.tensors.size());
    fflush(stderr);

    // Derive headDim from Q weight shape
    model.headDim = 0;
    for (auto& t : model.tensors) {
        if (t.name.find("blk.0.attn_q.weight") != std::string::npos && t.shape.size() == 2) {
            model.headDim = t.shape[1] / model.headCount;
            fprintf(stderr, "[uploader] Derived headDim=%u from Q weight shape [%u, %u]\n",
                    model.headDim, t.shape[0], t.shape[1]);
            break;
        }
    }
    if (model.headDim == 0) {
        model.headDim = model.embeddingLength / model.headCount;
        fprintf(stderr, "[uploader] WARNING: Could not derive headDim, using dim/headCount=%u\n", model.headDim);
    }

    return model;
}

static void loadTokenizerFromGGUF(Tokenizer& tokenizer, const notllama::GGUFMetadata& meta) {
    tokenizer.loadFromGGUF(meta.tokens, meta.merges, meta.bosId, meta.eosId, meta.padId, meta.unkId);
}

ModelDesc WeightUploader::loadFromGGUF(const std::string& ggufPath, Tokenizer* tokenizer) {
    notllama::GGUFLoader loader;
    if (!loader.load(ggufPath)) {
        fprintf(stderr, "[uploader] Failed to load GGUF: %s\n", ggufPath.c_str()); fflush(stderr);
        return ModelDesc();
    }

    const auto& meta = loader.metadata();
    ModelDesc model;
    model.architecture = meta.architecture;
    model.blockCount = meta.nLayers;
    model.embeddingLength = meta.nEmbd;
    model.feedForwardLength = meta.nFF;
    model.headCount = meta.nHeads;
    model.headCountKv = meta.nKVHeads;
    model.vocabSize = meta.nVocab;
    model.contextLength = meta.nCtx;

    const auto& tensors = loader.tensors();
    model.tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        TensorDesc desc;
        desc.name = t.name;
        desc.format = ggmlToQuantFormat(static_cast<int>(t.type));
        if (desc.format == QuantFormat::UNSUPPORTED) {
            fprintf(stderr, "  [!] UNSUPPORTED GGUF format for %s, skipping\n", t.name.c_str());
            fflush(stderr);
            continue;
        }
        desc.shape.reserve(t.dims.size());
        for (auto d : t.dims) desc.shape.push_back(static_cast<uint32_t>(d));
        desc.nDims = static_cast<uint32_t>(t.dims.size());
        desc.sizeBytes = t.nbytes;
        desc.binOffset = t.offset;
        desc.binSize = t.nbytes;
        auto qm = notllama::getQuantMeta(t.type);
        desc.blockSize = qm.blockSize;
        desc.blockElements = qm.blockSize;
        desc.gpuAddress = 0;
        desc.buffer = VK_NULL_HANDLE;
        desc.memory = VK_NULL_HANDLE;
        model.tensors.push_back(desc);
    }

    // Derive headDim from Q weight shape [inDim, outDim] where outDim = n_heads * headDim
    model.headDim = 0;
    for (auto& t : model.tensors) {
        if (t.name.find("blk.0.attn_q.weight") != std::string::npos && t.shape.size() == 2) {
            if (model.headCount > 0) {
                model.headDim = t.shape[1] / model.headCount;
                fprintf(stderr, "[uploader] Derived headDim=%u from Q weight shape [%u, %u]\n",
                        model.headDim, t.shape[0], t.shape[1]);
            }
            break;
        }
    }
    if (model.headDim == 0 && model.headCount > 0) {
        model.headDim = model.embeddingLength / model.headCount;
        fprintf(stderr, "[uploader] WARNING: Could not derive headDim, using dim/headCount=%u\n", model.headDim);
    }

    binData_ = loader.data();
    if (tokenizer) {
        loadTokenizerFromGGUF(*tokenizer, meta);
        fprintf(stderr, "[uploader] Tokenizer loaded: %zu tokens, %zu merges\n",
                meta.tokens.size(), meta.merges.size());
        fflush(stderr);
    }
    fprintf(stderr, "[uploader] GGUF loaded: %s | L=%u dim=%u heads=%u/%u | %zu tensors | %zu MB data (deferred upload)\n",
            model.architecture.c_str(), model.blockCount, model.embeddingLength,
            model.headCount, model.headCountKv, model.tensors.size(),
            binData_.size() / 1024 / 1024);
    fflush(stderr);
    return model;
}

bool WeightUploader::uploadTensor(TensorDesc& desc) {
    if (desc.binOffset + desc.binSize > binData_.size()) {
        fprintf(stderr, "  [uploadTensor] OOB: offset %zu + size %zu > binData %zu\n",
                desc.binOffset, desc.binSize, binData_.size()); fflush(stderr);
        return false;
    }

    // Create device-local GPU buffer
    VkDeviceAddress addr = 0;
    VkDeviceMemory bufMem = VK_NULL_HANDLE;
    VkBuffer buffer = createGpuBuffer(desc.binSize, &addr, &bufMem);
    if (buffer == VK_NULL_HANDLE) {
        fprintf(stderr, "  [uploadTensor] createGpuBuffer failed for %s\n", desc.name.c_str()); fflush(stderr);
        return false;
    }

    // Create staging buffer
    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = desc.binSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    VkResult r = vkCreateBuffer(device, &stagingInfo, nullptr, &staging);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [uploadTensor] staging buffer create failed: %d\n", r); fflush(stderr);
        freeTensor(desc);
        return false;
    }

    VkMemoryRequirements stagingMemReq;
    vkGetBufferMemoryRequirements(device, staging, &stagingMemReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t stagingMemType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((stagingMemReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            stagingMemType = i;
            break;
        }
    }
    if (stagingMemType == UINT32_MAX) {
        fprintf(stderr, "  [uploadTensor] No HOST_VISIBLE|HOST_COHERENT memory\n"); fflush(stderr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    VkMemoryAllocateInfo stagingAlloc = {};
    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAlloc.allocationSize = stagingMemReq.size;
    stagingAlloc.memoryTypeIndex = stagingMemType;

    VkDeviceMemory stagingMem;
    r = vkAllocateMemory(device, &stagingAlloc, nullptr, &stagingMem);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [uploadTensor] staging memory alloc failed: %d\n", r); fflush(stderr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    r = vkBindBufferMemory(device, staging, stagingMem, 0);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [uploadTensor] staging bind failed: %d\n", r); fflush(stderr);
        vkFreeMemory(device, stagingMem, nullptr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    void* mapped = nullptr;
    r = vkMapMemory(device, stagingMem, 0, desc.binSize, 0, &mapped);
    if (r != VK_SUCCESS || !mapped) {
        fprintf(stderr, "  [uploadTensor] staging map failed: %d\n", r); fflush(stderr);
        vkFreeMemory(device, stagingMem, nullptr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    std::memcpy(mapped, binData_.data() + desc.binOffset, desc.binSize);
    vkUnmapMemory(device, stagingMem);

    // Command pool + buffer for copy
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    VkCommandPool copyPool;
    r = vkCreateCommandPool(device, &poolInfo, nullptr, &copyPool);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [uploadTensor] command pool create failed: %d\n", r); fflush(stderr);
        vkFreeMemory(device, stagingMem, nullptr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = copyPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer copyCmd;
    r = vkAllocateCommandBuffers(device, &allocInfo, &copyCmd);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [uploadTensor] command buffer alloc failed: %d\n", r); fflush(stderr);
        vkDestroyCommandPool(device, copyPool, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
        vkDestroyBuffer(device, staging, nullptr);
        freeTensor(desc);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(copyCmd, &beginInfo);

    VkBufferCopy copyRegion = {};
    copyRegion.size = desc.binSize;
    vkCmdCopyBuffer(copyCmd, staging, buffer, 1, &copyRegion);
    vkEndCommandBuffer(copyCmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCmd;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);

    VkQueue submitQueue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &submitQueue);
    vkQueueSubmit(submitQueue, 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, copyPool, 1, &copyCmd);
    vkDestroyCommandPool(device, copyPool, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
    vkDestroyBuffer(device, staging, nullptr);

    desc.buffer = buffer;
    desc.gpuAddress = addr;
    desc.memory = bufMem;

    fprintf(stderr, "  [uploadTensor] %s uploaded @ 0x%llx (%zu bytes, format=%u)\n",
            desc.name.c_str(), (unsigned long long)addr, desc.binSize, (uint32_t)desc.format);
    fflush(stderr);

    return true;
}

bool WeightUploader::uploadLayer(ModelDesc& model, uint32_t layerIndex) {
    uint32_t count = 0;
    uint32_t totalTensors = 0;

    for (auto& t : model.tensors) {
        if (!tensorBelongsToLayer(t.name, layerIndex))
            continue;

        totalTensors++;
        // Skip if already uploaded (gpuAddress != 0)
        if (t.gpuAddress != 0) {
            count++;
            continue;
        }

        if (uploadTensor(t)) {
            count++;
        } else {
            fprintf(stderr, "  [uploadLayer] FAILED: %s (layer %u)\n", t.name.c_str(), layerIndex);
            fflush(stderr);
        }
    }

    if (totalTensors == 0) {
        fprintf(stderr, "[uploadLayer] No tensors found for layer %u\n", layerIndex);
        fflush(stderr);
        return false;
    }

    fprintf(stderr, "[uploadLayer] Layer %u: %u/%u tensors uploaded\n", layerIndex, count, totalTensors);
    fflush(stderr);

    return count > 0;
}

ModelDesc WeightUploader::load(const std::string& jsonPath, const std::string& binPath) {
    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        fprintf(stderr, "[uploader] Failed to open JSON: %s\n", jsonPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    nlohmann::json j;
    try {
        jsonFile >> j;
    } catch (const nlohmann::json::parse_error& e) {
        fprintf(stderr, "[uploader] JSON parse error: %s\n", e.what()); fflush(stderr);
        return ModelDesc();
    }

    ModelDesc model;
    auto& m = j["model"];
    model.architecture = m.value("architecture", "unknown");
    model.blockCount = m.value("block_count", 0);
    model.embeddingLength = m.value("embedding_length", 0);
    model.feedForwardLength = m.value("feed_forward_length", 0);
    model.headCount = m.value("attention.head_count", 0);
    model.headCountKv = m.value("attention.head_count_kv", 0);
    model.vocabSize = m.value("vocab_size", 0);
    model.contextLength = m.value("context_length", 0);

    fprintf(stderr, "[uploader] JSON parsed: %s | blocks=%u dim=%u heads=%u\n",
            model.architecture.c_str(), model.blockCount, model.embeddingLength, model.headCount);
    fflush(stderr);

    std::ifstream binFile(binPath, std::ios::binary | std::ios::ate);
    if (!binFile.is_open()) {
        fprintf(stderr, "[uploader] Failed to open binary: %s\n", binPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    std::streampos binEnd = binFile.tellg();
    if (binEnd <= 0) {
        fprintf(stderr, "[uploader] Binary file empty: %s\n", binPath.c_str()); fflush(stderr);
        return ModelDesc();
    }
    size_t binSize = static_cast<size_t>(binEnd);
    binFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> binData;
    try {
        binData.resize(binSize);
    } catch (const std::bad_alloc&) {
        fprintf(stderr, "[uploader] Out of memory: cannot allocate %zu MB for binary\n",
                binSize / 1024 / 1024); fflush(stderr);
        return ModelDesc();
    }
    binFile.read(reinterpret_cast<char*>(binData.data()), binSize);
    if (!binFile) {
        fprintf(stderr, "[uploader] Failed to read %zu bytes from binary\n", binSize); fflush(stderr);
        return ModelDesc();
    }
    fprintf(stderr, "[uploader] Binary loaded: %zu MB\n", binSize / 1024 / 1024);
    fflush(stderr);

    // Create staging buffer — one buffer, reused for all tensors
    // Staging buffer copies raw compressed bytes to GPU; dequant happens on GPU shaders
    VkDeviceSize maxTensorSize = 0;
    for (auto& t : j["tensors"]) {
        VkDeviceSize s = t.value("bin_size", 0);
        if (s > maxTensorSize) maxTensorSize = s;
    }
    fprintf(stderr, "[uploader] Max tensor: %zu MB\n", (size_t)(maxTensorSize / 1024 / 1024));
    fflush(stderr);

    if (maxTensorSize == 0) {
        fprintf(stderr, "[uploader] No tensors found in JSON\n"); fflush(stderr);
        return ModelDesc();
    }

    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = maxTensorSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    VkResult sr = vkCreateBuffer(device, &stagingInfo, nullptr, &staging);
    fprintf(stderr, "[uploader] staging create: %d\n", sr); fflush(stderr);
    if (sr != VK_SUCCESS) { fprintf(stderr, "[FATAL] staging buffer create failed\n"); return model; }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, staging, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
             (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        fprintf(stderr, "[uploader] No HOST_VISIBLE|HOST_COHERENT memory\n"); fflush(stderr);
        return model;
    }

    VkMemoryAllocateInfo stagingAlloc = {};
    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAlloc.allocationSize = memReq.size;
    stagingAlloc.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory stagingMem;
    VkResult sr2 = vkAllocateMemory(device, &stagingAlloc, nullptr, &stagingMem);
    fprintf(stderr, "[uploader] staging alloc: %d (%zu bytes)\n", sr2, (size_t)memReq.size); fflush(stderr);
    if (sr2 != VK_SUCCESS) return model;

    VkResult sr3 = vkBindBufferMemory(device, staging, stagingMem, 0);
    fprintf(stderr, "[uploader] staging bind: %d\n", sr3); fflush(stderr);
    if (sr3 != VK_SUCCESS) { vkFreeMemory(device, stagingMem, nullptr); vkDestroyBuffer(device, staging, nullptr); return model; }

    void* mapped = nullptr;
    VkResult sr4 = vkMapMemory(device, stagingMem, 0, maxTensorSize, 0, &mapped);
    fprintf(stderr, "[uploader] staging map: %d ptr=%p\n", sr4, mapped); fflush(stderr);
    if (sr4 != VK_SUCCESS || !mapped) return model;

    // Command buffer for all copies
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool uploadPool;
    VkResult poolR = vkCreateCommandPool(device, &poolInfo, nullptr, &uploadPool);
    if (poolR != VK_SUCCESS) { fprintf(stderr, "[FATAL] command pool create failed\n"); return model; }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = uploadPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer uploadCmd;
    VkResult allocR = vkAllocateCommandBuffers(device, &allocInfo, &uploadCmd);
    if (allocR != VK_SUCCESS) { fprintf(stderr, "[FATAL] command buffer alloc failed\n"); vkDestroyCommandPool(device, uploadPool, nullptr); return model; }

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult beginR = vkBeginCommandBuffer(uploadCmd, &beginInfo);
    if (beginR != VK_SUCCESS) { fprintf(stderr, "[FATAL] begin command buffer failed\n"); return model; }

    size_t tensorIdx = 0;
    size_t totalTensors = j["tensors"].size();
    for (auto& t : j["tensors"]) {
        TensorDesc desc;
        desc.name = t.value("name", "");
        desc.format = ggmlToQuantFormat(t.value("dtype_id", 0));
        if (desc.format == QuantFormat::UNSUPPORTED) {
            fprintf(stderr, "  [!] UNSUPPORTED format, skipping tensor\n"); fflush(stderr);
            tensorIdx++;
            continue;
        }
        desc.shape = t.value("shape", std::vector<uint32_t>{});
        desc.nDims = t.value("n_dims", 0);
        desc.sizeBytes = t.value<size_t>("size_bytes", 0);
        desc.binOffset = t.value<size_t>("bin_offset", 0);
        desc.binSize = t.value<size_t>("bin_size", 0);
        // Set correct block size and elements for the format
        if (isQuantized(desc.format)) {
            desc.blockSize = getBlockSize(desc.format);
            desc.blockElements = getBlockElements(desc.format);
        } else {
            desc.blockSize = (desc.format == QuantFormat::F16 || desc.format == QuantFormat::BF16) ? 2 : 4;
            desc.blockElements = 1;
        }

        fprintf(stderr, "\n[tensor %zu/%zu] %s binOffset=%zu binSize=%zu sizeBytes=%zu\n",
                tensorIdx + 1, totalTensors, desc.name.c_str(),
                desc.binOffset, desc.binSize, desc.sizeBytes);
        fflush(stderr);

        // Bounds check
        if (desc.binOffset + desc.binSize > binSize) {
            fprintf(stderr, "  [!] OUT OF BOUNDS: offset %zu + size %zu = %zu > binSize %zu\n",
                    desc.binOffset, desc.binSize, desc.binOffset + desc.binSize, binSize);
            fflush(stderr);
            tensorIdx++;
            continue;
        }

        // Upload raw bytes — GPU dequant shaders handle format conversion
        const uint8_t* uploadData = binData.data() + desc.binOffset;
        size_t uploadSize = desc.binSize;

        fprintf(stderr, "  [1] createGpuBuffer (%zu bytes)...\n", uploadSize); fflush(stderr);
        VkDeviceAddress addr = 0;
        VkDeviceMemory bufMem = VK_NULL_HANDLE;
        desc.buffer = createGpuBuffer(uploadSize, &addr, &bufMem);
        desc.gpuAddress = addr;
        desc.memory = bufMem;

        fprintf(stderr, "  [2] memcpy to staging...\n"); fflush(stderr);
        std::memcpy(mapped, uploadData, uploadSize);

        fprintf(stderr, "  [3] flush...\n"); fflush(stderr);
        VkMappedMemoryRange flushRange = {};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = stagingMem;
        flushRange.offset = 0;
        flushRange.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device, 1, &flushRange);

        fprintf(stderr, "  [4] vkCmdCopyBuffer...\n"); fflush(stderr);
        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = uploadSize;
        vkCmdCopyBuffer(uploadCmd, staging, desc.buffer, 1, &copyRegion);

        // Submit and WAIT after each tensor to prevent staging buffer overwrite.
        // The staging buffer is reused for all tensors, so we must ensure the GPU
        // finishes copying before we overwrite the staging data for the next tensor.
        vkEndCommandBuffer(uploadCmd);
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &uploadCmd;
        VkFence tensorFence;
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fenceInfo, nullptr, &tensorFence);
        VkQueue submitQueue;
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &submitQueue);
        vkQueueSubmit(submitQueue, 1, &submitInfo, tensorFence);
        vkWaitForFences(device, 1, &tensorFence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device, tensorFence, nullptr);

        fprintf(stderr, "  [5] push_back...\n"); fflush(stderr);
        model.tensors.push_back(desc);

        fprintf(stderr, "  [6] cout...\n"); fflush(stderr);
        std::cout << "Uploaded: " << desc.name << " @ 0x" << std::hex << addr
                  << " (" << std::dec << (desc.sizeBytes / 1024 / 1024) << " MB)\n";

        tensorIdx++;
        fprintf(stderr, "  [done]\n"); fflush(stderr);

        // Begin new command buffer for next tensor
        VkResult beginR = vkBeginCommandBuffer(uploadCmd, &beginInfo);
        if (beginR != VK_SUCCESS) {
            fprintf(stderr, "  [!] vkBeginCommandBuffer failed: %d — subsequent uploads may be lost!\n", beginR);
            fflush(stderr);
        }
    }

    // Command buffer from last iteration is already submitted; just clean up.
    fprintf(stderr, "\n[uploader] All %zu tensors uploaded.\n", model.tensors.size()); fflush(stderr);

    vkUnmapMemory(device, stagingMem);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    vkFreeCommandBuffers(device, uploadPool, 1, &uploadCmd);
    vkDestroyCommandPool(device, uploadPool, nullptr);

    fprintf(stderr, "[uploader] Model ready: %zu tensors\n", model.tensors.size()); fflush(stderr);

    // Derive headDim from Q weight shape [inDim, outDim] where outDim = n_heads * headDim
    model.headDim = 0;
    for (auto& t : model.tensors) {
        if (t.name.find("blk.0.attn_q.weight") != std::string::npos && t.shape.size() == 2) {
            // Weight is [inDim, outDim] — outDim = n_heads * headDim
            model.headDim = t.shape[1] / model.headCount;
            fprintf(stderr, "[uploader] Derived headDim=%u from Q weight shape [%u, %u]\n",
                    model.headDim, t.shape[0], t.shape[1]);
            break;
        }
    }
    if (model.headDim == 0) {
        model.headDim = model.embeddingLength / model.headCount;
        fprintf(stderr, "[uploader] WARNING: Could not derive headDim from weights, using dim/headCount=%u\n", model.headDim);
    }

    return model;
}

void WeightUploader::loadTokenizer(Tokenizer& tokenizer, const nlohmann::json& tokenizerJson) {
    auto vocab = tokenizerJson.value("vocab", std::vector<std::string>{});
    auto merges = tokenizerJson.value("merges", std::vector<std::string>{});
    auto special = tokenizerJson.value("special_tokens", nlohmann::json::object());

    uint32_t bos = special.value("bos", 1);
    uint32_t eos = special.value("eos", 2);
    uint32_t pad = special.value("pad", 0);
    uint32_t unk = special.value("unk", 3);

    tokenizer.loadFromGGUF(vocab, merges, bos, eos, pad, unk);
    fprintf(stderr, "[tokenizer] %zu tokens, %zu merges\n", vocab.size(), merges.size());
}

VkBuffer WeightUploader::createGpuBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem) {
    *outAddr = 0;
    *outMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    size_t alignedSize = (size + 3) & ~3;
    bufInfo.size = alignedSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &buffer);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [gpu] vkCreateBuffer failed: %d\n", r); fflush(stderr);
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == UINT32_MAX) {
        fprintf(stderr, "  [gpu] No DEVICE_LOCAL memory\n"); fflush(stderr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateFlagsInfo flagsInfo = {};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.pNext = &flagsInfo;

    VkDeviceMemory memory;
    r = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [gpu] vkAllocateMemory failed: %d (need %zu bytes)\n", r, (size_t)memReq.size); fflush(stderr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    r = vkBindBufferMemory(device, buffer, memory, 0);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [gpu] vkBindBufferMemory failed: %d\n", r); fflush(stderr);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    *outAddr = vkGetBufferDeviceAddress(device, &addrInfo);
    fprintf(stderr, "  [gpu] buf=%p mem=%p addr=0x%llx reqSize=%zu\n",
            (void*)buffer, (void*)memory, (unsigned long long)*outAddr, (size_t)memReq.size);
    fflush(stderr);
    *outMem = memory;

    return buffer;
}

void WeightUploader::freeTensor(const TensorDesc& desc) {
    if (desc.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, desc.buffer, nullptr);
    }
    if (desc.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, desc.memory, nullptr);
    }
}

void WeightUploader::freeAll(ModelDesc& model) {
    for (auto& t : model.tensors) {
        freeTensor(t);
        t.buffer = VK_NULL_HANDLE;
        t.memory = VK_NULL_HANDLE;
        t.gpuAddress = 0;
    }
}

void WeightUploader::OnAllLayersUploaded(ModelDesc& model) {
    switch (load_mode_) {
    case WeightLoadMode::VRAM:
        FreeSystemRAMCopy();
        fprintf(stderr, "[WeightUploader] VRAM mode: freed system RAM copy, all weights on GPU\n");
        break;
    case WeightLoadMode::MIRROR:
        fprintf(stderr, "[WeightUploader] MIRROR mode: keeping system RAM shadow copy (%.1f MB)\n",
                binData_.size() / (1024.0 * 1024.0));
        break;
    case WeightLoadMode::LAZY:
        fprintf(stderr, "[WeightUploader] LAZY mode: %zu tensors deferred, system RAM copy kept (%.1f MB)\n",
                model.tensors.size(), binData_.size() / (1024.0 * 1024.0));
        break;
    }
}

void WeightUploader::FreeSystemRAMCopy() {
    if (!binData_.empty()) {
        size_t freed = binData_.size();
        // Use clear + shrink_to_fit to actually release memory back to OS
        std::vector<uint8_t>().swap(binData_);
        fprintf(stderr, "[WeightUploader] Freed %.1f MB system RAM copy\n", freed / (1024.0 * 1024.0));
    }
}

} // namespace rdna4
