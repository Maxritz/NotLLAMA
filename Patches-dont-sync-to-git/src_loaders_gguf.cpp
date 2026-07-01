#include "loaders/gguf.h"
#include "rdna4.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable: 4996)
#include <io.h>
#endif

namespace notllama {

static constexpr uint32_t GGUF_MAGIC = 0x46554747;
static constexpr uint32_t GGUF_VERSION = 3;

GGUFQuantMeta getQuantMeta(GGUFTensorType type) {
 switch (type) {
 case GGUFTensorType::F32: return {type, 1, 4};
 case GGUFTensorType::F16: return {type, 1, 2};
 case GGUFTensorType::Q4_0: return {type, 32, 18};
 case GGUFTensorType::Q4_1: return {type, 32, 20};
 case GGUFTensorType::Q5_0: return {type, 32, 22};
 case GGUFTensorType::Q5_1: return {type, 32, 24};
 case GGUFTensorType::Q8_0: return {type, 32, 34};
 case GGUFTensorType::Q8_1: return {type, 32, 36};
 case GGUFTensorType::Q2_K: return {type, 256, 84};
 case GGUFTensorType::Q3_K: return {type, 256, 110};
 case GGUFTensorType::Q4_K: return {type, 256, 144};
 case GGUFTensorType::Q5_K: return {type, 256, 176};
 case GGUFTensorType::Q6_K: return {type, 256, 210};
 case GGUFTensorType::Q8_K: return {type, 256, 292};
 case GGUFTensorType::BF16: return {type, 1, 2};
 default: return {type, 1, 4};
 }
}

static bool readU32(FILE* f, uint32_t& v) { return fread(&v, 4, 1, f) == 1; }
static bool readU64(FILE* f, uint64_t& v) { return fread(&v, 8, 1, f) == 1; }
static bool readF32(FILE* f, float& v) { return fread(&v, 4, 1, f) == 1; }

static std::string readString(FILE* f) {
 uint64_t len;
 if (!readU64(f, len)) return "";
 std::string s(len, '\0');
 if (len > 0 && fread(s.data(), 1, len, f) != len) return "";
 return s;
}

static bool skipValue(FILE* f, GGUFType type);

static std::vector<std::string> readStringArray(FILE* f) {
 std::vector<std::string> out;
 uint32_t arrType;
 uint64_t arrLen;
 if (!readU32(f, arrType) || !readU64(f, arrLen)) return out;
 if (static_cast<GGUFType>(arrType) != GGUFType::STRING) {
 for (uint64_t i = 0; i < arrLen; ++i) skipValue(f, static_cast<GGUFType>(arrType));
 return out;
 }
 out.reserve(arrLen);
 for (uint64_t i = 0; i < arrLen; ++i) out.push_back(readString(f));
 return out;
}

static bool skipValue(FILE* f, GGUFType type) {
 switch (type) {
 case GGUFType::UINT8: case GGUFType::INT8: case GGUFType::BOOL: {
 uint8_t v; return fread(&v, 1, 1, f) == 1;
 }
 case GGUFType::UINT16: case GGUFType::INT16: {
 uint16_t v; return fread(&v, 2, 1, f) == 1;
 }
 case GGUFType::UINT32: case GGUFType::INT32: case GGUFType::FLOAT32: {
 uint32_t v; return fread(&v, 4, 1, f) == 1;
 }
 case GGUFType::UINT64: case GGUFType::INT64: case GGUFType::FLOAT64: {
 uint64_t v; return fread(&v, 8, 1, f) == 1;
 }
 case GGUFType::STRING: {
 readString(f); return true;
 }
 case GGUFType::ARRAY: {
 uint32_t arrType; readU32(f, arrType);
 uint64_t arrLen; readU64(f, arrLen);
 for (uint64_t i = 0; i < arrLen; ++i) skipValue(f, static_cast<GGUFType>(arrType));
 return true;
 }
 default: return false;
 }
}

// CRITICAL FIX: readIntByType now handles ALL integer/float types.
// Previously only handled 5 types; UINT16/INT16/UINT8/INT8 hit default
// which skipped the value and returned 0, misaligning the file pointer.
static uint64_t readIntByType(FILE* f, GGUFType type) {
 uint64_t val = 0;
 switch (type) {
 case GGUFType::UINT8:  { uint8_t  v; fread(&v, 1, 1, f); val = v; break; }
 case GGUFType::INT8:   { int8_t   v; fread(&v, 1, 1, f); val = (uint64_t)(int64_t)v; break; }
 case GGUFType::UINT16: { uint16_t v; fread(&v, 2, 1, f); val = v; break; }
 case GGUFType::INT16:  { int16_t  v; fread(&v, 2, 1, f); val = (uint64_t)(int64_t)v; break; }
 case GGUFType::UINT32: { uint32_t v; fread(&v, 4, 1, f); val = v; break; }
 case GGUFType::INT32:  { int32_t  v; fread(&v, 4, 1, f); val = (uint64_t)(int64_t)v; break; }
 case GGUFType::UINT64: { uint64_t v; fread(&v, 8, 1, f); val = v; break; }
 case GGUFType::INT64:  { int64_t  v; fread(&v, 8, 1, f); val = (uint64_t)v; break; }
 case GGUFType::FLOAT32:{ float    v; fread(&v, 4, 1, f); val = (uint64_t)(int64_t)v; break; }
 case GGUFType::FLOAT64:{ double   v; fread(&v, 8, 1, f); val = (uint64_t)(int64_t)v; break; }
 default:
 fprintf(stderr, "[GGUF WARNING] readIntByType: unhandled type %d, skipping\n", (int)type);
 skipValue(f, type);
 break;
 }
 return val;
}

bool GGUFLoader::load(const std::string& path) {
 FILE* f = fopen(path.c_str(), "rb");
 fprintf(stderr, "[GGUF] load(%s) f=%p\n", path.c_str(), (void*)f); fflush(stderr);
 if (!f) {
 std::cerr << "GGUF: cannot open " << path << std::endl;
 return false;
 }

 uint32_t magic, version;
 if (!readU32(f, magic) || magic != GGUF_MAGIC) {
 std::cerr << "GGUF: bad magic" << std::endl;
 fclose(f);
 return false;
 }
 if (!readU32(f, version) || version != GGUF_VERSION) {
 std::cerr << "GGUF: unsupported version " << version << std::endl;
 fclose(f);
 return false;
 }
 fprintf(stderr, "[GGUF] version=%u ok\n", version); fflush(stderr);

 uint64_t nTensors_header, nMetadata;
 readU64(f, nTensors_header);
 readU64(f, nMetadata);
 fprintf(stderr, "[GGUF] nTensors_header=%llu nMetadata=%llu\n",
 (unsigned long long)nTensors_header, (unsigned long long)nMetadata); fflush(stderr);

 for (uint64_t i = 0; i < nMetadata; ++i) {
 std::string key = readString(f);
 uint32_t typeU32;
 readU32(f, typeU32);
 GGUFType type = static_cast<GGUFType>(typeU32);

 fprintf(stderr, "[GGUF META %3llu] key='%s' type=%u ",
 (unsigned long long)i, key.c_str(), typeU32);
 fflush(stderr);

 size_t dot = key.find('.');
 std::string_view suffix = (dot != std::string::npos)
 ? std::string_view(key).substr(dot + 1) : std::string_view(key);

 if (key == "general.architecture") {
 meta_.architecture = readString(f);
 fprintf(stderr, "val='%s'\n", meta_.architecture.c_str());
 } else if (suffix == "embedding_length" || suffix == "hidden_size") {
 meta_.nEmbd = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nEmbd);
 } else if (suffix == "attention.head_count") {
 meta_.nHeads = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nHeads);
 } else if (suffix == "attention.head_count_kv") {
 meta_.nKVHeads = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nKVHeads);
 } else if (suffix == "block_count") {
 meta_.nLayers = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nLayers);
 } else if (suffix == "feed_forward_length" || suffix == "intermediate_size") {
 meta_.nFF = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nFF);
 } else if (suffix == "vocab_size") {
 meta_.nVocab = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nVocab);
 } else if (suffix == "context_length" || suffix == "max_position_embeddings") {
 meta_.nCtx = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", meta_.nCtx);
 } else if (suffix == "rope.freq_base") {
 if (type == GGUFType::FLOAT32) { readF32(f, meta_.ropeFreqBase); fprintf(stderr, "val=%f\n", meta_.ropeFreqBase); }
 else { skipValue(f, type); fprintf(stderr, "skipped (type=%u)\n", typeU32); }
 } else if (suffix == "rope.scale_linear" || suffix == "rope.freq_scale") {
 if (type == GGUFType::FLOAT32) { readF32(f, meta_.ropeScaling); fprintf(stderr, "val=%f\n", meta_.ropeScaling); }
 else { skipValue(f, type); fprintf(stderr, "skipped (type=%u)\n", typeU32); }
 }
 // Tokenizer: multiple key variants for different architectures/converters
 else if (key == "tokenizer.ggml.tokens" || key == "tokenizer.tokens" || key == "tokenizer.chars") {
 meta_.tokens = readStringArray(f);
 fprintf(stderr, "val=[%zu strings]\n", meta_.tokens.size());
 } else if (key == "tokenizer.ggml.merges" || key == "tokenizer.merges") {
 meta_.merges = readStringArray(f);
 fprintf(stderr, "val=[%zu strings]\n", meta_.merges.size());
 } else if (key == "tokenizer.ggml.model") {
 std::string model = readString(f);
 fprintf(stderr, "val='%s'\n", model.c_str());
 }
 // Tokenizer IDs: CRITICAL FIX — use readIntByType instead of hardcoded readU32
 else if (key == "tokenizer.ggml.bos_token_id" || key == "tokenizer.bos_token_id") {
 meta_.bosId = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u (type=%u)\n", meta_.bosId, typeU32);
 } else if (key == "tokenizer.ggml.eos_token_id" || key == "tokenizer.eos_token_id") {
 meta_.eosId = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u (type=%u)\n", meta_.eosId, typeU32);
 } else if (key == "tokenizer.ggml.padding_token_id" || key == "tokenizer.padding_token_id") {
 meta_.padId = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u (type=%u)\n", meta_.padId, typeU32);
 } else if (key == "tokenizer.ggml.unknown_token_id" || key == "tokenizer.unknown_token_id") {
 meta_.unkId = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u (type=%u)\n", meta_.unkId, typeU32);
 }
 // Gemma-specific keys
 else if (suffix == "attention.layer_norm_rms_epsilon") {
 skipValue(f, type); fprintf(stderr, "skipped (rms_eps)\n");
 } else if (suffix == "attention.key_length" || suffix == "attention.value_length") {
 uint32_t v = (uint32_t)readIntByType(f, type);
 fprintf(stderr, "val=%u\n", v);
 } else {
 skipValue(f, type);
 fprintf(stderr, "skipped\n");
 }
 fflush(stderr);
 }

 if (meta_.nHeads > 0 && meta_.nEmbd > 0) meta_.headDim = meta_.nEmbd / meta_.nHeads;
 if (meta_.nCtx == 0) meta_.nCtx = 2048;
 if (meta_.ropeFreqBase == 0) meta_.ropeFreqBase = 10000.0f;

 uint64_t nTensors = nTensors_header;

 tensors_.resize(nTensors);
 for (uint64_t i = 0; i < nTensors; ++i) {
 auto& t = tensors_[i];
 t.name = readString(f);
 uint32_t nDims;
 readU32(f, nDims);
 t.dims.resize(nDims);
 for (uint32_t d = 0; d < nDims; ++d) readU64(f, t.dims[d]);
 uint32_t typeU32;
 readU32(f, typeU32);
 t.type = static_cast<GGUFTensorType>(typeU32);
 readU64(f, t.offset);
 tensorMap_[t.name] = static_cast<int>(i);
 }

 int64_t dataStart = _ftelli64(f);
 int64_t aligned = (dataStart + 31) & ~31LL;
 if (aligned > dataStart) _fseeki64(f, aligned - dataStart, SEEK_CUR);
 dataStart = _ftelli64(f);

 _fseeki64(f, 0, SEEK_END);
 int64_t fileSize = _ftelli64(f);
 dataSize_ = (size_t)(fileSize - dataStart);
 fprintf(stderr, "[GGUF] dataStart=%lld fileSize=%lld dataSize=%zu\n", (long long)dataStart, (long long)fileSize, dataSize_); fflush(stderr);

 for (auto& t : tensors_) {
 auto qm = getQuantMeta(t.type);
 uint64_t nElements = 1;
 for (auto d : t.dims) nElements *= d;
 t.nbytes = (nElements / qm.blockSize) * qm.bytesPerBlock;
 if (t.type == GGUFTensorType::F32) t.nbytes = nElements * 4;
 else if (t.type == GGUFTensorType::F16 || t.type == GGUFTensorType::BF16) t.nbytes = nElements * 2;
 }

 data_.resize(dataSize_);
 _fseeki64(f, dataStart, SEEK_SET);
 fread(data_.data(), 1, dataSize_, f);
 fclose(f);

 std::cout << "GGUF: " << path << std::endl << std::flush;
 std::cout << " Arch: " << meta_.architecture << std::endl;
 std::cout << " L=" << meta_.nLayers << " H=" << meta_.nHeads << " KV=" << meta_.nKVHeads
 << " D=" << meta_.nEmbd << " FF=" << meta_.nFF << " V=" << meta_.nVocab << std::endl;
 std::cout << " " << tensors_.size() << " tensors, " << (dataSize_ / (1024*1024)) << " MB data" << std::endl;
 std::cout << " Tokens=" << meta_.tokens.size() << " Merges=" << meta_.merges.size() << std::endl;

 return true;
}

#if 0 // Legacy upload path
void GGUFLoader::uploadToGPU(VkDevice dev, VkPhysicalDevice physDev,
 VkCommandPool pool, VkQueue queue) {
 // ... existing code ...
}

rdna4::GpuBuffer GGUFLoader::getTensorBuffer(const std::string& name) const {
 auto it = tensorMap_.find(name);
 if (it == tensorMap_.end()) return {};
 return gpuBuffers_[it->second];
}
#endif // Legacy upload path

int GGUFLoader::tensorIndex(const std::string& name) const {
 auto it = tensorMap_.find(name);
 return (it != tensorMap_.end()) ? it->second : -1;
}

void GGUFLoader::printInfo() const {
 std::cout << "GGUF Tensors:" << std::endl;
 for (size_t i = 0; i < tensors_.size(); ++i) {
 auto& t = tensors_[i];
 std::cout << " [" << i << "] " << t.name << " (";
 for (size_t d = 0; d < t.dims.size(); ++d) {
 if (d > 0) std::cout << "x";
 std::cout << t.dims[d];
 }
 std::cout << ") type=" << static_cast<int>(t.type) << " size=" << t.nbytes << std::endl;
 }
}

} // namespace notllama
