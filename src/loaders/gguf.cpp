#include "loaders/gguf.h"
#include "rdna4.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <string_view>

#ifdef _MSC_VER
#pragma warning(disable: 4996)
#include <io.h>
#endif

namespace notllama {

GGUFLoader::~GGUFLoader() {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile((LPCVOID)mappedBase_);
        data_ = nullptr;
        mappedBase_ = nullptr;
    }
    if (mapHandle_) {
        CloseHandle((HANDLE)mapHandle_);
        mapHandle_ = nullptr;
    }
    if (fileHandle_) {
        CloseHandle((HANDLE)fileHandle_);
        fileHandle_ = nullptr;
    }
#endif
    // POSIX mmap path is left as a future exercise; for now Linux builds
    // still use the std::vector fallback below if mmap is not enabled.
}

static constexpr uint32_t GGUF_MAGIC = 0x46554747;
static constexpr uint32_t GGUF_VERSION = 3;

GGUFQuantMeta getQuantMeta(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::F32:     return {type, 1, 4};
        case GGUFTensorType::F16:     return {type, 1, 2};
        case GGUFTensorType::Q4_0:    return {type, 32, 18};
        case GGUFTensorType::Q4_1:    return {type, 32, 20};
        case GGUFTensorType::Q5_0:    return {type, 32, 22};
        case GGUFTensorType::Q5_1:    return {type, 32, 24};
        case GGUFTensorType::Q8_0:    return {type, 32, 34};
        case GGUFTensorType::Q8_1:    return {type, 32, 36};
        case GGUFTensorType::Q2_K:    return {type, 256, 84};
        case GGUFTensorType::Q3_K:    return {type, 256, 110};
        case GGUFTensorType::Q4_K:    return {type, 256, 144};
        case GGUFTensorType::Q5_K:    return {type, 256, 176};
        case GGUFTensorType::Q6_K:    return {type, 256, 210};
        case GGUFTensorType::Q8_K:    return {type, 256, 292};
        case GGUFTensorType::BF16:    return {type, 1, 2};
        default:                      return {type, 1, 4};
    }
}

static bool readU32(FILE* f, uint32_t& v) { return fread(&v, 4, 1, f) == 1; }
static bool readU64(FILE* f, uint64_t& v) { return fread(&v, 8, 1, f) == 1; }
static bool readF32(FILE* f, float& v)    { return fread(&v, 4, 1, f) == 1; }

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

// Read an integer value whose GGUF type is known (handles uint32, uint64, int32, int64,
// and single-element arrays of those types — e.g. gemma4 stores head_count_kv as an array)
static uint64_t readIntByType(FILE* f, GGUFType type) {
    uint64_t val = 0;
    switch (type) {
        case GGUFType::UINT32: { uint32_t v; fread(&v, 4, 1, f); val = v; break; }
        case GGUFType::INT32:  { int32_t  v; fread(&v, 4, 1, f); val = v; break; }
        case GGUFType::UINT64: { uint64_t v; fread(&v, 8, 1, f); val = v; break; }
        case GGUFType::INT64:  { int64_t  v; fread(&v, 8, 1, f); val = v; break; }
        case GGUFType::FLOAT32: { float v; fread(&v, 4, 1, f); val = (uint64_t)v; break; }
        case GGUFType::ARRAY: {
            uint32_t arrType; readU32(f, arrType);
            uint64_t arrLen; readU64(f, arrLen);
            if (arrLen == 1) {
                // Recurse to read the single scalar element
                val = readIntByType(f, static_cast<GGUFType>(arrType));
            } else {
                // Multi-element array: not a scalar, skip all elements
                for (uint64_t i = 0; i < arrLen; ++i)
                    skipValue(f, static_cast<GGUFType>(arrType));
            }
            break;
        }
        default: skipValue(f, type); break;
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
    readU64(f, nTensors_header);  // GGUF v3: tensor_count before metadata_kv_count
    readU64(f, nMetadata);
    fprintf(stderr, "[GGUF] nTensors_header=%llu nMetadata=%llu\n",
            (unsigned long long)nTensors_header, (unsigned long long)nMetadata); fflush(stderr);

    for (uint64_t i = 0; i < nMetadata; ++i) {
        std::string key = readString(f);
        uint32_t typeU32;
        readU32(f, typeU32);
        GGUFType type = static_cast<GGUFType>(typeU32);

        // Print EVERY key before processing
        fprintf(stderr, "[GGUF META %3llu] key='%s' type=%u ",
                (unsigned long long)i, key.c_str(), typeU32);
        fflush(stderr);

        // Extract suffix after first dot for architecture-agnostic matching
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
            else { skipValue(f, type); fprintf(stderr, "skipped\n"); }
        } else if (suffix == "rope.scale_linear" || suffix == "rope.freq_scale") {
            if (type == GGUFType::FLOAT32) { readF32(f, meta_.ropeScaling); fprintf(stderr, "val=%f\n", meta_.ropeScaling); }
            else { skipValue(f, type); fprintf(stderr, "skipped\n"); }
        } else if (key == "tokenizer.ggml.tokens") {
            meta_.tokens = readStringArray(f);
            fprintf(stderr, "val=[%zu strings]\n", meta_.tokens.size());
        } else if (key == "tokenizer.ggml.merges") {
            meta_.merges = readStringArray(f);
            fprintf(stderr, "val=[%zu strings]\n", meta_.merges.size());
        } else if (key == "tokenizer.ggml.bos_token_id") {
            meta_.bosId = (uint32_t)readIntByType(f, type);
            fprintf(stderr, "val=%u\n", meta_.bosId);
        } else if (key == "tokenizer.ggml.eos_token_id") {
            meta_.eosId = (uint32_t)readIntByType(f, type);
            fprintf(stderr, "val=%u\n", meta_.eosId);
        } else if (key == "tokenizer.ggml.padding_token_id") {
            meta_.padId = (uint32_t)readIntByType(f, type);
            fprintf(stderr, "val=%u\n", meta_.padId);
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            meta_.unkId = (uint32_t)readIntByType(f, type);
            fprintf(stderr, "val=%u\n", meta_.unkId);
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
        // Round up to full blocks to avoid truncation
        uint64_t nBlocks = (nElements + qm.blockSize - 1) / qm.blockSize;
        t.nbytes = nBlocks * qm.bytesPerBlock;
        if (t.type == GGUFTensorType::F32) t.nbytes = nElements * 4;
        else if (t.type == GGUFTensorType::F16 || t.type == GGUFTensorType::BF16) t.nbytes = nElements * 2;
    }

#ifdef _WIN32
    // Memory-map the file so models >4 GB do not need a single contiguous vector allocation.
    fclose(f);
    f = nullptr;

    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "GGUF: cannot reopen for mmap " << path << std::endl;
        return false;
    }
    LARGE_INTEGER fsLi;
    fsLi.QuadPart = fileSize;
    HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY,
                                    fsLi.HighPart, fsLi.LowPart, NULL);
    if (!hMap) {
        std::cerr << "GGUF: CreateFileMapping failed (size " << fileSize << ")" << std::endl;
        CloseHandle(hFile);
        return false;
    }
    mappedBase_ = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!mappedBase_) {
        std::cerr << "GGUF: MapViewOfFile failed" << std::endl;
        CloseHandle(hMap);
        CloseHandle(hFile);
        return false;
    }
    fileHandle_ = hFile;
    mapHandle_ = hMap;
    data_ = mappedBase_ + dataStart;
    mappedSize_ = (size_t)fileSize;
#else
    // Non-Windows fallback: read tensor data into a vector.
    fallback_data_.resize(dataSize_);
    _fseeki64(f, dataStart, SEEK_SET);
    fread(fallback_data_.data(), 1, dataSize_, f);
    data_ = fallback_data_.data();
    fclose(f);
    f = nullptr;
#endif

    std::cout << "GGUF: " << path << std::endl << std::flush;
    std::cout << "  Arch: " << meta_.architecture << std::endl;
    std::cout << "  L=" << meta_.nLayers << " H=" << meta_.nHeads << " KV=" << meta_.nKVHeads
              << " D=" << meta_.nEmbd << " FF=" << meta_.nFF << " V=" << meta_.nVocab << std::endl;
    std::cout << "  " << tensors_.size() << " tensors, " << (dataSize_ / (1024*1024)) << " MB data" << std::endl;

    return true;
}

#if 0  // Legacy upload path — not used by modular engine (uses WeightUploader::uploadTensor)
void GGUFLoader::uploadToGPU(VkDevice dev, VkPhysicalDevice physDev,
                             VkCommandPool pool, VkQueue queue) {
    gpuBuffers_.resize(tensors_.size());

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    auto findHostMemType = [&](uint32_t typeBits) -> uint32_t {
        for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
            if ((typeBits & (1 << m)) &&
                (memProps.memoryTypes[m].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                return m;
            }
        }
        return 0;
    };

    auto findDeviceMemType = [&](uint32_t typeBits) -> uint32_t {
        for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
            if ((typeBits & (1 << m)) &&
                (memProps.memoryTypes[m].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                return m;
            }
        }
        return 0;
    };

    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        auto qm = getQuantMeta(t.type);

        rdna4::QuantFormat fmt = rdna4::QuantFormat::F32;
        switch (t.type) {
            case GGUFTensorType::F32:  fmt = rdna4::QuantFormat::F32; break;
            case GGUFTensorType::F16:  fmt = rdna4::QuantFormat::F16; break;
            case GGUFTensorType::Q8_0: fmt = rdna4::QuantFormat::Q8_0; break;
            case GGUFTensorType::Q6_K: fmt = rdna4::QuantFormat::Q6_K; break;
            case GGUFTensorType::Q4_K: fmt = rdna4::QuantFormat::Q4_K; break;
            case GGUFTensorType::Q4_0: fmt = rdna4::QuantFormat::Q4_0; break;
            case GGUFTensorType::Q5_0: fmt = rdna4::QuantFormat::Q5_0; break;
            case GGUFTensorType::Q3_K: fmt = rdna4::QuantFormat::Q3_K; break;
            case GGUFTensorType::Q2_K: fmt = rdna4::QuantFormat::Q2_K; break;
            default: break;
        }


        // Staging buffer
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = t.nbytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer staging;
        vkCreateBuffer(dev, &bci, nullptr, &staging);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(dev, staging, &memReqs);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = memReqs.size;
        mai.memoryTypeIndex = findHostMemType(memReqs.memoryTypeBits);

        VkDeviceMemory stagingMem;
        vkAllocateMemory(dev, &mai, nullptr, &stagingMem);
        vkBindBufferMemory(dev, staging, stagingMem, 0);

        void* mapped;
        vkMapMemory(dev, stagingMem, 0, t.nbytes, 0, &mapped);
        memcpy(mapped, data_.data() + t.offset, t.nbytes);
        vkUnmapMemory(dev, stagingMem);

        // Device buffer
        VkBufferCreateInfo dbci{};
        dbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        dbci.size = t.nbytes;
        dbci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer deviceBuf;
        vkCreateBuffer(dev, &dbci, nullptr, &deviceBuf);

        VkMemoryRequirements devReqs;
        vkGetBufferMemoryRequirements(dev, deviceBuf, &devReqs);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo devAlloc{};
        devAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        devAlloc.allocationSize = devReqs.size;
        devAlloc.memoryTypeIndex = findDeviceMemType(devReqs.memoryTypeBits);
        devAlloc.pNext = &flagsInfo;

        VkDeviceMemory devMem;
        vkAllocateMemory(dev, &devAlloc, nullptr, &devMem);
        vkBindBufferMemory(dev, deviceBuf, devMem, 0);

        // Copy staging -> device
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(dev, &cbai, &cmd);

        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbbi);
        VkBufferCopy copyRegion{};
        copyRegion.size = t.nbytes;
        vkCmdCopyBuffer(cmd, staging, deviceBuf, 1, &copyRegion);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(dev, pool, 1, &cmd);

        VkBufferDeviceAddressInfo bdaInfo{};
        bdaInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bdaInfo.buffer = deviceBuf;
        VkDeviceAddress addr = vkGetBufferDeviceAddress(dev, &bdaInfo);

        gpuBuffers_[i].buffer = deviceBuf;
        gpuBuffers_[i].mem.memory = devMem;
        gpuBuffers_[i].mem.size = devReqs.size;
        gpuBuffers_[i].size = t.nbytes;
        gpuBuffers_[i].desc.deviceAddress = addr;
        gpuBuffers_[i].desc.range = t.nbytes;
        gpuBuffers_[i].desc.format = fmt;
        gpuBuffers_[i].desc.stride = qm.blockSize;

        vkDestroyBuffer(dev, staging, nullptr);
        vkFreeMemory(dev, stagingMem, nullptr);
    }

    std::cout << "  Uploaded " << gpuBuffers_.size() << " tensors to GPU" << std::endl << std::flush;
}

rdna4::GpuBuffer GGUFLoader::getTensorBuffer(const std::string& name) const {
    auto it = tensorMap_.find(name);
    if (it == tensorMap_.end()) return {};
    return gpuBuffers_[it->second];
}
#endif  // Legacy upload path

int GGUFLoader::tensorIndex(const std::string& name) const {
    auto it = tensorMap_.find(name);
    return (it != tensorMap_.end()) ? it->second : -1;
}

void GGUFLoader::printInfo() const {
    std::cout << "GGUF Tensors:" << std::endl;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        std::cout << "  [" << i << "] " << t.name << " (";
        for (size_t d = 0; d < t.dims.size(); ++d) {
            if (d > 0) std::cout << "x";
            std::cout << t.dims[d];
        }
        std::cout << ") type=" << static_cast<int>(t.type) << " size=" << t.nbytes << std::endl;
    }
}

} // namespace notllama
