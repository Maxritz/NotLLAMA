#include "loaders/gguf.h"
#include "rdna4.hpp"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifdef _MSC_VER
#pragma warning(disable: 4996)
#endif

namespace notllama {

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

bool GGUFLoader::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
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

    uint64_t nMetadata;
    readU64(f, nMetadata);

    for (uint64_t i = 0; i < nMetadata; ++i) {
        std::string key = readString(f);
        uint32_t typeU32;
        readU32(f, typeU32);
        GGUFType type = static_cast<GGUFType>(typeU32);

        if (key == "general.architecture") {
            meta_.architecture = readString(f);
        } else if (key == "llama.embedding_length" || key == "qwen2.embedding_length") {
            readU32(f, meta_.nEmbd);
        } else if (key == "llama.attention.head_count" || key == "qwen2.attention.head_count") {
            readU32(f, meta_.nHeads);
        } else if (key == "llama.attention.head_count_kv" || key == "qwen2.attention.head_count_kv") {
            readU32(f, meta_.nKVHeads);
        } else if (key == "llama.block_count" || key == "qwen2.block_count") {
            readU32(f, meta_.nLayers);
        } else if (key == "llama.feed_forward_length" || key == "qwen2.feed_forward_length") {
            readU32(f, meta_.nFF);
        } else if (key == "llama.vocab_size" || key == "qwen2.vocab_size") {
            readU32(f, meta_.nVocab);
        } else if (key == "llama.context_length" || key == "qwen2.context_length") {
            readU32(f, meta_.nCtx);
        } else if (key == "llama.rope.freq_base" || key == "qwen2.rope.freq_base") {
            readF32(f, meta_.ropeFreqBase);
        } else {
            skipValue(f, type);
        }
    }

    if (meta_.nHeads > 0 && meta_.nEmbd > 0) meta_.headDim = meta_.nEmbd / meta_.nHeads;
    if (meta_.nCtx == 0) meta_.nCtx = 2048;
    if (meta_.ropeFreqBase == 0) meta_.ropeFreqBase = 10000.0f;

    uint64_t nTensors;
    readU64(f, nTensors);

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

    long dataStart = ftell(f);
    long aligned = (dataStart + 31) & ~31L;
    if (aligned > dataStart) fseek(f, aligned - dataStart, SEEK_CUR);
    dataStart = ftell(f);

    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    dataSize_ = fileSize - dataStart;

    for (auto& t : tensors_) {
        auto qm = getQuantMeta(t.type);
        uint64_t nElements = 1;
        for (auto d : t.dims) nElements *= d;
        t.nbytes = (nElements / qm.blockSize) * qm.bytesPerBlock;
        if (t.type == GGUFTensorType::F32) t.nbytes = nElements * 4;
        else if (t.type == GGUFTensorType::F16 || t.type == GGUFTensorType::BF16) t.nbytes = nElements * 2;
    }

    data_.resize(dataSize_);
    fseek(f, dataStart, SEEK_SET);
    fread(data_.data(), 1, dataSize_, f);
    fclose(f);

    std::cout << "GGUF: " << path << std::endl << std::flush;
    std::cout << "  Arch: " << meta_.architecture << std::endl;
    std::cout << "  L=" << meta_.nLayers << " H=" << meta_.nHeads << " KV=" << meta_.nKVHeads
              << " D=" << meta_.nEmbd << " FF=" << meta_.nFF << " V=" << meta_.nVocab << std::endl;
    std::cout << "  " << tensors_.size() << " tensors, " << (dataSize_ / (1024*1024)) << " MB data" << std::endl;

    return true;
}

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
