#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"
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
        case 0: return QuantFormat::F32;
        case 1: return QuantFormat::F16;
        case 2: return QuantFormat::Q4_0;
        case 3: return QuantFormat::Q4_1;
        case 6: return QuantFormat::Q5_0;
        case 7: return QuantFormat::Q5_1;
        case 8: return QuantFormat::Q8_0;
        case 10: return QuantFormat::Q2_K;
        case 11: return QuantFormat::Q3_K;
        case 12: return QuantFormat::Q4_K;
        case 13: return QuantFormat::Q5_K;
        case 14: return QuantFormat::Q6_K;
        case 15: return QuantFormat::Q8_K;
        default: return QuantFormat::F32;
    }
}

static bool isQuantized(QuantFormat fmt) {
    return fmt != QuantFormat::F32 && fmt != QuantFormat::F16;
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
        // Layout: d(fp16)@0, dmin(fp16)@2, scales_int8(12)@4, qs(128)@16
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 144;
            if (bs + 144 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            float dmin = readF16Cpu(srcData, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint8_t sc = srcData[bs + 4 + subBlock];
            uint32_t qsOffset = bs + 16 + (eleInBlock / 2);
            if (qsOffset >= srcSize) continue;
            uint8_t bval = srcData[qsOffset];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            out[i] = d * (float)sc * ((float)nibble - 8.0f) + dmin;
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
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 290;
            if (bs + 290 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            uint32_t subBlock = eleInBlock / 32;
            int sc = (int8_t)srcData[bs + 256 + subBlock];
            uint32_t qIdx = bs + 2 + eleInBlock;
            if (qIdx >= srcSize) continue;
            int q = (int8_t)srcData[qIdx];
            out[i] = d * (float)sc * (float)q;
        }
        return true;
    }
    if (format == QuantFormat::Q3_K) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 256;
            uint32_t eleInBlock = i % 256;
            uint32_t bs = blockIdx * 112;
            if (bs + 112 > srcSize) continue;
            float d = readF16Cpu(srcData, bs);
            float dmin = readF16Cpu(srcData, bs + 2);
            uint32_t subBlock = eleInBlock / 32;
            uint32_t subEle = eleInBlock % 32;
            uint8_t sc = srcData[bs + 4 + subBlock];
            uint32_t byteIdx = bs + 12 + subBlock * 24 + subEle / 2;
            if (byteIdx >= srcSize) continue;
            uint8_t bval = srcData[byteIdx];
            uint32_t nibble = (subEle % 2 == 0) ? (bval & 0x0F) : ((bval >> 4) & 0x0F);
            out[i] = d * (float)(int8_t)sc * (float)(int)(nibble - 8) - dmin;
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
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 22;
            if (bs + 22 > srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            uint32_t byteIdx = bs + 2 + (eleInBlock * 5 / 8);
            if (byteIdx >= srcSize) continue;
            uint8_t bval = srcData[byteIdx];
            uint32_t shift = (eleInBlock * 5) % 8;
            uint32_t q5 = (bval >> shift) & 0x1F;
            if (q5 > 15) q5 |= 0xE0;
            out[i] = delta * ((float)(int)q5 - 16.0f);
        }
        return true;
    }
    if (format == QuantFormat::Q5_1) {
        for (uint32_t i = 0; i < nElements; ++i) {
            uint32_t blockIdx = i / 32;
            uint32_t eleInBlock = i % 32;
            uint32_t bs = blockIdx * 24;
            if (bs + 24 > srcSize) continue;
            float delta = readF16Cpu(srcData, bs);
            float deltaMin = readF16Cpu(srcData, bs + 2);
            uint32_t byteIdx = bs + 4 + (eleInBlock * 5 / 8);
            if (byteIdx >= srcSize) continue;
            uint8_t bval = srcData[byteIdx];
            uint32_t shift = (eleInBlock * 5) % 8;
            uint32_t q5 = (bval >> shift) & 0x1F;
            out[i] = delta * (float)q5 + deltaMin;
        }
        return true;
    }
    // Unknown format — output zeros
    fprintf(stderr, "[cpu-dequant] WARNING: unsupported format %u, outputting zeros\n", (uint32_t)format);
    return false;
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
    // Size for worst-case: quantized tensor dequantized to F32 (up to 4× raw size)
    VkDeviceSize maxTensorSize = 0;
    for (auto& t : j["tensors"]) {
        VkDeviceSize s = t.value("bin_size", 0);
        QuantFormat fmt = ggmlToQuantFormat(t.value("dtype_id", 0));
        if (isQuantized(fmt)) {
            // F32 output: nElements * 4. nElements = size_bytes / bytes_per_element_approx
            // Safe upper bound: bin_size * 4 (Q4_0 is 18 bytes per 32 elements = 0.56 B/elem -> F32 = 4 B/elem -> 7.1×)
            s = s * 8;
        }
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
    vkCreateCommandPool(device, &poolInfo, nullptr, &uploadPool);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = uploadPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer uploadCmd;
    vkAllocateCommandBuffers(device, &allocInfo, &uploadCmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(uploadCmd, &beginInfo);

    size_t tensorIdx = 0;
    size_t totalTensors = j["tensors"].size();
    for (auto& t : j["tensors"]) {
        TensorDesc desc;
        desc.name = t.value("name", "");
        desc.format = ggmlToQuantFormat(t.value("dtype_id", 0));
        desc.shape = t.value("shape", std::vector<uint32_t>{});
        desc.nDims = t.value("n_dims", 0);
        desc.sizeBytes = t.value<size_t>("size_bytes", 0);
        desc.binOffset = t.value<size_t>("bin_offset", 0);
        desc.binSize = t.value<size_t>("bin_size", 0);
        desc.blockSize = t.value("quant_block_size", 1);
        desc.blockElements = t.value("quant_block_elements", 1);

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

        // CPU pre-dequant: if quantized, dequantize to F32 before GPU upload
        const uint8_t* uploadData = binData.data() + desc.binOffset;
        size_t uploadSize = desc.binSize;
        std::vector<float> dequantF32;

        if (isQuantized(desc.format)) {
            uint32_t nElements = 1;
            for (auto d : desc.shape) nElements *= d;
            fprintf(stderr, "  [dequant] CPU pre-dequant %s (%u elements, format=%u)...\n",
                    desc.name.c_str(), nElements, (uint32_t)desc.format); fflush(stderr);
            if (cpuDequantToFloat(binData.data() + desc.binOffset, desc.binSize,
                                  desc.format, desc.blockSize, desc.blockElements,
                                  nElements, dequantF32)) {
                uploadData = reinterpret_cast<const uint8_t*>(dequantF32.data());
                uploadSize = dequantF32.size() * sizeof(float);
                fprintf(stderr, "  [dequant] -> F32 (%zu bytes)\n", uploadSize); fflush(stderr);
                desc.format = QuantFormat::F32;
                desc.sizeBytes = uploadSize;
            } else {
                fprintf(stderr, "  [dequant] FAILED, uploading raw bytes\n"); fflush(stderr);
            }
        }

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
    if (desc.buffer) vkDestroyBuffer(device, desc.buffer, nullptr);
    if (desc.memory) vkFreeMemory(device, desc.memory, nullptr);
}

void WeightUploader::freeAll(ModelDesc& model) {
    for (auto& t : model.tensors) freeTensor(t);
    model.tensors.clear();
}

} // namespace rdna4
