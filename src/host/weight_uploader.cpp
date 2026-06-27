#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

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
        case 12: return QuantFormat::Q4_K;
        case 13: return QuantFormat::Q5_K;
        case 14: return QuantFormat::Q6_K;
        case 15: return QuantFormat::Q8_K;
        default: return QuantFormat::F32;
    }
}

ModelDesc WeightUploader::load(const std::string& jsonPath, const std::string& binPath) {
    std::ifstream jsonFile(jsonPath);
    nlohmann::json j;
    jsonFile >> j;

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

    std::cerr << "  [uploader] JSON parsed: " << model.architecture << "\n";

    std::ifstream binFile(binPath, std::ios::binary | std::ios::ate);
    size_t binSize = binFile.tellg();
    binFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> binData(binSize);
    binFile.read(reinterpret_cast<char*>(binData.data()), binSize);
    std::cerr << "  [uploader] Binary loaded: " << (binSize / 1024 / 1024) << " MB\n";

    // Create staging buffer — one buffer, reused for all tensors
    VkDeviceSize maxTensorSize = 0;
    for (auto& t : j["tensors"]) {
        VkDeviceSize s = t.value("bin_size", 0);
        if (s > maxTensorSize) maxTensorSize = s;
    }
    std::cerr << "  [uploader] Max tensor: " << (maxTensorSize / 1024 / 1024) << " MB\n";

    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = maxTensorSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    VkResult sr = vkCreateBuffer(device, &stagingInfo, nullptr, &staging);
    fprintf(stderr, "  [uploader] staging buffer create: %d\n", sr); fflush(stderr);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, staging, &memReq);
    fprintf(stderr, "  [uploader] staging mem req: %zu bytes\n", (size_t)memReq.size); fflush(stderr);

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
        fprintf(stderr, "  [uploader] No HOST_VISIBLE|HOST_COHERENT memory for staging\n"); fflush(stderr);
        return model;
    }

    VkMemoryAllocateInfo stagingAlloc = {};
    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAlloc.allocationSize = memReq.size;
    stagingAlloc.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory stagingMem;
    VkResult sr2 = vkAllocateMemory(device, &stagingAlloc, nullptr, &stagingMem);
    fprintf(stderr, "  [uploader] staging alloc: %d (%zu bytes)\n", sr2, (size_t)memReq.size); fflush(stderr);
    if (sr2 != VK_SUCCESS) return model;

    VkResult sr3 = vkBindBufferMemory(device, staging, stagingMem, 0);
    fprintf(stderr, "  [uploader] staging bind: %d\n", sr3); fflush(stderr);

    void* mapped = nullptr;
    VkResult sr4 = vkMapMemory(device, stagingMem, 0, maxTensorSize, 0, &mapped);
    fprintf(stderr, "  [uploader] staging map: %d, ptr=%p\n", sr4, mapped); fflush(stderr);
    if (sr4 != VK_SUCCESS || !mapped) return model;

    // Command buffer for all copies
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
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

    for (auto& t : j["tensors"]) {
        TensorDesc desc;
        desc.name = t.value("name", "");
        desc.format = ggmlToQuantFormat(t.value("dtype_id", 0));
        desc.shape = t.value("shape", std::vector<uint32_t>{});
        desc.nDims = t.value("n_dims", 0);
        desc.sizeBytes = t.value("size_bytes", 0);
        desc.binOffset = t.value("bin_offset", 0);
        desc.binSize = t.value("bin_size", 0);
        desc.blockSize = t.value("quant_block_size", 1);
        desc.blockElements = t.value("quant_block_elements", 1);

        VkDeviceAddress addr = 0;
        VkDeviceMemory bufMem = VK_NULL_HANDLE;
        std::cerr << "  [uploader] Allocating GPU buffer for " << desc.name 
                  << " (" << (desc.binSize / 1024) << " KB)\n";
        desc.buffer = createGpuBuffer(desc.binSize, &addr, &bufMem);
        desc.gpuAddress = addr;
        desc.memory = bufMem;

        // Copy data into persistent staging buffer, record copy
        std::memcpy(mapped, binData.data() + desc.binOffset, desc.binSize);
        VkMappedMemoryRange flushRange = {};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = stagingMem;
        flushRange.offset = 0;
        flushRange.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(device, 1, &flushRange);

        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = desc.binSize;
        vkCmdCopyBuffer(uploadCmd, staging, desc.buffer, 1, &copyRegion);

        model.tensors.push_back(desc);
        std::cout << "Queued: " << desc.name << " @ 0x" << std::hex << addr
                  << " (" << std::dec << (desc.sizeBytes / 1024 / 1024) << " MB)\n";
    }

    vkEndCommandBuffer(uploadCmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &uploadCmd;

    VkQueue uploadQueue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &uploadQueue);
    vkQueueSubmit(uploadQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(uploadQueue);

    vkUnmapMemory(device, stagingMem);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    vkFreeCommandBuffers(device, uploadPool, 1, &uploadCmd);
    vkDestroyCommandPool(device, uploadPool, nullptr);

    std::cout << "All " << model.tensors.size() << " tensors uploaded.\n";
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
    std::cout << "Tokenizer loaded: " << vocab.size() << " tokens, " << merges.size() << " merges\n";
}

VkBuffer WeightUploader::createGpuBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem) {
    *outAddr = 0;
    *outMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
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
    fprintf(stderr, "  [gpu] vkCreateBuffer OK\n"); fflush(stderr);

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
        fprintf(stderr, "  [gpu] No DEVICE_LOCAL memory type\n"); fflush(stderr);
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
    fprintf(stderr, "  [gpu] vkAllocateMemory OK (%zu bytes)\n", (size_t)memReq.size); fflush(stderr);

    r = vkBindBufferMemory(device, buffer, memory, 0);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "  [gpu] vkBindBufferMemory failed: %d\n", r); fflush(stderr);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }
    fprintf(stderr, "  [gpu] vkBindBufferMemory OK\n"); fflush(stderr);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    fprintf(stderr, "  [gpu] calling vkGetBufferDeviceAddress\n"); fflush(stderr);
    *outAddr = vkGetBufferDeviceAddress(device, &addrInfo);
    fprintf(stderr, "  [gpu] vkGetBufferDeviceAddress returned: 0x%llx\n", (unsigned long long)*outAddr); fflush(stderr);
    *outMem = memory;

    return buffer;
}

void WeightUploader::freeTensor(const TensorDesc& desc) {
    if (desc.buffer) vkDestroyBuffer(device, desc.buffer, nullptr);
    if (desc.memory) vkFreeMemory(device, desc.memory, nullptr);
}

} // namespace rdna4
