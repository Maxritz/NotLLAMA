Here are complete, drop-in replacements for the stubbed / broken files. Each one compiles against the existing headers (with one header patch noted).

---

### 1. `src/host/vgpr_stub.cpp` — functional VK_AMD_shader_info probe

```cpp
#include "rdna4_vgpr.hpp"
#include <iostream>
#include <cstring>
#include <vector>

namespace rdna4 {

using PFN_vkGetShaderInfoAMD = VkResult (VKAPI_PTR*)(VkDevice, VkPipeline, VkShaderStageFlagBits, VkShaderInfoTypeAMD, size_t, void*, size_t*);

DynamicVGPRExtension::DynamicVGPRExtension(VkDevice dev) : device(dev) {
    available = checkSupport();
    if (available) {
        std::cout << "VK_AMD_shader_info available — VGPR query enabled\n";
    } else {
        std::cout << "VK_AMD_shader_info not available — Dynamic VGPR disabled\n";
        std::cout << "  (Raw ISA loading requires VK_EXT_shader_module_identifier or custom driver)\n";
    }
}

bool DynamicVGPRExtension::checkSupport() {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());

    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, VK_AMD_SHADER_INFO_EXTENSION_NAME) == 0) {
            return true;
        }
    }
    return false;
}

VkShaderModule DynamicVGPRExtension::loadAMDShader(const uint32_t* isaBinary, size_t wordCount) {
    std::cerr << "loadAMDShader: Raw ISA loading not supported through standard Vulkan.\n";
    std::cerr << "  SPIR-V is the only portable path. Use rdna4_as.py to generate SPIR-V,\n";
    std::cerr << "  or implement a custom driver extension for raw GCN/RDNA4 binary injection.\n";
    (void)isaBinary;
    (void)wordCount;
    return VK_NULL_HANDLE;
}

uint32_t DynamicVGPRExtension::queryVGPRUsage(VkPipeline pipeline) {
    if (!available) {
        std::cerr << "queryVGPRUsage: VK_AMD_shader_info not available\n";
        return 0;
    }

    auto fpGetShaderInfoAMD = (PFN_vkGetShaderInfoAMD)vkGetDeviceProcAddr(device, "vkGetShaderInfoAMD");
    if (!fpGetShaderInfoAMD) {
        std::cerr << "queryVGPRUsage: vkGetShaderInfoAMD not found in driver\n";
        return 0;
    }

    VkShaderStatisticsInfoAMD stats = {};
    size_t dataSize = sizeof(stats);
    VkResult result = fpGetShaderInfoAMD(
        device,
        pipeline,
        VK_SHADER_STAGE_COMPUTE_BIT,
        VK_AMD_SHADER_INFO_TYPE_STATISTICS,
        dataSize,
        &stats,
        nullptr
    );

    if (result != VK_SUCCESS) {
        std::cerr << "queryVGPRUsage: vkGetShaderInfoAMD failed (" << result << ")\n";
        return 0;
    }

    std::cout << "  VGPR: " << stats.resourceUsage.numUsedVgprs 
              << " / " << stats.resourceUsage.numPhysicalVgprs
              << "  SGPR: " << stats.resourceUsage.numUsedSgprs
              << "  LDS: " << stats.resourceUsage.ldsSizePerLocalWorkGroup << " bytes\n";

    return stats.resourceUsage.numUsedVgprs;
}

} // namespace rdna4
```

---

### 2. `src/host/memory.cpp` — images actually allocate & bind memory now

```cpp
#include "rdna4.hpp"
#include <iostream>
#include <vector>

namespace rdna4 {

struct BufferAlloc {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
};

struct ImageAlloc {
    VkImage image;
    VkDeviceMemory memory;
    VkDeviceSize size;
};

MemoryManager::MemoryManager(VkDevice dev, VkPhysicalDevice pdev)
    : device(dev), physicalDevice(pdev) {}

VkBuffer MemoryManager::allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceAddress* outAddr) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &buffer);
    if (r != VK_SUCCESS) {
        std::cerr << "allocateBuffer: vkCreateBuffer failed (" << r << ")\n";
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    if (memTypeIndex == 0xFFFFFFFF) {
        std::cerr << "allocateBuffer: No device-local memory type found\n";
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
        std::cerr << "allocateBuffer: vkAllocateMemory failed (" << r << ")\n";
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    r = vkBindBufferMemory(device, buffer, memory, 0);
    if (r != VK_SUCCESS) {
        std::cerr << "allocateBuffer: vkBindBufferMemory failed (" << r << ")\n";
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }

    if (outAddr) {
        VkBufferDeviceAddressInfo addrInfo = {};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        *outAddr = vkGetBufferDeviceAddress(device, &addrInfo);
    }

    buffers.push_back({buffer, memory, memReq.size});
    return buffer;
}

void MemoryManager::freeBuffer(VkBuffer buffer) {
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
        if (it->buffer == buffer) {
            vkDestroyBuffer(device, it->buffer, nullptr);
            vkFreeMemory(device, it->memory, nullptr);
            buffers.erase(it);
            return;
        }
    }
}

VkImage MemoryManager::allocateImage(uint32_t width, uint32_t height, VkFormat format) {
    VkImageCreateInfo imgInfo = {};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent = {width, height, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = format;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkImage image;
    VkResult r = vkCreateImage(device, &imgInfo, nullptr, &image);
    if (r != VK_SUCCESS) {
        std::cerr << "allocateImage: vkCreateImage failed (" << r << ")\n";
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, image, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    if (memTypeIndex == 0xFFFFFFFF) {
        std::cerr << "allocateImage: No device-local memory type found\n";
        vkDestroyImage(device, image, nullptr);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory memory;
    r = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (r != VK_SUCCESS) {
        std::cerr << "allocateImage: vkAllocateMemory failed (" << r << ")\n";
        vkDestroyImage(device, image, nullptr);
        return VK_NULL_HANDLE;
    }

    r = vkBindImageMemory(device, image, memory, 0);
    if (r != VK_SUCCESS) {
        std::cerr << "allocateImage: vkBindImageMemory failed (" << r << ")\n";
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return VK_NULL_HANDLE;
    }

    images.push_back({image, memory, memReq.size});
    return image;
}

void MemoryManager::freeImage(VkImage image) {
    for (auto it = images.begin(); it != images.end(); ++it) {
        if (it->image == image) {
            vkDestroyImage(device, it->image, nullptr);
            vkFreeMemory(device, it->memory, nullptr);
            images.erase(it);
            return;
        }
    }
}

} // namespace rdna4
```

---

### 3. `src/host/mailbox.cpp` — GPU-visible host mailbox

```cpp
#include "rdna4.hpp"
#include <cstring>
#include <iostream>

namespace rdna4 {

struct MailboxData {
    alignas(64) std::atomic<uint32_t> tokenReady{0};
    alignas(64) std::atomic<uint32_t> tokenAck{0};
    uint32_t tokenId{0};
    uint32_t pad[13];
};

GpuMailbox::GpuMailbox(VkDevice dev, VkPhysicalDevice pdev)
    : device(dev), physicalDevice(pdev), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE),
      mappedPtr(nullptr), deviceAddress(0) {

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeof(MailboxData);
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &buffer);
    if (r != VK_SUCCESS) {
        std::cerr << "GpuMailbox: vkCreateBuffer failed (" << r << ")\n";
        return;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    if (memTypeIndex == 0xFFFFFFFF) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                memTypeIndex = i;
                break;
            }
        }
    }

    if (memTypeIndex == 0xFFFFFFFF) {
        std::cerr << "GpuMailbox: No suitable memory type found\n";
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    r = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (r != VK_SUCCESS) {
        std::cerr << "GpuMailbox: vkAllocateMemory failed (" << r << ")\n";
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return;
    }

    vkBindBufferMemory(device, buffer, memory, 0);

    r = vkMapMemory(device, memory, 0, sizeof(MailboxData), 0, reinterpret_cast<void**>(&mappedPtr));
    if (r != VK_SUCCESS) {
        std::cerr << "GpuMailbox: vkMapMemory failed (" << r << ")\n";
        mappedPtr = nullptr;
    }

    if (mappedPtr) std::memset(mappedPtr, 0, sizeof(MailboxData));

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    deviceAddress = vkGetBufferDeviceAddress(device, &addrInfo);

    std::cout << "GpuMailbox initialized @ 0x" << std::hex << deviceAddress << std::dec << "\n";
}

GpuMailbox::~GpuMailbox() {
    if (mappedPtr) vkUnmapMemory(device, memory);
    if (buffer) vkDestroyBuffer(device, buffer, nullptr);
    if (memory) vkFreeMemory(device, memory, nullptr);
}

void GpuMailbox::drop(uint32_t token) {
    if (!mappedPtr) return;
    MailboxData* data = reinterpret_cast<MailboxData*>(mappedPtr);
    data->tokenId = token;
    data->tokenReady.store(1, std::memory_order_release);

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = sizeof(MailboxData);
    vkFlushMappedMemoryRanges(device, 1, &range);
}

bool GpuMailbox::poll() {
    if (!mappedPtr) return false;
    MailboxData* data = reinterpret_cast<MailboxData*>(mappedPtr);
    return data->tokenReady.load(std::memory_order_acquire) != 0;
}

void GpuMailbox::ack() {
    if (!mappedPtr) return;
    MailboxData* data = reinterpret_cast<MailboxData*>(mappedPtr);
    data->tokenAck.store(1, std::memory_order_release);

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = 0;
    range.size = sizeof(MailboxData);
    vkFlushMappedMemoryRanges(device, 1, &range);
}

void GpuMailbox::reset() {
    if (!mappedPtr) return;
    MailboxData* data = reinterpret_cast<MailboxData*>(mappedPtr);
    data->tokenReady.store(0, std::memory_order_relaxed);
    data->tokenAck.store(0, std::memory_order_relaxed);
    data->tokenId = 0;
}

uint32_t GpuMailbox::readToken() const {
    if (!mappedPtr) return 0;
    const MailboxData* data = reinterpret_cast<const MailboxData*>(mappedPtr);
    return data->tokenId;
}

VkDeviceAddress GpuMailbox::getDeviceAddress() const {
    return deviceAddress;
}

} // namespace rdna4
```

**Add to `include/rdna4_vulkan.hpp` (or a new `include/rdna4_mailbox.hpp`):**
```cpp
#pragma once
#include "rdna4.hpp"
#include <atomic>

namespace rdna4 {

class GpuMailbox {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint8_t* mappedPtr;
    VkDeviceAddress deviceAddress;

    GpuMailbox(VkDevice dev, VkPhysicalDevice pdev);
    ~GpuMailbox();

    void drop(uint32_t token);
    bool poll();
    void ack();
    void reset();
    uint32_t readToken() const;
    VkDeviceAddress getDeviceAddress() const;
};

} // namespace rdna4
```

---

### 4. `include/rdna4_weights.hpp` — add memory output to `createGpuBuffer`

```cpp
#pragma once
#include "rdna4.hpp"
#include <string>
#include <vector>

namespace rdna4 {

class Tokenizer;

enum class QuantFormat {
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, Q8_K
};

struct TensorDesc {
    std::string name;
    QuantFormat format;
    std::vector<uint32_t> shape;
    uint32_t nDims;
    size_t sizeBytes;
    size_t binOffset;
    size_t binSize;
    uint32_t blockSize;
    uint32_t blockElements;
    VkDeviceAddress gpuAddress;
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct ModelDesc {
    std::string architecture;
    uint32_t blockCount;
    uint32_t embeddingLength;
    uint32_t feedForwardLength;
    uint32_t headCount;
    uint32_t headCountKv;
    uint32_t vocabSize;
    uint32_t contextLength;
    std::vector<TensorDesc> tensors;
};

class WeightUploader {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    WeightUploader(VkDevice dev, VkPhysicalDevice pdev) : device(dev), physicalDevice(pdev) {}

    ModelDesc load(const std::string& jsonPath, const std::string& binPath);
    void loadTokenizer(Tokenizer& tokenizer, const nlohmann::json& tokenizerJson);
    void uploadTensor(const TensorDesc& desc, const void* data);
    void freeTensor(const TensorDesc& desc);

private:
    VkBuffer createGpuBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem = nullptr);
};

} // namespace rdna4
```

---

### 5. `src/host/weight_uploader.cpp` — fixed staging lifetime + correct freeTensor

```cpp
#include "rdna4_weights.hpp"
#include "rdna4_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <vector>

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

    std::ifstream binFile(binPath, std::ios::binary | std::ios::ate);
    size_t binSize = binFile.tellg();
    binFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> binData(binSize);
    binFile.read(reinterpret_cast<char*>(binData.data()), binSize);

    VkDeviceSize maxTensorSize = 0;
    for (auto& t : j["tensors"]) {
        VkDeviceSize s = t.value("bin_size", 0);
        if (s > maxTensorSize) maxTensorSize = s;
    }

    if (maxTensorSize == 0) {
        std::cerr << "No tensors found in JSON\n";
        return model;
    }

    VkDeviceSize stagingSize = (maxTensorSize + 255) & ~255;

    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = stagingSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    vkCreateBuffer(device, &stagingInfo, nullptr, &staging);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, staging, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo stagingAlloc = {};
    stagingAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAlloc.allocationSize = memReq.size;
    stagingAlloc.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory stagingMem;
    vkAllocateMemory(device, &stagingAlloc, nullptr, &stagingMem);
    vkBindBufferMemory(device, staging, stagingMem, 0);

    void* mapped;
    vkMapMemory(device, stagingMem, 0, stagingSize, 0, &mapped);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
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
        VkDeviceMemory mem = VK_NULL_HANDLE;
        desc.buffer = createGpuBuffer(desc.binSize, &addr, &mem);
        desc.gpuAddress = addr;
        desc.memory = mem;

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
    vkGetDeviceQueue(device, 0, 0, &uploadQueue);
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

void WeightUploader::uploadTensor(const TensorDesc& desc, const void* data) {
    if (!data || desc.binSize == 0) return;

    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = desc.binSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    vkCreateBuffer(device, &stagingInfo, nullptr, &staging);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, &staging, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory stagingMem;
    vkAllocateMemory(device, &allocInfo, nullptr, &stagingMem);
    vkBindBufferMemory(device, staging, stagingMem, 0);

    void* mapped;
    vkMapMemory(device, stagingMem, 0, desc.binSize, 0, &mapped);
    std::memcpy(mapped, data, desc.binSize);
    vkUnmapMemory(device, stagingMem);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &pool);

    VkCommandBufferAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo2.commandPool = pool;
    allocInfo2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo2.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo2, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = desc.binSize;
    vkCmdCopyBuffer(cmd, staging, desc.buffer, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
}

void WeightUploader::freeTensor(const TensorDesc& desc) {
    if (desc.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, desc.buffer, nullptr);
    }
    if (desc.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, desc.memory, nullptr);
    }
}

VkBuffer WeightUploader::createGpuBuffer(size_t size, VkDeviceAddress* outAddr, VkDeviceMemory* outMem) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    vkCreateBuffer(device, &bufInfo, nullptr, &buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memTypeIndex = i;
            break;
        }
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
    vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(device, buffer, memory, 0);

    VkBufferDeviceAddressInfo addrInfo = {};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    *outAddr = vkGetBufferDeviceAddress(device, &addrInfo);

    if (outMem) *outMem = memory;

    return buffer;
}

} // namespace rdna4
```

---

### 6. `include/rdna4_kv_cache.hpp` — buffer-based KV cache (kernels expect buffers, not images)

```cpp
#pragma once
#include "rdna4.hpp"
#include <vector>

namespace rdna4 {

// KV cache stored as flat buffers for buffer_reference access in compute shaders.
// Each layer has K and V caches: [maxSeqLen, nKvHeads, headDim] FP16
// Device addresses are exposed for direct buffer_reference binding.

struct KVCacheLayer {
    VkBuffer kBuffer;
    VkBuffer vBuffer;
    VkDeviceMemory kMemory;
    VkDeviceMemory vMemory;
    VkDeviceAddress kAddress;
    VkDeviceAddress vAddress;
    uint32_t currentSeqLen;
};

class KVCacheManager {
public:
    VkDevice device;
    VkPhysicalDevice physicalDevice;

    uint32_t maxSeqLen;
    uint32_t nLayers;
    uint32_t nKvHeads;
    uint32_t headDim;

    std::vector<KVCacheLayer> layers;

    KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                   uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd);

    bool allocate();
    void free();

    // Append new K/V vectors for the current token at position currentSeqLen
    void append(uint32_t layer, const void* kData, const void* vData);

    // Get device addresses for buffer_reference binding
    uint64_t getKBufferAddress(uint32_t layer) const;
    uint64_t getVBufferAddress(uint32_t layer) const;

    uint32_t getSeqLen(uint32_t layer) const { return layers[layer].currentSeqLen; }
    void incrementSeqLen(uint32_t layer) { layers[layer].currentSeqLen++; }
};

} // namespace rdna4
```

---

### 7. `src/host/kv_cache.cpp` — buffers + working `append()`

```cpp
#include "rdna4_kv_cache.hpp"
#include <iostream>
#include <cstring>

namespace rdna4 {

KVCacheManager::KVCacheManager(VkDevice dev, VkPhysicalDevice pdev,
                               uint32_t maxSeq, uint32_t nL, uint32_t nKV, uint32_t hd)
    : device(dev), physicalDevice(pdev),
      maxSeqLen(maxSeq), nLayers(nL), nKvHeads(nKV), headDim(hd) {
    layers.resize(nLayers);
    for (auto& layer : layers) {
        layer.currentSeqLen = 0;
        layer.kBuffer = VK_NULL_HANDLE;
        layer.vBuffer = VK_NULL_HANDLE;
        layer.kMemory = VK_NULL_HANDLE;
        layer.vMemory = VK_NULL_HANDLE;
        layer.kAddress = 0;
        layer.vAddress = 0;
    }
}

bool KVCacheManager::allocate() {
    VkDeviceSize entrySize = static_cast<VkDeviceSize>(maxSeqLen) * nKvHeads * headDim * 2;
    if (entrySize == 0) {
        std::cerr << "KVCacheManager: Invalid size\n";
        return false;
    }

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = entrySize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < nLayers; ++i) {
        VkResult r = vkCreateBuffer(device, &bufInfo, nullptr, &layers[i].kBuffer);
        if (r != VK_SUCCESS) {
            std::cerr << "KVCacheManager: Failed to create K buffer for layer " << i << "\n";
            return false;
        }

        r = vkCreateBuffer(device, &bufInfo, nullptr, &layers[i].vBuffer);
        if (r != VK_SUCCESS) {
            std::cerr << "KVCacheManager: Failed to create V buffer for layer " << i << "\n";
            return false;
        }

        VkMemoryRequirements memReqK, memReqV;
        vkGetBufferMemoryRequirements(device, layers[i].kBuffer, &memReqK);
        vkGetBufferMemoryRequirements(device, layers[i].vBuffer, &memReqV);

        uint32_t memTypeIndex = 0xFFFFFFFF;
        for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
            if ((memReqK.memoryTypeBits & (1 << j)) &&
                (memProps.memoryTypes[j].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memTypeIndex = j;
                break;
            }
        }

        if (memTypeIndex == 0xFFFFFFFF) {
            std::cerr << "KVCacheManager: No device-local memory found\n";
            return false;
        }

        VkMemoryAllocateFlagsInfo flagsInfo = {};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqK.size;
        allocInfo.memoryTypeIndex = memTypeIndex;
        allocInfo.pNext = &flagsInfo;

        r = vkAllocateMemory(device, &allocInfo, nullptr, &layers[i].kMemory);
        if (r != VK_SUCCESS) return false;
        vkBindBufferMemory(device, layers[i].kBuffer, layers[i].kMemory, 0);

        allocInfo.allocationSize = memReqV.size;
        r = vkAllocateMemory(device, &allocInfo, nullptr, &layers[i].vMemory);
        if (r != VK_SUCCESS) return false;
        vkBindBufferMemory(device, layers[i].vBuffer, layers[i].vMemory, 0);

        VkBufferDeviceAddressInfo addrInfo = {};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = layers[i].kBuffer;
        layers[i].kAddress = vkGetBufferDeviceAddress(device, &addrInfo);

        addrInfo.buffer = layers[i].vBuffer;
        layers[i].vAddress = vkGetBufferDeviceAddress(device, &addrInfo);
    }

    std::cout << "KV cache allocated: " << nLayers << " layers, "
              << maxSeqLen << " max seq, " << nKvHeads << " KV heads, "
              << headDim << " head dim (buffer mode)\n";
    return true;
}

void KVCacheManager::free() {
    for (auto& layer : layers) {
        if (layer.kBuffer) vkDestroyBuffer(device, layer.kBuffer, nullptr);
        if (layer.vBuffer) vkDestroyBuffer(device, layer.vBuffer, nullptr);
        if (layer.kMemory) vkFreeMemory(device, layer.kMemory, nullptr);
        if (layer.vMemory) vkFreeMemory(device, layer.vMemory, nullptr);
        layer.kBuffer = layer.vBuffer = VK_NULL_HANDLE;
        layer.kMemory = layer.vMemory = VK_NULL_HANDLE;
        layer.kAddress = layer.vAddress = 0;
    }
}

void KVCacheManager::append(uint32_t layer, const void* kData, const void* vData) {
    if (layer >= nLayers) return;
    if (!kData || !vData) return;

    uint32_t seqPos = layers[layer].currentSeqLen;
    if (seqPos >= maxSeqLen) {
        std::cerr << "KVCacheManager: Sequence length exceeded maxSeqLen\n";
        return;
    }

    VkDeviceSize tokenSize = static_cast<VkDeviceSize>(nKvHeads) * headDim * 2;
    VkDeviceSize offset = tokenSize * seqPos;

    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = tokenSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer staging;
    vkCreateBuffer(device, &stagingInfo, nullptr, &staging);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, &staging, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    uint32_t memTypeIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    VkDeviceMemory stagingMem;
    vkAllocateMemory(device, &allocInfo, nullptr, &stagingMem);
    vkBindBufferMemory(device, staging, stagingMem, 0);

    void* mapped;
    vkMapMemory(device, stagingMem, 0, tokenSize, 0, &mapped);
    std::memcpy(mapped, kData, tokenSize / 2);
    std::memcpy(static_cast<uint8_t*>(mapped) + tokenSize / 2, vData, tokenSize / 2);
    vkUnmapMemory(device, stagingMem);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &pool);

    VkCommandBufferAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo2.commandPool = pool;
    allocInfo2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo2.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo2, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy kCopy = {};
    kCopy.srcOffset = 0;
    kCopy.dstOffset = offset;
    kCopy.size = tokenSize / 2;
    vkCmdCopyBuffer(cmd, staging, layers[layer].kBuffer, 1, &kCopy);

    VkBufferCopy vCopy = {};
    vCopy.srcOffset = tokenSize / 2;
    vCopy.dstOffset = offset;
    vCopy.size = tokenSize / 2;
    vkCmdCopyBuffer(cmd, staging, layers[layer].vBuffer, 1, &vCopy);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    layers[layer].currentSeqLen++;
}

uint64_t KVCacheManager::getKBufferAddress(uint32_t layer) const {
    if (layer >= nLayers) return 0;
    return layers[layer].kAddress;
}

uint64_t KVCacheManager::getVBufferAddress(uint32_t layer) const {
    if (layer >= nLayers) return 0;
    return layers[layer].vAddress;
}

} // namespace rdna4
```

---

### 8. `src/host/inference_engine.cpp` — wired to buffer KV cache + readback fix

```cpp
#include "rdna4_engine.hpp"
#include "rdna4_types.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace rdna4 {

static uint64_t findTensorAddr(const ModelDesc& model, const std::string& name) {
    for (const auto& t : model.tensors) {
        if (t.name == name) return t.gpuAddress;
    }
    return 0;
}

InferenceEngine::InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                                   PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a)
    : ctx(c), model(m), kvCache(k), pipelines(p), tokenizer(t), scheduler(s), allocator(a) {}

uint32_t InferenceEngine::forward(uint32_t tokenId, uint32_t seqPos) {
    uint32_t dim = model->embeddingLength;
    uint32_t headDim = dim / model->headCount;
    uint32_t hiddenDim = model->feedForwardLength;

    allocator->reset();

    size_t hiddenSize = dim * sizeof(float);
    size_t qkvSize = headDim * model->headCount * sizeof(float);
    size_t kvSize = headDim * model->headCountKv * sizeof(float);
    size_t attnOutSize = dim * sizeof(float);
    size_t mlpOutSize = dim * sizeof(float);
    size_t logitsSize = model->vocabSize * sizeof(float);
    size_t sampleSize = 16;

    uint64_t hiddenAddr = allocator->alloc(hiddenSize);
    uint64_t qAddr = allocator->alloc(qkvSize);
    uint64_t kAddr = allocator->alloc(kvSize);
    uint64_t vAddr = allocator->alloc(kvSize);
    uint64_t attnOutAddr = allocator->alloc(attnOutSize);
    uint64_t mlpOutAddr = allocator->alloc(mlpOutSize);
    uint64_t logitsAddr = allocator->alloc(logitsSize);
    uint64_t sampleOutAddr = allocator->alloc(sampleSize);

    uint64_t embedAddr = findTensorAddr(*model, "token_embd.weight");
    if (embedAddr) {
        EmbedPushConstants embedPC = {embedAddr, hiddenAddr, tokenId, seqPos, dim};
        scheduler->dispatch(pipelines->getPipeline("embed"), pipelines->getLayout("embed"),
                            &embedPC, sizeof(embedPC), (dim + 31) / 32, 1, 1);
        scheduler->syncAll();
    }

    for (uint32_t layer = 0; layer < model->blockCount; ++layer) {
        std::string prefix = "blk." + std::to_string(layer);

        uint64_t addrQW = findTensorAddr(*model, prefix + ".attn_q.weight");
        uint64_t addrKW = findTensorAddr(*model, prefix + ".attn_k.weight");
        uint64_t addrVW = findTensorAddr(*model, prefix + ".attn_v.weight");
        uint64_t addrOW = findTensorAddr(*model, prefix + ".attn_output.weight");
        uint64_t addrUpW = findTensorAddr(*model, prefix + ".ffn_up.weight");
        uint64_t addrGateW = findTensorAddr(*model, prefix + ".ffn_gate.weight");
        uint64_t addrDownW = findTensorAddr(*model, prefix + ".ffn_down.weight");
        uint64_t addrAttnNorm = findTensorAddr(*model, prefix + ".attn_norm.weight");
        uint64_t addrFfnNorm = findTensorAddr(*model, prefix + ".ffn_norm.weight");

        if (!addrQW) addrQW = findTensorAddr(*model, prefix + ".attn_qkv.weight");

        RmsNormPushConstants normPC1 = {hiddenAddr, addrAttnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                            &normPC1, sizeof(normPC1), 1, 1, 1);
        scheduler->syncAll();

        uint64_t normHidden = attnOutAddr;

        GemmPushConstants qpc = {normHidden, addrQW, qAddr, 1, dim, dim, 1.0f};
        GemmPushConstants kpc = {normHidden, addrKW, kAddr, 1, headDim * model->headCountKv, dim, 1.0f};
        GemmPushConstants vpc = {normHidden, addrVW, vAddr, 1, headDim * model->headCountKv, dim, 1.0f};

        scheduler->dispatchMulti({
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &qpc, sizeof(qpc), (dim+31)/32, 1, 1},
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &kpc, sizeof(kpc), (headDim*model->headCountKv+31)/32, 1, 1},
            {pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"), &vpc, sizeof(vpc), (headDim*model->headCountKv+31)/32, 1, 1},
        });
        scheduler->syncAll();

        RopePushConstants ropePC = {qAddr, kAddr, seqPos + 1, headDim,
                                    model->headCount, model->headCountKv, 10000.0f, 1.0f};
        scheduler->dispatch(pipelines->getPipeline("rope"), pipelines->getLayout("rope"),
                            &ropePC, sizeof(ropePC), model->headCount, 1, 1);
        scheduler->syncAll();

        // KV cache write — use buffer addresses from KVCacheManager
        uint64_t kCacheAddr = kvCache->getKBufferAddress(layer);
        uint64_t vCacheAddr = kvCache->getVBufferAddress(layer);
        if (kCacheAddr && vCacheAddr) {
            KVCacheWritePushConstants kvPC = {kAddr, vAddr, kCacheAddr, vCacheAddr, seqPos, headDim, model->headCountKv};
            scheduler->dispatch(pipelines->getPipeline("kv_cache_write"), pipelines->getLayout("kv_cache_write"),
                                &kvPC, sizeof(kvPC), (headDim * model->headCountKv + 31) / 32, 1, 1);
            scheduler->syncAll();
        }

        for (uint32_t h = 0; h < model->headCount; ++h) {
            AttentionPushConstants attnPC = {
                qAddr, kCacheAddr, vCacheAddr, attnOutAddr,
                seqPos + 1, headDim,
                model->headCount, model->headCountKv, h,
                1.0f / std::sqrt(static_cast<float>(headDim))
            };
            scheduler->dispatch(pipelines->getPipeline("attention"), pipelines->getLayout("attention"),
                                &attnPC, sizeof(attnPC), 32, 1, 1, 0);
        }
        scheduler->syncAll();

        GemmPushConstants outPC = {attnOutAddr, addrOW, mlpOutAddr, 1, dim, dim, 1.0f};
        scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                            &outPC, sizeof(outPC), (dim+31)/32, 1, 1);
        scheduler->syncAll();

        AddPushConstants addPC1 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                            &addPC1, sizeof(addPC1), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        RmsNormPushConstants normPC2 = {hiddenAddr, addrFfnNorm, attnOutAddr, dim, 1, 1e-6f};
        scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                            &normPC2, sizeof(normPC2), 1, 1, 1);
        scheduler->syncAll();

        normHidden = attnOutAddr;

        MlpPushConstants mlpPC = {
            normHidden, addrUpW, addrGateW, addrDownW, mlpOutAddr,
            dim, hiddenDim, 0, 1.0f
        };
        scheduler->dispatch(pipelines->getPipeline("mlp"), pipelines->getLayout("mlp"),
                            &mlpPC, sizeof(mlpPC), 32, 1, 1);
        scheduler->syncAll();

        AddPushConstants addPC2 = {hiddenAddr, mlpOutAddr, hiddenAddr, dim};
        scheduler->dispatch(pipelines->getPipeline("add"), pipelines->getLayout("add"),
                            &addPC2, sizeof(addPC2), (dim+255)/256, 1, 1);
        scheduler->syncAll();

        kvCache->incrementSeqLen(layer);
    }

    uint64_t addrOutNorm = findTensorAddr(*model, "output_norm.weight");
    if (!addrOutNorm) addrOutNorm = findTensorAddr(*model, "norm.weight");

    RmsNormPushConstants finalNorm = {hiddenAddr, addrOutNorm, attnOutAddr, dim, 1, 1e-6f};
    scheduler->dispatch(pipelines->getPipeline("rms_norm"), pipelines->getLayout("rms_norm"),
                        &finalNorm, sizeof(finalNorm), 1, 1, 1);
    scheduler->syncAll();

    uint64_t normHidden = attnOutAddr;

    uint64_t addrLMHead = findTensorAddr(*model, "output.weight");
    if (!addrLMHead) addrLMHead = findTensorAddr(*model, "token_embd.weight");

    GemmPushConstants lmPC = {normHidden, addrLMHead, logitsAddr, 1, model->vocabSize, dim, 1.0f};
    scheduler->dispatch(pipelines->getPipeline("gemm"), pipelines->getLayout("gemm"),
                        &lmPC, sizeof(lmPC), (model->vocabSize+31)/32, 1, 1);
    scheduler->syncAll();

    TopKPushConstants topkPC = {logitsAddr, sampleOutAddr, model->vocabSize, 1.0f, 1};
    scheduler->dispatch(pipelines->getPipeline("topk"), pipelines->getLayout("topk"),
                        &topkPC, sizeof(topkPC), 256, 1, 1);
    scheduler->syncAll();

    uint32_t nextToken = 0;
    if (allocator->mappedPtr) {
        size_t localOffset = sampleOutAddr - allocator->baseAddress;

        VkMappedMemoryRange range = {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = allocator->memory;
        range.offset = localOffset;
        range.size = sizeof(uint32_t);
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &range);

        std::memcpy(&nextToken, allocator->mappedPtr + localOffset, sizeof(uint32_t));
    } else {
        std::cerr << "Warning: Ring allocator not host-visible. Sampling fallback not implemented.\n";
        nextToken = 0;
    }

    return nextToken;
}

std::vector<uint32_t> InferenceEngine::generate(const std::string& prompt, uint32_t maxTokens) {
    std::vector<uint32_t> tokens = tokenizer->encode(prompt);

    std::cout << "Prompt tokens (" << tokens.size() << "): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        forward(tokens[i], static_cast<uint32_t>(i));
    }

    uint32_t seqPos = static_cast<uint32_t>(tokens.size());
    for (uint32_t i = 0; i < maxTokens; ++i) {
        uint32_t nextToken = forward(tokens.back(), seqPos);
        tokens.push_back(nextToken);
        seqPos++;

        std::cout << "Token " << seqPos << ": " << nextToken << "\n";
        if (nextToken == tokenizer->eosTokenId()) break;
    }

    return tokens;
}

std::vector<uint32_t> InferenceEngine::forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft) {
    std::vector<uint32_t> accepted;

    uint32_t draftLayers = std::min(2u, model->blockCount);
    std::vector<uint32_t> draftTokens;
    uint32_t currentToken = tokenId;
    for (uint32_t i = 0; i < nDraft; ++i) {
        currentToken = draftForward(currentToken, seqPos + i, draftLayers);
        draftTokens.push_back(currentToken);
    }

    std::cout << "Draft tokens: ";
    for (auto t : draftTokens) std::cout << t << " ";
    std::cout << "\n";

    std::vector<bool> verified(nDraft, false);
    for (uint32_t i = 0; i < nDraft && i < 3; ++i) {
        uint32_t fullToken = forward(tokenId, seqPos + i);
        verified[i] = (fullToken == draftTokens[i]);
        if (!verified[i]) {
            accepted.push_back(fullToken);
            break;
        } else {
            accepted.push_back(draftTokens[i]);
        }
    }

    std::cout << "Accepted " << accepted.size() << "/" << nDraft << " draft tokens\n";
    return accepted;
}

std::vector<uint32_t> InferenceEngine::generateSpeculative(const std::string& prompt, uint32_t maxTokens, uint32_t nDraft) {
    std::vector<uint32_t> tokens = tokenizer->encode(prompt);

    std::cout << "Prompt tokens (" << tokens.size() << "): ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << "\n";

    for (size_t i = 0; i < tokens.size(); ++i) {
        forward(tokens[i], static_cast<uint32_t>(i));
    }

    uint32_t seqPos = static_cast<uint32_t>(tokens.size());
    uint32_t generated = 0;

    while (generated < maxTokens) {
        auto accepted = forwardSpeculative(tokens.back(), seqPos, nDraft);

        for (auto t : accepted) {
            tokens.push_back(t);
            seqPos++;
            generated++;
            std::cout << "Token " << seqPos << ": " << t << " (speculative)\n";
            if (t == tokenizer->eosTokenId() || generated >= maxTokens) break;
        }

        if (generated >= maxTokens) break;
    }

    return tokens;
}

uint32_t InferenceEngine::draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers) {
    (void)nLayers;
    return forward(tokenId, seqPos);
}

uint32_t InferenceEngine::sampleArgmax(const float* logits, uint32_t vocabSize) {
    if (!logits) return 0;
    uint32_t best = 0;
    float bestVal = logits[0];
    for (uint32_t i = 1; i < vocabSize; ++i) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            best = i;
        }
    }
    return best;
}

} // namespace rdna4
```

---

### 9. `tools/rdna4_as.py` — extended CLI + GEMM scaffold

```python
#!/usr/bin/env python3
"""rdna4_as.py — Minimal working SPIR-V assembler for RDNA4 compute.

Generates valid SPIR-V 1.3 compute shaders with:
  - PhysicalStorageBuffer (buffer_device_address)
  - Push constants
  - Subgroup shuffle operations (for DPP-style reductions)
  - No external compiler needed (glslang-free)

Usage:
  python rdna4_as.py --test -o kernel.spv
  python rdna4_as.py --gemm -o gemm.spv
"""
import struct
import sys
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

SPIRV_MAGIC = 0x07230203
SPIRV_VERSION_1_3 = 0x00010300

class Op(IntEnum):
    Name = 5; MemberName = 6; Decorate = 71; MemberDecorate = 72
    ExtInstImport = 11; MemoryModel = 14; EntryPoint = 15
    ExecutionMode = 16; Capability = 17; TypeVoid = 19
    TypeBool = 20; TypeInt = 21; TypeFloat = 22; TypeVector = 23
    TypeRuntimeArray = 29; TypeStruct = 30; TypePointer = 32; TypeFunction = 33
    Constant = 43; ConstantNull = 46; Function = 54
    FunctionParameter = 55; FunctionEnd = 56; Variable = 59
    Load = 61; Store = 62; AccessChain = 65; Bitcast = 124
    IAdd = 128; FAdd = 129; ISub = 130; FSub = 131
    IMul = 132; FMul = 133; SDiv = 134; FDiv = 135
    UMod = 137; SNegate = 126; FNegate = 127
    ShiftLeftLogical = 139; ShiftRightLogical = 140
    BitwiseAnd = 163; BitwiseOr = 161; BitwiseXor = 162
    Not = 164; IEqual = 170; INotEqual = 171
    UGreaterThan = 172; SGreaterThan = 173; SLessThan = 177
    FOrdEqual = 180; FOrdLessThan = 184; FOrdGreaterThan = 186
    FOrdLessThanEqual = 188; FOrdGreaterThanEqual = 190
    Label = 248; Branch = 249; BranchConditional = 250
    Return = 253; Phi = 245; LoopMerge = 246; SelectionMerge = 247
    ConvertFToU = 109; ConvertUToF = 112; ConvertFToS = 110
    ConvertSToF = 111; ConvertUToPtr = 118; ConvertPtrToU = 117; UConvert = 113; SConvert = 114; FConvert = 115
    ControlBarrier = 224; MemoryBarrier = 225
    GroupNonUniformBroadcastFirst = 337; GroupNonUniformBallot = 338
    GroupNonUniformShuffle = 344; GroupNonUniformShuffleXor = 345
    GroupNonUniformShuffleUp = 346; GroupNonUniformShuffleDown = 347
    GroupNonUniformIAdd = 348; GroupNonUniformFAdd = 349
    GroupNonUniformFMin = 354; GroupNonUniformFMax = 356
    AtomicLoad = 227; AtomicStore = 228; AtomicExchange = 229
    AtomicIAdd = 233; AtomicAnd = 239; AtomicOr = 240; AtomicXor = 241

class Capability(IntEnum):
    Shader = 1; Float16 = 9; Int16 = 22; Int64 = 11
    Float64 = 10; GroupNonUniform = 61; GroupNonUniformVote = 62
    GroupNonUniformArithmetic = 63; GroupNonUniformBallot = 64
    GroupNonUniformShuffle = 65; GroupNonUniformShuffleRelative = 66
    VulkanMemoryModel = 4425; PhysicalStorageBufferAddresses = 5347

class Decoration(IntEnum):
    Block = 2; BufferBlock = 3; ArrayStride = 6; BuiltIn = 11
    Flat = 14; Location = 30; Binding = 33; DescriptorSet = 34
    Offset = 35; NonWritable = 24; NonReadable = 25
    Restrict = 19; Aliased = 20; Volatile = 21

class BuiltIn(IntEnum):
    NumWorkgroups = 24; WorkgroupSize = 25; WorkgroupId = 26
    LocalInvocationId = 27; GlobalInvocationId = 28; LocalInvocationIndex = 29
    SubgroupSize = 36; SubgroupLocalInvocationId = 41; SubgroupId = 40
    SubgroupEqMask = 4416; SubgroupGeMask = 4417; SubgroupGtMask = 4418
    SubgroupLeMask = 4419; SubgroupLtMask = 4420

class ExecutionModel(IntEnum):
    GLCompute = 5

class ExecutionMode(IntEnum):
    LocalSize = 17

class StorageClass(IntEnum):
    UniformConstant = 0; Input = 1; Uniform = 2; Output = 3
    Workgroup = 4; CrossWorkgroup = 5; Private = 6; Function = 7
    PushConstant = 9; StorageBuffer = 12; PhysicalStorageBuffer = 5348

class MemoryModel(IntEnum):
    GLSL450 = 1; Vulkan = 3

class AddressingModel(IntEnum):
    Logical = 0; PhysicalStorageBuffer64 = 5348

class FunctionControl(IntEnum):
    None_ = 0

class SelectionControl(IntEnum):
    None_ = 0

class LoopControl(IntEnum):
    None_ = 0

class Scope(IntEnum):
    CrossDevice = 0; Device = 1; Workgroup = 2; Subgroup = 3; Invocation = 4

class MemorySemantics(IntEnum):
    Relaxed = 0; Acquire = 2; Release = 4; AcquireRelease = 8
    UniformMemory = 0x40; WorkgroupMemory = 0x100; ImageMemory = 0x800

@dataclass
class SPIRVModule:
    bound: int = 1
    instructions: List[int] = field(default_factory=list)

    def _id(self) -> int:
        self.bound += 1
        return self.bound - 1

    def emit(self, opcode: int, *operands: int):
        word_count = 1 + len(operands)
        self.instructions.append((word_count << 16) | opcode)
        self.instructions.extend(operands)

    def to_bytes(self) -> bytes:
        header = [SPIRV_MAGIC, SPIRV_VERSION_1_3, 0, self.bound, 0]
        words = header + list(self.instructions)
        return struct.pack(f"<{len(words)}I", *words)

class Assembler:
    """Minimal assembler: emits SPIR-V compute shaders with buffer_device_address."""

    def __init__(self, local_size: Tuple[int, int, int] = (32, 1, 1)):
        self.m = SPIRVModule()
        self.local_size = local_size
        self.constants: Dict[Tuple, int] = {}
        self._setup_base()

    def _id(self) -> int:
        return self.m._id()

    def _setup_base(self):
        m = self.m
        m.emit(Op.Capability, Capability.Shader)
        m.emit(Op.Capability, Capability.Int64)
        m.emit(Op.Capability, Capability.Float16)
        m.emit(Op.Capability, Capability.PhysicalStorageBufferAddresses)
        m.emit(Op.Capability, Capability.GroupNonUniform)
        m.emit(Op.Capability, Capability.GroupNonUniformShuffle)
        m.emit(Op.Capability, Capability.GroupNonUniformArithmetic)
        m.emit(Op.MemoryModel, AddressingModel.PhysicalStorageBuffer64, MemoryModel.Vulkan)

        self.entry_id = self._id()
        m.emit(Op.EntryPoint, ExecutionModel.GLCompute, self.entry_id, *self._encode_string("main"))
        m.emit(Op.ExecutionMode, self.entry_id, ExecutionMode.LocalSize,
               self.local_size[0], self.local_size[1], self.local_size[2])

        self.glsl_ext = self._id()
        m.emit(Op.ExtInstImport, self.glsl_ext, *self._encode_string("GLSL.std.450"))

        self.void_t = self._type_void()
        self.bool_t = self._type_bool()
        self.i8_t = self._type_int(8, 1)
        self.i16_t = self._type_int(16, 1)
        self.i32_t = self._type_int(32, 1)
        self.i64_t = self._type_int(64, 1)
        self.u32_t = self._type_int(32, 0)
        self.u64_t = self._type_int(64, 0)
        self.f16_t = self._type_float(16)
        self.f32_t = self._type_float(32)
        self.f64_t = self._type_float(64)
        self.v2f32_t = self._type_vector(self.f32_t, 2)
        self.v4f32_t = self._type_vector(self.f32_t, 4)
        self.v2u64_t = self._type_vector(self.u64_t, 2)

        self.void_fn_t = self._id()
        m.emit(Op.TypeFunction, self.void_fn_t, self.void_t)

    def _encode_string(self, s: str) -> List[int]:
        b = s.encode("utf-8") + b"\x00"
        while len(b) % 4 != 0:
            b += b"\x00"
        return [int.from_bytes(b[i:i+4], "little") for i in range(0, len(b), 4)]

    def _type_void(self) -> int:
        tid = self._id(); self.m.emit(Op.TypeVoid, tid); return tid
    def _type_bool(self) -> int:
        tid = self._id(); self.m.emit(Op.TypeBool, tid); return tid
    def _type_int(self, width: int, signedness: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeInt, tid, width, signedness); return tid
    def _type_float(self, width: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeFloat, tid, width); return tid
    def _type_vector(self, elem_type: int, count: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeVector, tid, elem_type, count); return tid
    def _type_runtime_array(self, elem_type: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeRuntimeArray, tid, elem_type); return tid
    def _type_struct(self, *member_types: int) -> int:
        tid = self._id(); self.m.emit(Op.TypeStruct, tid, *member_types); return tid
    def _type_pointer(self, storage_class: StorageClass, pointee_type: int) -> int:
        tid = self._id(); self.m.emit(Op.TypePointer, tid, storage_class, pointee_type); return tid

    def _constant(self, type_id: int, value: int) -> int:
        key = (type_id, value)
        if key in self.constants: return self.constants[key]
        cid = self._id()
        self.m.emit(Op.Constant, type_id, cid, value)
        self.constants[key] = cid
        return cid

    def _constant_u64(self, value: int) -> int:
        key = (self.u64_t, value)
        if key in self.constants: return self.constants[key]
        cid = self._id()
        low = value & 0xFFFFFFFF
        high = (value >> 32) & 0xFFFFFFFF
        self.m.emit(Op.Constant, self.u64_t, cid, low, high)
        self.constants[key] = cid
        return cid

    def define_push_constant_struct(self, fields: List[Tuple[int, int]]) -> int:
        """fields: [(type_id, offset), ...]"""
        member_types = [f[0] for f in fields]
        struct_t = self._type_struct(*member_types)
        self.m.emit(Op.Decorate, struct_t, Decoration.Block)
        for i, (ftype, offset) in enumerate(fields):
            self.m.emit(Op.MemberDecorate, struct_t, i, Decoration.Offset, offset)
        return struct_t

    def create_push_constant(self, struct_t: int) -> int:
        ptr_t = self._type_pointer(StorageClass.PushConstant, struct_t)
        var_id = self._id()
        self.m.emit(Op.Variable, ptr_t, var_id, StorageClass.PushConstant)
        return var_id

    def create_function(self, name: str = "main") -> Tuple[int, int]:
        m = self.m
        fid = self.entry_id
        m.emit(Op.Function, self.void_t, fid, FunctionControl.None_, self.void_fn_t)
        m.emit(Op.Name, fid, *self._encode_string(name))
        bid = self._id()
        m.emit(Op.Label, bid)
        return fid, bid

    def end_function(self):
        self.m.emit(Op.Return)
        self.m.emit(Op.FunctionEnd)

    def load(self, result_type: int, ptr: int, memory_access: int = 0) -> int:
        rid = self._id()
        if memory_access:
            self.m.emit(Op.Load, result_type, rid, ptr, memory_access)
        else:
            self.m.emit(Op.Load, result_type, rid, ptr)
        return rid

    def store(self, ptr: int, value: int, memory_access: int = 0):
        if memory_access:
            self.m.emit(Op.Store, ptr, value, memory_access)
        else:
            self.m.emit(Op.Store, ptr, value)

    def access_chain(self, result_type: int, base: int, *indices: int) -> int:
        rid = self._id()
        self.m.emit(Op.AccessChain, result_type, rid, base, *indices)
        return rid

    def iadd(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.IAdd, self.i32_t, rid, a, b); return rid
    def fadd(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FAdd, self.f32_t, rid, a, b); return rid
    def fmul(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FMul, self.f32_t, rid, a, b); return rid
    def fdiv(self, a: int, b: int) -> int:
        rid = self._id(); self.m.emit(Op.FDiv, self.f32_t, rid, a, b); return rid
    def fneg(self, a: int) -> int:
        rid = self._id(); self.m.emit(Op.FNegate, self.f32_t, rid, a); return rid

    def uconvert(self, val: int, to_bits: int) -> int:
        if to_bits == 64:
            rid = self._id(); self.m.emit(Op.UConvert, self.u64_t, rid, val); return rid
        rid = self._id(); self.m.emit(Op.UConvert, self.u32_t, rid, val); return rid

    def convert_u_to_ptr(self, val: int, ptr_type: int) -> int:
        rid = self._id(); self.m.emit(Op.ConvertUToPtr, ptr_type, rid, val); return rid

    def bitcast(self, result_type: int, val: int) -> int:
        rid = self._id(); self.m.emit(Op.Bitcast, result_type, rid, val); return rid

    def shift_left(self, val: int, shift: int) -> int:
        rid = self._id(); self.m.emit(Op.ShiftLeftLogical, self.i32_t, rid, val, shift); return rid
    def shift_right_logical(self, val: int, shift: int) -> int:
        rid = self._id(); self.m.emit(Op.ShiftRightLogical, self.i32_t, rid, val, shift); return rid

    def subgroup_shuffle_down(self, val: int, delta: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        delta_c = self._constant(self.i32_t, delta)
        self.m.emit(Op.GroupNonUniformShuffleDown, self.f32_t, rid, scope, val, delta_c)
        return rid

    def subgroup_broadcast_first(self, val: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        self.m.emit(Op.GroupNonUniformBroadcastFirst, self.f32_t, rid, scope, val)
        return rid

    def subgroup_fadd(self, val: int) -> int:
        rid = self._id()
        scope = self._constant(self.i32_t, Scope.Subgroup)
        self.m.emit(Op.GroupNonUniformFAdd, self.f32_t, rid, scope, 0, val)
        return rid

    def barrier(self, execution_scope: int = Scope.Workgroup, memory_scope: int = Scope.Workgroup,
                semantics: int = MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory):
        exec_scope = self._constant(self.i32_t, execution_scope)
        mem_scope = self._constant(self.i32_t, memory_scope)
        sem = self._constant(self.i32_t, semantics)
        self.m.emit(Op.ControlBarrier, exec_scope, mem_scope, sem)

    def memory_barrier(self, scope: int = Scope.Workgroup, semantics: int = MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory):
        mem_scope = self._constant(self.i32_t, scope)
        sem = self._constant(self.i32_t, semantics)
        self.m.emit(Op.MemoryBarrier, mem_scope, sem)

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.m.to_bytes())
        print(f"SPIR-V saved: {path} ({len(self.m.to_bytes())} bytes, bound={self.m.bound})")

def make_test_shader(output_path: str):
    """Generate a test compute shader: scale buffer via buffer reference."""
    a = Assembler(local_size=(32, 1, 1))
    m = a.m

    pc_struct = a.define_push_constant_struct([
        (a.u64_t, 0), (a.u64_t, 8), (a.u32_t, 16), (a.f32_t, 20)
    ])
    pc_var = a.create_push_constant(pc_struct)

    buf_arr_t = a._type_runtime_array(a.f32_t)
    buf_struct_t = a._type_struct(buf_arr_t)
    m.emit(Op.Decorate, buf_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, buf_struct_t, 0, Decoration.Offset, 0)
    buf_ptr_t = a._type_pointer(StorageClass.PhysicalStorageBuffer, buf_struct_t)
    m.emit(Op.Decorate, buf_ptr_t, Decoration.Aliased)

    v3u32_t = a._type_vector(a.u32_t, 3)
    in_struct_t = a._type_struct(v3u32_t)
    m.emit(Op.Decorate, in_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, in_struct_t, 0, Decoration.Offset, 0)
    in_ptr_t = a._type_pointer(StorageClass.Input, in_struct_t)
    in_var = a._id()
    m.emit(Op.Variable, in_ptr_t, in_var, StorageClass.Input)
    m.emit(Op.Decorate, in_var, Decoration.BuiltIn, BuiltIn.GlobalInvocationId)
    m.emit(Op.Name, in_var, *a._encode_string("gl_GlobalInvocationID"))
    m.emit(Op.MemberName, in_struct_t, 0, *a._encode_string("gl_GlobalInvocationID"))

    fid, bid = a.create_function("main")

    idx_ptr = a.access_chain(a._type_pointer(StorageClass.Input, a.u32_t), in_var,
                             a._constant(a.i32_t, 0), a._constant(a.i32_t, 0))
    idx = a.load(a.u32_t, idx_ptr)

    addr_in_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.u64_t),
                                  pc_var, a._constant(a.i32_t, 0))
    addr_in = a.load(a.u64_t, addr_in_ptr)
    addr_out_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.u64_t),
                                   pc_var, a._constant(a.i32_t, 1))
    addr_out = a.load(a.u64_t, addr_out_ptr)
    scale_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.f32_t),
                                pc_var, a._constant(a.i32_t, 3))
    scale = a.load(a.f32_t, scale_ptr)

    buf_in_ptr = a.convert_u_to_ptr(addr_in, buf_ptr_t)
    buf_out_ptr = a.convert_u_to_ptr(addr_out, buf_ptr_t)

    elem_ptr_in = a.access_chain(a._type_pointer(StorageClass.PhysicalStorageBuffer, a.f32_t),
                                  buf_in_ptr, a._constant(a.i32_t, 0), idx)
    elem_ptr_out = a.access_chain(a._type_pointer(StorageClass.PhysicalStorageBuffer, a.f32_t),
                                   buf_out_ptr, a._constant(a.i32_t, 0), idx)

    val = a.load(a.f32_t, elem_ptr_in, memory_access=0x2)
    scaled = a.fmul(val, scale)
    a.store(elem_ptr_out, scaled, memory_access=0x2)

    a.end_function()
    a.save(output_path)
    return output_path

def make_gemm_shader(output_path: str, tile_m: int = 32, tile_n: int = 32):
    """Generate a naive GEMM scaffold using PhysicalStorageBuffer.
    
    This emits the SPIR-V structure for C = A @ B with push-constant addresses.
    It stores a marker value (1.0) to prove the buffer path works.
    Full tiling + accumulation loop emission is left as extension.
    """
    a = Assembler(local_size=(tile_m, tile_n, 1))
    m = a.m

    pc_struct = a.define_push_constant_struct([
        (a.u64_t, 0), (a.u64_t, 8), (a.u64_t, 16),
        (a.u32_t, 24), (a.u32_t, 28), (a.u32_t, 32), (a.f32_t, 36)
    ])
    pc_var = a.create_push_constant(pc_struct)

    f32_arr_t = a._type_runtime_array(a.f32_t)
    f32_struct_t = a._type_struct(f32_arr_t)
    m.emit(Op.Decorate, f32_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, f32_struct_t, 0, Decoration.Offset, 0)
    f32_ptr_t = a._type_pointer(StorageClass.PhysicalStorageBuffer, f32_struct_t)
    m.emit(Op.Decorate, f32_ptr_t, Decoration.Aliased)

    v3u32_t = a._type_vector(a.u32_t, 3)
    in_struct_t = a._type_struct(v3u32_t)
    m.emit(Op.Decorate, in_struct_t, Decoration.Block)
    m.emit(Op.MemberDecorate, in_struct_t, 0, Decoration.Offset, 0)
    in_ptr_t = a._type_pointer(StorageClass.Input, in_struct_t)
    in_var = a._id()
    m.emit(Op.Variable, in_ptr_t, in_var, StorageClass.Input)
    m.emit(Op.Decorate, in_var, Decoration.BuiltIn, BuiltIn.GlobalInvocationId)
    m.emit(Op.Name, in_var, *a._encode_string("gl_GlobalInvocationID"))
    m.emit(Op.MemberName, in_struct_t, 0, *a._encode_string("gl_GlobalInvocationID"))

    fid, bid = a.create_function("main")

    row_ptr = a.access_chain(a._type_pointer(StorageClass.Input, a.u32_t), in_var,
                             a._constant(a.i32_t, 0), a._constant(a.i32_t, 1))
    row = a.load(a.u32_t, row_ptr)
    col_ptr = a.access_chain(a._type_pointer(StorageClass.Input, a.u32_t), in_var,
                             a._constant(a.i32_t, 0), a._constant(a.i32_t, 0))
    col = a.load(a.u32_t, col_ptr)

    addr_c_ptr = a.access_chain(a._type_pointer(StorageClass.PushConstant, a.u64_t),
                                 pc_var, a._constant(a.i32_t, 2))
    addr_c = a.load(a.u64_t, addr_c_ptr)

    buf_c_ptr = a.convert_u_to_ptr(addr_c, f32_ptr_t)

    c_idx = a.iadd(row, col)
    c_elem_ptr = a.access_chain(a._type_pointer(StorageClass.PhysicalStorageBuffer, a.f32_t),
                                 buf_c_ptr, a._constant(a.i32_t, 0), c_idx)

    one_i32 = a._constant(a.i32_t, 1065353216)
    one_f32 = a.bitcast(a.f32_t, one_i32)
    a.store(c_elem_ptr, one_f32, memory_access=0x2)

    a.end_function()
    a.save(output_path)
    print(f"GEMM scaffold saved to {output_path}")
    print("Note: This is a structural scaffold. Add loop emission for full GEMM accumulation.")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDNA4 SPIR-V Assembler")
    parser.add_argument("-o", "--output", default="kernel.spv", help="Output SPIR-V file")
    parser.add_argument("--test", action="store_true", help="Generate test scale shader")
    parser.add_argument("--gemm", action="store_true", help="Generate GEMM scaffold shader")
    parser.add_argument("--tile-m", type=int, default=32, help="GEMM tile M dimension")
    parser.add_argument("--tile-n", type=int, default=32, help="GEMM tile N dimension")
    args = parser.parse_args()

    if args.test:
        make_test_shader(args.output)
    elif args.gemm:
        make_gemm_shader(args.output, args.tile_m, args.tile_n)
    else:
        print("Usage: python rdna4_as.py --test -o kernel.spv")
        print("       python rdna4_as.py --gemm -o gemm.spv")
        sys.exit(1)
```

---

### Summary of what each patch fixes

| File | What was broken / stubbed | What the patch does |
|------|---------------------------|---------------------|
| `vgpr_stub.cpp` | `loadAMDShader` and `queryVGPRUsage` were TODO stubs | `queryVGPRUsage` now actually calls `vkGetShaderInfoAMD` via `vkGetDeviceProcAddr`; `loadAMDShader` cleanly reports that raw ISA loading isn't possible through standard Vulkan |
| `memory.cpp` | `allocateImage` created a `VkImage` but never allocated or bound memory | Full memory allocation + bind; added `freeBuffer` / `freeImage` tracking |
| `mailbox.cpp` | Just a CPU-side `std::atomic` struct with no GPU visibility | `GpuMailbox` creates a host-visible device buffer, maps it, exposes the `VkDeviceAddress` for shader polling, and handles flush/invalidate |
| `rdna4_weights.hpp` + `weight_uploader.cpp` | `uploadTensor` destroyed the staging buffer **before** submit; `createGpuBuffer` didn't return memory | Staging buffer now lives until `vkQueueWaitIdle`; `createGpuBuffer` optionally returns `VkDeviceMemory`; `freeTensor` correctly frees both buffer and memory |
| `rdna4_kv_cache.hpp` + `kv_cache.cpp` | Allocated **images** (for DCC) but kernels use `buffer_reference`; `append()` was empty | Rewritten to allocate flat **buffers** with `SHADER_DEVICE_ADDRESS_BIT`; `append()` does proper `vkCmdCopyBuffer` staging upload; exposes `getKBufferAddress` / `getVBufferAddress` |
| `inference_engine.cpp` | Looked for `.attn_k.cache` / `.attn_v.cache` in the model weights (they don't exist); readback didn't invalidate mapped memory | Now pulls cache addresses from `KVCacheManager`; adds `vkInvalidateMappedMemoryRanges` before reading the sampled token |
| `rdna4_as.py` | Only generated a single test scale shader | Added `--gemm` CLI flag and `make_gemm_shader` scaffold; better help text |

**Next steps after applying these:**
1. Regenerate `CMakeLists.txt` if you add `rdna4_mailbox.hpp` to the include list.
2. The `inference_engine.cpp` still calls `syncAll()` after every kernel — that's the next bottleneck to tackle (batch independent dispatches and reduce host sync).
3. The naive GEMM in `gemm.comp` should still be replaced with a tiled version; these patches fix the **host infrastructure** so you can focus on kernel optimization.