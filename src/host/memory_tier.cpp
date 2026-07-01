#include "memory_tier.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>

namespace notllama {

// ---- Aligned CPU allocation ----
static void* AlignedAlloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

static void AlignedFree(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ---- Constructor / Destructor ----

MemoryTierManager::MemoryTierManager(VkDevice device,
                                     VkPhysicalDevice physical_device,
                                     uint32_t queue_family_index)
    : device_(device), physical_device_(physical_device),
      queue_family_index_(queue_family_index) {
    AutoDetectVRAMBudget();
}

MemoryTierManager::~MemoryTierManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& buf : buffers_) {
        if (buf->gpu_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buf->gpu_buffer, nullptr);
        }
        if (buf->gpu_memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buf->gpu_memory, nullptr);
        }
        if (buf->cpu_data) {
            AlignedFree(buf->cpu_data);
        }
        if (!config_.spill_directory.empty()) {
            auto path = GetSpillPath(buf->name);
            if (std::filesystem::exists(path)) {
                std::filesystem::remove(path);
            }
        }
    }
    buffers_.clear();
}

void MemoryTierManager::AutoDetectVRAMBudget() {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += mem_props.memoryHeaps[i].size;
        }
    }
    config_.vram_budget_bytes = static_cast<size_t>(total);
    // Reserve 10% for OS/driver
    config_.vram_budget_bytes = config_.vram_budget_bytes * 9 / 10;

    fprintf(stderr, "[MemoryTier] VRAM budget: %.1f GB\n",
            config_.vram_budget_bytes / (1024.0 * 1024.0 * 1024.0));
}

void MemoryTierManager::Configure(const TierConfig& cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = cfg;
    if (config_.vram_budget_bytes == 0) {
        AutoDetectVRAMBudget();
    }
    if (!config_.spill_directory.empty()) {
        std::filesystem::create_directories(config_.spill_directory);
    }
    fprintf(stderr, "[MemoryTier] Configured: gpu_layers=%d split_mode=%s spill=%s\n",
            config_.gpu_layers, config_.split_mode.c_str(),
            config_.spill_directory.empty() ? "none" : config_.spill_directory.c_str());
}

// ---- Buffer Allocation ----

TieredBuffer* MemoryTierManager::AllocateBuffer(const std::string& name,
                                                  size_t size_bytes,
                                                  uint32_t layer_index,
                                                  bool is_layer) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already exists
    auto it = name_map_.find(name);
    if (it != name_map_.end()) return it->second;

    auto buf = std::make_unique<TieredBuffer>();
    buf->name = name;
    buf->layer_index = layer_index;
    buf->size_bytes = size_bytes;
    buf->current_tier = MemoryTier::SYSTEM_RAM;
    buf->is_layer_weight = is_layer;

    buf->cpu_data = AlignedAlloc(size_bytes, 64);
    if (!buf->cpu_data) {
        fprintf(stderr, "[MemoryTier] FAILED to allocate %zu bytes RAM for %s\n",
                size_bytes, name.c_str());
        return nullptr;
    }
    buf->cpu_capacity = size_bytes;
    ram_used_ += size_bytes;

    TieredBuffer* ptr = buf.get();
    name_map_[name] = ptr;
    layer_map_[layer_index].push_back(ptr);
    buffers_.push_back(std::move(buf));

    return ptr;
}

bool MemoryTierManager::WriteCPUData(TieredBuffer* buf, const void* data, size_t size) {
    if (!buf || !buf->cpu_data || !data) return false;
    size_t to_copy = (size < buf->cpu_capacity) ? size : buf->cpu_capacity;
    std::memcpy(buf->cpu_data, data, to_copy);
    return true;
}

// ---- GPU Upload (Promote to VRAM) ----

bool MemoryTierManager::PromoteToVRAM(TieredBuffer* buf) {
    if (!buf) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    if (buf->current_tier == MemoryTier::VRAM && buf->gpu_address != 0) {
        return true;
    }

    // Restore from disk if needed
    if (buf->current_tier == MemoryTier::DISK) {
        if (!RestoreFromDisk(buf)) return false;
    }

    // Check budget
    if (buf->size_bytes > GetVRAMAvailable()) {
        fprintf(stderr, "[MemoryTier] VRAM full: need %zu MB for '%s', have %zu MB\n",
                buf->size_bytes / (1024*1024), buf->name.c_str(),
                GetVRAMAvailable() / (1024*1024));
        return false;
    }

    // Create GPU buffer
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = buf->size_bytes;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer gpu_buf = VK_NULL_HANDLE;
    VkResult r = vkCreateBuffer(device_, &buf_info, nullptr, &gpu_buf);
    if (r != VK_SUCCESS) return false;

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device_, gpu_buf, &mem_req);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(device_, gpu_buf, nullptr);
        return false;
    }

    VkMemoryAllocateFlagsInfo flags = {};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.pNext = &flags;
    alloc.allocationSize = mem_req.size;
    alloc.memoryTypeIndex = mem_type;

    VkDeviceMemory gpu_mem = VK_NULL_HANDLE;
    r = vkAllocateMemory(device_, &alloc, nullptr, &gpu_mem);
    if (r != VK_SUCCESS) {
        vkDestroyBuffer(device_, gpu_buf, nullptr);
        return false;
    }

    r = vkBindBufferMemory(device_, gpu_buf, gpu_mem, 0);
    if (r != VK_SUCCESS) {
        vkFreeMemory(device_, gpu_mem, nullptr);
        vkDestroyBuffer(device_, gpu_buf, nullptr);
        return false;
    }

    // Upload via staging buffer
    if (buf->cpu_data && buf->size_bytes > 0) {
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_index_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        vkCreateCommandPool(device_, &pool_info, nullptr, &cmd_pool);

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cmd_alloc = {};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = cmd_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd);

        // Staging buffer (host-visible)
        VkBufferCreateInfo staging_info = {};
        staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_info.size = buf->size_bytes;
        staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging = VK_NULL_HANDLE;
        vkCreateBuffer(device_, &staging_info, nullptr, &staging);

        VkMemoryRequirements staging_req;
        vkGetBufferMemoryRequirements(device_, staging, &staging_req);

        uint32_t staging_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((staging_req.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                staging_type = i;
                break;
            }
        }

        if (staging_type != UINT32_MAX) {
            VkMemoryAllocateInfo staging_alloc = {};
            staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            staging_alloc.allocationSize = staging_req.size;
            staging_alloc.memoryTypeIndex = staging_type;

            VkDeviceMemory staging_mem = VK_NULL_HANDLE;
            vkAllocateMemory(device_, &staging_alloc, nullptr, &staging_mem);
            vkBindBufferMemory(device_, staging, staging_mem, 0);

            void* mapped = nullptr;
            vkMapMemory(device_, staging_mem, 0, buf->size_bytes, 0, &mapped);
            if (mapped) {
                std::memcpy(mapped, buf->cpu_data, buf->size_bytes);
                vkUnmapMemory(device_, staging_mem);
            }

            VkCommandBufferBeginInfo begin = {};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin);

            VkBufferCopy copy = {};
            copy.size = buf->size_bytes;
            vkCmdCopyBuffer(cmd, staging, gpu_buf, 1, &copy);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo submit = {};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;

            VkQueue queue = VK_NULL_HANDLE;
            vkGetDeviceQueue(device_, queue_family_index_, 0, &queue);
            vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);

            vkFreeMemory(device_, staging_mem, nullptr);
            vkDestroyBuffer(device_, staging, nullptr);
        }

        vkFreeCommandBuffers(device_, cmd_pool, 1, &cmd);
        vkDestroyCommandPool(device_, cmd_pool, nullptr);
    }

    // Get device address
    VkBufferDeviceAddressInfo addr_info = {};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = gpu_buf;
    buf->gpu_address = vkGetBufferDeviceAddress(device_, &addr_info);
    buf->gpu_buffer = gpu_buf;
    buf->gpu_memory = gpu_mem;
    buf->current_tier = MemoryTier::VRAM;
    vram_used_ += buf->size_bytes;

    fprintf(stderr, "[MemoryTier] '%s' -> VRAM (%.1f MB used)\n",
            buf->name.c_str(), vram_used_ / (1024.0 * 1024.0));
    return true;
}

// ---- GPU Download (Demote to RAM) ----

bool MemoryTierManager::DemoteToRAM(TieredBuffer* buf) {
    if (!buf) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    if (buf->current_tier != MemoryTier::VRAM) return true;

    // Download data from GPU
    if (buf->gpu_buffer != VK_NULL_HANDLE && buf->cpu_data) {
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_index_;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        vkCreateCommandPool(device_, &pool_info, nullptr, &cmd_pool);

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cmd_alloc = {};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = cmd_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd);

        // Readback staging buffer
        VkBufferCreateInfo staging_info = {};
        staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_info.size = buf->size_bytes;
        staging_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer staging = VK_NULL_HANDLE;
        vkCreateBuffer(device_, &staging_info, nullptr, &staging);

        VkMemoryRequirements staging_req;
        vkGetBufferMemoryRequirements(device_, staging, &staging_req);

        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

        uint32_t staging_type = UINT32_MAX;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((staging_req.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                staging_type = i;
                break;
            }
        }

        if (staging_type != UINT32_MAX) {
            VkMemoryAllocateInfo staging_alloc = {};
            staging_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            staging_alloc.allocationSize = staging_req.size;
            staging_alloc.memoryTypeIndex = staging_type;

            VkDeviceMemory staging_mem = VK_NULL_HANDLE;
            vkAllocateMemory(device_, &staging_alloc, nullptr, &staging_mem);
            vkBindBufferMemory(device_, staging, staging_mem, 0);

            VkCommandBufferBeginInfo begin = {};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin);

            VkBufferCopy copy = {};
            copy.size = buf->size_bytes;
            vkCmdCopyBuffer(cmd, buf->gpu_buffer, staging, 1, &copy);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo submit = {};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;

            VkQueue queue = VK_NULL_HANDLE;
            vkGetDeviceQueue(device_, queue_family_index_, 0, &queue);
            vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);

            void* mapped = nullptr;
            vkMapMemory(device_, staging_mem, 0, buf->size_bytes, 0, &mapped);
            if (mapped) {
                std::memcpy(buf->cpu_data, mapped, buf->size_bytes);
                vkUnmapMemory(device_, staging_mem);
            }

            vkFreeMemory(device_, staging_mem, nullptr);
            vkDestroyBuffer(device_, staging, nullptr);
        }

        vkFreeCommandBuffers(device_, cmd_pool, 1, &cmd);
        vkDestroyCommandPool(device_, cmd_pool, nullptr);
    }

    // Free GPU resources
    if (buf->gpu_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buf->gpu_buffer, nullptr);
        buf->gpu_buffer = VK_NULL_HANDLE;
    }
    if (buf->gpu_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, buf->gpu_memory, nullptr);
        buf->gpu_memory = VK_NULL_HANDLE;
    }
    buf->gpu_address = 0;
    buf->current_tier = MemoryTier::SYSTEM_RAM;
    vram_used_ -= buf->size_bytes;

    fprintf(stderr, "[MemoryTier] '%s' -> RAM (%.1f MB freed)\n",
            buf->name.c_str(), buf->size_bytes / (1024.0 * 1024.0));
    return true;
}

// ---- Disk Spill ----

bool MemoryTierManager::SpillToDisk(TieredBuffer* buf) {
    if (!buf || config_.spill_directory.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    if (buf->current_tier == MemoryTier::VRAM) {
        if (!DemoteToRAM(buf)) return false;
    }
    if (!buf->cpu_data) return false;

    auto path = GetSpillPath(buf->name);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    f.write(static_cast<const char*>(buf->cpu_data), static_cast<std::streamsize>(buf->size_bytes));
    if (!f.good()) return false;

    AlignedFree(buf->cpu_data);
    buf->cpu_data = nullptr;
    ram_used_ -= buf->size_bytes;
    buf->current_tier = MemoryTier::DISK;
    disk_used_ += buf->size_bytes;
    return true;
}

bool MemoryTierManager::RestoreFromDisk(TieredBuffer* buf) {
    if (!buf) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    if (buf->current_tier != MemoryTier::DISK) return true;

    auto path = GetSpillPath(buf->name);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);

    buf->cpu_data = AlignedAlloc(static_cast<size_t>(size), 64);
    if (!buf->cpu_data) return false;

    f.read(static_cast<char*>(buf->cpu_data), size);
    if (!f.good()) {
        AlignedFree(buf->cpu_data);
        buf->cpu_data = nullptr;
        return false;
    }

    buf->cpu_capacity = static_cast<size_t>(size);
    buf->current_tier = MemoryTier::SYSTEM_RAM;
    disk_used_ -= buf->size_bytes;
    ram_used_ += buf->size_bytes;

    std::filesystem::remove(path);
    return true;
}

// ---- Layer Management ----

bool MemoryTierManager::LoadLayerToVRAM(uint32_t layer_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_map_.find(layer_index);
    if (it == layer_map_.end()) return false;

    bool all_ok = true;
    for (auto* buf : it->second) {
        if (buf->current_tier != MemoryTier::VRAM) {
            if (!PromoteToVRAM(buf)) all_ok = false;
        }
    }
    return all_ok;
}

bool MemoryTierManager::OffloadLayerToRAM(uint32_t layer_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_map_.find(layer_index);
    if (it == layer_map_.end()) return false;

    for (auto* buf : it->second) {
        if (buf->current_tier == MemoryTier::VRAM) {
            DemoteToRAM(buf);
        }
    }
    return true;
}

bool MemoryTierManager::IsLayerInVRAM(uint32_t layer_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_map_.find(layer_index);
    if (it == layer_map_.end()) return false;
    for (auto* buf : it->second) {
        if (buf->current_tier != MemoryTier::VRAM) return false;
    }
    return !it->second.empty();
}

TieredBuffer* MemoryTierManager::GetLayerBuffer(const std::string& name, uint32_t layer_index) {
    (void)layer_index;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_map_.find(name);
    if (it == name_map_.end()) return nullptr;

    auto* buf = it->second;
    if (buf->current_tier == MemoryTier::DISK) {
        RestoreFromDisk(buf);
    }
    if (buf->current_tier == MemoryTier::SYSTEM_RAM && buf->size_bytes <= GetVRAMAvailable()) {
        PromoteToVRAM(buf);
    }
    return buf;
}

TieredBuffer* MemoryTierManager::FindBuffer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name_map_.find(name);
    return (it != name_map_.end()) ? it->second : nullptr;
}

void MemoryTierManager::FreeBuffer(TieredBuffer* buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lock(mutex_);

    if (buf->gpu_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buf->gpu_buffer, nullptr);
        vram_used_ -= buf->size_bytes;
    }
    if (buf->gpu_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, buf->gpu_memory, nullptr);
    }
    if (buf->cpu_data) {
        AlignedFree(buf->cpu_data);
        ram_used_ -= buf->size_bytes;
    }
    buf->gpu_buffer = VK_NULL_HANDLE;
    buf->gpu_memory = VK_NULL_HANDLE;
    buf->cpu_data = nullptr;
    buf->gpu_address = 0;
}

// ---- Stats ----

size_t MemoryTierManager::GetVRAMAvailable() const {
    if (vram_used_ >= config_.vram_budget_bytes) return 0;
    return config_.vram_budget_bytes - vram_used_;
}

std::string MemoryTierManager::GetStatsString() const {
    char buf[256];
    snprintf(buf, sizeof(buf),
        "VRAM: %.1f/%.1f GB (%.0f%%) | RAM: %.1f GB | Disk: %.1f MB | Buffers: %zu",
        vram_used_ / (1024.0 * 1024.0 * 1024.0),
        config_.vram_budget_bytes / (1024.0 * 1024.0 * 1024.0),
        100.0 * vram_used_ / (config_.vram_budget_bytes + 1),
        ram_used_ / (1024.0 * 1024.0 * 1024.0),
        disk_used_ / (1024.0 * 1024.0),
        buffers_.size());
    return std::string(buf);
}

std::vector<uint32_t> MemoryTierManager::ComputeVRAMLayout(
    size_t num_layers, const std::vector<size_t>& layer_sizes) {

    std::vector<uint32_t> vram_layers;
    size_t remaining = GetVRAMBudget();

    // Reserve for global weights and KV cache
    const size_t reserve = 300 * 1024 * 1024;
    if (remaining > reserve) remaining -= reserve;

    // gpu_layers override
    if (config_.gpu_layers >= 0) {
        uint32_t n = static_cast<uint32_t>(std::min(
            static_cast<size_t>(config_.gpu_layers), num_layers));
        for (uint32_t i = 0; i < n; i++) vram_layers.push_back(i);
        fprintf(stderr, "[MemoryTier] gpu_layers=%d: layers 0..%u in VRAM\n",
                config_.gpu_layers, n > 0 ? n - 1 : 0);
        return vram_layers;
    }

    // Greedy fit
    for (size_t i = 0; i < num_layers && i < layer_sizes.size(); i++) {
        if (layer_sizes[i] <= remaining) {
            vram_layers.push_back(static_cast<uint32_t>(i));
            remaining -= layer_sizes[i];
        } else {
            break;
        }
    }
    fprintf(stderr, "[MemoryTier] Auto-fit: %zu layers in VRAM (%.1f GB remaining)\n",
            vram_layers.size(), remaining / (1024.0 * 1024.0 * 1024.0));
    return vram_layers;
}

std::string MemoryTierManager::GetSpillPath(const std::string& name) {
    std::string sanitized = name;
    for (auto& c : sanitized) {
        if (c == '/' || c == '\\' || c == ':') c = '_';
    }
    return config_.spill_directory + "/" + sanitized + ".spill";
}

} // namespace notllama
