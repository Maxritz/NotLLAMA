#include "engine/ring_allocator_adapter.hpp"
#include <stdexcept>

namespace notllama {

RingAllocatorAdapter::RingAllocatorAdapter(VkDevice device, VkPhysicalDevice physical_device,
                                           size_t weight_size, size_t transient_size, size_t kv_size)
    : device_(device), physical_device_(physical_device) {
    weight_ = std::make_unique<rdna4::RingAllocator>(device_, physical_device_, weight_size);
    transient_ = std::make_unique<rdna4::RingAllocator>(device_, physical_device_, transient_size);
    kv_ = std::make_unique<rdna4::RingAllocator>(device_, physical_device_, kv_size);
}

RingAllocatorAdapter::~RingAllocatorAdapter() = default;

rdna4::RingAllocator* RingAllocatorAdapter::GetAllocator(MemoryType type) {
    switch (type) {
        case MemoryType::WEIGHT: return weight_.get();
        case MemoryType::TRANSIENT: return transient_.get();
        case MemoryType::KV_CACHE: return kv_.get();
    }
    return nullptr;
}

GpuAllocation RingAllocatorAdapter::Allocate(MemoryType type, size_t size, size_t alignment) {
    auto* alloc = GetAllocator(type);
    if (!alloc) return {};

    uint64_t addr = alloc->alloc(size, alignment);
    if (addr == 0) return {};

    GpuAllocation result{};
    result.buffer = alloc->buffer;
    result.offset = static_cast<VkDeviceSize>(addr - alloc->baseAddress);
    result.size = size;
    result.slab_id = static_cast<uint32_t>(type);
    result.device_address = addr;
    return result;
}

void RingAllocatorAdapter::Free(MemoryType type, const GpuAllocation& alloc) {
    // Ring allocator uses a circular buffer pattern - allocations are
    // overwritten when the ring wraps around. Individual frees are not
    // supported. Call Reset(type) to clear all allocations of a given type.
    (void)type;
    (void)alloc;
}

void RingAllocatorAdapter::Reset(MemoryType type) {
    auto* alloc = GetAllocator(type);
    if (alloc) alloc->reset();
}

void* RingAllocatorAdapter::Map(const GpuAllocation& alloc) {
    auto* a = GetAllocator(static_cast<MemoryType>(alloc.slab_id));
    if (!a || !a->mappedPtr) return nullptr;
    return a->mappedPtr + alloc.offset;
}

void RingAllocatorAdapter::Unmap(const GpuAllocation& alloc) {
    (void)alloc;
    // Host-coherent mapping; no explicit flush/invalidate needed here.
}

size_t RingAllocatorAdapter::GetNonCoherentAtomSize() const {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    return props.limits.nonCoherentAtomSize;
}

} // namespace notllama
