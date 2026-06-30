#pragma once
#include "engine/imemory_allocator.hpp"
#include "rdna4_allocator.hpp"
#include <memory>
#include <vector>

namespace notllama {

// Bridge adapter: maps IMemoryAllocator onto three rdna4::RingAllocator instances.
// This is a temporary implementation; replace with a true slab allocator later.
class RingAllocatorAdapter : public IMemoryAllocator {
public:
    RingAllocatorAdapter(VkDevice device, VkPhysicalDevice physical_device,
                         size_t weight_size, size_t transient_size, size_t kv_size);
    ~RingAllocatorAdapter() override;

    GpuAllocation Allocate(MemoryType type, size_t size, size_t alignment) override;
    void Free(MemoryType type, const GpuAllocation& alloc) override;

    void* Map(const GpuAllocation& alloc) override;
    void Unmap(const GpuAllocation& alloc) override;

    size_t GetNonCoherentAtomSize() const override;

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;

    std::unique_ptr<rdna4::RingAllocator> weight_;
    std::unique_ptr<rdna4::RingAllocator> transient_;
    std::unique_ptr<rdna4::RingAllocator> kv_;

    rdna4::RingAllocator* GetAllocator(MemoryType type);
};

} // namespace notllama
