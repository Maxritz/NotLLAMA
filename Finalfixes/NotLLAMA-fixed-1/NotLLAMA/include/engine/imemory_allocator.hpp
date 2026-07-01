#pragma once
#include "types.hpp"
#include <vulkan/vulkan.h>
#include <cstddef>

namespace notllama {

class IMemoryAllocator {
public:
    virtual ~IMemoryAllocator() = default;

    virtual GpuAllocation Allocate(MemoryType type, size_t size, size_t alignment) = 0;
    virtual void Free(MemoryType type, const GpuAllocation& alloc) = 0;

    virtual void* Map(const GpuAllocation& alloc) = 0;
    virtual void Unmap(const GpuAllocation& alloc) = 0;

    virtual size_t GetNonCoherentAtomSize() const = 0;
};

} // namespace notllama
