#pragma once
#include "types.hpp"
#include <cstddef>
#include <cstdint>

namespace notllama {

class IQuantization {
public:
    virtual ~IQuantization() = default;

    virtual DataType GetType() = 0;
    virtual size_t PackedBytesPerElement() = 0;
    virtual void CompressTile(const float* src, uint8_t* dst, size_t tile_size) = 0;

    virtual uint32_t GetShaderSpecializationID() = 0;
    virtual size_t GetBlockAlignment() = 0;
    virtual bool ValidateBlockAlignment(const void* data) = 0;
};

} // namespace notllama
