#pragma once
#include "engine/iquantization.hpp"
#include <memory>

namespace notllama {

// Factory for GGUF-style quantization adapters.
// Each adapter reports packed block size, element count per block, shader
// specialization ID, and can compress a float tile to the GPU block layout.
std::unique_ptr<IQuantization> CreateQuantization(DataType type);

} // namespace notllama
