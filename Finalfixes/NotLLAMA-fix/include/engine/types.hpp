#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <array>

namespace notllama {

enum class DataType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    FP8_E4M3,
    FP8_E5M2,
    UNKNOWN
};

enum class MemoryType {
    WEIGHT,
    TRANSIENT,
    KV_CACHE
};

enum class PipelineVariant {
    FAST,
    SAFE
};

enum class AccumulatorType {
    FP16,
    FP32
};

enum class VendorID {
    AMD = 0x1002,
    NVIDIA = 0x10DE,
    INTEL = 0x8086,
    UNKNOWN = 0
};

struct VendorProfile {
    VendorID vendor = VendorID::UNKNOWN;
    uint32_t subgroup_size = 32;
    bool cooperative_matrix = false;
    bool wave32 = true;
    bool wave64 = false;
};

struct GpuAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    uint32_t slab_id = 0;
    VkDeviceAddress device_address = 0;
};

struct TensorMeta {
    std::array<uint32_t, 4> dims = {};
    uint32_t num_dims = 0;
    std::array<uint32_t, 4> strides = {};
    DataType dtype = DataType::F32;
    GpuAllocation alloc;
    uint32_t block_size = 0;
    GpuAllocation scales;
};

struct SpecializationMap {
    uint32_t tile_m = 64;
    uint32_t tile_n = 64;
    uint32_t tile_k = 16;
    uint32_t workgroup_x = 64;
    uint32_t workgroup_y = 1;
    uint32_t head_dim = 128;
    uint32_t subgroup_size = 32;
    AccumulatorType accum_type = AccumulatorType::FP32;
};

struct BatchMetadata {
    static constexpr uint32_t MAX_BATCH_SIZE = 64;
    static constexpr uint32_t MAX_BLOCKS = 256;

    uint32_t num_active_sequences = 0;

    struct SequenceInfo {
        uint32_t seq_id = 0;
        uint32_t length = 0;
        uint32_t num_blocks = 0;
        uint32_t position_offset = 0;
        std::array<uint32_t, MAX_BLOCKS> block_table = {};
    };

    std::array<SequenceInfo, MAX_BATCH_SIZE> seq_info = {};
};

struct FrameMetadata {
    uint32_t frame_index = 0;
    GpuAllocation metadata_buffer;
    TensorMeta input;
    TensorMeta output;
    TensorMeta weights;
};

} // namespace notllama
