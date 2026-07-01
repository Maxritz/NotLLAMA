#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include "vulkan/vk_device.h"

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct PipelineBundle {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

class VulkanBackend {
public:
    DeviceInfo dev;

    // Command resources
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool batch_mode = false;
    VkCommandBuffer batch_cmd = VK_NULL_HANDLE;

    // Descriptor pool
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_3buf_layout = VK_NULL_HANDLE; // 3 SSBOs
    VkDescriptorSetLayout desc_2buf_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_1buf_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_4buf_layout = VK_NULL_HANDLE;

    // Pre-allocated descriptor sets (24 each to prevent overwrite in batch mode)
    static constexpr int DESC_POOL_COUNT = 24;
    VkDescriptorSet desc_3buf_set[DESC_POOL_COUNT] = {};
    VkDescriptorSet desc_2buf_set[DESC_POOL_COUNT] = {};
    VkDescriptorSet desc_1buf_set[DESC_POOL_COUNT] = {};
    VkDescriptorSet desc_4buf_set[DESC_POOL_COUNT] = {};
    int desc_idx = 0;

    // Shader pipelines
    PipelineBundle gemm_fp32_pipe;
    PipelineBundle gemm_q4_0_pipe;
    PipelineBundle gemm_q8_0_pipe;
    PipelineBundle gemm_wmma_pipe;     // requires GL_KHR_cooperative_matrix
    PipelineBundle convert_f32_f16_pipe;
    PipelineBundle rms_norm_pipe;
    PipelineBundle rope_pipe;
    PipelineBundle silu_mul_pipe;
    PipelineBundle token_embed_pipe;
    PipelineBundle add_pipe;
    PipelineBundle attn_scores_pipe;   // scalar fallback if WMMA unavailable
    PipelineBundle attn_wmma_pipe;     // WMMA attention scores
    PipelineBundle attn_softmax_pipe;
    PipelineBundle attn_value_pipe;
    PipelineBundle argmax_pipe;

    // Staging pool for host<->device transfers
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkDeviceSize staging_size = 0;
    VkDeviceSize staging_offset = 0;

    bool init(VkInstance instance);
    bool load_pipelines(const char* spv_dir);
    void cleanup();

    // Buffer management
    GpuBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags mem_props, bool mapped = false);
    void destroy_buffer(const GpuBuffer& buf);
    void upload_to_buffer(const GpuBuffer& dst, const void* data, VkDeviceSize size);
    void download_from_buffer(const GpuBuffer& src, void* data, VkDeviceSize size);
    void copy_buffer(const GpuBuffer& src, const GpuBuffer& dst, VkDeviceSize size);

    // Dispatch helpers
    void begin_batch();
    void end_batch();
    VkCommandBuffer current_cmd();
    void push_constants(VkPipelineLayout layout, const void* data, uint32_t size);
    void bind_descriptors(VkPipelineLayout layout, VkDescriptorSet set);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void barrier();

    // Descriptor helpers
    void update_desc_3buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1, VkBuffer b2,
                          VkDeviceSize off0 = 0, VkDeviceSize off1 = 0, VkDeviceSize off2 = 0);
    void update_desc_2buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1);
    void update_desc_1buf(VkDescriptorSet set, VkBuffer b0);
    void update_desc_4buf(VkDescriptorSet set, VkBuffer b0, VkBuffer b1, VkBuffer b2, VkBuffer b3);

    // GEMM: auto-selects WMMA or scalar
    void gemm(const GpuBuffer& A, const GpuBuffer& B, const GpuBuffer& C,
              uint32_t M, uint32_t N, uint32_t K,
              bool A_is_fp16 = false, bool B_is_fp16 = false);

    // Quantized GEMM: B is Q4_0 or Q8_0, num_blocks = ceil(K / 32)
    void gemm_q4_0(const GpuBuffer& A, const GpuBuffer& B_q, const GpuBuffer& C,
                   uint32_t M, uint32_t N, uint32_t K, uint32_t num_blocks);
    void gemm_q8_0(const GpuBuffer& A, const GpuBuffer& B_q, const GpuBuffer& C,
                   uint32_t M, uint32_t N, uint32_t K, uint32_t num_blocks);

    // Kernels
    void add(const GpuBuffer& a, const GpuBuffer& b, const GpuBuffer& c, uint32_t count);
    void rms_norm(const GpuBuffer& input, const GpuBuffer& weight, const GpuBuffer& output,
                  uint32_t dim, float eps, uint32_t rows = 1);
    void rope(const GpuBuffer& data, uint32_t head_dim, uint32_t num_heads,
              uint32_t position, float theta_base);
    void silu_mul(const GpuBuffer& gate, const GpuBuffer& up, const GpuBuffer& output, uint32_t count);
    void token_embedding(const GpuBuffer& table, const GpuBuffer& output,
                         uint32_t token_id, uint32_t embed_dim, uint32_t vocab_size, float scale);
    void attention_scores_wmma(const GpuBuffer& Q, const GpuBuffer& K_cache,
                               const GpuBuffer& scores, uint32_t head_dim,
                               uint32_t seq_len, uint32_t max_seq, float scale,
                               uint32_t num_q_heads, uint32_t gqa_ratio, uint32_t off_q);
    void attention_softmax(const GpuBuffer& scores, uint32_t seq_len,
                           uint32_t max_seq, uint32_t num_q_heads);
    void attention_value(const GpuBuffer& scores, const GpuBuffer& V_cache,
                         const GpuBuffer& output, uint32_t head_dim,
                         uint32_t seq_len, uint32_t max_seq,
                         uint32_t num_q_heads, uint32_t gqa_ratio);

    // GPU argmax: finds index of max value in logits buffer
    void gpu_argmax(const GpuBuffer& logits, const GpuBuffer& result, uint32_t vocab_size);

    // Exposed for testing
    bool load_shader(const char* name, const uint32_t* spirv, size_t size, PipelineBundle& out);
    bool create_descriptor_resources();
    bool create_pipeline(const uint32_t* spirv, size_t size,
                         VkDescriptorSetLayout desc_layout, uint32_t push_size,
                         PipelineBundle& out, uint32_t required_subgroup_size = 0);
    GpuBuffer create_fp16_copy(const GpuBuffer& src_f32, uint32_t count);
};
