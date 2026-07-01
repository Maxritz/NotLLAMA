#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <string>
#include "vulkan/vk_backend.h"

struct ModelConfig {
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_kv_head = 0;
    uint32_t head_dim = 0;
    uint32_t n_vocab = 0;
    uint32_t max_seq_len = 0;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-5f;
};

class InferenceEngine {
public:
    VulkanBackend backend;
    ModelConfig config;

    // GPU scratch buffers
    GpuBuffer residual_gpu;
    GpuBuffer work_buf;
    GpuBuffer attn_out_buf;
    GpuBuffer ffn_gate_buf;
    GpuBuffer ffn_up_buf;
    GpuBuffer ffn_out_buf;
    GpuBuffer scores_buf;
    GpuBuffer logits_gpu;
    GpuBuffer result_buf;   // uint32 for GPU argmax
    GpuBuffer k_cache_gpu;
    GpuBuffer v_cache_gpu;

    // Weights
    GpuBuffer embed_table_gpu;
    GpuBuffer output_weight;    // LM head (may equal embed_table for weight-tied)
    GpuBuffer output_norm_weight;
    struct LayerWeights {
        GpuBuffer wq, wk, wv, wo;
        GpuBuffer wgate, wup, wdown;
        GpuBuffer attn_norm, ffn_norm;
    };
    std::vector<LayerWeights> layers;

    InferenceEngine();
    ~InferenceEngine();

    bool init(VkInstance instance);
    void cleanup();

    // Load from preprocessed binary + json
    bool load_model(const std::string& bin_path, const std::string& json_path);
    bool create_pipelines(const char* spv_dir);

    // Single forward pass for one token at given position
    uint32_t forward(uint32_t token, uint32_t position);

private:
    bool cleaned_ = false;
    bool create_buffers();
    bool upload_weight(const std::string& name, const float* data, uint32_t count, GpuBuffer& dst);
};
