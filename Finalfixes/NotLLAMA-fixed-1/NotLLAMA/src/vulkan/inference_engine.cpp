#include "vulkan/inference_engine.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Binary format helpers ──

struct TensorEntry {
    uint64_t data_off, data_size;
    uint32_t name_off, name_len;
};

static bool load_binary(const std::vector<char>& d,
                         std::unordered_map<std::string, const float*>& out) {
    if (d.size() < 20) return false;
    if (memcmp(d.data(), "NRDT", 4) != 0) return false;
    uint32_t count, name_table_size;
    memcpy(&count, d.data() + 4, 4);
    memcpy(&name_table_size, d.data() + 8, 4);

    std::vector<TensorEntry> entries(count);
    memcpy(entries.data(), d.data() + 20, count * sizeof(TensorEntry));

    uint32_t name_table_off = 20 + count * sizeof(TensorEntry);
    uint32_t data_region_off = name_table_off + ((name_table_size + 15) & ~15);

    out.clear();
    for (auto& e : entries) {
        std::string name(d.data() + name_table_off + e.name_off, e.name_len);
        out[name] = (const float*)(d.data() + e.data_off);
    }
    return !out.empty();
}

// ── Engine ──

InferenceEngine::InferenceEngine() {}
InferenceEngine::~InferenceEngine() { cleanup(); }

bool InferenceEngine::init(VkInstance instance) {
    if (!backend.init(instance)) {
        fprintf(stderr, "Failed to init VulkanBackend\n");
        return false;
    }
    return true;
}

bool InferenceEngine::create_buffers() {
    auto& c = config;
    VkBufferUsageFlags s = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    auto make = [&](uint32_t bytes) {
        return backend.create_buffer(bytes, s, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    };

    residual_gpu  = make(c.n_embd * sizeof(float));
    work_buf      = make(c.n_embd * sizeof(float));
    attn_out_buf  = make(c.n_head * c.head_dim * sizeof(float));
    ffn_gate_buf  = make(c.n_ff * sizeof(float));
    ffn_up_buf    = make(c.n_ff * sizeof(float));
    ffn_out_buf   = make(c.n_embd * sizeof(float));
    scores_buf    = make(c.n_head * c.max_seq_len * sizeof(float));
    logits_gpu    = make(c.n_vocab * sizeof(float));
    result_buf    = make(sizeof(uint32_t));
    k_cache_gpu   = make(c.n_kv_head * c.max_seq_len * c.head_dim * sizeof(float));
    v_cache_gpu   = make(c.n_kv_head * c.max_seq_len * c.head_dim * sizeof(float));

    return residual_gpu.buffer && logits_gpu.buffer && result_buf.buffer;
}

bool InferenceEngine::create_pipelines(const char* spv_dir) {
    return backend.load_pipelines(spv_dir);
}

bool InferenceEngine::upload_weight(const std::string& name, const float* data,
                                     uint32_t count, GpuBuffer& dst) {
    dst = backend.create_buffer(count * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!dst.buffer) { fprintf(stderr, "Failed to create buffer for %s\n", name.c_str()); return false; }
    backend.upload_to_buffer(dst, data, count * sizeof(float));
    return true;
}

bool InferenceEngine::load_model(const std::string& bin_path, const std::string& json_path) {
    // Read JSON config
    std::ifstream jf(json_path);
    if (!jf) { fprintf(stderr, "Cannot open %s\n", json_path.c_str()); return false; }
    json j;
    try { jf >> j; } catch (...) { fprintf(stderr, "JSON parse error\n"); return false; }

    auto& m = j["model"];
    config.n_layers    = m["block_count"];
    config.n_embd      = m["embedding_length"];
    config.n_ff        = m["feed_forward_length"];
    config.n_head      = m["attention.head_count"];
    config.n_kv_head   = m.value("attention.head_count_kv", config.n_head);
    config.n_vocab     = m["vocab_size"];
    config.max_seq_len = m.value("context_length", 2048u);
    config.head_dim    = config.n_embd / config.n_head;

    // Read binary
    std::ifstream bf(bin_path, std::ios::binary | std::ios::ate);
    if (!bf) { fprintf(stderr, "Cannot open %s\n", bin_path.c_str()); return false; }
    size_t bin_sz = (size_t)bf.tellg();
    std::vector<char> bin(bin_sz);
    bf.seekg(0);
    bf.read(bin.data(), bin_sz);

    std::unordered_map<std::string, const float*> tensors;
    if (!load_binary(bin, tensors)) {
        fprintf(stderr, "Bad binary format\n");
        return false;
    }

    auto get = [&](const std::string& name) -> const float* {
        auto it = tensors.find(name);
        if (it == tensors.end()) { fprintf(stderr, "Missing tensor: %s\n", name.c_str()); return nullptr; }
        return it->second;
    };

    auto get_bytes = [&](const std::string& name) -> uint32_t {
        for (auto& t : j["tensors"]) {
            if (t["name"] == name)
                return t["size_bytes"];
        }
        return 0;
    };

    // Allocate scratch buffers
    if (!create_buffers()) return false;

    // Upload weights
    if (!upload_weight("token_embd.weight",
        get("token_embd.weight"), config.n_vocab * config.n_embd, embed_table_gpu))
        return false;

    if (!upload_weight("output_norm.weight",
        get("output_norm.weight"), config.n_embd, output_norm_weight))
        return false;

    // LM head — might be weight-tied
    auto* ow = get("output.weight");
    if (ow && ow != tensors["token_embd.weight"]) {
        if (!upload_weight("output.weight", ow, config.n_vocab * config.n_embd, output_weight))
            return false;
    } else {
        // Weight-tied: use embed table
        output_weight = embed_table_gpu;
    }

    // Per-layer weights
    layers.resize(config.n_layers);
    for (uint32_t i = 0; i < config.n_layers; i++) {
        auto& l = layers[i];
        std::string p = "blk." + std::to_string(i) + ".";
        uint32_t qs = config.n_head * config.head_dim;
        uint32_t ks = config.n_kv_head * config.head_dim;

        if (!upload_weight(p + "attn_q.weight", get(p + "attn_q.weight"), config.n_embd * qs, l.wq)
         || !upload_weight(p + "attn_k.weight", get(p + "attn_k.weight"), config.n_embd * ks, l.wk)
         || !upload_weight(p + "attn_v.weight", get(p + "attn_v.weight"), config.n_embd * ks, l.wv)
         || !upload_weight(p + "attn_output.weight", get(p + "attn_output.weight"), qs * config.n_embd, l.wo)
         || !upload_weight(p + "ffn_gate.weight", get(p + "ffn_gate.weight"), config.n_embd * config.n_ff, l.wgate)
         || !upload_weight(p + "ffn_up.weight", get(p + "ffn_up.weight"), config.n_embd * config.n_ff, l.wup)
         || !upload_weight(p + "ffn_down.weight", get(p + "ffn_down.weight"), config.n_ff * config.n_embd, l.wdown)
         || !upload_weight(p + "attn_norm.weight", get(p + "attn_norm.weight"), config.n_embd, l.attn_norm)
         || !upload_weight(p + "ffn_norm.weight", get(p + "ffn_norm.weight"), config.n_embd, l.ffn_norm))
            return false;
    }

    // Zero KV cache
    std::vector<float> zeros(config.n_kv_head * config.max_seq_len * config.head_dim, 0);
    backend.upload_to_buffer(k_cache_gpu, zeros.data(), zeros.size() * sizeof(float));
    backend.upload_to_buffer(v_cache_gpu, zeros.data(), zeros.size() * sizeof(float));

    printf("Model loaded: %u layers, %u dim, %u vocab, %u ctx\n",
           config.n_layers, config.n_embd, config.n_vocab, config.max_seq_len);
    return true;
}

uint32_t InferenceEngine::forward(uint32_t token, uint32_t position) {
    auto& c = config;
    float embed_scale = sqrtf((float)c.n_embd);
    float attn_scale = 1.0f / sqrtf((float)c.head_dim);
    uint32_t qs = c.n_head * c.head_dim;
    uint32_t ks = c.n_kv_head * c.head_dim;
    uint32_t gqa = c.n_head / c.n_kv_head;

    // Embedding
    backend.begin_batch();
    backend.token_embedding(embed_table_gpu, residual_gpu, token, c.n_embd, c.n_vocab, embed_scale);
    backend.end_batch();

    for (uint32_t l = 0; l < c.n_layers; l++) {
        auto& w = layers[l];
        backend.begin_batch();

        // Pre-attention RMS norm
        backend.rms_norm(residual_gpu, w.attn_norm, work_buf, c.n_embd, c.rms_eps);
        backend.barrier();

        // Q projection
        backend.gemm(work_buf, w.wq, attn_out_buf, 1, qs, c.n_embd, false, false);
        backend.barrier();
        backend.rope(attn_out_buf, c.head_dim, c.n_head, position, c.rope_theta);
        backend.barrier();

        // K projection
        backend.gemm(work_buf, w.wk, ffn_gate_buf, 1, ks, c.n_embd, false, false);
        backend.barrier();
        backend.rope(ffn_gate_buf, c.head_dim, c.n_kv_head, position, c.rope_theta);
        backend.barrier();

        // V projection
        backend.gemm(work_buf, w.wv, ffn_up_buf, 1, ks, c.n_embd, false, false);
        backend.barrier();

        // KV cache copy (per-head for GQA)
        for (uint32_t h = 0; h < c.n_kv_head; h++) {
            VkDeviceSize so = h * c.head_dim * sizeof(float);
            VkDeviceSize doff = (h * c.max_seq_len + position) * c.head_dim * sizeof(float);
            VkBufferCopy bc{bc.srcOffset = so, bc.dstOffset = doff, bc.size = c.head_dim * sizeof(float)};
            vkCmdCopyBuffer(backend.current_cmd(), ffn_gate_buf.buffer, k_cache_gpu.buffer, 1, &bc);
            vkCmdCopyBuffer(backend.current_cmd(), ffn_up_buf.buffer, v_cache_gpu.buffer, 1, &bc);
        }
        backend.barrier();

        // Attention
        backend.attention_scores_wmma(attn_out_buf, k_cache_gpu, scores_buf,
                                       c.head_dim, position + 1, c.max_seq_len,
                                       attn_scale, c.n_head, gqa, 0);
        backend.barrier();
        backend.attention_softmax(scores_buf, position + 1, c.max_seq_len, c.n_head);
        backend.barrier();
        backend.attention_value(scores_buf, v_cache_gpu, attn_out_buf,
                                c.head_dim, position + 1, c.max_seq_len,
                                c.n_head, gqa);
        backend.barrier();

        // Output projection + residual
        backend.gemm(attn_out_buf, w.wo, work_buf, 1, c.n_embd, qs, false, false);
        backend.barrier();
        backend.add(work_buf, residual_gpu, residual_gpu, c.n_embd);
        backend.barrier();

        // Post-attention RMS norm
        backend.rms_norm(residual_gpu, w.ffn_norm, work_buf, c.n_embd, c.rms_eps);
        backend.barrier();

        // FFN gate + up
        backend.gemm(work_buf, w.wgate, ffn_gate_buf, 1, c.n_ff, c.n_embd, false, false);
        backend.barrier();
        backend.gemm(work_buf, w.wup, ffn_up_buf, 1, c.n_ff, c.n_embd, false, false);
        backend.barrier();
        backend.silu_mul(ffn_gate_buf, ffn_up_buf, ffn_gate_buf, c.n_ff);
        backend.barrier();
        backend.gemm(ffn_gate_buf, w.wdown, ffn_out_buf, 1, c.n_embd, c.n_ff, false, false);
        backend.barrier();
        backend.add(ffn_out_buf, residual_gpu, residual_gpu, c.n_embd);

        backend.end_batch();
    }

    // Output norm + unembed + argmax
    backend.begin_batch();
    backend.rms_norm(residual_gpu, output_norm_weight, work_buf, c.n_embd, c.rms_eps);
    backend.end_batch();

    backend.begin_batch();
    backend.gemm(work_buf, output_weight.buffer ? output_weight : embed_table_gpu,
                 logits_gpu, 1, c.n_vocab, c.n_embd, false, false);
    backend.end_batch();

    backend.begin_batch();
    backend.gpu_argmax(logits_gpu, result_buf, c.n_vocab);
    backend.end_batch();

    uint32_t sampled;
    backend.download_from_buffer(result_buf, &sampled, sizeof(sampled));
    return sampled;
}

void InferenceEngine::cleanup() {
    if (cleaned_) return;
    auto destroy = [&](GpuBuffer& b) { if (b.buffer) { backend.destroy_buffer(b); b = GpuBuffer{}; } };
    destroy(residual_gpu);
    destroy(work_buf);
    destroy(attn_out_buf);
    destroy(ffn_gate_buf);
    destroy(ffn_up_buf);
    destroy(ffn_out_buf);
    destroy(scores_buf);
    destroy(logits_gpu);
    destroy(result_buf);
    destroy(embed_table_gpu);
    // Weight-tied models alias output_weight to embed_table_gpu; avoid double-free.
    if (output_weight.buffer == embed_table_gpu.buffer) output_weight = {};
    destroy(output_weight);
    destroy(k_cache_gpu);
    destroy(v_cache_gpu);
    destroy(output_norm_weight);
    for (auto& l : layers) {
        destroy(l.wq); destroy(l.wk); destroy(l.wv); destroy(l.wo);
        destroy(l.wgate); destroy(l.wup); destroy(l.wdown);
        destroy(l.attn_norm); destroy(l.ffn_norm);
    }
    layers.clear();
    backend.cleanup();
    cleaned_ = true;
}
