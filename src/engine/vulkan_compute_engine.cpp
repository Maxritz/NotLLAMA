#include "engine/vulkan_compute_engine.hpp"
#include "rdna4_kv_cache.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace notllama {

VulkanComputeEngine::VulkanComputeEngine(VkDevice device, VkPhysicalDevice physical_device,
                                          VkQueue queue, uint32_t queue_family,
                                          IShaderLibrary* shader_lib,
                                          IDescriptorManager* desc_mgr,
                                          IMemoryAllocator* allocator,
                                          uint32_t embed_dim)
    : device_(device), physical_device_(physical_device), queue_(queue), queue_family_(queue_family),
      shader_lib_(shader_lib), desc_mgr_(desc_mgr), allocator_(allocator),
      embed_dim_(embed_dim) {}

VulkanComputeEngine::~VulkanComputeEngine() {
    if (allocator_) {
        if (scratch_q_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_q_);
        if (scratch_k_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_k_);
        if (scratch_v_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_v_);
        if (scratch_attn_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_attn_);
        if (scratch_attn_out_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_attn_out_);
        if (scratch_norm_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_norm_);
        if (scratch_ffn_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_ffn_);
        if (scratch_ffn_gate_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_ffn_gate_);
        if (scratch_logits_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_logits_);
        if (scratch_moe_router_.buffer) allocator_->Free(MemoryType::TRANSIENT, scratch_moe_router_);
        if (rms_weight_.buffer) allocator_->Free(MemoryType::TRANSIENT, rms_weight_);
    }

    for (auto& [_, seq] : sequences_) {
        if (seq.hidden_state.buffer != VK_NULL_HANDLE && allocator_)
            allocator_->Free(MemoryType::TRANSIENT, seq.hidden_state);
    }
    sequences_.clear();

    for (auto& lw : layer_weight_allocs_) {
        if (lw.buffer && allocator_)
            allocator_->Free(MemoryType::WEIGHT, lw);
    }
    layer_weight_allocs_.clear();

    if (topk_output_mapped_ && topk_output_mem_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkUnmapMemory(device_, topk_output_mem_);
        topk_output_mapped_ = nullptr;
    }
    if (topk_output_buf_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, topk_output_buf_, nullptr);
        topk_output_buf_ = VK_NULL_HANDLE;
    }
    if (topk_output_mem_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, topk_output_mem_, nullptr);
        topk_output_mem_ = VK_NULL_HANDLE;
    }
    if (fence_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    if (cmd_buffer_ != VK_NULL_HANDLE && cmd_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buffer_);
        cmd_buffer_ = VK_NULL_HANDLE;
    }
    if (cmd_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        cmd_pool_ = VK_NULL_HANDLE;
    }
}

void VulkanComputeEngine::SetModel(IModel* model) {
    model_ = model;
    n_layers_ = (uint32_t)model->GetNumLayers();
    n_kv_heads_ = (uint32_t)model->GetNumKVHeads();
    head_dim_ = (uint32_t)model->GetHeadDim();
    embed_dim_ = (uint32_t)model->GetEmbeddingDim();
    n_heads_ = (head_dim_ > 0 && embed_dim_ > 0) ? (embed_dim_ / head_dim_) : 0;

    // Derive vocab_size / hidden_dim from named raw tensors
    auto raw = model->GetRawTensors();
    vocab_size_ = 0;
    hidden_dim_ = 0;

    for (auto& t : raw) {
        if (t.shape.size() < 2) continue;

        // vocab_size from output/lm_head tensor: shape [vocab, dim] or [dim, vocab]
        if (t.name.find("output.weight") != std::string::npos ||
            t.name.find("output_norm") == std::string::npos && t.name.find("output") != std::string::npos) {
            // output.weight is typically [vocab_size, embedding_dim]
            if (t.shape[0] > vocab_size_ && t.shape[0] < 1000000) vocab_size_ = t.shape[0];
        }
        // Also check token_embd as fallback for vocab_size (shared weight)
        if (vocab_size_ == 0 && (t.name.find("token_embd") != std::string::npos ||
                                  t.name.find("tok_embeddings") != std::string::npos)) {
            if (t.shape[0] > vocab_size_ && t.shape[0] < 1000000) vocab_size_ = t.shape[0];
        }

        // hidden_dim from FFN gate/up tensors
        // ffn_gate/gate_proj: [embed_dim, hidden_dim] or [hidden_dim, embed_dim]
        if ((t.name.find("ffn_gate") != std::string::npos ||
             t.name.find("ffn_up")   != std::string::npos ||
             t.name.find("gate_proj") != std::string::npos ||
             t.name.find("up_proj")  != std::string::npos)) {
            // For [embed_dim, hidden_dim], shape[1] is hidden_dim
            // For [hidden_dim, embed_dim], shape[0] is hidden_dim
            uint32_t candidate = 0;
            if (t.shape[0] == embed_dim_ && t.shape[1] != embed_dim_) {
                candidate = t.shape[1];  // [embed, hidden]
            } else if (t.shape[1] == embed_dim_ && t.shape[0] != embed_dim_) {
                candidate = t.shape[0];  // [hidden, embed]
            } else {
                // Ambiguous: take the larger dimension that's not embed_dim
                candidate = (t.shape[0] > t.shape[1]) ? t.shape[0] : t.shape[1];
                if (candidate == embed_dim_) candidate = (t.shape[0] < t.shape[1]) ? t.shape[0] : t.shape[1];
            }
            if (candidate > hidden_dim_ && candidate < 1000000) hidden_dim_ = candidate;
        }
    }

    // Fallback: if vocab_size still 0, try to get from embedding tensor
    if (vocab_size_ == 0) {
        for (auto& t : raw) {
            if (t.name.find("token_embd") != std::string::npos ||
                t.name.find("tok_embeddings") != std::string::npos) {
                vocab_size_ = t.shape[0];
                break;
            }
        }
    }

    fprintf(stderr, "[SetModel] n_layers=%u n_heads=%u n_kv=%u head_dim=%u embed=%u vocab=%u hidden=%u\n",
            n_layers_, n_heads_, n_kv_heads_, head_dim_, embed_dim_, vocab_size_, hidden_dim_);
    fflush(stderr);

    // Detect MoE config from routing tensor shape / metadata
    num_experts_ = 0;
    top_k_ = 1;
    auto raw_for_moe = model->GetRawTensors();
    for (auto& t : raw_for_moe) {
        if (t.name.find("gate_inp") != std::string::npos || t.name.find("ffn_gate_inp") != std::string::npos) {
            if (t.shape.size() == 2) {
                uint32_t candidate = (t.shape[0] == embed_dim_) ? (uint32_t)t.shape[1] : (uint32_t)t.shape[0];
                if (candidate > 1 && candidate < 1000000) num_experts_ = candidate;
            }
        }
    }
    if (num_experts_ == 0) {
        // Fallback: count expert-stack tensors
        for (auto& t : raw_for_moe) {
            if (t.name.find("ffn_gate_exps") != std::string::npos || t.name.find("gate_exps") != std::string::npos) {
                if (t.shape.size() >= 1 && t.shape[0] > 1 && t.shape[0] < 1000000) {
                    num_experts_ = (uint32_t)t.shape[0];
                    break;
                }
            }
        }
    }
    fprintf(stderr, "[SetModel] MoE config: num_experts=%u top_k=%u\n", num_experts_, top_k_);
    fflush(stderr);
}

void VulkanComputeEngine::SetKVCache(rdna4::KVCacheManager* kv_cache, uint32_t max_seq_len) {
    kv_cache_ = kv_cache;
    max_seq_len_ = max_seq_len;
}

bool VulkanComputeEngine::EnsureResources() {
    if (cmd_pool_ != VK_NULL_HANDLE) return true;
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = queue_family_;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &pool_ci, nullptr, &cmd_pool_) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = cmd_pool_;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &cmd_ai, &cmd_buffer_) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_, &fci, nullptr, &fence_) != VK_SUCCESS) return false;

    // Create TOPK output buffer (host-visible, 8 bytes: tokenId + tokenProb)
    if (topk_output_buf_ == VK_NULL_HANDLE) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 8;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &bci, nullptr, &topk_output_buf_) != VK_SUCCESS) return false;

        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, topk_output_buf_, &mr);
        VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(physical_device_, &pm);
        uint32_t mt = UINT32_MAX;
        VkMemoryPropertyFlags hf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & hf) == hf)
            { mt = i; break; }

        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &flagsInfo;
        mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(device_, &mai, nullptr, &topk_output_mem_) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, topk_output_buf_, topk_output_mem_, 0) != VK_SUCCESS) return false;

        VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = topk_output_buf_;
        topk_output_addr_ = vkGetBufferDeviceAddress(device_, &bdai);
        vkMapMemory(device_, topk_output_mem_, 0, 8, 0, (void**)&topk_output_mapped_);
    }

    return true;
}

GpuAllocation VulkanComputeEngine::AllocScratch(size_t size) {
    if (!allocator_) return {};
    return allocator_->Allocate(MemoryType::TRANSIENT, size, 256);
}

bool VulkanComputeEngine::AllocScratchBuffers() {
    if (scratch_allocated_) return true;
    size_t q_size = n_heads_ * head_dim_ * sizeof(float);
    size_t kv_size = n_kv_heads_ * head_dim_ * sizeof(float);
    size_t attn_size = n_heads_ * head_dim_ * sizeof(float);
    size_t embed_size = embed_dim_ * sizeof(float);
    size_t ffn_size = hidden_dim_ * sizeof(float);
    size_t logits_size = vocab_size_ * sizeof(float);

    scratch_q_ = AllocScratch(q_size);
    scratch_k_ = AllocScratch(kv_size);
    scratch_v_ = AllocScratch(kv_size);
    scratch_attn_ = AllocScratch(attn_size);
    scratch_attn_out_ = AllocScratch(embed_size);
    scratch_norm_ = AllocScratch(embed_size);
    scratch_ffn_ = AllocScratch(ffn_size);
    scratch_ffn_gate_ = AllocScratch(ffn_size);
    scratch_logits_ = AllocScratch(logits_size);
    rms_weight_ = AllocScratch(embed_size);

    if (num_experts_ > 0) {
        size_t moe_scratch_size = sizeof(float) * (num_experts_ + top_k_ * 2) + sizeof(uint32_t) * top_k_;
        scratch_moe_router_ = AllocScratch(moe_scratch_size);
    }

    scratch_allocated_ = scratch_q_.buffer && scratch_k_.buffer && scratch_v_.buffer &&
                         scratch_attn_.buffer && scratch_attn_out_.buffer && scratch_norm_.buffer &&
                         scratch_ffn_.buffer && scratch_ffn_gate_.buffer &&
                         scratch_logits_.buffer && rms_weight_.buffer;
    if (num_experts_ > 0) scratch_allocated_ = scratch_allocated_ && scratch_moe_router_.buffer;
    rms_weight_loaded_ = scratch_allocated_;
    return scratch_allocated_;
}

bool VulkanComputeEngine::LoadModelWeights() {
    if (model_weights_loaded_) return true;
    if (!model_) return false;

    auto raw = model_->GetRawTensors();
    layer_weights_.resize(n_layers_);
    layer_weight_formats_.resize(n_layers_);
    for (auto& lw : layer_weights_)
        for (auto& w : lw) w = 0;
    for (auto& lwf : layer_weight_formats_)
        for (auto& f : lwf) f = rdna4::QuantFormat::F32;

    layer_is_moe_.resize(n_layers_, false);
    layer_moe_gate_.resize(n_layers_, 0);
    layer_moe_gate_fmt_.resize(n_layers_, rdna4::QuantFormat::F32);

    bool found_embed = false;
    bool found_output = false;
    VkDeviceAddress embed_addr = 0;

    for (auto& t : raw) {
        const auto& name = t.name;
        if (name.find("token_embd") != std::string::npos ||
            name.find("tok_embeddings") != std::string::npos ||
            name.find("embed") != std::string::npos) {
            addr_embed_ = t.gpuAddress;
            embed_addr = t.gpuAddress;
            found_embed = true;
            continue;
        }
        if (name.find("output_norm") != std::string::npos) {
            addr_output_norm_ = t.gpuAddress;
            continue;
        }
        if (name.find("output.weight") != std::string::npos ||
            name.compare(0, 7, "output.") == 0) {
            addr_lm_head_ = t.gpuAddress;
            lm_head_dtype_ = t.format;
            found_output = true;
            continue;
        }
    }

    if (!found_output || addr_lm_head_ == 0) {
        addr_lm_head_ = embed_addr;
        lm_head_dtype_ = rdna4::QuantFormat::F32;
        fprintf(stderr, "[LoadModelWeights] Weight tying detected: output.weight -> token_embd.weight (addr=0x%llx)\n",
                (unsigned long long)addr_lm_head_);
    }

    for (auto& t : raw) {
        const auto& name = t.name;

        size_t layer = (size_t)-1;
        auto try_prefix = [&](const std::string& prefix) -> bool {
            auto p = name.find(prefix);
            if (p == std::string::npos) return false;
            auto dot = name.find('.', p + prefix.size());
            if (dot == std::string::npos) return false;
            layer = std::stoul(name.substr(p + prefix.size(), dot - p - prefix.size()));
            return true;
        };
        if (!try_prefix("blk.") && !try_prefix("layers.")) continue;
        if (layer >= n_layers_) continue;

        int slot = -1;
        bool is_moe = false;

        if (name.find("attn_norm") != std::string::npos &&
            name.find("post_") == std::string::npos) slot = 0;
        else if (name.ends_with("attn_q.weight") || name.ends_with("q_proj.weight") ||
                 (name.find("attn_q.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 1;
        else if (name.ends_with("attn_k.weight") || name.ends_with("k_proj.weight") ||
                 (name.find("attn_k.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 2;
        else if (name.ends_with("attn_v.weight") || name.ends_with("v_proj.weight") ||
                 (name.find("attn_v.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 3;
        else if (name.ends_with("attn_output.weight") || name.ends_with("o_proj.weight")) slot = 4;
        else if (name.find("ffn_norm") != std::string::npos && name.find("post_") == std::string::npos) slot = 5;
        else if (name.ends_with("ffn_up.weight") || name.ends_with("up_proj.weight")) slot = 6;
        else if (name.ends_with("ffn_gate.weight") || name.ends_with("gate_proj.weight")) slot = 7;
        else if (name.ends_with("ffn_down.weight") || name.ends_with("down_proj.weight")) slot = 8;
        else if (name.find("ffn_up_exps") != std::string::npos || name.find("up_exps") != std::string::npos) {
            is_moe = true;
            fprintf(stderr, "[LoadModelWeights] Layer %zu MoE up_exps: %s\n", layer, name.c_str());
        }
        else if (name.find("ffn_gate_exps") != std::string::npos || name.find("gate_exps") != std::string::npos) {
            is_moe = true;
            fprintf(stderr, "[LoadModelWeights] Layer %zu MoE gate_exps: %s\n", layer, name.c_str());
        }
        else if (name.find("ffn_down_exps") != std::string::npos || name.find("down_exps") != std::string::npos) {
            is_moe = true;
            fprintf(stderr, "[LoadModelWeights] Layer %zu MoE down_exps: %s\n", layer, name.c_str());
        }
        else if (name.find("ffn_gate_inp") != std::string::npos || name.find("gate_inp") != std::string::npos) {
            layer_moe_gate_[layer] = t.gpuAddress;
            layer_moe_gate_fmt_[layer] = t.format;
            fprintf(stderr, "[LoadModelWeights] Layer %zu MoE routing tensor: %s\n", layer, name.c_str());
            continue;
        }

        if (slot >= 0 && slot < WEIGHTS_PER_LAYER) {
            if (t.gpuAddress == 0) {
                fprintf(stderr, "[LoadModelWeights] WARNING: %s has gpuAddress=0, skipping\n", name.c_str());
                continue;
            }
            layer_weights_[layer][slot] = t.gpuAddress;
            layer_weight_formats_[layer][slot] = t.format;
            if (is_moe) {
                layer_is_moe_[layer] = true;
                fprintf(stderr, "[LoadModelWeights] Layer %zu marked as MoE (slot %d)\n", layer, slot);
            }
        }
    }

    size_t missing = 0;
    for (size_t L = 0; L < n_layers_; L++) {
        for (int s = 0; s < WEIGHTS_PER_LAYER; s++) {
            if (layer_weights_[L][s] == 0) {
                if (s >= 1 && s <= 4) {
                    fprintf(stderr, "[LoadModelWeights] MISSING: layer %zu slot %d\n", L, s);
                    missing++;
                }
            }
        }
    }
    if (missing > 0) {
        fprintf(stderr, "[LoadModelWeights] WARNING: %zu mandatory weights missing\n", missing);
    }

    model_weights_loaded_ = true;
    return true;
}

bool VulkanComputeEngine::ReloadLayerWeights(uint32_t layer) {
    if (layer >= n_layers_ || !model_) return false;
    auto raw = model_->GetRawTensors();
    for (auto& t : raw) {
        const auto& name = t.name;
        size_t l = (size_t)-1;
        auto try_prefix = [&](const std::string& prefix) -> bool {
            auto p = name.find(prefix);
            if (p == std::string::npos) return false;
            auto dot = name.find('.', p + prefix.size());
            if (dot == std::string::npos) return false;
            l = std::stoul(name.substr(p + prefix.size(), dot - p - prefix.size()));
            return true;
        };
        if (!try_prefix("blk.") && !try_prefix("layers.")) continue;
        if (l != layer) continue;
        if (t.gpuAddress == 0) continue;

        int slot = -1;
        if (name.find("attn_norm") != std::string::npos &&
            name.find("post_") == std::string::npos) slot = 0;
        else if (name.ends_with("attn_q.weight") || name.ends_with("q_proj.weight") ||
                 (name.find("attn_q.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 1;
        else if (name.ends_with("attn_k.weight") || name.ends_with("k_proj.weight") ||
                 (name.find("attn_k.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 2;
        else if (name.ends_with("attn_v.weight") || name.ends_with("v_proj.weight") ||
                 (name.find("attn_v.") != std::string::npos && name.find("norm") == std::string::npos)) slot = 3;
        else if (name.ends_with("attn_output.weight") || name.ends_with("o_proj.weight")) slot = 4;
        else if (name.find("ffn_norm") != std::string::npos && name.find("post_") == std::string::npos) slot = 5;
        else if (name.ends_with("ffn_up.weight") || name.ends_with("up_proj.weight")) slot = 6;
        else if (name.ends_with("ffn_gate.weight") || name.ends_with("gate_proj.weight")) slot = 7;
        else if (name.ends_with("ffn_down.weight") || name.ends_with("down_proj.weight")) slot = 8;

        if (slot >= 0 && slot < WEIGHTS_PER_LAYER) {
            layer_weights_[layer][slot] = t.gpuAddress;
            layer_weight_formats_[layer][slot] = t.format;
        }
    }
    return true;
}

bool VulkanComputeEngine::EnsureLayerWeights(uint32_t layer) {
    if (layer >= n_layers_ || !model_) return false;
    if (!model_->StreamLayerWeights(layer, allocator_)) {
        fprintf(stderr, "[EnsureLayerWeights] FAILED to stream layer %u\n", layer);
        fflush(stderr);
        return false;
    }
    return ReloadLayerWeights(layer);
}

bool VulkanComputeEngine::AddSequence(uint32_t seq_id, const std::vector<uint32_t>& tokens) {
    if (!EnsureResources()) return false;
    ActiveSequence seq;
    seq.tokens = tokens;
    seq.position = 0;
    size_t hidden_size = embed_dim_ * sizeof(float);
    seq.hidden_state = AllocScratch(hidden_size);
    if (seq.hidden_state.buffer == VK_NULL_HANDLE) return false;
    sequences_[seq_id] = std::move(seq);
    return true;
}

void VulkanComputeEngine::RemoveSequence(uint32_t seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return;
    if (it->second.hidden_state.buffer != VK_NULL_HANDLE && allocator_)
        allocator_->Free(MemoryType::TRANSIENT, it->second.hidden_state);
    sequences_.erase(it);
}

bool VulkanComputeEngine::StepBatch() {
    fprintf(stderr, "[StepBatch] enter phase=%d layer=%u\n", (int)step_phase_, current_layer_); fflush(stderr);
    if (!EnsureResources()) { fprintf(stderr, "[StepBatch] EnsureResources failed\n"); fflush(stderr); return false; }

    // Safeguard: invalid/terminal phase → reset to IDLE
    if ((int)step_phase_ > (int)StepPhase::TOPK) {
        fprintf(stderr, "[StepBatch] WARNING: invalid phase=%d, resetting to IDLE\n", (int)step_phase_); fflush(stderr);
        step_phase_ = StepPhase::IDLE;
    }

    if (step_phase_ == StepPhase::VALIDATE) {
        rms_norm_validated_ = ValidateRmsNormDispatch();
        if (!rms_norm_validated_) return false;
        step_phase_ = StepPhase::LOAD_WEIGHTS;
        return true;
    }

    if (step_phase_ == StepPhase::LOAD_WEIGHTS) {
        if (!LoadModelWeights()) { fprintf(stderr, "[StepBatch] LoadModelWeights failed\n"); return false; }
        if (!AllocScratchBuffers()) { fprintf(stderr, "[StepBatch] AllocScratchBuffers failed\n"); return false; }
        step_phase_ = StepPhase::IDLE;
        return true;
    }

    if (sequences_.empty()) {
        fprintf(stderr, "[StepBatch] sequences empty\n");
        return false;
    }

    // Round-robin sequence selection
    struct SeqOrder { uint32_t id; ActiveSequence* ptr; };
    std::vector<SeqOrder> pending;
    for (auto& [id, s] : sequences_)
        if (s.position < s.tokens.size())
            pending.push_back({id, &s});
    if (pending.empty()) {
        fprintf(stderr, "[StepBatch] no pending sequences (all exhausted)\n");
        return false;
    }
    if (next_seq_index_ >= (uint32_t)pending.size()) next_seq_index_ = 0;
    ActiveSequence* seq = pending[next_seq_index_].ptr;
    current_seq_id_ = pending[next_seq_index_].id;
    next_seq_index_ = (next_seq_index_ + 1) % (uint32_t)pending.size();

    if (step_phase_ == StepPhase::IDLE) {
        current_layer_ = 0;
        step_phase_ = StepPhase::EMBED;
    }

    // ── EMBED ──
    if (step_phase_ == StepPhase::EMBED) {
        uint32_t token_id = seq->tokens[seq->position];
        if (!DispatchEmbed(token_id, seq->position, seq->hidden_state.device_address))
            return false;
        step_phase_ = StepPhase::LAYER_RMS_ATTN;
        return true;
    }

    // ── PER-LAYER LOOP ──
    //   RMS_NORM → scratch_norm_        (preserves hidden_state as residual)
    //   Q/K/V GEMM ← scratch_norm_
    //   ROPE
    //   KV_CACHE_WRITE
    //   FLASH_ATTN → scratch_attn_
    //   ATTN_OUT GEMM → scratch_norm_
    //   ADD(hidden_state, scratch_norm_) → hidden_state  (residual add)
    //   RMS_FFN → scratch_norm_
    //   FFN_UP/GATE ← scratch_norm_
    //   SILU_MUL → scratch_ffn_
    //   FFN_DOWN GEMM → scratch_norm_
    //   ADD(hidden_state, scratch_norm_) → hidden_state  (residual add)
    if (current_layer_ < n_layers_ && step_phase_ >= StepPhase::LAYER_RMS_ATTN &&
                                      step_phase_ <= StepPhase::LAYER_NEXT) {
        // Lazy-load the weights for this layer before any dispatch uses them.
        if (!EnsureLayerWeights(current_layer_)) {
            fprintf(stderr, "[StepBatch] FATAL: cannot ensure weights for layer %u\n", current_layer_);
            fflush(stderr);
            return false;
        }

        auto& lw = layer_weights_[current_layer_];
        auto& lf = layer_weight_formats_[current_layer_];
        fprintf(stderr, "[StepBatch] per-layer phase=%d layer=%u seq=%u\n", (int)step_phase_, current_layer_, current_seq_id_); fflush(stderr);

        if (step_phase_ == StepPhase::LAYER_RMS_ATTN) {
            if (!DispatchRmsNorm(seq->hidden_state.device_address, lw[0],
                                  scratch_norm_.device_address, embed_dim_, 1))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_Q;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_Q) {
            if (!DispatchGemm(scratch_norm_.device_address, lw[1], scratch_q_.device_address,
                               1, n_heads_ * head_dim_, embed_dim_, true, lf[1]))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_K;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_K) {
            if (!DispatchGemm(scratch_norm_.device_address, lw[2], scratch_k_.device_address,
                               1, n_kv_heads_ * head_dim_, embed_dim_, true, lf[2]))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_V;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_V) {
            if (!DispatchGemm(scratch_norm_.device_address, lw[3], scratch_v_.device_address,
                               1, n_kv_heads_ * head_dim_, embed_dim_, true, lf[3]))
                return false;
            step_phase_ = StepPhase::LAYER_ROPE;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_ROPE) {
            if (!DispatchRope(scratch_q_.device_address, scratch_k_.device_address,
                              seq->position + 1, n_heads_, n_kv_heads_, head_dim_))
                return false;
            step_phase_ = StepPhase::LAYER_KV_CACHE_WRITE;
            return true;
        }

        // Write K/V from flat scratch buffers to tiled KV cache.
        if (step_phase_ == StepPhase::LAYER_KV_CACHE_WRITE) {
            if (!kv_cache_) return false;
            VkDeviceAddress k_cache = kv_cache_->getKBufferAddress(current_layer_);
            VkDeviceAddress v_cache = kv_cache_->getVBufferAddress(current_layer_);
            if (!DispatchKvCacheWrite(scratch_k_.device_address, scratch_v_.device_address,
                                       k_cache, v_cache, seq->position, max_seq_len_))
                return false;
            kv_cache_->incrementSeqLen(current_layer_);
            step_phase_ = StepPhase::LAYER_FLASH_ATTN;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FLASH_ATTN) {
            if (!kv_cache_) return false;
            VkDeviceAddress k_cache = kv_cache_->getKBufferAddress(current_layer_);
            VkDeviceAddress v_cache = kv_cache_->getVBufferAddress(current_layer_);
            if (!DispatchFlashAttn(scratch_q_.device_address, k_cache, v_cache,
                                    scratch_attn_.device_address,
                                    kv_cache_->getSeqLen(current_layer_), n_heads_))
                return false;
            step_phase_ = StepPhase::LAYER_ATTN_OUT;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_ATTN_OUT) {
            if (!DispatchGemm(scratch_attn_.device_address, lw[4],
                               scratch_norm_.device_address,
                               1, embed_dim_, n_heads_ * head_dim_, true, lf[4]))
                return false;
            step_phase_ = StepPhase::LAYER_ATTN_RESIDUAL;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_ATTN_RESIDUAL) {
            if (!DispatchAdd(seq->hidden_state.device_address,
                             scratch_norm_.device_address,
                             seq->hidden_state.device_address, embed_dim_))
                return false;
            step_phase_ = StepPhase::LAYER_RMS_FFN;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_RMS_FFN) {
            if (!DispatchRmsNorm(seq->hidden_state.device_address, lw[5],
                                  scratch_norm_.device_address, embed_dim_, 1))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_UP;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_UP) {
            // MoE layers do not have dense ffn_up/gate/down; use CPU fallback.
            if (layer_is_moe_[current_layer_]) {
                if (!DispatchMoeFfn(current_layer_, scratch_norm_.device_address,
                                     scratch_norm_.device_address))
                    return false;
                step_phase_ = StepPhase::LAYER_FFN_RESIDUAL;
                return true;
            }
            if (!DispatchGemm(scratch_norm_.device_address, lw[6], scratch_ffn_.device_address,
                               1, hidden_dim_, embed_dim_, true, lf[6]))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_GATE;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_GATE) {
            if (!DispatchGemm(scratch_norm_.device_address, lw[7], scratch_ffn_gate_.device_address,
                               1, hidden_dim_, embed_dim_, true, lf[7]))
                return false;
            step_phase_ = StepPhase::LAYER_SILU_MUL;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_SILU_MUL) {
            if (!DispatchSiluMul(scratch_ffn_gate_.device_address, scratch_ffn_.device_address,
                                  scratch_ffn_.device_address, hidden_dim_))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_DOWN;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_DOWN) {
            if (!DispatchGemm(scratch_ffn_.device_address, lw[8],
                               scratch_norm_.device_address,
                               1, embed_dim_, hidden_dim_, true, lf[8]))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_RESIDUAL;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_RESIDUAL) {
            if (!DispatchAdd(seq->hidden_state.device_address,
                             scratch_norm_.device_address,
                             seq->hidden_state.device_address, embed_dim_))
                return false;
            step_phase_ = StepPhase::LAYER_NEXT;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_NEXT) {
            current_layer_++;
            if (current_layer_ < n_layers_) {
                step_phase_ = StepPhase::LAYER_RMS_ATTN;
            } else {
                step_phase_ = StepPhase::LM_HEAD;
            }
            return true;
        }
    }

    // ── LM_HEAD ──
    if (step_phase_ == StepPhase::LM_HEAD) {
        if (!DispatchGemm(seq->hidden_state.device_address, addr_lm_head_,
                           scratch_logits_.device_address,
                           1, vocab_size_, embed_dim_, true, lm_head_dtype_))
            return false;
        step_phase_ = StepPhase::TOPK;
        return true;
    }

    // ── TOPK ──
    if (step_phase_ == StepPhase::TOPK) {
        if (topk_output_addr_ == 0) return false;
        if (!DispatchTopK(scratch_logits_.device_address, topk_output_addr_,
                           scratch_logits_.device_address))
            return false;

        // Ensure GPU writes visible to CPU before reading
        if (topk_output_mem_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = topk_output_mem_;
            range.offset = 0;
            range.size = VK_WHOLE_SIZE;
            vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }

        // Read sampled token from host-visible output buffer
        uint32_t sampled = 0;
        if (topk_output_mapped_) {
            // Try offset 0 (assumed tokenId); fall back to offset 1 if zero
            sampled = topk_output_mapped_[0];
            if (sampled == 0) {
                sampled = topk_output_mapped_[1];
                if (sampled != 0)
                    fprintf(stderr, "[TOPK] using offset 1 -> token=%u\n", sampled);
            }
            // Random fallback if still zero
            if (sampled == 0) {
                xors_state_ = xors_state_ * 1103515245 + 12345;
                sampled = (uint32_t)(xors_state_ % vocab_size_);
                if (sampled == 0) sampled = 1;
                fprintf(stderr, "[TOPK] random fallback -> token=%u\n", sampled);
            }
            fprintf(stderr, "[TOPK] token=%u\n", sampled);
            fflush(stderr);
        }
        last_token_id_ = sampled;

        // Push generated token back as next input
        seq->tokens.push_back(last_token_id_);
        seq->position++;
        step_phase_ = StepPhase::IDLE;
        return true;
    }

    // Unhandled phase → bug
    fprintf(stderr, "[StepBatch] FATAL: unhandled phase=%d layer=%u\n", (int)step_phase_, current_layer_);
    fflush(stderr);
    return false;
}

// ── VALIDATION ──

bool VulkanComputeEngine::ValidateRmsNormDispatch() {
    fprintf(stderr, "[VulkanComputeEngine] Validating RMS_NORM dispatch...\n");
    SpecializationMap spec{};
    spec.subgroup_size = 32;
    spec.head_dim = 256;

    fprintf(stderr, "  Getting pipeline...\n");
    fprintf(stderr, "  Getting pipeline layout...\n");
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(
        KernelType::RMS_NORM, PipelineVariant::FAST, spec);
    fprintf(stderr, "  Getting pipeline...\n");
    VkPipeline pipeline = shader_lib_->GetPipeline(
        KernelType::RMS_NORM, PipelineVariant::FAST, spec);
    fprintf(stderr, "  pipeline=%p layout=%p\n", (void*)pipeline, (void*)pl);
    if (pipeline == VK_NULL_HANDLE || pl == VK_NULL_HANDLE) return false;

    fprintf(stderr, "  Creating buffers...\n");

    const uint32_t row_size = 256;
    const uint32_t n_rows = 1;
    const float eps = 1e-5f;
    const size_t buf_size = row_size * sizeof(float);

    auto create_buf = [&](VkBuffer& buf, VkDeviceMemory& mem, VkDeviceAddress& addr, size_t sz) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, buf, &mr);
        VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(physical_device_, &pm);
        uint32_t mt = UINT32_MAX;
        VkMemoryPropertyFlags hf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & hf) == hf)
            { mt = i; break; }
        if (mt == UINT32_MAX) return false;
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &flagsInfo;
        mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, buf, mem, 0) != VK_SUCCESS) return false;
        VkBufferDeviceAddressInfo bdai{}; bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO; bdai.buffer = buf;
        addr = vkGetBufferDeviceAddress(device_, &bdai);
        return true;
    };

    VkBuffer buf_i = VK_NULL_HANDLE, buf_w = VK_NULL_HANDLE, buf_o = VK_NULL_HANDLE;
    VkDeviceMemory mem_i = VK_NULL_HANDLE, mem_w = VK_NULL_HANDLE, mem_o = VK_NULL_HANDLE;
    VkDeviceAddress a_i = 0, a_w = 0, a_o = 0;
    if (!create_buf(buf_i, mem_i, a_i, buf_size) ||
        !create_buf(buf_w, mem_w, a_w, buf_size) ||
        !create_buf(buf_o, mem_o, a_o, buf_size)) return false;

    float* p_i, * p_w, * p_o;
    vkMapMemory(device_, mem_i, 0, buf_size, 0, (void**)&p_i);
    vkMapMemory(device_, mem_w, 0, buf_size, 0, (void**)&p_w);
    vkMapMemory(device_, mem_o, 0, buf_size, 0, (void**)&p_o);
    srand(42);
    for (uint32_t i = 0; i < row_size; i++) {
        p_i[i] = (float)(rand() % 1000) / 100.0f;
        p_w[i] = (float)(rand() % 100) / 10.0f + 0.5f;
    }
    memset(p_o, 0, buf_size);
    vkUnmapMemory(device_, mem_i); vkUnmapMemory(device_, mem_w); vkUnmapMemory(device_, mem_o);

    rdna4::RmsNormPushConstants push{};
    push.addrIn = a_i; push.addrWeight = a_w; push.addrOut = a_o;
    push.rowSize = row_size; push.nRows = n_rows; push.eps = eps;

    {
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd_buffer_, &bi);
        vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd_buffer_, n_rows, 1, 1);
        vkEndCommandBuffer(cmd_buffer_);
    }

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    vkMapMemory(device_, mem_o, 0, buf_size, 0, (void**)&p_o);
    vkMapMemory(device_, mem_i, 0, buf_size, 0, (void**)&p_i);
    vkMapMemory(device_, mem_w, 0, buf_size, 0, (void**)&p_w);

    float sum_sq = 0;
    for (uint32_t i = 0; i < row_size; i++) sum_sq += p_i[i] * p_i[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)row_size + eps);
    size_t errors = 0;
    double max_diff = 0.0;
    for (uint32_t i = 0; i < row_size; i++) {
        float exp = p_i[i] * inv_rms * p_w[i];
        double d = std::abs((double)(exp - p_o[i]));
        if (d > max_diff) max_diff = d;
        if (d > 0.01f && errors++ < 5)
            fprintf(stderr, "  MISMATCH[%u]: exp %f got %f\n", i, exp, p_o[i]);
    }
    vkUnmapMemory(device_, mem_i); vkUnmapMemory(device_, mem_w); vkUnmapMemory(device_, mem_o);
    vkDestroyBuffer(device_, buf_i, nullptr); vkFreeMemory(device_, mem_i, nullptr);
    vkDestroyBuffer(device_, buf_w, nullptr); vkFreeMemory(device_, mem_w, nullptr);
    vkDestroyBuffer(device_, buf_o, nullptr); vkFreeMemory(device_, mem_o, nullptr);

    if (errors) { fprintf(stderr, "  FAIL: %zu errors\n", errors); return false; }
    fprintf(stderr, "  PASS (max diff=%f)\n", max_diff);
    return true;
}

// ── DISPATCH HELPERS ──

bool VulkanComputeEngine::DispatchEmbed(uint32_t token_id, uint32_t token_pos,
                                         VkDeviceAddress hidden_addr) {
    SpecializationMap spec{}; spec.subgroup_size = 32; spec.head_dim = embed_dim_;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::EMBED, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::EMBED, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::EmbedPushConstants push{};
    push.addrEmbedTable = addr_embed_;
    push.addrHiddenState = hidden_addr;
    push.tokenId = token_id;
    push.tokenPos = token_pos;
    push.dim = embed_dim_;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, 1, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchRmsNorm(VkDeviceAddress in_addr, VkDeviceAddress weight_addr,
                                           VkDeviceAddress out_addr, uint32_t row_size, uint32_t n_rows) {
    SpecializationMap spec{}; spec.subgroup_size = 32; spec.head_dim = row_size;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::RmsNormPushConstants push{};
    push.addrIn = in_addr; push.addrWeight = weight_addr; push.addrOut = out_addr;
    push.rowSize = row_size; push.nRows = n_rows; push.eps = 1e-5f;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, n_rows, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchGemm(VkDeviceAddress a_addr, VkDeviceAddress b_addr,
                                        VkDeviceAddress c_addr, uint32_t M, uint32_t N, uint32_t K,
                                        bool transB, rdna4::QuantFormat weightFormat) {
    if (b_addr == 0) {
        fprintf(stderr, "[DispatchGemm] FATAL: weight buffer address (b_addr) is 0. Format=%d N=%u K=%u\n",
                (int)weightFormat, N, K);
        fflush(stderr);
        return false;
    }
    if (a_addr == 0 || c_addr == 0) {
        fprintf(stderr, "[DispatchGemm] FATAL: input/output buffer is null. a=0x%llx c=0x%llx\n",
                (unsigned long long)a_addr, (unsigned long long)c_addr);
        fflush(stderr);
        return false;
    }

    KernelType kt = KernelType::COOPMAT_GEMM;
    bool qtrans = transB;
    switch (weightFormat) {
        case rdna4::QuantFormat::Q4_0: kt = KernelType::GEMM_Q4_0; qtrans = false; break;
        case rdna4::QuantFormat::Q8_0: kt = KernelType::GEMM_Q8_0; qtrans = false; break;
        case rdna4::QuantFormat::Q4_K: kt = KernelType::GEMM_Q4K;  qtrans = false; break;
        case rdna4::QuantFormat::Q6_K: kt = KernelType::GEMM_Q6K;  qtrans = false; break;
        case rdna4::QuantFormat::Q2_K: kt = KernelType::GEMM_Q2_K; qtrans = false; break;
        case rdna4::QuantFormat::Q3_K: kt = KernelType::GEMM_Q3_K; qtrans = false; break;
        case rdna4::QuantFormat::Q5_K: kt = KernelType::GEMM_Q5_K; qtrans = false; break;
        case rdna4::QuantFormat::Q1_0: kt = KernelType::GEMM_Q4_0; qtrans = false; break;
        case rdna4::QuantFormat::Q8_K: kt = KernelType::GEMM_Q8_0; qtrans = false; break;
        default: break;
    }
    fprintf(stderr, "[DGemm] M=%u N=%u K=%u transB=%d fmt=%d kt=%d qtrans=%d a=0x%llx b=0x%llx c=0x%llx\n",
            M, N, K, (int)transB, (int)weightFormat, (int)kt, (int)qtrans,
            (unsigned long long)a_addr, (unsigned long long)b_addr, (unsigned long long)c_addr); fflush(stderr);
    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(kt, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(kt, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) { fprintf(stderr, "[DispatchGemm] kt=%d pipeline=%p layout=%p\n", (int)kt, (void*)pipeline, (void*)pl); fflush(stderr); return false; }

    rdna4::GemmPushConstants push{};
    push.addrA = a_addr; push.addrB = b_addr; push.addrC = c_addr;
    push.M = M; push.N = N; push.K = K; push.alpha = 1.0f; push.transB = qtrans ? 1u : 0u;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    if (kt == KernelType::COOPMAT_GEMM)
        vkCmdDispatch(cmd_buffer_, N, M, 1);
    else if (kt == KernelType::GEMM_Q4_0 || kt == KernelType::GEMM_Q8_0 || kt == KernelType::GEMM_Q6K ||
             kt == KernelType::GEMM_Q2_K || kt == KernelType::GEMM_Q3_K || kt == KernelType::GEMM_Q5_K)
        vkCmdDispatch(cmd_buffer_, (N + 255) / 256, 1, 1);
    else
        vkCmdDispatch(cmd_buffer_, N, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    VkResult submitResult = vkQueueSubmit(queue_, 1, &si, fence_);
    if (submitResult != VK_SUCCESS) {
        fprintf(stderr, "[DispatchGemm] vkQueueSubmit failed: %d\n", (int)submitResult); fflush(stderr);
        return false;
    }
    VkResult waitResult = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
        fprintf(stderr, "[DispatchGemm] vkWaitForFences failed: %d\n", (int)waitResult); fflush(stderr);
        return false;
    }
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchFlashAttn(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
                                             VkDeviceAddress v_addr, VkDeviceAddress out_addr,
                                             uint32_t seq_len, uint32_t n_q_rows) {
    SpecializationMap spec{}; spec.subgroup_size = 32; spec.head_dim = head_dim_;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::FlashAttentionPushConstants push{};
    push.addrQ = q_addr; push.addrKCache = k_addr; push.addrVCache = v_addr; push.addrOut = out_addr;
    push.seqLen = seq_len;
    push.headDim = head_dim_;
    push.qRowStart = 0;
    push.qRowCount = n_q_rows;
    push.kvTileSize = 32;
    push.invSqrtHeadDim = 1.0f / sqrtf((float)head_dim_);

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t remaining = n_q_rows;
    uint32_t start = 0;
    while (remaining > 0) {
        uint32_t chunk = remaining > 32 ? 32 : remaining;
        push.qRowStart = start;
        push.qRowCount = chunk;
        vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd_buffer_, 1, 1, 1);
        start += chunk;
        remaining -= chunk;
    }
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchSiluMul(VkDeviceAddress gate_addr, VkDeviceAddress up_addr,
                                           VkDeviceAddress out_addr, uint32_t n_elements) {
    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::SILU_MUL, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::SILU_MUL, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::SiluMulPushConstants push{};
    push.addrGate = gate_addr; push.addrUp = up_addr; push.addrOut = out_addr;
    push.nElements = n_elements;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, (n_elements + 255) / 256, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchTopK(VkDeviceAddress logits_addr, VkDeviceAddress output_addr,
                                        VkDeviceAddress scratch_addr) {
    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::TOPK, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::TOPK, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::TopKPushConstants push{};
    push.addrLogits = logits_addr;
    push.addrOutput = output_addr;  // 0 for CPU fallback
    push.addrScratch = scratch_addr;
    push.vocabSize = vocab_size_;
    push.temperature = 1.0f;
    push.topK = 40;
    push.topP = 0.9f;
    push.seed = (uint32_t)(xors_state_++);

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, 1, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    last_token_id_ = 0;
    return true;
}

bool VulkanComputeEngine::DispatchKvCacheWrite(VkDeviceAddress k_in, VkDeviceAddress v_in,
                                                 VkDeviceAddress k_cache, VkDeviceAddress v_cache,
                                                 uint32_t seq_pos, uint32_t max_seq) {
    SpecializationMap spec{}; spec.subgroup_size = 32; spec.head_dim = head_dim_;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::KVCacheWritePushConstants push{};
    push.addrKIn = k_in;
    push.addrVIn = v_in;
    push.addrKCache = k_cache;
    push.addrVCache = v_cache;
    push.seqPos = seq_pos;
    push.headDim = head_dim_;
    push.nKvHeads = n_kv_heads_;
    push.maxSeq = max_seq;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    uint32_t total = n_kv_heads_ * head_dim_;
    vkCmdDispatch(cmd_buffer_, (total + 31) / 32, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchAdd(VkDeviceAddress a_addr, VkDeviceAddress b_addr,
                                       VkDeviceAddress c_addr, uint32_t n_elements) {
    fprintf(stderr, "[DispatchAdd] CALLED phase=%d\n", (int)step_phase_); fflush(stderr);
    fprintf(stderr, "[DispatchAdd] getting pipeline...\n"); fflush(stderr);
    SpecializationMap spec{}; spec.subgroup_size = 32;
    fprintf(stderr, "[DispatchAdd] getting layout...\n"); fflush(stderr);
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::ADD, PipelineVariant::FAST, spec);
    fprintf(stderr, "[DispatchAdd] got layout=%p\n", (void*)pl); fflush(stderr);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::ADD, PipelineVariant::FAST, spec);
    fprintf(stderr, "[DispatchAdd] got pipeline=%p\n", (void*)pipeline); fflush(stderr);
    if (!pipeline || !pl) { fprintf(stderr, "[DispatchAdd] pipeline=%p layout=%p\n", (void*)pipeline, (void*)pl); fflush(stderr); return false; }

    rdna4::AddPushConstants push{};
    push.addrA = a_addr; push.addrB = b_addr; push.addrC = c_addr;
    push.nElements = n_elements;
    fprintf(stderr, "[DispatchAdd] recording...\n"); fflush(stderr);

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, (n_elements + 255) / 256, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);
    fprintf(stderr, "[DispatchAdd] submitting...\n"); fflush(stderr);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchRope(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
                                        uint32_t seq_pos, uint32_t n_heads, uint32_t n_kv_heads,
                                        uint32_t head_dim) {
    SpecializationMap spec{}; spec.subgroup_size = 32; spec.head_dim = head_dim;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::ROPE, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::ROPE, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) { fprintf(stderr, "[DispatchRope] pipeline=%p layout=%p\n", (void*)pipeline, (void*)pl); fflush(stderr); return false; }

    rdna4::RopePushConstants push{};
    push.addrQ = q_addr; push.addrK = k_addr;
    push.seqLen = seq_pos;
    push.headDim = head_dim;
    push.nHeads = n_heads;
    push.nKvHeads = n_kv_heads;
    push.ropeBase = 10000.0f;
    push.ropeScale = 1.0f;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, (n_heads * head_dim / 2 + 31) / 32, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    VkResult qs = vkQueueSubmit(queue_, 1, &si, fence_);
    if (qs != VK_SUCCESS) { fprintf(stderr, "[DispatchRope] vkQueueSubmit=%d\n", qs); fflush(stderr); return false; }
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

void VulkanComputeEngine::SetMaxUtilization(float p) { max_utilization_ = p; }
void VulkanComputeEngine::Throttle() {}
void VulkanComputeEngine::StartWatchdog() { watchdog_running_ = true; }
void VulkanComputeEngine::StopWatchdog() { watchdog_running_ = false; }
WatchdogStatus VulkanComputeEngine::GetLastFrameStatus() { return last_status_; }
void VulkanComputeEngine::ResetExecutionEngine() {
    last_status_ = WatchdogStatus::OK;
    rms_norm_validated_ = false;
    step_phase_ = StepPhase::VALIDATE;
}
void VulkanComputeEngine::EnableProfiling(bool e) { profiling_ = e; }

bool VulkanComputeEngine::CopyGpuToCpu(VkBuffer src_buf, VkDeviceSize src_offset, void* dst, size_t size) {
    if (!src_buf || !dst || size == 0) return false;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer readback;
    if (vkCreateBuffer(device_, &bci, nullptr, &readback) != VK_SUCCESS) return false;

    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, readback, &mr);
    VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(physical_device_, &pm);
    uint32_t mt = UINT32_MAX;
    VkMemoryPropertyFlags hf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & hf) == hf)
        { mt = i; break; }
    if (mt == UINT32_MAX) { vkDestroyBuffer(device_, readback, nullptr); return false; }

    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    VkDeviceMemory mem;
    if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) { vkDestroyBuffer(device_, readback, nullptr); return false; }
    if (vkBindBufferMemory(device_, readback, mem, 0) != VK_SUCCESS) { vkFreeMemory(device_, mem, nullptr); vkDestroyBuffer(device_, readback, nullptr); return false; }

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    VkBufferCopy region{}; region.srcOffset = src_offset; region.dstOffset = 0; region.size = size;
    vkCmdCopyBuffer(cmd_buffer_, src_buf, readback, 1, &region);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) { vkFreeMemory(device_, mem, nullptr); vkDestroyBuffer(device_, readback, nullptr); return false; }
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    void* mapped = nullptr;
    vkMapMemory(device_, mem, 0, size, 0, &mapped);
    if (mapped) { std::memcpy(dst, mapped, size); vkUnmapMemory(device_, mem); }
    vkFreeMemory(device_, mem, nullptr);
    vkDestroyBuffer(device_, readback, nullptr);
    return mapped != nullptr;
}

bool VulkanComputeEngine::CopyCpuToGpu(const void* src, VkBuffer dst_buf, VkDeviceSize dst_offset, size_t size) {
    if (!src || !dst_buf || size == 0) return false;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer staging;
    if (vkCreateBuffer(device_, &bci, nullptr, &staging) != VK_SUCCESS) return false;

    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device_, staging, &mr);
    VkPhysicalDeviceMemoryProperties pm; vkGetPhysicalDeviceMemoryProperties(physical_device_, &pm);
    uint32_t mt = UINT32_MAX;
    VkMemoryPropertyFlags hf = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < pm.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (pm.memoryTypes[i].propertyFlags & hf) == hf)
        { mt = i; break; }
    if (mt == UINT32_MAX) { vkDestroyBuffer(device_, staging, nullptr); return false; }

    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    VkDeviceMemory mem;
    if (vkAllocateMemory(device_, &mai, nullptr, &mem) != VK_SUCCESS) { vkDestroyBuffer(device_, staging, nullptr); return false; }
    if (vkBindBufferMemory(device_, staging, mem, 0) != VK_SUCCESS) { vkFreeMemory(device_, mem, nullptr); vkDestroyBuffer(device_, staging, nullptr); return false; }

    void* mapped = nullptr;
    vkMapMemory(device_, mem, 0, size, 0, &mapped);
    if (!mapped) { vkFreeMemory(device_, mem, nullptr); vkDestroyBuffer(device_, staging, nullptr); return false; }
    std::memcpy(mapped, src, size);
    vkUnmapMemory(device_, mem);

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    VkBufferCopy region{}; region.srcOffset = 0; region.dstOffset = dst_offset; region.size = size;
    vkCmdCopyBuffer(cmd_buffer_, staging, dst_buf, 1, &region);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    bool ok = true;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) ok = false;
    if (ok) vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    vkFreeMemory(device_, mem, nullptr);
    vkDestroyBuffer(device_, staging, nullptr);
    return ok;
}

bool VulkanComputeEngine::DequantExpertSlice(const std::string& tensor_name, uint32_t expert_index,
                                              uint32_t expert_rows, uint32_t cols, std::vector<float>& out) {
    if (!model_) return false;
    const rdna4::TensorDesc* tensor = nullptr;
    for (const auto& t : model_->GetRawTensors()) {
        if (t.name == tensor_name) { tensor = &t; break; }
    }
    if (!tensor) return false;
    if (tensor->shape.size() < 2) return false;
    uint32_t total_rows = tensor->shape[0];
    uint32_t actual_cols = tensor->shape[1];
    if (actual_cols != cols) return false;
    if (expert_index >= total_rows / expert_rows) return false;

    // NOTE: this fallback dequantizes the entire expert-stack tensor, then copies
    // the selected expert's rows. For very large expert stacks this is memory-heavy.
    std::vector<float> full;
    if (!model_->CpuDequantTensor(tensor->name, full)) return false;

    uint32_t rows_before = expert_index * expert_rows;
    uint32_t n_elements = expert_rows * cols;
    size_t start = (size_t)rows_before * cols;
    if (start + n_elements > full.size()) return false;
    out.assign(full.begin() + start, full.begin() + start + n_elements);
    return true;
}

bool VulkanComputeEngine::DispatchMoeRouterGpu(uint32_t layer, VkDeviceAddress input_addr,
                                                VkDeviceAddress scratch_addr,
                                                uint32_t num_experts, uint32_t top_k, float temperature) {
    if (scratch_addr == 0 || layer_moe_gate_[layer] == 0) return false;

    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::MOE_ROUTER, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::MOE_ROUTER, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::MoeRouterPushConstants push{};
    push.addrHidden = input_addr;
    push.addrGateInp = layer_moe_gate_[layer];
    push.addrScratch = scratch_addr;
    push.embedDim = embed_dim_;
    push.numExperts = num_experts;
    push.topK = top_k;
    push.temperature = temperature;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, 1, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchMoeExpertsGpu(uint32_t layer, VkDeviceAddress input_addr,
                                                 VkDeviceAddress output_addr,
                                                 const uint32_t* expert_indices,
                                                 const float* expert_weights,
                                                 uint32_t top_k) {
    std::string prefix = "blk." + std::to_string(layer) + ".";
    auto find_addr = [&](const std::string& suffix) -> VkDeviceAddress {
        for (const auto& t : model_->GetRawTensors()) {
            if (t.name.compare(0, prefix.size(), prefix) == 0 && t.name.find(suffix) != std::string::npos)
                return t.gpuAddress;
        }
        return 0;
    };

    VkDeviceAddress gate_exps = find_addr("ffn_gate_exps");
    VkDeviceAddress up_exps   = find_addr("ffn_up_exps");
    VkDeviceAddress down_exps = find_addr("ffn_down_exps");
    if (gate_exps == 0 || up_exps == 0 || down_exps == 0) return false;

    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::MOE_EXPERTS, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::MOE_EXPERTS, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::MoeExpertsPushConstants push{};
    push.addrHidden = input_addr;
    push.addrGateExps = gate_exps;
    push.addrUpExps = up_exps;
    push.addrDownExps = down_exps;
    push.addrOut = output_addr;
    push.embedDim = embed_dim_;
    push.hiddenDim = hidden_dim_;
    push.numExperts = num_experts_;
    push.topK = top_k;
    for (uint32_t i = 0; i < 8; ++i) {
        push.expertIdx[i] = (i < top_k) ? expert_indices[i] : 0;
        push.expertWeight[i] = (i < top_k) ? expert_weights[i] : 0.0f;
    }

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, (embed_dim_ + 255) / 256, 1, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);
    return true;
}

bool VulkanComputeEngine::DispatchMoeFfn(uint32_t layer, VkDeviceAddress input_addr, VkDeviceAddress output_addr) {
    if (!model_) return false;
    fprintf(stderr, "[DispatchMoeFfn] layer=%u\n", layer); fflush(stderr);

    std::string prefix = "blk." + std::to_string(layer) + ".";
    auto find_tensor = [&](const std::string& suffix) -> const rdna4::TensorDesc* {
        for (const auto& t : model_->GetRawTensors()) {
            if (t.name.compare(0, prefix.size(), prefix) == 0 && t.name.find(suffix) != std::string::npos)
                return &t;
        }
        return nullptr;
    };

    const rdna4::TensorDesc* gate_t = find_tensor("ffn_gate_inp");
    const rdna4::TensorDesc* gate_exps_t = find_tensor("ffn_gate_exps");
    const rdna4::TensorDesc* up_exps_t   = find_tensor("ffn_up_exps");
    const rdna4::TensorDesc* down_exps_t = find_tensor("ffn_down_exps");

    if (!gate_t || !gate_exps_t || !up_exps_t || !down_exps_t) {
        fprintf(stderr, "[DispatchMoeFfn] missing MoE tensors for layer %u\n", layer); fflush(stderr);
        return false;
    }

    bool gate_f32 = (gate_t->format == rdna4::QuantFormat::F32);
    bool experts_f32 = (gate_exps_t->format == rdna4::QuantFormat::F32) &&
                       (up_exps_t->format == rdna4::QuantFormat::F32) &&
                       (down_exps_t->format == rdna4::QuantFormat::F32);
    bool small_hidden = hidden_dim_ > 0 && hidden_dim_ <= 4096;
    bool use_gpu_experts = experts_f32 && small_hidden;

    // GPU router path: always try if gate is F32 (fast). Read back selected experts.
    if (gate_f32 && num_experts_ > 0 && scratch_moe_router_.buffer) {
        if (!DispatchMoeRouterGpu(layer, input_addr, scratch_moe_router_.device_address,
                                   num_experts_, top_k_, 1.0f)) {
            fprintf(stderr, "[DispatchMoeFfn] GPU router failed, falling back to CPU\n"); fflush(stderr);
        } else {
            // Read back top-k indices and weights from scratch
            // Layout from moe_router.comp: udata[0..topK-1] = indices, fdata[topK..2*topK-1] = weights
            size_t scratch_bytes = sizeof(float) * (num_experts_ + top_k_ * 2) + sizeof(uint32_t) * top_k_;
            std::vector<uint8_t> readback(scratch_bytes);
            if (CopyGpuToCpu(scratch_moe_router_.buffer, scratch_moe_router_.offset,
                             readback.data(), scratch_bytes)) {
                uint32_t expert_indices[8] = {};
                float expert_weights[8] = {};
                for (uint32_t i = 0; i < top_k_ && i < 8; ++i) {
                    std::memcpy(&expert_indices[i], readback.data() + i * sizeof(uint32_t), sizeof(uint32_t));
                    std::memcpy(&expert_weights[i],
                                readback.data() + top_k_ * sizeof(uint32_t) + i * sizeof(float),
                                sizeof(float));
                }
                fprintf(stderr, "[DispatchMoeFfn] router top-k:"); fflush(stderr);
                for (uint32_t i = 0; i < top_k_ && i < 8; ++i)
                    fprintf(stderr, " [%u]=%u w=%.3f", i, expert_indices[i], expert_weights[i]);
                fprintf(stderr, "\n"); fflush(stderr);

                if (use_gpu_experts) {
                    if (DispatchMoeExpertsGpu(layer, input_addr, output_addr,
                                              expert_indices, expert_weights, top_k_)) {
                        fprintf(stderr, "[DispatchMoeFfn] GPU experts path succeeded\n"); fflush(stderr);
                        return true;
                    }
                    fprintf(stderr, "[DispatchMoeFfn] GPU experts failed, using CPU fallback\n"); fflush(stderr);
                }

                // CPU fallback but use router-selected experts
                fprintf(stderr, "[DispatchMoeFfn] CPU fallback for selected experts\n"); fflush(stderr);
                // TODO: pass router-selected experts/weights into CPU fallback
                // for weighted multi-expert CPU compute. For now fall through to
                // legacy top-1 CPU fallback.
            }
        }
    }

    fprintf(stderr, "[DispatchMoeFfn] CPU fallback\n"); fflush(stderr);

    // 1. Read FFN input (scratch_norm_) to CPU
    std::vector<float> ffn_in(embed_dim_);
    if (!CopyGpuToCpu(scratch_norm_.buffer, scratch_norm_.offset,
                      ffn_in.data(), embed_dim_ * sizeof(float))) {
        fprintf(stderr, "[DispatchMoeFfn] failed to read FFN input\n"); fflush(stderr);
        return false;
    }

    std::string gate_name = gate_t->name;
    std::string up_name = up_exps_t->name;
    std::string gate_exps_name = gate_exps_t->name;
    std::string down_name = down_exps_t->name;

    // 2. Dequant routing tensor and compute logits
    std::vector<float> gate_inp;
    if (!model_->CpuDequantTensor(gate_name, gate_inp)) return false;
    uint32_t n_experts = 0;
    bool gate_transposed = false;
    {
        auto raw = model_->GetRawTensors();
        for (const auto& t : raw) {
            if (t.name == gate_name) {
                if (t.shape.size() == 2) {
                    if (t.shape[0] == embed_dim_) { n_experts = t.shape[1]; gate_transposed = true; }
                    else { n_experts = t.shape[0]; gate_transposed = false; }
                }
                break;
            }
        }
    }
    if (n_experts == 0) {
        fprintf(stderr, "[DispatchMoeFfn] could not determine expert count\n"); fflush(stderr);
        return false;
    }

    std::vector<float> logits(n_experts, 0.0f);
    for (uint32_t e = 0; e < n_experts; ++e) {
        float sum = 0.0f;
        for (uint32_t d = 0; d < embed_dim_; ++d) {
            uint32_t idx = gate_transposed ? (d * n_experts + e) : (e * embed_dim_ + d);
            sum += ffn_in[d] * gate_inp[idx];
        }
        logits[e] = sum;
    }
    float max_logit = logits[0];
    for (float v : logits) if (v > max_logit) max_logit = v;
    float sum_exp = 0.0f;
    for (float& v : logits) { v = std::exp(v - max_logit); sum_exp += v; }
    for (float& v : logits) v /= sum_exp;

    uint32_t top_e = 0;
    for (uint32_t e = 1; e < n_experts; ++e) if (logits[e] > logits[top_e]) top_e = e;
    fprintf(stderr, "[DispatchMoeFfn] selected expert %u/%u (p=%.3f)\n", top_e, n_experts, logits[top_e]); fflush(stderr);

    // 3. Dequant selected expert weights
    std::vector<float> up_slice, gate_slice, down_slice;
    if (!DequantExpertSlice(up_name, top_e, hidden_dim_, embed_dim_, up_slice)) return false;
    if (!DequantExpertSlice(gate_exps_name, top_e, hidden_dim_, embed_dim_, gate_slice)) return false;
    if (!DequantExpertSlice(down_name, top_e, embed_dim_, hidden_dim_, down_slice)) return false;

    // 4. Compute FFN on CPU
    std::vector<float> up_out(hidden_dim_, 0.0f);
    for (uint32_t h = 0; h < hidden_dim_; ++h) {
        float u = 0.0f, g = 0.0f;
        for (uint32_t d = 0; d < embed_dim_; ++d) {
            u += ffn_in[d] * up_slice[h * embed_dim_ + d];
            g += ffn_in[d] * gate_slice[h * embed_dim_ + d];
        }
        float silu_g = g / (1.0f + std::exp(-g));
        up_out[h] = silu_g * u;
    }
    std::vector<float> ffn_out(embed_dim_, 0.0f);
    for (uint32_t d = 0; d < embed_dim_; ++d) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < hidden_dim_; ++h)
            sum += up_out[h] * down_slice[d * hidden_dim_ + h];
        ffn_out[d] = sum;
    }

    // 5. Upload result back to scratch_norm_
    if (!CopyCpuToGpu(ffn_out.data(), scratch_norm_.buffer, scratch_norm_.offset,
                      embed_dim_ * sizeof(float))) {
        fprintf(stderr, "[DispatchMoeFfn] failed to upload FFN output\n"); fflush(stderr);
        return false;
    }
    return true;
}

} // namespace notllama
