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
    if (topk_output_mapped_) vkUnmapMemory(device_, topk_output_mem_);
    if (topk_output_buf_) vkDestroyBuffer(device_, topk_output_buf_, nullptr);
    if (topk_output_mem_) vkFreeMemory(device_, topk_output_mem_, nullptr);
    if (fence_ != VK_NULL_HANDLE) vkDestroyFence(device_, fence_, nullptr);
    if (cmd_buffer_ != VK_NULL_HANDLE && cmd_pool_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buffer_);
    if (cmd_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    for (auto& [_, seq] : sequences_) {
        if (seq.hidden_state.buffer != VK_NULL_HANDLE && allocator_)
            allocator_->Free(MemoryType::TRANSIENT, seq.hidden_state);
    }
}

void VulkanComputeEngine::SetModel(IModel* model) {
    model_ = model;
    n_layers_ = (uint32_t)model->GetNumLayers();
    n_heads_ = (uint32_t)(model->GetHeadDim() ? model->GetNumKVHeads() : 0);
    n_kv_heads_ = (uint32_t)model->GetNumKVHeads();
    head_dim_ = (uint32_t)model->GetHeadDim();
    auto tensors = model->GetWeightTensors();
    vocab_size_ = 0;
    hidden_dim_ = 0;
    for (auto& t : tensors) {
        if (t.num_dims >= 2) {
            if (t.dims[0] > vocab_size_) vocab_size_ = t.dims[0];
            if (t.dims[1] > hidden_dim_ && t.dims[1] < 100000) hidden_dim_ = t.dims[1];
        }
    }
    embed_dim_ = (uint32_t)model->GetEmbeddingDim();
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
    size_t ffn_size = hidden_dim_ * sizeof(float);
    size_t logits_size = vocab_size_ * sizeof(float);

    scratch_q_ = AllocScratch(q_size);
    scratch_k_ = AllocScratch(kv_size);
    scratch_v_ = AllocScratch(kv_size);
    scratch_attn_ = AllocScratch(attn_size);
    scratch_ffn_ = AllocScratch(ffn_size);
    scratch_ffn_gate_ = AllocScratch(ffn_size);
    scratch_logits_ = AllocScratch(logits_size);
    rms_weight_ = AllocScratch(embed_dim_ * sizeof(float));

    scratch_allocated_ = scratch_q_.buffer && scratch_k_.buffer && scratch_v_.buffer &&
                         scratch_attn_.buffer && scratch_ffn_.buffer && scratch_ffn_gate_.buffer &&
                         scratch_logits_.buffer && rms_weight_.buffer;
    rms_weight_loaded_ = scratch_allocated_;
    return scratch_allocated_;
}

bool VulkanComputeEngine::LoadModelWeights() {
    if (model_weights_loaded_) return true;
    if (!model_) return false;

    auto raw = model_->GetRawTensors();
    layer_weights_.resize(n_layers_);
    for (auto& lw : layer_weights_)
        for (auto& w : lw) w = 0;

    for (auto& t : raw) {
        const auto& name = t.name;
        if (name.find("token_embd") != std::string::npos ||
            name.find("tok_embeddings") != std::string::npos ||
            name.find("embed") != std::string::npos) {
            addr_embed_ = t.gpuAddress;
            continue;
        }
        if ((name.find("output") != std::string::npos && name.find("norm") == std::string::npos) ||
            name.find("head") != std::string::npos) {
            addr_lm_head_ = t.gpuAddress;
            continue;
        }
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
        if (name.find("attn_norm") != std::string::npos) slot = 0;
        else if (name.find("attn_q") != std::string::npos || name.find("q_proj") != std::string::npos) slot = 1;
        else if (name.find("attn_k") != std::string::npos || name.find("k_proj") != std::string::npos) slot = 2;
        else if (name.find("attn_v") != std::string::npos || name.find("v_proj") != std::string::npos) slot = 3;
        else if (name.find("attn_output") != std::string::npos || name.find("o_proj") != std::string::npos) slot = 4;
        else if (name.find("ffn_norm") != std::string::npos) slot = 5;
        else if (name.find("ffn_up") != std::string::npos || name.find("up_proj") != std::string::npos) slot = 6;
        else if (name.find("ffn_gate") != std::string::npos || name.find("gate_proj") != std::string::npos) slot = 7;
        else if (name.find("ffn_down") != std::string::npos || name.find("down_proj") != std::string::npos) slot = 8;
        if (slot >= 0 && slot < WEIGHTS_PER_LAYER) {
            layer_weights_[layer][slot] = t.gpuAddress;
        }
    }

    model_weights_loaded_ = true;
    return true;
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
    if (!EnsureResources()) return false;

    if (step_phase_ == StepPhase::VALIDATE) {
        rms_norm_validated_ = ValidateRmsNormDispatch();
        if (!rms_norm_validated_) return false;
        step_phase_ = StepPhase::LOAD_WEIGHTS;
        return true;
    }

    if (step_phase_ == StepPhase::LOAD_WEIGHTS) {
        if (!LoadModelWeights()) return false;
        if (!AllocScratchBuffers()) return false;
        step_phase_ = StepPhase::IDLE;
        return true;
    }

    if (sequences_.empty()) return false;

    // Round-robin sequence selection
    struct SeqOrder { uint32_t id; ActiveSequence* ptr; };
    std::vector<SeqOrder> pending;
    for (auto& [id, s] : sequences_)
        if (s.position < s.tokens.size())
            pending.push_back({id, &s});
    if (pending.empty()) return false;
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
    // Each layer: RMS_ATTN → Q_K_V GEMMs → FLASH_ATTN → ATTN_OUT → RMS_FFN → FFN_UP/GATE → SILU_MUL → FFN_DOWN
    if (current_layer_ < n_layers_ && step_phase_ >= StepPhase::LAYER_RMS_ATTN &&
                                      step_phase_ <= StepPhase::LAYER_NEXT) {
        auto& lw = layer_weights_[current_layer_];

        if (step_phase_ == StepPhase::LAYER_RMS_ATTN) {
            if (!DispatchRmsNorm(seq->hidden_state.device_address, lw[0],
                                  seq->hidden_state.device_address, embed_dim_, 1))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_Q;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_Q) {
            if (!DispatchGemm(seq->hidden_state.device_address, lw[1], scratch_q_.device_address,
                               1, n_heads_ * head_dim_, embed_dim_, true))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_K;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_K) {
            if (!DispatchGemm(seq->hidden_state.device_address, lw[2], scratch_k_.device_address,
                               1, n_kv_heads_ * head_dim_, embed_dim_, true))
                return false;
            step_phase_ = StepPhase::LAYER_QKV_V;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_QKV_V) {
            if (!DispatchGemm(seq->hidden_state.device_address, lw[3], scratch_v_.device_address,
                               1, n_kv_heads_ * head_dim_, embed_dim_, true))
                return false;
            step_phase_ = StepPhase::LAYER_KV_CACHE_WRITE;
            return true;
        }

        // Write K/V from flat scratch buffers to tiled KV cache,
        // reconciling the layout mismatch between GEMM output and cache.
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
            // Read Q from scratch (flat, per-head), K/V from tiled cache
            if (!DispatchFlashAttn(scratch_q_.device_address, k_cache, v_cache,
                                    scratch_attn_.device_address,
                                    kv_cache_->getSeqLen(current_layer_), n_heads_))
                return false;
            step_phase_ = StepPhase::LAYER_ATTN_OUT;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_ATTN_OUT) {
            // attention output projection + residual
            if (!DispatchGemm(scratch_attn_.device_address, lw[4],
                               scratch_attn_.device_address,
                               1, embed_dim_, n_heads_ * head_dim_, true))
                return false;
            // For now, attn_out overwrites the scratch; residual add is skipped.
            // TODO: add residual: hidden_state += attn_out
            step_phase_ = StepPhase::LAYER_RMS_FFN;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_RMS_FFN) {
            if (!DispatchRmsNorm(seq->hidden_state.device_address, lw[5],
                                  seq->hidden_state.device_address, embed_dim_, 1))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_UP;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_UP) {
            if (!DispatchGemm(seq->hidden_state.device_address, lw[6], scratch_ffn_.device_address,
                               1, hidden_dim_, embed_dim_, true))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_GATE;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_GATE) {
            if (!DispatchGemm(seq->hidden_state.device_address, lw[7], scratch_ffn_gate_.device_address,
                               1, hidden_dim_, embed_dim_, true))
                return false;
            step_phase_ = StepPhase::LAYER_SILU_MUL;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_SILU_MUL) {
            // SiLU(gate) * up → ffn buffer
            if (!DispatchSiluMul(scratch_ffn_gate_.device_address, scratch_ffn_.device_address,
                                  scratch_ffn_.device_address, hidden_dim_))
                return false;
            step_phase_ = StepPhase::LAYER_FFN_DOWN;
            return true;
        }

        if (step_phase_ == StepPhase::LAYER_FFN_DOWN) {
            if (!DispatchGemm(scratch_ffn_.device_address, lw[8],
                               seq->hidden_state.device_address,
                               1, embed_dim_, hidden_dim_, true))
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
                           1, vocab_size_, embed_dim_, true))
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
        // Read sampled token from host-visible output buffer
        if (topk_output_mapped_) {
            last_token_id_ = topk_output_mapped_[0];
        } else {
            last_token_id_ = 0;
        }
        seq->position++;
        step_phase_ = StepPhase::IDLE;
        if (seq->position >= seq->tokens.size()) {
            step_phase_ = StepPhase::DONE;
        }
        return true;
    }

    return step_phase_ == StepPhase::DONE;
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
                                        bool transB) {
    SpecializationMap spec{}; spec.subgroup_size = 32;
    VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::COOPMAT_GEMM, PipelineVariant::FAST, spec);
    VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::COOPMAT_GEMM, PipelineVariant::FAST, spec);
    if (!pipeline || !pl) return false;

    rdna4::GemmPushConstants push{};
    push.addrA = a_addr; push.addrB = b_addr; push.addrC = c_addr;
    push.M = M; push.N = N; push.K = K; push.alpha = 1.0f; push.transB = transB ? 1u : 0u;

    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd_buffer_, &bi);
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd_buffer_, N, M, 1);
    vkEndCommandBuffer(cmd_buffer_);

    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_buffer_;
    if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
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

} // namespace notllama
