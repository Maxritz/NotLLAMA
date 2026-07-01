#include "engine/vulkan_compute_engine.hpp"
#include "engine/vulkan_shader_library.hpp"
#include "engine/vulkan_descriptor_manager.hpp"
#include "engine/ring_allocator_adapter.hpp"
#include "engine/shader_compiler.hpp"
#include "engine/model_adapter.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace notllama {

VulkanComputeEngine::VulkanComputeEngine(VkDevice device, VkPhysicalDevice physical_device,
 VkQueue queue, uint32_t queue_family,
 IShaderLibrary* shader_lib, IDescriptorManager* desc_mgr,
 IMemoryAllocator* allocator, uint32_t embed_dim)
 : device_(device), physical_device_(physical_device), queue_(queue),
 queue_family_(queue_family), shader_lib_(shader_lib),
 desc_mgr_(desc_mgr), allocator_(allocator), embed_dim_(embed_dim) {

 VkCommandPoolCreateInfo cpci{};
 cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
 cpci.queueFamilyIndex = queue_family_;
 cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
 vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_);

 VkCommandBufferAllocateInfo cbai{};
 cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
 cbai.commandPool = cmd_pool_;
 cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
 cbai.commandBufferCount = 1;
 vkAllocateCommandBuffers(device_, &cbai, &cmd_buffer_);

 VkFenceCreateInfo fci{};
 fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
 fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
 vkCreateFence(device_, &fci, nullptr, &fence_);

 VkBufferCreateInfo bci{};
 bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
 bci.size = 16;
 bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
 bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
 vkCreateBuffer(device_, &bci, nullptr, &topk_output_buf_);

 VkMemoryRequirements mr;
 vkGetBufferMemoryRequirements(device_, topk_output_buf_, &mr);

 VkPhysicalDeviceMemoryProperties memProps;
 vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

 uint32_t memIdx = 0;
 for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
 if ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
 (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
 (mr.memoryTypeBits & (1u << i))) {
 memIdx = i;
 break;
 }
 }

 VkMemoryAllocateInfo mai{};
 mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
 mai.allocationSize = mr.size;
 mai.memoryTypeIndex = memIdx;
 vkAllocateMemory(device_, &mai, nullptr, &topk_output_mem_);
 vkBindBufferMemory(device_, topk_output_buf_, topk_output_mem_, 0);
 vkMapMemory(device_, topk_output_mem_, 0, VK_WHOLE_SIZE, 0,
 reinterpret_cast<void**>(&topk_output_mapped_));

 VkBufferDeviceAddressInfo bda{};
 bda.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
 bda.buffer = topk_output_buf_;
 topk_output_addr_ = vkGetBufferDeviceAddress(device_, &bda);
}

VulkanComputeEngine::~VulkanComputeEngine() {
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
 if (cmd_buffer_ != VK_NULL_HANDLE && cmd_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE)
 vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buffer_);
 cmd_buffer_ = VK_NULL_HANDLE;
 if (cmd_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
 vkDestroyCommandPool(device_, cmd_pool_, nullptr);
 cmd_pool_ = VK_NULL_HANDLE;
 }
 for (auto& [_, seq] : sequences_) {
 if (seq.hidden_state.buffer != VK_NULL_HANDLE && allocator_)
 allocator_->Free(MemoryType::TRANSIENT, seq.hidden_state);
 }
}

void VulkanComputeEngine::SetModel(IModel* model) {
 model_ = model;
 n_layers_ = (uint32_t)model->GetNumLayers();
 head_dim_ = (uint32_t)model->GetHeadDim();
 embed_dim_ = (uint32_t)model->GetEmbeddingDim();
 n_heads_ = (head_dim_ > 0) ? (embed_dim_ / head_dim_) : 0;
 n_kv_heads_ = (uint32_t)model->GetNumKVHeads();

 auto tensors = model->GetWeightTensors();
 vocab_size_ = 0;
 hidden_dim_ = 0;
 for (auto& t : tensors) {
 if (t.num_dims >= 2) {
 if (t.dims[0] > vocab_size_) vocab_size_ = t.dims[0];
 if (t.dims[1] > hidden_dim_ && t.dims[1] < 100000) hidden_dim_ = t.dims[1];
 }
 }

 layer_weights_.clear();
 layer_weight_formats_.clear();
 model_weights_loaded_ = false;
 scratch_allocated_ = false;
}

void VulkanComputeEngine::SetKVCache(rdna4::KVCacheManager* kv_cache, uint32_t max_seq_len) {
 kv_cache_ = kv_cache;
 max_seq_len_ = max_seq_len;
}

bool VulkanComputeEngine::AddSequence(uint32_t seq_id, const std::vector<uint32_t>& tokens) {
 if (sequences_.find(seq_id) != sequences_.end()) return false;
 ActiveSequence seq;
 seq.tokens = tokens;
 seq.position = 0;
 seq.hidden_state = AllocScratch(embed_dim_ * sizeof(float));
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
 fprintf(stderr, "[StepBatch] ENTER phase=%d layer=%u seq=%u\n",
 (int)step_phase_, current_layer_,
 sequences_.count(current_seq_id_) ? sequences_[current_seq_id_].position : 999);
 fflush(stderr);

 if (!EnsureResources()) {
 fprintf(stderr, "[StepBatch] EnsureResources FAILED\n");
 return false;
 }

 if (step_phase_ > StepPhase::TOPK) {
 fprintf(stderr, "[StepBatch] WARNING: phase=%d invalid, reset to IDLE\n", (int)step_phase_);
 fflush(stderr);
 step_phase_ = StepPhase::IDLE;
 }

 if (step_phase_ == StepPhase::VALIDATE) {
 if (!ValidateRmsNormDispatch()) {
 fprintf(stderr, "[StepBatch] RMS_NORM validation failed\n");
 fflush(stderr);
 return false;
 }
 rms_norm_validated_ = true;
 step_phase_ = StepPhase::LOAD_WEIGHTS;
 return true;
 }

 if (step_phase_ == StepPhase::LOAD_WEIGHTS) {
 if (!LoadModelWeights()) {
 fprintf(stderr, "[StepBatch] LoadModelWeights failed\n");
 fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::IDLE;
 return true;
 }

 if (step_phase_ == StepPhase::IDLE) {
 if (sequences_.empty()) {
 fprintf(stderr, "[StepBatch] IDLE but no active sequences\n");
 fflush(stderr);
 return true;
 }
 auto it = sequences_.find(current_seq_id_);
 for (size_t i = 0; i < sequences_.size(); ++i) {
 if (it == sequences_.end() || ++it == sequences_.end()) it = sequences_.begin();
 }
 current_seq_id_ = it->first;

 auto& seq = it->second;
 if (seq.position < seq.tokens.size()) {
 step_phase_ = StepPhase::EMBED;
 fprintf(stderr, "[StepBatch] IDLE -> EMBED seq=%u pos=%u/%zu\n",
 current_seq_id_, seq.position, seq.tokens.size());
 } else {
 step_phase_ = StepPhase::LM_HEAD;
 fprintf(stderr, "[StepBatch] IDLE -> LM_HEAD seq=%u pos=%u (prompt exhausted)\n",
 current_seq_id_, seq.position);
 }
 fflush(stderr);
 current_layer_ = 0;
 return true;
 }

 if (step_phase_ == StepPhase::EMBED) {
 auto& seq = sequences_[current_seq_id_];
 uint32_t token_id = seq.tokens[seq.position];
 fprintf(stderr, "[StepBatch] EMBED token_id=%u at position=%u\n", token_id, seq.position);
 fflush(stderr);
 if (!DispatchEmbed(token_id, seq.position, seq.hidden_state.device_address)) {
 fprintf(stderr, "[StepBatch] EMBED dispatch failed\n");
 fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_RMS_ATTN;
 current_layer_ = 0;
 return true;
 }

 if (step_phase_ >= StepPhase::LAYER_RMS_ATTN && step_phase_ <= StepPhase::LAYER_FFN_RESIDUAL) {
 auto& seq = sequences_[current_seq_id_];
 const auto& lw = layer_weights_[current_layer_];
 const auto& lf = layer_weight_formats_[current_layer_];

 VkDeviceAddress k_cache = 0, v_cache = 0;
 if (kv_cache_) {
 k_cache = kv_cache_->GetKCacheAddress(current_layer_, current_seq_id_);
 v_cache = kv_cache_->GetVCacheAddress(current_layer_, current_seq_id_);
 }

 if (step_phase_ == StepPhase::LAYER_RMS_ATTN) {
 if (!DispatchRmsNorm(seq.hidden_state.device_address, lw[0],
 scratch_norm_.device_address, embed_dim_, 1)) {
 fprintf(stderr, "[StepBatch] LAYER_RMS_ATTN failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_QKV_Q;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_QKV_Q) {
 if (!DispatchGemm(scratch_norm_.device_address, lw[1], scratch_q_.device_address,
 1, n_heads_ * head_dim_, embed_dim_, true, lf[1])) {
 fprintf(stderr, "[StepBatch] LAYER_QKV_Q failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_QKV_K;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_QKV_K) {
 if (!DispatchGemm(scratch_norm_.device_address, lw[2], scratch_k_.device_address,
 1, n_kv_heads_ * head_dim_, embed_dim_, true, lf[2])) {
 fprintf(stderr, "[StepBatch] LAYER_QKV_K failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_QKV_V;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_QKV_V) {
 if (!DispatchGemm(scratch_norm_.device_address, lw[3], scratch_v_.device_address,
 1, n_kv_heads_ * head_dim_, embed_dim_, true, lf[3])) {
 fprintf(stderr, "[StepBatch] LAYER_QKV_V failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_ROPE;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_ROPE) {
 if (!DispatchRope(scratch_q_.device_address, scratch_k_.device_address,
 seq.position, head_dim_, n_heads_, n_kv_heads_,
 model_->GetRoPEBase(), model_->GetRoPEScale())) {
 fprintf(stderr, "[StepBatch] LAYER_ROPE failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_KV_CACHE_WRITE;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_KV_CACHE_WRITE) {
 if (k_cache && v_cache) {
 if (!DispatchKvCacheWrite(scratch_k_.device_address, scratch_v_.device_address,
 k_cache, v_cache, seq.position, max_seq_len_)) {
 fprintf(stderr, "[StepBatch] LAYER_KV_CACHE_WRITE failed\n"); fflush(stderr);
 return false;
 }
 }
 step_phase_ = StepPhase::LAYER_FLASH_ATTN;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_FLASH_ATTN) {
 uint32_t seq_len = seq.position + 1;
 if (!DispatchFlashAttn(scratch_q_.device_address, k_cache, v_cache,
 scratch_attn_.device_address, seq_len, 1)) {
 fprintf(stderr, "[StepBatch] LAYER_FLASH_ATTN failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_ATTN_OUT;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_ATTN_OUT) {
 if (!DispatchGemm(scratch_attn_.device_address, lw[4], scratch_norm_.device_address,
 1, embed_dim_, n_heads_ * head_dim_, true, lf[4])) {
 fprintf(stderr, "[StepBatch] LAYER_ATTN_OUT failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_ATTN_RESIDUAL;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_ATTN_RESIDUAL) {
 if (!DispatchAdd(seq.hidden_state.device_address, scratch_norm_.device_address,
 seq.hidden_state.device_address, embed_dim_)) {
 fprintf(stderr, "[StepBatch] LAYER_ATTN_RESIDUAL failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_RMS_FFN;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_RMS_FFN) {
 if (!DispatchRmsNorm(seq.hidden_state.device_address, lw[5],
 scratch_norm_.device_address, embed_dim_, 1)) {
 fprintf(stderr, "[StepBatch] LAYER_RMS_FFN failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_FFN_UP;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_FFN_UP) {
 if (!DispatchGemm(scratch_norm_.device_address, lw[6], scratch_ffn_.device_address,
 1, hidden_dim_, embed_dim_, true, lf[6])) {
 fprintf(stderr, "[StepBatch] LAYER_FFN_UP failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_FFN_GATE;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_FFN_GATE) {
 if (!DispatchGemm(scratch_norm_.device_address, lw[7], scratch_ffn_gate_.device_address,
 1, hidden_dim_, embed_dim_, true, lf[7])) {
 fprintf(stderr, "[StepBatch] LAYER_FFN_GATE failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_SILU_MUL;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_SILU_MUL) {
 if (!DispatchSiluMul(scratch_ffn_gate_.device_address, scratch_ffn_.device_address,
 scratch_ffn_.device_address, hidden_dim_)) {
 fprintf(stderr, "[StepBatch] LAYER_SILU_MUL failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_FFN_DOWN;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_FFN_DOWN) {
 if (!DispatchGemm(scratch_ffn_.device_address, lw[8], scratch_norm_.device_address,
 1, embed_dim_, hidden_dim_, true, lf[8])) {
 fprintf(stderr, "[StepBatch] LAYER_FFN_DOWN failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_FFN_RESIDUAL;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_FFN_RESIDUAL) {
 if (!DispatchAdd(seq.hidden_state.device_address, scratch_norm_.device_address,
 seq.hidden_state.device_address, embed_dim_)) {
 fprintf(stderr, "[StepBatch] LAYER_FFN_RESIDUAL failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::LAYER_NEXT;
 return true;
 }

 if (step_phase_ == StepPhase::LAYER_NEXT) {
 ++current_layer_;
 if (current_layer_ >= n_layers_) {
 step_phase_ = StepPhase::LM_HEAD;
 } else {
 step_phase_ = StepPhase::LAYER_RMS_ATTN;
 }
 return true;
 }
 }

 if (step_phase_ == StepPhase::LM_HEAD) {
 auto& seq = sequences_[current_seq_id_];
 fprintf(stderr, "[StepBatch] LM_HEAD dispatch seq=%u\n", current_seq_id_);
 fflush(stderr);
 if (!DispatchGemm(seq.hidden_state.device_address, addr_lm_head_, scratch_logits_.device_address,
 1, vocab_size_, embed_dim_, true, lm_head_dtype_)) {
 fprintf(stderr, "[StepBatch] LM_HEAD failed\n"); fflush(stderr);
 return false;
 }
 step_phase_ = StepPhase::TOPK;
 return true;
 }

 if (step_phase_ == StepPhase::TOPK) {
 fprintf(stderr, "[StepBatch] TOPK dispatch\n"); fflush(stderr);
 if (!DispatchTopK(scratch_logits_.device_address, topk_output_addr_, 0)) {
 fprintf(stderr, "[StepBatch] TOPK dispatch failed\n"); fflush(stderr);
 return false;
 }

 if (topk_output_mem_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
 VkMappedMemoryRange range{};
 range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
 range.memory = topk_output_mem_;
 range.offset = 0;
 range.size = VK_WHOLE_SIZE;
 vkInvalidateMappedMemoryRanges(device_, 1, &range);
 }

 uint32_t sampled = topk_output_mapped_[0];
 if (sampled == 0 && topk_output_mapped_[1] != 0) {
 sampled = topk_output_mapped_[1];
 fprintf(stderr, "[StepBatch] TOPK fallback: offset 1 -> token=%u\n", sampled);
 fflush(stderr);
 }
 if (sampled == 0) {
 xors_state_ = xors_state_ * 1103515245 + 12345;
 sampled = (uint32_t)(xors_state_ % vocab_size_);
 if (sampled == 0) sampled = 1;
 fprintf(stderr, "[StepBatch] TOPK random fallback: token=%u\n", sampled);
 fflush(stderr);
 }

 last_token_id_ = sampled;
 auto& seq = sequences_[current_seq_id_];
 seq.tokens.push_back(sampled);
 ++seq.position;
 fprintf(stderr, "[StepBatch] TOPK sampled token=%u seq_pos=%u seq_len=%zu\n",
 sampled, seq.position, seq.tokens.size());
 fflush(stderr);

 step_phase_ = StepPhase::IDLE;
 return true;
 }

 fprintf(stderr, "[StepBatch] FATAL: unhandled phase=%d\n", (int)step_phase_);
 fflush(stderr);
 return false;
}

void VulkanComputeEngine::SetMaxUtilization(float percent) { max_utilization_ = percent; }
void VulkanComputeEngine::Throttle() {}
void VulkanComputeEngine::StartWatchdog() { watchdog_running_ = true; }
void VulkanComputeEngine::StopWatchdog() { watchdog_running_ = false; }
WatchdogStatus VulkanComputeEngine::GetLastFrameStatus() { return last_status_; }
void VulkanComputeEngine::ResetExecutionEngine() {
 step_phase_ = StepPhase::VALIDATE;
 current_layer_ = 0;
 current_seq_id_ = 0;
 next_seq_index_ = 0;
}
void VulkanComputeEngine::EnableProfiling(bool enable) { profiling_ = enable; }

bool VulkanComputeEngine::EnsureResources() {
 if (!rms_norm_validated_) {
 if (!ValidateRmsNormDispatch()) return false;
 rms_norm_validated_ = true;
 }
 if (!model_weights_loaded_) {
 if (!LoadModelWeights()) return false;
 }
 if (!scratch_allocated_) {
 if (!AllocScratchBuffers()) return false;
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
 scratch_norm_ = AllocScratch(embed_dim_ * sizeof(float));
 scratch_ffn_ = AllocScratch(ffn_size);
 scratch_ffn_gate_ = AllocScratch(ffn_size);
 scratch_logits_ = AllocScratch(logits_size);
 rms_weight_ = AllocScratch(embed_dim_ * sizeof(float));

 scratch_allocated_ = scratch_q_.buffer && scratch_k_.buffer && scratch_v_.buffer &&
 scratch_attn_.buffer && scratch_norm_.buffer &&
 scratch_ffn_.buffer && scratch_ffn_gate_.buffer &&
 scratch_logits_.buffer && rms_weight_.buffer;
 rms_weight_loaded_ = scratch_allocated_;
 return scratch_allocated_;
}

bool VulkanComputeEngine::LoadModelWeights() {
 if (model_weights_loaded_) return true;
 if (!model_) return false;

 auto raw = model_->GetRawTensors();
 auto meta = model_->GetWeightTensors();
 layer_weights_.resize(n_layers_);
 layer_weight_formats_.resize(n_layers_);
 for (auto& lw : layer_weights_)
 for (auto& w : lw) w = 0;
 for (auto& lf : layer_weight_formats_)
 for (auto& f : lf) f = DataType::F32;

 size_t meta_idx = 0;
 for (auto& t : raw) {
 const std::string& name = t.name;
 int layer = -1;
 int slot = -1;

 if (name.find("token_embd") != std::string::npos) {
 addr_embed_ = t.gpuAddress;
 ++meta_idx;
 continue;
 }

 if ((name.find("output") != std::string::npos && name.find("norm") == std::string::npos) ||
 name.find("head") != std::string::npos) {
 addr_lm_head_ = t.gpuAddress;
 if (meta_idx < meta.size()) {
 lm_head_dtype_ = meta[meta_idx].dtype;
 }
 ++meta_idx;
 continue;
 }

 if (name.find("blk.") != std::string::npos) {
 size_t dot = name.find(".");
 size_t dot2 = name.find(".", dot + 1);
 if (dot != std::string::npos && dot2 != std::string::npos) {
 layer = std::stoi(name.substr(dot + 1, dot2 - dot - 1));
 }

 if (name.find("attn_norm") != std::string::npos) slot = 0;
 else if (name.find("attn_q") != std::string::npos) slot = 1;
 else if (name.find("attn_k") != std::string::npos) slot = 2;
 else if (name.find("attn_v") != std::string::npos) slot = 3;
 else if (name.find("attn_output") != std::string::npos) slot = 4;
 else if (name.find("ffn_norm") != std::string::npos) slot = 5;
 else if (name.find("ffn_up") != std::string::npos) slot = 6;
 else if (name.find("ffn_gate") != std::string::npos) slot = 7;
 else if (name.find("ffn_down") != std::string::npos) slot = 8;
 }

 if (layer >= 0 && layer < (int)n_layers_ && slot >= 0 && slot < WEIGHTS_PER_LAYER) {
 layer_weights_[layer][slot] = t.gpuAddress;
 if (meta_idx < meta.size()) {
 layer_weight_formats_[layer][slot] = meta[meta_idx].dtype;
 }
 }

 ++meta_idx;
 }

 model_weights_loaded_ = true;
 return true;
}

KernelType VulkanComputeEngine::GetGemmKernelType(DataType dtype) {
 switch (dtype) {
 case DataType::Q4_0: return KernelType::GEMM_Q4_0;
 case DataType::Q8_0: return KernelType::GEMM_Q8_0;
 case DataType::Q4_K: return KernelType::GEMM_Q4K;
 case DataType::Q6_K: return KernelType::GEMM_Q6K;
 case DataType::F32:
 case DataType::F16:
 default:
 return KernelType::COOPMAT_GEMM;
 }
}

bool VulkanComputeEngine::DispatchEmbed(uint32_t token_id, uint32_t token_pos, VkDeviceAddress hidden_addr) {
 (void)token_pos;
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::EMBED, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::EMBED, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::EmbedPushConstants push{};
 push.addrEmbed = addr_embed_;
 push.addrHidden = hidden_addr;
 push.tokenId = token_id;
 push.embedDim = embed_dim_;
 push.vocabSize = vocab_size_;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, (embed_dim_ + 255) / 256, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchRmsNorm(VkDeviceAddress in_addr, VkDeviceAddress weight_addr,
 VkDeviceAddress out_addr, uint32_t row_size, uint32_t n_rows) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::RmsNormPushConstants push{};
 push.addrIn = in_addr;
 push.addrOut = out_addr;
 push.addrWeight = weight_addr;
 push.rowSize = row_size;
 push.nRows = n_rows;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, n_rows, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchGemm(VkDeviceAddress a_addr, VkDeviceAddress b_addr, VkDeviceAddress c_addr,
 uint32_t M, uint32_t N, uint32_t K, bool transB, DataType weight_dtype) {
 KernelType kernel_type = GetGemmKernelType(weight_dtype);
 bool actual_transB = transB;
 if (kernel_type != KernelType::COOPMAT_GEMM) {
 actual_transB = false;
 }

 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(kernel_type, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(kernel_type, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::GemmPushConstants push{};
 push.addrA = a_addr;
 push.addrB = b_addr;
 push.addrC = c_addr;
 push.M = M;
 push.N = N;
 push.K = K;
 push.alpha = 1.0f;
 push.transB = actual_transB ? 1u : 0u;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

 if (kernel_type == KernelType::COOPMAT_GEMM) {
 vkCmdDispatch(cmd_buffer_, N, M, 1);
 } else if (kernel_type == KernelType::GEMM_Q4_0) {
 vkCmdDispatch(cmd_buffer_, (N + 255) / 256, 1, 1);
 } else {
 vkCmdDispatch(cmd_buffer_, N, 1, 1);
 }

 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchAdd(VkDeviceAddress a_addr, VkDeviceAddress b_addr, VkDeviceAddress c_addr,
 uint32_t n_elements) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::ADD, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::ADD, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::AddPushConstants push{};
 push.addrA = a_addr;
 push.addrB = b_addr;
 push.addrC = c_addr;
 push.nElements = n_elements;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, (n_elements + 255) / 256, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchRope(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
 uint32_t position, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
 float rope_base, float rope_scale) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 spec.head_dim = head_dim;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::ROPE, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::ROPE, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::RopePushConstants push{};
 push.addrQ = q_addr;
 push.addrK = k_addr;
 push.seqLen = position + 1;
 push.headDim = head_dim;
 push.nHeads = n_heads;
 push.nKvHeads = n_kv_heads;
 push.ropeBase = rope_base;
 push.ropeScale = rope_scale;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

 uint32_t pairs_per_head = head_dim / 2;
 uint32_t total_pairs = n_heads * pairs_per_head;
 vkCmdDispatch(cmd_buffer_, (total_pairs + 31) / 32, 1, 1);

 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchFlashAttn(VkDeviceAddress q_addr, VkDeviceAddress k_addr,
 VkDeviceAddress v_addr, VkDeviceAddress out_addr,
 uint32_t seq_len, uint32_t n_q_rows) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 spec.head_dim = head_dim_;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::FlashAttnPushConstants push{};
 push.addrQ = q_addr;
 push.addrK = k_addr;
 push.addrV = v_addr;
 push.addrOut = out_addr;
 push.seqLen = seq_len;
 push.nQRows = n_q_rows;
 push.headDim = head_dim_;
 push.nHeads = n_heads_;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, n_heads_, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchSiluMul(VkDeviceAddress gate_addr, VkDeviceAddress up_addr,
 VkDeviceAddress out_addr, uint32_t n_elements) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::SILU_MUL, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::SILU_MUL, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::SiluMulPushConstants push{};
 push.addrGate = gate_addr;
 push.addrUp = up_addr;
 push.addrOut = out_addr;
 push.nElements = n_elements;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, (n_elements + 255) / 256, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchTopK(VkDeviceAddress logits_addr, VkDeviceAddress output_addr,
 VkDeviceAddress scratch_addr) {
 (void)scratch_addr;
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::TOPK, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::TOPK, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::TopKPushConstants push{};
 push.addrLogits = logits_addr;
 push.addrOutput = output_addr;
 push.vocabSize = vocab_size_;
 push.k = 1;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, 1, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::DispatchKvCacheWrite(VkDeviceAddress k_in, VkDeviceAddress v_in,
 VkDeviceAddress k_cache, VkDeviceAddress v_cache,
 uint32_t seq_pos, uint32_t max_seq) {
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::KV_CACHE_WRITE, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) return false;

 rdna4::KvCacheWritePushConstants push{};
 push.addrKIn = k_in;
 push.addrVIn = v_in;
 push.addrKCache = k_cache;
 push.addrVCache = v_cache;
 push.seqPos = seq_pos;
 push.maxSeqLen = max_seq;
 push.nKvHeads = n_kv_heads_;
 push.headDim = head_dim_;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, n_kv_heads_, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) return false;
 vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
 vkResetFences(device_, 1, &fence_);
 return true;
}

bool VulkanComputeEngine::ValidateRmsNormDispatch() {
 if (!rms_weight_loaded_) return false;
 SpecializationMap spec{};
 spec.subgroup_size = 32;
 VkPipelineLayout pl = shader_lib_->GetPipelineLayout(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
 VkPipeline pipeline = shader_lib_->GetPipeline(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
 if (!pipeline || !pl) {
 fprintf(stderr, "[ValidateRmsNorm] Pipeline not found\n");
 fflush(stderr);
 return false;
 }

 rdna4::RmsNormPushConstants push{};
 push.addrIn = rms_weight_.device_address;
 push.addrOut = rms_weight_.device_address;
 push.addrWeight = rms_weight_.device_address;
 push.rowSize = 1;
 push.nRows = 1;

 VkCommandBufferBeginInfo bi{};
 bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
 vkBeginCommandBuffer(cmd_buffer_, &bi);
 vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
 vkCmdPushConstants(cmd_buffer_, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
 vkCmdDispatch(cmd_buffer_, 1, 1, 1);
 vkEndCommandBuffer(cmd_buffer_);

 VkSubmitInfo si{};
 si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
 si.commandBufferCount = 1;
 si.pCommandBuffers = &cmd_buffer_;
 if (vkQueueSubmit(queue_, 1, &si, fence_) != VK_SUCCESS) {
 fprintf(stderr, "[ValidateRmsNorm] Queue submit failed\n");
 fflush(stderr);
 return false;
 }
 VkResult wait = vkWaitForFences(device_, 1, &fence_, VK_TRUE, 1000000000);
 if (wait != VK_SUCCESS) {
 fprintf(stderr, "[ValidateRmsNorm] Fence wait failed: %d\n", wait);
 fflush(stderr);
 return false;
 }
 vkResetFences(device_, 1, &fence_);
 return true;
}

} // namespace notllama
