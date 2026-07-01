#include "engine/vulkan_shader_library.hpp"
#include "engine/idevice.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

namespace notllama {

namespace fs = std::filesystem;

bool PipelineKey::operator==(const PipelineKey& other) const noexcept {
    return type == other.type &&
           variant == other.variant &&
           spec.tile_m == other.spec.tile_m &&
           spec.tile_n == other.spec.tile_n &&
           spec.tile_k == other.spec.tile_k &&
           spec.workgroup_x == other.spec.workgroup_x &&
           spec.workgroup_y == other.spec.workgroup_y &&
           spec.head_dim == other.spec.head_dim &&
           spec.subgroup_size == other.spec.subgroup_size &&
           spec.accum_type == other.spec.accum_type;
}

size_t PipelineKeyHash::operator()(const PipelineKey& k) const noexcept {
    size_t h = static_cast<size_t>(k.type);
    h ^= static_cast<size_t>(k.variant) << 4;
    h ^= static_cast<size_t>(k.spec.tile_m) << 8;
    h ^= static_cast<size_t>(k.spec.tile_n) << 12;
    h ^= static_cast<size_t>(k.spec.tile_k) << 16;
    h ^= static_cast<size_t>(k.spec.workgroup_x) << 20;
    h ^= static_cast<size_t>(k.spec.workgroup_y) << 24;
    h ^= static_cast<size_t>(k.spec.head_dim) << 28;
    h ^= static_cast<size_t>(k.spec.subgroup_size) << 2;
    h ^= static_cast<size_t>(k.spec.accum_type) << 6;
    return h;
}

VulkanShaderLibrary::VulkanShaderLibrary(VkDevice device, const std::string& shader_dir,
                                                 const VendorProfile& profile)
    : device_(device), shader_dir_(shader_dir), profile_(profile) {}

VulkanShaderLibrary::~VulkanShaderLibrary() {
    for (auto& [_, layout] : named_layouts_) {
        if (layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout, nullptr);
    }
    for (auto& [_, pipeline] : named_pipelines_) {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline, nullptr);
    }
    for (auto& [_, layout] : layouts_) {
        if (layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout, nullptr);
    }
    for (auto& [_, pipeline] : pipelines_) {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline, nullptr);
    }
    for (auto& [_, module] : modules_) {
        if (module != VK_NULL_HANDLE) vkDestroyShaderModule(device_, module, nullptr);
    }
    if (pipeline_cache_ != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
}

bool VulkanShaderLibrary::LoadPipelineCache(const std::string& path) {
    std::vector<char> data;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        size_t size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        data.resize(size);
        file.read(data.data(), size);
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (!data.empty()) {
        ci.initialDataSize = data.size();
        ci.pInitialData = data.data();
    }

    return vkCreatePipelineCache(device_, &ci, nullptr, &pipeline_cache_) == VK_SUCCESS;
}

std::string VulkanShaderLibrary::KernelTypeToString(KernelType type) const {
    switch (type) {
        case KernelType::RMS_NORM: return "rms_norm";
        case KernelType::RMS_NORM_MUL: return "rms_norm_mul";
        case KernelType::ADD_RMS_NORM: return "add_rms_norm";
        case KernelType::QKV_PROJ: return "gemm";
        case KernelType::ROPE: return "rope";
        case KernelType::FLASH_ATTN: return "flash_attention";
        case KernelType::ATTENTION: return "attention";
        case KernelType::MLP_UP: return "gemm";
        case KernelType::MLP_GATE: return "gemm";
        case KernelType::MLP_DOWN: return "gemm";
        case KernelType::MLP_FUSED_GATEUP: return "mlp_fused_gateup";
        case KernelType::LM_HEAD: return "gemm";
        case KernelType::EMBED: return "embed";
        case KernelType::EMBED_Q8_0: return "embed_q8_0";
        case KernelType::DEQUANT: return "dequantize";
        case KernelType::DEQUANT_TURBO: return "dequant_turbo";
        case KernelType::DEQUANTIZE_TEST: return "dequantize_test";
        case KernelType::KV_CACHE_WRITE: return "kv_cache_write";
        case KernelType::KV_CACHE_QUANT: return "kv_cache_quantize";
        case KernelType::KV_CACHE_QUANT_Q8_0: return "kv_cache_quantize_q8_0";
        case KernelType::KV_CACHE_DEQUANT: return "kv_cache_dequant";
        case KernelType::COPY_F32_F16: return "cpy_f32_f16";
        case KernelType::COPY_F32_BF16: return "cpy_f32_bf16";
        case KernelType::COPY_F16_F32: return "cpy_f16_f32";
        case KernelType::COPY_BF16_F32: return "cpy_bf16_f32";
        case KernelType::SILU_MUL: return "silu_mul";
        case KernelType::TOPK: return "topk";
        case KernelType::ADD: return "add";
        case KernelType::COMPRESS_CONTEXT: return "compress_context";
        case KernelType::GEMM: return "gemm";
        case KernelType::GEMM_Q4_0: return "gemm_q4_0";
        case KernelType::GEMM_Q8_0: return "gemm_q8_0";
        case KernelType::GEMM_Q6K: return "gemm_q6k";
        case KernelType::GEMM_Q4K: return "gemm_q4k";
        case KernelType::GEMM_TURBO: return "gemm_turbo";
        case KernelType::MATMUL_Q4_0: return "matmul_q4_0";
        case KernelType::MATMUL_Q8_0: return "matmul_q8_0";
        case KernelType::COOPMAT_GEMM: return "cooperative_gemm";
        case KernelType::BDA_TEST: return "bda_test";
        case KernelType::KERNEL_ENTRY: return "kernel_entry";
    }
    return "unknown";
}

std::string VulkanShaderLibrary::FusedKernelTypeToString(FusedKernelType type) const {
    switch (type) {
        case FusedKernelType::RMS_QKV: return "rms_qkv";
        case FusedKernelType::ROPE_FLASHATTN: return "rope_flashattn";
        case FusedKernelType::MLP_UPGATE: return "mlp_upgate";
        case FusedKernelType::MLP_DOWN_RESIDUAL: return "mlp_down_residual";
    }
    return "unknown";
}

std::string VulkanShaderLibrary::GetShaderPath(KernelType type) const {
    return shader_dir_ + "/" + KernelTypeToString(type) + ".spv";
}

std::string VulkanShaderLibrary::GetShaderPath(FusedKernelType type) const {
    return shader_dir_ + "/" + FusedKernelTypeToString(type) + ".spv";
}

std::vector<std::string> VulkanShaderLibrary::BuildDefines(const SpecializationMap& spec) const {
    std::vector<std::string> defs;
    defs.push_back("SUBGROUP_SIZE=" + std::to_string(spec.subgroup_size));
    defs.push_back("TILE_M=" + std::to_string(spec.tile_m));
    defs.push_back("TILE_N=" + std::to_string(spec.tile_n));
    defs.push_back("TILE_K=" + std::to_string(spec.tile_k));
    defs.push_back("HEAD_DIM=" + std::to_string(spec.head_dim));
    defs.push_back("WORKGROUP_X=" + std::to_string(spec.workgroup_x));
    defs.push_back("WORKGROUP_Y=" + std::to_string(spec.workgroup_y));
    defs.push_back("ACCUM_TYPE=" + std::to_string(static_cast<uint32_t>(spec.accum_type)));
    switch (profile_.vendor) {
        case VendorID::AMD:    defs.push_back("VENDOR_AMD=1"); break;
        case VendorID::NVIDIA: defs.push_back("VENDOR_NVIDIA=1"); break;
        default: break;
    }
    // Cooperative matrix is a dead end on AMD Vulkan — glslc SDK 1.4.350 doesn't
    // support GL_KHR_cooperative_matrix, and the GPU reports zero usable configs.
    // The wave32 + DPP path is the correct RDNA4 approach (llama.cpp PR 19625).
    return defs;
}

bool VulkanShaderLibrary::LoadSpv(const std::string& path, std::vector<uint32_t>& out_code) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;

    size_t size = static_cast<size_t>(file.tellg());
    if (size % 4 != 0) return false;

    file.seekg(0, std::ios::beg);
    out_code.resize(size / 4);
    file.read(reinterpret_cast<char*>(out_code.data()), size);
    return true;
}

bool VulkanShaderLibrary::CreateComputePipeline(const PipelineKey& key,
                                                 const std::vector<uint32_t>& spv,
                                                 VkPipeline* out_pipeline,
                                                 VkPipelineLayout* out_layout) {
    VkShaderModule module = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        if (vkCreateShaderModule(device_, &ci, nullptr, &module) != VK_SUCCESS) return false;
        modules_[GetShaderPath(key.type)] = module;
    }

    VkDescriptorSetLayout set_layouts[1] = {};
    uint32_t set_count = 0;
    if (bindless_layout_ != VK_NULL_HANDLE) {
        set_layouts[0] = bindless_layout_;
        set_count = 1;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = set_count;
    layout_ci.pSetLayouts = set_count > 0 ? set_layouts : nullptr;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 128;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;

    if (vkCreatePipelineLayout(device_, &layout_ci, nullptr, out_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, module, nullptr);
        return false;
    }

    VkSpecializationMapEntry entries[5] = {};
    entries[0] = {0, offsetof(SpecializationMap, tile_m), sizeof(uint32_t)};
    entries[1] = {1, offsetof(SpecializationMap, tile_n), sizeof(uint32_t)};
    entries[2] = {2, offsetof(SpecializationMap, head_dim), sizeof(uint32_t)};
    entries[3] = {3, offsetof(SpecializationMap, subgroup_size), sizeof(uint32_t)};
    entries[4] = {4, offsetof(SpecializationMap, accum_type), sizeof(uint32_t)};

    VkSpecializationInfo spec_info{};
    spec_info.mapEntryCount = 5;
    spec_info.pMapEntries = entries;
    spec_info.dataSize = sizeof(SpecializationMap);
    spec_info.pData = &key.spec;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {};
    if (profile_.wave32) {
        subgroup_size_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroup_size_ci.requiredSubgroupSize = 32;
    }

    VkPipelineShaderStageCreateInfo stage_ci{};
    stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = module;
    stage_ci.pName = "main";
    stage_ci.pSpecializationInfo = &spec_info;
    if (profile_.wave32) stage_ci.pNext = &subgroup_size_ci;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage = stage_ci;
    pipeline_ci.layout = *out_layout;

    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci, nullptr, out_pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, *out_layout, nullptr);
        vkDestroyShaderModule(device_, module, nullptr);
        *out_layout = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanShaderLibrary::CreateNamedComputePipeline(const std::string& name,
                                                       const std::vector<uint32_t>& spv,
                                                       VkPipeline* out_pipeline,
                                                       VkPipelineLayout* out_layout) {
    VkShaderModule module = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        if (vkCreateShaderModule(device_, &ci, nullptr, &module) != VK_SUCCESS) return false;
        modules_[name] = module;
    }

    VkDescriptorSetLayout set_layouts[1] = {};
    uint32_t set_count = 0;
    if (bindless_layout_ != VK_NULL_HANDLE) {
        set_layouts[0] = bindless_layout_;
        set_count = 1;
    }

    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = set_count;
    layout_ci.pSetLayouts = set_count > 0 ? set_layouts : nullptr;
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 128;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &pc_range;

    if (vkCreatePipelineLayout(device_, &layout_ci, nullptr, out_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, module, nullptr);
        return false;
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {};
    if (profile_.wave32) {
        subgroup_size_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroup_size_ci.requiredSubgroupSize = 32;
    }

    VkPipelineShaderStageCreateInfo stage_ci{};
    stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_ci.module = module;
    stage_ci.pName = "main";
    if (profile_.wave32) stage_ci.pNext = &subgroup_size_ci;

    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage = stage_ci;
    pipeline_ci.layout = *out_layout;
    pipeline_ci.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_ci.basePipelineIndex = -1;

    VkResult result = vkCreateComputePipelines(device_, pipeline_cache_, 1, &pipeline_ci, nullptr, out_pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, *out_layout, nullptr);
        vkDestroyShaderModule(device_, module, nullptr);
        *out_layout = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanShaderLibrary::CompileKernel(KernelType type, PipelineVariant variant, const SpecializationMap& spec) {
    PipelineKey key{type, variant, spec};
    if (pipelines_.count(key)) return true;

    std::string src_path = shader_dir_ + "/" + KernelTypeToString(type) + ".comp";
    std::string spv_path = GetShaderPath(type);
    std::vector<uint32_t> code;

    ShaderCompileOptions opts;
    opts.src_path = src_path;
    opts.cache_dir = shader_dir_ + "/cache";
    opts.defines = BuildDefines(spec);

    std::string log;
    bool compiled = compiler_.Compile(opts, code, &log);
    if (!compiled) {
        if (!LoadSpv(spv_path, code)) {
            std::cerr << "Failed to compile or load shader for " << KernelTypeToString(type)
                      << ": " << log << "\n";
            return false;
        }
    }

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (!CreateComputePipeline(key, code, &pipeline, &layout)) return false;

    pipelines_[key] = pipeline;
    layouts_[key] = layout;
    return true;
}

bool VulkanShaderLibrary::CompileNamedShader(const std::string& name, const SpecializationMap& spec) {
    if (named_pipelines_.count(name)) return true;

    std::string src_path = shader_dir_ + "/" + name + ".comp";
    std::string spv_path = shader_dir_ + "/" + name + ".spv";
    std::vector<uint32_t> code;

    ShaderCompileOptions opts;
    opts.src_path = src_path;
    opts.cache_dir = shader_dir_ + "/cache";
    opts.defines = BuildDefines(spec);

    std::string log;
    bool compiled = compiler_.Compile(opts, code, &log);
    if (!compiled) {
        if (!LoadSpv(spv_path, code)) {
            std::cerr << "Failed to compile or load named shader '" << name
                      << "': " << log << "\n";
            return false;
        }
    }

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (!CreateNamedComputePipeline(name, code, &pipeline, &layout)) return false;

    named_pipelines_[name] = pipeline;
    named_layouts_[name] = layout;
    return true;
}

VkPipeline VulkanShaderLibrary::GetNamedPipeline(const std::string& name, const SpecializationMap& spec) {
    auto it = named_pipelines_.find(name);
    if (it != named_pipelines_.end()) return it->second;

    if (!CompileNamedShader(name, spec)) return VK_NULL_HANDLE;
    return named_pipelines_[name];
}

bool VulkanShaderLibrary::PrecompileAll(const SpecializationMap& spec) {
    if (!compiler_.IsAvailable()) {
        std::cerr << "PrecompileAll: glslc not available\n";
        return false;
    }

    if (!fs::exists(shader_dir_)) {
        std::cerr << "PrecompileAll: shader directory not found: " << shader_dir_ << "\n";
        return false;
    }

    size_t attempted = 0;
    size_t succeeded = 0;
    for (const auto& entry : fs::directory_iterator(shader_dir_)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".comp") continue;

        std::string name = entry.path().stem().string();
        ShaderCompileOptions opts;
        opts.src_path = entry.path().string();
        opts.cache_dir = shader_dir_ + "/cache";
        opts.defines = BuildDefines(spec);

        std::vector<uint32_t> code;
        std::string log;
        ++attempted;
        if (compiler_.Compile(opts, code, &log)) {
            ++succeeded;
        } else {
            std::cerr << "PrecompileAll: failed to compile " << name << ": " << log << "\n";
        }
    }

    std::cout << "PrecompileAll: " << succeeded << "/" << attempted
              << " shaders compiled/cached\n";
    return succeeded > 0 || attempted == 0;
}

bool VulkanShaderLibrary::CompileFastVariants(const IQuantization* quant, const SpecializationMap& spec) {
    (void)quant;
    bool ok = true;
    ok &= CompileKernel(KernelType::RMS_NORM, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::QKV_PROJ, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::ROPE, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::MLP_UP, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::MLP_GATE, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::MLP_DOWN, PipelineVariant::FAST, spec);
    ok &= CompileKernel(KernelType::LM_HEAD, PipelineVariant::FAST, spec);
    return ok;
}

void VulkanShaderLibrary::CompileSafeVariantsBackground(const IQuantization* quant) {
    (void)quant;
    // TODO: background compilation of SAFE variants
}

VkPipeline VulkanShaderLibrary::GetPipeline(KernelType type, PipelineVariant variant, const SpecializationMap& spec) {
    PipelineKey key{type, variant, spec};
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;

    if (!CompileKernel(type, variant, spec)) return VK_NULL_HANDLE;
    return pipelines_[key];
}

VkPipelineLayout VulkanShaderLibrary::GetPipelineLayout(KernelType type, PipelineVariant variant, const SpecializationMap& spec) {
    PipelineKey key{type, variant, spec};
    auto it = layouts_.find(key);
    if (it != layouts_.end()) return it->second;

    // Auto-compile if not yet created
    if (!CompileKernel(type, variant, spec)) return VK_NULL_HANDLE;
    return layouts_[key];
}

VkPipeline VulkanShaderLibrary::GetFusedPipeline(FusedKernelType type, PipelineVariant variant, const SpecializationMap& spec) {
    (void)type;
    (void)variant;
    (void)spec;
    // TODO: fused kernel support
    return VK_NULL_HANDLE;
}

bool VulkanShaderLibrary::ValidateKernel(VkPipeline pipeline, IDevice* device) {
    (void)device;
    return pipeline != VK_NULL_HANDLE;
}

void VulkanShaderLibrary::SetPreferredSubgroupSize(uint32_t size) {
    preferred_subgroup_size_ = size;
}

bool VulkanShaderLibrary::EmergencyDegrade(KernelType type, SpecializationMap& out_reduced_spec) {
    (void)type;
    out_reduced_spec.tile_m = 32;
    out_reduced_spec.tile_n = 32;
    out_reduced_spec.tile_k = 16;
    out_reduced_spec.workgroup_x = 32;
    out_reduced_spec.workgroup_y = 1;
    return true;
}

} // namespace notllama
