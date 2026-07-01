#pragma once
#include "engine/ishader_library.hpp"
#include "engine/shader_compiler.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace notllama {

struct PipelineKey {
    KernelType type;
    PipelineVariant variant;
    SpecializationMap spec;

    bool operator==(const PipelineKey& other) const noexcept;
};

struct PipelineKeyHash {
    size_t operator()(const PipelineKey& k) const noexcept;
};

class IQuantization;

class VulkanShaderLibrary : public IShaderLibrary {
public:
    explicit VulkanShaderLibrary(VkDevice device, const std::string& shader_dir,
                                 const VendorProfile& profile = {});
    ~VulkanShaderLibrary() override;

    // Set the bindless descriptor set layout (AMD-safe, no update-after-bind).
    // When set, all pipeline layouts will include this as set=0.
    void SetBindlessSetLayout(VkDescriptorSetLayout layout) { bindless_layout_ = layout; }

    bool LoadPipelineCache(const std::string& path) override;

    bool CompileFastVariants(const IQuantization* quant, const SpecializationMap& spec) override;
    void CompileSafeVariantsBackground(const IQuantization* quant) override;

    bool CompileNamedShader(const std::string& name, const SpecializationMap& spec) override;
    VkPipeline GetNamedPipeline(const std::string& name, const SpecializationMap& spec) override;
    bool PrecompileAll(const SpecializationMap& spec) override;

    VkPipeline GetPipeline(KernelType type, PipelineVariant variant, const SpecializationMap& spec) override;
    VkPipelineLayout GetPipelineLayout(KernelType type, PipelineVariant variant, const SpecializationMap& spec) override;
    VkPipeline GetFusedPipeline(FusedKernelType type, PipelineVariant variant, const SpecializationMap& spec) override;

    bool ValidateKernel(VkPipeline pipeline, IDevice* device) override;
    void SetPreferredSubgroupSize(uint32_t size) override;
    bool EmergencyDegrade(KernelType type, SpecializationMap& out_reduced_spec) override;

private:
    VkDevice device_;
    std::string shader_dir_;
    VendorProfile profile_;
    ShaderCompiler compiler_;
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    uint32_t preferred_subgroup_size_ = 32;
    VkDescriptorSetLayout bindless_layout_ = VK_NULL_HANDLE;

    std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelines_;
    std::unordered_map<PipelineKey, VkPipelineLayout, PipelineKeyHash> layouts_;
    std::unordered_map<std::string, VkShaderModule> modules_;

    // Named pipelines are keyed by shader stem (e.g. "gemm_q4_0", "dequantize").
    std::unordered_map<std::string, VkPipeline> named_pipelines_;
    std::unordered_map<std::string, VkPipelineLayout> named_layouts_;

    std::string KernelTypeToString(KernelType type) const;
    std::string FusedKernelTypeToString(FusedKernelType type) const;
    std::string GetShaderPath(KernelType type) const;
    std::string GetShaderPath(FusedKernelType type) const;

    std::vector<std::string> BuildDefines(const SpecializationMap& spec) const;

    bool LoadSpv(const std::string& path, std::vector<uint32_t>& out_code) const;
    bool CreateComputePipeline(const PipelineKey& key, const std::vector<uint32_t>& spv,
                                VkPipeline* out_pipeline, VkPipelineLayout* out_layout);
    bool CreateNamedComputePipeline(const std::string& name, const std::vector<uint32_t>& spv,
                                     VkPipeline* out_pipeline, VkPipelineLayout* out_layout);
    bool CompileKernel(KernelType type, PipelineVariant variant, const SpecializationMap& spec);
};

} // namespace notllama
