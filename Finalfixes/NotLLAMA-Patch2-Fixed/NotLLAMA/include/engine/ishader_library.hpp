#pragma once
#include "types.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <cstdint>

namespace notllama {

enum class KernelType {
    RMS_NORM,
    RMS_NORM_MUL,
    ADD_RMS_NORM,
    QKV_PROJ,
    ROPE,
    FLASH_ATTN,
    ATTENTION,
    MLP_UP,
    MLP_GATE,
    MLP_DOWN,
    MLP_FUSED_GATEUP,
    LM_HEAD,
    EMBED,
    EMBED_Q8_0,
    DEQUANT,
    DEQUANT_TURBO,
    DEQUANTIZE_TEST,
    KV_CACHE_WRITE,
    KV_CACHE_QUANT,
    KV_CACHE_QUANT_Q8_0,
    KV_CACHE_DEQUANT,
    COPY_F32_F16,
    COPY_F32_BF16,
    COPY_F16_F32,
    COPY_BF16_F32,
    SILU_MUL,
    TOPK,
    ADD,
    COMPRESS_CONTEXT,
    GEMM,
    GEMM_Q4_0,
    GEMM_Q8_0,
    GEMM_Q6K,
    GEMM_Q4K,
    GEMM_TURBO,
    MATMUL_Q4_0,
    MATMUL_Q8_0,
    COOPMAT_GEMM,
    BDA_TEST,
    KERNEL_ENTRY
};

enum class FusedKernelType {
    RMS_QKV,
    ROPE_FLASHATTN,
    MLP_UPGATE,
    MLP_DOWN_RESIDUAL
};

class IDevice;
class IQuantization;

class IShaderLibrary {
public:
    virtual ~IShaderLibrary() = default;

    virtual bool LoadPipelineCache(const std::string& path) = 0;

    virtual bool CompileFastVariants(const IQuantization* quant, const SpecializationMap& spec) = 0;
    virtual void CompileSafeVariantsBackground(const IQuantization* quant) = 0;

    virtual bool CompileNamedShader(const std::string& name, const SpecializationMap& spec) = 0;
    virtual VkPipeline GetNamedPipeline(const std::string& name, const SpecializationMap& spec) = 0;
    virtual bool PrecompileAll(const SpecializationMap& spec) = 0;

    virtual VkPipeline GetPipeline(KernelType type, PipelineVariant variant, const SpecializationMap& spec) = 0;
    virtual VkPipelineLayout GetPipelineLayout(KernelType type, PipelineVariant variant, const SpecializationMap& spec) = 0;
    virtual VkPipeline GetFusedPipeline(FusedKernelType type, PipelineVariant variant, const SpecializationMap& spec) = 0;

    virtual bool ValidateKernel(VkPipeline pipeline, IDevice* device) = 0;
    virtual void SetPreferredSubgroupSize(uint32_t size) = 0;
    virtual bool EmergencyDegrade(KernelType type, SpecializationMap& out_reduced_spec) = 0;
};

} // namespace notllama
