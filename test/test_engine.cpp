#include "engine/engine.hpp"
#include "rdna4_vulkan.hpp"
#include <cstdio>
#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

using namespace notllama;

static std::string FindShaderDir(const char* argv0) {
    namespace fs = std::filesystem;
    fs::path exe_dir = fs::path(argv0).parent_path();
    std::vector<fs::path> candidates = {
        exe_dir / "shaders",
        exe_dir / ".." / ".." / "shaders",
        fs::current_path() / "shaders",
    };
    for (const auto& c : candidates) {
        fs::path canon = fs::weakly_canonical(c);
        if (fs::exists(canon / "rms_norm.comp")) {
            return canon.string();
        }
    }
    return "shaders";
}

int main(int argc, char** argv) {
    (void)argc;

    fprintf(stderr, "Engine modular interface test\n");

    rdna4::VulkanContext ctx;
    bool ok = ctx.init();
    assert(ok);
    printf("Vulkan context initialized\n");

    {
        VulkanDevice device(&ctx);
        assert(device.GetLogicalDevice() != VK_NULL_HANDLE);
        assert(device.GetPhysicalDevice() != VK_NULL_HANDLE);
        assert(device.GetComputeQueue(0) != VK_NULL_HANDLE);
        printf("VulkanDevice adapter ok\n");

        VendorProfile profile = device.GetVendorProfile();
        printf("Vendor: %s  Subgroup: %u  Wave32: %d\n",
               profile.vendor == VendorID::AMD ? "AMD" :
               profile.vendor == VendorID::NVIDIA ? "NVIDIA" : "UNKNOWN",
               profile.subgroup_size,
               profile.wave32 ? 1 : 0);

        RingAllocatorAdapter allocator(ctx.device, ctx.physicalDevice,
                                       16 * 1024 * 1024,
                                       16 * 1024 * 1024,
                                       16 * 1024 * 1024);
        printf("RingAllocatorAdapter created\n");

        GpuAllocation test_alloc = allocator.Allocate(MemoryType::TRANSIENT, 1024, 256);
        assert(test_alloc.buffer != VK_NULL_HANDLE);
        assert(test_alloc.size == 1024);
        assert(test_alloc.device_address != 0);
        printf("Allocation ok\n");

        VulkanDescriptorManager desc_mgr(ctx.device, ctx.physicalDevice, &allocator);
        assert(desc_mgr.IsValid());
        uint32_t idx = desc_mgr.RegisterBuffer(test_alloc.buffer);
        assert(idx == 0);
        assert(desc_mgr.GetBindlessSet() != VK_NULL_HANDLE);
        VkDescriptorSetLayout bindless_layout = desc_mgr.GetBindlessSetLayout();
        assert(bindless_layout != VK_NULL_HANDLE);
        printf("DescriptorManager ok (bindless layout enabled)\n");

        std::string shader_dir = FindShaderDir(argv[0]);
        printf("Using shader dir: %s\n", shader_dir.c_str());

        VulkanShaderLibrary shader_lib(ctx.device, shader_dir, profile);
        // Wire the bindless descriptor set layout into all pipeline layouts
        shader_lib.SetBindlessSetLayout(bindless_layout);

        // Create the compute engine with real dependencies
        VulkanComputeEngine compute_engine(ctx.device, ctx.physicalDevice, ctx.queues[0], ctx.queueFamilyIndex,
                                            &shader_lib, &desc_mgr, &allocator);
        compute_engine.StartWatchdog();
        assert(compute_engine.GetLastFrameStatus() == WatchdogStatus::OK);

        // Test RMS_NORM dispatch validation (compiles kernel, dispatches, verifies)
        printf("Testing RMS_NORM dispatch validation...\n");
        bool step_ok = compute_engine.StepBatch();
        if (step_ok) {
            printf("RMS_NORM dispatch validation PASSED\n");
        } else {
            // If glslc is not available, PrecompileAll won't work — skip gracefully
            ShaderCompiler check_compiler;
            if (check_compiler.IsAvailable()) {
                printf("RMS_NORM dispatch validation FAILED (compiler available but dispatch failed)\n");
                return 1;
            }
            printf("RMS_NORM dispatch validation SKIPPED (glslc not available)\n");
        }

        compute_engine.StopWatchdog();
        printf("VulkanComputeEngine dispatch ok\n");

        // Test ShaderLibrary with bindless layout
        assert(shader_lib.ValidateKernel(VK_NULL_HANDLE, &device) == false);

        SpecializationMap spec{};
        spec.subgroup_size = profile.subgroup_size;
        spec.head_dim = 128;

        VkPipeline test_pipeline = shader_lib.GetPipeline(KernelType::FLASH_ATTN, PipelineVariant::FAST, spec);
        (void)test_pipeline;

        printf("Testing PrecompileAll...\n");
        bool precompiled = shader_lib.PrecompileAll(spec);
        printf("PrecompileAll: %s\n", precompiled ? "ok" : "failed/no shaders");

        printf("ShaderLibrary created with bindless layout\n");

        VulkanDebugContext debug_ctx(ctx.instance, ctx.device);
        debug_ctx.NameObject(reinterpret_cast<uint64_t>(ctx.device), VK_OBJECT_TYPE_DEVICE, "NotLLAMA device");
        printf("DebugContext ok\n");

        printf("All engine interface tests PASSED\n");
    }

    ctx.cleanup();
    return 0;
}
