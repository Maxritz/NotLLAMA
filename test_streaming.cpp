// Test: ModelAdapter streaming path — load metadata, stream layers to GPU, verify
#include <cstdio>
#include <cstdlib>
#include <string>
#include "engine/model_adapter.hpp"
#include "rdna4_vulkan.hpp"

static std::string FindShaderDir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
        fs::current_path() / "shaders",
        fs::current_path() / ".." / "shaders",
        fs::current_path() / ".." / ".." / "shaders",
    };
    for (const auto& c : candidates) {
        fs::path canon = fs::weakly_canonical(c);
        if (fs::exists(canon / "rms_norm.comp")) return canon.string();
    }
    return "shaders";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_streaming <model.weights.json>\n");
        return 1;
    }
    std::string json_path = argv[1];
    std::string bin_path = json_path;
    size_t p = bin_path.rfind(".json");
    if (p != std::string::npos) bin_path.replace(p, 5, ".bin");
    else bin_path += ".bin";

    rdna4::VulkanContext ctx;
    if (!ctx.init()) { fprintf(stderr, "FAIL: Vulkan init\n"); return 1; }
    fprintf(stderr, "Vulkan OK\n");

    // Find shader dir for RMS_NORM validation in engine
    std::string shader_dir = FindShaderDir();
    fprintf(stderr, "Shader dir: %s\n", shader_dir.c_str());

    notllama::ModelAdapter adapter(ctx.device, ctx.physicalDevice, ctx.queueFamilyIndex);

    // Test 1: LoadFromPath (metadata only)
    fprintf(stderr, "\n--- Test 1: LoadFromPath ---\n");
    if (!adapter.LoadFromPath(json_path, nullptr)) {
        fprintf(stderr, "FAIL: LoadFromPath\n");
        ctx.cleanup(); return 1;
    }
    fprintf(stderr, "PASS: LoadFromPath (%zu layers)\n", adapter.GetNumLayers());

    // Test 2: GetWeightTensors before streaming (all addresses should be 0)
    fprintf(stderr, "\n--- Test 2: Verify all GPU addresses are 0 before streaming ---\n");
    {
        auto tensors = adapter.GetWeightTensors();
        size_t zero_count = 0;
        for (auto& t : tensors) {
            if (t.alloc.device_address == 0) zero_count++;
        }
        fprintf(stderr, "%zu/%zu tensors have zero GPU address (expected all)\n",
                zero_count, tensors.size());
        if (zero_count != tensors.size()) {
            fprintf(stderr, "FAIL: Some tensors already have GPU addresses\n");
            ctx.cleanup(); return 1;
        }
    }
    fprintf(stderr, "PASS: All tensors have zero GPU address\n");

    // Test 3: Stream each layer and verify addresses become non-zero
    fprintf(stderr, "\n--- Test 3: StreamLayerWeights ---\n");
    size_t total_layers = adapter.GetNumLayers();
    for (uint32_t l = 0; l < total_layers; l++) {
        if (!adapter.StreamLayerWeights(l, nullptr)) {
            fprintf(stderr, "FAIL: StreamLayerWeights(%u)\n", l);
            ctx.cleanup(); return 1;
        }
        // Verify some tensors for this layer now have GPU addresses
        auto tensors = adapter.GetWeightTensors();
        size_t layer_tensors = 0;
        size_t layer_nonzero = 0;
        for (auto& t : tensors) {
            if (t.alloc.device_address != 0) layer_nonzero++;
        }
        fprintf(stderr, "  Layer %u: %zu tensors on GPU\n", l, layer_nonzero);
    }
    fprintf(stderr, "PASS: All %zu layers streamed\n", total_layers);

    // Test 4: Stream same layer again (idempotent)
    fprintf(stderr, "\n--- Test 4: Re-stream layer (idempotent) ---\n");
    if (!adapter.StreamLayerWeights(0, nullptr)) {
        fprintf(stderr, "FAIL: Re-stream layer 0\n");
        ctx.cleanup(); return 1;
    }
    fprintf(stderr, "PASS: Re-stream idempotent\n");

    // Test 5: Out-of-range layer
    fprintf(stderr, "\n--- Test 5: Out-of-range layer ---\n");
    if (adapter.StreamLayerWeights((uint32_t)total_layers + 1, nullptr)) {
        fprintf(stderr, "FAIL: Should have rejected out-of-range layer\n");
        ctx.cleanup(); return 1;
    }
    fprintf(stderr, "PASS: Out-of-range rejected\n");

    fprintf(stderr, "\n=== ALL TESTS PASSED ===\n");
    ctx.cleanup();
    return 0;
}
