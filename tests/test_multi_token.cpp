#include "vulkan/inference_engine.h"
#include <cstdio>
#include <cstdint>
#include <vector>

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkApplicationInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion = VK_API_VERSION_1_4;
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;

    InferenceEngine eng;
    if (!eng.init(inst)) return 1;
    if (!eng.load_model("e:/OLLAMA-Models/GGUF/stories260k.bin",
                        "../../stories260K.weights.json")) return 1;
    if (!eng.create_pipelines("shaders_spv")) return 1;

    // Autoregressive generation
    uint32_t num_tokens = 20;
    uint32_t token = 1;  // BOS
    std::vector<uint32_t> generated;

    printf("=== Generating %u tokens ===\n", num_tokens);
    for (uint32_t pos = 0; pos < num_tokens; pos++) {
        token = eng.forward(token, pos);
        generated.push_back(token);
        printf("  [%2u] token=%u (score=%.2f)\n", pos, token, 0.0f);
        if (token == 2) { printf("  (EOS)\n"); break; }  // EOS
    }

    printf("\nGenerated sequence: ");
    for (auto t : generated) printf("%u ", t);
    printf("\n");

    // Determinism check: re-run and compare
    printf("\n=== Determinism check ===\n");
    std::vector<uint32_t> gen2;
    token = 1;
    for (uint32_t pos = 0; pos < generated.size() && pos < num_tokens; pos++) {
        token = eng.forward(token, pos);
        gen2.push_back(token);
        if (token == 2) break;
    }

    bool match = (generated.size() == gen2.size());
    if (match) {
        for (size_t i = 0; i < generated.size(); i++)
            if (generated[i] != gen2[i]) { match = false; break; }
    }
    printf("Deterministic: %s\n", match ? "YES" : "NO");

    eng.cleanup();
    vkDestroyInstance(inst, nullptr);
    printf("\n%s\n", match ? "PASS" : "FAIL");
    return match ? 0 : 1;
}
