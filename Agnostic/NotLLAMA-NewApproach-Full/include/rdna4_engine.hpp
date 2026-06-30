#pragma once
#include "rdna4.hpp"
#include "rdna4_vulkan.hpp"
#include "rdna4_weights.hpp"
#include "rdna4_kv_cache.hpp"
#include "rdna4_pipeline.hpp"
#include "rdna4_tokenizer.hpp"
#include "rdna4_scheduler.hpp"
#include "rdna4_allocator.hpp"
#include <string>
#include <vector>

namespace rdna4 {

class InferenceEngine {
public:
    VulkanContext* ctx;
    ModelDesc* model;
    KVCacheManager* kvCache;
    PipelineBuilder* pipelines;
    Tokenizer* tokenizer;
    Scheduler* scheduler;
    RingAllocator* allocator;

    // Dequantization staging buffer (reused for every dequantize dispatch)
    VkBuffer dequantBuffer = VK_NULL_HANDLE;
    VkDeviceMemory dequantMemory = VK_NULL_HANDLE;
    VkDeviceAddress dequantAddr = 0;
    VkDeviceAddress dequantBufEnd = 0;
    size_t dequantCapacity = 0;

    // Persistent embedding cache (dequantized once, reused across all tokens)
    VkBuffer embedCacheBuffer = VK_NULL_HANDLE;
    VkDeviceMemory embedCacheMemory = VK_NULL_HANDLE;
    VkDeviceAddress embedCacheAddr = 0;
    size_t embedCacheSize = 0;
    bool embedCacheReady = false;

    // Persistent dequantized weight buffer for kernel_entry.comp (float32).
    // Holds ALL layer weights dequantized at init. Size ≈ nLayers × 68 MB.
    VkBuffer weightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory weightMemory = VK_NULL_HANDLE;
    VkDeviceAddress weightBufferAddr = 0;
    size_t weightBufferSize = 0;

    // LayerParams array — one per layer, written once, read by GPU kernel.
    VkBuffer layerParamsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory layerParamsMemory = VK_NULL_HANDLE;
    VkDeviceAddress layerParamsAddr = 0;

    // Scratch buffer for kernel_entry.comp (residual + Q + K + V).
    VkBuffer scratchBuffer = VK_NULL_HANDLE;
    VkDeviceMemory scratchMemory = VK_NULL_HANDLE;
    VkDeviceAddress scratchAddr = 0;

    // Mailbox + output token buffer for kernel_entry.comp.
    VkBuffer mailboxBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mailboxMemory = VK_NULL_HANDLE;
    VkDeviceAddress mailboxAddr = 0;

    VkBuffer outputTokenBuffer = VK_NULL_HANDLE;
    VkDeviceMemory outputTokenMemory = VK_NULL_HANDLE;
    VkDeviceAddress outputTokenAddr = 0;

    // Hidden state buffer for kernel_entry.comp (persistent across tokens).
    // Sized for maxContext × dim floats. Kernel writes activations here.
    VkBuffer hiddenStateBuffer = VK_NULL_HANDLE;
    VkDeviceMemory hiddenStateMemory = VK_NULL_HANDLE;
    VkDeviceAddress hiddenStateAddr = 0;

    // Host-readable logits buffer for kernel_entry.comp.
    // Kernel writes logits here; test harness reads via mapped memory.
    VkBuffer logitsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory logitsMemory = VK_NULL_HANDLE;
    VkDeviceAddress logitsAddr = 0;
    float* logitsMapped = nullptr;

    InferenceEngine(VulkanContext* c, ModelDesc* m, KVCacheManager* k,
                      PipelineBuilder* p, Tokenizer* t, Scheduler* s, RingAllocator* a);

    bool initDequantBuffer();  // Allocate reusable dequant staging buffer
    void cleanupDequantBuffer();
    bool initEmbedCache();  // Allocate persistent embedding cache
    void cleanupEmbedCache();
    bool initWeightBuffer();   // Dequant all weights to float32 persistent buffer
    void cleanupWeightBuffer();
    bool initLayerParams();    // Build LayerParams array in GPU-visible buffer
    bool initKernelEntryBuffers();  // Mailbox, output, scratch, layerParams
    void cleanupKernelEntryBuffers();

    bool kernelEntryReady = false;  // True if all kernel_entry buffers are initialized

    // Release all engine-owned GPU resources. Safe to call multiple times.
    void cleanup();

    uint32_t forward(uint32_t tokenId, uint32_t seqPos);

    // Address of the logits buffer from the most recent forward() / forwardPartial().
    // Valid only after a successful forward call; 0 if no forward has run.
    uint64_t lastLogitsAddr = 0;
    size_t lastLogitsOffset = 0;

    // Speculative decode: draft N tokens, verify in parallel
    std::vector<uint32_t> forwardSpeculative(uint32_t tokenId, uint32_t seqPos, uint32_t nDraft);

    std::vector<uint32_t> generate(const std::string& prompt, uint32_t maxTokens);

    // Generate with speculative decoding
    std::vector<uint32_t> generateSpeculative(const std::string& prompt, uint32_t maxTokens, uint32_t nDraft);

    uint32_t sampleGpu(uint64_t logitsAddr, uint32_t vocabSize, uint64_t sampleOutAddr, uint64_t scratchAddr);
    uint32_t sampleArgmax(const float* logits, uint32_t vocabSize);

private:
    // Core forward pass with optional layer count limit
    uint32_t forwardPartial(uint32_t tokenId, uint32_t seqPos, uint32_t maxLayers);

    // One-dispatch forward via kernel_entry.comp (persistent kernel with mailbox polling)
    uint32_t forwardKernelEntry(uint32_t tokenId, uint32_t seqPos);

    // Fast draft forward using fewer layers (e.g., first 2 layers only)
    uint32_t draftForward(uint32_t tokenId, uint32_t seqPos, uint32_t nLayers);

    // Verify a draft token against the full model — returns predicted next token
    uint32_t verifyDraftToken(uint32_t draftToken, uint32_t seqPos);
};

} // namespace rdna4
