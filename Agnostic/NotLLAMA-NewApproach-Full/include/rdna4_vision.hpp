#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rdna4 {

// ============================================================================
// Vision encoder types (CLIP, SigLIP, etc.)
// ============================================================================
enum class VisionEncoderType : uint32_t {
    CLIP        = 0,
    SIGLIP      = 1,
    DINOv2      = 2,
    CUSTOM      = 3,
};

// ============================================================================
// Image preprocessing config
// ============================================================================
struct VisionConfig {
    VisionEncoderType encoderType = VisionEncoderType::CLIP;
    uint32_t imageSize = 336;      // input resolution
    uint32_t patchSize = 14;
    uint32_t numChannels = 3;
    uint32_t hiddenSize = 768;
    uint32_t numLayers = 12;
    uint32_t numHeads = 12;
    uint32_t intermediateSize = 3072;
    std::string imageMean = "0.48145466,0.4578275,0.40821073";
    std::string imageStd  = "0.26862954,0.26130258,0.27577711";
};

// ============================================================================
// Processed image patch embeddings
// ============================================================================
struct ImagePatches {
    std::vector<float> patchEmbeddings; // [nPatches, hiddenSize]
    uint32_t nPatches = 0;
    uint32_t hiddenSize = 0;
};

// ============================================================================
// Vision-language model interface
// ============================================================================
class VisionEngine {
public:
    explicit VisionEngine(const VisionConfig& cfg);

    // Load vision encoder weights
    bool loadWeights(const std::string& jsonPath, const std::string& binPath);

    // Encode image to patch embeddings
    ImagePatches encodeImage(const std::vector<uint8_t>& rgbaPixels,
                             uint32_t width, uint32_t height);

    // Encode from file path
    ImagePatches encodeImageFile(const std::string& path);

    // Project vision features to language model space
    std::vector<float> projectToLlmSpace(const ImagePatches& patches);

    // Build multimodal prompt: <image_tokens> + text_tokens
    std::vector<uint32_t> buildMultimodalPrompt(
        const ImagePatches& patches,
        const std::vector<uint32_t>& textTokens,
        uint32_t imageTokenId);

    uint32_t getImageTokenCount() const;

private:
    VisionConfig cfg_;
    bool loaded_ = false;
};

} // namespace rdna4
