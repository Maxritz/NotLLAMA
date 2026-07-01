#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rdna4 {

struct InferenceEngine;
struct ModelHParams;

// ============================================================================
// Embedding extraction modes
// ============================================================================
enum class EmbeddingMode : uint32_t {
    LAST_HIDDEN = 0,   // Last layer hidden state
    MEAN_POOL   = 1,   // Mean of all token hidden states
    CLS_TOKEN   = 2,   // CLS token embedding (BERT-style)
    POOLING     = 3,   // Custom pooling layer
    LAYER_N     = 4,   // Specific layer output
};

// ============================================================================
// Embedding request
// ============================================================================
struct EmbeddingRequest {
    std::vector<uint32_t> tokens;
    EmbeddingMode mode = EmbeddingMode::LAST_HIDDEN;
    uint32_t targetLayer = 0;      // for LAYER_N mode
    bool normalize = true;
    std::string poolingType = "mean"; // mean, max, cls
};

// ============================================================================
// Embedding result
// ============================================================================
struct EmbeddingResult {
    std::vector<float> embedding;
    uint32_t dim = 0;
    bool normalized = false;
    float norm = 0.0f;
};

// ============================================================================
// Embedding engine — extract hidden states for RAG / similarity
// ============================================================================
class EmbeddingEngine {
public:
    explicit EmbeddingEngine(InferenceEngine* engine);

    // Compute embedding for a token sequence
    EmbeddingResult compute(const EmbeddingRequest& req);

    // Batch compute (more efficient)
    std::vector<EmbeddingResult> computeBatch(const std::vector<EmbeddingRequest>& reqs);

    // Similarity metrics
    static float cosineSimilarity(const std::vector<float>& a,
                                  const std::vector<float>& b);
    static float dotProduct(const std::vector<float>& a,
                            const std::vector<float>& b);
    static float euclideanDistance(const std::vector<float>& a,
                                   const std::vector<float>& b);

    // Dimensionality
    uint32_t getEmbeddingDim() const;

private:
    InferenceEngine* engine_;
};

} // namespace rdna4
