#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace rdna4 {

struct ModelHParams;
struct Scheduler;
struct PipelineBuilder;

// ============================================================================
// MoE routing decision per token
// ============================================================================
struct MoeRouting {
    uint32_t tokenIndex = 0;
    uint32_t topK = 0;
    uint32_t experts[8] = {};       // top-k expert indices
    float    weights[8] = {};       // gating weights
    float    normFactor = 1.0f;     // load-balancing normalization
};

// ============================================================================
// MoE execution config
// ============================================================================
struct MoeConfig {
    uint32_t nExperts = 0;
    uint32_t nExpertsPerToken = 0;  // top-k
    uint32_t expertDim = 0;
    uint32_t sharedExpertDim = 0;
    uint32_t nSharedExperts = 0;
    bool     useGatingSoftmax = true;
    bool     useLoadBalancing = false;
    float    loadBalancingLoss = 0.01f;
};

// ============================================================================
// MoE manager — sparse expert dispatch + weight routing
// ============================================================================
class MoeManager {
public:
    explicit MoeManager(Scheduler* sched, PipelineBuilder* pipes);

    // Initialize from model hparams
    bool init(const ModelHParams& hparams);

    // Gate + route: input [batch, dim] -> expert assignments
    // Returns false if no MoE layers in model
    bool routeBatch(const float* hiddenIn, uint32_t batchSize,
                    std::vector<MoeRouting>& outRoutes);

    // Dispatch experts for a single token (GPU path)
    // This dispatches the moe_dispatch.comp shader
    bool dispatchExperts(uint32_t layerIndex,
                         uint64_t hiddenAddr,
                         uint64_t gateAddr,
                         uint64_t expertWeightsAddr,
                         uint64_t outAddr,
                         const MoeRouting& routing);

    // Dispatch shared expert MLP (always runs)
    bool dispatchSharedExpert(uint32_t layerIndex,
                              uint64_t hiddenAddr,
                              uint64_t sharedGateAddr,
                              uint64_t sharedUpAddr,
                              uint64_t sharedDownAddr,
                              uint64_t outAddr);

    // Batch MoE dispatch (processes all tokens in one submit)
    bool dispatchMoeLayerBatch(uint32_t layerIndex,
                               uint64_t hiddenAddr,
                               uint64_t gateAddr,
                               uint64_t expertWeightsAddr,
                               uint64_t sharedExpertAddr,
                               uint64_t outAddr,
                               const std::vector<MoeRouting>& routes);

    // CPU fallback for MoE (when layer is offloaded)
    void computeMoeCpu(const float* hiddenIn, float* hiddenOut,
                       const MoeConfig& cfg,
                       const std::vector<float*>& expertWeights);

    // Check if a layer is an MoE layer
    bool isMoeLayer(uint32_t layerIndex) const;

    const MoeConfig& config() const { return cfg_; }

private:
    Scheduler* sched_;
    PipelineBuilder* pipes_;
    MoeConfig cfg_;
    std::vector<bool> moeLayerMask_;
};

} // namespace rdna4
