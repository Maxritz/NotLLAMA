#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace rdna4 {

// ============================================================================
// Supported model architectures (parity with llama.cpp)
// ============================================================================
enum class ModelArch : uint32_t {
    UNKNOWN     = 0,
    LLAMA       = 1,   // LLaMA, LLaMA-2, LLaMA-3, CodeLLaMA
    MISTRAL     = 2,   // Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
    MIXTRAL     = 3,   // Explicit MoE variant
    QWEN        = 4,   // Qwen, Qwen-2, Qwen-2.5, Qwen-Coder
    QWEN2MOE    = 5,   // Qwen2-MoE
    PHI         = 6,   // Phi-1, Phi-2, Phi-3, Phi-4
    PHI3        = 7,   // Phi-3/4 explicit
    GEMMA       = 8,   // Gemma, Gemma-2
    GEMMA2      = 9,   // Gemma-2 explicit
    FALCON      = 10,  // Falcon 7B/40B/180B
    GPT2        = 11,  // GPT-2
    GPTJ        = 12,  // GPT-J
    GPTNEOX     = 13,  // GPT-NeoX, Pythia, StableLM
    MPT         = 14,  // MPT
    RWKV        = 15,  // RWKV
    BERT        = 16,  // BERT, RoBERTa (embeddings/reranking)
    T5          = 17,  // T5, Flan-T5, UL2
    DEEPSEEK    = 18,  // DeepSeek-V2/V3 (MLA attention)
    DEEPSEEK2   = 19,  // DeepSeek-V2 with MLA
    DEEPSEEK3   = 20,  // DeepSeek-V3 with MoE + MLA
    COMMAND_R   = 21,  // Cohere Command-R
    DBRX        = 22,  // DBRX (Databricks)
    XVERSE      = 23,  // XVERSE
    ORION       = 24,  // Orion
    MINICPM     = 25,  // MiniCPM
    STABLELM    = 26,  // StableLM
    STARCODER   = 27,  // StarCoder, StarCoder2
    REFACT      = 28,  // Refact
    BLOOM       = 29,  // BLOOM
    MAMBA       = 30,  // Mamba, Jamba
    GRANITE     = 31,  // IBM Granite
    NEMOTRON    = 32,  // Nemotron
    CHATGLM     = 33,  // ChatGLM
    LLAVA       = 34,  // LLaVA (vision-language)
    BAICHUAN    = 35,  // Baichuan
    YI          = 36,  // Yi
    COHERE      = 37,  // Cohere (non-Command-R)
    OLMO        = 38,  // OLMo
    ARCTIC      = 39,  // Snowflake Arctic
    COUNT       = 40,
};

// ============================================================================
// Model hyperparameters (architecture-agnostic, populated from GGUF metadata)
// ============================================================================
struct ModelHParams {
    ModelArch arch = ModelArch::UNKNOWN;
    std::string archName;

    // Core dims
    uint32_t vocabSize = 0;
    uint32_t vocabSizeUsed = 0;
    uint32_t dim = 0;              // embedding dimension (hidden_size)
    uint32_t nLayers = 0;          // num_hidden_layers
    uint32_t nHeads = 0;           // num_attention_heads
    uint32_t nHeadsKv = 0;         // num_key_value_heads
    uint32_t headDim = 0;          // attention_head_dim (usually dim / nHeads)
    uint32_t ffnDim = 0;           // intermediate_size / feed_forward_length
    uint32_t contextLength = 0;    // max_position_embeddings
    uint32_t ropeDim = 0;          // rope dimensionality (partial rotary)

    // Normalization
    float    rmsNormEps = 1e-5f;
    bool     useRmsNorm = true;
    bool     useLayerNorm = false;

    // RoPE
    float    ropeBase = 10000.0f;
    float    ropeScale = 1.0f;
    bool     ropeScaled = false;
    float    ropeFreqScale = 1.0f;
    float    ropeFreqBase = 10000.0f;
    uint32_t ropeNTimes = 0;

    // Activation
    enum class ActFn { SILU, GELU, GELU_TANH, RELU, RELU2, SWIGLU };
    ActFn    actFn = ActFn::SILU;

    // Attention variants
    bool     useGqa = true;        // grouped query attention
    bool     useMla = false;       // multi-head latent attention (DeepSeek-V2/V3)
    bool     useSlidingWindow = false;
    uint32_t slidingWindow = 0;
    bool     useAlibi = false;
    bool     useClampedKqv = false;
    float    attnSoftcap = 0.0f;   // Gemma2 softcapping
    bool     useQKNorm = false;    // Qwen2.5 / modern archs

    // MoE
    bool     isMoe = false;
    uint32_t nExperts = 0;
    uint32_t nExpertsUsed = 0;     // top-k experts per token
    uint32_t moeFreq = 0;          // every N layers is MoE
    bool     moeSharedExpert = false;
    uint32_t nSharedExperts = 0;

    // Tie weights
    bool     tieWordEmbeddings = false;
    bool     tieLMHead = false;

    // Quantization overrides
    QuantFormat defaultWeightFormat = QuantFormat::F16;
    bool        useImatrix = false;
    std::string imatrixPath;

    // Special tokens
    uint32_t bosTokenId = 0;
    uint32_t eosTokenId = 0;
    uint32_t padTokenId = 0;
    uint32_t unkTokenId = 0;
    uint32_t sepTokenId = 0;
    uint32_t clsTokenId = 0;
    uint32_t maskTokenId = 0;

    // Chat template
    std::string chatTemplate;
};

// ============================================================================
// Per-layer configuration (allows layer-specific overrides)
// ============================================================================
struct LayerConfig {
    uint32_t layerIndex = 0;
    bool     isMoeLayer = false;
    uint32_t nExpertsThisLayer = 0;
    uint32_t ffnDimThisLayer = 0;
    bool     useSlidingWindow = false;
    QuantFormat weightFormat = QuantFormat::F16;
};

// ============================================================================
// Architecture descriptor — factory + capability bits
// ============================================================================
struct ArchDescriptor {
    ModelArch arch;
    const char* name;
    const char* ggufKeyPrefix;   // e.g. "llama", "qwen2", "deepseek2"
    bool supportsGqa;
    bool supportsMoe;
    bool supportsMla;
    bool supportsVision;
    bool supportsToolCall;
    bool supportsImatrix;
};

class ModelArchRegistry {
public:
    static ModelArchRegistry& instance();

    const ArchDescriptor* lookup(ModelArch arch) const;
    const ArchDescriptor* lookupByName(const std::string& name) const;
    ModelArch parseGGUFArch(const std::string& ggufKey) const;

    // Populate hparams from GGUF metadata (json map)
    bool populateHParams(const std::unordered_map<std::string, nlohmann::json>& meta,
                         ModelHParams& out) const;

private:
    ModelArchRegistry();
    std::vector<ArchDescriptor> descriptors_;
};

} // namespace rdna4
