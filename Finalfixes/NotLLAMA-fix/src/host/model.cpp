#include "rdna4.hpp"
#include "loaders/gguf.h"

#include <iostream>
#include <cstring>
#include <cmath>

namespace rdna4 {

std::unique_ptr<Model> Model::load(
    Context& ctx, MemoryManager& mem, Scheduler& sched,
    const std::string& path, std::string* error)
{
    auto model = std::unique_ptr<Model>(new Model);
    model->m_ctx = &ctx;
    model->m_mem = &mem;
    model->m_sched = &sched;

    notllama::GGUFLoader loader;
    if (!loader.load(path)) {
        if (error) *error = "Failed to load GGUF: " + path;
        return nullptr;
    }

    const auto& meta = loader.metadata();
    auto& cfg = model->m_config;
    if (meta.nLayers > 0) cfg.nLayers = meta.nLayers;
    if (meta.nHeads > 0) cfg.nHeads = meta.nHeads;
    if (meta.nKVHeads > 0) cfg.nKvHeads = meta.nKVHeads;
    if (meta.nEmbd > 0) cfg.dim = meta.nEmbd;
    if (meta.nFF > 0) cfg.hiddenDim = meta.nFF;
    if (meta.nVocab > 0) cfg.vocabSize = meta.nVocab;
    if (meta.headDim > 0) cfg.headDim = meta.headDim;
    if (meta.nCtx > 0) cfg.seqLenMax = meta.nCtx;
    if (meta.ropeFreqBase > 0) cfg.ropeFreqBase = meta.ropeFreqBase;

    std::cout << "Model: dim=" << cfg.dim << " L=" << cfg.nLayers
              << " H=" << cfg.nHeads << " KV=" << cfg.nKvHeads
              << " D=" << cfg.headDim << " FF=" << cfg.hiddenDim
              << " V=" << cfg.vocabSize << " ctx=" << cfg.seqLenMax << std::endl;

    loader.uploadToGPU(ctx.device(), ctx.physicalDevice(), ctx.acePool(0), ctx.aceQueue(0));

    // Allocate KV-cache images
    model->m_kvCacheK.resize(cfg.nLayers);
    model->m_kvCacheV.resize(cfg.nLayers);
    for (uint32_t layer = 0; layer < cfg.nLayers; ++layer) {
        auto kResult = mem.allocateImage(cfg.seqLenMax, cfg.nKvHeads * cfg.headDim, 1,
                                         VK_FORMAT_R16G16_SFLOAT, true);
        if (!kResult) { if (error) *error = "Failed to allocate K cache"; return nullptr; }
        model->m_kvCacheK[layer] = *kResult;

        auto vResult = mem.allocateImage(cfg.seqLenMax, cfg.nKvHeads * cfg.headDim, 1,
                                         VK_FORMAT_R16G16_SFLOAT, true);
        if (!vResult) { if (error) *error = "Failed to allocate V cache"; return nullptr; }
        model->m_kvCacheV[layer] = *vResult;
    }

    std::cout << "KV cache: " << cfg.nLayers << " layers x 2 x "
              << cfg.seqLenMax << " x " << (cfg.nKvHeads * cfg.headDim) << " FP16" << std::endl;

    std::cout << "Model loaded successfully." << std::endl;
    return model;
}

Model::~Model() {
    if (m_mem) {
        for (auto& img : m_kvCacheK) m_mem->freeImage(img);
        for (auto& img : m_kvCacheV) m_mem->freeImage(img);
    }
}

} // namespace rdna4
