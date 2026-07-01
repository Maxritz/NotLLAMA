#include "rdna4_context_manager.hpp"
#include <algorithm>
#include <iostream>

namespace rdna4 {

ContextManager::ContextManager(const Config& cfg) : cfg_(cfg) {}

uint32_t ContextManager::createSequence(uint32_t systemPromptLen) {
    auto seq = std::make_unique<SequenceContext>();
    seq->seqId = nextSeqId_++;
    if (systemPromptLen > 0) {
        scoreSystemPromptHigh(seq->seqId, systemPromptLen);
    }
    uint32_t id = seq->seqId;
    sequences_[id] = std::move(seq);
    return id;
}

bool ContextManager::appendTokens(uint32_t seqId, const std::vector<uint32_t>& tokens) {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return false;
    SequenceContext& seq = *it->second;
    for (auto t : tokens) {
        seq.tokens.push_back(t);
        seq.tokenImportance.push_back(0.5f);
    }
    return maybeCompact(seqId);
}

bool ContextManager::appendToken(uint32_t seqId, uint32_t token) {
    return appendTokens(seqId, { token });
}

bool ContextManager::maybeCompact(uint32_t seqId) {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return false;
    SequenceContext& seq = *it->second;
    float ratio = static_cast<float>(seq.tokens.size()) / static_cast<float>(cfg_.maxContextLength);
    if (ratio > cfg_.compactionThreshold) {
        compactNow(seqId);
        return false;
    }
    return true;
}

void ContextManager::compactNow(uint32_t seqId) {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return;
    SequenceContext& seq = *it->second;
    switch (cfg_.strategy) {
        case CompactionStrategy::SLIDING_WINDOW: doSlidingWindowCompact(seq); break;
        case CompactionStrategy::HALF_SLIDE:     doHalfSlideCompact(seq);     break;
        case CompactionStrategy::FIFO:           doFifoCompact(seq);          break;
        case CompactionStrategy::IMPORTANCE:     doImportanceCompact(seq);    break;
        case CompactionStrategy::SUMMARY:        doSummaryCompact(seq);       break;
        default: break;
    }
    compactionCount_++;
}

const std::vector<uint32_t>& ContextManager::getTokens(uint32_t seqId) const {
    static const std::vector<uint32_t> empty;
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return empty;
    return it->second->tokens;
}

uint32_t ContextManager::getTokenCount(uint32_t seqId) const {
    auto it = sequences_.find(seqId);
    return (it != sequences_.end()) ? static_cast<uint32_t>(it->second->tokens.size()) : 0;
}

float ContextManager::getUsageRatio(uint32_t seqId) const {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return 0.0f;
    return static_cast<float>(it->second->tokens.size()) / static_cast<float>(cfg_.maxContextLength);
}

void ContextManager::setKvShiftCallback(KvShiftCallback cb) {
    kvShiftCb_ = std::move(cb);
}

void ContextManager::setTokenImportance(uint32_t seqId, uint32_t pos, float importance) {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return;
    if (pos < it->second->tokenImportance.size())
        it->second->tokenImportance[pos] = importance;
}

void ContextManager::scoreSystemPromptHigh(uint32_t seqId, uint32_t len) {
    auto it = sequences_.find(seqId);
    if (it == sequences_.end()) return;
    auto& scores = it->second->tokenImportance;
    for (uint32_t i = 0; i < len && i < scores.size(); ++i)
        scores[i] = 1.0f;
}

void ContextManager::resetSequence(uint32_t seqId) {
    auto it = sequences_.find(seqId);
    if (it != sequences_.end()) {
        it->second->tokens.clear();
        it->second->tokenImportance.clear();
        it->second->kvCachePos = 0;
        it->second->kvCacheBase = 0;
    }
}

void ContextManager::removeSequence(uint32_t seqId) {
    sequences_.erase(seqId);
}

uint32_t ContextManager::getSequenceCount() const {
    return static_cast<uint32_t>(sequences_.size());
}

uint32_t ContextManager::getTotalTokens() const {
    uint32_t total = 0;
    for (const auto& [id, seq] : sequences_)
        total += static_cast<uint32_t>(seq->tokens.size());
    return total;
}

uint32_t ContextManager::getCompactionCount() const {
    return compactionCount_;
}

void ContextManager::doSlidingWindowCompact(SequenceContext& seq) {
    uint32_t targetLen = static_cast<uint32_t>(cfg_.maxContextLength * cfg_.compactionTarget);
    uint32_t keep = cfg_.preserveSystemPrompt ? std::max(cfg_.systemPromptTokens, targetLen / 2) : 0;
    if (seq.tokens.size() <= targetLen) return;
    uint32_t drop = static_cast<uint32_t>(seq.tokens.size()) - targetLen;
    if (keep >= drop) drop = 0;
    seq.tokens.erase(seq.tokens.begin() + keep, seq.tokens.begin() + keep + drop);
    seq.tokenImportance.erase(seq.tokenImportance.begin() + keep, seq.tokenImportance.begin() + keep + drop);
    if (kvShiftCb_) kvShiftCb_(seq.seqId, seq.kvCacheBase, seq.kvCacheBase + keep, targetLen);
}

void ContextManager::doHalfSlideCompact(SequenceContext& seq) {
    uint32_t targetLen = static_cast<uint32_t>(cfg_.maxContextLength * cfg_.compactionTarget);
    if (seq.tokens.size() <= targetLen) return;
    uint32_t dropLen = static_cast<uint32_t>(seq.tokens.size()) / 2;
    uint32_t keep = cfg_.preserveSystemPrompt ? cfg_.systemPromptTokens : 0;
    if (keep + dropLen > seq.tokens.size()) return;
    seq.tokens.erase(seq.tokens.begin() + keep, seq.tokens.begin() + keep + dropLen);
    seq.tokenImportance.erase(seq.tokenImportance.begin() + keep, seq.tokenImportance.begin() + keep + dropLen);
    if (kvShiftCb_) kvShiftCb_(seq.seqId, seq.kvCacheBase, seq.kvCacheBase + keep, targetLen);
}

void ContextManager::doFifoCompact(SequenceContext& seq) {
    uint32_t targetLen = static_cast<uint32_t>(cfg_.maxContextLength * cfg_.compactionTarget);
    if (seq.tokens.size() <= targetLen) return;
    uint32_t drop = static_cast<uint32_t>(seq.tokens.size()) - targetLen;
    seq.tokens.erase(seq.tokens.begin(), seq.tokens.begin() + drop);
    seq.tokenImportance.erase(seq.tokenImportance.begin(), seq.tokenImportance.begin() + drop);
    if (kvShiftCb_) kvShiftCb_(seq.seqId, seq.kvCacheBase, seq.kvCacheBase + drop, targetLen);
}

void ContextManager::doImportanceCompact(SequenceContext& seq) {
    uint32_t targetLen = static_cast<uint32_t>(cfg_.maxContextLength * cfg_.compactionTarget);
    if (seq.tokens.size() <= targetLen) return;
    auto& scores = seq.tokenImportance;
    std::vector<uint32_t> idx(seq.tokens.size());
    for (uint32_t i = 0; i < seq.tokens.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        return scores[a] > scores[b];
    });
    std::vector<uint32_t> keptTokens;
    std::vector<float> keptScores;
    for (uint32_t i = 0; i < targetLen && i < seq.tokens.size(); ++i) {
        keptTokens.push_back(seq.tokens[idx[i]]);
        keptScores.push_back(scores[idx[i]]);
    }
    seq.tokens = std::move(keptTokens);
    seq.tokenImportance = std::move(keptScores);
    if (kvShiftCb_) kvShiftCb_(seq.seqId, seq.kvCacheBase, seq.kvCacheBase, targetLen);
}

void ContextManager::doSummaryCompact(SequenceContext& seq) {
    // Placeholder: summary compaction not yet implemented
    // Falls through to half-slide behavior
    uint32_t targetLen = static_cast<uint32_t>(cfg_.maxContextLength * cfg_.compactionTarget);
    if (seq.tokens.size() <= targetLen) return;
    uint32_t dropLen = static_cast<uint32_t>(seq.tokens.size()) / 2;
    seq.tokens.erase(seq.tokens.begin(), seq.tokens.begin() + dropLen);
    seq.tokenImportance.erase(seq.tokenImportance.begin(), seq.tokenImportance.begin() + dropLen);
}

} // namespace rdna4
