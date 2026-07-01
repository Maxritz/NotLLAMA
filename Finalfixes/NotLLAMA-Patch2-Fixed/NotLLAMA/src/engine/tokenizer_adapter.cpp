#include "engine/tokenizer_adapter.hpp"

namespace notllama {

TokenizerAdapter::TokenizerAdapter(rdna4::Tokenizer* tokenizer) : tokenizer_(tokenizer) {}

std::vector<uint32_t> TokenizerAdapter::Encode(const std::string& text) {
    if (!tokenizer_) return {};
    return tokenizer_->encode(text);
}

std::string TokenizerAdapter::Decode(const std::vector<uint32_t>& tokens) {
    if (!tokenizer_) return {};
    return tokenizer_->decode(tokens);
}

uint32_t TokenizerAdapter::GetEOSToken() const {
    return tokenizer_ ? tokenizer_->eosTokenId() : 0;
}

uint32_t TokenizerAdapter::GetBOSToken() const {
    return tokenizer_ ? tokenizer_->bosTokenId() : 0;
}

uint32_t TokenizerAdapter::GetPadToken() const {
    return tokenizer_ ? tokenizer_->padTokenId() : 0;
}

uint32_t TokenizerAdapter::GetNewlineToken() const {
    // Existing tokenizer does not expose a dedicated newline token.
    return tokenizer_ ? tokenizer_->eosTokenId() : 0;
}

} // namespace notllama
