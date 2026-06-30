#pragma once
#include "engine/itokenizer.hpp"
#include "rdna4_tokenizer.hpp"

namespace notllama {

class TokenizerAdapter : public ITokenizer {
public:
    explicit TokenizerAdapter(rdna4::Tokenizer* tokenizer);
    ~TokenizerAdapter() override = default;

    std::vector<uint32_t> Encode(const std::string& text) override;
    std::string Decode(const std::vector<uint32_t>& tokens) override;

    uint32_t GetEOSToken() const override;
    uint32_t GetBOSToken() const override;
    uint32_t GetPadToken() const override;
    uint32_t GetNewlineToken() const override;

private:
    rdna4::Tokenizer* tokenizer_;
};

} // namespace notllama
