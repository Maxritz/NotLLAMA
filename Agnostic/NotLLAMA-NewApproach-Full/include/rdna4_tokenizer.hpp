#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>

namespace rdna4 {

// BPE tokenizer with Byte-Fallback decoding.
// Loads vocab and merges from GGUF metadata.
class Tokenizer {
public:
    std::unordered_map<std::string, uint32_t> vocab;
    std::vector<std::string> idToToken;
    std::vector<std::pair<std::string, std::string>> merges;

    // Special tokens
    uint32_t bosId = 1;
    uint32_t eosId = 2;
    uint32_t padId = 0;
    uint32_t unkId = 3;
    std::string bosToken = "<s>";
    std::string eosToken = "</s>";
    std::string padToken = "<pad>";
    std::string unkToken = "<unk>";

    // GGUF metadata keys we look for
    void loadFromGGUF(const std::vector<std::string>& tokens,
                      const std::vector<std::string>& mergeRules,
                      uint32_t bos, uint32_t eos, uint32_t pad, uint32_t unk);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;

    uint32_t bosTokenId() const { return bosId; }
    uint32_t eosTokenId() const { return eosId; }
    uint32_t padTokenId() const { return padId; }

private:
    std::vector<std::string> byteFallbackEncode(const std::string& text) const;
    std::string byteFallbackDecode(const std::string& token) const;
    std::vector<std::string> bpeMerge(const std::vector<std::string>& wordPieces) const;
};

} // namespace rdna4
