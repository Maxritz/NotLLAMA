#include "rdna4_tokenizer.hpp"
#include <sstream>
#include <algorithm>
#include <cstring>

namespace rdna4 {

void Tokenizer::loadFromGGUF(const std::vector<std::string>& tokens,
                              const std::vector<std::string>& mergeRules,
                              uint32_t bos, uint32_t eos, uint32_t pad, uint32_t unk) {
    bosId = bos;
    eosId = eos;
    padId = pad;
    unkId = unk;

    idToToken = tokens;
    for (size_t i = 0; i < tokens.size(); ++i) {
        vocab[tokens[i]] = static_cast<uint32_t>(i);
    }

    for (const auto& rule : mergeRules) {
        size_t spacePos = rule.find(' ');
        if (spacePos != std::string::npos) {
            merges.push_back({rule.substr(0, spacePos), rule.substr(spacePos + 1)});
        }
    }
}

std::vector<std::string> Tokenizer::byteFallbackEncode(const std::string& text) const {
    std::vector<std::string> pieces;
    for (unsigned char c : text) {
        // Byte fallback: map bytes to special tokens like <0xXX>
        std::string byteToken = "<0x" + std::to_string(c) + ">";
        auto it = vocab.find(byteToken);
        if (it != vocab.end()) {
            pieces.push_back(byteToken);
        } else {
            pieces.push_back(std::string(1, static_cast<char>(c)));
        }
    }
    return pieces;
}

std::string Tokenizer::byteFallbackDecode(const std::string& token) const {
    // Reverse byte fallback: <0xXX> -> byte
    if (token.size() > 4 && token.substr(0, 3) == "<0x" && token.back() == '>') {
        try {
            int val = std::stoi(token.substr(3, token.size() - 4), nullptr, 16);
            return std::string(1, static_cast<char>(val));
        } catch (...) {}
    }
    return token;
}

std::vector<std::string> Tokenizer::bpeMerge(const std::vector<std::string>& wordPieces) const {
    if (wordPieces.empty()) return wordPieces;

    auto pieces = wordPieces;

    for (const auto& merge : merges) {
        std::vector<std::string> newPieces;
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (i + 1 < pieces.size() && pieces[i] == merge.first && pieces[i+1] == merge.second) {
                newPieces.push_back(pieces[i] + pieces[i+1]);
                i++;
            } else {
                newPieces.push_back(pieces[i]);
            }
        }
        pieces = newPieces;
        if (pieces.size() == 1) break;
    }

    return pieces;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> result;
    result.push_back(bosId);

    // Simple word-level tokenization (pre-tokenization)
    // Real BPE uses regex patterns like GPT-2's pre-tokenizer
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        // Try to find the word directly in vocab
        auto it = vocab.find(word);
        if (it != vocab.end()) {
            result.push_back(it->second);
            continue;
        }

        // Try subword decomposition
        std::vector<std::string> wordPieces;
        for (size_t i = 0; i < word.size(); ) {
            size_t longest = 0;
            uint32_t bestId = unkId;
            std::string bestToken;

            for (size_t len = 1; len <= word.size() - i && len <= 32; ++len) {
                std::string sub = word.substr(i, len);
                auto vit = vocab.find(sub);
                if (vit != vocab.end() && len > longest) {
                    longest = len;
                    bestId = vit->second;
                    bestToken = sub;
                }
            }

            if (longest > 0) {
                wordPieces.push_back(bestToken);
                i += longest;
            } else {
                // Byte fallback
                wordPieces.push_back(std::string(1, word[i]));
                i++;
            }
        }

        // Apply BPE merges
        wordPieces = bpeMerge(wordPieces);

        for (const auto& piece : wordPieces) {
            auto vit = vocab.find(piece);
            if (vit != vocab.end()) {
                result.push_back(vit->second);
            } else {
                result.push_back(unkId);
            }
        }
    }

    result.push_back(eosId);
    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    for (uint32_t id : tokens) {
        if (id == bosId || id == eosId || id == padId) continue;
        if (id < idToToken.size()) {
            result += byteFallbackDecode(idToToken[id]);
        }
    }
    return result;
}

} // namespace rdna4
