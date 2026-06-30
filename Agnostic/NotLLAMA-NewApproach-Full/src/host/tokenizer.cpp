#include "rdna4_tokenizer.hpp"
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdio>

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
        char hex[8];
        snprintf(hex, sizeof(hex), "<0x%02X>", c);
        std::string byteToken(hex);
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

// GPT-2 style pre-tokenization: split text into chunks matching regex patterns
static std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0;
    while (i < text.size()) {
        char c = text[i];

        // Contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (c == '\'' && i + 1 < text.size()) {
            char next = text[i + 1];
            if (next == 's' || next == 't' || next == 'd' || next == 'm') {
                chunks.push_back(text.substr(i, 2));
                i += 2;
                continue;
            }
            if (i + 2 < text.size()) {
                std::string tri = text.substr(i, 3);
                if (tri == "'re" || tri == "'ve" || tri == "'ll") {
                    chunks.push_back(tri);
                    i += 3;
                    continue;
                }
            }
        }

        // Leading space + letters
        if (c == ' ') {
            size_t start = i;
            i++;  // consume space
            size_t letterStart = i;
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) i++;
            if (i > letterStart) {
                chunks.push_back(text.substr(start, i - start));
                continue;
            }
            // Just a space, will be handled by whitespace rule
            chunks.push_back(" ");
            continue;
        }

        // Letters (no leading space)
        if (std::isalpha(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) i++;
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // Numbers (with optional leading space)
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) i++;
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // Non-space/non-letter/non-number sequences
        if (c != ' ' && !std::isalpha(static_cast<unsigned char>(c)) && !std::isdigit(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < text.size()) {
                char nc = text[i];
                if (nc == ' ' || std::isalpha(static_cast<unsigned char>(nc)) || std::isdigit(static_cast<unsigned char>(nc))) break;
                i++;
            }
            if (i > start) {
                chunks.push_back(text.substr(start, i - start));
                continue;
            }
        }

        // Whitespace (spaces, tabs, newlines)
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            size_t start = i;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) i++;
            chunks.push_back(text.substr(start, i - start));
            continue;
        }

        // Fallback: single character
        chunks.push_back(std::string(1, c));
        i++;
    }
    return chunks;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> result;
    result.push_back(bosId);

    // GPT-2 style pre-tokenization
    auto chunks = pretokenize(text);

    for (const auto& word : chunks) {
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
                unsigned char ch = static_cast<unsigned char>(word[i]);
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", ch);
                std::string byteToken(hex);
                auto bit = vocab.find(byteToken);
                if (bit != vocab.end()) {
                    wordPieces.push_back(byteToken);
                } else {
                    wordPieces.push_back(std::string(1, word[i]));
                }
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
