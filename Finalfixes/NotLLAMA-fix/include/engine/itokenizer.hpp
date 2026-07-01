#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace notllama {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<uint32_t> Encode(const std::string& text) = 0;
    virtual std::string Decode(const std::vector<uint32_t>& tokens) = 0;

    virtual uint32_t GetEOSToken() const = 0;
    virtual uint32_t GetBOSToken() const = 0;
    virtual uint32_t GetPadToken() const = 0;
    virtual uint32_t GetNewlineToken() const = 0;
};

} // namespace notllama
