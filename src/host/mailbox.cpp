#include "rdna4.hpp"
#include <atomic>
#include <cstdint>

namespace rdna4 {

struct Mailbox {
    alignas(64) std::atomic<uint32_t> tokenReady{0};
    alignas(64) std::atomic<uint32_t> tokenAck{0};
    uint32_t tokenId;
    uint32_t pad[13];

    void drop(uint32_t token) {
        tokenId = token;
        tokenReady.store(1, std::memory_order_release);
    }

    bool poll() {
        return tokenReady.load(std::memory_order_acquire) != 0;
    }

    void ack() {
        tokenAck.store(1, std::memory_order_release);
    }
};

} // namespace rdna4
