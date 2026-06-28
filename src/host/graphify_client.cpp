#include "rdna4_graphify.hpp"
#include <iostream>

namespace rdna4 {

struct GraphifyClient::Impl {
    GraphifyConfig config;
};

GraphifyClient::GraphifyClient(const GraphifyConfig& cfg)
    : pImpl(std::make_unique<Impl>())
{
    pImpl->config = cfg;
}

GraphifyClient::~GraphifyClient() = default;

GraphQueryResult GraphifyClient::query(const std::string& question) {
    GraphQueryResult result;
    result.question = question;
    result.answer = "stub: query not yet implemented";
    return result;
}

bool GraphifyClient::isStale() const {
    return false;
}

bool GraphifyClient::updateGraph() {
    // TODO: subprocess graphify rebuild
    return false;
}

std::vector<std::string> GraphifyClient::getRelatedNodes(const std::string& symbol, int depth) {
    (void)depth;
    return { symbol };
}

void GraphifyClient::clearCache() {
    // TODO: clear LRU cache
}

bool GraphifyClient::isAvailable() {
    return false;
}

} // namespace rdna4
