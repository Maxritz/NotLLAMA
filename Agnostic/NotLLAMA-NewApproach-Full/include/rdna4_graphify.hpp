#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace rdna4 {

struct GraphQueryResult {
    std::string question;
    std::string answer;
    std::vector<std::string> sourceNodes;
    std::vector<std::string> sourceLocations;
    float confidence = 0.0f;
    uint64_t queryTimeMs = 0;
};

struct GraphifyConfig {
    std::string graphPath = "graphify-out/graph.json";
    bool autoUpdate = true;
    bool queryBeforeRead = true;
    std::string preferredMode = "dfs";
    uint32_t tokenBudget = 1500;
    bool budgetCap = true;
    uint32_t cacheSize = 32;
};

class GraphifyClient {
public:
    explicit GraphifyClient(const GraphifyConfig& cfg = {});
    ~GraphifyClient();

    GraphQueryResult query(const std::string& question);
    bool isStale() const;
    bool updateGraph();
    std::vector<std::string> getRelatedNodes(const std::string& symbol, int depth = 2);
    void clearCache();

    static bool isAvailable();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace rdna4
