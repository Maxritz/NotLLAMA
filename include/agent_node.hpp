#pragma once
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <queue>
#include <chrono>
#include <functional>
#include <nlohmann/json.hpp>

namespace notllama {

// ---------------------------------------------------------------------------
// Agent Protocol — every NotLLAMA instance can be an Agent Node.
// Nodes talk to each other over HTTP/JSON, forming a mesh of reasoning peers.
// ---------------------------------------------------------------------------

// Identity of a remote agent peer
struct AgentPeer {
    std::string name;          // Human-readable name (e.g., "deepseek-node")
    std::string host;          // IP or hostname
    int port = 0;              // HTTP port
    std::string model_loaded;  // Which model this node runs
    std::vector<std::string> tags; // Capabilities (code, english, math, etc.)
    std::chrono::steady_clock::time_point last_seen;
    bool alive = false;
    float avg_response_ms = 0;
};

// A reasoning request sent between agents
struct AgentReasonRequest {
    std::string request_id;
    std::string from_agent;
    std::string query;         // The actual prompt/question
    uint32_t max_tokens = 256;
    float temperature = 0.8f;
    bool stream = false;
    std::vector<std::string> already_asked; // Agents already consulted (prevent loops)
    uint32_t hop_count = 0;    // How many hops so far (prevent infinite relay)
    static constexpr uint32_t MAX_HOPS = 5;
};

// A reasoning response returned by an agent
struct AgentReasonResponse {
    std::string request_id;
    std::string from_agent;
    std::string answer;
    float confidence = 1.0f;
    uint64_t tokens_generated = 0;
    uint64_t elapsed_ms = 0;
    bool needs_more_reasoning = false;
    std::vector<std::string> suggested_followups;
};

// Model-creation request (for distributed training / model distillation)
struct ModelCreateRequest {
    std::string model_name;
    std::string base_model_type;   // e.g., "llama", "qwen", "gemma"
    size_t size_mb = 0;            // Target size
    std::string quant_format;      // e.g., "Q4_K", "Q6_K", "Q8_0"
    std::string architecture_type; // "dense", "moe", "mixed"
    std::vector<std::string> source_agents; // Which agents to pull knowledge from
    bool distill_from_conversation = false; // Learn from agent-to-agent chat
};

// Debug log entry for agent-to-agent conversation
struct AgentLogEntry {
    std::chrono::system_clock::time_point timestamp;
    std::string from_agent;
    std::string to_agent;
    std::string level;       // "DEBUG", "INFO", "WARN", "ERROR", "REASON"
    std::string message;
    nlohmann::json metadata;
};

// Configuration for an agent node
struct AgentConfig {
    std::string agent_name = "notllama-agent";   // Unique name
    int agent_port = 0;                            // 0 = no agent listener
    std::vector<std::string> agent_peers;          // "host:port,name,tags" entries
    bool use_graphify = false;                     // External Graphify available?
    std::string graphify_url;                      // Graphify endpoint
    bool use_mcp = false;                          // External MCP server available?
    std::string mcp_url;                           // MCP endpoint
    bool enable_reason_sharing = true;             // Share reasoning with peers
    bool enable_model_distill = false;             // Allow cross-agent distillation
    std::string log_file;                          // Circular debug log path
    size_t log_max_size_mb = 50;                   // Per-log max size
    uint32_t log_verbosity = 2;                    // 0=errors only, 1=info, 2=debug, 3=trace
};

// ---------------------------------------------------------------------------
// AgentNode — core of the distributed system.
// Each NotLLAMA process can run one AgentNode that connects to peers.
// ---------------------------------------------------------------------------
class AgentNode {
public:
    AgentNode(const AgentConfig& cfg);
    ~AgentNode();

    // Peer management
    void AddPeer(const std::string& host, int port,
                 const std::string& name, const std::string& tags);
    void RemovePeer(const std::string& name);
    std::vector<AgentPeer> GetPeers() const;
    void RefreshPeers();          // Ping all peers, update alive status

    // Reasoning: ask a specific peer
    AgentReasonResponse AskPeer(const std::string& peer_name,
                                 const AgentReasonRequest& req);

    // Reasoning: broadcast to all capable peers, aggregate best response
    AgentReasonResponse AskAll(const AgentReasonRequest& req);

    // Reasoning: route query to best peer based on tags/capability
    AgentReasonResponse RouteQuery(const std::string& query,
                                    uint32_t max_tokens = 256);

    // Handle incoming reasoning request (called by web_server)
    nlohmann::json HandleReasonRequest(const nlohmann::json& req_json);

    // Handle incoming status ping
    nlohmann::json HandleStatusRequest();

    // Model creation across agents
    bool RequestModelCreation(const ModelCreateRequest& req);
    nlohmann::json HandleModelCreateRequest(const nlohmann::json& req_json);

    // Logging
    void Log(const std::string& level, const std::string& message,
             const nlohmann::json& meta = {});
    void LogReason(const std::string& to_agent, const std::string& query,
                   const std::string& response, float confidence);
    std::vector<AgentLogEntry> GetRecentLogs(size_t count = 100) const;
    void FlushLogs();

    // Status
    bool IsRunning() const { return running_.load(); }
    std::string GetName() const { return config_.agent_name; }
    int GetPort() const { return config_.agent_port; }

    // Web endpoint handlers (called by WebServer)
    std::string HandleAgentEndpoint(const std::string& subpath,
                                     const nlohmann::json& body);

    static std::string Version() { return "0.3.0"; }

private:
    AgentConfig config_;
    std::map<std::string, AgentPeer> peers_; // name -> Peer
    mutable std::mutex peers_mutex_;

    std::atomic<bool> running_{false};
    std::thread heartbeat_thread_;
    std::thread log_flush_thread_;

    // Log ring buffer
    std::vector<AgentLogEntry> log_buffer_;
    mutable std::mutex log_mutex_;
    size_t log_write_index_ = 0;
    size_t log_max_entries_ = 10000;
    std::atomic<size_t> log_sequence_{0};

    // Internal
    void HeartbeatLoop();        // Keep peer connections alive
    void LogFlushLoop();         // Periodic log flush
    bool HttpPost(const std::string& host, int port,
                  const std::string& path,
                  const nlohmann::json& body,
                  nlohmann::json& out_response);
    std::vector<std::string> ParsePeerEntry(const std::string& entry);
    void RotateLogFile();
    std::string GetLogPath(const std::string& suffix);
};

} // namespace notllama
