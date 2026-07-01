#include "agent_node.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// Platform sockets for HTTP client (peer-to-peer)
// ---------------------------------------------------------------------------
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    #define AGENT_CLOSE closesocket
    #define AGENT_EWOULDBLOCK WSAEWOULDBLOCK
    typedef int socklen_t;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <netdb.h>
    #define AGENT_CLOSE close
    #define AGENT_EWOULDBLOCK EWOULDBLOCK
#endif

namespace notllama {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string TimestampISO() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return ss.str();
}

static std::string EscapeJsonString(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

AgentNode::AgentNode(const AgentConfig& cfg) : config_(cfg) {
    log_max_entries_ = static_cast<size_t>(cfg.log_max_size_mb) * 1024 * 1024 / 256;
    log_buffer_.reserve(log_max_entries_);

    // Parse initial peers from config
    for (const auto& entry : cfg.agent_peers) {
        auto parts = ParsePeerEntry(entry);
        if (parts.size() >= 3) {
            int port = 8080;
            try { port = std::stoi(parts[1]); } catch (...) {}
            AddPeer(parts[0], port, parts[2],
                    parts.size() > 3 ? parts[3] : "");
        }
    }

    running_ = true;
    heartbeat_thread_ = std::thread(&AgentNode::HeartbeatLoop, this);
    if (!cfg.log_file.empty()) {
        log_flush_thread_ = std::thread(&AgentNode::LogFlushLoop, this);
    }

    Log("INFO", "AgentNode started: " + cfg.agent_name +
        " | port=" + std::to_string(cfg.agent_port) +
        " | peers=" + std::to_string(peers_.size()));
}

AgentNode::~AgentNode() {
    running_ = false;
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    if (log_flush_thread_.joinable()) log_flush_thread_.join();
    FlushLogs();
}

// ---------------------------------------------------------------------------
// Peer Management
// ---------------------------------------------------------------------------

void AgentNode::AddPeer(const std::string& host, int port,
                        const std::string& name, const std::string& tags) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    AgentPeer p;
    p.name = name;
    p.host = host;
    p.port = port;
    p.last_seen = std::chrono::steady_clock::now();
    p.alive = false;

    // Parse tags
    std::stringstream ss(tags);
    std::string tag;
    while (std::getline(ss, tag, ',')) {
        if (!tag.empty()) p.tags.push_back(tag);
    }

    peers_[name] = p;
    Log("INFO", "Peer added: " + name + " @ " + host + ":" + std::to_string(port));
}

void AgentNode::RemovePeer(const std::string& name) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    peers_.erase(name);
    Log("INFO", "Peer removed: " + name);
}

std::vector<AgentPeer> AgentNode::GetPeers() const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    std::vector<AgentPeer> result;
    for (const auto& [name, p] : peers_) { (void)name; result.push_back(p); }
    return result;
}

void AgentNode::RefreshPeers() {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (auto& [name, peer] : peers_) { (void)name;
        nlohmann::json resp;
        auto start = std::chrono::steady_clock::now();
        bool ok = HttpPost(peer.host, peer.port, "/agent/status", {}, resp);
        auto elapsed = std::chrono::steady_clock::now() - start;
        float ms = std::chrono::duration<float, std::milli>(elapsed).count();

        peer.last_seen = std::chrono::steady_clock::now();
        peer.alive = ok;
        if (ok) {
            peer.avg_response_ms = peer.avg_response_ms * 0.7f + ms * 0.3f;
            if (resp.contains("model_loaded")) {
                peer.model_loaded = resp["model_loaded"].get<std::string>();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP Client (synchronous, for peer-to-peer)
// ---------------------------------------------------------------------------

bool AgentNode::HttpPost(const std::string& host, int port,
                         const std::string& path,
                         const nlohmann::json& body,
                         nlohmann::json& out_response) {
#ifdef _WIN32
    WSADATA wsa;
    static bool wsa_init = false;
    if (!wsa_init) { WSAStartup(MAKEWORD(2, 2), &wsa); wsa_init = true; }
#endif

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    // Resolve hostname
    hostent* he = gethostbyname(host.c_str());
    if (!he) { AGENT_CLOSE(sock); return false; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    // Set timeout
#ifdef _WIN32
    DWORD timeout = 5000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));
#else
    struct timeval tv;
    tv.tv_sec = 5; tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        AGENT_CLOSE(sock);
        return false;
    }

    // Build HTTP POST
    std::string json_body = body.dump();
    std::stringstream request;
    request << "POST " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << ":" << port << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << json_body.size() << "\r\n";
    request << "Connection: close\r\n";
    request << "\r\n";
    request << json_body;

    std::string req_str = request.str();
    if (send(sock, req_str.c_str(), static_cast<int>(req_str.size()), 0) < 0) {
        AGENT_CLOSE(sock);
        return false;
    }

    // Read response
    char buffer[65536];
    std::string response;
    int n;
    while ((n = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[n] = '\0';
        response += buffer;
    }
    AGENT_CLOSE(sock);

    // Parse JSON from body (skip headers)
    size_t body_start = response.find("\r\n\r\n");
    if (body_start == std::string::npos) return false;
    body_start += 4;

    try {
        out_response = nlohmann::json::parse(response.substr(body_start));
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> AgentNode::ParsePeerEntry(const std::string& entry) {
    std::vector<std::string> parts;
    std::stringstream ss(entry);
    std::string part;
    while (std::getline(ss, part, ',')) {
        parts.push_back(part);
    }
    return parts;
}

// ---------------------------------------------------------------------------
// Reasoning
// ---------------------------------------------------------------------------

AgentReasonResponse AgentNode::AskPeer(const std::string& peer_name,
                                        const AgentReasonRequest& req) {
    AgentReasonResponse resp;
    resp.request_id = req.request_id;
    resp.from_agent = peer_name;

    AgentPeer peer;
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto it = peers_.find(peer_name);
        if (it == peers_.end()) {
            resp.answer = "[ERROR: Peer '" + peer_name + "' not found]";
            return resp;
        }
        peer = it->second;
    }

    nlohmann::json req_json;
    req_json["request_id"] = req.request_id;
    req_json["from_agent"] = config_.agent_name;
    req_json["query"] = req.query;
    req_json["max_tokens"] = req.max_tokens;
    req_json["temperature"] = req.temperature;
    req_json["already_asked"] = req.already_asked;
    req_json["hop_count"] = req.hop_count + 1;

    nlohmann::json peer_resp;
    auto start = std::chrono::steady_clock::now();
    bool ok = HttpPost(peer.host, peer.port, "/agent/reason", req_json, peer_resp);
    auto elapsed = std::chrono::steady_clock::now() - start;
    resp.elapsed_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    if (ok && peer_resp.contains("answer")) {
        resp.answer = peer_resp["answer"].get<std::string>();
        resp.confidence = peer_resp.value("confidence", 0.8f);
        resp.tokens_generated = peer_resp.value("tokens_generated", 0);
    } else {
        resp.answer = "[ERROR: No response from " + peer_name + "]";
        resp.confidence = 0.0f;
    }

    LogReason(peer_name, req.query, resp.answer, resp.confidence);
    return resp;
}

AgentReasonResponse AgentNode::AskAll(const AgentReasonRequest& req) {
    AgentReasonResponse best;
    best.confidence = 0.0f;

    auto peers = GetPeers();
    for (auto& peer : peers) {
        if (!peer.alive) continue;
        if (std::find(req.already_asked.begin(), req.already_asked.end(),
                      peer.name) != req.already_asked.end()) continue;

        auto resp = AskPeer(peer.name, req);
        if (resp.confidence > best.confidence) {
            best = resp;
        }
    }

    if (best.answer.empty()) {
        best.answer = "[No peers available to answer query]";
        best.confidence = 0.0f;
    }
    return best;
}

AgentReasonResponse AgentNode::RouteQuery(const std::string& query,
                                           uint32_t max_tokens) {
    AgentReasonRequest req;
    req.request_id = config_.agent_name + "-" +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    req.from_agent = config_.agent_name;
    req.query = query;
    req.max_tokens = max_tokens;
    req.already_asked.push_back(config_.agent_name);

    // If only one peer, ask directly
    auto peers = GetPeers();
    if (peers.size() == 1 && peers[0].alive) {
        return AskPeer(peers[0].name, req);
    }

    // Otherwise ask all and pick best
    return AskAll(req);
}

// ---------------------------------------------------------------------------
// Incoming request handlers (called by WebServer)
// ---------------------------------------------------------------------------

nlohmann::json AgentNode::HandleReasonRequest(const nlohmann::json& req_json) {
    std::string query = req_json.value("query", "");
    std::string request_id = req_json.value("request_id", "unknown");
    std::string from = req_json.value("from_agent", "unknown");
    uint32_t hop = req_json.value("hop_count", 0);

    Log("DEBUG", "Reasoning request from " + from + ": " + query.substr(0, 100));

    // Prevent infinite loops
    if (hop > AgentReasonRequest::MAX_HOPS) {
        return {
            {"answer", "[HOP_LIMIT: Query has been relayed too many times]"},
            {"confidence", 0.0},
            {"request_id", request_id}
        };
    }

    // If we can't answer locally, forward to another peer
    auto peers = GetPeers();
    std::vector<std::string> asked = req_json.value("already_asked",
                                                      std::vector<std::string>{});
    asked.push_back(config_.agent_name);

    for (auto& peer : peers) {
        if (!peer.alive) continue;
        if (std::find(asked.begin(), asked.end(), peer.name) != asked.end()) continue;

        AgentReasonRequest fwd;
        fwd.request_id = request_id;
        fwd.from_agent = config_.agent_name;
        fwd.query = query;
        fwd.max_tokens = req_json.value("max_tokens", 256);
        fwd.already_asked = asked;
        fwd.hop_count = hop;

        auto resp = AskPeer(peer.name, fwd);
        if (resp.confidence > 0.3f) {
            nlohmann::json j;
            j["answer"] = resp.answer;
            j["confidence"] = resp.confidence;
            j["tokens_generated"] = resp.tokens_generated;
            j["request_id"] = request_id;
            j["from_agent"] = config_.agent_name;
            return j;
        }
    }

    // Return local indication that we received it but can't answer
    return {
        {"answer", "[ACK: Agent '" + config_.agent_name +
                   "' received query but cannot answer locally. Forwarded to " +
                   std::to_string(peers.size()) + " peers.]"},
        {"confidence", 0.1},
        {"request_id", request_id},
        {"from_agent", config_.agent_name}
    };
}

nlohmann::json AgentNode::HandleStatusRequest() {
    auto peers = GetPeers();
    size_t alive_count = 0;
    for (const auto& p : peers) if (p.alive) alive_count++;

    return {
        {"agent_name", config_.agent_name},
        {"version", Version()},
        {"agent_port", config_.agent_port},
        {"peers_total", peers.size()},
        {"peers_alive", alive_count},
        {"use_graphify", config_.use_graphify},
        {"graphify_url", config_.graphify_url},
        {"use_mcp", config_.use_mcp},
        {"mcp_url", config_.mcp_url},
        {"reason_sharing", config_.enable_reason_sharing},
        {"model_distill", config_.enable_model_distill}
    };
}

std::string AgentNode::HandleAgentEndpoint(const std::string& subpath,
                                            const nlohmann::json& body) {
    nlohmann::json result;
    if (subpath == "/agent/reason" || subpath == "agent/reason") {
        result = HandleReasonRequest(body);
    } else if (subpath == "/agent/status" || subpath == "agent/status") {
        result = HandleStatusRequest();
    } else if (subpath == "/agent/model/create" || subpath == "agent/model/create") {
        result = HandleModelCreateRequest(body);
    } else {
        result = {{"error", "Unknown agent endpoint: " + subpath}};
    }
    return result.dump();
}

// ---------------------------------------------------------------------------
// Model Creation
// ---------------------------------------------------------------------------

bool AgentNode::RequestModelCreation(const ModelCreateRequest& req) {
    nlohmann::json j;
    j["model_name"] = req.model_name;
    j["base_model_type"] = req.base_model_type;
    j["size_mb"] = req.size_mb;
    j["quant_format"] = req.quant_format;
    j["architecture_type"] = req.architecture_type;
    j["source_agents"] = req.source_agents;
    j["distill_from_conversation"] = req.distill_from_conversation;

    // Send to all peers that support model creation
    auto peers = GetPeers();
    bool any_ok = false;
    for (auto& peer : peers) {
        if (!peer.alive) continue;
        bool has_code = false;
        for (const auto& t : peer.tags) if (t == "code" || t == "train") has_code = true;

        nlohmann::json resp;
        if (HttpPost(peer.host, peer.port, "/agent/model/create", j, resp)) {
            any_ok = true;
            Log("INFO", "Model creation request sent to " + peer.name +
                ": " + req.model_name);
        }
    }
    return any_ok;
}

nlohmann::json AgentNode::HandleModelCreateRequest(const nlohmann::json& req_json) {
    std::string name = req_json.value("model_name", "");
    std::string type = req_json.value("base_model_type", "");
    size_t size_mb = req_json.value("size_mb", 0);
    std::string quant = req_json.value("quant_format", "Q4_K");
    std::string arch = req_json.value("architecture_type", "dense");

    Log("INFO", "Model creation request: " + name + " (" + type + ", " +
        std::to_string(size_mb) + "MB, " + quant + ", " + arch + ")");

    // This is a coordination point — the actual model creation would involve
    // spinning up training or conversion. Here we acknowledge and log.
    return {
        {"status", "accepted"},
        {"model_name", name},
        {"agent", config_.agent_name},
        {"message", "Model creation queued. Use --create-model to finalize locally."}
    };
}

// ---------------------------------------------------------------------------
// Logging (circular buffer + file)
// ---------------------------------------------------------------------------

void AgentNode::Log(const std::string& level, const std::string& message,
                    const nlohmann::json& meta) {
    if (config_.log_verbosity < 2 && level == "DEBUG") return;
    if (config_.log_verbosity < 1 && level == "INFO") return;

    AgentLogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.from_agent = config_.agent_name;
    entry.to_agent = "";
    entry.level = level;
    entry.message = message;
    entry.metadata = meta;

    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (log_buffer_.size() < log_max_entries_) {
            log_buffer_.push_back(entry);
        } else {
            log_buffer_[log_write_index_] = entry;
            log_write_index_ = (log_write_index_ + 1) % log_max_entries_;
        }
    }
    log_sequence_++;
}

void AgentNode::LogReason(const std::string& to_agent, const std::string& query,
                          const std::string& response, float confidence) {
    AgentLogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.from_agent = config_.agent_name;
    entry.to_agent = to_agent;
    entry.level = "REASON";
    entry.message = "Q: " + query.substr(0, 200) + " | A: " + response.substr(0, 200);
    entry.metadata = {
        {"confidence", confidence},
        {"query_full", query},
        {"response_full", response}
    };

    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        if (log_buffer_.size() < log_max_entries_) {
            log_buffer_.push_back(entry);
        } else {
            log_buffer_[log_write_index_] = entry;
            log_write_index_ = (log_write_index_ + 1) % log_max_entries_;
        }
    }
    log_sequence_++;
}

std::vector<AgentLogEntry> AgentNode::GetRecentLogs(size_t count) const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::vector<AgentLogEntry> result;
    size_t total = log_buffer_.size();
    size_t start = (total <= count) ? 0 : total - count;
    for (size_t i = start; i < total; i++) {
        size_t idx = (log_write_index_ + i) % total;
        result.push_back(log_buffer_[idx]);
    }
    return result;
}

void AgentNode::FlushLogs() {
    if (config_.log_file.empty()) return;

    std::vector<AgentLogEntry> to_write;
    {
        std::lock_guard<std::mutex> lock(log_mutex_);
        to_write = log_buffer_;
    }

    std::ofstream f(config_.log_file, std::ios::app);
    if (!f) return;

    for (const auto& e : to_write) {
        auto t = std::chrono::system_clock::to_time_t(e.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            e.timestamp.time_since_epoch()) % 1000;

        f << "[" << std::put_time(std::gmtime(&t), "%Y-%m-%d %H:%M:%S");
        f << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";
        f << " [" << e.level << "]";
        f << " [" << e.from_agent;
        if (!e.to_agent.empty()) f << " -> " << e.to_agent;
        f << "] " << e.message << "\n";
    }
    f.close();

    // Rotate if file too large
    RotateLogFile();
}

void AgentNode::RotateLogFile() {
    std::ifstream check(config_.log_file, std::ios::binary | std::ios::ate);
    if (!check) return;
    auto size = check.tellg();
    check.close();

    size_t max_bytes = config_.log_max_size_mb * 1024 * 1024;
    if (size < 0 || static_cast<size_t>(size) < max_bytes) return;

    // Rotate: move current to .1, .1 to .2, etc. (keep 3 files)
    std::string base = config_.log_file;
    std::remove((base + ".2").c_str());
    std::rename((base + ".1").c_str(), (base + ".2").c_str());
    std::rename(base.c_str(), (base + ".1").c_str());
}

std::string AgentNode::GetLogPath(const std::string& suffix) {
    if (config_.log_file.empty()) return "";
    size_t dot = config_.log_file.find_last_of('.');
    if (dot == std::string::npos) return config_.log_file + suffix;
    return config_.log_file.substr(0, dot) + suffix + config_.log_file.substr(dot);
}

// ---------------------------------------------------------------------------
// Background threads
// ---------------------------------------------------------------------------

void AgentNode::HeartbeatLoop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        if (!running_.load()) break;
        RefreshPeers();
    }
}

void AgentNode::LogFlushLoop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (!running_.load()) break;
        FlushLogs();
    }
}

} // namespace notllama
