#include "web_server.hpp"

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    #pragma comment(lib, "ws2_32.lib")
    #define CLOSE_SOCKET closesocket
    #define MSG_NOSIGNAL 0
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <poll.h>
    #define CLOSE_SOCKET close
#endif

#include <cstring>
#include <cstdio>
#include <sstream>
#include <chrono>
#include <thread>

// Cross-platform: ssize_t, errno, recv/send
#ifdef _WIN32
    typedef int ssize_t_win;
    typedef SSIZE_T ssize_t;
    #define SOCK_ERRNO WSAGetLastError()
    #define SOCK_EAGAIN WSAEWOULDBLOCK
    #define SOCK_EWOULDBLOCK WSAEWOULDBLOCK
    #define SOCK_SEND_FLAGS 0
#else
    typedef ssize_t ssize_t;
    #define SOCK_ERRNO errno
    #define SOCK_EAGAIN EAGAIN
    #define SOCK_EWOULDBLOCK EWOULDBLOCK
    #define SOCK_SEND_FLAGS MSG_NOSIGNAL
#endif

// Cross-platform helpers for poll/non-blocking I/O
#ifdef _WIN32
    typedef ULONG nfds_t;
    #define SET_NONBLOCKING(fd) do { u_long mode = 1; ioctlsocket((fd), FIONBIO, &mode); } while(0)
    #define PLATFORM_POLL(fds, nfds, timeout) WSAPoll((fds), (nfds), (timeout))
    #define PLATFORM_POLLIN 0x0100
    #define PLATFORM_POLLERR 0x0001
    #define PLATFORM_POLLHUP 0x0002
    #define ERRNO_EINTR (WSAGetLastError() == WSAEINTR)
#else
    #define SET_NONBLOCKING(fd) do { int fl = fcntl((fd), F_GETFL, 0); fcntl((fd), F_SETFL, fl | O_NONBLOCK); } while(0)
    #define PLATFORM_POLL(fds, nfds, timeout) poll((fds), (nfds), (timeout))
    #define PLATFORM_POLLIN POLLIN
    #define PLATFORM_POLLERR POLLERR
    #define PLATFORM_POLLHUP POLLHUP
    #define ERRNO_EINTR (errno == EINTR)
#endif
#include <iomanip>
#include <random>

namespace notllama {

// ---- HTTP helpers ----

static std::string http_status(int code) {
    switch (code) {
        case 200: return "200 OK";
        case 201: return "201 Created";
        case 400: return "400 Bad Request";
        case 401: return "401 Unauthorized";
        case 404: return "404 Not Found";
        case 405: return "405 Method Not Allowed";
        case 429: return "429 Too Many Requests";
        case 500: return "500 Internal Server Error";
        case 503: return "503 Service Unavailable";
        default:  return "500 Internal Server Error";
    }
}

static std::string make_http_response(int code, const std::string& content_type,
                                       const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << http_status(code) << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: keep-alive\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        << "\r\n"
        << body;
    return oss.str();
}

static std::string make_sse_chunk(const std::string& data) {
    return "data: " + data + "\n\n";
}

// ---- WebServer ----

WebServer::WebServer(MultiModelManager* model_mgr,
                     ModelRouter* router,
                     MTPEngine* mtp,
                     const std::string& host,
                     int32_t port,
                     int32_t timeout_seconds)
    : model_mgr_(model_mgr), router_(router), mtp_(mtp),
      host_(host), port_(port), timeout_seconds_(timeout_seconds) {}

WebServer::~WebServer() { Stop(); }

bool WebServer::Start() {
    if (running_.load()) return true;

    server_fd_ = MakeSocket();
    if (server_fd_ < 0) {
        fprintf(stderr, "[WebServer] Failed to create socket\n");
        return false;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "[WebServer] bind failed on %s:%d\n", host_.c_str(), port_);
        CLOSE_SOCKET(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 128) < 0) {
        fprintf(stderr, "[WebServer] listen failed\n");
        CLOSE_SOCKET(server_fd_);
        server_fd_ = -1;
        return false;
    }

    // Set server socket non-blocking
    SET_NONBLOCKING(server_fd_);

    running_ = true;
    stop_flag_ = false;
    server_thread_ = std::thread(&WebServer::Run, this);

    fprintf(stderr, "[WebServer] Listening on http://%s:%d\n", host_.c_str(), port_);
    return true;
}

void WebServer::Stop() {
    stop_flag_ = true;
    running_ = false;
    if (server_fd_ >= 0) {
        CLOSE_SOCKET(server_fd_);
        server_fd_ = -1;
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    fprintf(stderr, "[WebServer] Stopped\n");
}

int WebServer::MakeSocket() {
    return socket(AF_INET, SOCK_STREAM, 0);
}

void WebServer::Run() {
    const int MAX_CLIENTS = 64;
    std::vector<struct pollfd> fds;
    fds.reserve(MAX_CLIENTS + 1);

    fds.push_back({server_fd_, PLATFORM_POLLIN, 0});

    while (!stop_flag_.load()) {
        int ret = PLATFORM_POLL(fds.data(), static_cast<nfds_t>(fds.size()), 100); // 100ms timeout
        if (ret < 0) {
            if (ERRNO_EINTR) continue;
            break;
        }

        // Check server socket for new connections
        if (fds[0].revents & PLATFORM_POLLIN) {
            sockaddr_in client_addr{};
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            int client_fd = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
            if (client_fd >= 0) {
                // Set client socket non-blocking
                SET_NONBLOCKING(client_fd);
                fds.push_back({client_fd, PLATFORM_POLLIN, 0});
            }
        }

        // Handle client sockets
        for (size_t i = 1; i < fds.size(); ) {
            if (fds[i].revents & (PLATFORM_POLLIN | PLATFORM_POLLERR | PLATFORM_POLLHUP)) {
                if (fds[i].revents & PLATFORM_POLLIN) {
                    HandleRequest(fds[i].fd);
                }
                CLOSE_SOCKET(fds[i].fd);
                fds.erase(fds.begin() + i);
            } else {
                i++;
            }
        }

        // Limit max connections
        while (fds.size() > static_cast<size_t>(MAX_CLIENTS + 1)) {
            CLOSE_SOCKET(fds.back().fd);
            fds.pop_back();
        }
    }

    // Clean up remaining client sockets
    for (size_t i = 1; i < fds.size(); i++) {
        CLOSE_SOCKET(fds[i].fd);
    }
}

void WebServer::HandleRequest(int client_fd) {
    char buffer[65536];
    ssize_t total = 0;

    // Read request with timeout
    auto start = std::chrono::steady_clock::now();
    while (total < static_cast<ssize_t>(sizeof(buffer)) - 1) {
        ssize_t n = recv(client_fd, buffer + total, sizeof(buffer) - 1 - static_cast<size_t>(total), 0);
        if (n > 0) {
            total += n;
            buffer[total] = '\0';
            // Check if we have the full HTTP header
            if (strstr(buffer, "\r\n\r\n") != nullptr) break;
        } else if (n == 0) {
            break; // Connection closed
        } else {
            int err = SOCK_ERRNO;
            if (err == SOCK_EAGAIN || err == SOCK_EWOULDBLOCK) {
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (elapsed > std::chrono::seconds(timeout_seconds_)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            break;
        }
    }

    if (total == 0) return;
    buffer[total] = '\0';
    std::string request(buffer, static_cast<size_t>(total));

    std::string method = ParseMethod(request);
    std::string path = ParsePath(request);
    std::string body = ParseBody(request);
    std::string auth = ParseHeader(request, "Authorization");

    std::string response = RouteRequest(method, path, body, auth);

    send(client_fd, response.c_str(), static_cast<int>(response.size()), SOCK_SEND_FLAGS);
}

std::string WebServer::ParseMethod(const std::string& request) {
    size_t space = request.find(' ');
    if (space == std::string::npos) return "GET";
    return request.substr(0, space);
}

std::string WebServer::ParsePath(const std::string& request) {
    size_t first = request.find(' ');
    if (first == std::string::npos) return "/";
    size_t second = request.find(' ', first + 1);
    if (second == std::string::npos) return request.substr(first + 1);
    return request.substr(first + 1, second - first - 1);
}

std::string WebServer::ParseBody(const std::string& request) {
    size_t pos = request.find("\r\n\r\n");
    if (pos == std::string::npos) return "";
    return request.substr(pos + 4);
}

std::string WebServer::ParseHeader(const std::string& request, const std::string& header) {
    std::string search = header + ": ";
    size_t pos = request.find(search);
    if (pos == std::string::npos) return "";
    size_t start = pos + search.length();
    size_t end = request.find("\r\n", start);
    if (end == std::string::npos) return request.substr(start);
    return request.substr(start, end - start);
}

bool WebServer::CheckAuth(const std::string& auth_header) {
    if (api_key_.empty()) return true; // No auth required
    std::string expected = "Bearer " + api_key_;
    return auth_header == expected;
}

std::string WebServer::RouteRequest(const std::string& method,
                                     const std::string& path,
                                     const std::string& body,
                                     const std::string& auth_header) {
    // OPTIONS (CORS preflight)
    if (method == "OPTIONS") {
        return make_http_response(204, "text/plain", "");
    }

    // Auth check
    if (!CheckAuth(auth_header)) {
        return make_http_response(401, "application/json",
            nlohmann::json{{"error", {{"message", "Invalid API key"},
                                       {"type", "authentication_error"}}}}.dump());
    }

    try {
        // Health check
        if (method == "GET" && path == "/health") {
            return HandleHealth();
        }
        // Properties
        if (method == "GET" && (path == "/props" || path == "/v1/props")) {
            return HandleProps();
        }
        // List models
        if (method == "GET" && (path == "/v1/models" || path == "/models")) {
            return HandleListModels();
        }
        // Completions
        if (method == "POST" && (path == "/v1/completions" || path == "/completions")) {
            return HandleCompletions(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Chat completions
        if (method == "POST" && (path == "/v1/chat/completions" || path == "/chat/completions")) {
            return HandleChatCompletions(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Embeddings
        if (method == "POST" && enable_embeddings_ &&
            (path == "/v1/embeddings" || path == "/embeddings")) {
            return HandleEmbeddings(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Reranking
        if (method == "POST" && enable_reranking_ &&
            (path == "/v1/rerank" || path == "/rerank")) {
            return HandleRerank(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Tokenize
        if (method == "POST" && (path == "/tokenize" || path == "/v1/tokenize")) {
            return HandleTokenize(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Detokenize
        if (method == "POST" && (path == "/detokenize" || path == "/v1/detokenize")) {
            return HandleDetokenize(nlohmann::json::parse(body.empty() ? "{}" : body));
        }
        // Agent endpoints (distributed agent protocol)
        if (path.starts_with("/agent/")) {
            if (agent_handler_) {
                std::string subpath = path.substr(1); // remove leading /
                return make_http_response(200, "application/json",
                    agent_handler_(subpath, nlohmann::json::parse(body.empty() ? "{}" : body)));
            }
            return make_http_response(503, "application/json",
                nlohmann::json{{"error", "Agent node not initialized"}}.dump());
        }
    } catch (const nlohmann::json::exception& e) {
        return MakeErrorResponse(400, std::string("JSON parse error: ") + e.what(),
                                  "invalid_request_error");
    } catch (const std::exception& e) {
        return MakeErrorResponse(500, e.what(), "internal_error");
    }

    return make_http_response(404, "application/json",
        nlohmann::json{{"error", "Endpoint not found: " + path}}.dump());
}

std::string WebServer::HandleHealth() {
    auto models = model_mgr_->ListModels();
    nlohmann::json j;
    j["status"] = "ok";
    j["models_loaded"] = models.size();
    j["vram_total_mb"] = model_mgr_->GetTotalVRAM() / (1024 * 1024);
    j["vram_used_mb"] = model_mgr_->GetUsedVRAM() / (1024 * 1024);
    j["vram_available_mb"] = model_mgr_->GetAvailableVRAM() / (1024 * 1024);
    return make_http_response(200, "application/json", j.dump(2));
}

std::string WebServer::HandleProps() {
    nlohmann::json j;
    j["system_prompt"] = "";
    j["default_generation_settings"] = {
        {"n_ctx", 4096},
        {"n_predict", -1},
        {"model", model_mgr_->ListModels().empty() ? "" : model_mgr_->ListModels()[0]},
        {"seed", -1},
        {"temperature", 0.8},
        {"top_p", 0.95},
        {"top_k", 40},
        {"repeat_penalty", 1.1}
    };
    j["total_slots"] = 1;
    return make_http_response(200, "application/json", j.dump(2));
}

std::string WebServer::HandleListModels() {
    nlohmann::json j;
    j["object"] = "list";
    auto models = model_mgr_->ListModels();
    for (const auto& id : models) {
        auto* inst = model_mgr_->GetModel(id);
        if (!inst) continue;
        nlohmann::json m;
        m["id"] = id;
        m["object"] = "model";
        m["owned_by"] = "notllama";
        m["permission"] = nlohmann::json::array();
        m["tags"] = inst->tags;
        m["context_window"] = inst->max_context;
        j["data"].push_back(m);
    }
    return make_http_response(200, "application/json", j.dump(2));
}

std::string WebServer::HandleCompletions(const nlohmann::json& req) {
    CompletionRequest creq;
    creq.model = req.value("model", "");
    creq.prompt = req.value("prompt", "");
    creq.max_tokens = req.value("max_tokens", -1);
    creq.n_predict = req.value("n_predict", creq.max_tokens);
    creq.temperature = req.value("temperature", 0.8f);
    creq.top_p = req.value("top_p", 0.95f);
    creq.top_k = req.value("top_k", 40);
    creq.frequency_penalty = req.value("frequency_penalty", 0.0f);
    creq.presence_penalty = req.value("presence_penalty", 0.0f);
    creq.repeat_penalty = req.value("repeat_penalty", 1.1f);
    creq.seed = req.value("seed", -1);
    creq.stream = req.value("stream", false);
    if (req.contains("stop")) {
        if (req["stop"].is_string()) creq.stop.push_back(req["stop"]);
        else if (req["stop"].is_array()) {
            for (const auto& s : req["stop"]) creq.stop.push_back(s);
        }
    }

    if (creq.prompt.empty()) {
        return MakeErrorResponse(400, "Missing 'prompt' field", "invalid_request_error");
    }

    // Route to appropriate model
    auto decision = router_->Route(creq.prompt);
    std::string model_id = creq.model.empty() ? decision.primary_model_id : creq.model;
    if (model_id.empty()) {
        return MakeErrorResponse(400, "No model specified and no default available",
                                  "invalid_request_error");
    }

    auto* inst = model_mgr_->GetModel(model_id);
    if (!inst) {
        return MakeErrorResponse(404, "Model not found: " + model_id, "not_found");
    }

    auto response = DoCompletion(creq);

    nlohmann::json j;
    j["id"] = response.id;
    j["object"] = "text_completion";
    j["created"] = response.created;
    j["model"] = response.model;
    for (const auto& c : response.choices) {
        nlohmann::json choice;
        choice["index"] = c.index;
        choice["text"] = c.text;
        choice["finish_reason"] = c.finish_reason.empty() ? nullptr : c.finish_reason;
        j["choices"].push_back(choice);
    }
    j["usage"] = {
        {"prompt_tokens", response.usage.prompt_tokens},
        {"completion_tokens", response.usage.completion_tokens},
        {"total_tokens", response.usage.total_tokens}
    };

    return make_http_response(200, "application/json", j.dump());
}

std::string WebServer::HandleChatCompletions(const nlohmann::json& req) {
    CompletionRequest creq;
    creq.model = req.value("model", "");
    creq.max_tokens = req.value("max_tokens", -1);
    creq.n_predict = req.value("n_predict", creq.max_tokens);
    creq.temperature = req.value("temperature", 0.8f);
    creq.top_p = req.value("top_p", 0.95f);
    creq.top_k = req.value("top_k", 40);
    creq.seed = req.value("seed", -1);
    creq.stream = req.value("stream", false);

    // Convert messages to prompt
    std::string prompt;
    if (req.contains("messages") && req["messages"].is_array()) {
        for (const auto& msg : req["messages"]) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            if (role == "system") prompt += "System: " + content + "\n";
            else if (role == "user") prompt += "User: " + content + "\n";
            else if (role == "assistant") prompt += "Assistant: " + content + "\n";
            else prompt += role + ": " + content + "\n";
        }
        prompt += "Assistant: ";
    } else {
        prompt = req.value("prompt", "");
    }
    creq.prompt = prompt;

    if (creq.prompt.empty()) {
        return MakeErrorResponse(400, "Missing 'messages' or 'prompt' field", "invalid_request_error");
    }

    // Route and generate
    auto decision = router_->Route(creq.prompt);
    std::string model_id = creq.model.empty() ? decision.primary_model_id : creq.model;
    if (model_id.empty()) {
        auto models = model_mgr_->ListModels();
        if (!models.empty()) model_id = models[0];
    }

    auto* inst = model_mgr_->GetModel(model_id);
    if (!inst) {
        return MakeErrorResponse(404, "Model not found: " + model_id, "not_found");
    }

    auto response = DoCompletion(creq);

    nlohmann::json j;
    j["id"] = response.id;
    j["object"] = "chat.completion";
    j["created"] = response.created;
    j["model"] = response.model;
    for (const auto& c : response.choices) {
        nlohmann::json choice;
        choice["index"] = c.index;
        choice["message"] = {{"role", "assistant"}, {"content", c.text}};
        choice["finish_reason"] = c.finish_reason.empty() ? nullptr : c.finish_reason;
        j["choices"].push_back(choice);
    }
    j["usage"] = {
        {"prompt_tokens", response.usage.prompt_tokens},
        {"completion_tokens", response.usage.completion_tokens},
        {"total_tokens", response.usage.total_tokens}
    };

    return make_http_response(200, "application/json", j.dump());
}

std::string WebServer::HandleEmbeddings(const nlohmann::json& req) {
    std::string input = req.value("input", "");
    std::string model_id = req.value("model", "");
    if (input.empty()) {
        return MakeErrorResponse(400, "Missing 'input' field", "invalid_request_error");
    }
    auto models = model_mgr_->ListModels();
    if (model_id.empty() && !models.empty()) model_id = models[0];

    // Placeholder: return zero embedding
    // Real implementation would run the model's embedding layer
    nlohmann::json j;
    j["object"] = "list";
    nlohmann::json data;
    data["object"] = "embedding";
    data["index"] = 0;
    data["embedding"] = std::vector<float>(768, 0.0f); // 768-dim placeholder
    j["data"] = nlohmann::json::array({data});
    j["model"] = model_id;
    j["usage"] = {{"prompt_tokens", 0}, {"total_tokens", 0}};

    return make_http_response(200, "application/json", j.dump());
}

std::string WebServer::HandleRerank(const nlohmann::json& req) {
    std::string query = req.value("query", "");
    auto documents = req.value("documents", std::vector<std::string>{});
    if (query.empty() || documents.empty()) {
        return MakeErrorResponse(400, "Missing 'query' or 'documents'", "invalid_request_error");
    }

    // Placeholder: return documents in original order with score 1.0
    nlohmann::json j;
    j["results"] = nlohmann::json::array();
    for (size_t i = 0; i < documents.size(); i++) {
        j["results"].push_back({
            {"index", static_cast<int>(i)},
            {"relevance_score", 1.0},
            {"document", documents[i]}
        });
    }
    return make_http_response(200, "application/json", j.dump());
}

std::string WebServer::HandleTokenize(const nlohmann::json& req) {
    std::string text = req.value("text", "");
    std::string model_id = req.value("model", "");
    auto models = model_mgr_->ListModels();
    if (model_id.empty() && !models.empty()) model_id = models[0];

    auto* inst = model_mgr_->GetModel(model_id);
    nlohmann::json j;
    j["tokens"] = nlohmann::json::array();

    // Simple whitespace tokenization fallback
    if (!inst || inst->adapter->GetTokenizer().idToToken.empty()) {
        size_t start = 0;
        while (start < text.size()) {
            size_t end = text.find(' ', start);
            if (end == std::string::npos) end = text.size();
            if (end > start) {
                // Return token IDs as byte values for now
                for (size_t i = start; i < end; i++) {
                    j["tokens"].push_back(static_cast<uint8_t>(text[i]));
                }
            }
            start = end + 1;
        }
    } else {
        // Use model tokenizer
        const auto& tok = inst->adapter->GetTokenizer();
        // Simple character-by-character tokenization
        for (char c : text) {
            auto it = tok.vocab.find(std::string(1, c));
            if (it != tok.vocab.end()) {
                j["tokens"].push_back(it->second);
            } else {
                j["tokens"].push_back(static_cast<uint8_t>(c));
            }
        }
    }

    return make_http_response(200, "application/json", j.dump());
}

std::string WebServer::HandleDetokenize(const nlohmann::json& req) {
    auto tokens = req.value("tokens", std::vector<uint32_t>{});
    std::string model_id = req.value("model", "");
    auto models = model_mgr_->ListModels();
    if (model_id.empty() && !models.empty()) model_id = models[0];

    auto* inst = model_mgr_->GetModel(model_id);
    std::string text;

    if (!inst || inst->adapter->GetTokenizer().idToToken.empty()) {
        // Fallback: interpret as ASCII
        for (uint32_t t : tokens) {
            if (t < 256) text += static_cast<char>(t);
        }
    } else {
        const auto& tok = inst->adapter->GetTokenizer();
        for (uint32_t t : tokens) {
            if (t < tok.idToToken.size()) {
                text += tok.idToToken[t];
            }
        }
    }

    nlohmann::json j;
    j["text"] = text;
    return make_http_response(200, "application/json", j.dump());
}

CompletionResponse WebServer::DoCompletion(const CompletionRequest& req) {
    CompletionResponse resp;
    resp.id = "notllama-" + std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    resp.created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    resp.model = req.model;

    std::string model_id = req.model;
    if (model_id.empty()) {
        auto models = model_mgr_->ListModels();
        if (!models.empty()) model_id = models[0];
    }

    auto* inst = model_mgr_->GetModel(model_id);
    if (!inst) {
        CompletionResponse::Choice c;
        c.text = "[Error: Model not available]";
        c.finish_reason = "error";
        resp.choices.push_back(c);
        return resp;
    }

    // Tokenize the prompt (simple char-level for now)
    std::vector<uint32_t> prompt_tokens;
    for (char c : req.prompt) {
        prompt_tokens.push_back(static_cast<uint8_t>(c));
    }

    // Set MTP config
    MTPConfig mtp_cfg;
    mtp_cfg.n_predict = req.max_tokens > 0 ? req.max_tokens : 128;
    mtp_cfg.temperature = req.temperature;
    mtp_->SetConfig(mtp_cfg);
    if (req.seed >= 0) mtp_->SetSeed(req.seed);

    // Generate
    uint32_t max_tokens = req.max_tokens > 0 ? static_cast<uint32_t>(req.max_tokens) : 128;
    auto generated = mtp_->Generate(model_id, prompt_tokens, max_tokens);

    // Convert tokens to text
    std::string text;
    const auto& tok = inst->adapter->GetTokenizer();
    if (!tok.idToToken.empty()) {
        for (uint32_t t : generated) {
            if (t < tok.idToToken.size()) text += tok.idToToken[t];
            else text += "?";
        }
    } else {
        for (uint32_t t : generated) {
            if (t < 256) text += static_cast<char>(t);
        }
    }

    CompletionResponse::Choice c;
    c.text = text;
    c.finish_reason = generated.size() < max_tokens ? "stop" : "length";
    resp.choices.push_back(c);

    resp.usage.prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
    resp.usage.completion_tokens = static_cast<int32_t>(generated.size());
    resp.usage.total_tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens;

    return resp;
}

std::string WebServer::MakeErrorResponse(int code, const std::string& message,
                                          const std::string& type) {
    nlohmann::json j;
    j["error"] = {{"message", message}, {"type", type}};
    return make_http_response(code, "application/json", j.dump());
}

} // namespace notllama
