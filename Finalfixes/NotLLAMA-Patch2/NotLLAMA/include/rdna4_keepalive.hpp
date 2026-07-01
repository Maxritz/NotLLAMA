#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <future>

namespace rdna4 {

// ============================================================================
// Event types that can trigger actions
// ============================================================================
enum class EventType : uint32_t {
    TIMER       = 0,  // Periodic interval
    HTTP        = 1,  // HTTP webhook / endpoint hit
    FILE_CHANGE = 2,  // File system watcher
    API_POLL    = 3,  // Poll external API for changes
    PROCESS     = 4,  // External process stdout / exit
    GPU_SIGNAL  = 5,  // Inference completion signal
    MANUAL      = 6,  // User-triggered
    CHAIN       = 7,  // Triggered by another action completion
    CONDITION   = 8,  // Conditional expression becomes true
};

// ============================================================================
// Trigger condition descriptor
// ============================================================================
struct TriggerCondition {
    EventType type = EventType::TIMER;
    std::string name;              // human-readable trigger name
    std::string target;            // URL, file path, API endpoint, etc.
    uint32_t intervalMs = 5000;    // for TIMER / API_POLL
    std::string method = "GET";    // for HTTP / API_POLL
    std::string headers;           // JSON string of HTTP headers
    std::string bodyFilter;        // regex or JSONPath filter for response
    bool enabled = true;
    uint32_t maxRetries = 3;
    uint32_t cooldownMs = 1000;    // debounce between triggers
};

// ============================================================================
// Action types
// ============================================================================
enum class ActionType : uint32_t {
    INFERENCE   = 0,  // Run model inference
    HTTP_POST   = 1,  // Send HTTP POST
    HTTP_GET    = 2,  // Send HTTP GET
    EXECUTE     = 3,  // Execute shell command / script
    FILE_WRITE  = 4,  // Write to file
    FILE_READ   = 5,  // Read from file
    NOTIFY      = 6,  // Send notification (toast, webhook, etc.)
    SCRIPT      = 7,  // Run embedded script
    CHAIN       = 8,  // Chain to another work template
    GPU_COMPUTE = 9,  // Dispatch custom GPU compute
};

// ============================================================================
// Action descriptor
// ============================================================================
struct Action {
    ActionType type = ActionType::INFERENCE;
    std::string name;
    std::string target;            // URL, file path, model ID, etc.
    std::string payload;           // body, prompt template, script source
    std::unordered_map<std::string, std::string> params;
    uint32_t timeoutMs = 30000;
    bool async = false;            // run in background vs block
};

// ============================================================================
// Work Template — a reusable "do until" or "on event" workflow
// ============================================================================
struct WorkTemplate {
    std::string id;
    std::string name;
    std::string description;

    // Triggers that start this workflow
    std::vector<TriggerCondition> triggers;

    // Actions to execute (in order, unless branching)
    std::vector<Action> actions;

    // Control flow
    bool loop = false;             // "do until" — repeat actions until stop condition
    std::string untilCondition;    // expression that ends loop (e.g. "result.done == true")
    uint32_t maxIterations = 1000; // safety cap for loops
    bool onErrorContinue = false;  // continue to next action on failure
    std::string onErrorAction;     // action ID to run on error

    // Scheduling
    bool runAtStartup = false;
    std::chrono::system_clock::time_point scheduledTime;
};

// ============================================================================
// Script engine — executes user-defined logic
// ============================================================================
class ScriptEngine {
public:
    ScriptEngine();
    ~ScriptEngine();

    // Load a script from string (Lua-like syntax or simple expression)
    bool loadScript(const std::string& id, const std::string& source);

    // Execute script with input variables, returns output map
    std::unordered_map<std::string, std::string>
    execute(const std::string& id,
            const std::unordered_map<std::string, std::string>& inputs);

    // Evaluate a boolean expression (for triggers / until conditions)
    bool evaluateCondition(const std::string& expr,
                           const std::unordered_map<std::string, std::string>& vars);

    // Built-in functions
    void registerBuiltin(const std::string& name,
                         std::function<std::string(const std::vector<std::string>&)> fn);

    // Template substitution: "Hello {{name}}" + {"name":"World"} -> "Hello World"
    static std::string substitute(const std::string& templ,
                                  const std::unordered_map<std::string, std::string>& vars);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Keep-Alive engine — background event loop + workflow executor
// ============================================================================
class KeepAliveEngine {
public:
    struct Config {
        uint32_t pollIntervalMs = 1000;
        uint32_t workerThreads = 2;
        bool     enableHttpServer = true;
        uint16_t httpPort = 8081;    // separate from main server
        bool     enableFileWatcher = true;
        bool     logAllEvents = true;
    };

    explicit KeepAliveEngine(const Config& cfg);
    ~KeepAliveEngine();

    // Start background polling threads
    bool start();
    void stop();

    // Register a work template
    bool registerTemplate(const WorkTemplate& tmpl);
    void unregisterTemplate(const std::string& id);

    // Trigger a template manually
    bool triggerTemplate(const std::string& id,
                         const std::unordered_map<std::string, std::string>& vars);

    // Add a one-shot trigger (e.g. "poll this URL every 5s until condition met")
    uint64_t addOneShot(const TriggerCondition& trigger,
                        const std::vector<Action>& actions,
                        const std::string& stopCondition);
    void cancelOneShot(uint64_t handle);

    // Set the inference engine callback (for INFERENCE actions)
    using InferenceCallback = std::function<std::string(const std::string& prompt,
                                                        uint32_t maxTokens,
                                                        float temperature)>;
    void setInferenceCallback(InferenceCallback cb);

    // Set the API client (for HTTP actions)
    class ApiClient;
    void setApiClient(std::shared_ptr<ApiClient> client);

    // Stats
    uint64_t getEventCount() const;
    uint64_t getActionCount() const;
    std::vector<std::string> listActiveTemplates() const;

    // External API access: POST work to this engine
    bool postWork(const std::string& jsonPayload);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// HTTP / API client for external services
// ============================================================================
class KeepAliveEngine::ApiClient {
public:
    virtual ~ApiClient() = default;

    virtual std::string httpGet(const std::string& url,
                                const std::unordered_map<std::string, std::string>& headers,
                                uint32_t timeoutMs) = 0;

    virtual std::string httpPost(const std::string& url,
                                 const std::unordered_map<std::string, std::string>& headers,
                                 const std::string& body,
                                 uint32_t timeoutMs) = 0;

    virtual bool webSocketConnect(const std::string& url) = 0;
    virtual bool webSocketSend(const std::string& msg) = 0;
    virtual void setWebSocketCallback(std::function<void(const std::string&)> cb) = 0;
};

} // namespace rdna4
