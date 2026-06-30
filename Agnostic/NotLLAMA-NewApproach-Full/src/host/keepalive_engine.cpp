#include "rdna4_keepalive.hpp"
#include <iostream>

namespace rdna4 {

// ============================================================================
// ScriptEngine
// ============================================================================

struct ScriptEngine::Impl {
    struct Script {
        std::string source;
    };
    std::unordered_map<std::string, Script> scripts;
    std::unordered_map<std::string, std::function<std::string(const std::vector<std::string>&)>> builtins;
};

ScriptEngine::ScriptEngine() : impl_(std::make_unique<Impl>()) {}
ScriptEngine::~ScriptEngine() = default;

bool ScriptEngine::loadScript(const std::string& id, const std::string& source) {
    Impl::Script s;
    s.source = source;
    impl_->scripts[id] = s;
    return true;
}

std::unordered_map<std::string, std::string>
ScriptEngine::execute(const std::string& id,
                      const std::unordered_map<std::string, std::string>& inputs) {
    // TODO: implement script evaluation
    (void)id;
    return inputs;
}

bool ScriptEngine::evaluateCondition(const std::string& expr,
                                     const std::unordered_map<std::string, std::string>& vars) {
    // TODO: implement expression evaluation
    (void)expr;
    (void)vars;
    return false;
}

void ScriptEngine::registerBuiltin(const std::string& name,
                                    std::function<std::string(const std::vector<std::string>&)> fn) {
    impl_->builtins[name] = std::move(fn);
}

std::string ScriptEngine::substitute(const std::string& templ,
                                      const std::unordered_map<std::string, std::string>& vars) {
    std::string result = templ;
    for (const auto& [key, val] : vars) {
        std::string pattern = "{{" + key + "}}";
        size_t pos = 0;
        while ((pos = result.find(pattern, pos)) != std::string::npos) {
            result.replace(pos, pattern.length(), val);
            pos += val.length();
        }
    }
    return result;
}

// ============================================================================
// KeepAliveEngine
// ============================================================================

struct KeepAliveEngine::Impl {
    Config config;
    std::unordered_map<std::string, WorkTemplate> templates;
    InferenceCallback inferCb;
    std::shared_ptr<ApiClient> apiClient;
    std::atomic<uint64_t> eventCount{0};
    std::atomic<uint64_t> actionCount{0};
    bool running = false;
};

KeepAliveEngine::KeepAliveEngine(const Config& cfg)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = cfg;
}

KeepAliveEngine::~KeepAliveEngine() {
    stop();
}

bool KeepAliveEngine::start() {
    if (impl_->running) return false;
    impl_->running = true;
    // TODO: start background polling threads
    return true;
}

void KeepAliveEngine::stop() {
    impl_->running = false;
    // TODO: stop background threads
}

bool KeepAliveEngine::registerTemplate(const WorkTemplate& tmpl) {
    impl_->templates[tmpl.id] = tmpl;
    return true;
}

void KeepAliveEngine::unregisterTemplate(const std::string& id) {
    impl_->templates.erase(id);
}

bool KeepAliveEngine::triggerTemplate(const std::string& id,
                                       const std::unordered_map<std::string, std::string>& vars) {
    // TODO: execute template actions
    (void)vars;
    auto it = impl_->templates.find(id);
    if (it == impl_->templates.end()) return false;
    impl_->actionCount++;
    return true;
}

uint64_t KeepAliveEngine::addOneShot(const TriggerCondition& trigger,
                                      const std::vector<Action>& actions,
                                      const std::string& stopCondition) {
    // TODO: schedule one-shot trigger
    (void)trigger;
    (void)actions;
    (void)stopCondition;
    return 0;
}

void KeepAliveEngine::cancelOneShot(uint64_t handle) {
    (void)handle;
}

void KeepAliveEngine::setInferenceCallback(InferenceCallback cb) {
    impl_->inferCb = std::move(cb);
}

void KeepAliveEngine::setApiClient(std::shared_ptr<ApiClient> client) {
    impl_->apiClient = std::move(client);
}

uint64_t KeepAliveEngine::getEventCount() const {
    return impl_->eventCount.load();
}

uint64_t KeepAliveEngine::getActionCount() const {
    return impl_->actionCount.load();
}

std::vector<std::string> KeepAliveEngine::listActiveTemplates() const {
    std::vector<std::string> ids;
    for (const auto& [id, tmpl] : impl_->templates)
        ids.push_back(id);
    return ids;
}

bool KeepAliveEngine::postWork(const std::string& jsonPayload) {
    // TODO: parse and execute work from JSON
    (void)jsonPayload;
    return false;
}

} // namespace rdna4
