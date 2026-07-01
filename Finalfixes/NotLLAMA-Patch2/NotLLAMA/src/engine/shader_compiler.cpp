#include "engine/shader_compiler.hpp"
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

namespace notllama {

namespace fs = std::filesystem;

namespace {

std::string QuotePath(const std::string& path) {
    return "\"" + path + "\"";
}

} // anonymous namespace

ShaderCompiler::ShaderCompiler() {
    available_ = FindGlslc();
}

bool ShaderCompiler::IsAvailable() const {
    return available_;
}

bool ShaderCompiler::FindGlslc() {
    const char* vulkan_sdk = std::getenv("VULKAN_SDK");
    if (vulkan_sdk) {
        std::vector<std::string> candidates = {
            std::string(vulkan_sdk) + "/Bin/glslc.exe",
            std::string(vulkan_sdk) + "/bin/glslc.exe",
            std::string(vulkan_sdk) + "/Bin/glslc",
            std::string(vulkan_sdk) + "/bin/glslc",
        };
        for (const auto& path : candidates) {
            if (fs::exists(path)) {
                glslc_path_ = path;
                return true;
            }
        }
    }

    // Fallback: try a plain command and hope it is on PATH.
    int rc = std::system("where glslc >nul 2>&1");
    if (rc == 0) {
        glslc_path_ = "glslc";
        return true;
    }
    return false;
}

std::string ShaderCompiler::BuildCacheKey(const ShaderCompileOptions& options) {
    std::string joined = options.src_path + ";" + options.target_env + ";" + options.entry_point;
    for (const auto& d : options.defines) {
        joined += ";" + d;
    }
    size_t h = std::hash<std::string>{}(joined);

    std::string base = fs::path(options.src_path).stem().string();
    std::ostringstream oss;
    oss << base << "_" << std::hex << h << ".spv";
    return oss.str();
}

std::string ShaderCompiler::GetCachePath(const std::string& cache_dir,
                                         const std::string& key) {
    return (fs::path(cache_dir) / key).string();
}

bool ShaderCompiler::ReadSpvFile(const std::string& path,
                                 std::vector<uint32_t>& out_data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamoff end = file.tellg();
    if (end < 0 || static_cast<size_t>(end) % sizeof(uint32_t) != 0) {
        return false;
    }
    size_t size = static_cast<size_t>(end);

    file.seekg(0, std::ios::beg);
    out_data.resize(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(out_data.data()), size);
    return file.good();
}

bool ShaderCompiler::RunGlslc(const ShaderCompileOptions& options,
                              const std::string& out_path,
                              std::string* out_log) {
    auto normalize = [](const std::string& p) -> std::string {
        std::string r = p;
        for (auto& c : r) if (c == '/') c = '\\';
        return r;
    };
    std::string glslc = normalize(glslc_path_);
    std::string src = normalize(options.src_path);
    std::string out = normalize(out_path);

    // Build command line: glslc -fshader-stage=compute --target-env=... -o out src
    std::string cmd = QuotePath(glslc);
    cmd += " -fshader-stage=compute";
    cmd += " --target-env=" + options.target_env;
    for (const auto& d : options.defines) cmd += " -D" + d;
    cmd += " -o " + QuotePath(out);
    cmd += " " + QuotePath(src);

#ifdef _WIN32
    // Create pipes for stdout/stderr
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE hStdoutRead = nullptr, hStdoutWrite = nullptr;
    HANDLE hStderrRead = nullptr, hStderrWrite = nullptr;
    CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0);
    CreatePipe(&hStderrRead, &hStderrWrite, &sa, 0);

    // Ensure write handles are not inherited
    SetHandleInformation(hStdoutWrite, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(hStderrWrite, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStderrWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.dwFlags = STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};
    // CreateProcessA needs a mutable command line buffer
    std::vector<char> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr,           // application name
        cmd_buf.data(),    // command line
        nullptr,           // process security
        nullptr,           // thread security
        TRUE,              // inherit handles
        CREATE_NO_WINDOW,  // creation flags
        nullptr,           // environment
        nullptr,           // current directory
        &si, &pi);

    // Close write ends in parent immediately
    CloseHandle(hStdoutWrite);
    CloseHandle(hStderrWrite);

    if (!ok) {
        CloseHandle(hStdoutRead);
        CloseHandle(hStderrRead);
        DWORD err = GetLastError();
        if (out_log) *out_log = "CreateProcessA failed, error=" + std::to_string(err);
        return false;
    }

    // Read stderr (glslc writes errors here)
    std::string stderr_output;
    {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(hStderrRead, buf, sizeof(buf) - 1, &n, nullptr) && n > 0) {
            buf[n] = '\0';
            stderr_output += buf;
        }
    }
    CloseHandle(hStderrRead);

    // Read stdout
    std::string stdout_output;
    {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(hStdoutRead, buf, sizeof(buf) - 1, &n, nullptr) && n > 0) {
            buf[n] = '\0';
            stdout_output += buf;
        }
    }
    CloseHandle(hStdoutRead);

    // Wait for process to finish
    WaitForSingleObject(pi.hProcess, 30000); // 30s timeout
    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    if (exit_code != 0) {
        if (out_log) {
            *out_log = "glslc returned exit code " + std::to_string(exit_code);
            if (!stderr_output.empty()) *out_log += "\n" + stderr_output;
            if (!stdout_output.empty()) *out_log += "\n" + stdout_output;
        }
        return false;
    }

    if (!fs::exists(out)) {
        if (out_log) {
            *out_log = "glslc exited 0 but output not found: " + out;
            if (!stderr_output.empty()) *out_log += "\n" + stderr_output;
        }
        return false;
    }

    return true;
#else
    // Non-Windows: use system()
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        if (out_log) *out_log = "glslc returned exit code " + std::to_string(rc);
        return false;
    }
    if (!fs::exists(out)) {
        if (out_log) *out_log = "glslc exited 0 but output not found: " + out;
        return false;
    }
    return true;
#endif
}

bool ShaderCompiler::Compile(const ShaderCompileOptions& options,
                             std::vector<uint32_t>& out_spv,
                             std::string* out_log) {
    if (!available_) {
        if (out_log) *out_log = "glslc not found; set VULKAN_SDK or add glslc to PATH";
        return false;
    }

    if (!fs::exists(options.src_path)) {
        if (out_log) *out_log = "shader source not found: " + options.src_path;
        return false;
    }

    std::string key = BuildCacheKey(options);
    std::string cache_path = GetCachePath(options.cache_dir, key);

    if (!options.force_recompile && fs::exists(cache_path)) {
        if (ReadSpvFile(cache_path, out_spv)) {
            return true;
        }
        // Cache read failed; fall through to recompile.
    }

    try {
        fs::create_directories(options.cache_dir);
    } catch (const fs::filesystem_error& e) {
        if (out_log) {
            *out_log = std::string("failed to create shader cache dir: ") + e.what();
        }
        return false;
    }

    if (!RunGlslc(options, cache_path, out_log)) {
        return false;
    }

    if (!ReadSpvFile(cache_path, out_spv)) {
        if (out_log) *out_log = "glslc succeeded but output SPIR-V could not be read";
        return false;
    }
    return true;
}

} // namespace notllama
