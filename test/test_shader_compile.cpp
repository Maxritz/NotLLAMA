#include "engine/shader_compiler.hpp"
#include <cstdio>
#include <filesystem>
#include <string>

using namespace notllama;

static std::string FindShaderDir() {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
        fs::current_path() / "shaders",
        fs::current_path() / ".." / "shaders",
        fs::current_path() / ".." / ".." / "shaders",
    };
    for (const auto& c : candidates) {
        fs::path canon = fs::weakly_canonical(c);
        if (fs::exists(canon / "test_add.comp")) {
            return canon.string();
        }
    }
    return "shaders";
}

int main() {
    printf("Shader compiler standalone test\n");

    ShaderCompiler compiler;
    if (!compiler.IsAvailable()) {
        printf("glslc not found\n");
        return 1;
    }
    printf("glslc available\n");

    std::string shader_dir = FindShaderDir();
    printf("Shader dir: %s\n", shader_dir.c_str());

    // Try to compile test_add.comp
    ShaderCompileOptions opts{};
    opts.src_path = shader_dir + "/test_add.comp";
    opts.cache_dir = shader_dir + "/cache";
    opts.defines.push_back("SUBGROUP_SIZE=32");

    std::vector<uint32_t> spv;
    std::string log;
    bool ok = compiler.Compile(opts, spv, &log);
    if (ok) {
        printf("test_add.comp compiled: %zu words\n", spv.size());
    } else {
        printf("FAILED: %s\n", log.c_str());
        return 1;
    }

    // Try rms_norm.comp
    opts.src_path = shader_dir + "/rms_norm.comp";
    ok = compiler.Compile(opts, spv, &log);
    if (ok) {
        printf("rms_norm.comp compiled: %zu words\n", spv.size());
    } else {
        printf("rms_norm.comp FAILED: %s\n", log.c_str());
    }

    // Try gemm.comp
    opts.src_path = shader_dir + "/gemm.comp";
    ok = compiler.Compile(opts, spv, &log);
    if (ok) {
        printf("gemm.comp compiled: %zu words\n", spv.size());
    } else {
        printf("gemm.comp FAILED: %s\n", log.c_str());
    }

    // Try dequantize.comp
    opts.src_path = shader_dir + "/dequantize.comp";
    ok = compiler.Compile(opts, spv, &log);
    if (ok) {
        printf("dequantize.comp compiled: %zu words\n", spv.size());
    } else {
        printf("dequantize.comp FAILED: %s\n", log.c_str());
    }

    // Compile all .comp files in shader directory
    namespace fs = std::filesystem;
    size_t attempted = 0, succeeded = 0, failed = 0;
    std::vector<std::string> failures;
    for (const auto& entry : fs::directory_iterator(shader_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".comp") continue;

        std::string name = entry.path().stem().string();
        opts.src_path = entry.path().string();
        opts.force_recompile = true;
        ++attempted;
        if (compiler.Compile(opts, spv, &log)) {
            ++succeeded;
            printf("  OK: %s (%zu words)\n", name.c_str(), spv.size());
        } else {
            ++failed;
            failures.push_back(name + ": " + log);
            printf("  FAIL: %s\n", name.c_str());
        }
    }

    printf("\nResult: %zu/%zu compiled, %zu failed\n", succeeded, attempted, failed);
    if (!failures.empty()) {
        printf("\nFailures:\n");
        for (const auto& f : failures) printf("  %s\n", f.c_str());
        return 1;
    }
    return 0;
}
