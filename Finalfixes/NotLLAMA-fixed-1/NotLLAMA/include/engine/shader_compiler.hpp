#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace notllama {

// Options for a single GLSL -> SPIR-V compile via the system glslc compiler.
struct ShaderCompileOptions {
    std::string src_path;        // Path to .comp source file
    std::string entry_point = "main";
    std::string target_env = "vulkan1.2";
    std::vector<std::string> defines;  // e.g. "WAVE_SIZE=64"
    std::string cache_dir = "shaders/cache";
    bool force_recompile = false;
};

// Runtime GLSL compiler wrapper. Finds the Vulkan SDK glslc, compiles source
// on demand, and caches SPIR-V output. This lets shaders adapt to the user's
// GPU (wave size, vendor, available extensions) instead of shipping fixed .spv.
class ShaderCompiler {
public:
    ShaderCompiler();

    bool IsAvailable() const;

    // Compile a compute shader. Returns SPIR-V words in out_spv.
    // On failure, out_log (if non-null) receives the compiler stderr.
    bool Compile(const ShaderCompileOptions& options,
                 std::vector<uint32_t>& out_spv,
                 std::string* out_log = nullptr);

    static std::string BuildCacheKey(const ShaderCompileOptions& options);
    static std::string GetCachePath(const std::string& cache_dir,
                                    const std::string& key);

private:
    std::string glslc_path_;
    bool available_ = false;

    bool FindGlslc();
    bool ReadSpvFile(const std::string& path, std::vector<uint32_t>& out_data);
    bool RunGlslc(const ShaderCompileOptions& options,
                  const std::string& out_path,
                  std::string* out_log);
};

} // namespace notllama
