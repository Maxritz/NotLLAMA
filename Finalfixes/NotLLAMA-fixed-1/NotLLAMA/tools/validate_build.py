#!/usr/bin/env python3
"""validate_build.py — Check project structure and tool chain."""

import sys
import os
import subprocess
from pathlib import Path

def check_file(path, desc):
    if Path(path).exists():
        print(f"  [OK] {desc}")
        return True
    else:
        print(f"  [MISSING] {desc}: {path}")
        return False

def check_tool(cmd, desc):
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        print(f"  [OK] {desc}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  [MISSING] {desc}: {cmd} not found")
        return False

def main():
    print("RDNA4 LLaMA Build Validation\n")
    print("=" * 40)

    all_ok = True

    print("\n--- Project Structure ---")
    files = [
        ("CMakeLists.txt", "Build system"),
        ("include/rdna4.hpp", "Core header"),
        ("include/rdna4_types.hpp", "Push constant types"),
        ("include/rdna4_vulkan.hpp", "Vulkan context"),
        ("include/rdna4_weights.hpp", "Weight uploader"),
        ("include/rdna4_kv_cache.hpp", "KV cache manager"),
        ("include/rdna4_pipeline.hpp", "Pipeline builder"),
        ("include/rdna4_scheduler.hpp", "Multi-ACE scheduler"),
        ("include/rdna4_engine.hpp", "Inference engine"),
        ("include/rdna4_profiler.hpp", "Profiler"),
        ("include/rdna4_allocator.hpp", "Ring allocator"),
        ("src/host/context.cpp", "Vulkan init"),
        ("src/host/scheduler.cpp", "Scheduler dispatch"),
        ("src/host/inference_engine.cpp", "Forward pass"),
        ("src/kernels/gemm.comp", "GEMM kernel"),
        ("src/kernels/attention.comp", "Attention kernel"),
        ("src/kernels/mlp.comp", "MLP kernel"),
        ("src/kernels/rope.comp", "RoPE kernel"),
        ("src/kernels/topk.comp", "Top-k kernel"),
        ("src/kernels/add.comp", "Add kernel"),
        ("src/kernels/rms_norm.comp", "RMS norm kernel"),
        ("src/kernels/embed.comp", "Embed kernel"),
        ("src/kernels/kv_cache_write.comp", "KV cache write kernel"),
        ("tools/rdna4_as.py", "SPIR-V assembler"),
        ("tools/gguf_loader.py", "GGUF parser"),
        ("tools/weight_converter.py", "Weight converter"),
    ]
    for path, desc in files:
        all_ok &= check_file(path, desc)

    print("\n--- Tool Chain ---")
    tools = [
        ("cmake", "CMake"),
        ("python3", "Python 3"),
    ]
    for cmd, desc in tools:
        all_ok &= check_tool(cmd, desc)

    # Check glslangValidator
    glslang = os.environ.get("GLSLANG_VALIDATOR", "glslangValidator")
    all_ok &= check_tool(glslang, "glslangValidator")

    print("\n--- Python Tools ---")
    try:
        from rdna4_as import Assembler
        print("  [OK] rdna4_as.py imports")
    except Exception as e:
        print(f"  [FAIL] rdna4_as.py: {e}")
        all_ok = False

    try:
        from gguf_loader import GGUFLoader
        print("  [OK] gguf_loader.py imports")
    except Exception as e:
        print(f"  [FAIL] gguf_loader.py: {e}")
        all_ok = False

    print("\n" + "=" * 40)
    if all_ok:
        print("All checks passed! Ready to build.")
        print("\nNext steps:")
        print("  mkdir build && cd build")
        print("  cmake .. -DGLSLANG_VALIDATOR=/path/to/glslangValidator")
        print("  cmake --build .")
        return 0
    else:
        print("Some checks failed. Fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
