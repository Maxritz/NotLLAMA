#!/bin/bash
# NotLLAMA Critical Fixes — Quick Apply Script
# Usage: cd /path/to/NotLLAMA/repo && bash /path/to/apply_all_patches.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "NotLLAMA Critical Fixes — Patch Application"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "src/host/inference_engine.cpp" ]; then
    echo "ERROR: src/host/inference_engine.cpp not found."
    echo "Please run this script from the NotLLAMA repository root."
    exit 1
fi

echo "Target directory: $(pwd)"
echo ""

# Apply each patch
echo "[1/3] Applying inference_engine.patch (CRITICAL barrier fix + FFN scratch fix)..."
patch -p1 < "$SCRIPT_DIR/inference_engine.patch"
echo ""

echo "[2/3] Applying rope.patch (RoPE angle fix)..."
patch -p1 < "$SCRIPT_DIR/rope.patch"
echo ""

echo "[3/3] Applying rms_norm.patch (RMS norm parallelism)..."
patch -p1 < "$SCRIPT_DIR/rms_norm.patch"
echo ""

echo "=========================================="
echo "All patches applied successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Rebuild:  cmake --build build --target NotLLAMA"
echo "  2. Test:     ./build/NotLLAMA --model model.gguf --prompt 'Hello' --tokens 10"
echo "  3. Verify:   Check that MaxAE < 0.01 (was 10-24 before fix)"
echo ""
