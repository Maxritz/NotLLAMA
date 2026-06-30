# tools/ — Python Tooling Standards

## Purpose
Python scripts for model conversion, weight inspection, GGUF parsing, and RAG tools.

## Ownership
Python tools are owned by the build/CI pipeline. They are standalone scripts, not imported by the C++ runtime.

## Local Contracts

### Python Requirements
- Python 3.10+ required.
- Use `argparse` for CLI entry points.
- Use `typing` annotations (Python 3.9+ style: `list[int]`, `dict[str, float]`).
- No external dependencies unless strictly necessary. Prefer stdlib.
- If numpy is needed, guard the import and provide a pure-Python fallback for basic operations.

### Script Structure
```python
#!/usr/bin/env python3
"""One-line description."""
import argparse
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("-o", "--output", type=Path, default=Path("out.json"))
    args = parser.parse_args()
    # ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### GGUF / Quantization Rules
- K-quant formats use **d-last layout** (qs first, d at end) **except Q8_K** which is d-first.
- Q2_K block size: 84 bytes (scales[16] + qs[64] + d + dmin)
- Q3_K block size: 110 bytes (hmask[32] + qs[64] + scales[12] + d)
- Q4_K block size: 144 bytes (qs[128] + scales[12] + d + dmin)
- Q5_K block size: 176 bytes (qs[128] + qh[32] + scales[12] + d + dmin)
- Q6_K block size: 210 bytes (ql[128] + qh[64] + scales[16] + d)
- Q8_K block size: 292 bytes (float32 d + qs[256] + bsums[16]) — d-first exception
- Nibble extraction in Q4 formats: alternate low/high nibble. `index 0 -> low, index 1 -> high, index 2 -> low, ...`
- Validation functions must check `d` (delta/scale) at the **correct end-of-block offset**, not offset 0.

### Error Handling
- Use `try/except` around file I/O. Print clear error messages to stderr.
- Validate file sizes before allocating buffers.
- Check `is_open()` on all streams before reading.
- Return non-zero exit codes on failure.

## Work Guidance

### Adding a New Tool
1. Create script in `tools/`.
2. Add to root `AGENTS.md` Child DOX Index under `tools/`.
3. Document usage in script docstring.
4. If the tool produces JSON output, validate against a schema in `docs/`.

### Tool Checklist
- [ ] Has `#!/usr/bin/env python3` shebang
- [ ] Has `if __name__ == "__main__"` guard
- [ ] Returns `int` from `main()`
- [ ] Uses `argparse`
- [ ] Handles `FileNotFoundError`, `PermissionError`
- [ ] Validates inputs before processing
- [ ] Prints to stdout, errors to stderr

## Verification
- Run script with `--help` to verify argparse works.
- Run on a small test file (e.g., a 1-tensor GGUF) to verify basic functionality.

## Child DOX Index
- None (leaf directory)
