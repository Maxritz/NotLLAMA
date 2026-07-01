#!/usr/bin/env python3
"""DOX lint tool — validates AGENTS.md compliance, push constant sizes,
shader SPIR-V presence, TODO/FIXME counts, and large files."""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
INCLUDE = REPO_ROOT / "include"
KERNELS = SRC / "kernels"
BUILD_SHADERS = REPO_ROOT / "build" / "shaders"

CRITICAL_FAILURE = False

# ── Helpers ──────────────────────────────────────────────────────────────

def has_agents_md(dirpath: Path) -> bool:
    return (dirpath / "AGENTS.md").is_file()

def find_parent_agents(dirpath: Path) -> Path | None:
    for parent in dirpath.parents:
        amd = parent / "AGENTS.md"
        if amd.is_file():
            content = amd.read_text(encoding="utf-8", errors="replace")
            if dirpath.name in content:
                return amd
    return None

def get_required_sections(filepath: Path) -> dict[str, bool]:
    """Check for required section headers (case-insensitive)."""
    required = ["Purpose", "Ownership", "Verification", "Child DOX Index"]
    optional_work = ["Work Guidance", "Local Contracts"]

    if not filepath.is_file():
        return {s: False for s in required + optional_work}

    text = filepath.read_text(encoding="utf-8", errors="replace").lower()
    result = {}
    for s in required:
        result[s] = f"## {s.lower()}" in text
    for s in optional_work:
        result[s] = f"## {s.lower()}" in text
    result["_has_work_or_contracts"] = any(result[s] for s in optional_work)
    return result

# ── Check 1: AGENTS.md Coverage ─────────────────────────────────────────

def check_agents_coverage() -> list[dict]:
    rows = []
    dirs_to_check = ["src", "include"]

    for top in dirs_to_check:
        top_path = REPO_ROOT / top
        for dirpath, _dirnames, filenames in os.walk(top_path):
            dirpath = Path(dirpath)
            has_cpp = any(f.endswith((".cpp", ".hpp", ".comp")) for f in filenames)
            if not has_cpp:
                continue
            has_amd = has_agents_md(dirpath)
            parent_path = None
            if not has_amd:
                parent_path = find_parent_agents(dirpath)
            rows.append({
                "dir": str(dirpath.relative_to(REPO_ROOT)),
                "has_amd": has_amd,
                "parent": str(parent_path.relative_to(REPO_ROOT)) if parent_path else "N/A",
            })
    return rows

# ── Check 2: AGENTS.md Sections ─────────────────────────────────────────

def check_agents_sections(agents_files: list[Path]) -> list[dict]:
    rows = []
    for f in agents_files:
        secs = get_required_sections(f)
        rows.append({
            "file": str(f.relative_to(REPO_ROOT)),
            "purpose": secs["Purpose"],
            "ownership": secs["Ownership"],
            "work_guidance": secs["Work Guidance"],
            "local_contracts": secs["Local Contracts"],
            "verification": secs["Verification"],
            "child_dox": secs["Child DOX Index"],
            "has_work_or_contracts": secs["_has_work_or_contracts"],
        })
    return rows

# ── Check 3: Push Constant Size ─────────────────────────────────────────

FIELD_SIZES = {
    "uint32_t": 4,
    "uint64_t": 8,
    "float": 4,
    "int32_t": 4,
    "int64_t": 8,
    "bool": 4,  # aligned
}

PUSH_CONST_STRUCT_RE = re.compile(
    r"struct\s+(\w+PushConstants?\w*)\s*\{([^}]+)\};"
)
FIELD_RE = re.compile(
    r"\b(uint32_t|uint64_t|float|int32_t|int64_t|bool)\s+\w+"
)

def estimate_push_const_size(struct_body: str) -> int:
    size = 0
    max_align = 8  # largest scalar is uint64_t
    offset = 0
    for match in FIELD_RE.finditer(struct_body):
        ftype = match.group(1)
        fsize = FIELD_SIZES.get(ftype, 4)
        # align
        if offset % fsize != 0:
            offset = (offset + fsize - 1) // fsize * fsize
        offset += fsize
        size = offset
    # round up to max_align
    if size % max_align != 0:
        size = (size + max_align - 1) // max_align * max_align
    return size

def check_push_constants() -> list[dict]:
    rows = []
    headers = [
        INCLUDE / "rdna4_types.hpp",
        INCLUDE / "rdna4_compression.hpp",
    ]
    for hdr in headers:
        if not hdr.is_file():
            continue
        text = hdr.read_text(encoding="utf-8", errors="replace")
        # Remove comments
        text = re.sub(r"//.*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        for match in PUSH_CONST_STRUCT_RE.finditer(text):
            name = match.group(1)
            body = match.group(2)
            size = estimate_push_const_size(body)
            ok = size <= 128
            rows.append({
                "struct": name,
                "size": size,
                "status": "OK" if ok else "OVER 128",
            })
            if not ok:
                global CRITICAL_FAILURE
                CRITICAL_FAILURE = True
    return rows

# ── Check 4: Shader Compilation ─────────────────────────────────────────

def check_shaders() -> list[dict]:
    rows = []
    if not KERNELS.is_dir():
        return rows
    for comp_file in sorted(KERNELS.glob("*.comp")):
        spv_name = comp_file.with_suffix(".spv").name
        spv_path = BUILD_SHADERS / spv_name
        exists = spv_path.is_file()
        if not exists:
            global CRITICAL_FAILURE
            CRITICAL_FAILURE = True
        rows.append({
            "shader": comp_file.name,
            "spv_exists": exists,
        })
    return rows

# ── Check 5: TODO/FIXME Tracker ─────────────────────────────────────────

def count_todos_fixmes(filepath: Path) -> tuple[int, int]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    todos = len(re.findall(r"\bTODO\b", text, re.IGNORECASE))
    fixmes = len(re.findall(r"\bFIXME\b", text, re.IGNORECASE))
    return todos, fixmes

# ── Check 6: Large Files ────────────────────────────────────────────────

def find_large_files() -> list[dict]:
    large = []
    for root_dir in [SRC, INCLUDE]:
        if not root_dir.is_dir():
            continue
        for f in sorted(root_dir.rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".hpp", ".comp"):
                try:
                    line_count = sum(1 for _ in f.open("rb"))
                except Exception:
                    continue
                if line_count > 5000:
                    large.append({
                        "file": str(f.relative_to(REPO_ROOT)),
                        "lines": line_count,
                    })
    return large

# ── Main ────────────────────────────────────────────────────────────────

def main():
    global CRITICAL_FAILURE

    # Collect all AGENTS.md files
    agents_files = sorted(REPO_ROOT.rglob("AGENTS.md"))

    print("# DOX Lint Report\n")

    # ── AGENTS.md Coverage ──
    coverage = check_agents_coverage()
    print("## AGENTS.md Coverage\n")
    print("| Directory | Has AGENTS.md | Listed in Parent |")
    print("|-----------|---------------|------------------|")
    for r in coverage:
        print(f"| {r['dir']} | {'YES' if r['has_amd'] else 'NO'} | {r['parent']} |")
        if not r['has_amd'] and r['parent'] == "N/A":
            CRITICAL_FAILURE = True
    print()

    # ── AGENTS.md Sections ──
    sections = check_agents_sections(agents_files)
    print("## AGENTS.md Sections\n")
    print("| File | Purpose | Ownership | Work Guidance | Local Contracts | Verification | Child DOX |")
    print("|------|---------|-----------|---------------|-----------------|--------------|-----------|")
    for r in sections:
        print(f"| {r['file']} | {'Y' if r['purpose'] else '-'} | {'Y' if r['ownership'] else '-'} | "
              f"{'Y' if r['work_guidance'] else '-'} | {'Y' if r['local_contracts'] else '-'} | "
              f"{'Y' if r['verification'] else '-'} | {'Y' if r['child_dox'] else '-'} |")
        if not r['purpose'] or not r['ownership'] or not r['has_work_or_contracts']:
            CRITICAL_FAILURE = True
    print()

    # ── Push Constants ──
    push_consts = check_push_constants()
    print("## Push Constant Size Check\n")
    print("| Struct | Estimated Size | Status |")
    print("|--------|---------------|--------|")
    for r in push_consts:
        print(f"| {r['struct']} | {r['size']} | {r['status']} |")
    print()

    # ── Shader Compilation ──
    shaders = check_shaders()
    print("## Shader Compilation Check\n")
    print("| Shader | SPIR-V Exists |")
    print("|--------|---------------|")
    for r in shaders:
        print(f"| {r['shader']} | {'YES' if r['spv_exists'] else 'NO'} |")
    print()

    # ── TODO/FIXME Summary ──
    print("## TODO/FIXME Summary\n")
    heavy_files = []
    for root_dir in [SRC, INCLUDE]:
        if not root_dir.is_dir():
            continue
        for f in sorted(root_dir.rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".hpp", ".comp"):
                todos, fixmes = count_todos_fixmes(f)
                if todos + fixmes > 5:
                    heavy_files.append({
                        "file": str(f.relative_to(REPO_ROOT)),
                        "todos": todos,
                        "fixmes": fixmes,
                    })
    print("| File | TODO | FIXME |")
    print("|------|------|-------|")
    for r in heavy_files:
        print(f"| {r['file']} | {r['todos']} | {r['fixmes']} |")
    if not heavy_files:
        print("_(no files with >5 TODOs/FIXMEs)_")
    print()

    # ── Large Files ──
    large = find_large_files()
    print("## Large Files\n")
    print("| File | Lines |")
    print("|------|-------|")
    for r in large:
        print(f"| {r['file']} | {r['lines']} |")
    if not large:
        print("_(no files >5000 lines)_")
    print()

    print("## Exit Code")
    code = 1 if CRITICAL_FAILURE else 0
    print(f"```\n{code}\n```")
    sys.exit(code)

if __name__ == "__main__":
    main()
