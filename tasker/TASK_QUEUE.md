# Task Queue — Updated
# Last updated: 2026-06-28

## Completed

- [x] fix_001–fix_007: All shader/kernel bugs fixed
- [x] feat_001: test_inference.cpp created
- [x] feat_002: cpu_reference.cpp/h created
- [x] forwardPartial() implemented
- [x] Speculative decode methods implemented

## In Progress

| # | Task ID | Title | Status |
|---|---------|-------|--------|
| 1 | test_build_run | Build & run test_inference.cpp | PENDING |
| 2 | test_debug | Fix any test failures kernel-by-kernel | BLOCKED on #1 |
| 3 | fix_exit_crash | Fix exit crash (0xC0000005) | PENDING |
| 4 | speculative_test | Test speculative decode pipeline | PENDING |
| 5 | cli_tools | Build CLI tools (llama-cli equiv) | PENDING |

## Future: MoE + Ornito

| # | Task ID | Title | Effort |
|---|---------|-------|--------|
| 6 | moe_support | Add MoE (mixture of experts) support | 2 weeks |
| 7 | ornito_support | Add Ornito model support (Qwen variant) | 1 week |

## Archived (done by DeepSeek)

- feat_003: llama.cpp feature audit — PROMPT prepared
