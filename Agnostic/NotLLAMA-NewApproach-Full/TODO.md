# TODO — Next Work Items

## Immediate (fix test_inference crash — code written, pending user build)

1. **test_inference crashes with 0xC0000005** — root cause identified: `test_inference.cpp` used a local `computeLogitsOffset()` that did not match the ring-allocator layout in `forwardPartial()`. Fix written: read back from `engine.lastLogitsOffset`. Build/run pending user confirmation.

2. **test_inference GPU/CPU logits comparison** — once crash is fixed, verify max absolute error < 0.01f

## High Priority

3. **Fix exit crash (0xC0000005)** — main.exe also crashes on cleanup. Review destruction order.

4. **Test speculative decode pipeline** — draftForward, verifyDraftToken, forwardSpeculative all implemented but untested

5. ~~**attention.comp: headDim=8 bug** — `hdPerThread = hd / 32` returns 0 for small headDim. Must clamp to min 1. Same in flash_attention.comp. Affects stories260K (headDim=8).~~ — Fixed in both `attention.comp` and `flash_attention.comp` (`max(hd/32u,1u)` + bounds checks).

## Medium Priority

6. **Build CLI tools** — llama-cli equivalent (interactive chat), llama-server (HTTP API), llama-bench (benchmarks)

7. **llama.cpp feature gap audit** — generate FEATURE_AUDIT.md comparing against llama.cpp features

## Future

8. **MoE support** — mixture of experts architecture

9. **Ornito model support** — Qwen variant architecture

10. **Cooperative matrix GEMM** — re-add VK_KHR_cooperative_matrix

11. ~~**kernel_entry.comp host wiring** — shader was written but forwardKernelEntry() didn't exist.~~ — Fixed. `forwardKernelEntry()` writes mailbox, dispatches one workgroup, reads next token. `forward()` auto-selects kernel_entry when `kernelEntryReady=true`.

## Known Issues

- Dequant staging buffer is sized for the largest simultaneous tensor set (MLP up+gate+down, ~270 MB for VibeThinker-3B), capped at 512 MB. `dequantWeight()` reuses the buffer per dispatch; MLP weights are packed at offsets 0 / upSize / upSize+gateSize.
- `embed.comp` uses `EmbedTableRef` as `float[]` — must match persistent embed cache format
- Flash attention kernel uses subgroup shuffle/ballot extensions — may not work on all hardware
- topk is single-threaded (tid==0 only) — functional but slow
