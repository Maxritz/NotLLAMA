# TODO — Next Work Items

## Immediate (fix crash)

1. **test_inference crashes with 0xC0000005** — forwardPartial crashes during GPU forward pass. Debug logging added to `inference_engine.cpp`. Need to trace exact crash point (embed dispatch, first GEMM, attention, etc.)

2. **Missing `engine.cleanup()` in test_inference.cpp** — cleanup order may differ from main.cpp. Add full cleanup sequence.

## High Priority

3. **GPU vs CPU logits comparison** — once crash is fixed, verify max absolute error < 0.01f

4. **Fix exit crash (0xC0000005)** — main.exe also crashes on cleanup. Review destruction order.

5. **Test speculative decode pipeline** — draftForward, verifyDraftToken, forwardSpeculative all implemented but untested

## Medium Priority

6. **Build CLI tools** — llama-cli equivalent (interactive chat), llama-server (HTTP API), llama-bench (benchmarks)

7. **llama.cpp feature gap audit** — generate FEATURE_AUDIT.md comparing against llama.cpp features

## Future

8. **MoE support** — mixture of experts architecture

9. **Ornito model support** — Qwen variant architecture

10. **Cooperative matrix GEMM** — re-add VK_KHR_cooperative_matrix

11. **kernel_entry.comp phases 1 & 2** — implement attn/MLP dispatches for persistent kernel path

## Known Issues

- Dequant staging buffer is 128MB but each layer's Q6_K weights need much more — `dequantWeight()` reuses same buffer per layer (should work if dispatches are sequential)
- `embed.comp` uses `EmbedTableRef` as `float[]` — must match persistent embed cache format
- Flash attention kernel uses subgroup shuffle/ballot extensions — may not work on all hardware
- topk is single-threaded (tid==0 only) — functional but slow
