# AGENTS.md — training/
## Child DOX Contract: Training Pipeline for Graphify-Optimized Models

**Status:** Binding work contract for the `training/` subtree.  
**Scope:** All training loop code, data curation, curriculum scheduling, mixed precision, quantization-aware training (QAT), and on-device training for AMD ROCm (RDNA4/RDNA2).  
**Parent:** `../AGENTS.md` (root). This doc inherits all root rules and specializes them for training.  
**Self-Diagnostic Mode:** INHERITED from parent §3 — Agent MUST verify prerequisites before any implementation work.

---

## 1. Purpose

This module trains LLMs optimized for the Graphify token-minimization pipeline. Training produces models that:
- Generate grounded, citation-backed responses with minimal hallucination.
- Operate efficiently within strict token budgets (inference) and memory budgets (training).
- Are quantized-aware for Q4/Q8/BF16 inference on AMD ROCm hardware.

---

## 2. Ownership

| Boundary | Owner | Contact |
|----------|-------|---------|
| Training orchestration | `training/AGENTS.md` (this doc) | — |
| Data curation / KG verification | `training/data/AGENTS.md` | Child DOX |
| Model architecture / QAT layers | `training/models/AGENTS.md` | Child DOX |
| Evaluation / hallucination metrics | `training/eval/AGENTS.md` | Child DOX |

---

## 3. Inherited Rules (Cannot Be Weakened)

All rules from parent `AGENTS.md` §3-§4 apply here without modification:
- §3 Self-Diagnostic: Prerequisites Check (MANDATORY FIRST STEP) — **Agent MUST run diagnostic before ANY training code.**
- §3.1 Diagnostic Checklist (CHK-01 through CHK-10).
- §3.2 Diagnostic Output Format.
- §3.3 Setup Mode vs. Development Mode.
- §3.4 Conditional Implementation Paths.
- §4.1 Token Budget Is Non-Negotiable (extended to training batch budgets).
- §4.2 Retrieved Facts Must Be Grounded (extended to training data verification).
- §4.3 Context Is Tiered — Never Flat (extended to Training Tiers TT1-TT4).
- §4.4 Graphify Is Primary Retrieval and Ground Truth (training data ground truth).
- §4.5 No Code Without a Trace (training loops, data loaders, loss functions require FULL traces).
- §4.6 Hallucination Prevention Starts at Training (data curation, curriculum, evaluation).
- §4.7 Agent Must Help With Setup (generate scripts, Docker files, SETUP.md guides).

---

## 4. Local Contracts (Training-Specific)

### 4.1 Hardware Target Contracts

| GPU | Architecture | ROCm Version | Compute API | Memory Budget | Default Precision |
|-----|-------------|--------------|-------------|---------------|-------------------|
| RX 9070 XT | RDNA4 (gfx1201) | 7.1+ | HIP / rocWMMA | 16 GB VRAM | BF16 / FP8 |
| RX 6700 XT | RDNA2 (gfx1031) | 7.1+ | HIP / Vulkan fallback | 12 GB VRAM | BF16 |

- Training scripts MUST detect `gfx1201` vs `gfx1031` at runtime and select appropriate kernels.
- If ROCm is unstable on gfx1031, Vulkan compute compute shaders MUST be available as fallback.
- Gradient checkpointing is MANDATORY for models > 7B parameters on ≤ 16GB VRAM.

### 4.2 Data Curation Pipeline

1. **Source Ingestion:** Raw data enters via `raw/`. Supported formats: JSONL, Parquet, HuggingFace datasets.
2. **Entity Extraction:** Run spaCy / lightweight NER on every sample. Extract entities for KG lookup.
3. **KG Verification:** For TT1/TT2 targets, every factual claim is verified against Graphify KG.
   - Confidence threshold: TT1 ≥ 0.9, TT2 ≥ 0.8.
   - Rejection rate logged. If > 50% rejected, source is flagged and pipeline aborts.
4. **Tier Assignment:**
   - TT1: 100% KG-verified, human-curated or high-trust synthetic.
   - TT2: ≥ 90% KG-verified, synthetic with KG grounding.
   - TT3: Heuristic-filtered web data, no KG verification required.
   - TT4: Unverified synthetic, raw exploration data.
5. **Deduplication:** Per-tier deduplication using MinHash + entity resolution. Duplicate samples waste training compute.
6. **Tokenization:** Use the same tokenizer as inference. Sequence length ≤ `max_sequence_tokens` (default 2048). Padding waste < 10%.

### 4.3 Curriculum Schedule

| Phase | Steps (%) | Allowed Tiers | Learning Rate | Special Conditions |
|-------|-----------|---------------|---------------|-------------------|
| 1 — Foundation | 0-30% | TT1 only | base_lr | Warmup to base_lr over 5% of steps |
| 2 — Expansion | 30-60% | TT1 + TT2 | base_lr * 0.8 | Linear decay starts at 50% |
| 3 — Generalization | 60-85% | TT1 + TT2 + TT3 | base_lr * 0.5 | Introduce longer sequences (up to 4096) |
| 4 — RLHF + QAT | 85-100% | All tiers + RLHF | base_lr * 0.2 | Enable QAT layers, reward model = KG verifier |

- Curriculum scheduler MUST lock data loaders to `allowed_tiers` per phase. No TT4 in Phase 1-2.
- Phase transitions are step-based, not epoch-based. Track global step count.

### 4.4 Mixed Precision & Memory Strategy

1. **Default Precision:** BF16 for all training. FP8 only if ROCm 7.1+ supports `rocWMMA` FP8 on gfx1201.
2. **Loss Scaling:** Use `torch.cuda.amp.GradScaler` for FP16/FP8. BF16 does not require loss scaling.
3. **Gradient Checkpointing:**
   - Enable for transformer layers when `model_size > 7B` AND `VRAM <= 16GB`.
   - Tradeoff: 20-30% more compute, 40-50% less memory. This is acceptable.
4. **Activation Checkpointing:** Store only attention weights and recompute feed-forward activations during backward.
5. **Optimizer States:** Use 8-bit AdamW (`bitsandbytes` or custom) to reduce optimizer state memory by 2x.

### 4.5 Quantization-Aware Training (QAT)

1. **Timing:** QAT fake quantization layers are enabled ONLY in Phase 4 (final 15% of training).
2. **Target Formats:** Q4_0, Q4_K, Q8_0, BF16. Must match inference quantization exactly.
3. **Implementation:**
   - Replace `nn.Linear` with `FakeQuantizedLinear` that applies `quantize → dequantize` during forward.
   - Gradients flow through the dequantized weights (straight-through estimator).
   - Per-layer quantization error logged. If any layer error > 2%, reduce quantization granularity for that layer.
4. **Validation:** After QAT enablement, run forward on 100 held-out samples. Assert MSE(full_precision, fake_quantized) < 1%.

### 4.6 Evaluation & Checkpointing

1. **Evaluation Frequency:** Every `eval_steps` (default 500).
2. **Evaluation Set:** KG-held-out validation set (facts NOT seen during training). Minimum 1000 samples.
3. **Metrics:**
   - Perplexity (standard LM metric).
   - Hallucination rate: % of generated claims without KG source (target < 2%).
   - Token efficiency: % of prompt tokens used in response (target > 80%).
   - KG coverage: % of factual queries answered from KG (target > 85%).
4. **Checkpoint Criteria:**
   - Save ONLY if `hallucination_rate < 0.02` AND `perplexity < best_perplexity * 1.05`.
   - If criteria fail, log failure and continue training. Do not save.
   - Best checkpoint is promoted to `checkpoints/best/`. Others go to `checkpoints/history/` with TTL (auto-delete after 7 days).
5. **Final Export:** After training, export to GGUF/Q4/Q8 format using the same quantization scheme as QAT. Validate with `llama.cpp` or `ollama` inference test.

### 4.7 On-Device Training (AMD ROCm)

1. **Kernel Selection:**
   - gfx1201: Use `rocWMMA` for FP16/BF16 GEMM, `hipBLASLt` for general matrix ops.
   - gfx1031: Use standard `hipBLAS`. If ROCm is unstable, compile Vulkan compute shaders for training ops.
2. **Wave Size:** gfx1201 defaults to Wave32. gfx1031 defaults to Wave64. Do not override unless trace confirms benefit.
3. **Memory Pool:** Use `hipMalloc` with memory pool for activation buffers. Avoid repeated alloc/free during training loop.
4. **Compilation:** Use `hipcc` with `-mno-wavefrontsize64` for gfx1201 Wave32 kernels. Use `-mcpu=gfx1031` for RDNA2.
5. **Profiling:** Every training run MUST capture `rocprof` trace for the first 100 steps. Verify kernel occupancy > 70% and no host-side bottlenecks.

### 4.8 Self-Diagnostic for Training (Extension of Parent §3)

Before ANY training task, the agent MUST run the parent diagnostic (CHK-01 through CHK-10) PLUS these training-specific checks:

| Check | What to Verify | If Missing | Severity |
|-------|--------------|------------|----------|
| **CHK-TR-01** | PyTorch with ROCm support (`torch.version.hip`) | Training cannot run on AMD GPU. | **BLOCKING** |
| **CHK-TR-02** | Training data directory structure (`data/raw/`, `data/tt1/`, etc.) | No staging for data curation. | **BLOCKING** |
| **CHK-TR-03** | Model checkpoint directory (`checkpoints/`) | No place to save checkpoints. | **WARNING** |
| **CHK-TR-04** | `rocminfo` or `vulkaninfo` executable (GPU detection) | Cannot verify GPU architecture. | **WARNING** |
| **CHK-TR-05** | `bitsandbytes` or custom 8-bit optimizer | Optimizer states will overflow VRAM. | **WARNING** |

If CHK-TR-01 or CHK-TR-02 fails, the agent MUST enter SETUP MODE and provide installation commands:
```bash
# CHK-TR-01: PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# For ROCm 7.1 (when available):
# pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# CHK-TR-02: Training data directories
mkdir -p data/{raw,processed,tt1,tt2,tt3,tt4}
mkdir -p checkpoints/{best,history}

# CHK-TR-03: Checkpoint directory (auto-create in script)
# CHK-TR-04: ROCm info
# Usually installed with ROCm package: /opt/rocm/bin/rocminfo

# CHK-TR-05: 8-bit optimizer
pip install bitsandbytes
# Or for custom: compile from source with hipcc
```

---

## 5. Verification Requirements

### 5.1 Pre-Training Checks

Before any training run begins, verify:

- [ ] Data verification: 100% of TT1 data, ≥ 90% of TT2 data has been KG-fact-checked.
- [ ] Token efficiency: Padding waste < 10% across all training batches.
- [ ] Curriculum integrity: Training script enforces phase-locked tier exposure (no TT4 in Phase 1).
- [ ] Memory trace: GPU memory usage trace confirms no OOM at max batch size + gradient checkpointing.
- [ ] Checkpoint criteria: Evaluation script measures hallucination rate and token efficiency, not just loss.
- [ ] Hardware detection: Script correctly identifies gfx1201 vs gfx1031 and selects appropriate kernels.
- [ ] QAT readiness: Fake quantization layers are implemented and validated against inference quantization.
- [ ] **Prerequisites: All BLOCKING diagnostic checks (CHK-01 through CHK-07, CHK-TR-01, CHK-TR-02) PASS.**

### 5.2 Runtime Assertions (Training)

These MUST be present in training code:

```python
# At data loader batch construction
assert padding_tokens / total_tokens < 0.10,     "Padding waste exceeds 10% — fix bucketing"

# At curriculum phase boundary
assert current_phase in ALLOWED_TIERS[training_step],     f"Training step {training_step} sees forbidden tier {current_tier}"

# At checkpoint save
assert eval_hallucination_rate < 0.02,     f"Checkpoint hallucination rate {eval_hallucination_rate} exceeds 2% — discarding"

# At synthetic data ingestion
assert synthetic_verification_passed or tier_assignment == "TT4",     "Unverified synthetic data assigned to TT1/TT2/TT3 — data corruption"

# At QAT enablement
assert current_step >= 0.85 * total_steps,     f"QAT enabled at step {current_step}, but Phase 4 starts at {0.85 * total_steps}"

# At GPU kernel dispatch
assert gpu_arch in ["gfx1201", "gfx1031"],     f"Unknown GPU architecture {gpu_arch} — cannot select kernels"

# At prerequisite diagnostic (training-specific)
assert torch.version.hip is not None,     "CHK-TR-01 FAIL: PyTorch not compiled with ROCm. Install: pip install torch --index-url https://download.pytorch.org/whl/rocm5.7"
assert os.path.exists("data/raw/"),     "CHK-TR-02 FAIL: Training data directory missing. Run: mkdir -p data/{raw,processed,tt1,tt2,tt3,tt4}"
```

---

## 6. Child DOX Index

- `training/data/` — Data curation pipeline, KG verification filter, synthetic data generator, tier assignment logic
- `training/models/` — Model architecture, quantization-aware layers, gradient checkpointing hooks, ROCm kernel wrappers
- `training/eval/` — Hallucination evaluator, token efficiency benchmark, KG-held-out test sets, `rocprof` profiling harness

---

## 7. Closeout Rule

Every task ending in this subtree MUST:
1. Re-check changed paths against the DOX chain (`training/AGENTS.md` → `../AGENTS.md` → `../CLAUDE.md`).
2. Update the nearest owning AGENTS.md if the change affects purpose, scope, contracts, workflows, or artifacts.
3. Refresh every affected Child DOX Index.
4. Remove stale or contradictory text immediately.
5. Run verification checks (§5.1, §5.2) when relevant.
6. Report any docs intentionally left unchanged and why.

---

*AGENTS.md v2.0 | training/ | Child DOX Contract | Inherits Self-Diagnostics from Parent*
