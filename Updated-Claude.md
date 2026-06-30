# CLAUDE.md — Graphify Token Optimization Project
## Behavioral Contract for Coding Assistants

**Status:** Binding behavioral rules. Merge with project-specific `AGENTS.md` as needed.  
**Scope:** All code, configuration, and documentation written for the Graphify token-optimization pipeline, including model training, data curation, curriculum design, and quantization-aware training.  
**Parent:** `AGENTS.md` (root DOX contract). This doc specializes the generic guidelines for this project.  
**Self-Diagnostic Mode:** ENABLED — Agent MUST verify prerequisites before any implementation work.

---

## 1. Think Before Coding

### 1.1 State Assumptions Explicitly

Before writing any retrieval, context assembly, generation, training loop, data loader, or loss function logic, you MUST state:
- What token budget this code targets (default: 4K context, 1.2K reserve for inference; max_sequence_tokens=2048, padding_waste<10% for training).
- What retrieval backend is primary (Graphify KG) vs fallback (vector store).
- What tier this context belongs to (T1/T2/T3/T4 for inference; TT1/TT2/TT3/TT4 for training data).
- What training phase this targets (Phase 1-4) and which tiers are allowed.
- **What prerequisites are currently satisfied vs. missing (run diagnostic per AGENTS.md §3).**
- If uncertain about any of these, ASK. Do not pick silently.

### 1.2 Surface Tradeoffs

When a design choice affects token count, hallucination risk, training convergence, or memory usage, present the tradeoff:
- "Adding this validation step costs +50 tokens per call but reduces hallucination rate by 1.2%."
- "Using full history here costs 800 tokens. Summarized history costs 120 tokens but may lose temporal nuance."
- "Gradient checkpointing costs 20% more compute but allows 2x batch size on 16GB VRAM."
- "Training on TT4 (unverified) data in Phase 1 may teach hallucinations before the model learns grounded patterns."
- If a simpler approach exists, say so. Push back when the user request is over-engineered.

### 1.3 Stop on Confusion

If the retrieval strategy, context tier, training phase, curriculum tier exposure, quantization target, **or prerequisite status** is unclear:
1. Name what is confusing.
2. Reference the relevant `AGENTS.md` section.
3. Ask for clarification before proceeding.

---

## 2. Self-Diagnostic Behavior (MANDATORY)

### 2.1 Run Diagnostic Before Any Implementation

**Rule: The agent MUST run the prerequisites diagnostic (AGENTS.md §3.1) at the start of EVERY task. If any BLOCKING check fails, the agent MUST STOP implementation and enter SETUP MODE.**

**Diagnostic Flow:**
```
User requests feature/task
│
├── Run diagnostic checklist (CHK-01 through CHK-10)
│   ├── ALL BLOCKING checks PASS → Enter DEVELOPMENT MODE
│   │   └── Proceed with implementation per §3-§7
│   └── ANY BLOCKING check FAILS → Enter SETUP MODE
│       └── STOP implementation immediately
│       └── Output diagnostic results + setup guide
│       └── Offer to generate setup scripts
│       └── WAIT for user confirmation before proceeding
│
└── User confirms "setup complete" → Re-run diagnostic → Enter DEVELOPMENT MODE
```

### 2.2 SETUP MODE Behavior

When in SETUP MODE, the agent MUST:

1. **Output a clear diagnostic banner:**
   ```
   ╔══════════════════════════════════════════════════════════════════╗
   ║  PREREQUISITES MISSING — IMPLEMENTATION BLOCKED                ║
   ╠══════════════════════════════════════════════════════════════════╣
   ║  Failed Checks: [CHK-01, CHK-03, CHK-05]                     ║
   ╚══════════════════════════════════════════════════════════════════╝
   ```

2. **Explain WHY each missing component is BLOCKING:**
   - CHK-01 (Graphify KG missing): "Without a knowledge graph, facts cannot be verified. Hallucination prevention is impossible. Every generated claim would be ungrounded."
   - CHK-04 (Token Counter missing): "Without a tokenizer, I cannot enforce token budgets. Context assembly would overflow silently, burning tokens and potentially crashing the LLM backend."

3. **Provide copy-paste ready installation commands:**
   ```bash
   # CHK-01: Graphify Knowledge Graph
   pip install neo4j networkx
   # Or Docker:
   docker run -d -p 7474:7474 -p 7687:7687 --name neo4j      -e NEO4J_AUTH=neo4j/password neo4j:latest

   # CHK-03: Entity Extractor
   pip install spacy
   python -m spacy download en_core_web_sm

   # CHK-04: Token Counter
   pip install tiktoken transformers

   # CHK-05: Embedding Model
   pip install sentence-transformers

   # CHK-06: LLM Backend (local)
   pip install llama-cpp-python
   # Or (cloud)
   export OPENAI_API_KEY=your_key_here

   # CHK-07: Training Environment
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   # Or for ROCm 7.1:
   pip install torch --index-url https://download.pytorch.org/whl/rocm7.1
   ```

4. **Offer to generate setup scripts:**
   - "I can generate a `setup/install_prerequisites.sh` (or `.ps1` for Windows) that automates all of this. Would you like me to?"
   - "I can create a `docker-compose.yml` with Neo4j, Chroma, and the embedding model pre-configured."

5. **Create a `SETUP.md` guide** documenting:
   - What each component does and why it's required.
   - How to verify installation (test commands).
   - How to configure connection strings / API keys.
   - How to initialize the Graphify schema (Cypher queries for Neo4j, or Python scripts for NetworkX).

6. **DO NOT write implementation code** for the requested feature until the user confirms all BLOCKING prerequisites are installed.

### 2.3 DEVELOPMENT MODE Entry

The agent enters DEVELOPMENT MODE ONLY when:
- All BLOCKING checks (CHK-01 through CHK-07) pass.
- The user explicitly confirms: "setup complete", "all installed", "proceed", or similar.

**Transition Rule:** The agent MUST re-run the diagnostic after user confirmation. If checks still fail, remain in SETUP MODE. Do not assume the user fixed everything.

### 2.4 Partial Prerequisites (WARNING Checks)

If only WARNING checks fail (CHK-08, CHK-09, CHK-10):
- The agent MAY proceed with implementation BUT MUST flag the limitations.
- Example: "CHK-08 (Graphify schema empty) is a WARNING. I can build the pipeline, but retrieval will return zero facts until you populate the KG. I'll include a schema initialization script in the output."
- The agent MUST include fallback logic for missing WARNING components (e.g., if KG is empty, skip KG traversal and use vector search exclusively, with a log warning).

---

## 3. Simplicity First

### 3.1 Minimum Code That Solves the Problem

- No features beyond what was asked.
- No abstractions for single-use retrieval logic.
- No "flexibility" or "configurability" that wasn't requested.
  - Exception: Token budget parameters (max_context_tokens, response_reserve) MUST be configurable per AGENTS.md §4.1.
  - Exception: Training hyperparameters (learning_rate, batch_size, gradient_accumulation, max_sequence_tokens, curriculum_phase) MUST be configurable per AGENTS.md §5.5.
- No error handling for impossible scenarios (e.g., do not handle negative token counts unless the input path allows them).
- No training loop complexity beyond what the hardware supports. If the target is RX 9070 XT 16GB, do not write code that assumes 80GB A100 memory.

### 3.2 Token Count Is a First-Class Constraint

Every function that assembles, retrieves, generates, trains, or curates data MUST account for token cost:
- If you write a retrieval function, it MUST accept and respect a max_retrieved_tokens parameter.
- If you write a summarizer, it MUST accept a target_summary_tokens parameter.
- If you write a prompt template, it MUST include a token budget comment.
- If you write a data loader, it MUST accept max_sequence_tokens and packing_strategy parameters, and MUST log padding waste per batch.
- If you write a training script, it MUST declare effective_batch_size = batch_size * gradient_accumulation * world_size and verify it fits in GPU memory via a trace.

Ask yourself: "Would a senior engineer say this retrieval logic is burning tokens unnecessarily?" Or: "Would a senior ML engineer say this training loop is wasting compute on padding or unverified data?" If yes, simplify.

---

## 4. Surgical Changes

### 4.1 Touch Only What You Must

When editing existing RAG, context-management, training, or data curation code:
- Do not "improve" adjacent retrieval logic, comment formatting, or prompt templates.
- Do not refactor context tiers that aren't broken.
- Do not change training hyperparameters, curriculum phases, or data tier assignments unless asked.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code (e.g., an unused vector search fallback), mention it — do not delete it unless asked.
- If you notice training data loaded without KG verification, flag it as a CRITICAL issue per AGENTS.md §4.6.

### 4.2 Clean Up Your Own Orphans

When your changes create unused code:
- Remove imports, variables, or helper functions that YOUR changes made unused.
- Do not remove pre-existing dead code unless asked.
- Remove training data transforms that YOUR changes made redundant (e.g., old padding logic replaced by packing).

**The test:** Every changed line should trace directly to the user's request OR to a token-optimization/hallucination-prevention requirement in AGENTS.md.

---

## 5. Goal-Driven Execution

### 5.1 Define Success Criteria Before Coding

Transform every task into verifiable goals:

| Vague Request | Verifiable Goal |
|---------------|-----------------|
| "Add context trimming" | "Write tests: (a) history > 1000 tokens triggers trim, (b) T1 facts never dropped, (c) total tokens ≤ budget - reserve" |
| "Fix hallucination bug" | "Write test reproducing unsourced claim, then make verification catch it and reject response" |
| "Optimize retrieval" | "Benchmark: KG hit rate ≥ 85%, avg retrieved tokens ≤ 600, precision ≥ 90%" |
| "Add training data pipeline" | "Write tests: (a) TT1 data 100% KG-verified, (b) padding waste < 10%, (c) curriculum phase locks tier exposure, (d) synthetic data flagged TT4 if unverified" |
| "Implement QAT" | "Write tests: (a) Q4/Q8/BF16 fake quantization layers match inference path, (b) gradient flow verified through quantized layers, (c) checkpoint eval on KG-held-out set passes" |
| "Train on local GPU" | "Trace confirms batch size fits 16GB VRAM with gradient checkpointing, activation checkpointing, and BF16 mixed precision" |

### 5.2 Brief Plan for Multi-Step Tasks

Before implementing, state:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Example:
```
1. Implement entity extractor → verify: "USA" extracted from "United States of America"
2. Wire extractor to Graphify query path → verify: KG traversal receives entity list, not raw string
3. Add token budget enforcement to assembler → verify: prompt > budget aborts with error
```

Training example:
```
1. Implement KG-verified data filter → verify: TT1 data has 100% KG triple match, TT2 has ≥ 90%
2. Implement dynamic sequence bucketing → verify: padding waste < 10% across 100 batches
3. Implement curriculum phase scheduler → verify: Phase 1 sees only TT1, Phase 2 sees TT1+TT2, etc.
4. Implement Q4 fake quantization layers → verify: forward pass matches inference quantization, backward passes gradients
5. Integrate ROCm 7.1 mixed precision → verify: loss scaling stable, no NaN gradients, memory trace fits 16GB
```

---

## 6. Truth/False Logic Validation (MANDATORY)

### 6.1 No Code Writes Without a Trace

Per AGENTS.md §4.5 and parent DOX: Every new algorithm, I/O path, data format, concurrency, memory/buffer resize, training loop, data loader, loss function, or gradient path MUST have a trace before implementation.

### 6.2 Full Trace Template (Mandatory for New Logic)

Use this exact format. Do not skip sections.

```markdown
## TRACE: [Feature Name]

### Input State
| Variable | Value | Source |
|----------|-------|--------|
| query | user string | input |
| history | list[Turn] | session state |
| budget | 4000 | config |
| reserve | 1200 | config |
| batch_size | 4 | config |
| max_seq_len | 2048 | config |
| curriculum_phase | 2 | config |

### Flow Diagram
```
function(args)
│
├── condition?
│   ├── TRUE → action A
│   └── FALSE → action B
```

### Decision Tree
```
if (condition):
  TRUE → do A
  FALSE → do B
```

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| [condition] | [expected] | [actual] | [PASS/FAIL] |

### Completeness Check
- Branch coverage: ALL branches reachable and tested? [YES/NO]
- State completeness: ALL variable values handled (including error states, OOM, NaN loss)? [YES/NO]
- Goal reachability: EVERY valid input reaches output without silent failure? [YES/NO]

### Resource Ordering
- Producer → Consumer: [pair]
- DAG ordering verified (write before read)? [YES/NO]
- Synchronization/barrier placement: BEFORE consumer access? [YES/NO]
- Deadlock/circular wait risk: [NONE/LOW/HIGH]
- GPU memory ordering: activation checkpoint save → backward recomputation → optimizer step? [YES/NO]

### Load Conditions
- Retrieved facts / chunks: [count] within safe limits? [PASS/FAIL]
- Memory staging: [bytes] within budget? [PASS/FAIL]
- Batch/dispatch storm risk: [YES/NO]
- GPU VRAM usage: [MB] within 16GB limit with gradient checkpointing? [PASS/FAIL]
- Training step time: [ms] within target? [PASS/FAIL]

### Distillation
- Verdict: PASS / FAIL
- Bugs: [list or "none"]
- Patterns: [reusable insights]
- Assumptions: [what the code assumes — validate these]

### Failure Severity
| Issue | Severity | Action |
|-------|----------|--------|
| [description] | CRITICAL / MAJOR / MINOR | [abort / flag+retry / log+continue] |

### VERDICT: PASS / FAIL
```

### 6.3 Lightweight Trace Template (Mechanical Changes)

For refactors, renames, or moves within an existing pattern:

```markdown
## LIGHTWEIGHT TRACE: [Change Description]

### What Changed
[One-line description]

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| Logic unchanged? | YES | [diff confirms] | PASS |
| All call sites updated? | YES | [grep confirms] | PASS |
| Types/signatures match? | YES | [compiler confirms] | PASS |
| Training data tier unchanged? | YES | [data audit confirms] | PASS |
| Hyperparameter scaling correct? | YES | [lr = base_lr * sqrt(effective_batch_size) confirms] | PASS |

### Assumptions
- [Any assumptions that must hold]

### VERDICT: PASS / FAIL
```

### 6.4 Failure Severity Definitions (Project-Specific)

| Level | Condition | Action |
|-------|-----------|--------|
| CRITICAL | Context exceeds budget after compression, hallucination confirmed (unsourced claim > 5%), race condition in producer/consumer chain, data corruption in retrieved facts, training data contains unverified facts in TT1/TT2, OOM during training due to un-traced memory usage, NaN loss from unstable mixed precision, curriculum violation (TT4 in Phase 1), **prerequisite missing (BLOCKING diagnostic fail)** | STOP. Fix before any code writes. |
| MAJOR | Wrong tier assignment (T1 dropped), missing branch coverage in context assembly, compilation error in retrieval logic, undefined behavior in token counter, resource leak in vector store connection, padding waste > 10%, checkpoint hallucination rate > 2%, gradient checkpointing not enabled for >7B model on 16GB VRAM, QAT layers mismatch inference path | Flag. List specific issues. Retry with guidance. |
| MINOR | Style inconsistency, missing token budget comment, non-optimal chunk size, misleading precedence in relevance scoring, learning rate not scaled with batch size, suboptimal sequence bucketing, missing training log metadata | Log. Continue with main task. Fix later. |

### 6.5 Common Pitfalls (Check These First)

1. **Token count ambiguity:** Does size mean characters, tokens, or bytes? Document and validate. Off-by-factor bugs silently blow budgets.
2. **Tier violation:** Never drop T1 (immutable) facts to make room. If T1 + T2 exceeds budget - reserve, ABORT — do not silently demote.
   - Training extension: Never expose TT4 (unverified) data in Phase 1 or 2. If data loader loads TT4 in early phases, ABORT — this teaches hallucinations.
3. **Retrieval ordering:** Graphify KG traversal MUST run before vector fallback. Reversing this wastes tokens on low-relevance chunks.
4. **Synchronization ordering:** If context assembly and retrieval run concurrently, the barrier MUST be BEFORE assembly reads retrieved facts, not after.
5. **Chunk size limits:** Vector store chunks MUST be ≤ 512 tokens. Larger chunks waste retrieval budget.
6. **Confidence threshold:** KG edges below 0.6 MUST be skipped. Including them risks hallucination from low-confidence facts.
   - Training extension: Training data with KG confidence < 0.8 MUST NOT enter TT1. Data with confidence < 0.6 MUST be TT4 or excluded.
7. **Memory ambiguity in training:** Does batch_size mean per-GPU, per-node, or effective? Document and validate. Off-by-factor bugs cause OOM or under-utilization.
8. **Gradient accumulation mismatch:** Learning rate MUST scale with effective batch size. lr = base_lr * sqrt(effective_batch_size) or linear scaling with warmup. Unscaled LR causes divergence.
9. **Quantization mismatch:** QAT fake quantization MUST match inference quantization exactly (Q4_0, Q4_K, Q8_0, BF16). Mismatch causes accuracy collapse at inference.
10. **Curriculum leak:** If training data is shuffled globally, TT4 samples may leak into Phase 1. Use phase-locked data subsets, not global shuffle + filtering.
11. **Prerequisite blindness:** Do not assume the user has installed Neo4j, FAISS, or spaCy. Run the diagnostic. If missing, STOP and help install.

### 6.6 Debug Point Extraction: From Trace to Instrumentation

Instrument the trace points, not random locations:

| Trace Section | Debug Point | What to Log / Assert |
|---------------|-------------|----------------------|
| **Input State** | Function entry | Log all input values; assert non-null for pointers; assert > 0 for sizes |
| **Flow Diagram — Branch** | Every if / switch / ? : | Log which branch taken; assert condition matches expected |
| **Truth Table — Condition** | Every evaluated expression | Assert actual == expected; if mismatch, dump full state and abort |
| **Completeness — Error State** | Every error path | Assert error path reachable in tests; log error code / reason |
| **Resource Ordering — Producer** | Write completion | Assert write finished; log bytes written / facts retrieved |
| **Resource Ordering — Sync Point** | Barrier / lock / signal | Assert sync object valid; log wait duration (detect deadlocks) |
| **Resource Ordering — Consumer** | Read start | Assert read index / offset within bounds; log first element / fact |
| **Load Conditions** | Pre-dispatch / pre-allocation | Assert count < limit, assert size < budget; log actual vs limit |
| **Training — Data Loader** | Batch construction | Assert padding_tokens / total_tokens < 0.10; assert tier in ALLOWED_TIERS[phase] |
| **Training — Forward Pass** | Loss computation | Assert not NaN; assert loss within expected range; log loss value |
| **Training — Backward Pass** | Gradient check | Assert no inf/NaN gradients; log gradient norm |
| **Training — Checkpoint** | Save trigger | Assert eval_hallucination_rate < 0.02; assert eval_perplexity < threshold; log metrics |
| **Prerequisite — Diagnostic** | Setup check | Assert CHK-01 through CHK-07 pass before DEVELOPMENT MODE entry |

**Debug Point Template:**
```python
# For every row in the Truth Table:
assert condition == expected, f"[file:line] condition={actual} expected={expected}"

# For every branch in the Flow Diagram:
logger.info(f"[file:line] branch: <description> — taken={branch_name}")

# For every Producer → Consumer pair:
assert producer.completed, "[file:line] Producer write not completed before sync"
assert consumer.offset < buffer.size, "[file:line] Consumer read out of bounds"

# For every Load Condition:
assert retrieved_count < MAX_RETRIEVED, f"[file:line] retrieved={retrieved_count} > limit={MAX_RETRIEVED}"
assert total_tokens < budget - reserve, f"[file:line] tokens={total_tokens} > budget={budget - reserve}"

# For training data loader:
assert padding_tokens / total_tokens < 0.10, f"[file:line] padding_waste={padding_tokens/total_tokens:.2%} > 10%"
assert batch_tier in ALLOWED_TIERS[phase], f"[file:line] tier={batch_tier} not allowed in phase={phase}"

# For training forward/backward:
assert not torch.isnan(loss), f"[file:line] NaN loss at step={step}"
assert not torch.isinf(grad_norm), f"[file:line] Inf gradient norm at step={step}"

# For checkpoint evaluation:
assert eval_hallucination < 0.02, f"[file:line] hallucination={eval_hallucination:.2%} > 2% — discarding checkpoint"

# For prerequisite diagnostic:
assert neo4j_client.connected, "[file:line] CHK-01 FAIL — Neo4j not reachable. Entering SETUP MODE."
assert tokenizer is not None, "[file:line] CHK-04 FAIL — Tokenizer not loaded. Entering SETUP MODE."
```

---

## 7. RAG-Specific Coding Rules

### 7.1 Retrieval Functions

Every retrieval function MUST:
1. Accept max_tokens and min_confidence parameters.
2. Return a structured result: List[Fact] where each Fact has subject, predicate, object, source_id, confidence, token_count.
3. Log the retrieval path taken (KG exact / KG hop-1 / KG hop-2 / vector fallback).
4. Assert that sum(f.token_count for f in facts) <= max_tokens.

### 7.2 Context Assembly Functions

Every context assembler MUST:
1. Accept budget and reserve parameters.
2. Return assembled_prompt and metadata (token breakdown per tier).
3. Abort with structured error if compression cannot meet budget.
4. Never silently truncate — truncation is a CRITICAL failure per §6.4.

### 7.3 Generation Wrapper Functions

Every generation wrapper MUST:
1. Accept grounding_facts and enforce citation constraints.
2. Parse the response for claims and cross-reference against grounding_facts.
3. Reject responses where unsourced claims > 5% of total claims.
4. Return the response with a verification_passed boolean.

---

## 8. Training-Specific Coding Rules

### 8.1 Data Curation Functions

Every data curation pipeline MUST:
1. Accept tier_target (TT1/TT2/TT3/TT4) and kg_confidence_threshold parameters.
2. For TT1/TT2: Run every factual claim through Graphify KG lookup. Claims without KG triple or with confidence < threshold are REJECTED for TT1/TT2.
3. For TT3: Apply heuristic filtering (source quality, language model perplexity, duplicate detection).
4. For TT4: Accept unverified data but flag it explicitly. Log the unverified ratio.
5. Return a Dataset object with tier metadata attached to every sample.
6. Assert that len(rejected_for_tier) / len(input) < 0.50 — if >50% of data is rejected, the source quality is too low and MUST be flagged.

### 8.2 Data Loader Functions

Every training data loader MUST:
1. Accept max_sequence_tokens and packing_strategy ("none", "greedy", "optimal") parameters.
2. Implement dynamic bucketing: group sequences by length into buckets of size ≤ max_sequence_tokens. No batch mixes 128-token and 2048-token sequences.
3. Log padding_tokens / total_tokens per batch. Assert this ratio < 0.10.
4. Accept allowed_tiers parameter (list). Assert that every sample in the batch has sample.tier in allowed_tiers.
5. Return input_ids, attention_mask, labels, and tier_metadata for every batch.

### 8.3 Curriculum Scheduler Functions

Every curriculum scheduler MUST:
1. Accept total_steps and phase_boundaries (list of step counts, e.g., [0.30, 0.60, 0.85]).
2. Map each phase to allowed tiers:
   - Phase 1 (0-30%): TT1 only.
   - Phase 2 (30-60%): TT1 + TT2.
   - Phase 3 (60-85%): TT1 + TT2 + TT3.
   - Phase 4 (85-100%): All tiers + RLHF.
3. Return current_allowed_tiers for any given step.
4. Assert that the data loader's allowed_tiers matches current_allowed_tiers at every step. Mismatch is a CRITICAL failure.

### 8.4 Training Loop Functions

Every training loop MUST:
1. Declare effective_batch_size = per_gpu_batch_size * gradient_accumulation_steps * world_size.
2. Scale learning rate: lr = base_lr * scaling_fn(effective_batch_size) where scaling_fn is linear or sqrt with warmup.
3. Enable gradient checkpointing for models > 7B parameters on ≤ 16GB VRAM. Assert this is active via model.gradient_checkpointing_enabled.
4. Use mixed precision (BF16 or FP8) to maximize batch size. Assert torch.cuda.amp or torch.bfloat16 is active.
5. Log every step: loss, gradient norm, learning rate, tokens/sec, GPU memory used, padding waste %.
6. Run evaluation every eval_steps on a KG-held-out validation set. Metrics: perplexity, hallucination rate (vs KG ground truth), token efficiency.
7. Save checkpoint ONLY if eval_hallucination_rate < 0.02 AND eval_perplexity < best_perplexity * 1.05. Otherwise, discard.

### 8.5 Quantization-Aware Training (QAT) Functions

Every QAT implementation MUST:
1. Accept target_quantization (Q4_0, Q4_K, Q8_0, BF16) and match the inference quantization scheme exactly.
2. Implement fake quantization layers that simulate inference quantization during forward but allow gradient flow during backward.
3. Assert that QAT forward output matches inference quantization output (within 1% relative error) on a held-out sample.
4. Apply QAT only in the final 20% of training steps (Phase 4). Assert current_step >= 0.80 * total_steps before enabling QAT layers.
5. Log quantization error (MSE between full-precision and fake-quantized forward) per layer.

### 8.6 On-Device Training Functions (AMD ROCm)

Every on-device training script for AMD hardware MUST:
1. Detect architecture: gfx1201 (RDNA4) or gfx1031 (RDNA2).
2. For RDNA4: Use ROCm 7.1+, enable WMMA/rocWMMA for matrix ops, use hipBLASLt for GEMM.
3. For RDNA2: Use ROCm 7.1+ with gfx1031 targeting, or Vulkan compute if ROCm is unstable. Use Wave32 by default.
4. Trace GPU memory usage before training starts: model_size + activations + gradients + optimizer_states < VRAM_budget. Assert this is true.
5. If OOM occurs, automatic fallback: reduce batch size → enable gradient checkpointing → reduce sequence length → abort with memory trace.
6. Use torch.compile with ROCm backend if available. Log compilation time and verify no graph breaks in the training loop.

---

## 9. Reusable Templates

### Template A: Retrieval Validation
```python
retrieve_facts(query, max_tokens=800, min_confidence=0.6)
│
├── extract_entities(query)
│   ├── entities found → kg_traversal(entities)
│   └── no entities → vector_fallback(query)
│
├── kg_traversal(entities)
│   ├── exact_match(entity) → add facts
│   ├── hop_1_neighbors(entity) → filter confidence >= min_confidence
│   ├── hop_2_path(entity) → filter confidence >= min_confidence
│   └── no matches → vector_fallback(query)
│
├── vector_fallback(query)
│   ├── embed(query)
│   ├── search(top_n=5, threshold=0.75)
│   └── add chunks as facts (confidence=0.7, source_id=vector_store)
│
├── deduplicate(facts)
│   └── entity_resolution() → merge duplicates
│
├── rank_by_relevance(facts)
│   └── hybrid_score = graph_distance * 0.4 + vector_similarity * 0.6
│
├── truncate_to_budget(facts, max_tokens)
│   ├── sum(token_count) <= max_tokens → return all
│   └── sum(token_count) > max_tokens → drop lowest score until under budget
│
└── return facts
```

### Template B: Context Assembly Validation
```python
assemble_context(query, history, facts, budget=4000, reserve=1200)
│
├── system_prompt = load_compressed_system_prompt()
│   ├── file.open(path)
│   │   ├── is_open() == false → return error  [CRITICAL if missing]
│   │   └── is_open() == true ↓
│   └── read → system_tokens = count_tokens(system_prompt)
│
├── query_tokens = count_tokens(query)
│
├── history_tokens = sum(count_tokens(turn) for turn in history)
│   ├── history_tokens > budget * 0.25 → trim_history(history)
│   │   ├── keep last 2 turns (T2) — full
│   │   ├── summarize turns 3-5 (T3) — semantic summary
│   │   ├── extract key facts from turns >5 (T3) — drop raw text
│   │   └── recompute history_tokens
│   └── history_tokens <= budget * 0.25 → use full history
│
├── fact_tokens = sum(f.token_count for f in facts)
│   ├── fact_tokens > budget * 0.30 → rank_and_truncate(facts, top_n=10)
│   └── fact_tokens <= budget * 0.30 → use all facts
│
├── total = system_tokens + query_tokens + history_tokens + fact_tokens
│   ├── total > (budget - reserve) → emergency_compress()
│   │   ├── drop lowest-relevance facts (T4)
│   │   ├── aggressive summarization of T3 history
│   │   ├── truncate oldest T3 summaries to key facts
│   │   └── recompute total
│   │       ├── still > (budget - reserve) → ABORT with error  [CRITICAL]
│   │       └── <= (budget - reserve) → proceed
│   └── total <= (budget - reserve) → proceed
│
└── return assembled_prompt, metadata={system, query, history, facts, total}
```

### Template C: Producer/Consumer Synchronization (Retrieval → Assembly)
```python
Producer (Retrieval) → Consumer (Assembly)
│
├── Producer: retrieve_facts(query)
│   └── writes to shared_facts_buffer
│   └── [WRITE operation]
│
├── Sync point: facts_ready_event.set()
│   ├── BEFORE consumer access?  [YES = CORRECT]
│   └── AFTER consumer access?   [NO = RACE CONDITION]
│
├── Consumer: assemble_context(..., facts=shared_facts_buffer)
│   └── reads from shared_facts_buffer
│   └── [READ operation]
│
└── VERIFY: No circular waits (retrieval waits for assembly, assembly waits for retrieval)
```

### Template D: Training Data Curation Pipeline
```python
curate_dataset(raw_data, target_tier="TT1", kg_confidence_threshold=0.8)
│
├── load_raw_data(source)
│   ├── file.open(path)
│   │   ├── is_open() == false → return error  [CRITICAL]
│   │   └── is_open() == true ↓
│   └── parse → samples
│
├── for each sample in samples:
│   ├── extract_claims(sample.text)
│   │   └── claims = [Fact(subject, predicate, object, ...)]
│   │
│   ├── if target_tier in ["TT1", "TT2"]:
│   │   ├── verify_claims_against_kg(claims)
│   │   │   ├── all claims have KG triple with confidence >= threshold → ACCEPT
│   │   │   ├── some claims missing or low confidence → REJECT for TT1/TT2
│   │   │   └── log rejected_claims
│   │   └──
│   ├── elif target_tier == "TT3":
│   │   ├── heuristic_filter(sample)
│   │   │   ├── source_quality > 0.6 AND perplexity < 50 AND not duplicate → ACCEPT
│   │   │   └── otherwise → REJECT
│   │   └──
│   └── elif target_tier == "TT4":
│       ├── ACCEPT (unverified)
│       └── log unverified_ratio
│
├── assert len(rejected) / len(samples) < 0.50
│   ├── FAIL → flag source quality, abort curation  [CRITICAL]
│   └── PASS → proceed
│
├── assign_tier_metadata(samples)
│   └── each sample.tier = target_tier
│
└── return Dataset(samples)
```

### Template E: Training Loop with Curriculum
```python
train(model, dataset, config)
│
├── initialize
│   ├── detect_gpu_architecture() → gfx1201 or gfx1031
│   ├── enable_mixed_precision(BF16 or FP8)
│   ├── enable_gradient_checkpointing(if model.params > 7B)
│   ├── trace_memory_budget(model, config.batch_size, config.max_seq_len)
│   │   ├── model_size + activations + gradients + optimizer_states < VRAM
│   │   │   ├── PASS → proceed
│   │   │   └── FAIL → reduce batch_size or enable more checkpointing  [CRITICAL]
│   └──
│
├── setup_data_loaders(dataset)
│   ├── create_buckets(sequence_lengths)
│   ├── setup_packing_strategy(config.packing)
│   └── verify_padding_waste < 10% on 100 sample batches
│
├── setup_curriculum_scheduler(total_steps, phase_boundaries)
│   ├── phase_1: allowed_tiers=[TT1]
│   ├── phase_2: allowed_tiers=[TT1, TT2]
│   ├── phase_3: allowed_tiers=[TT1, TT2, TT3]
│   └── phase_4: allowed_tiers=[TT1, TT2, TT3, TT4] + enable_qat
│
├── for step in range(total_steps):
│   ├── current_tiers = curriculum_scheduler.get_allowed_tiers(step)
│   ├── batch = data_loader.next_batch(allowed_tiers=current_tiers)
│   │   ├── assert all(s.tier in current_tiers for s in batch)
│   │   └── assert padding_waste < 0.10
│   │
│   ├── forward_pass(batch)
│   │   ├── with autocast():
│   │   │   └── logits = model(batch.input_ids)
│   │   └── loss = compute_loss(logits, batch.labels)
│   │       ├── assert not torch.isnan(loss)  [CRITICAL]
│   │       └── log loss, tokens/sec, memory_used
│   │
│   ├── backward_pass()
│   │   ├── scaler.scale(loss).backward()
│   │   ├── grad_norm = clip_gradients()
│   │   └── assert not torch.isinf(grad_norm)  [CRITICAL]
│   │
│   ├── optimizer_step()
│   │   ├── scaler.step(optimizer)
│   │   └── scaler.update()
│   │
│   ├── if step % eval_steps == 0:
│   │   ├── eval_results = evaluate_on_kg_held_out(model)
│   │   ├── eval_hallucination = eval_results.hallucination_rate
│   │   ├── eval_perplexity = eval_results.perplexity
│   │   ├── log eval_results
│   │   │
│   │   ├── if eval_hallucination < 0.02 AND eval_perplexity < best * 1.05:
│   │   │   ├── save_checkpoint(model, optimizer, step)
│   │   │   └── best_perplexity = eval_perplexity
│   │   └── else:
│   │       └── discard checkpoint, log failure reason
│   │
│   └── if step >= 0.80 * total_steps:
│       ├── enable_qat_layers(model, target_quantization=config.qat_target)
│       └── assert qat_forward_matches_inference(model, sample)  [MAJOR if fail]
│
└── final_eval = evaluate_on_kg_held_out(model)
    ├── assert final_eval.hallucination_rate < 0.02  [CRITICAL]
    └── export_quantized_model(model, format=config.qat_target)
```

### Template F: Prerequisite Diagnostic Output
```python
run_diagnostic()
│
├── CHK-01: check_graphify_kg()
│   ├── neo4j_client.connect() OR networkx_graph.load()
│   ├── PASS → log "Graphify KG reachable"
│   └── FAIL → log "CHK-01 FAIL: No KG backend. Install: pip install neo4j networkx"
│
├── CHK-02: check_vector_store()
│   ├── faiss_index.load() OR chroma_client.heartbeat()
│   ├── PASS → log "Vector store ready"
│   └── FAIL → log "CHK-02 FAIL: No vector store. Install: pip install faiss-cpu chromadb"
│
├── CHK-03: check_entity_extractor()
│   ├── spacy.load("en_core_web_sm")
│   ├── PASS → log "Entity extractor ready"
│   └── FAIL → log "CHK-03 FAIL: spaCy model missing. Run: python -m spacy download en_core_web_sm"
│
├── CHK-04: check_token_counter()
│   ├── tiktoken.get_encoding("cl100k_base") OR transformers.AutoTokenizer
│   ├── PASS → log "Token counter ready"
│   └── FAIL → log "CHK-04 FAIL: No tokenizer. Install: pip install tiktoken transformers"
│
├── CHK-05: check_embedding_model()
│   ├── sentence_transformers.SentenceTransformer("BAAI/bge-small-en")
│   ├── PASS → log "Embedding model ready"
│   └── FAIL → log "CHK-05 FAIL: No embedding model. Install: pip install sentence-transformers"
│
├── CHK-06: check_llm_backend()
│   ├── llama_cpp.Llama(model_path=...) OR openai_client.api_key
│   ├── PASS → log "LLM backend ready"
│   └── FAIL → log "CHK-06 FAIL: No LLM backend. Install: pip install llama-cpp-python OR set OPENAI_API_KEY"
│
├── CHK-07: check_training_env()
│   ├── import torch; torch.cuda.is_available() OR torch.version.hip
│   ├── PASS → log "Training environment ready"
│   └── FAIL → log "CHK-07 FAIL: PyTorch/ROCm not installed. See: https://pytorch.org/get-started/locally/"
│
├── CHK-08: check_graphify_schema()
│   ├── kg_schema.has_nodes() AND kg_schema.has_edges()
│   ├── PASS → log "Graphify schema populated"
│   └── WARNING → log "CHK-08 WARNING: KG schema empty. Retrieval will return zero facts. Run schema init."
│
├── CHK-09: check_training_data_dirs()
│   ├── os.path.exists("data/raw/") AND os.path.exists("data/tt1/")
│   ├── PASS → log "Training data directories ready"
│   └── WARNING → log "CHK-09 WARNING: Training data dirs missing. Create: mkdir -p data/{raw,processed,tt1,tt2,tt3,tt4}"
│
├── CHK-10: check_rocm_vulkan()
│   ├── os.path.exists("/opt/rocm/bin/rocminfo") OR os.path.exists("C:\VulkanSDK\")
│   ├── PASS → log "ROCm/Vulkan SDK detected"
│   └── WARNING → log "CHK-10 WARNING: ROCm/Vulkan not detected. GPU acceleration unavailable. Install from amd.com"
│
└── compile_results()
    ├── ALL BLOCKING PASS → return DEVELOPMENT_MODE
    └── ANY BLOCKING FAIL → return SETUP_MODE, output_guide()
```

---

## 10. User Preferences (Project-Specific)

- Do not build, run, or execute any code, tests, or workloads without explicit user confirmation.
- Write the full code first; debug/run only after the user confirms.
- When the user requests a durable behavior change (e.g., new tier policy, different budget defaults, new training phase boundaries, different QAT target), record it in the nearest AGENTS.md or child AGENTS.md.
- When the user requests training on specific hardware (RX 9070 XT, RX 6700 XT), the trace MUST include memory verification for that exact GPU before any training code is written.
- **When the user requests ANY feature, the agent MUST run the prerequisite diagnostic first. If BLOCKING checks fail, the agent MUST enter SETUP MODE and guide installation. NO exceptions.**

---

*CLAUDE.md v3.0 | Graphify Token Optimization + Training + Self-Diagnostics | Behavioral Contract*
