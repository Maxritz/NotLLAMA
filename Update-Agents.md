# AGENTS.md — Graphify Token Optimization Project
## Root DOX Contract: Minimize Token Burn · Maximize Output Quality · Prevent Hallucinations

**Status:** Binding work contract for this project subtree.  
**Scope:** All RAG pipeline design, context management, retrieval logic, LLM interaction code, and model training pipelines.  
**Parent:** None (root). Child docs must be created per durable boundary.  
**Self-Diagnostic Mode:** ENABLED — Agent MUST verify prerequisites before any implementation work.

---

## 1. Purpose

This project optimizes LLM inference (cloud and local) and model training by minimizing token consumption through structured retrieval, aggressive context compression, hallucination-resistant generation, and grounded training data curation. Graphify (Knowledge Graph + Vector Store hybrid) is the canonical retrieval backend and ground truth source for training data verification. Every agent working this tree must enforce the rules below.

---

## 2. Ownership

| Boundary | Owner | Contact |
|----------|-------|---------|
| Root pipeline design | AGENTS.md (this doc) | — |
| Graphify KG schema | `graphify/AGENTS.md` | Child DOX |
| Context trimmer / compressor | `context/AGENTS.md` | Child DOX |
| Hallucination verifier | `verify/AGENTS.md` | Child DOX |
| Prompt assembly | `prompts/AGENTS.md` | Child DOX |
| Metrics / feedback loop | `metrics/AGENTS.md` | Child DOX |
| Local inference optimization | `local/AGENTS.md` | Child DOX |
| Training pipeline | `training/AGENTS.md` | Child DOX |
| Data curation / KG-grounded dataset | `training/data/AGENTS.md` | Child DOX |

---

## 3. Self-Diagnostic: Prerequisites Check (MANDATORY FIRST STEP)

**Rule: Before writing ANY code, the agent MUST run a prerequisites diagnostic. If prerequisites are missing, STOP implementation and guide the user through installation.**

### 3.1 Diagnostic Checklist

Before any task begins, verify the following infrastructure exists. If any check fails, the agent MUST NOT proceed with implementation. Instead, it MUST output the diagnostic results and a setup guide.

| Check | What to Verify | If Missing | Severity |
|-------|--------------|------------|----------|
| **CHK-01** | Graphify Knowledge Graph backend (Neo4j, NetworkX, or ArangoDB) | RAG cannot ground facts. Hallucination prevention impossible. | **BLOCKING** |
| **CHK-02** | Vector Store (FAISS, Chroma, pgvector, or Weaviate) | Fallback retrieval unavailable. KG misses become failures. | **BLOCKING** |
| **CHK-03** | Entity Extractor (spaCy, Flair, or lightweight NER model) | Queries cannot be mapped to KG entities. Retrieval precision collapses. | **BLOCKING** |
| **CHK-04** | Token Counter (tiktoken, HuggingFace tokenizer, or custom) | Token budgets cannot be enforced. Context overflow inevitable. | **BLOCKING** |
| **CHK-05** | Embedding Model (Sentence-Transformers, BGE, or E5) | Vector search cannot compute similarity. KG fallback broken. | **BLOCKING** |
| **CHK-06** | LLM Backend (local: llama.cpp/ollama/vllm; cloud: OpenAI/Anthropic API key) | No generation target. Pipeline has no output stage. | **BLOCKING** |
| **CHK-07** | Python environment with PyTorch / JAX / ROCm (for training) | Training loops cannot execute. | **BLOCKING for training tasks** |
| **CHK-08** | Graphify schema initialized (nodes, edges, confidence properties) | KG exists but is empty. Retrieval returns zero facts. | **WARNING** |
| **CHK-09** | Training data directory structure (`raw/`, `processed/`, `tt1/`, `tt2/`, `tt3/`, `tt4/`) | Training pipeline has no input staging. | **WARNING for training tasks** |
| **CHK-10** | ROCm 7.1+ or Vulkan SDK 1.4.35+ (for AMD GPU training/inference) | Cannot leverage RDNA4/RDNA2 hardware acceleration. | **WARNING** |

### 3.2 Diagnostic Output Format

If any BLOCKING check fails, the agent MUST output:

```
╔══════════════════════════════════════════════════════════════════╗
║  PREREQUISITES MISSING — IMPLEMENTATION BLOCKED                ║
╠══════════════════════════════════════════════════════════════════╣
║  Failed Checks: [CHK-01, CHK-03, CHK-05]                        ║
║                                                                  ║
║  Required Before Proceeding:                                     ║
║  1. Graphify KG backend (Neo4j/NetworkX) — CHK-01               ║
║     Why: Without a knowledge graph, facts cannot be verified.     ║
║     Install: pip install neo4j networkx                          ║
║     Or: docker run -p 7474:7474 -p 7687:7687 neo4j:latest       ║
║                                                                  ║
║  2. Entity Extractor (spaCy) — CHK-03                           ║
║     Why: Queries need entity extraction before KG traversal.    ║
║     Install: pip install spacy && python -m spacy download en_core_web_sm ║
║                                                                  ║
║  3. Embedding Model — CHK-05                                    ║
║     Why: Vector similarity search requires embeddings.          ║
║     Install: pip install sentence-transformers                   ║
║     Or: Use HuggingFace transformers with BGE model              ║
╚══════════════════════════════════════════════════════════════════╝

Next Step: Install the above, then re-run diagnostic.
Agent will resume implementation once all BLOCKING checks pass.
```

### 3.3 Setup Mode vs. Development Mode

The agent operates in two modes:

| Mode | Trigger | Behavior |
|------|---------|----------|
| **SETUP MODE** | Any BLOCKING prerequisite missing | Agent guides installation, provides copy-paste commands, explains why each component is needed. NO code implementation. |
| **DEVELOPMENT MODE** | All BLOCKING prerequisites satisfied | Agent proceeds with implementation per CLAUDE.md rules. |

**Transition Rule:** The user MUST explicitly confirm "setup complete" or "all prerequisites installed" before the agent exits SETUP MODE and enters DEVELOPMENT MODE. The agent MUST NOT assume setup is done without user confirmation.

### 3.4 Conditional Implementation Paths

If the user asks for a feature but prerequisites are partially met, the agent MUST:

1. **Assess what exists:** Run the diagnostic checklist.
2. **Identify gaps:** List missing BLOCKING and WARNING items.
3. **Propose staged implementation:**
   - **Stage A (Setup):** Install missing prerequisites. Provide scripts, Docker commands, or pip installs.
   - **Stage B (Scaffold):** Create directory structure, config files, and stub interfaces for missing components.
   - **Stage C (Implement):** Full implementation only after Stage A and B are confirmed complete.
4. **Offer help:** "Do you want me to generate the setup scripts for the missing components?"

**Example:**
```
User: "Build the context trimmer"
Agent: "CHK-04 (Token Counter) is missing. I cannot enforce token budgets without a tokenizer.

I can:
1. Generate a setup script to install tiktoken + HuggingFace tokenizer
2. Create a minimal tokenizer stub using regex/word-count as fallback
3. Wait for you to install the real tokenizer

Which would you prefer?"
```

---

## 4. Global Rules (No Child May Weaken These)

### 4.1 Token Budget Is Non-Negotiable

- Every prompt assembled for an LLM call MUST declare a `max_context_tokens` and `response_reserve` before generation.
- The assembled context (system + query + retrieved + history) MUST NOT exceed `max_context_tokens - response_reserve`.
- If compression cannot bring the prompt under budget, the call MUST abort with a structured error — never silently truncate.
- Default budget for 4K models: `max_context_tokens=4000`, `response_reserve=1200`.
- **Training extension:** Every training batch MUST declare `max_sequence_tokens` and `gradient_accumulation_steps` to stay within GPU memory budget. Sequence packing MUST minimize padding token waste.

### 4.2 Retrieved Facts Must Be Grounded

- Every claim in a generated response MUST trace to a retrieved fact triple (subject, predicate, object, source_id, confidence).
- Claims without a source triple MUST be flagged with an uncertainty marker (`[unverified]`) or omitted.
- The verifier (see §6) MUST reject responses where unsourced claims exceed 5% of total claims.
- **Training extension:** Every training example labeled as "factual" MUST be verified against Graphify before inclusion. Synthetic training data generated by LLMs MUST be fact-checked against the KG before entering the training set.

### 4.3 Context Is Tiered — Never Flat

Conversation history and retrieved context MUST be organized into tiers. No flat concatenation.

| Tier | Priority | Drop Policy | Max Token Share |
|------|----------|-------------|-----------------|
| T1 — Immutable | 1 | Never | 30% |
| T2 — High Value | 2 | Summarize before drop | 25% |
| T3 — Compressible | 3 | Summarize → extract → drop | 15% |
| T4 — Droppable | 4 | Drop first | 0% (overflow) |

**Training extension:** Training data is tiered by reliability and complexity:

| Training Tier | Source | Verification | Usage |
|---------------|--------|--------------|-------|
| TT1 — Gold | Human-curated, KG-verified | 100% fact-checked | Core pre-training, never dropped |
| TT2 — Silver | High-quality synthetic, KG-grounded | Spot-checked ≥ 90% | Fine-tuning, summarizable |
| TT3 — Bronze | General web crawl, vector-filtered | Heuristic filtering | Continual pre-training, compressible |
| TT4 — Unverified | Raw synthetic, unfiltered | None | Exploration / RL only, droppable on memory pressure |

### 4.4 Graphify Is Primary Retrieval and Ground Truth

- Vector search is a FALLBACK for Graphify KG misses only.
- If a query entity exists in the KG, traversal MUST be attempted before vector search.
- Hybrid mode (KG + vector) is allowed only when the query classification is "complex reasoning" or "multi-hop".
- **Training extension:** Graphify is the canonical ground truth for training data verification. Any training example claiming a fact MUST have a KG triple supporting it, or it MUST be labeled as TT3/TT4 (unverified).

### 4.5 No Code Without a Trace

Per parent DOX: **No implementation work begins without a Truth/False Logic Validation trace.**
- FULL trace for: new algorithms, new I/O paths, new data formats, concurrency, memory/buffer resize, security boundaries, training loops, data loaders, loss functions, gradient paths.
- LIGHTWEIGHT trace for: refactors, renames, moves within existing patterns, hyperparameter tweaks, learning rate schedules.
- NONE for: comments, docs, build flags, config values, training logs, tensorboard summaries.
- The trace MUST be written in the same session as the code and committed to the task record.

### 4.6 Hallucination Prevention Starts at Training

- **Data Curation Rule:** Training datasets MUST be filtered through Graphify fact-checking before use. Facts not in KG or marked low-confidence MUST be excluded from TT1/TT2.
- **Synthetic Data Rule:** LLM-generated training examples MUST be run through the same verifier as inference outputs. Unverified synthetic data goes to TT4 only.
- **Curriculum Rule:** Training MUST follow curriculum: TT1 → TT2 → TT3 → TT4. Never expose model to unverified data before it has learned grounded patterns.
- **Evaluation Rule:** Training checkpoints MUST be evaluated on hallucination rate (against KG ground truth), not just perplexity or loss. A model with low loss but high hallucination is a FAILED checkpoint.

### 4.7 Agent Must Help With Setup

- If the user is missing prerequisites, the agent MUST generate setup scripts, Docker Compose files, or step-by-step installation guides.
- The agent MUST explain WHY each component is needed (link to the diagnostic check).
- The agent MUST provide copy-paste ready commands (pip, conda, docker, apt, etc.).
- The agent MUST offer to create a `setup/` directory with installation scripts and a `SETUP.md` guide.
- The agent MUST NOT say "install X and come back" without providing the exact commands to do so.

---

## 5. Local Contracts (Per-Module Rules)

### 5.1 Query Processing (Inference)

1. **Entity Extraction First:** Every query MUST be run through an entity extractor before retrieval. No raw query string goes to the vector store.
2. **Query Classification:** Classify as `factual`, `opinion`, `creative`, `procedural`. Retrieval strategy depends on this classification.
3. **Query Embedding:** If vector search is needed, the query embedding MUST use the same model as the chunk embeddings (dimension lock).

### 5.2 Retrieval Orchestration (Inference)

1. **KG Traversal Order:**
   - Exact entity match → 1-hop neighbors → 2-hop path → vector fallback.
   - Each hop MUST include edge confidence. Edges below 0.6 are skipped.
2. **Vector Search Constraints:**
   - Top-N ≤ 5 chunks per query.
   - Chunk size ≤ 512 tokens.
   - Relevance threshold ≥ 0.75 (cosine similarity).
3. **Deduplication:** Retrieved facts from KG and vector store MUST be deduplicated by entity resolution before assembly. Duplicate facts waste tokens.

### 5.3 Context Assembly (Inference)

1. **Assembly Order:**
   ```
   system_prompt (compressed, hard-coded)
   + current_query (entity-mapped)
   + T1_retrieved_facts (structured triples)
   + T2_recent_history (last 2 turns, full)
   + T3_older_history (summarized or key-fact extracted)
   + working_memory (structured key-value, if any)
   ```
2. **Compression Pipeline:**
   - Entity abbreviation map (e.g., "The United States of America" → "USA").
   - Redundant phrase removal (e.g., "it is important to note that" → delete).
   - Semantic summarization of T3 history (lightweight model, ≤ 128 tokens per summary).
3. **Emergency Compression:** If total > budget - reserve:
   - Step 1: Drop T4 (lowest-relevance vector chunks).
   - Step 2: Aggressive summarization of T3.
   - Step 3: Truncate oldest T3 summaries to key facts only.
   - Step 4: If still over budget, ABORT with error — do not drop T1 or T2.

### 5.4 Generation Constraints (Inference)

1. **Citation Requirement:** Every factual claim in the response MUST include a citation to its source triple ID.
2. **Uncertainty Flagging:** If a claim is generated without a KG source, it MUST be wrapped in `[unverified]`.
3. **Self-Consistency Check:** After generation, the response MUST be parsed for claims and cross-referenced against the retrieved context. Mismatches trigger regeneration with stricter grounding.

### 5.5 Training Pipeline Contracts

1. **Data Verification Pipeline:**
   ```
   raw_data → entity_extractor → kg_lookup → confidence_filter → tier_assignment → training_set
   ```
   - `kg_lookup`: Every factual claim in the data MUST match a KG triple or be flagged.
   - `confidence_filter`: Only facts with confidence ≥ 0.8 enter TT1/TT2.
   - `tier_assignment`: Based on verification status and source quality.

2. **Token-Efficient Training:**
   - **Sequence Packing:** Training sequences MUST be packed to `max_sequence_tokens` with minimal padding. Padding waste MUST be < 10% of batch tokens.
   - **Dynamic Bucketing:** Batches MUST group sequences of similar length. No batch contains both 128-token and 2048-token sequences.
   - **Gradient Accumulation:** If batch size is limited by memory, gradient accumulation steps MUST be declared and accounted for in learning rate scaling.

3. **Curriculum Learning Schedule:**
   - Phase 1 (0-30% steps): TT1 only — Gold facts, KG-verified.
   - Phase 2 (30-60% steps): TT1 + TT2 — Add high-quality synthetic, KG-grounded.
   - Phase 3 (60-85% steps): TT1 + TT2 + TT3 — Add general web data, heuristic-filtered.
   - Phase 4 (85-100% steps): All tiers + RLHF — Introduce TT4 for exploration, with KG verifier as reward model.

4. **Checkpoint Evaluation:**
   - Every checkpoint MUST pass: `hallucination_rate < 2%` on KG-held-out set.
   - Every checkpoint MUST pass: `token_efficiency > 80%` (measured as useful tokens / total tokens in validation batches).
   - Checkpoints failing either criterion MUST be discarded, not promoted.

5. **Mixed Precision & Quantization:**
   - Training MUST use BF16 or FP8 (if hardware supports) to maximize batch size within GPU memory.
   - Quantization-Aware Training (QAT) for target inference formats (Q4, Q8, BF16) MUST be applied in the final 20% of training.
   - Gradient checkpointing MUST be enabled for models > 7B parameters to fit in 16GB VRAM.

6. **On-Device Training (Local AI):**
   - Training runs on AMD RX 9070 XT (RDNA4) MUST use ROCm 7.1+ with WMMA/rocWMMA for matrix ops.
   - Training runs on RDNA2 (RX 6700 XT) MUST use Vulkan compute or ROCm with gfx1031 targeting.
   - Memory budget: 16GB VRAM → max batch size determined by trace, not guess. Use activation checkpointing aggressively.

### 5.6 Feedback Loop (Inference + Training)

1. **User Feedback:** Every response MUST capture 👍 / 👎 feedback.
2. **Retrieval Accuracy:** Log which retrieved facts were cited in the response. Facts never cited get their edge confidence decayed.
3. **Gap Detection:** Queries that miss the KG (vector fallback) MUST be logged as "knowledge gaps" for future KG population.
4. **Training Feedback:** Failed inference responses (high hallucination, user 👎) MUST be added to the training gap log. These become priority synthetic data generation targets.

---

## 6. Verification Requirements

### 6.1 Pre-Deployment Checks (Inference)

Before any module is considered complete, verify:

- [ ] Token budget enforcement: Prompt never exceeds `max_context_tokens - response_reserve`.
- [ ] Hallucination rate: < 2% on held-out test set (manual audit + automated fact-checking).
- [ ] Retrieval precision: > 90% of retrieved facts are cited in the response.
- [ ] Context efficiency: > 80% of tokens in the assembled prompt are used in the response (no dead weight).
- [ ] KG coverage: > 85% of factual queries hit the KG (not vector fallback).

### 6.2 Pre-Training Checks (Training)

Before any training run begins, verify:

- [ ] Data verification: 100% of TT1 data, ≥ 90% of TT2 data has been KG-fact-checked.
- [ ] Token efficiency: Padding waste < 10% across all training batches.
- [ ] Curriculum integrity: Training script enforces phase-locked tier exposure (no TT4 in Phase 1).
- [ ] Memory trace: GPU memory usage trace confirms no OOM at max batch size + gradient checkpointing.
- [ ] Checkpoint criteria: Evaluation script measures hallucination rate and token efficiency, not just loss.

### 6.3 Runtime Assertions (Inference)

These MUST be present in production code:

```python
# At context assembly entry
assert total_tokens <= max_context_tokens - response_reserve,     "Context exceeds budget — aborting"

# At retrieval exit
assert len(retrieved_facts) > 0 or query_classification != "factual",     "Factual query returned zero facts — falling back to vector"

# At generation exit
assert unsourced_claims / total_claims <= 0.05,     "Too many unsourced claims — rejecting response"
```

### 6.4 Runtime Assertions (Training)

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
```

---

## 7. Child DOX Index

Create child AGENTS.md files when a folder becomes a durable boundary:

- `graphify/` — KG schema, traversal logic, entity resolution, confidence scoring
- `context/` — Tier trimming, summarization, compression, budget enforcement
- `verify/` — Fact extraction, claim verification, uncertainty flagging, regeneration triggers
- `prompts/` — Prompt templates, entity mapping, citation formatting, system prompt compression
- `metrics/` — Retrieval accuracy logging, user feedback ingestion, gap detection, edge confidence decay
- `local/` — Quantized embedding inference, ONNX runtime config, query caching, batch retrieval
- `training/` — Training loop orchestration, curriculum scheduler, checkpoint manager, mixed precision config
- `training/data/` — Data curation pipeline, KG verification filter, synthetic data generator, tier assignment logic
- `training/models/` — Model architecture, quantization-aware layers, gradient checkpointing hooks
- `training/eval/` — Hallucination evaluator, token efficiency benchmark, KG-held-out test sets

---

## 8. Closeout Rule

Every task ending in this subtree MUST:
1. Re-check changed paths against the DOX chain (this doc + any child AGENTS.md).
2. Update the nearest owning AGENTS.md if the change affects purpose, scope, contracts, workflows, or artifacts.
3. Refresh every affected Child DOX Index.
4. Remove stale or contradictory text immediately.
5. Run verification checks (§6.1, §6.2) when relevant.
6. Report any docs intentionally left unchanged and why.

---

*AGENTS.md v3.0 | Graphify Token Optimization + Training + Self-Diagnostics | Binding DOX Contract*
