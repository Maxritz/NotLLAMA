# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

- Do not build, run, or execute any code, tests, or workloads without explicit user confirmation.
- Write the full code first; debug/run only after the user confirms.

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Truth/False Logic Validation (MANDATORY)

> **Rule: No code writes without a trace first. The trace is faster than the debugger.**

This framework catches logic errors, state mishandling, and resource ordering bugs *before* implementation — through structured static analysis. It also tells you exactly where to instrument your code when things go wrong at runtime.

### Proof of Value

Real bugs caught by this framework in production codebases:

| Bug | Caught By | Cost If Missed |
|-----|-----------|----------------|
| File stream `bad_alloc` crash | Missing `is_open()` branch in flow diagram | Unhandled exception / crash on bad path |
| Array index off-by-2× | Truth table showed computed count vs actual buffer bounds | Buffer overflow → data corruption |
| Operator precedence "bug" | Mathematical proof in truth table showed working code was correct | Would have introduced a bug by "fixing" working code |
| Race condition in producer/consumer | Resource ordering DAG showed missing synchronization | Intermittent corruption / crash |

**The trace finds bugs. It confirms correctness. And it tells you where to attach debug instrumentation.**

### Calibration: Full vs Lightweight Trace

Not every change needs exhaustive analysis. Use risk calibration:

| Change Type | Trace Level | Reasoning |
|-------------|-------------|-----------|
| New algorithm, new I/O path, new data format, concurrency logic, memory/buffer resize, security boundary | **FULL** | Invisible or high-impact bugs (corruption, leaks, races, overflows) |
| Refactor within existing pattern (rename, const, extract function, move) | **LIGHTWEIGHT** | Verify logic unchanged |
| Comments, docs, build flags, config values | **NONE** | No runtime effect |

**Default to FULL if unsure.** A 10-minute trace beats a 3-hour debug session.

### Full Trace Format (Mandatory for New Logic)

```markdown
## TRACE: [Feature Name]

### Input State
| Variable | Value | Source |
|----------|-------|--------|
| x        | 42    | caller |
| format   | RAW   | config |

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
if (x > 0):
  TRUE → do A
  FALSE → do B
```

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|
| x > 0     | TRUE     | TRUE   | PASS  |
| format == RAW | TRUE | TRUE   | PASS  |

### Completeness Check
- Branch coverage: ALL branches reachable and tested? [YES/NO]
- State completeness: ALL variable values handled (including error states)? [YES/NO]
- Goal reachability: EVERY valid input reaches output without silent failure? [YES/NO]

### Resource Ordering
- Producer → Consumer: [pair]
- DAG ordering verified (write before read)? [YES/NO]
- Synchronization/barrier placement: BEFORE consumer access? [YES/NO]
- Deadlock/circular wait risk: [NONE/LOW/HIGH]

### Load Conditions
- Compute units / threads: [count] within safe limits? [PASS/FAIL]
- Memory staging: [bytes] within budget? [PASS/FAIL]
- Batch/dispatch storm risk: [YES/NO]

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

### Lightweight Trace Format (Mechanical Changes)

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

### Assumptions
- [Any assumptions that must hold]

### VERDICT: PASS / FAIL
```

### Failure Severity Definitions

| Level | Condition | Action |
|-------|-----------|--------|
| **CRITICAL** | Buffer overflow possible, race condition confirmed, data corruption, unhandled exception, security boundary violation, infinite loop | **STOP.** Fix before any code writes. |
| **MAJOR** | Wrong format conversion, missing branch coverage, compilation error, undefined behavior, resource leak | **Flag.** List specific issues. Retry with guidance. |
| **MINOR** | Style inconsistency, missing documentation, non-optimal performance, misleading precedence | **Log.** Continue with main task. Fix later. |

### Common Pitfalls (Check These First)

1. **File stream state**: Always check `is_open()` before `tellg()` / `read()`. A failed stream's `tellg()` returns `-1` → cast to unsigned = `SIZE_MAX` → massive allocation → crash.
2. **Size ambiguity**: Does `size` mean compressed bytes, element count, or byte count? Document and validate. Off-by-factor bugs are silent killers.
3. **Operator precedence**: In C-family languages, `|` has lower precedence than `<<` and `-`. Always parenthesize bitwise arithmetic: `(a | (b << 4)) - c`, not `a | (b << 4) - c`.
4. **Synchronization ordering**: Lock/barrier/semaphore must be **BEFORE** consumer access, not after. Sequence: `producer write` → `sync point` → `consumer read`.
5. **Resource limits**: Cap batch sizes, thread counts, and memory allocations to known safe thresholds. Chunk if exceeded.
6. **Temporary buffer sizing**: Size buffers for the operation, not the entire dataset. Temporary staging should not persist beyond the operation scope.

### Debug Point Extraction: From Trace to Instrumentation

The trace identifies exactly where reality can diverge from expectation. Those points are your debug instrumentation targets. Do not add logs or breakpoints randomly — instrument the trace points.

| Trace Section | Debug Point | What to Log / Assert |
|---------------|-------------|----------------------|
| **Input State** | Function entry | Log all input values; assert non-null for pointers, assert > 0 for sizes |
| **Flow Diagram — Branch** | Every `if` / `switch` / `? :` | Log which branch was taken; assert the condition matches expected |
| **Truth Table — Condition** | Every evaluated expression | Assert `actual == expected`; if mismatch, dump full state and abort |
| **Completeness — Error State** | Every error path | Assert the error path is reachable in tests; log error code / reason |
| **Resource Ordering — Producer** | Write completion | Assert write finished successfully; log bytes written / checksum |
| **Resource Ordering — Sync Point** | Barrier / lock / signal | Assert sync object is valid; log wait duration (detect deadlocks) |
| **Resource Ordering — Consumer** | Read start | Assert read index / offset within bounds; log first element / hash |
| **Load Conditions** | Pre-dispatch / pre-allocation | Assert count < limit, assert size < budget; log actual vs limit |

**Debug Point Template:**
```
For every row in the Truth Table:
  → Add: ASSERT(condition == expected) or LOG("[file:line] condition=%d expected=%d", actual, expected)

For every branch in the Flow Diagram:
  → Add: LOG("[file:line] branch: <description>") at the taken path

For every Producer → Consumer pair:
  → Add: ASSERT(producer.completed) before sync point
  → Add: ASSERT(consumer.offset < buffer.size) at read start

For every Load Condition:
  → Add: ASSERT(load < threshold) before dispatch/allocation
```

### Reusable Templates

**Template A: File I/O Validation**
```
loadFile(path)
│
├── file.open(path)
│   ├── is_open() == false → return error  [CRITICAL if missing]
│   └── is_open() == true ↓
│
├── file.parse() or file.tellg()
│   ├── parse_error / tellg() == -1 → return error  [CRITICAL if missing]
│   └── success ↓
│
├── size = (size_t)file.tellg()
│   ├── size <= 0 → return error  [MAJOR if missing]
│   └── size > 0 ↓
│
├── buffer.resize(size)
│   ├── bad_alloc → return error  [CRITICAL if missing]
│   └── success ↓
│
└── file.read(buffer)
    ├── bytes_read != size → return error  [MAJOR if missing]
    └── success → return buffer
```

**Template B: Computed Size Validation**
```
calculateElements(byteSize, format)
│
├── What does byteSize represent?
│   ├── Compressed bytes → nElements = decompressRatio(byteSize)
│   ├── Element count → nElements = byteSize
│   └── Unknown → abort / assert  [CRITICAL if ambiguous]
│
├── VERIFY: Does nElements match actual buffer bounds?
│   ├── nElements * elemSize > bufferSize → BUG  [CRITICAL]
│   └── nElements * elemSize <= bufferSize → OK
│
└── VERIFY: Does consumer use correct count?
```

**Template C: Producer/Consumer Synchronization**
```
Producer → Consumer Chain
│
├── Producer writes to shared resource
│   └── [WRITE operation]
│
├── Sync point (barrier / lock / fence / signal)
│   ├── BEFORE consumer access?  [YES = CORRECT]
│   └── AFTER consumer access?   [NO = RACE CONDITION]
│
├── Consumer reads from shared resource
│   └── [READ operation]
│
└── VERIFY: No circular waits (A waits for B, B waits for A)
```

### RAG / Knowledge Graph Distillation

Every trace produces durable knowledge. Extract these fields:

| Field | Extract? | Why |
|-------|----------|-----|
| Verdict (PASS/FAIL) | **YES** | Node label for search |
| Bug description (one-liner) | **YES** | Primary search key |
| Root cause pattern | **YES** | Reusable rule — e.g., "size ambiguity" |
| Assumption that failed | **YES** | Prevents repeat — e.g., "Assumed files always exist" |
| Severity + action taken | **YES** | Teaches escalation rules |
| Debug points extracted | **YES** | Reusable instrumentation pattern |
| Full truth table | **NO** | Too noisy; keep in version control |
| Full code | **NO** | Already in source |

**Graph Node Format:**
```json
{
  "id": "trace_[component]_[bug]_[date]",
  "type": "validation",
  "component": "[component name]",
  "verdict": "FAIL",
  "bug": "[one-line description]",
  "pattern": "[reusable rule]",
  "assumption_broken": "[what was assumed]",
  "severity": "CRITICAL",
  "fix": "[what was done]",
  "debug_points": ["ASSERT(...)", "LOG(...)"],
  "related": ["trace_[other]"]
}
```

**When to Feed the Graph:**

| Trigger | Action |
|---------|--------|
| Trace finds a CRITICAL bug | **Immediate** — prevents the next engineer from repeating it |
| Trace confirms a false positive | **Immediate** — prevents future "fixes" of working code |
| Lightweight trace on refactor | **Skip** — no new knowledge generated |
| Trace finds only MINOR issues | **Batch** — feed at end of week |

**The trace is faster than the debugger.**

## Child DOX Index

<!-- List your child AGENTS.md files here. Example:
- `src/` — Source code ownership and contracts
- `tools/` — Tooling and scripts
-->
