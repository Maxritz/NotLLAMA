# Truth/False Logic Validation
## A Pre-Code Static Analysis Framework

> **Rule: No code writes without a trace first. The trace is faster than the debugger.**

This framework catches logic errors, state mishandling, and resource ordering bugs *before* implementation — through structured static analysis. It also tells you exactly where to instrument your code when things go wrong at runtime.

---

## Proof of Value

Real bugs caught by this framework in production codebases:

| Bug | Caught By | Cost If Missed |
|-----|-----------|----------------|
| File stream `bad_alloc` crash | Missing `is_open()` branch in flow diagram | Unhandled exception / crash on bad path |
| Array index off-by-2× | Truth table showed computed count vs actual buffer bounds | Buffer overflow → data corruption |
| Operator precedence "bug" | Mathematical proof in truth table showed working code was correct | Would have introduced a bug by "fixing" working code |
| Race condition in producer/consumer | Resource ordering DAG showed missing synchronization | Intermittent corruption / crash |

**The trace finds bugs. It confirms correctness. And it tells you where to attach debug instrumentation.**

---

## Calibration: Full vs Lightweight Trace

Not every change needs exhaustive analysis. Use risk calibration:

| Change Type | Trace Level | Reasoning |
|-------------|-------------|-----------|
| New algorithm, new I/O path, new data format, concurrency logic, memory/buffer resize, security boundary | **FULL** | Invisible or high-impact bugs (corruption, leaks, races, overflows) |
| Refactor within existing pattern (rename, const, extract function, move) | **LIGHTWEIGHT** | Verify logic unchanged |
| Comments, docs, build flags, config values | **NONE** | No runtime effect |

**Default to FULL if unsure.** A 10-minute trace beats a 3-hour debug session.

---

## Full Trace Format (Mandatory for New Logic)

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

---

## Lightweight Trace Format (Mechanical Changes)

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

---

## Failure Severity Definitions

| Level | Condition | Action |
|-------|-----------|--------|
| **CRITICAL** | Buffer overflow possible, race condition confirmed, data corruption, unhandled exception, security boundary violation, infinite loop | **STOP.** Fix before any code writes. |
| **MAJOR** | Wrong format conversion, missing branch coverage, compilation error, undefined behavior, resource leak | **Flag.** List specific issues. Retry with guidance. |
| **MINOR** | Style inconsistency, missing documentation, non-optimal performance, misleading precedence | **Log.** Continue with main task. Fix later. |

---

## Common Pitfalls (Check These First)

1. **File stream state**: Always check `is_open()` before `tellg()` / `read()`. A failed stream's `tellg()` returns `-1` → cast to unsigned = `SIZE_MAX` → massive allocation → crash.
2. **Size ambiguity**: Does `size` mean compressed bytes, element count, or byte count? Document and validate. Off-by-factor bugs are silent killers.
3. **Operator precedence**: In C-family languages, `|` has lower precedence than `<<` and `-`. Always parenthesize bitwise arithmetic: `(a | (b << 4)) - c`, not `a | (b << 4) - c`.
4. **Synchronization ordering**: Lock/barrier/semaphore must be **BEFORE** consumer access, not after. Sequence: `producer write` → `sync point` → `consumer read`.
5. **Resource limits**: Cap batch sizes, thread counts, and memory allocations to known safe thresholds. Chunk if exceeded.
6. **Temporary buffer sizing**: Size buffers for the operation, not the entire dataset. Temporary staging should not persist beyond the operation scope.

---

## Debug Point Extraction: From Trace to Instrumentation

The trace identifies exactly where reality can diverge from expectation. Those points are your debug instrumentation targets. Do not add logs or breakpoints randomly — instrument the trace points.

### Where to Attach Debug Points

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

### Debug Point Template (Copy-Paste)

After writing the trace, generate debug instrumentation by scanning each section:

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

### Example: Trace → Debug Points

**Trace excerpt:**
```
loadFile(path)
│
├── file.open(path)
│   ├── is_open() == false → return error
│   └── is_open() == true ↓
│
├── size = (size_t)file.tellg()
│   ├── size <= 0 → return error
│   └── size > 0 ↓
│
└── buffer.resize(size)
    ├── bad_alloc → return error
    └── success → return buffer
```

**Extracted debug points:**
```cpp
// 1. Input State
LOG("loadFile: path=%s", path.c_str());
ASSERT(!path.empty());

// 2. Branch: file.open
std::ifstream file(path);
LOG("loadFile: is_open=%d", file.is_open());
ASSERT(file.is_open());  // Truth table: is_open() == true

// 3. Condition: size = tellg()
auto pos = file.tellg();
LOG("loadFile: tellg=%ld", (long)pos);
ASSERT(pos > 0);  // Truth table: size > 0

// 4. Load Condition: resize
size_t size = (size_t)pos;
LOG("loadFile: size=%zu", size);
ASSERT(size < MAX_BUFFER_SIZE);  // Load condition

// 5. Error State: bad_alloc
try {
    buffer.resize(size);
} catch (const std::bad_alloc&) {
    LOG("loadFile: bad_alloc on size=%zu", size);
    ASSERT(false);  // Completeness: error path must be reachable in tests
    return error;
}
```

### Why This Works

- **No spray-and-pray**: You don't add 50 printfs and hope one catches the bug. You add 6 asserts at the exact points the trace identified as risky.
- **Self-documenting**: The debug points map 1:1 to the trace. When a bug hits, you know which truth table row failed.
- **Testable**: The asserts become unit test assertions. The logs become integration test expectations.

---

## Reusable Templates

### Template A: File I/O Validation
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

### Template B: Computed Size Validation
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

### Template C: Producer/Consumer Synchronization
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

---

## RAG / Knowledge Graph Distillation

Every trace produces durable knowledge. If your project uses a RAG system or knowledge graph, extract these fields:

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

### Graph Node Format (Example)

```json
{
  "id": "trace_file_io_badalloc_2026-06-28",
  "type": "validation",
  "component": "FileLoader",
  "verdict": "FAIL",
  "bug": "Unhandled bad_alloc on missing file path",
  "pattern": "File stream state must be checked before every operation",
  "assumption_broken": "Files always exist and open successfully",
  "severity": "CRITICAL",
  "fix": "Add is_open() check before tellg(); catch bad_alloc on resize",
  "debug_points": [
    "ASSERT(file.is_open()) after open",
    "ASSERT(pos > 0) after tellg",
    "ASSERT(size < MAX_BUFFER) before resize"
  ],
  "related": ["trace_size_ambiguity", "trace_buffer_overflow"]
}
```

### Query Patterns Your RAG Should Answer

- *"Why did the file loader crash on bad path?"* → Returns the stream validation trace
- *"What bugs involve size/count ambiguity?"* → Returns all size-related traces
- *"What traces confirmed false positives?"* → Returns traces that proved working code correct
- *"What CRITICAL bugs were found in static analysis?"* → Returns all CRITICAL severities
- *"Where should I add debug points for file I/O?"* → Returns debug point patterns from traces

### When to Feed the Graph

| Trigger | Action |
|---------|--------|
| Trace finds a CRITICAL bug | **Immediate** — prevents the next engineer from repeating it |
| Trace confirms a false positive | **Immediate** — prevents future "fixes" of working code |
| Lightweight trace on refactor | **Skip** — no new knowledge generated |
| Trace finds only MINOR issues | **Batch** — feed at end of week |

### Readiness Check

If your RAG cannot answer these 3 queries correctly, do not enable distillation yet:

1. *"List all CRITICAL bugs found in file I/O validation"* → Should return the relevant trace
2. *"What patterns involve size/count misinterpretation?"* → Should return the relevant trace
3. *"What traces confirmed false positives?"* → Should return the relevant trace

---

## Team Integration Guide

### For Individual Developers

1. Before writing new logic, pick the correct trace level (Full / Lightweight / None)
2. Write the trace in a scratch file, PR description, or commit message
3. Validate the truth table with real values, not assumptions
4. Extract debug points from the trace — instrument before you run
5. Only implement after VERDICT: PASS
6. If VERDICT: FAIL, fix the trace first, then code

### For Code Reviewers

1. Require a trace for any PR containing new algorithms, I/O paths, or concurrency
2. Check that the truth table covers error states (null, empty, overflow, failure)
3. Verify resource ordering has a sync point before every consumer read
4. Check that debug points are attached at every truth table row and branch
5. Reject CRITICAL-severity traces without documented fixes

### For Project Maintainers

1. Store traces in `docs/traces/` or PR descriptions — they are thinking tools, not permanent artifacts
2. Feed CRITICAL and false-positive traces into your knowledge graph
3. Review trace effectiveness quarterly: Are the same bug patterns recurring? If yes, the trace is not being written or not being read.

---

## Value Map: What the Trace Gives You

The trace is not overhead. It is a single artifact that generates multiple downstream deliverables. Write it once, get docs, tests, debug points, review checklists, security audits, and performance targets.

### 1. Function Documentation

The **Input State + Flow Diagram** becomes your function's living documentation:

```
loadFile(path)
│
├── file.open(path)
│   ├── is_open() == false → return error
│   └── is_open() == true ↓
│
├── size = (size_t)file.tellg()
│   ├── size <= 0 → return error
│   └── size > 0 ↓
│
└── buffer.resize(size)
    ├── bad_alloc → return error
    └── success → return buffer
```

**5 lines of ASCII teaches the logic.** The source code is implementation detail.

### 2. Debug Point Instrumentation

The **Truth Table** maps 1:1 to asserts and logs. Every row where `actual != expected` is an instrumentation target. See the [Debug Point Extraction](#debug-point-extraction-from-trace-to-instrumentation) section above.

### 3. Automatic Test Case Generation

The **Truth Table IS the test matrix**. Every row is a test case:

| Trace Row | Generated Test |
|-----------|----------------|
| `x > 0 → TRUE` | `TEST(function, positive_input)` |
| `x > 0 → FALSE` | `TEST(function, zero_input)` |
| `is_open() == false` | `TEST(function, missing_file)` |
| `bad_alloc` path | `TEST(function, oversized_allocation)` |

The **Completeness Check** guarantees you didn't miss a boundary.

### 4. API Contract Specification

The **Input State + Completeness Check** becomes your function's contract:

```
Contract: loadFile(path)
├── Precondition: path is non-empty string
├── Precondition: file at path exists and is readable
├── Postcondition: returns buffer on success
├── Postcondition: returns error on missing file, parse error, bad_alloc
└── Invariant: no unhandled exceptions escape
```

This is better than docstrings because it is **provable** — every condition has a truth table row.

### 5. Code Review Checklist

Reviewers verify against the trace instead of reading code blindly:

| Reviewer Check | Source |
|----------------|--------|
| Did you check `is_open()`? | Flow Diagram branch |
| Did you handle `size <= 0`? | Truth Table row |
| Is the barrier before the consumer? | Resource Ordering |
| Did you assert load < threshold? | Load Conditions |
| Is the error path reachable in tests? | Completeness Check |

A PR without a trace is **unreviewable** — the reviewer has to reverse-engineer your assumptions.

### 6. Onboarding & Knowledge Transfer

A new developer doesn't read 500 lines of shader code to understand a pipeline. They read the trace. The ASCII flow diagram is a **logic map** that makes the source code navigable.

### 7. Refactoring Safety Net

The **Lightweight Trace** proves you didn't break anything:

| Before Refactor | After Refactor |
|-----------------|----------------|
| Truth table: 6 rows, all PASS | Truth table: 6 rows, all PASS |
| Flow diagram: 3 branches | Flow diagram: 3 branches, same paths |

If the truth table changes, the refactor changed logic. If it stays identical, the refactor is pure.

### 8. Incident Post-Mortem

When a bug escapes to production, the missing trace shows exactly what wasn't considered:

```
Bug: Crash on empty file
Post-mortem: Trace was never written for loadFile()
Missing: Truth table row for size == 0
Missing: Flow diagram branch for empty file
Action: Write trace for all I/O functions
```

The trace doesn't just prevent bugs — it **diagnoses process failures**.

### 9. Performance Profiling Targets

The **Load Conditions** section tells you exactly where to instrument timers:

| Load Condition | Profiler Hook |
|----------------|---------------|
| Workgroups < 1M | `PROFILE(dispatch_count)` |
| Staging < 512 MB | `PROFILE(allocator_peak)` |
| Sync before consumer | `PROFILE(barrier_wait_ns)` |

You don't profile randomly. You profile the points the trace identified as load-sensitive.

### 10. Security Audit

The **Input State + Assumptions** reveals attack surfaces:

| Assumption | Security Risk |
|------------|---------------|
| "Path is always valid" | Path traversal injection |
| "Size is always positive" | Integer underflow → massive allocation |
| "Format is always valid" | Out-of-bounds switch fallthrough |
| "Producer always completes" | Denial of service via deadlock |

The trace forces you to document trust boundaries. Security reviewers read the assumptions section first.

### 11. CI/CD Gating

The trace can be a **merge requirement**:

```
CI Check:
├── Trace present for new I/O or concurrency? [YES/NO]
├── Truth table covers error states? [YES/NO]
├── Debug points extracted? [YES/NO]
├── VERDICT == PASS? [YES/NO]
└── Any CRITICAL without fix? [NO]
```

A bot can validate trace format. A human validates correctness.

### 12. Regression Prevention via RAG

When the same bug pattern appears twice, the RAG flags it:

```
Developer writes trace for new function
├── RAG query: "What traces involve size ambiguity?"
├── Returns: 3 previous traces with same pattern
├── Alert: "You are about to repeat a known bug pattern"
└── Action: Review previous fix before implementing
```

The trace becomes **organizational memory**, not just personal notes.

### Summary Table

| Application | What the Trace Provides |
|-------------|------------------------|
| **Function docs** | Input State + Flow Diagram |
| **Debug points** | Truth Table → ASSERT/LOG mapping |
| **Test cases** | Truth Table rows → unit tests |
| **API contracts** | Input State + Completeness Check |
| **Code review** | Checklist for every branch and state |
| **Onboarding** | ASCII flow diagram replaces source reading |
| **Refactoring** | Lightweight trace proves logic unchanged |
| **Post-mortem** | Shows what wasn't considered |
| **Performance** | Load Conditions → profiler hooks |
| **Security** | Assumptions → attack surface audit |
| **CI gating** | Format validation + VERDICT check |
| **Regression prevention** | RAG patterns alert before repeat |

---