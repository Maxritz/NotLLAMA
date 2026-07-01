# Truth/False Logic Validation — Gap Evaluation

## GAP 1: Flow Diagram

### Current State
Text decision trees only. No visual execution path.

### Problem
For complex flows (nested loops, async waits, GPU dispatch chains), text trees become unreadable. You can't see the full path at a glance.

### What's Needed
ASCII flow diagrams embedded in the trace. Not external tools — inline in the markdown.

### Example (dequantIfNeeded)
```
dequantIfNeeded(tensor, addr, nElem, format)
│
├── format == F32?
│   ├── TRUE → return addr (no work)
│   └── FALSE ↓
│
├── stagingAddr = alloc(nElem * 4)
│   ├── 0 (OOM) → return addr (fallback, log warning)
│   └── nonzero ↓
│
├── dispatch(dequant, addr→staging)
├── barrier(WRITE→READ)
└── return stagingAddr
```

### Verdict
**Add ASCII flow diagrams to every trace.** Not optional for anything with >3 branches.

---

## GAP 2: Logical Completeness Check

### Current State
Truth table checks individual conditions. No completeness check.

### Problem
You can have all rows PASS but still miss:
- Unreachable code paths
- Undefined states (what happens if format = 99?)
- Goal state not reachable from some input

### What's Needed
Three additional checks:

#### Check A: Branch Coverage
```
For every if/else/switch:
  - Is EVERY branch reachable?
  - Is EVERY branch tested in the truth table?
  - Are there dead code paths?
```

#### Check B: State Completeness
```
For every variable:
  - What are ALL possible values?
  - Is EVERY value handled?
  - What happens on unexpected values?
```

#### Check C: Goal Reachability
```
For every valid input:
  - Can the goal state be reached?
  - Is there a path from input to output?
  - Are there infinite loops or early exits that prevent completion?
```

### Example (dequantize.comp)
```
format values: 0,1,2,3,6,7,8,9,10,11,12,13,14,15
Handled:       0,1,2,3,6,7,8,9,10,11,12,13,14,15
Missing:       4,5,16-30,31+ (falls through to "output zeros")
Verdict:       PARTIAL — unknown formats produce zeros, not error
```

### Verdict
**Add branch coverage, state completeness, and goal reachability checks.**

---

## GAP 3: Resource Ordering

### Current State
Race condition checklist exists but doesn't check ordering.

### Problem
A checklist says "no race" but doesn't verify:
- Producer writes BEFORE consumer reads (DAG)
- No circular waits (deadlock)
- Barrier is BEFORE the race-prone access, not after

### What's Needed
DAG check for every producer→consumer pair:

```
PRODUCER                    CONSUMER
────────                    ────────
dequant writes staging  →   GEMM reads staging

Ordering check:
  dequant dispatch timestamp < GEMM dispatch timestamp?
  barrier between them?
  barrier BEFORE GEMM read? → YES = CORRECT
  barrier AFTER GEMM read?  → YES = RACE
```

### Deadlock Check
```
For every barrier:
  - Is it in the same command buffer?
  - Is it on the same queue?
  - Could it wait on itself? (circular)
```

### Barrier Placement Check
```
For every producer→consumer:
  - Where is the barrier relative to the access?
  - BEFORE consumer access? → CORRECT
  - AFTER consumer access?  → RACE (too late)
  - NO barrier?             → RACE
```

### Example (dequant→GEMM)
```
Sequence in command buffer:
  1. vkCmdDispatch(dequant)     # WRITE to staging
  2. vkCmdPipelineBarrier()     # WAIT for write
  3. vkCmdDispatch(gemm)        # READ from staging

DAG: dequant → barrier → gemm
Barrier placement: BEFORE gemm read
Verdict: CORRECT
```

### Verdict
**Add DAG ordering, deadlock check, and barrier placement verification.**

---

## GAP 4: Distillation Rules

### Current State
Not defined. What goes to graphify/RAG?

### Problem
Without rules, you either:
- Extract everything (noise)
- Extract nothing (knowledge lost)
- Extract wrong things (misleading)

### What Gets Extracted

| Category | Extract? | Example |
|----------|----------|---------|
| Verdict (PASS/FAIL) | YES | "dequant trace: PASS" |
| Bugs found | YES | "BUG: nElements wrong for Q4_0" |
| Patterns | YES | "Vulkan sync: barrier before consumer read" |
| Counter-examples | YES | "Q4_0 without calcNumElements: FAIL" |
| Full truth table | NO | Too detailed for graph |
| Full code | NO | Already in source |
| Assumptions | YES | "Assumes ring buffer has space" |
| Load conditions | YES | "GPU load: 80% at 8192 workgroups" |

### Extraction Format
```
Node: [feature_name]_[date]
Type: validation
Verdict: PASS/FAIL
Bugs: [list]
Patterns: [list]
Assumptions: [list]
Load: [GPU%, VRAM%]
```

### Verdict
**Define extraction rules. Extract verdict, bugs, patterns, assumptions, load. Skip full tables and code.**

---

## GAP 5: Failure Modes

### Current State
"VERDICT: PASS / FAIL" — no defined action on FAIL.

### Problem
FAIL could mean:
- Abort and rewrite (waste of time if bug is minor)
- Flag and retry (good for small issues)
- Log and move on (dangerous — silent failures)

### What's Needed
Three failure severity levels:

#### Level 1: CRITICAL (Abort)
```
Conditions:
- Buffer overflow possible
- Race condition confirmed
- GPU hang likely
- Data corruption

Action: STOP. Fix before any code writes.
```

#### Level 2: MAJOR (Flag + Retry)
```
Conditions:
- Wrong format conversion
- Missing branch coverage
- Suboptimal but functional

Action: List specific issues. Retry with guidance.
```

#### Level 3: MINOR (Log + Continue)
```
Conditions:
- Style inconsistency
- Missing documentation
- Non-optimal performance

Action: Log the issue. Continue with main task.
```

### Example (dequant trace)
```
BUG 1: nElements wrong for Q4_0 → CRITICAL (data corruption)
BUG 2: F32 byte indexing wrong → CRITICAL (data corruption)
BUG 3: uint16_t missing extension → MAJOR (won't compile)
BUG 4: int/uint type mismatch → MAJOR (won't compile)

Action: Fix CRITICAL bugs before building. Fix MAJOR bugs during build.
```

### Verdict
**Add three severity levels: CRITICAL (abort), MAJOR (flag+retry), MINOR (log+continue).**

---

## UPDATED TRUTH/FALSE LOGIC RULE

### Complete Checklist

```
□ Input State documented
□ Decision tree written (ASCII flow diagram)
□ Truth table with ALL branches covered
□ Branch coverage check (no unreachable code)
□ State completeness check (all values handled)
□ Goal reachability check (every input → output)
□ Race condition check (producer before consumer)
□ DAG ordering verified
□ Deadlock check (no circular waits)
□ Barrier placement check (before consumer access)
□ Load conditions (GPU < 80%, VRAM < 80%)
□ Resource ordering (allocation before use)
□ Distillation: verdict, bugs, patterns, assumptions
□ Failure severity: CRITICAL/MAJOR/MINOR
□ Action defined for each failure
```

### Updated Format

```markdown
## TRACE: [Feature Name]

### Input State
| Variable | Value | Source |
|----------|-------|--------|

### Flow Diagram
```
[ASCII flow diagram]
```

### Decision Tree
```
[if/else branches]
```

### Truth Table
| Condition | Expected | Actual | PASS? |
|-----------|----------|--------|-------|

### Completeness Check
- Branch coverage: ALL branches tested? [YES/NO]
- State completeness: ALL values handled? [YES/NO]
- Goal reachability: EVERY input → output? [YES/NO]

### Resource Ordering
- Producer → Consumer: [pair]
- DAG: [ordering]
- Barrier placement: [BEFORE/AFTER consumer]
- Deadlock risk: [NONE/LOW/HIGH]

### Load Conditions
- GPU load: [value] < 80%? [PASS/FAIL]
- VRAM usage: [value] < 80%? [PASS/FAIL]
- Dispatch storm: [count] < 1M? [PASS/FAIL]

### Distillation
- Verdict: PASS/FAIL
- Bugs: [list]
- Patterns: [list]
- Assumptions: [list]

### Failure Severity
| Issue | Severity | Action |
|-------|----------|--------|

### VERDICT: PASS / FAIL
```
