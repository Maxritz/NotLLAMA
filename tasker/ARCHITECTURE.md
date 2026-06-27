# Auto-Tasker — DeepSeek Work Delegation System

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (me)                     │
│                                                         │
│  1. Read task queue (pending/)                          │
│  2. Generate prompt from template                       │
│  3. Spawn DeepSeek agent via Task tool                  │
│  4. Agent reads code, implements fix, writes files      │
│  5. Agent returns result                                │
│  6. I validate: build → test → lint                     │
│  7. If fail → create fix task, goto 2                   │
│  8. If pass → move to completed/, next task             │
└─────────────────────────────────────────────────────────┘
```

## Task Format (JSON in pending/)

```json
{
  "id": "fix_001",
  "type": "fix|implement|audit|test",
  "priority": 1,
  "title": "Fix rope.comp paired variable naming",
  "prompt_file": "prompts/fix_rope.md",
  "files_to_read": ["src/kernels/rope.comp", "include/rdna4_types.hpp"],
  "files_to_write": ["src/kernels/rope.comp"],
  "validation": {
    "build": true,
    "shader_compile": true,
    "test_exe": null
  },
  "max_retries": 3,
  "retries": 0,
  "status": "pending",
  "created": "2024-01-01T00:00:00Z"
}
```

## Task Types

| Type | Description | Validation |
|------|-------------|------------|
| `fix` | Fix a bug | build + test |
| `implement` | Add new feature | build + test |
| `audit` | Review code | build only |
| `test` | Write tests | build + run tests |
| `refactor` | Restructure code | build + test |
| `shader` | Modify compute shader | build + shader compile |

## Validation Pipeline

1. **Build check**: `cmake --build . --config Release`
2. **Shader compile**: All `.spv` files regenerated
3. **Test run**: Execute test binary if exists
4. **Smoke test**: `rdna4_llama.exe` runs without crash
5. **Output check**: Compare GPU vs CPU reference

## Files

- `tasker/tasks/` — Active task queue
- `tasker/prompts/` — Prompt templates
- `tasker/logs/` — Execution logs
- `tasker/ARCHITECTURE.md` — This file
