# Task: {{TITLE}}

## Context

You are working on an RDNA4 LLM inference engine at `C:\Users\rr\Desktop\Notllama-loc`. The engine runs VibeThinker-3B.Q6_K (qwen2 architecture) end-to-end on Vulkan compute shaders.

## Goal

{{DESCRIPTION}}

## Task

{{TASK细节}}

## Constraints

- Read ALL relevant source files before making changes
- Follow existing code conventions (see AGENTS.md)
- Do NOT add comments unless asked
- Do NOT modify files not listed in the task
- Build must pass: `cd build && cmake --build . --config Release`
- If shader modified: recompile with `glslc -V shader.comp -o shader.spv`

## Files to Read

{{FILES_TO_READ}}

## Files to Modify

{{FILES_TO_WRITE}}

## Expected Output

{{EXPECTED_OUTPUT}}
