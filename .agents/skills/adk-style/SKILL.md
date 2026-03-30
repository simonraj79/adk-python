---
name: adk-style
description: ADK development style guide — architecture patterns, testing best practices, Python idioms, and codebase conventions. Use when writing code, tests, or reviewing PRs for the ADK project. Triggers on "write tests", "best practice", "code style", "how should I", "convention", "pattern", "review code".
---

# ADK Style Guide

## Development Conventions
[references/development.md](references/development.md) — public API vs internal methods, comments, file organization, imports, Pydantic patterns, formatting

## Architecture (references/architecture/)
- [BaseNode](references/architecture/base-node.md) — node contract, output/streaming, state/routing, HITL, configuration
- [Context](references/architecture/context.md) — 1:1 node-context mapping, InvocationContext singleton, property reference
- [NodeRunner](references/architecture/node-runner.md) — two communication channels, execution flow, output delegation
- [Runner](references/architecture/runner.md) — Runner vs NodeRunner vs Workflow separation
- [Checkpoint and Resume](references/architecture/checkpoint-resume.md) — HITL lifecycle, `rerun_on_resume`, `execution_id`

## Testing
[references/testing.md](references/testing.md) — core principles, 9 rules for writing ADK tests, test structure template
