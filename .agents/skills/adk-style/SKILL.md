---
name: adk-style
description: ADK development style guide — architecture patterns, testing best practices, Python idioms, and codebase conventions. Use when writing code, tests, or reviewing PRs for the ADK project. Triggers on "write tests", "best practice", "code style", "how should I", "convention", "pattern", "review code".
---

# ADK Style Guide


## Code Paths
[references/code_paths.md](references/code_paths.md) — Key files and directories along the different agent and workflow code paths.

## Beta Launch Scope (Priority)
[references/beta_launch_scope.md](references/beta_launch_scope.md) — Scope decisions for the upcoming beta launch. **This path is the current priority. All analysis and development should focus here.**

## Alpha Launch Scope
[references/alpha_launch_scope.md](references/alpha_launch_scope.md) — Scope for the alpha launch (legacy).


## Architecture (references/architecture/)
- [BaseNode](references/architecture/base-node.md) — node contract, output/streaming, state/routing, HITL, configuration
- [Context](references/architecture/context.md) — 1:1 node-context mapping, InvocationContext singleton, property reference
- [NodeRunner](references/architecture/node-runner.md) — two communication channels, execution flow, output delegation
- [Runner](references/architecture/runner.md) — Runner vs NodeRunner vs Workflow separation
- [Checkpoint and Resume](references/architecture/checkpoint-resume.md) — HITL lifecycle, `rerun_on_resume`, `run_id`
- [Workflow](references/architecture/workflow.md) — graph orchestration, dynamic nodes (tracking/dedup/resume), transitive dynamic nodes, interrupt propagation, design rules for node authors
- [Observability](references/architecture/observability.md) — span-on-Context design, NodeRunner integration, correlated logs, metrics
- [LLM Context Orchestration](references/architecture/llm-context-orchestration.md) — relationship between events and LLM context, task delegation translation, branch isolation. Use when modifying event processing, context preparation for LLMs, or debugging context pollution issues.

## Style Guide (references/style/)
- [API Principles](references/style/api-principles.md) — stability, backward compatibility, and self-containment.
- [Visibility](references/style/visibility.md) — naming conventions for module-private, internal, and package-private visibility.
- [Imports](references/style/imports.md) — relative vs absolute imports, `TYPE_CHECKING` patterns.
- [Typing](references/style/typing.md) — strong typing, avoiding Any, and bare type names.
- [Pydantic Patterns](references/style/pydantic.md) — Pydantic v2 usage, private attributes, post-init setup.
- [Formatting](references/style/formatting.md) — indentation, line limits, and running pre-commit hooks.
- [Documentation](references/style/documentation.md) — comments and docstrings.
- [Logging](references/style/logging.md) — lazy evaluation and log levels.
- [File Organization](references/style/file-organization.md) — file headers and class organization.

## Testing
[references/style/testing.md](references/style/testing.md) — core principles, 9 rules for writing ADK tests, test structure template
