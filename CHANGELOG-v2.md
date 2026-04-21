# Changelog

## [2.0.0-beta.1](https://github.com/google/adk-python/compare/v2.0.0-beta.0...v2.0.0-beta.1) (2026-04-21)

### Highlights

*   **Transition to Beta**: Updated documentation to reflect the project's move to Beta phase ([84e792fd](https://github.com/google/adk-python/commit/84e792fd1ef1b946e5bcacbaf54834cb91f557a0)).
*   **Security Fix**: Resolved a potential RCE vulnerability related to nested YAML configurations ([2b54c4ac](https://github.com/google/adk-python/commit/2b54c4ac37c3ec8bdbb23f4ed6d7523d7700d95f)).
*   **Documentation & Style**: Modularized the ADK style guide and added new topics ([9bb21795](https://github.com/google/adk-python/commit/9bb217955a176422c8678eae56af23b531340aa8)).
*   **Workflow Orchestration Core**:
    *   Added the full `Workflow(BaseNode)` graph orchestration implementation ([c25d86f1](https://github.com/google/adk-python/commit/c25d86f1adf6b32e5cc780880acf3d2e15e0b984)).
    *   Introduced `NodeRunner` for per-node execution isolation ([0b3e7043](https://github.com/google/adk-python/commit/0b3e7043c47228f0ba1d89d6c86a0cf6a67ebd31)).
    *   Added `DefaultNodeScheduler` for standalone node resume via `ctx.run_node()` ([a68e70d5](https://github.com/google/adk-python/commit/a68e70d5da23eaafc5db5611205d34976cb98e5a)).
*   **Agent Execution Models**:
    *   Added explicit `ReAct` loop nodes to replace legacy single agent flows ([94453619](https://github.com/google/adk-python/commit/944536191ea16b050ba0a38ad5783108dd0b79bc)).
    *   Supported `BaseNode` as the root in both the Runner and the CLI ([91b56b9a](https://github.com/google/adk-python/commit/91b56b9ab07eca1e9b7b590e2ad6eee52898c0bc), [1c2a81bb](https://github.com/google/adk-python/commit/1c2a81bb675f98ba06ffcd5423f2c5aedce94bee)).
*   **State & Resumption**:
    *   Added Human-in-the-loop (HITL) resume via event reconstruction ([ca327329](https://github.com/google/adk-python/commit/ca32732991c89a3a0ab88a43874182af3e937a13)).
    *   Added lazy scan deduplication and resume for dynamic nodes ([d600b195](https://github.com/google/adk-python/commit/d600b195bd5831cb6bc460e704178b739880576f)).
    *   Supported flushing state/artifact deltas onto yielded events ([00153088](https://github.com/google/adk-python/commit/0015308820a86e076581547ae585ade853f82707)).
*   **Performance**:
    *   Optimized execution by bypassing the Mesh for leaf single-turn `LlmAgent` instances ([d864917f](https://github.com/google/adk-python/commit/d864917fb6e8c82c4a67941cc8e8d11f9f7540d1)).


## [2.0.0-alpha.3](https://github.com/google/adk-python/compare/v2.0.0-alpha.2...v2.0.0-alpha.3) (2026-04-09)

### Features

* **Workflow Orchestration:** Added Workflow(BaseNode) graph orchestration implementation, support for lazy scan deduplication and resume for dynamic nodes, partial resume for nested workflows, and state/artifact delta bundling onto yielded events.
* **CLI and Web UI:** Added support for Workflow graph visualization in web UI, improved graph readability with distinct icons and shapes, and active node rendering in event graph.
* **Documentation:** Added reference documentation for development (skills like adk-style, adk-git, and observability architecture).

## [2.0.0-alpha.1](https://github.com/google/adk-python/compare/v2.0.0-alpha.0...v2.0.0-alpha.1) (2026-03-18)

### Features

Introduces two major capabilities:
* Workflow runtime: graph-based execution engine for composing
  deterministic execution flows for agentic apps, with support for
  routing, fan-out/fan-in, loops, retry, state management, dynamic
  nodes, human-in-the-loop, and nested workflows
* Task API: structured agent-to-agent delegation with multi-turn
  task mode, single-turn controlled output, mixed delegation
  patterns, human-in-the-loop, and task agents as workflow nodes
