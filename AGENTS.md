# AI Coding Assistant Context

This document provides context for AI coding assistants (Antigravity, Gemini CLI, etc.) to understand the ADK Python project and assist with development.

## Skills index — when to invoke which

The repository ships seven task-scoped skills under `.agents/skills/`. Each
folder contains a `SKILL.md` with YAML frontmatter (`name`, `description`)
and supporting reference docs. The `description` field is the
auto-activation trigger — when the user's request matches a skill's
triggers, read the matching `SKILL.md` first and follow it. For tasks
that span multiple skills, layer them (e.g., `adk-workflow` to build,
then `adk-style` to polish, then `adk-debug` to verify).

| Skill | Invoke when… | Auto-trigger |
|---|---|---|
| **`adk-architecture`** | You need to understand or design ADK's runtime: event flow, state management, `BaseNode`, `Workflow`, `NodeRunner`, `Context`, checkpoint/resume, observability, LLM context orchestration. Use for "how does X work?", "design of…", core API changes. | yes |
| **`adk-workflow`** | You are **building** an agent or workflow: defining `LlmAgent` / `FunctionNode` / `LlmAgentWrapper` / `JoinNode` / `ParallelWorker`, wiring `Edge`s, conditional routing, fan-out/fan-in, retry, HITL via `RequestInput`, parallel workers, MCP tools, session state, `mode='task'` / `mode='single_turn'`. The day-to-day "make a new agent" skill. | yes |
| **`adk-debug`** | You are debugging or verifying agent behavior: inspecting sessions, tracing event flow, tool-call issues, LLM/model problems, choosing between `adk web` (browser UI + REST) and `adk run` (CLI). Always consult before adding print-statement debugging. | yes |
| **`adk-style`** | You are writing or reviewing code/tests: Python idioms, imports, typing, Pydantic v2 patterns, formatting, logging, file organization, ADK testing rules. Apply on every code edit before commit. | yes |
| **`adk-sample-creator`** | The user wants to add a new sample under `contributing/` (e.g., dynamic nodes, fan-out/fan-in, standalone agents). Covers required folder/file shape (`agent.py`, `README.md`) and naming. | yes |
| **`adk-git`** | Any git operation — commit, push, pull, rebase, branch, PR, cherry-pick. Documents the project's Conventional Commits format (`<type>(<scope>): <description>`). Consult before composing any commit message. | yes |
| **`adk-setup`** | The user is setting up a local dev environment from scratch: Python 3.11+, `uv` package manager, venv, dependency install. Required for contributors getting started. | **no — explicit only** (skill sets `disable-model-invocation: true`; do not volunteer it unless the user asks for setup help) |

### Common skill combinations

- **Building a new agent or workflow** → `adk-workflow` (build) + `adk-style` (polish) + `adk-debug` (verify) + `adk-git` (commit).
- **Designing a framework-level change** → `adk-architecture` (design) + `adk-style` (implement) + `adk-debug` (verify) + `adk-git` (commit).
- **Adding a contributing sample** → `adk-sample-creator` (scaffold) + `adk-workflow` (logic) + `adk-style` (polish).
- **Diagnosing a misbehaving demo** → `adk-debug` first; reach for `adk-architecture` only if the issue is in the runtime, not the demo.

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.

### Key Components

- **Agent**: Blueprint defining identity, instructions, and tools.
- **Runner**: Stateless execution engine that orchestrates agent execution.
- **Tool**: Functions/capabilities agents can call.
- **Session**: Conversation state management.
- **Memory**: Long-term recall across sessions.
- **Workflow** (ADK 2.0): Graph-based orchestration of complex, multi-step agent interactions.
- **BaseNode** (ADK 2.0): Contract for all nodes, supporting output streaming and human-in-the-loop steps.
- **Context** (ADK 2.0): Holds execution state and telemetry context mapped 1:1 to nodes.

For details on how the Runner works and the invocation lifecycle, please refer to the `adk-architecture` skill and the referenced documentation therein.

## Project Architecture

For detailed architecture patterns, component descriptions, and core interfaces, please refer to the **`adk-architecture`** skill at `.agents/skills/adk-architecture/SKILL.md`.

## Development Setup

The project uses `uv` for package management and Python 3.11+. Please refer to the **`adk-setup`** skill at `.agents/skills/adk-setup/SKILL.md` for detailed instructions.

---

## Local fork (simonraj79/adk-python) — additions

Everything above this divider is upstream. Everything below is specific to
this fork and should be preserved across upstream re-baselines.

### Active branch

- `v2-migration` — current development branch, baselined on the `v2.0.0b1`
  upstream tag.
- `main` — frozen on v1.31, kept as safety net. Do not work on `main`.
- `backup/pre-v2-migration` — branch + tag preserving the pre-migration
  `main` HEAD.

### Repository layout (this fork)

In addition to upstream framework code under `src/google/adk/`, this fork
holds **local agent demos** for teaching purposes. Both v1 archives and
v2 rewrites coexist so learners can compare approaches side by side:

- `V1_*` folders (e.g., `V1_level_1_agent/`, `V1_level_1a_agent/`) —
  **archived v1 demos**, reference only. Source is intact but they were
  written against ADK 1.31. They still run on v2 in most cases (the
  loader is backwards-compatible for the `LlmAgent` shape) and serve
  as a "before" exhibit for each v2 rewrite.
- `level_*_agent/` folders — **v2 rewrites**. Currently populated:
  `level_1_agent/`, `level_1a_agent/`. The remaining slots (`level_2`,
  `level_3`, `level_4`, `level_4a`) are written one at a time, in
  order, and each one should highlight a specific v2 idiom (leaf
  fast-path, `RunConfig`-driven Live config, `Workflow(BaseNode)`
  graphs, dynamic-node spawning, etc.) rather than mechanically
  copying the v1 logic.

### Agent levels — what each demo teaches

Six demos under this fork, ascending in capability. Each one introduces
exactly one new v2 idiom on top of the previous, so a learner can read
them in order and trace a clear capability ladder. Folders are
`level_*_agent/` for the v2 rewrites; the v1 originals live alongside
in `V1_level_*_agent/` for before/after comparison.

#### Level 1 — Connected Problem-Solver (`level_1_agent/`)

**Capability**: a single LLM agent that connects to one external tool
(Google Search) to answer questions requiring real-time information.

**v2 idioms genuinely in use**:
- `Agent` (alias for `LlmAgent`) as a leaf — runs on v2's
  Mesh-bypass fast path (`CHANGELOG-v2.md`: "Optimized execution by
  bypassing the Mesh for leaf single-turn `LlmAgent` instances").
- `tools=[google_search]` — Gemini built-in (no auto-swap needed
  because there's only one tool).
- `output_key="last_answer"` — v2's state-delta-on-events flushing,
  so the response is observable in the State panel and as
  `Event.actions.state_delta`.

**Single file**: `agent.py`. Sample query: *"What's the latest from
JPMorgan on AI agents?"*.

#### Level 1a — Voice variant of Level 1 (`level_1a_agent/`)

**Capability**: same one-tool shape as L1 but running on a Gemini Live
model so the user can speak the question and the agent speaks back.

**v2 idioms genuinely in use**:
- Model declared as a plain string (`gemini-3.1-flash-live-preview`),
  no `Gemini(...)` wrapper. Voice and modality config moved off the
  agent and onto `RunConfig` (`run_config.py:195`) — same agent file
  can use Zephyr in one session and Kore in another via the Live
  Flags panel.
- `after_model_callback` (`_suppress_telemetry_only_responses`) that
  rewrites Live's per-chunk `usage_metadata` and
  `live_session_resumption_update` messages into empty
  `LlmResponse()`s, triggering the framework's event-skip
  early-return at `base_llm_flow.py:1023-1033`. Without this, a
  single voice turn produces 200+ empty bubbles in adk-web's chat
  panel (one per audio chunk's token-count telemetry). See
  AGENTS.md gotcha entry.

**Two failure modes documented in the docstring**: native-audio Gemini
API models fail audio-input negotiation with adk-web's bidi worklet;
`gemini-3.1-flash-live-preview` (half-cascade) is the v1-proven
working path on Gemini API. Native audio works on Vertex.

**Single file**: `agent.py`. To run: `adk web .`, pick `level_1a_agent`,
**click the mic icon** (typing fails — bidi-only Live model has no
`generateContent` endpoint).

#### Level 2 — Strategic Problem-Solver (`level_2_agent/`)

**Capability**: planning + parallel execution. The agent decomposes a
question into sub-questions, fans them out concurrently, and
synthesises a brief.

**v2 idioms genuinely in use**:
- `Workflow(BaseNode)` graph-orchestration runtime (the v2 headline
  feature — `CHANGELOG-v2.md`).
- Edge declarations: `("START", process_input, classify, route_input)`
  + dict routing `(route_input, {"greeting": greeter, "research": planner})`.
- `output_schema` on classifier (Pydantic `Literal[...]` for
  routing) + `output_key` for instruction injection downstream.
- `ctx.run_node()` dynamic fan-out — the planner LLM decides how
  many sub-questions to spawn at runtime; `fan_out_research`
  spawns one researcher per sub-question via dynamic node
  scheduling (`CHANGELOG-v2.md`: "DefaultNodeScheduler for
  standalone node resume").
- `Event(state={...})` for inter-node data flow — *replaces* v1's
  `save_research_note` / `get_research_notes` custom scratchpad
  tools. Pure framework state.
- `@node(rerun_on_resume=True)` on the dynamic-fan-out orchestrator
  so it's HITL-resumable.

**Single file**: `agent.py`. Sample queries: *"hi"* (greeting branch),
*"how is solid-state battery research progressing across LFP, sulfide,
and oxide chemistries?"* (3 sub-questions, parallel).

#### Level 3 — Collaborative Multi-Agent System (`level_3_agent/`)

**Capability**: a coordinator delegates work to a team of specialist
sub-agents (search / analyst / writer), each with a distinct role and
non-overlapping tools.

**v2 idioms genuinely in use**:
- `sub_agents=[...]` + `mode='single_turn'` on each specialist —
  framework auto-derives `_SingleTurnAgentTool` instances on the
  coordinator at `model_post_init` time (`llm_agent.py:982-994`).
  This is the v2 idiom that *replaces* v1's manual
  `tools=[AgentTool(agent=X), ...]` wrapping. The
  `request_task_<name>` delegation tool is auto-generated.
- `disallow_transfer_to_parent=True` and
  `disallow_transfer_to_peers=True` on every specialist — required
  because `agent_transfer.py:152-188` injects a `transfer_to_agent`
  function tool that conflicts with `google_search` (built-in /
  function combo Gemini rejects). Caught only by end-to-end tool
  calls, not by static load checks. See AGENTS.md migration pattern
  #2's "critical sub-pattern" callout.
- Pydantic `input_schema` / `output_schema` on every sub-agent —
  typed call-and-return contracts that *replace* v1's prose-only
  ("save a note with topic, finding, source") instructions.
- `PlanReActPlanner()` on the coordinator — visible
  `/PLANNING/ /REASONING/ /ACTION/` blocks in the chat panel,
  showing the multi-agent orchestration logic inline.
- `output_schema=Brief` on `report_writer_agent` (no built-in tools
  → `set_model_response` injection works without conflict).

**Single file**: `agent.py`. Sample query: *"compare mRNA vs
viral-vector vaccine platforms"* (multi-source delegation).

#### Level 4 — Self-Evolving System (`level_4_agent/`)

**Capability**: meta-reasoning + dynamic agent creation. The
coordinator routes to a fixed BI team (data_fetcher / analyst /
report_writer), but if it detects a capability gap it transfers to
`agent_creator` which builds a new specialist on demand. Runtime
specialists persist across server restarts via the
`runtime_agents/` YAML library on disk.

**v2 idioms genuinely in use** (everything from L3, plus):
- `code_executor=BuiltInCodeExecutor()` on `analyst_agent` — Gemini's
  hosted Python sandbox, separate from `tools=[]`. Pre-imports
  pandas + matplotlib + numpy.
- `BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True))`
  on `analyst_agent` — Gemini's native thinking surfaced as separate
  THOUGHT events. Analyst plans cell layout BEFORE writing code,
  defeating the v1 "blank PNG" gotcha (multi-cell figure drift).
- `mode='task'` on `agent_creator` for multi-turn HITL — the creator
  can ask "shall I proceed?" and wait for the user's reply, then
  call `create_specialist` and `finish_task` to auto-return to the
  coordinator. v2's structured equivalent of the v1 "transfer to
  sub_agent and beg the LLM to transfer back" pattern.
- `gemini-3.1-pro-preview` model on `agent_creator` — needed for the
  chained tool decision (draft → confirm → call create_specialist
  → call finish_task). Flash exhibits empty-STOP responses on
  this chain. Confirmed via Context7 model card.
- `disallow_transfer_to_parent=True` on `agent_creator` too —
  narrows its tool surface from `[transfer_to_agent,
  create_specialist, finish_task]` to `[create_specialist,
  finish_task]`, eliminating tool-choice paralysis.
- `before_agent_callback` on the coordinator —
  `_rehydrate_runtime_tools` rebuilds `root_agent.tools` every turn
  as `_INITIAL_TOOLS + runtime_tools` (auto-derived sub_agent tools
  + per-spec runtime AgentTools loaded from session state AND the
  on-disk YAML library).
- `PlanReActPlanner()` on the coordinator — visible meta-reasoning
  during routing decisions (e.g., "step 2: does any runtime
  specialist's description match? if so, call THAT one; do NOT
  recreate").
- Runtime specialists wrap as `AgentTool` (NOT `sub_agents`)
  because v2's `sub_agents → _SingleTurnAgentTool` auto-derivation
  is init-time only — mutating sub_agents post-init won't register
  new tools. **Pedagogical split: fixed teams use `sub_agents`;
  runtime teams use `AgentTool`.**
- Restart-aware persistence: `has_capability` checks both
  in-session state AND the on-disk YAML library, so a fresh session
  asked to "build f1_data_agent" sees the existing YAML and
  short-circuits rather than silently overwriting.

**Multi-file structure**:
- `agent.py` — coordinator + 4 sub-agents wired with v2 idioms.
- `safety.py` — non-negotiable allowlist of tool names runtime
  specialists may use (validate_spec rejects anything else).
- `registry.py` — capability persistence across turns AND restarts;
  `has_capability` is disk-aware; `hydrate_capabilities` returns
  `state ∪ disk` AgentTools deduped by name (state wins).
- `creator_tools.py` — `create_specialist`: validate → smoke-test →
  persist to state → write YAML audit copy.
- `tools.py` — safe AST-based `calculator` (rejects all injection
  attempts in unit tests).
- `runtime_agents/*.yaml` — disk library of runtime specialists,
  reloaded on every `before_agent_callback`. Survives `adk web`
  restart.

**Allowlist (4 tools)**: `google_search`, `get_current_date`,
`calculator`, `load_web_page`.

**Sample queries**: *"hi"* (greeting), *"give me a board update on
NVIDIA's revenue trend"* (fixed-team analytical with chart),
*"build a specialist that pulls F1 race results"* (capability gap →
HITL creation).

#### Level 4a — Self-Evolving System over MCP (`level_4a_agent/`)

**Capability**: structurally identical to L4, but `data_fetcher_agent`
gains a third tool source — **gahmen-mcp**, a domain-specific MCP
server providing data.gov.sg + SingStat (Department of Statistics
Singapore) data. The diff vs L4 is intentionally narrow.

**v2 idioms genuinely in use** (everything from L4, plus):
- `McpToolset(connection_params=StdioConnectionParams(...),
  tool_filter=[...])` — v2's MCP primitive. `tool_filter` narrows
  the toolset to the 8 read-only `datagovsg_*` / `singstat_*` tools
  (download orchestration tools deliberately excluded).
- `StdioConnectionParams` launches the gahmen-mcp server via
  `npx tsx vendor/gahmen-mcp/src/stdio_entry.ts` — a 15-line
  TypeScript wrapper that adapts the upstream Smithery
  `createStatelessServer` factory to `StdioServerTransport`.
- `data_fetcher_agent` has 3 tool sources at runtime: 10 function
  tools sent to Gemini after auto-swap (`GoogleSearchTool(bypass=True)` →
  `GoogleSearchAgentTool` + `load_web_page` + 8× `MCPTool`s).
  Source routing is LLM-driven via instruction.
- Runtime allowlist sentinel pattern: 8 MCP tool names map to a
  `_MCP_SENTINEL` placeholder; `safety.resolve_tools()` swaps each
  set of MCP names per-spec into a fresh `McpToolset(tool_filter=...)`.
  Each runtime specialist gets only the MCP tools it asked for —
  enforced at the framework boundary, not just at spec-validation
  time.
- Graceful degradation: if `vendor/gahmen-mcp/` is missing, the
  framework logs `Failed to get tools from toolset McpToolset` and
  the rest of the BI team continues to work — the MCP-using
  specialists fail at first tool call, but `google_search` /
  `load_web_page` remain available. v2 resilience moment.

**Prerequisite**: vendor the gahmen-mcp server. From inside
`level_4a_agent/`:
```powershell
git clone https://github.com/<your-fork>/gahmen-mcp.git vendor/gahmen-mcp
cd vendor\gahmen-mcp
npm install
# the stdio_entry.ts wrapper is already in place if you cloned this fork
```

**Multi-file structure** (delta vs L4):
- `agent.py` — narrow extension: top docstring, +1 import, +1 tool
  on `data_fetcher_agent`, instruction enumerates the 8 MCP
  allowlist additions, name/description say "level_4a".
- `safety.py` — extends L4 allowlist with 8 MCP sentinels +
  `resolve_tools` MCP-narrowing path.
- `mcp_toolset.py` — **new file**: gahmen-mcp wiring (`McpToolset`
  + `StdioConnectionParams`, `tool_filter` to the read-only set).
- `creator_tools.py`, `registry.py`, `tools.py`, `__init__.py` —
  identical to L4 (the registry routes through `safety.resolve_tools`
  so MCP "just works").
- `vendor/gahmen-mcp/` — the cloned + npm-installed MCP server,
  with our `src/stdio_entry.ts` wrapper. Gitignored.

**Allowlist (12 tools)**: L4's four (`google_search`,
`get_current_date`, `calculator`, `load_web_page`) + 5 datagovsg_* +
3 singstat_* = 12.

**Sample query**: *"I want to understand Singapore's labour market —
can you pull the resident unemployment rate trend over the last
several years from SingStat and give me a brief?"* — verified
end-to-end; data_fetcher's LLM correctly picked `singstat_search_resources`
then `singstat_get_table_data`, analyst rendered a 34-year chart with
crisis-by-crisis annotations, writer produced typed Brief.

### v1 → v2 migration patterns

The v2 thesis in one sentence: **every v2 primitive replaces a v1
"please-LLM-follow-this-prompt" caveat with a structural guarantee.**
When migrating a v1 demo, scan for these anti-patterns and apply the
listed v2 replacement. Each pattern below is grounded in either the
`adk-workflow` / `adk-architecture` skills, an upstream `contributing/`
sample, or a `CHANGELOG-v2.md` highlight — cited inline.

#### 1. Graph workflow replaces "prompt-driven multi-step plan"

**v1 anti-pattern**: a single `LlmAgent` with a long `instruction` like
*"1. classify the message, 2. call the right handler, 3. never skip
steps, 4. respond in the requested format"*. The control flow lives
inside the prompt; the LLM may skip a step, pick the wrong handler, or
just respond conversationally without calling any tool.

**v2 replacement**: `Workflow(name=..., edges=[...])` with explicit
nodes and edges. The classifier becomes one node, the router a function
node returning `Event(route=...)`, the handlers separate nodes — and
control flow lives in code, not prose.

**Migration recipe**:
1. Identify the steps in the v1 instruction. Each numbered step usually
   becomes a node.
2. Make the classification step its own `Agent` with a Pydantic
   `output_schema` (e.g., `Literal["bug", "billing", "feature"]`) and
   `output_key` so downstream nodes read the decision from state.
3. Add a tiny function node `def route(node_input: dict): return
   Event(route=node_input["category"])`.
4. Wire branches with dict routing: `(route, {"bug": handle_bug,
   "billing": handle_billing, "feature": handle_feature})`.
5. Delete every "first do X, then do Y, never skip" sentence from the
   remaining prompts — the graph enforces it now.

**Why**: routing decisions belong in code, where they're deterministic
and inspectable. The `level_2_agent` rewrite is the canonical example
in this repo (`level_2_agent/agent.py`); the upstream reference is
`contributing/workflow_samples/loop/agent.py`.

#### 2. `mode='task'` / `mode='single_turn'` replace manual sub-agent handoff

**v1 anti-pattern**: a coordinator `Agent` with `sub_agents=[...]` and
prompt instructions like *"after the specialist finishes, take control
back and respond to the user."* In practice, control often stuck inside
the sub-agent or the coordinator missed the return.

**v2 replacement**: set `mode='task'` or `mode='single_turn'` on the
sub-agent. Both delegate via the `request_task_{name}` tool (instead of
`transfer_to_agent`) and **auto-return to the coordinator** on finish:

| Mode             | User can chat with sub-agent? | Use for                                              |
|------------------|-------------------------------|------------------------------------------------------|
| `'chat'` (default) | yes (full transfer)           | General assistants, escalation                       |
| `'task'`           | yes (multi-turn, can clarify) | Sub-agent that may need follow-up info               |
| `'single_turn'`    | no (autonomous, one shot)     | Pure utility specialists (look-up, format, classify) |

**Migration recipe**:
1. For each sub-agent, ask: *does the user need to chat with it
   directly?* If no, set `mode='single_turn'`. If only for clarifying
   questions, set `mode='task'`. Otherwise leave `mode` unset (chat).
2. If the sub-agent has `output_schema`, `task` and `single_turn` use
   the schema as the structured return contract automatically — drop
   any "respond in this JSON shape" prompt instructions.
3. Delete coordinator prompt sentences telling it to "take control
   back" — auto-return handles it.

**Why**: stable coordinator/specialist topology. Source:
`adk-workflow` skill ("Agent Modes: Chat, Task, and Single-Turn",
references/task-mode.md); enforced at
`src/google/adk/agents/llm_agent.py:312` (the `mode` Literal).

**Critical sub-pattern — sub-agents with built-in tools**: if a
`single_turn` / `task` sub-agent uses a built-in tool like
`google_search`, also set `disallow_transfer_to_parent=True` and
`disallow_transfer_to_peers=True` on the sub-agent. By default ADK
injects a `transfer_to_agent` function tool on every sub-agent whose
parent transfer is allowed (`agent_transfer.py:152-188`). That
function tool then conflicts with the built-in (Gemini's "Built-in
tools and Function Calling cannot be combined" — the v1 gotcha #24
situation, very much alive in v2). `mode` alone does NOT suppress this
injection — it's a separate code path. The disallow flags are
correct anyway for the call-and-return pattern: `single_turn`
sub-agents auto-return via `request_task` and should never transfer.
This bug typically surfaces only at request time (the static loader
sees no conflict because the injection happens at request build), so
end-to-end tool-call tests are the only reliable way to catch it.

#### 3. `@node` + `RequestInput` + `ctx.resume_inputs` replace ad-hoc HITL

**v1 anti-pattern**: a custom tool that returned a string like
*"Please ask the user to approve and then call this tool again with
their answer"*, plus prompt instructions to "wait for the user." No
checkpoint, no resumability — if the process restarted, state was lost.

**v2 replacement**: a `@node`-decorated function that yields
`RequestInput(message=..., response_schema=...)` to pause execution.
The framework checkpoints, the UI surfaces the prompt, and on resume
the user's answer arrives at `ctx.resume_inputs` (note: the canonical
attribute is `resume_inputs`, **not** `resume_data` — confirmed at
`src/google/adk/runners.py:437` and `workflow/_workflow.py:224`).

**Migration recipe**:
```python
from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow import node

@node(rerun_on_resume=True)
async def approval_gate(ctx: Context, node_input: dict):
    if node_input["amount"] <= 100:
        yield Event(state={"decision": "auto_approved"})
        return
    # Pause until human responds. ctx.resume_inputs will be populated
    # with their answer when execution resumes.
    yield RequestInput(
        message=f"Approve refund of ${node_input['amount']}?",
        response_schema={"type": "string", "enum": ["yes", "no"]},
    )
    decision = ctx.resume_inputs  # human's answer
    yield Event(state={"decision": decision})
```

For multi-step HITL, wrap the workflow in an `App` with
`ResumabilityConfig(is_resumable=True)` so state survives process
restart. For simple one-shot HITL, the non-resumable mode replays the
session events from START — no `App` needed.

**Why**: durable, resumable HITL is a v2 framework guarantee, not a
prompt convention. Source: `adk-workflow` skill ("Human-in-the-Loop",
references/human-in-the-loop.md); checkpoint-resume mechanics in the
`adk-architecture` skill (`references/architecture/checkpoint-resume.md`).

#### 4. `RunConfig` replaces `Gemini(model=..., speech_config=...)` for voice

**v1 anti-pattern**: voice / Live config baked onto the agent itself:
```python
model=Gemini(
    model="gemini-3.1-flash-live-preview",
    speech_config=types.SpeechConfig(voice_config=...),
)
```
Same agent file couldn't run different voices in different sessions.

**v2 replacement**: agent declares only the model id string; voice and
all Live runtime knobs live on `RunConfig`
(`src/google/adk/agents/run_config.py:195`). The web UI's "Live Flags"
panel exposes them at runtime; programmatic callers pass a `RunConfig`
to `runner.run_live(run_config=...)`.

| RunConfig field                    | What it controls                                           |
|------------------------------------|------------------------------------------------------------|
| `speech_config`                    | Prebuilt voice (Zephyr, Kore, Puck, …)                     |
| `response_modalities`              | `["AUDIO"]` / `["TEXT"]`                                   |
| `output_audio_transcription`       | Transcribe model's speech (default on)                     |
| `input_audio_transcription`        | Transcribe user's speech (default on)                      |
| `enable_affective_dialog`          | Native-audio only — emotion-aware responses                |
| `proactivity`                      | Native-audio only — model speaks unprompted when relevant  |
| `session_resumption`               | Reconnect-after-disconnect (Vertex only on Gemini API)     |

**Migration recipe**:
1. Replace `model=Gemini(model="…", speech_config=…)` with `model="…"`
   (just the id string).
2. Drop the `from google.adk.models.google_llm import Gemini` import.
3. If the demo is for `adk web`, voice picking is automatic via the
   Live Flags panel — no agent change needed.

**Caveat (Gemini API only)**: native-audio Live models
(`gemini-2.5-flash-native-audio-*`) currently fail audio-input
negotiation with `adk web`'s bidi worklet on the Gemini API backend —
the connection establishes but no audio is transcribed. Use
`gemini-3.1-flash-live-preview` (half-cascade) on Gemini API; native
audio works on Vertex. Documented in the `level_1a_agent/agent.py`
docstring with the upgrade path.

#### 5. Framework state replaces custom `save_X` / `get_X` scratchpad tools

**v1 anti-pattern**: a pair of custom tools wrapping
`tool_context.state` to let one agent's findings flow to another:
```python
def save_research_note(...): tool_context.state["notes"] = ...
def get_research_notes(...): return tool_context.state["notes"]
```
The LLM had to remember to call `save_X` after every step.

**v2 replacement**: `Event(state={...})` from any node + `{key?}`
instruction injection in the next agent. No tools, no LLM compliance
risk, fewer round-trips.

**Migration recipe**:
1. Replace `save_research_note(topic, finding, source)` with
   `yield Event(state={"findings": findings_md})` from the function
   node that *produces* the data.
2. Replace `get_research_notes()` with `{findings?}` in the
   downstream agent's `instruction`.
3. Delete both custom tools and remove them from `tools=[...]`.
4. If the agent that produced the findings was an LLM agent, use
   `output_key="findings"` on it instead of an explicit
   `Event(state=...)` — same effect, one fewer node.

**Why**: state is now a first-class workflow primitive with delta
flushing onto yielded events (`CHANGELOG-v2.md`: "Supported flushing
state/artifact deltas onto yielded events"). The `level_2_agent`
rewrite drops two custom tools by applying this pattern.

#### 6. Leaf `LlmAgent` is the fast path — don't force-fit `Workflow`

**v1 → v2 cargo-cult risk**: not every v1 `LlmAgent` should become a
`Workflow` graph in v2. For a single-tool, single-turn agent (e.g.,
Level 1's "google search and answer"), wrapping in a `Workflow`
*disables* the v2 leaf optimisation:

> Optimized execution by bypassing the Mesh for leaf single-turn
> `LlmAgent` instances. — `CHANGELOG-v2.md`

**Rule of thumb**: keep an agent as a leaf `LlmAgent` if **all** of:
- One LLM call per turn (no internal multi-step plan).
- One tool or zero tools.
- No branching control flow (no greeting carve-out, no classification).

Promote to `Workflow` only when the v1 prompt was doing real
orchestration that should be deterministic. The `level_1_agent` and
`level_1a_agent` v2 rewrites stay as `LlmAgent`; the `level_2_agent`
rewrite becomes a `Workflow` because its v1 prompt was doing
PLAN/EXECUTE/SYNTHESISE orchestration.

#### 7. `output_schema` + tools no longer needs a workaround

**v1 anti-pattern**: combining `output_schema` with `tools` failed at
the LLM layer (Gemini refused `googleSearch` + function tools in one
request without `bypass_multi_tools_limit=True`, and even with it,
structured output was incompatible).

**v2 replacement**: just set both. The framework now uses an internal
`set_model_response` shim (see
`contributing/samples/output_schema_with_tools/agent.py`) so an agent
can ground via `google_search` *and* return a typed Pydantic object.

**Migration recipe**: drop `bypass_multi_tools_limit=True` from
`GoogleSearchTool(...)` calls; use the bare `google_search` built-in
import. Add `output_schema=YourModel` if you want structured output.

#### Gotchas during migration

| Pitfall | What happens | Fix |
|---|---|---|
| Agent with `output_schema=Foo` feeds a function node typed `node_input: Foo` | `node_input` is actually a **`dict`**, not a Pydantic instance | Type-hint as `dict` and access by key, OR call `Foo.model_validate(node_input)` inside the node |
| `route` node's `Event(route=...)` value doesn't match a key in the routing dict | Workflow halts with no obvious error | Make the classifier's `output_schema` use a `Literal["..."]` so the route values are constrained at the Pydantic boundary |
| START outputs `types.Content`, but downstream function hint is `: str` | v2 auto-extracts `.parts[0].text` for `: str` hints — works, but doesn't generalise to multimodal input | If your demo accepts images/audio, hint as `: types.Content` and inspect `parts` explicitly |
| `parallel_worker=True` on an LLM agent receives a single value (not a list) | Worker produces no output, JoinNode hangs | `parallel_worker` requires the predecessor to output a `list`. For runtime-determined N (e.g., LLM-decided sub-question count), use `ctx.run_node()` inside an `@node` orchestrator instead — see `contributing/workflow_samples/dynamic_fan_out_fan_in/agent.py` |
| Dynamic-fan-out orchestrator without `@node(rerun_on_resume=True)` | If a worker hits HITL, the orchestrator can't resume cleanly | Always decorate dynamic orchestrators with `rerun_on_resume=True`. Lazy-scan dedup ensures only the missing workers re-run on resume (`CHANGELOG-v2.md`) |
| Different terminal nodes write to different `output_key`s | Downstream consumers must inspect which branch ran to find the user-facing text | Use the **same** `output_key` (e.g., `"last_brief"`) on every terminal LLM agent across branches |
| Auto-discovery picks up the `__pycache__` of a deleted demo | Demo appears in `adk web .` picker but loader fails with "No root_agent found" | Delete the orphan `__pycache__` (and `.adk/` session db) when emptying a `level_*_agent/` slot |
| Pre-existing agent samples (`from google.adk.agents.llm_agent import LlmAgent`) vs modern (`from google.adk import Agent`) | Both work in v2; mixing them across files looks messy | For new demos use `from google.adk import Agent` (matches `contributing/samples/google_search_agent/agent.py`); preserve original imports when re-baselining upstream |
| Sub-agent (`mode='single_turn'` / `'task'`) with `tools=[google_search]` (or any built-in) | Runtime error: "Built-in tools and Function Calling cannot be combined." Setting `mode` alone is **not** enough — `transfer_to_agent` is auto-injected on sub-agents from a separate code path (`agent_transfer.py:152-188`) | Set `disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True` on the sub-agent. This is the v2 form of v1's gotcha #24 — caught only by end-to-end tool-call tests, not by static load checks. Applies equally to `code_executor=BuiltInCodeExecutor()` (which is also a built-in surface, even though it's not in `tools=[]`) |
| Sub-agent on Gemini API has both `output_schema=...` and a built-in tool | Runtime error: "Built-in tools and Function Calling cannot be combined." `_OutputSchemaRequestProcessor` injects a `set_model_response` function tool because `can_use_output_schema_with_tools()` returns True only on Vertex (`utils/output_schema_utils.py:31-52`) | Drop `output_schema` on the sub-agent that uses the built-in tool; let it return free text. Keep `output_schema` on downstream sub-agents that don't use built-in tools (analyst, writer, etc.) |
| `mode='task'` agent with `tools=[...]` returns `finish_reason=STOP` and zero output tokens after multi-turn HITL on Flash | The Flash model can't reliably make chained tool decisions when the agent has 3+ overlapping tools (e.g., `transfer_to_agent` + `create_specialist` + `finish_task` all signaling "I'm done"). Empirical pattern: first turn drafts a spec (works), user replies "yes proceed" → empty STOP. Conversation history poisons subsequent turns with the same empty-pattern | Two-step fix: (1) set `disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True` to suppress the redundant `transfer_to_agent` injection, narrowing the tool surface to `[create_specialist, finish_task]`. (2) Upgrade the model to `gemini-3.1-pro-preview` (or `gemini-2.5-pro`) for the agent doing chained tool decisions — Pro 3.1 is described in the model card as supporting "compositional function calling" natively. Confirmed working on the Level 4 `agent_creator` |
| Coordinator's LLM ignores runtime specialists loaded from disk by `before_agent_callback` | The framework hydration is fine — the coordinator's tool list does include the disk specialists. But if the coordinator's *instruction* says runtime specialists "may appear if created earlier in the session", the LLM treats anything from a prior session as "not in this session, must recreate" | Rewrite the instruction to mention persistence explicitly ("PERSISTENT — across server restarts, via `runtime_agents/` library") and add an explicit decision-checklist step ("if a runtime specialist's description matches, call THAT specialist — do NOT recreate") before the gap-detection branch. Otherwise the disk-aware `has_capability` rejection is the only thing stopping needless recreation, costing a wasted creator round-trip every time |
| `adk web --reload_agents` silently misses a file change on Windows | The watchdog handler logs "Change detected" for files that get rewritten, but very-fast successive edits or saves through certain editors don't always trigger an event. Symptom: agent edits don't take effect, callback runs with stale code, behaviour diverges from source | **Do not rely on `--reload_agents` for development.** Standard workflow is kill-and-restart between every edit + test cycle (see "Local conventions" above for the kill snippet). It adds ~3 seconds per cycle but eliminates an entire class of "is the live agent the same as my source?" debugging. If you DO want to use hot-reload, prove it's working with a temporary `print()` in your callback before trusting it |
| Live (bidi audio) agent's chat panel fills with 200+ empty bubbles per voice turn | Gemini Live emits one LLM message per audio chunk, each carrying `usage_metadata` (token counting telemetry). Session resumption (when enabled in Live Flags) adds another stream of `live_session_resumption_update` messages. ADK's runtime emits an Event for each (`base_llm_flow.py:_postprocess_live`), even when there is no user-visible content. v2.0.0b1's adk-web chat panel renders one bubble per Event — so a single voice turn produces 200+ empty bubbles, drowning the actual transcription events. **Important**: `after_model_callback` does NOT fire in Live mode (the Live path goes through `_postprocess_live` which constructs events directly from `LlmResponse`, bypassing `_handle_after_model_callback`). A callback-based fix is silently dead code. Plugins' `on_event_callback` fires but cannot drop events — `runners.py` checks `event.partial` on the **original** event before plugin invocation, so plugin mutation cannot prevent persistence | **Subclass `LlmAgent` and override `_run_live_impl` to filter telemetry-only events from the yielded stream.** The override is structurally identical to the parent's (`llm_agent.py:527-535`) plus one `if _is_telemetry_only_event(event): continue` check. "Telemetry-only" = has `usage_metadata` or `live_session_resumption_update` but no `content`/`error_code`/`interrupted`/`turn_complete`/`input_transcription`/`output_transcription`/`grounding_metadata`. See `level_1a_agent/agent.py` for the implementation (`_LiveTelemetryFilteringAgent`). This is the only agent-level fix that actually works for the Live event stream |
| Pydantic `Field(default_factory=list)` on an `input_schema` / `output_schema` field | Runtime error: `"Object of type _HAS_DEFAULT_FACTORY_CLASS is not JSON serializable"` when ADK generates the Gemini function declaration. The Pydantic v2 sentinel for `default_factory` doesn't survive `json.dumps()` | Use `Field(description=...)` (no default) and require the caller to pass an explicit value. For "optional list" semantics, document in the description: "Pass an empty list if not applicable." Do **not** use `default_factory=...` or `default=[]` on schemas that get sent to Gemini |
| `mode='task'` agent with `input_schema=...` and multi-turn user dialogue | Validation error: `"Input should be a valid dictionary or instance of <Schema>"` on the second user turn. v2 enforces `input_schema` on **every** user turn within the task, not just the initial coordinator-delegated call (input is `types.Content`, fails Pydantic validation) | Drop `input_schema` on `task`-mode agents that need multi-turn HITL ("draft → ask user → wait for yes/no → continue"). The agent's instruction documents the expected shape; v2 uses the default flexible input. `input_schema` is only safe on `single_turn` agents (which only ever receive the structured initial call) |
| Instruction text contains accidental `{var}` literal patterns (e.g., f-string examples in a docstring) | Runtime error: `"Context variable not found: 'var'."` ADK's instruction templater (`utils/instructions_utils.py:124`) regex-matches `{anything}` and tries to look up `anything` in session state. There is **no escape syntax** for literal braces | Rewrite the instruction without `{var}` patterns. For example, `f"chart artists: {n_artists}"` → `print("chart artists:", n_artists)` (comma form). Or use `?` to make missing vars resolve to empty: `{n_artists?}`. The first form is preferred when the brace is not meant to be templated at all |

#### Migration ordering (this fork)

When rewriting v1 demos as v2, follow this order so each rewrite has a
clear teaching focus and doesn't reach for primitives the previous
levels haven't introduced:

1. **Level 1** — leaf `LlmAgent` fast path; `output_key` for state delta.
2. **Level 1a** — same shape, voice; `RunConfig`-driven Live config.
3. **Level 2** — `Workflow` graph; classifier + dict routing; dynamic
   fan-out via `ctx.run_node()`; `Event(state=...)` + `{key?}`
   instruction injection replacing custom scratchpad tools.
4. **Level 3** — sub-agent delegation with `mode='task'` /
   `mode='single_turn'`; auto-return to coordinator; `AgentTool` for
   call-and-return orchestration.
5. **Level 4** — meta-reasoning + dynamic agent creation; runtime node
   spawning from a registry; safety allowlist as a node, not a
   prompt rule. Coordinator gets `PlanReActPlanner()` for visible
   meta-reasoning; `agent_creator` runs on `gemini-3.1-pro-preview`
   for chained tool decisions; all sub-agents with built-in tools
   set `disallow_transfer_to_*` flags to suppress
   `transfer_to_agent` injection.
6. **Level 4a** — Level 4 + MCP toolset (`McpToolset` /
   `StdioConnectionParams`). Structurally a narrow extension of L4:
   `data_fetcher_agent` gains a third tool source (gahmen-mcp for
   Singapore Government data) alongside `google_search` and
   `load_web_page`. Runtime safety allowlist gains 8 MCP sentinel
   entries that resolve to per-spec narrowed `McpToolset` via
   `tool_filter`. **Prerequisite**: vendor the gahmen-mcp server at
   `level_4a_agent/vendor/gahmen-mcp/` with a stdio entry wrapper
   (`src/stdio_entry.ts` adapting `createStatelessServer` to
   `StdioServerTransport`); without it the agent loads but MCP tool
   calls fail at subprocess launch. The framework error is graceful
   (logged, not crashing) — a teachable v2 resilience moment.

### V1_* auto-discovery

The V1_* folders have no `__init__.py`, but ADK 2.0's loader treats them
as PEP 420 namespace packages and finds `agent.py` directly. They appear
in the `adk web .` picker alongside the v2 rewrites — that's a feature
(live before/after reference), not a bug. To hide a V1 demo without
deleting it, rename its `agent.py` (e.g., `agent.py.v1ref`).

### Migration history (informal)

The v1.31 → v2.0.0b1 transition is recorded in:

- `backup/pre-v2-migration` — branch + tag preserving the pre-migration
  `main` HEAD.
- The git log on `v2-migration` — commit `d26f7cf6` added the original
  v1 demos; commit `fc6ccf4b` archived them as `V1_*`. Levels 1a, 4,
  and 4a were deleted outright in that archive (not renamed) — they
  must be restored from `git show 18b3b909:level_1a_agent/agent.py`
  etc. before being prefixed `V1_`. This was done for `level_1a` when
  its v2 rewrite landed; do the same for 4 / 4a when their rewrites
  begin.

There is no formal migration plan document — earlier versions of this
file referenced `migration/v2_migration_plan.md`, which never made it
into the repo.

### Local conventions

- **Dev workflow: kill the server and restart before every test, do
  NOT rely on `--reload_agents`.** ADK caches agent modules in
  `sys.modules` after first load. The `--reload_agents` flag exists
  but its file watcher is unreliable on Windows — we observed silent
  missed-changes twice in one session (no "Change detected" log
  entry, callback running with stale code, behaviour diverging from
  the on-disk source). The reliable sequence after every edit:
  ```powershell
  # PowerShell on Windows
  netstat -ano | Select-String ":8000.*LISTENING" | ForEach-Object {
      ($_ -split "\s+")[-1] | ForEach-Object { taskkill /PID $_ /F }
  }
  adk web .
  ```
  Or in the Bash tool: `netstat -ano | grep :8000 | grep LISTENING | awk '{print $5}' | head -1 | xargs -I {} taskkill //PID {} //F` then re-launch. Adds ~3 seconds to the
  edit/test loop but eliminates the entire "is the live agent the same
  as my source?" debugging class. If you DO want hot reload, prove it's
  working by adding a temporary `print()` to the callback and checking
  the log before trusting it.
- Don't edit `src/google/adk/` locally to fix a demo — fix the demo.
- Don't merge `v2-migration` into `main` until all v2 demos are written
  and validated.
- When rewriting a v1 demo as v2: first restore the v1 source from git
  into a `V1_<name>/` folder (if not already archived), then write the
  fresh v2 version in `<name>/`. Don't copy v1 source into the v2 slot
  — write from scratch using v2 idioms. Walk through the **"v1 → v2
  migration patterns"** section above to identify which patterns apply
  before writing code; cross-reference with the `adk-workflow` and
  `adk-architecture` skills for the framework-level details.
- The v1-era `AGENT_LEVELS.md` ladder is historical; the v2 rewrites
  should reuse the *taxonomy* (Level 1 = connected, Level 2 =
  strategic, etc.) but redesign the *implementation* around v2
  primitives. The v1 ladder is preserved in git at commit `18b3b909`
  if needed as reference.
