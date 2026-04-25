# Agent Levels — Design & Comparison

This document outlines the progression of agent designs built in this project. Each level maps to [Google's agent taxonomy](https://cloud.google.com/discover/what-are-ai-agents) from the *Introduction to Agents* whitepaper. The levels represent **cumulative capabilities** — each level adds new abilities on top of the previous.

> **Note:** All tool-using agents (Level 1+) use the fundamental reason → act → observe cycle. The levels are NOT about different thinking patterns — they're about what capabilities the agent has.

---

## v1 → v2 update notes

This document was originally written against ADK 1.31. The **taxonomy is
unchanged for v2** — the capability ladder L0 → L1 → L2 → L3 → L4 → L4a is
still correct. What changed is the **wiring**: the v2 framework introduced
new primitives (graph workflows, `mode='task'/'single_turn'`,
`McpToolset`, leaf-`LlmAgent` fast path, `PlanReActPlanner` first-class,
disk-backed disallow_transfer flags) that replace some of the v1
hand-rolled patterns. Each level's section below has a **"v2 wiring update"**
callout summarising what changed.

For the full migration playbook (7 patterns + 12-row gotchas table),
see [`AGENTS.md`](AGENTS.md) — the "v1 → v2 migration patterns" section.
For the actual v2 source code of each demo, see the corresponding
`level_*_agent/` folder; the v1 originals are preserved for reference
under `V1_level_*_agent/`.

The biggest single change across the ladder: **fixed-team specialists
that v1 wrapped manually as `tools=[AgentTool(agent=X), ...]` are now
wired as `sub_agents=[...]` with a non-`chat` `mode`**. The framework
auto-derives the wrapper at `model_post_init` time. Runtime-injected
specialists (those created mid-session via Level 4's `agent_creator`)
still wrap as `AgentTool` because the auto-derivation is init-time only.

---

## Level 0 — `my_first_agent` (Core Reasoning System)

> **v2 wiring update**: NOT MIGRATED. The v1 demo lives in
> `V1_my_first_agent/` for reference; the v2 ladder starts at L1
> because the leaf-`LlmAgent` fast-path (introduced in v2) is exactly
> what L1 demonstrates, making L0 redundant. To run on v2, the v1
> source still works as-is.

**Taxonomy:** Level 0 — The Core Reasoning System

**New capability:** None — LLM in isolation

A minimal agent to verify the ADK setup. Single tool (`greet`), no external data, no state. Operates on pre-trained knowledge only.

- **Reasoning mode** — **none** by design. LLM in isolation with a single tool; no multi-step reasoning is possible or needed.

```
User → Agent → greet() → Response
```

### Flow

```
 [User message]
       |
       v
 +----------------------+
 | LlmAgent             |
 | my_first_agent       |
 +----------------------+
       |
       v
  /------------------\
 <   Name given?      >
  \------------------/
       |          |
      yes         no
       |          |
       v          |
 +------------+   |
 | greet(name)|   |
 +------------+   |
       |          |
       v          v
 +----------------------+
 | LlmAgent formats     |
 | reply                |
 +----------------------+
       |
       v
   (( Done ))
```

One optional tool call. No iteration, no state. Turn ends after the first model reply.

---

## Level 1 — `level_1_agent` (Connected Problem-Solver)

> **v2 wiring update**: structurally identical to v1 (still a single
> `LlmAgent` with `tools=[google_search]`), but now runs on v2's
> **Mesh-bypass leaf fast path** (`CHANGELOG-v2.md`). One small
> addition: `output_key="last_answer"` so the response auto-flushes
> as a `state_delta` on every yielded `Event` — visible in the State
> panel and in `Event.actions.state_delta` for downstream consumers
> (eval harnesses, transcript loggers). No prompt changes. v2 source
> at `level_1_agent/agent.py`.

**Taxonomy:** Level 1 — The Connected Problem-Solver

**New capability:** External tools (connection to the live world)

The agent connects to external tools to answer questions that require real-time information. Simple serial flow: identify what's needed → search → answer. One tool call per question.

- **Uses ADK's built-in `google_search` tool** — zero custom code, pure ADK configuration.
- **Serial flow** — identify need → call tool → use result to answer.
- **No planning** — handles one straightforward question at a time.
- **No state** — no persistent memory between tool calls.
- **Modeled after** the official ADK `google_search_agent` sample.
- **Reasoning mode** — **none** by design. A one-shot search-and-answer flow; explicit reasoning adds cost without quality gain. Reasoning is introduced in Level 2+ as the taxonomy's next capability jump.
- **Demonstrates:** ADK built-in tool use, the basic reason → act → observe cycle.

```
User → Identify need → google_search → Answer
```

### Flow

```
 [User question]
       |
       v
 +----------------------+
 | LlmAgent             |
 | level_1_agent        |
 +----------------------+
       |
       v
  /---------------------\
 <  Need real-time info? >
  \---------------------/
       |          |
      yes         no
       |          |
       v          |
 +----------------+|
 | google_search  ||
 | (built-in)     ||
 +----------------+|
       |          |
       v          v
 +----------------------+
 | LlmAgent synthesises |
 | grounded answer      |
 +----------------------+
       |
       v
   (( Done ))
```

Exactly one tool call when external info is needed. No planning, no loops, no state between turns.

---

## Level 1a — `level_1a_agent` (L1 + voice)

> **v2 wiring update**: significant changes.
>
> - **Voice config moved off the agent and onto `RunConfig`**
>   (`run_config.py:195`). Agent declares only `model="gemini-3.1-flash-live-preview"`
>   as a plain string — no more `Gemini(model=..., speech_config=...)`
>   wrapper. Voice (Zephyr / Kore / Puck), audio transcription
>   defaults, affective dialog, proactivity all become Live-Flags-panel
>   knobs at runtime. Same agent file can use Zephyr in one session
>   and Kore in another.
> - **`_LiveTelemetryFilteringAgent` subclass overrides `_run_live_impl`**
>   to drop telemetry-only events (`usage_metadata` / `live_session_resumption_update`
>   / `turn_complete` with no content). Without this, a single voice
>   turn produces 200+ empty bubbles in the chat panel because Gemini
>   Live emits one LLM message per audio chunk. **NB: an
>   `after_model_callback` does NOT work for Live mode** — the Live
>   path goes through `_postprocess_live` which bypasses
>   `_handle_after_model_callback`. Plugins also can't drop events
>   (the runner checks `event.partial` on the original event before
>   plugin invocation). The `_run_live_impl` override at the agent
>   yield boundary is the only filter point that works.
> - **Native-audio caveat**: `gemini-2.5-flash-native-audio-*` on Gemini
>   API + adk-web's bidi worklet currently fails to negotiate audio
>   input (server-side issue, not framework). Half-cascade
>   `gemini-3.1-flash-live-preview` is the v1-proven working path on
>   Gemini API. Native audio works on Vertex.
>
> v2 source at `level_1a_agent/agent.py` (~227 lines incl. extensive
> docstrings).

**Not a new taxonomy level.** A variant of Level 1 with a Gemini Live model wired in place of the stock text Flash, so the agent can be spoken to via the mic in adk web and responds with synthesised speech. Default model is `gemini-3.1-flash-live-preview` (half-cascade Live on Gemini API); voice is Zephyr by default (change `_VOICE_NAME` in the agent file to pick a different prebuilt voice).

**Why L1a, not an L3a clone with voice.** We originally built a multi-agent voice variant (`level_3a_agent`) but removed it. L3's `sub_agents` → `transfer_to_agent` path requires the model to emit a structured function call in bidi audio mode, which `gemini-3.1-flash-live-preview` does not reliably do — observed symptom was the coordinator greeting successfully then silently freezing on delegation turns, regardless of `thinking_level` (tried minimal, LOW, MEDIUM). The underlying `gemini-2.5-flash-native-audio-*` family does emit transfer calls but intermittently throws 1011 WebSocket internal errors on delegation-heavy turns. Neither mode is stable enough for a teaching demo. L1a removes the entire failure surface by dropping sub-agents: `google_search` is a **built-in** tool (handled internally by the model, not emitted as a function call) so voice + search work together without requiring the capability gap.

- **Single agent, single voice** — one `LlmAgent` with `model=Gemini(model=..., speech_config=...)`. No sub_agents means no auto-injected `transfer_to_agent`, so no `bypass_multi_tools_limit=True` needed either (see AGENTS.md gotcha #24 — applies only to sub-agents).
- **Voice-output discipline** — same "speak, don't write" rules we used in L3a (no markdown, no numbered citations, no URL recitation, inline domain attribution, no list enumeration), applied directly to the single agent's instruction. TTS pronounces every character literally, so text that would render cleanly in adk web's text bubble becomes painful audio without these rules.
- **Greeting carve-out** — the instruction handles "hi" / "hello" / "what can you do?" directly without triggering a search, with one short spoken sentence plus two voice-friendly example questions. Same teaching pattern we developed for L3 and L3a.
- **Voice-only on Gemini API** — all currently-available Gemini API Live models support only `bidiGenerateContent`, not `generateContent`. Typing text fails with HTTP 404 until a dual-mode Live model ships on Gemini API. Click the mic.
- **How to run** — `adk web .` from the repo root; select `level_1a_agent`, click the phone/mic icon, speak.
- **Demonstrates:** the minimum viable voice agent in ADK — `Gemini` model wrapper + `speech_config` + a built-in tool. Proof that Level 1's single-tool-call architecture maps cleanly to voice without any topology change.

---

## Level 2 — `level_2_agent` (Strategic Problem-Solver)

> **v2 wiring update**: this is the level that changed the most.
> v1 was a single `LlmAgent` with a long instruction *asking* the LLM
> to follow PLAN/EXECUTE/SYNTHESISE steps, plus two custom scratchpad
> tools (`save_research_note` / `get_research_notes`). v2 expresses
> the same workflow as a **`Workflow(BaseNode)` graph**:
>
> ```
> START → process_input → classify → ┬── greeting → greeter → END
>                                    └── research → planner →
>                                       fan_out_research → writer → END
> ```
>
> - **Classifier + dict routing** replace the v1 instruction's
>   "if greeting, respond directly" prose branch.
> - **Dynamic fan-out via `ctx.run_node()`** replaces the v1
>   prompt-driven sub-question loop. The planner LLM decides how many
>   sub-questions to spawn at runtime; one researcher per sub-question
>   runs concurrently. Pattern lifted from
>   `contributing/workflow_samples/dynamic_fan_out_fan_in/`.
> - **`Event(state={...})` + `{key?}` instruction injection** replaces
>   `save_research_note` / `get_research_notes` custom tools. The
>   orchestrator writes findings to `state["findings"]`; the writer
>   reads via `{findings?}`. Fewer tools, no LLM compliance risk.
> - **`@node(rerun_on_resume=True)`** on the dynamic-fan-out
>   orchestrator so it's HITL-resumable.
>
> v2 source at `level_2_agent/agent.py`. Compare side-by-side with
> `V1_level_2_agent/agent.py` — the v1 prompt's "1. PLAN: 2. EXECUTE:
> 3. SYNTHESISE:" instruction collapses into a graph that *enforces*
> those steps structurally rather than relying on the LLM to follow them.

**Taxonomy:** Level 2 — The Strategic Problem-Solver

**New capability:** Strategic planning + Context engineering

The agent can now handle **complex, multi-part goals** by planning a strategy upfront, executing multiple chained tool calls where each step feeds the next, and managing a scratchpad to accumulate context across steps.

- **ADK built-in `GoogleSearchTool`** with `bypass_multi_tools_limit=True` to mix with custom tools.
- **Custom tools for business logic only** — `save_research_note`, `get_research_notes` (scratchpad via `ToolContext.state`).
- **Plans before acting** — decomposes complex questions into 2–3 sub-questions.
- **Context engineering** — uses `ToolContext.state` scratchpad to curate findings across steps.
- **Structured output** — Research Brief with Key Findings, Analysis, Sources, Confidence & Gaps.
- **Reasoning mode** — **none** (no `planner` configured). The "PLAN / EXECUTE / SYNTHESISE" structure is prompt-driven, not framework-enforced. Gemini 2.5 still thinks natively, but thoughts are hidden. To make it an explicit reasoning agent, set `planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True))` or `planner=PlanReActPlanner()`.
- **Demonstrates:** Upfront planning, context engineering via `ToolContext.state`, scratchpad pattern.

```
User → PLAN (decompose) → EXECUTE (search + save) × 2–3
     → SYNTHESISE (get_research_notes → cross-reference)
     → DELIVER (structured report)
```

### Flow

```
 [User question]
       |
       v
 +------------------------------+
 | PLAN: decompose into         |
 | 2-3 sub-questions            |
 +------------------------------+
       |
       v
 +==================================+
 |  LOOP: once per sub-question     |
 |                                  |
 |   /------------------------\    |
 |  <  More sub-questions?     >---+---> exit loop
 |   \------------------------/    |      when "no"
 |            |                    |
 |           yes                   |
 |            v                    |
 |   +----------------------+      |
 |   | GoogleSearchTool     |      |
 |   | bypass_multi_tools   |      |
 |   +----------------------+      |
 |            |                    |
 |            v                    |
 |   +----------------------+      |
 |   | save_research_note   |      |
 |   | ToolContext.state    |      |
 |   | (append to list)     |      |
 |   +----------------------+      |
 |            |                    |
 |            +-- back to check ---+
 |                                  |
 +==================================+
       |
       v
 +---------------------------+
 | get_research_notes        |
 | (read whole scratchpad)   |
 +---------------------------+
       |
       v
 +---------------------------+
 | SYNTHESISE: cross-ref,    |
 | format structured brief   |
 +---------------------------+
       |
       v
   (( Done ))
```

The loop is the new thing at L2. A single agent executes N search-and-save cycles, accumulating findings in `ToolContext.state["research_notes"]`, then a final synthesis step reads the whole scratchpad and produces the structured brief.

---

## Level 3 — `level_3_agent` (Collaborative Multi-Agent System)

> **v2 wiring update**: the delegation primitive changed.
>
> - **`tools=[AgentTool(agent=X), ...]` → `sub_agents=[...]` with
>   `mode='single_turn'` on each specialist.** The framework
>   auto-derives `_SingleTurnAgentTool` instances on the coordinator
>   at `model_post_init` time (`llm_agent.py:982-994`). The
>   coordinator delegates via auto-generated `request_task_<name>`
>   function tools instead of manually-wrapped `AgentTool`s. Same
>   call-and-return semantics, less boilerplate.
> - **Pydantic `input_schema` / `output_schema`** on every specialist
>   define typed contracts at the boundary. Replaces v1's
>   prose-only ("save a note with topic, finding, source")
>   instructions. The `report_writer_agent` returns a typed `Brief`
>   model.
> - **`disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True`**
>   on every sub-agent with a built-in tool — required to suppress
>   ADK's auto-injected `transfer_to_agent` function tool, which
>   otherwise conflicts with `google_search` (the v1 gotcha #24
>   echo, alive in v2 at the Gemini API layer; suppression is the
>   v2-canonical fix). Caught only by end-to-end tool-call tests,
>   not static load checks.
> - **`PlanReActPlanner()` on the coordinator** — kept from v1; still
>   the right choice for runtime LLM-decided routing. Surfaces
>   `/PLANNING/ /REASONING/ /ACTION/ /FINAL_ANSWER/` blocks inline.
>
> v2 source at `level_3_agent/agent.py`. The pedagogical split between
> Levels 2 and 3 is now sharper: L2 uses `Workflow(BaseNode)`
> graph-orchestration (deterministic routing); L3 uses
> `LlmAgent`+`sub_agents` (LLM-decided runtime routing). Pick by
> whether the orchestration order is data-driven or content-driven.

**Taxonomy:** Level 3 — The Collaborative Multi-Agent System

**New capability:** Delegation to specialist agents + explicit plan-then-act reasoning

The paradigm shifts from a single agent to a **team of specialists**. A coordinator delegates work to specialist agents wrapped as `AgentTool`s, each with their own tools and expertise. The coordinator does NO work itself — it only orchestrates — and its reasoning is made visible via `PlanReActPlanner`.

- **4 agents** — 1 coordinator + 3 specialists (`search_agent`, `analyst_agent`, `writer_agent`).
- **Division of labour** — each specialist has a distinct role and non-overlapping tool set.
- **`AgentTool` for delegation, not `sub_agents`** — specialists are wrapped as `AgentTool(agent=X)` on the coordinator's `tools=[...]` list. ADK exposes two multi-agent primitives (`AGENTS.md:267`): `sub_agents` for conversation-transfer semantics (escalation), and `AgentTool` for call-and-return delegation (orchestration). A research coordinator that chains search → analyst → writer is doing orchestration, so `AgentTool` is the right primitive. Same pattern `level_4_agent` uses — L3→L4 now shares a foundation rather than arbitrarily using different ADK APIs.
- **Plain `google_search` built-in works directly** — because `AgentTool`-wrapped agents don't get `transfer_to_agent` auto-injected (unlike `sub_agents`), there's no function + built-in conflict to work around. No `GoogleSearchAgentTool` nested wrap needed. Cleaner, one layer fewer at runtime.
- **ADK-native data flow** — no custom glue code. Data moves between specialists via:
  - `output_key` — auto-writes a specialist's final text to session state (works identically with `AgentTool` as with `sub_agents`).
  - `{state_key?}` instruction injection — auto-reads state into the next agent's prompt.
  - `ToolContext.state` — for list-accumulation across multiple tool calls (scratchpad notes).
- **Reasoning mode** — **`PlanReActPlanner()` on the coordinator**. Prompt-level ReAct scaffold that forces the coordinator to emit an explicit PLANNING / REASONING / ACTION / FINAL_ANSWER structure before calling tools. Makes the multi-agent orchestration visible inline in the chat stream — learners see the coordinator decide "which specialist next?" rather than just observing the end result. Leaf specialists stay unplanned; their jobs (search, save, format) are mechanical. Alternative is `BuiltInPlanner(thinking_config=...)` which surfaces Gemini's native thinking as separate THOUGHT events in the trace panel — inline-scaffold was chosen because planning IS the coordinator's job so the plan belongs in the main output, not a side channel.
- **Demonstrates:** Multi-agent orchestration via `AgentTool` delegation, ADK-native inter-agent state flow, and explicit plan-then-act reasoning.

### Sub-Agent Roles

| Agent | Role | Tools | State I/O |
|---|---|---|---|
| **root (coordinator)** | Plans with `PlanReActPlanner()`, then calls specialists as tools | `AgentTool(search_agent)`, `AgentTool(analyst_agent)`, `AgentTool(writer_agent)` | — |
| **search_agent** | Searches the web | Plain `google_search` built-in — no wrap needed because this agent is wrapped as `AgentTool` (not `sub_agent`), so no `transfer_to_agent` is auto-injected, so there's no built-in + function conflict | writes → `last_search_result` via `output_key` |
| **analyst_agent** | Extracts findings from searches and reviews accumulated notes | `save_research_note` | reads `{last_search_result?}`, `{research_notes?}`; writes `research_notes` via `ToolContext.state` |
| **writer_agent** | Formats the final structured report | None — pure LLM reasoning | reads `{research_notes?}` |

```
User → Coordinator
         ├── delegates → search_agent (google_search) ──▶ state.last_search_result
         ├── delegates → analyst_agent (save_research_note) ──▶ state.research_notes (append)
         └── delegates → writer_agent (reads state.research_notes) ──▶ structured brief
```

### Flow

**Overview — the hub-and-spoke:**

```
          [User question]
                |
                v
    +===========================+
 +->|  COORDINATOR (LlmAgent)   |
 |  |  no tools, only sub_agents|
 |  +===========================+
 |              |
 |              v
 |      /-----------------\
 |     <  Which specialist? >
 |      \-----------------/
 |        |     |      |      |       \
 |     search  save  review  writer   enough info
 |        |     |      |      |         |
 |        v     v      v      v         v
 |     [spec] [spec] [spec] [spec]  (( Done:
 |        |     |      |      |      direct
 |        |     |      |      |      answer ))
 |        |     |      |      v
 |        |     |      |   (( Done:
 |        |     |      |    structured
 |        |     |      |    brief ))
 |        |     |      |
 +--------+-----+------+
      return edge: after each specialist finishes,
      control snaps back to the coordinator.
```

**Zoom — what each specialist does and what it writes to state:**

```
  search_agent                 analyst_agent (SAVE)
  +-------------------+        +----------------------------+
  | tools:            |        | reads: {last_search_result}|
  |   google_search   |        | tools:                     |
  |                   |        |   save_research_note       |
  | output_key:       |        |                            |
  |   last_search_    |        | effect: appends a note to  |
  |   result          |        |   state.research_notes     |
  |                   |        |   via ToolContext.state    |
  | effect: writes    |        +----------------------------+
  |   state.last_     |
  |   search_result   |        analyst_agent (REVIEW)
  +-------------------+        +----------------------------+
                               | reads: {research_notes}    |
  writer_agent                 | tools: none                |
  +-------------------+        | effect: summarises         |
  | reads:            |        |   patterns/gaps            |
  |   {research_notes}|        |   (text to coordinator)    |
  | tools: none       |        +----------------------------+
  | effect: produces  |
  |   structured      |
  |   brief (final    |
  |   reply)          |
  +-------------------+
```

The hub-and-spoke shape is the new thing at L3. After every specialist finishes, control returns to the coordinator, which decides whether to delegate again or terminate. The coordinator never runs a tool itself — all work happens inside specialists, and all data crossing between them flows through session state via three ADK primitives:

- **`output_key`** (write) — `search_agent`'s final text auto-lands in `state.last_search_result`.
- **`ToolContext.state`** (append-write) — `save_research_note` appends to the `state.research_notes` list.
- **`{state_key?}` instruction injection** (read) — `analyst_agent` and `writer_agent` read state directly from their prompts; no read-tool needed.

Two terminal states: the coordinator either routes to `writer_agent` for a structured brief, or — for a simple factual question that one search already answered — replies directly without invoking the writer.

---

## Level 4 — `level_4_agent` (Self-Evolving System)

> **v2 wiring update**: the most-evolved demo. Inherits all L3 changes
> (sub_agents+mode, disallow flags, typed schemas) and adds:
>
> - **`agent_creator` runs on `gemini-3.1-pro-preview`**. Multi-turn
>   HITL ("draft spec → ask user → call create_specialist → call
>   finish_task") is a chained tool decision; Gemini Flash exhibits
>   empty-STOP responses on this chain. Pro 3.1's compositional
>   function calling handles it reliably.
> - **`mode='task'` on `agent_creator`** for multi-turn HITL —
>   structurally cleaner than v1's "transfer to sub_agent and beg the
>   LLM to transfer back" pattern. `task` mode delegates via
>   `request_task_<name>` and auto-returns on `finish_task`.
> - **`disallow_transfer_to_parent=True`** also on `agent_creator` —
>   without this, the auto-injected `transfer_to_agent` adds a third
>   "I'm done" tool alongside `create_specialist` and `finish_task`,
>   causing tool-choice paralysis.
> - **`PlanReActPlanner()` on the coordinator** — newly added in v2.
>   Surfaces the meta-reasoning ("can my team handle this? does a
>   runtime specialist match? do I need agent_creator?") inline in
>   the chat panel. Half the L4 teaching value (visible
>   meta-reasoning) was missing from the initial rewrite; this restored
>   it.
> - **Restart-aware persistence**. v1 was session-only — runtime
>   specialists died with the session. v2 keeps the session-state
>   capability registry AND reads the on-disk `runtime_agents/*.yaml`
>   library on every turn. `has_capability` checks both, so a fresh
>   session asked to "build f1_data_agent" sees the existing YAML and
>   short-circuits rather than silently overwriting.
> - **Safety allowlist expanded**: 4 tools (`google_search`,
>   `get_current_date`, `calculator`, `load_web_page`). The
>   `calculator` is a safe AST-based math evaluator (in `tools.py`)
>   that rejects all injection attempts; the others are v2 built-ins
>   or function tools.
> - **Runtime specialists still wrap as `AgentTool`** (NOT
>   `sub_agents`) — because v2's `sub_agents → _SingleTurnAgentTool`
>   auto-derivation is init-time only. Mutating `sub_agents` mid-session
>   wouldn't register new tools. **Pedagogical split: fixed teams use
>   `sub_agents`; runtime teams use `AgentTool`.**
> - **The chart-rendering pattern from v1 carries forward unchanged**.
>   The hardened single-cell `plt.show()` + defensive `n_artists` check
>   defeats the blank-PNG bug under v2's `mode='single_turn'` shape too;
>   verified end-to-end (4-quarter Apple chart, 5-year NVIDIA dual-axis).
>
> v2 source: `level_4_agent/agent.py` + `safety.py` + `registry.py` +
> `creator_tools.py` + `tools.py` (~1100 lines total). Compare side-by-side
> with `V1_level_4_agent/`.

**Taxonomy:** Level 4 — The Self-Evolving System

**New capability:** Meta-reasoning + dynamic agent creation

The team is no longer fixed. A coordinator can detect that its specialists cannot cover the user's request and **spawn a new specialist** on demand. Once created, the new agent is available for delegation for the rest of the session. The system moves from "fixed team" (L3) to "team that grows when it hits a gap."

The Level 4 demo in this project is a **Business Intelligence (BI) Agent Team**. The fixed team answers analytical business questions (*"Compute Tesla FY24 revenue growth and compare vs industry"*). When a question requires a capability outside the fixed team (*"pull our live Salesforce pipeline"*), the coordinator delegates to `agent_creator`, which drafts a spec, asks the user to confirm, and registers a new specialist in session state.

- **5 starting agents + N runtime specialists** — 1 coordinator + 3 fixed specialists (`data_fetcher_agent`, `analyst_agent`, `report_writer_agent`) + 1 `agent_creator` sub-agent + any specialists created mid-session.
- **Every fixed specialist is an `AgentTool`** wrapping a leaf `LlmAgent`. The coordinator synthesizes their outputs like typed function returns. `agent_creator` is the only `sub_agent` because creation is a multi-turn conversation.
- **`BuiltInCodeExecutor` for math + charts** — `analyst_agent` does arithmetic, pandas exploration, and matplotlib chart generation in Gemini's server-side hosted Python sandbox (the framework just sets the `code_execution` tool flag; it does not run Python itself). Charts propagate to `adk web` via the framework's auto-save: `_code_execution.py:279-318` intercepts every `inline_data` image Part returned by `BuiltInCodeExecutor` and saves it to the configured `artifact_service` with an auto-generated `YYYYMMDD_HHMMSS.png` filename. The analyst is told to call ONLY `plt.show()` (never `savefig`, never `tool_context.save_artifact` — `tool_context` doesn't exist inside the sandbox; it's a runner-side object injected into Python function tools, not into Gemini's hosted code-execution environment).
- **`BuiltInPlanner` on `analyst_agent`** — Flash + `BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True))` is the AGENTS.md-sanctioned "reasoning model" knob for code-exec leaves. Pro on a `BuiltInCodeExecutor` agent has been observed to hang 6+ min under AFC (gotcha #21), so explicit thinking is layered on Flash instead. The planner makes the model plan its code-cell layout before writing code, which directly prevents the multi-cell drift that produces blank artifact versions (gotcha #20).
- **Runtime specialists persist across turns** — stored in `state.capabilities` as plain dicts; `before_agent_callback` rebuilds `root_agent.tools = _FIXED_TOOLS + hydrate_capabilities(state)` each turn.
- **Safety allowlist** — `safety.ALLOWED_TOOLS` caps what a dynamic specialist can touch. A runtime spec requesting an unknown tool name is rejected before construction.
- **HITL confirmation** — `agent_creator` is instructed to ask the user before calling `create_specialist`. Prompt-level gate, not framework-level Tool Confirmation.
- **Reasoning mode** — **none**. Meta-reasoning is prompt-driven (the coordinator's instruction tells it *"can my team handle this?"*). Explicit `BuiltInPlanner` / `PlanReActPlanner` is not wired.
- **Demonstrates:** `AgentTool` reliability patterns, `BuiltInCodeExecutor` with artifact-based chart propagation, dynamic Python agent construction, session-scoped capability registry, safety allowlist.

### Agent Roles

| Agent | Role | Tools | How it joins the coordinator |
|---|---|---|---|
| **root (`bi_coordinator`)** | Meta-reasoning router. Orchestrates fixed team; transfers to creator on gaps. | None directly — 3 `AgentTool`s + 1 `sub_agent` | N/A |
| **`data_fetcher_agent`** | Web search for business data | `google_search` built-in | `AgentTool(propagate_grounding_metadata=True)` |
| **`analyst_agent`** | All arithmetic + pandas + matplotlib charts | `code_executor=BuiltInCodeExecutor()`, `planner=BuiltInPlanner(thinking_config=...)` | `AgentTool()` — charts auto-saved as artifacts by `_code_execution.py` post-processor |
| **`report_writer_agent`** | Formats final BI brief with `{last_search_result?}` / `{last_analysis?}` state injection | None | `AgentTool(skip_summarization=True)` |
| **`agent_creator`** | Drafts + registers new specialists on demand | `create_specialist` function tool | `sub_agent` — conversational delegation |
| **runtime specialists** | Fills a capability gap (e.g., `crm_data_agent`, `f1_data_agent`) | Any from the safety allowlist | `AgentTool()` re-hydrated from `state.capabilities` each turn |

### Flow

```
                      [User question]
                             |
                             v
         +=====================================+
      +->|  BI COORDINATOR (LlmAgent)          |
      |  |  tools = FIXED_TOOLS + runtime_tools|
      |  +=====================================+
      |              |
      |              v
      |     /-------------------------\
      |    <  Can the current team      >
      |    <  handle this? (meta)       >
      |     \-------------------------/
      |          |                 |
      |         yes               no (GAP)
      |          |                 |
      |          v                 v
      |   delegate via      transfer to
      |   AgentTool:        agent_creator
      |    - data_fetcher   sub_agent:
      |    - analyst        1. draft spec
      |    - report_writer  2. ASK USER
      |    - runtime_spec   3. create_specialist
      |          |          4. validate+dedupe
      |          |          5. persist to
      |          |             state.capabilities
      |          |                 |
      |          v                 v
      |       (( Done ))   (new specialist
      |                     available NEXT turn
      |                     after re-hydration ))
      |                           |
      +---------------------------+
        before_agent_callback on next turn:
         tools = FIXED_TOOLS + hydrate_capabilities(state)
```

**Two terminal paths.** Either the coordinator orchestrates fixed + runtime tools to produce a final BI brief, or it transfers to `agent_creator` and the *user* confirms creation. After confirmation, the new specialist joins the team for the rest of the session.

**State primitives used:**

- `output_key` on `data_fetcher_agent` / `analyst_agent` → `state.last_search_result`, `state.last_analysis` for `report_writer_agent` to read via `{state_key?}` injection.
- `state.capabilities` — list of plain dicts, one per runtime specialist. Grows across turns. Rebuilt into `AgentTool`s by `hydrate_capabilities` on each invocation.
- `ToolContext.state` — mutated by `create_specialist` (a Python function tool, where `tool_context` is injected) to append new capabilities. The analyst does NOT mutate state directly: chart artifacts are produced by `BuiltInCodeExecutor` returning `inline_data` Parts, which the framework then auto-saves via `invocation_context.artifact_service` — no `tool_context` involved.

---

## Level 4a — `level_4a_agent` (L4 + MCP data source)

> **v2 wiring update**: structurally a *narrow* extension of L4 — the
> intentional "diff is one agent + one toolset". v2 inherits all L4
> changes (sub_agents+mode, disallow flags, Pro 3.1 creator,
> PlanReActPlanner on coordinator, disk-persistence) and adds:
>
> - **`data_fetcher_agent` becomes a 3-source agent**: `google_search`
>   (general web), `load_web_page` (specific URL), AND `McpToolset`
>   wrapping gahmen-mcp (8 read-only `datagovsg_*` / `singstat_*`
>   tools, narrowed via `tool_filter`). v1 was a *replacement* (only
>   MCP); v2 is *additive* — the LLM picks the right source per query
>   via the fetcher's instruction.
> - **`bypass_multi_tools_limit=True`** on the fetcher's `GoogleSearchTool`
>   triggers the v2 auto-swap to `GoogleSearchAgentTool` (function-
>   wrapped) when `multiple_tools=True`. Combined with `load_web_page`
>   and the 8 MCP tools, the final shape sent to Gemini is 10 function
>   tools + 0 built-ins → no Gemini "built-in + function" conflict.
> - **Safety allowlist sentinel pattern carried forward from v1**:
>   8 MCP tool names map to a `_MCP_SENTINEL` placeholder; per-runtime
>   `safety.resolve_tools()` swaps each set into a fresh
>   `McpToolset(tool_filter=[...])`. Each runtime specialist gets only
>   the MCP tools it asked for — enforced at the framework boundary.
> - **Stdio entry wrapper** (`vendor/gahmen-mcp/src/stdio_entry.ts`,
>   ~15 lines) bridges upstream's Smithery `createStatelessServer`
>   factory to ADK's expected `StdioServerTransport`. Launched via
>   `npx tsx`. The wrapper is the only part of the MCP server we
>   wrote; the rest is the cloned upstream repo.
> - **Graceful degradation**: if `vendor/gahmen-mcp/` is missing,
>   ADK logs `Failed to get tools from toolset McpToolset` and the
>   rest of the BI team continues to work — the MCP-using specialists
>   fail at first tool call, but `google_search` / `load_web_page`
>   remain available. v2 resilience moment.
>
> v2 source: `level_4a_agent/agent.py` + `mcp_toolset.py` + extended
> `safety.py` (the rest is identical to L4 — `creator_tools.py`,
> `registry.py`, `tools.py`). The `vendor/gahmen-mcp/` directory is
> gitignored; clone your fork there and run `npm install` once before
> using.

**Not a new taxonomy level.** A variant of Level 4 where `data_fetcher_agent` swaps the `google_search` built-in for an `McpToolset` pointed at the [`gahmen-mcp`](https://github.com/simonraj79/gahmen-mcp) Singapore Government Data MCP server (data.gov.sg + SingStat Table Builder).

The coordinator's meta-reasoning, `agent_creator`, `state.capabilities` re-hydration, and safety allowlist are identical to Level 4. Two things genuinely differ: the data source (MCP instead of `google_search`) and the **chart-propagation path** (see below).

- **Connection mode** — local stdio only, via `StdioConnectionParams` spawning `npx @smithery/cli run simonraj79/gahmen-mcp`. No auth required, no Smithery account needed.
- **Tool filter** — only the 8 read-oriented MCP tools are exposed; download-orchestration tools (`datagovsg_initiate_download`, `datagovsg_poll_download`) are hidden by default.
- **Safety allowlist extension** — `ALLOWED_TOOLS` gains `datagovsg_*` / `singstat_*` entries; resolution builds a **per-specialist `McpToolset`** narrowed by `tool_filter` so each runtime specialist only sees the MCP tools it declared.
- **Chart pipeline — same as L4** (`plt.show()` only → framework auto-saves `inline_data` Part → adk web Artifacts tab). Earlier drafts of this doc claimed L4 used an "explicit" `tool_context.save_artifact` path while L4a used an "implicit" one — that distinction was fictional. `tool_context` is not accessible inside Gemini's hosted code-execution sandbox, so `tool_context.save_artifact(...)` written in a prompt to `BuiltInCodeExecutor` raises `NameError`. Both L4 and L4a now share the single, hardened single-cell `plt.show()` pattern with a defensive `n_artists = len(ax.lines) + len(ax.patches) + len(ax.collections)` check before show (see `level_4a_agent/agent.py:78-101` and the matching block in L4).
- **Demonstrates:** `McpToolset` wiring, domain-specific BI on top of L4's framework, that Level 4's safety model scales to MCP tool surfaces without change.

Full design and rationale: [`New Agents/LEVEL_4A_MCP_PLAN.md`](New%20Agents/LEVEL_4A_MCP_PLAN.md).

---

## How Each Level Adds Capabilities

| | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Level 4a (variant) |
|---|---|---|---|---|---|---|
| **New capability** | Brain only | + Tools | + Planning + Context engineering | + Delegation to sub-agents | + Meta-reasoning + dynamic agent creation | = Level 4 capabilities, scoped to a domain-specific MCP data source |
| **Question type** | Static knowledge | Simple, single-fact | Complex, multi-part | Complex, multi-domain | Any, even requiring capabilities not in the starting team | Same as L4, but analytical questions are answered against Singapore government datasets |
| **Tool calls** | 0 | 1 per question | Multiple, chained | Multiple, distributed across agents | Multiple, through AgentTools + dynamically-added specialists | Same as L4; fetcher calls MCP tools instead of `google_search` |
| **Planning** | ❌ | ❌ | ✅ Upfront decomposition | ✅ Upfront + delegation | ✅ Upfront + delegation + team-composition decisions | ✅ Same as L4 |
| **State** | ❌ | ❌ | ✅ `ToolContext.state` scratchpad | ✅ Shared across agents via `output_key` + `{state_key?}` injection + `ToolContext.state` | ✅ L3 state + `state.capabilities` registry for runtime specialists | ✅ Same as L4 |
| **Search tool** | — | `google_search` built-in | `GoogleSearchTool(bypass_multi_tools_limit=True)` + custom function tools — `multiple_tools=True` triggers ADK's auto-swap at `llm_agent.py:151-157` to `GoogleSearchAgentTool` | Plain `google_search` built-in (inside `search_agent`) — works directly because the agent is wrapped as `AgentTool`, so no `transfer_to_agent` auto-injection, so no built-in + function conflict | `google_search` built-in (inside `data_fetcher_agent`) — same AgentTool pattern as L3 | **`McpToolset(connection_params=StdioConnectionParams(...))`** over `gahmen-mcp` (data.gov.sg + SingStat) — no `google_search` |
| **Code execution** | — | — | — | — | `BuiltInCodeExecutor` on `analyst_agent` (Flash + `BuiltInPlanner` thinking). Charts via single-cell `plt.show()` → framework auto-saves the `inline_data` image to the artifact service (`_code_execution.py:279-318`). Prompt forbids `savefig` / `tool_context.save_artifact` (the latter is a NameError inside the sandbox). | Same as L4. |
| **Number of agents** | 1 | 1 | 1 | 4 | 5 fixed + N runtime (session-scoped) | Same as L4 (5 fixed + N runtime) |
| **Output** | Plain text | Plain text | Structured report | Structured report | Structured BI brief + inline matplotlib charts | Same as L4 |
| **Reasoning mode** | None (no `planner`) | None (no `planner`) | None — planning is prompt-driven, not framework-enforced | **`PlanReActPlanner()` on the coordinator** — explicit Plan → Reason → Act → Final scaffold injected into the prompt, makes the delegation reasoning visible inline | None — meta-reasoning is prompt-driven via the coordinator's instruction | Same as L4 |
| **Key ADK concept** | Tool discovery | Tool use | Context engineering | **`AgentTool` delegation + `PlanReActPlanner` + `output_key` + instruction state injection** | `AgentTool` + `BuiltInCodeExecutor` + `before_agent_callback` re-hydration | **`McpToolset` + `StdioConnectionParams`** on top of L4's stack; safety allowlist extends to MCP tool names with per-specialist `tool_filter` narrowing |

> **Note on "reasoning mode":** None of the levels set `planner=...` on their agents. Gemini 2.5's *native* thinking still runs on every LLM call (thinking tokens are consumed internally), but thoughts are not surfaced and no framework-level plan-then-act scaffold is applied. To convert any agent into an explicit reasoning agent, add either `BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True))` (surfaces Gemini's native thinking) or `PlanReActPlanner()` (forces a prompt-level ReAct loop). See `contributing/samples/fields_planner/agent.py` for the canonical example.

> **Note on variants not in this table:** the table focuses on taxonomy axes. Demo variants that change only one orthogonal thing are documented inline above rather than as extra columns: `level_1a_agent` (L1 + Gemini Live voice; minimum viable voice agent, single tool, no sub-agent transfer) and `level_4a_agent` (L4 + domain-specific MCP data source — this one *is* in the table because it changes multiple axes at once). See the L1a and L4a sections for details. A multi-agent voice variant (`level_3a_agent`) was previously in the repo but removed after neither `gemini-2.5-flash-native-audio-*` nor `gemini-3.1-flash-live-preview` reliably supported the `transfer_to_agent` function call required for sub-agent delegation in audio mode.

---

## Reasoning Mode in ADK — API Reference

ADK exposes reasoning via two orthogonal layers.

### Layer 1 — Model-level thinking (`google.genai.types.ThinkingConfig`)

Controls Gemini's native thinking feature. Only works on models that support thinking (Gemini 2.5 family).

| Field | Type | Meaning |
|---|---|---|
| `thinking_budget` | `int` | Max thinking tokens. `0` = disabled, `-1` = automatic, default is model-dependent |
| `thinking_level` | `ThinkingLevel` enum | `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`, `THINKING_LEVEL_UNSPECIFIED` — coarser-grained alternative to `thinking_budget` |
| `include_thoughts` | `bool` | Whether thought tokens are surfaced in the response (default: hidden) |

### Layer 2 — ADK planners (`planner` field on `LlmAgent`)

Two concrete planners ship; you can also subclass `BasePlanner` for custom scaffolds.

| Planner | Mechanism | Works with |
|---|---|---|
| `BuiltInPlanner(thinking_config=...)` | Wraps a `ThinkingConfig`. Uses Gemini's **native** thinking. No prompt injection. | Gemini 2.5 family only |
| `PlanReActPlanner()` | Injects prompt instructions forcing `/*PLANNING*/`, `/*REASONING*/`, `/*ACTION*/`, `/*FINAL_ANSWER*/` tags around the response. Prompt-level ReAct scaffold. | **Any model** (Gemini, Claude, GPT via LiteLLM, etc.) |

### Usage patterns

```python
from google.adk.agents import Agent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.genai import types

# Pattern A — surface Gemini 2.5's native thinking with a controlled budget.
agent = Agent(
    model="gemini-2.5-pro",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MEDIUM,
            include_thoughts=True,
        ),
    ),
    ...
)

# Pattern B — force a ReAct-style plan/reason/act scaffold (works on any model).
agent = Agent(
    model="gemini-2.5-flash",
    planner=PlanReActPlanner(),
    ...
)

# Pattern C — per-agent disable thinking entirely (e.g. for a mechanical
# formatter like writer_agent where thinking is wasted tokens).
agent = Agent(
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
    ...
)
```

### Two places thinking can be configured

```python
# Option A — via planner (recommended; supports both BuiltInPlanner and PlanReActPlanner)
Agent(..., planner=BuiltInPlanner(thinking_config=...))

# Option B — directly on content config
Agent(..., generate_content_config=types.GenerateContentConfig(thinking_config=...))
```

If both are set, `planner` wins (`llm_agent.py:889-895` logs a warning). Prefer Option A — it composes with `PlanReActPlanner` and is the canonical placement per `contributing/samples/fields_planner/agent.py`.

### When to turn it on (and off) per level

| Level | Default | When to turn it on | When to turn it off |
|---|---|---|---|
| **L0** `my_first_agent` | None | Never — no multi-step reasoning is possible with one trivial tool | — |
| **L1** `level_1_agent` | None | Never — one-shot search-answer; adds cost without quality gain | — |
| **L2** `level_2_agent` | None | If planning decomposition misses sub-questions, or syntheses are shallow on hard questions. Try `BuiltInPlanner` with `thinking_level=MEDIUM` first. | — |
| **L3** `level_3_agent` | None on all four | Most likely escalation: `analyst_agent` in REVIEW mode (multi-note cross-referencing) | `writer_agent` can use `thinking_budget=0` — it's a template-formatter and thinking is wasted |
| **L4** `level_4_agent` | None on all five fixed agents | Most likely escalation: `bi_coordinator` (meta-reasoning is the new Level 4 job) or `agent_creator` (spec design is reasoning-heavy). Try `BuiltInPlanner` with `include_thoughts=True` to see gap-detection logic. | `report_writer_agent` (template) and `data_fetcher_agent` (mechanical) can use `thinking_budget=0` |

---

## Key Takeaways

1. **Level 0 → 1**: From closed system to **connected**. The agent can now answer questions that require real-time data by calling external tools.

2. **Level 1 → 2**: From simple to **strategic**. The agent gains upfront planning (question decomposition) and working memory (scratchpad). It can now handle complex, multi-part goals where each step feeds the next.

3. **Level 2 → 3**: From solo to **team**. The paradigm shifts from "one agent does everything" to "a coordinator delegates to specialists." The coordinator has no tools — only sub-agents.

4. **Level 3 → 4**: From fixed team to **self-evolving team**. The coordinator can detect capability gaps and spawn new specialists mid-session. Every specialist is an `AgentTool`; runtime specialists are persisted in session state and re-hydrated on each turn.

5. All tool-using agents share the same base **reason → act → observe** cycle. The levels add capabilities on top, not different thinking patterns.

---

## What's Next?

Potential **Level 5** direction (beyond the published taxonomy):

- **Framework-enforced reasoning.** Level 4 keeps meta-reasoning at the prompt level. Adding `BuiltInPlanner` (surfaced thinking) or `PlanReActPlanner` (enforced plan-then-act scaffold) to the coordinator would make the reasoning structure *part of the framework*, not an instruction the LLM can skip.
- **Cross-session persistence.** Level 4 is session-scoped — runtime specialists die with the session. A genuine L5 could promote successful specialists into committed YAML agents that survive restarts.
- **Code-generated tools**, not just agents. Level 4 assembles new *agents* from existing allowlisted tools. Generating new *tool code* at runtime (via a sandboxed `ContainerCodeExecutor`) would close the "whole capability" loop but significantly expand the blast-radius — real sandbox hardening required.
- **Evaluation-driven evolution.** A Level 5 system could measure each specialist's success rate (e.g., via ADK's eval framework) and retire or replace underperforming specialists automatically. See `contributing/samples/gepa/` for prompt-evolution prior art.
