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

> **v2 wiring update**: this is the level that changed the most. Two
> separate redesigns:
>
> **(1) Use case rebrand: research assistant → student day planner.**
> The taxonomy ("strategic planning + context engineering") is
> unchanged, but the demo pivoted from a research bot (which produced
> walls of text most learners never read) to a *student day planner*:
> the user describes commitments and goals, the agent extracts study
> topics, fans out parallel topic research, and assembles a
> Pomodoro-paced markdown timetable. Same shape as research
> (decompose → look up → assemble) but the output is concretely
> useful and the user describes their day naturally.
>
> **(2) Wiring overhaul.** v1 was a single `LlmAgent` with a long
> instruction *asking* the LLM to follow PLAN/EXECUTE/SYNTHESISE
> steps, plus two custom scratchpad tools (`save_research_note` /
> `get_research_notes`). v2 expresses the same workflow as a
> **`Workflow(BaseNode)` graph**:
>
> ```
> START → process_input → classify → ┬── quick → quick_answerer → END
>                                    └── plan  → anchor_today →
>                                                task_planner →
>                                                fan_out_research →
>                                                schedule_writer → END
> ```
>
> - **2-way classifier + inline greeting handling** — there is no
>   dedicated greeter node. Greetings ("hi", "hello", "what can you
>   do?") are classified as `quick` and routed to the
>   `quick_answerer`, whose instruction handles them as a special
>   case (introduces the agent + suggests two example queries).
>   Same idiom as Level 3 and Level 4 where the lead agent handles
>   greetings inline rather than via a routed sub-agent. Cleaner
>   graph viz (no greeter dead-end), consistent pattern across the
>   ladder.
> - **`anchor_today` function node** stashes today's date in state
>   (`today`, `today_human` keys). The planner LLM reads it via
>   `{today_human?}` instruction injection and computes concrete
>   dates ("Friday May 1", "Wednesday April 29") from the user's
>   relative references — no `get_current_date` tool call needed.
> - **Dynamic fan-out via `ctx.run_node()`** — the planner emits a
>   `study_topics: list[str]` (1-3 items, runtime-decided); one
>   researcher per topic spawns concurrently with `google_search`.
>   Pattern from `contributing/workflow_samples/dynamic_fan_out_fan_in/`.
> - **`Event(state={...})` + `{key?}` instruction injection** replaces
>   `save_research_note` / `get_research_notes` custom tools. The
>   orchestrator writes `commitments` and `topic_briefs` to state;
>   the writer reads via `{commitments?}` and `{topic_briefs?}`.
>   Fewer tools, no LLM compliance risk.
> - **`@node(rerun_on_resume=True)`** on the dynamic-fan-out
>   orchestrator so it's HITL-resumable.
> - **The `[NO DEFAULT]` warning on the route node is benign here.**
>   Pydantic's `Literal["quick", "plan"]` constraint on the
>   classifier's `output_schema` rejects any other value at the
>   schema layer before reaching `route_input`, so no dead-end is
>   reachable. Adding a `__DEFAULT__` fallback that points at one
>   of the two named routes is rejected by graph validation as a
>   duplicate edge; adding a no-op fallback node would be
>   structural noise for no behavioural gain.
>
> v2 source at `level_2_agent/agent.py`. Compare side-by-side with
> `V1_level_2_agent/agent.py` — the v1 prompt's "1. PLAN: 2.
> EXECUTE: 3. SYNTHESISE:" instruction collapses into a graph that
> *enforces* those steps structurally rather than relying on the
> LLM to follow them.

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

## Level 2b — `level_2b_agent` (Minimal Graph Router)

> **Variant of Level 2.** Same capability tier (strategic / context
> engineering via a graph), minimal shape: classify → route → handler.
> No fan-out, no parallel research, no synthesis. The point is to show
> the v2 graph routing primitive **in isolation**, without the
> dynamic-fan-out richness that makes the canonical L2 information-dense.

**Not a new taxonomy level.** New ADK 2.0 *primitives* are absorbed as
`Na/Nb/Nc` variants of the closest capability tier; they don't inflate
the L0–L4 ladder. Same rationale as L4a (MCP variant of L4) and L1a
(voice variant of L1).

**Inspired by** the `graph_router` demo in the *ADK 2.0 launch* video.
The video's pedagogical message: in v1 you'd cram "first classify, then
call the right handler, never skip steps" into one giant prompt and
**hope** the LLM followed it; in v2 the routing logic moves out of the
prompt and **into the graph**, so it's deterministic. This file is the
project's canonical illustration of that lesson.

**Use case**: customer support triage. The user sends a message; the
classifier emits one of four categories (`GREETING` / `BUG` / `BILLING`
/ `FEATURE`); the routing dict dispatches to a focused handler agent
that responds in-character for its category.

**v2 idioms genuinely in use**:
- `Workflow(BaseNode)` graph orchestration with **dict routing map**
  on the classifier output (`(route_input, {"BUG": bug_handler, ...})`).
- `output_schema=` Pydantic class with `Literal[...]` field — the
  framework rejects any value outside the enum at the schema layer, so
  no `__DEFAULT__` fallback edge is needed.
- LLM agents embedded directly in `edges=[...]` — auto-wrapped as nodes
  per the canonical v2 pattern (`adk-workflow` skill: "Place `Agent`
  instances directly in workflow edges").
- `Event(state={...})` from `process_input` to stash the user's request
  for handler `{request}` template injection.
- `output_key="last_response"` on each handler for state-delta
  observability.

**"Lead agent handles greetings inline" idiom — graph form**:
`AGENT_LEVELS.md` documents the convention that the agent the user
talks to handles greetings inline rather than via a dedicated greeter
sub-agent. In a coordinator-LlmAgent shape (L3, L4), that's a prompt
branch. In a Workflow shape, the equivalent is **a `GREETING` route on
the classifier**: the classifier is the "lead" the user's first
message hits, and a small `greet_user` agent handles only that route.
Same idiom, expressed in graph form. Avoids fighting `output_schema`
(greetings have no category to emit) and keeps the graph viz honest.

### Graph

```
START
  ↓
process_input  (function node — stash request in state)
  ↓
classify       (LlmAgent — output_schema=TicketCategory)
  ↓
route_input    (function node — emits Event(route=...))
  ├── "GREETING" → greet_user      → END
  ├── "BUG"      → bug_handler     → END
  ├── "BILLING"  → billing_handler → END
  └── "FEATURE"  → feature_handler → END
```

### Sample queries (verified end-to-end with `adk run --jsonl`)

- *"hi"* → GREETING → `greet_user` introduces capabilities + 3 example queries.
- *"the dashboard shows 500 errors every time I open the analytics page"* → BUG → `bug_handler` asks for repro / browser / OS / severity.
- *"how much does the Pro plan cost and what features are included?"* → BILLING → `billing_handler` quotes Pro pricing + storage + SSO + 20% annual discount.
- *"it would be great if you could add dark mode to the dashboard"* → FEATURE → `feature_handler` thanks the user, asks one use-case clarifier, sets backlog expectations.

**Demonstrates**: the v2 graph routing primitive on its own, without
the additional planning/fan-out machinery of the canonical L2. A clean
stepping-stone *into* L2 for readers who find the day-planner
information-dense. v2 source: `level_2b_agent/agent.py`.

---

## Level 2c — `level_2c_agent` (Dynamic Workflow + HITL Pause/Resume)

> **Variant of Level 2.** Same capability tier (strategic / context
> engineering via a graph). The new axis is **framework-enforced
> human-in-the-loop**: a `@node` function returns `RequestInput`, the
> runner physically halts the workflow until a human responds, and
> `App + ResumabilityConfig(is_resumable=True)` durably checkpoints
> state so the workflow survives a server restart while waiting.

**Not a new taxonomy level.** New ADK 2.0 *primitives* are absorbed as
`Na/Nb/Nc` variants of the closest capability tier; they don't inflate
the L0–L4 ladder. `RequestInput` is a new *primitive*, but the
underlying capability — workflow orchestration with a deterministic
gate — sits inside L2's strategic-planning tier. Same rationale as
L4a (MCP variant) and L1a (voice variant).

**Inspired by** the `refund_approval` demo in the *ADK 2.0 launch*
video. The video's pedagogical message: in v1, "human-in-the-loop"
was a prompt-level convention you hoped the LLM honoured; in v2, the
framework guarantees it — the workflow physically cannot continue
past `RequestInput` until a real response comes back, and the state
is durably checkpointed. That guarantee is what enables enterprise
approval / compliance / fraud-review patterns.

**Use case**: refund processing with a manager-approval gate. Under
$100 the workflow auto-approves and processes; at or above $100 it
pauses, asks a manager 'yes' or 'no', then either issues the refund
or records the rejection.

### How L2c's HITL differs from L4's HITL

| | L4 `agent_creator` | L2c `gate` |
|---|---|---|
| **Trigger** | LLM-discretionary — the model in `mode='task'` decides whether to ask | Framework-enforced — `if amount >= 100: return RequestInput(...)` |
| **Bypass-able by prompt drift?** | Yes (LLM might just decide to commit without asking) | No (the function returns the interrupt; the runner halts) |
| **Use when** | Clarifying a fuzzy spec — "is this team description right?" | Compliance gates that *must* hold — "every refund ≥ $100 needs sign-off" |
| **Recovery on restart** | Replay session events | Checkpointed via `ResumabilityConfig(is_resumable=True)` |

Different guarantees → both have legitimate uses. L4's HITL is more
flexible, L2c's is more enforceable.

### v2 idioms genuinely in use

- **`RequestInput(message=, response_schema=, payload=,
  interrupt_id=)`** from `google.adk.events.request_input`.
- **`@node(rerun_on_resume=True)`** on the gate — when the user
  responds, the gate **re-executes** with `ctx.resume_inputs`
  populated, letting one function handle both first-run (ask) and
  post-resume (decide) cases without a follow-up node. Without this
  flag, the user's response would just *become* the node's output
  (default `FunctionNode` behaviour), forcing a separate interpreter
  node downstream.
- **`App(root_agent=..., resumability_config=ResumabilityConfig(
  is_resumable=True))`** — durable checkpointing. Both `app` and
  `root_agent` are exported from `agent.py`; the loader picks `app`
  when present (per `references/human-in-the-loop.md`).
- **`output_schema=RefundRequest` + `output_key="refund_request"`**
  on the intake LLM — turns user free-text into a typed dict, stored
  in state for downstream function nodes to read via parameter-name
  resolution (`def gate(refund_request: dict)`).
- **`Event(state={"decision": ...}, output=...)`** to both update
  state AND pass the decision down the graph in one event.

### Graph

```
START
  ↓
process_input    (function — stash raw text)
  ↓
intake           (LlmAgent — parse → RefundRequest)
  ↓
gate             (@node async, rerun_on_resume=True)
  │ amount < $100 → emit RefundDecision(approved=True), continue
  │ amount ≥ $100 first run  → yield RequestInput, PAUSE
  │ amount ≥ $100 resumed    → emit RefundDecision based on response
  ↓
process_refund   (@node async — issue refund OR record rejection)
  ↓
  END
```

### Sample queries (verified end-to-end with `adk run --jsonl`)

- *"Process a $50 refund for customer C-001 — wrong size shipped."*
  → intake parses → gate auto-approves (under threshold) → process_refund mints `RFND-XXXXXXXX` confirmation. **No pause.**
- *"Refund $350 to customer C-002 — defective laptop returned."*
  → intake parses → gate emits `adk_request_input` function call with `longRunningToolIds: ["manager_approval"]`. Workflow halts. (Resume verified interactively in `adk web`: reply 'yes' → process; 'no' → reject.)

**On the wire**: `RequestInput` shows up as a function call to a
synthetic tool `adk_request_input` with the workflow's
`longRunningToolIds` field flagging the suspension. That's how the
framework communicates "this isn't a real tool call, this is a HITL
pause" to clients — `adk web` hooks it to a chat prompt; custom
clients respond with a `FunctionResponse` keyed to the same id.

**Demonstrates**: the third pillar of ADK 2.0 (dynamic workflows
with framework-enforced HITL pause/resume + automatic checkpointing).
v2 source: `level_2c_agent/agent.py` (single-file demo — workflow,
intake LLM, gate, process_refund, App wrapper all colocated).

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

## Level 3b — `level_3b_agent` (Dual-Mode Coordinator: Travel Planner)

> **Variant of Level 3.** Same capability tier (delegation to a fixed
> team of specialists). The new axis is which **mode** each peer
> specialist uses: one `mode='single_turn'` (autonomous) and one
> `mode='task'` (allowed to ask the user clarifying questions
> mid-task). Our canonical L3 only uses `single_turn`; only L4's
> `agent_creator` uses `mode='task'`. L3b fills the gap by demonstrating
> both modes side-by-side on peer specialists.

**Not a new taxonomy level.** Same rationale as L4a / L1a / L2b: new
ADK 2.0 *primitives* are absorbed as `Na/Nb/Nc` variants of the
closest capability tier; they don't inflate the L0–L4 ladder.

**Inspired by** the `travel_planner` demo in the *ADK 2.0 launch* video
("the moment the task completes, it will again auto return"). Same
shape as the canonical mixed-mode example in
`.agents/skills/adk-workflow/references/task-mode.md:188-232`.

**Use case**: travel planning. The coordinator delegates to:
- `weather_checker` — `mode='single_turn'`, autonomous one-shot. Looks
  up current + 3-day forecast for one city, returns a structured
  `WeatherForecast`, control auto-returns to the coordinator.
- `flight_booker` — `mode='task'`, multi-turn. Receives `origin` and
  `destination` only (no `date`), asks the user for the departure
  date, presents flight options, books the chosen one, returns a
  structured `FlightBooking` via `finish_task`.

**v2 idioms genuinely in use**:
- `sub_agents=[...]` with **mixed modes** on peer specialists. The
  framework auto-creates `request_task_weather_checker` and
  `request_task_flight_booker` tools on the coordinator.
- `input_schema` / `output_schema` Pydantic models on each specialist
  for typed coordinator↔specialist contracts.
- `mode='task'` HITL clarification — the booker pauses with
  `branch: "task:..."` until the user replies, then resumes naturally.
- `disallow_transfer_to_parent=True` + `disallow_transfer_to_peers=True`
  on both specialists — same gotcha #24 hygiene as L3 (suppresses the
  framework's auto-injection of `transfer_to_agent`).
- `PlanReActPlanner()` on the coordinator — same rationale as L3.

**Key design choice** — `date` deliberately omitted from
`FlightBookingInput`: forces the booker to ASK the user. If `date`
were in the schema, the coordinator's LLM might guess one. Omitting
it is the pedagogical hook for `mode='task'` — the framework gives the
agent a legitimate reason to interrupt and ask. Same trick the video
uses ("What is the exact date?").

**Tool/mode compatibility note** — L3b uses **only function tools**
(`get_weather`, `search_flights`, `book_flight`). No built-ins like
`google_search`, so gotcha #24 doesn't apply: a `mode='task'` agent's
auto-injected `FinishTaskTool` plus 2 function tools is fine; it only
becomes a problem when mixed with built-in tools. (For an L3-style
agent that *does* mix built-ins with `mode='task'`, see `AGENTS.md`
gotcha #24 — set `bypass_multi_tools_limit=True` to opt into the
auto-swap.)

### Sample queries (verified end-to-end with `adk run --jsonl`)

- *"hi"* / *"what can you do?"* → coordinator handles inline; no
  delegation.
- *"what's the weather in Paris today?"* → coordinator:
  `request_task_weather_checker(city="Paris")` → autonomous run, auto-
  return → coordinator emits final `WeatherForecast` summary.
- *"book me a flight from SFO to CDG"* → coordinator:
  `request_task_flight_booker(origin="SFO", destination="CDG")` →
  booker asks *"What date would you like to depart?"* → workflow
  pauses for user input. (Multi-turn — verify in `adk web`.)
- *"I'm going to Tokyo. Check the weather there, then book a flight
  from SFO."* → coordinator chains both:
  `weather_checker(Tokyo)` runs to completion (autonomous), then
  `flight_booker(SFO, Tokyo)` opens and asks for date. **Both modes
  in a single coordinator turn.**

**Demonstrates**: peer-specialist `mode='task'` HITL clarification,
mixed-mode chaining under one coordinator, the auto-injected
`request_task_<name>` / `set_model_response` / `finish_task` tool
trio in action. v2 source: `level_3b_agent/agent.py` +
`level_3b_agent/tools.py`.

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

## How Each Level Adds Capabilities (v2)

The table below reflects the **v2 wiring** (post-migration). The
**capability ladder** in row 1 is taxonomy-stable across v1 and v2 —
it's the underlying abilities, not the framework primitives. The rest
of the rows describe how each capability is implemented in v2.

| | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Level 4a (variant) |
|---|---|---|---|---|---|---|
| **New capability** | Brain only | + Tools | + Planning + Context engineering | + Delegation to specialists | + Meta-reasoning + dynamic agent creation | = Level 4 capabilities, scoped to a domain-specific MCP data source |
| **Demo use case** | Pre-trained knowledge only | One-shot search-and-answer | **Student day planner** — extracts commitments, parallel topic research, produces a Pomodoro-paced markdown timetable | Coordinator + 3 specialists (search / analyst / writer) producing a structured Brief | BI agent team with `agent_creator` that spawns new specialists on demand; runtime specialists persist to disk and survive `adk web` restart | Same as L4 with Singapore Government data (data.gov.sg + SingStat) added as a third source on `data_fetcher_agent` |
| **Question type** | Static knowledge | Simple, single-fact | Greeting OR quick factual ("what's the Pomodoro technique?") OR commitments-driven schedule request | Multi-domain analytical (research → analysis → brief) | Any analytical request; coordinator detects when the team can't cover it and delegates to creator | SG-flavoured analytical (or any L4 question — fetcher picks per query) |
| **Top-level shape** | `LlmAgent` | Leaf `LlmAgent` (Mesh-bypass fast path in v2) | `Workflow(BaseNode)` graph with edge declarations | `LlmAgent` + `sub_agents=[...]` w/ `mode='single_turn'` | Same as L3, plus `mode='task'` `agent_creator` for HITL | Same as L4 |
| **Tool calls** | 0 | 1 per question | Multiple, parallel via `ctx.run_node()` dynamic fan-out | Multiple, distributed across `request_task_<name>` calls to specialists | Multiple, through fixed-team specialists + runtime specialists (registered as `AgentTool`s on the coordinator at every turn) | Same as L4; fetcher routes between `google_search` / `load_web_page` / 8 MCP tools per query |
| **Planning** | ❌ | ❌ | ✅ Graph-enforced upfront decomposition | ✅ LLM-decided runtime routing made visible via `PlanReActPlanner()` | ✅ Same as L3, plus capability-gap detection (meta-reasoning) | ✅ Same as L4 |
| **State** | ❌ | `output_key` writes `state.last_answer` for delta observability | `Event(state={...})` from function/orchestrator nodes + `{key?}` instruction injection in agents downstream | `output_key` on terminal LLM agents; specialist returns are typed Pydantic objects flowing through coordinator's LLM context | L3 state + `state.capabilities` registry; `before_agent_callback` rebuilds runtime tools from state ∪ disk YAML library each turn | Same as L4 |
| **Search tool** | — | Plain `google_search` built-in (single tool, no auto-swap needed) | Plain `google_search` on each researcher (worker is alone, no conflict); fan-out wrapper handles parallelism | `google_search` on `search_agent` (single-tool sub-agent in `single_turn` mode + `disallow_transfer_to_*` flags suppresses the v1 gotcha #24 echo) | `GoogleSearchTool(bypass=True)` + `load_web_page` on data_fetcher → auto-swap to `GoogleSearchAgentTool` (function-wrapped) when `multiple_tools=True` | Same as L4 PLUS `McpToolset(connection_params=StdioConnectionParams(...))` over gahmen-mcp narrowed by `tool_filter` to 8 read-only tools |
| **Code execution** | — | — | — | — | `BuiltInCodeExecutor` on `analyst_agent` (Flash + `BuiltInPlanner` thinking). Charts via single-cell `plt.show()` → framework auto-saves the `inline_data` image to the artifact service (`_code_execution.py`). Hardened prompt forbids `savefig` / `tool_context.save_artifact` (the latter is a NameError inside the sandbox) | Same as L4 |
| **Delegation primitive** | — | — | n/a (single-agent graph) | `sub_agents=[...]` + `mode='single_turn'` → framework auto-derives `_SingleTurnAgentTool` instances on coordinator. `request_task_<name>` is the auto-generated function tool the coordinator calls | Same as L3 PLUS `mode='task'` on `agent_creator` for multi-turn HITL with auto-return on `finish_task`. Runtime specialists wrap as `AgentTool` (v2 `sub_agents` auto-derivation is init-time only — runtime injection has to use `AgentTool`) | Same as L4 |
| **Number of agents** | 1 | 1 | 1 graph + 5 LLM agents (classify, quick_answerer, task_planner, schedule_writer, researcher×N dynamic) | 1 coordinator + 3 specialist sub-agents | 1 coordinator + 4 sub-agents (3 fixed-team + creator) + N runtime specialists (session-scoped + disk-persisted) | Same as L4 (4 sub-agents + N runtime) |
| **Output** | Plain text | Plain text + `state.last_answer` delta | Structured markdown timetable for "plan" route; 2-3 sentence answer for "quick" route | Pydantic `Brief` (typed) | Pydantic `Brief` + matplotlib chart artifact | Same as L4 |
| **Reasoning mode** | None | None — one-shot question, planner adds cost without quality gain | None on the agents themselves; the *graph itself* enforces the plan-then-act structure (PLAN/EXECUTE/SYNTHESISE collapse into nodes) | **`PlanReActPlanner()` on the coordinator** — `/PLANNING/ /REASONING/ /ACTION/ /FINAL_ANSWER/` blocks visible inline in chat panel | **`PlanReActPlanner()` on the coordinator** + **`BuiltInPlanner(include_thoughts=True)` on `analyst_agent`** (Pro on a code-executor agent hangs under AFC — gotcha #21) | Same as L4 |
| **HITL** | — | — | — | — | `mode='task'` on `agent_creator` — multi-turn confirmation ("draft spec → ask user → call create_specialist → call finish_task") with auto-return | Same as L4 |
| **Persistence** | — | — | Session-scoped state via `Event(state={...})` | Session-scoped state via `output_key` | **Cross-restart**: `runtime_agents/*.yaml` library on disk, read on every turn by `before_agent_callback` | Same as L4 |
| **Key ADK 2.0 concept** | Tool discovery | Leaf-`LlmAgent` fast path; `output_key` state delta | **`Workflow(BaseNode)` graph + `ctx.run_node()` dynamic fan-out + `Event(state=...)` data flow** | **`sub_agents=[...]` + `mode='single_turn'` auto-delegation + `disallow_transfer_to_*` to suppress gotcha #24** | **`mode='task'` HITL + `before_agent_callback` runtime tool hydration + `BuiltInCodeExecutor` charts** | **`McpToolset` + `StdioConnectionParams`** on top of L4's stack; safety allowlist extends to MCP tool names with per-spec `tool_filter` narrowing |

> **Note on "reasoning mode":** at L1/L2 a `planner` adds cost without
> quality gain — L1 is one-shot, L2 has plan-then-act enforced by the
> graph itself. L3+ benefits from making the LLM-decided routing
> visible (`PlanReActPlanner()`) since the routing IS the reasoning.
> L4's `analyst_agent` uses `BuiltInPlanner` (Gemini's native thinking)
> rather than `PlanReActPlanner` because: (a) Pro on a
> `BuiltInCodeExecutor` agent hangs under AFC (gotcha #21), so we use
> Flash + native thinking instead; (b) the analyst's reasoning is
> code-cell layout, which native thinking handles well without prompt
> scaffolding. To wire `PlanReActPlanner()` into any leaf agent, set
> the `planner=` field; the v2 framework injects the
> `/PLANNING/ /REASONING/ /ACTION/ /FINAL_ANSWER/` scaffold around
> the response.

> **Note on variants not in this table:** L1a (Gemini Live voice
> variant of L1) and L4a (MCP variant of L4) are documented in their
> own sections above. L4a *is* in the table because it changes
> multiple axes (tool source, allowlist, vendoring); L1a is not
> because it only changes one axis (model identifier + a Live-event
> filtering subclass). A multi-agent voice variant (`level_3a_agent`)
> was previously in the repo but removed after neither
> `gemini-2.5-flash-native-audio-*` nor `gemini-3.1-flash-live-preview`
> reliably supported the `transfer_to_agent` function call required
> for sub-agent delegation in audio mode.

> **The "lead agent handles greetings inline" pattern is consistent
> across L2/L3/L4/L4a.** Earlier drafts had a dedicated `greeter`
> sub-agent on L2 and L3, which made the graph viz noisy and was
> structurally inconsistent with L4 (where greetings live in the
> coordinator's instruction prose). Now all four levels follow the
> same idiom: the lead agent the user actually talks to (L2's
> `quick_answerer`, L3/L4's coordinator, L4a's coordinator) detects
> greetings via prompt branching and responds inline with one
> sentence describing the agent + 2 example queries. Same idiom in
> different shells (Workflow node vs LlmAgent prompt).

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
