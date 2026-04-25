"""Level 4a — Self-Evolving BI System with a domain-specific MCP source.

Variant of `level_4_agent` with one additional capability: the
fixed-team `data_fetcher_agent` now also has access to **gahmen-mcp**
(Singapore Government Data — data.gov.sg + SingStat Table Builder)
via the v2 `McpToolset` primitive. Everything else (coordinator
meta-reasoning + `PlanReActPlanner`, `BuiltInCodeExecutor` analyst,
`agent_creator` on Gemini 3.1 Pro, disk-backed runtime-spec library,
safety allowlist with sentinel pattern for MCP tools) is structurally
identical to Level 4. Compare side-by-side with
`level_4_agent/agent.py` — the diff is intentionally narrow.

Prerequisite: the gahmen-mcp server must be vendored at
`level_4a_agent/vendor/gahmen-mcp/` with a stdio entry wrapper. See
`mcp_toolset.py` for the launch contract. Without it, the agent
module still loads and the rest of the BI team works — but any
specialist that calls a `datagovsg_*` / `singstat_*` tool will fail
to launch the subprocess.

Fixed team
----------
- `data_fetcher_agent`  — three sources, picked by the LLM per query:
  `google_search` (general web), `load_web_page` (specific URL), and
  `gahmen_toolset` (Singapore Government data — eight read-only MCP
  tools narrowed via `tool_filter`).
- `analyst_agent`       — pandas/matplotlib via `BuiltInCodeExecutor`
  (unchanged from L4).
- `report_writer_agent` — formats accumulated findings into a BI brief
  (unchanged from L4).

Level 4 additions
-----------------
- `agent_creator`       — builds new specialists on demand, with user
                          confirmation before persisting.
- Capability registry   — runtime specialists persist as dicts in
                          session state and are re-hydrated as
                          `AgentTool`s every turn.
- Safety allowlist      — `safety.py` is the non-negotiable gate on
                          which tools a runtime specialist may use.

What changed v1 → v2 (and why)
------------------------------
Five concrete v2 wins layered on top of the v1 design:

1. **Fixed team uses `sub_agents=[...]` + `mode='single_turn'` instead
   of `tools=[AgentTool(agent=X), ...]`** (the L3 lesson). The
   framework auto-derives `_SingleTurnAgentTool` instances and the
   specialists declare typed `input_schema` / `output_schema` Pydantic
   contracts. The coordinator delegates via auto-generated function
   tools named after each agent.

2. **`agent_creator` uses `mode='task'` for multi-turn confirmation
   HITL.** v1 used default chat mode + a sub_agent + manual
   transfer-back. v2 `task` mode keeps the multi-turn capability *with
   auto-return on `finish_task`*: the creator chats with the user
   ("Shall I proceed?"), receives the answer, calls
   `create_specialist`, then calls `finish_task` to return to the
   coordinator. No transfer plumbing.

3. **`disallow_transfer_to_parent=True` and
   `disallow_transfer_to_peers=True` on every fixed specialist with a
   built-in tool** (`data_fetcher_agent` uses `google_search`,
   `analyst_agent` uses `BuiltInCodeExecutor`). This is the v2
   migration gotcha #2 echo: ADK's `agent_transfer.py:152-188`
   auto-injects a `transfer_to_agent` function tool on every sub-agent
   whose parent transfer is allowed, which then conflicts with built-
   in tools (Gemini's "Built-in tools and Function Calling cannot be
   combined"). Suppressing the injection is mandatory here. Same fix
   as `level_3_agent`'s `search_agent`.

4. **`output_schema` dropped on agents that use built-in tools.** Same
   v2 limitation as L3: on Gemini API,
   `_OutputSchemaRequestProcessor` injects a `set_model_response`
   function tool that conflicts with built-ins. So
   `data_fetcher_agent` and `analyst_agent` return free text; only
   `report_writer_agent` (no built-ins) gets a typed `Brief`
   output_schema.

5. **Runtime specialists use `AgentTool` (the v1 pattern), not
   `sub_agents`** — because v2's `sub_agents → _SingleTurnAgentTool`
   auto-derivation runs once at `model_post_init` time
   (`llm_agent.py:982-994`), so mutating `sub_agents` from the
   `before_agent_callback` mid-session would not register new tools.
   `AgentTool` injection via `tools=[...]` works at every turn. This
   is the pedagogical split: **fixed teams use sub_agents; runtime
   teams use AgentTool**.

What stays from v1
------------------
- Three-specialist fixed team (search/analyse/write) plus the creator.
- `BuiltInCodeExecutor` on `analyst_agent` with the same hardened
  charting prompt — gotcha #20 (blank-version PNG) and gotcha #21
  (Pro hangs on code execution) still apply at the model layer.
- Three supporting modules: `safety.py` (allowlist), `registry.py`
  (state-backed capability persistence), `creator_tools.py`
  (`create_specialist` validate/smoke-test/persist).
- `before_agent_callback` rebuilds runtime tools each turn from
  `state.capabilities`, ensuring session isolation.

Sample queries to try
---------------------
- "hi"  →  coordinator handles directly.
- "What was Apple's Q1 2025 revenue and how does that compare to Q1 2024?"
        → fixed team: `data_fetcher_agent` → `analyst_agent` (computes
          deltas, makes a chart) → `report_writer_agent`.
- "Build a specialist that pulls Formula 1 race results."
        → coordinator detects gap → transfers to `agent_creator` →
          creator drafts spec, asks user to confirm → user says yes →
          `create_specialist` validates and persists → next turn the
          new `f1_data_agent` is callable from the coordinator.
"""

from __future__ import annotations

from typing import Literal

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.load_web_page import load_web_page
from google.genai import types
from pydantic import BaseModel
from pydantic import Field

from .creator_tools import create_specialist
from .mcp_toolset import gahmen_toolset
from .registry import hydrate_capabilities


# ---------------------------------------------------------------------------
# Pydantic schemas — typed contracts at every coordinator/specialist
# boundary. Schemas are dropped on agents that use built-in tools (see
# top docstring §4 for why); kept on schema-only agents.
# ---------------------------------------------------------------------------


class FetcherInput(BaseModel):
  """Argument schema the coordinator passes to data_fetcher_agent."""

  query: str = Field(
      description=(
          "Focused, self-contained search query for business data."
          " Examples: 'Apple Q1 2025 revenue', 'Tesla EV deliveries"
          " 2024 Q4', 'global lithium spot price October 2025'."
      )
  )


class AnalystInput(BaseModel):
  """Argument schema the coordinator passes to analyst_agent."""

  task: str = Field(
      description=(
          "What to compute or visualise, with the raw numbers / table"
          " inline. The analyst will run Python (pandas, matplotlib);"
          " never compute the answer in this argument — just describe"
          " the task."
      )
  )


class WriterInput(BaseModel):
  """Argument schema the coordinator passes to report_writer_agent."""

  topic: str = Field(description="The original user question / topic.")
  fetcher_findings: list[str] = Field(
      description=(
          "All raw findings from data_fetcher_agent calls. Pass an"
          " empty list if no fetcher calls were made."
      ),
  )
  analyst_findings: list[str] = Field(
      description=(
          "All printed numeric summaries from analyst_agent calls."
          " Quote VERBATIM in the brief — do not round. Pass an empty"
          " list if no analyst calls were made."
      ),
  )


class Brief(BaseModel):
  """Final structured BI brief returned from report_writer_agent."""

  title: str = Field(description="Short title for the brief.")
  executive_summary: str = Field(
      description="1-2 paragraph plain-English answer to the question."
  )
  key_metrics: str = Field(
      description=(
          "Markdown bullet list of the headline numbers, quoted"
          " verbatim from the analyst."
      )
  )
  analysis: str = Field(
      description="2-3 paragraph synthesis with inline source attribution."
  )
  sources: list[str] = Field(description="Source domains cited.")
  confidence_and_gaps: str = Field(
      description="Where the brief is most/least confident."
  )


# NOTE: there is intentionally no `CreatorMission` input_schema for
# `agent_creator` — see the agent_creator definition below for the v2
# task-mode multi-turn HITL caveat that explains why.


# ---------------------------------------------------------------------------
# Fixed team — three specialists + creator. Wired as `sub_agents` on the
# coordinator so v2's framework auto-derives `_SingleTurnAgentTool` /
# `_TaskAgentTool` entries (see top docstring §1, §2).
# ---------------------------------------------------------------------------


data_fetcher_agent = Agent(
    name="data_fetcher_agent",
    model="gemini-2.5-flash",
    description=(
        "Fetches public business data from three sources: Google"
        " Search (general web), specific URLs (via load_web_page),"
        " AND Singapore Government data via the gahmen-mcp MCP server"
        " (data.gov.sg datasets and SingStat statistical tables)."
        " Returns text facts with source attribution; does not"
        " compute or interpret."
    ),
    mode="single_turn",
    input_schema=FetcherInput,
    # NO output_schema: would inject set_model_response and conflict with
    # built-in tools on Gemini API. See top docstring (L4 §4).
    instruction=(
        "You are a business-data research specialist with three"
        " sources. Pick the right one for each query:\n"
        "  - google_search: general web search. Default for"
        " international companies, recent news, market context.\n"
        "  - load_web_page: fetch text of a specific URL. Use when the"
        " user provides a URL or you found a clearly relevant link"
        " in a search result.\n"
        "  - gahmen-mcp tools (datagovsg_*, singstat_*): Singapore"
        " Government data ONLY. Prefer SingStat for official macro /"
        " demographic / labour-market time series; prefer data.gov.sg"
        " for granular operational datasets. Use these when the"
        " query is clearly about Singapore (mentions SG / Singapore /"
        " HDB / MOM / etc., or asks about SG-specific economic /"
        " social statistics).\n\n"
        "Return raw facts as plain text with source attribution"
        " inline (e.g., 'According to data.gov.sg dataset"
        " <name>, ...' or 'According to bloomberg.com, ...'). Do not"
        " compute or interpret — that is the analyst's job."
    ),
    # Three tool surfaces:
    #   1. GoogleSearchTool (built-in, with bypass=True so the
    #      auto-swap converts it to GoogleSearchAgentTool when
    #      multiple_tools=True — which it is here).
    #   2. load_web_page (plain function tool).
    #   3. gahmen_toolset (McpToolset; expands to N function tools at
    #      canonical_tools time, narrowed by tool_filter to the eight
    #      read-only datagovsg_* / singstat_* tools).
    #
    # Final shape sent to Gemini: all function tools (search built-in
    # gets auto-swapped to function form). No built-in/function
    # conflict; same wiring trick as L4's data_fetcher and the runtime
    # allowlist.
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        load_web_page,
        gahmen_toolset,
    ],
    # Mandatory: prevents the auto-injected transfer_to_agent function
    # tool that would conflict with built-in tool surfaces. v2
    # migration gotcha #2 — see AGENTS.md.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


analyst_agent = Agent(
    name="analyst_agent",
    # Flash + BuiltInPlanner per AGENTS.md gotcha #21: Pro on a
    # BuiltInCodeExecutor leaf can hang 6+ min under AFC. The planner
    # turns Gemini's native thinking on for Flash so the model plans
    # cell layout before writing code — directly addressing gotcha #20
    # ("a later code cell closes/re-saves the figure → blank Version 1
    # overwrite hides the real chart at Version 0").
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
    description=(
        "Business analyst. Performs arithmetic (CAGR, percent change,"
        " weighted averages, ratios, arbitrary expressions), explores"
        " tabular data with pandas, and generates matplotlib charts."
        " Returns a text summary; charts surface in adk web via the"
        " framework's auto-save of BuiltInCodeExecutor image output"
        " into the artifact service."
    ),
    mode="single_turn",
    input_schema=AnalystInput,
    # NO output_schema: BuiltInCodeExecutor is a built-in tool, same
    # set_model_response conflict as data_fetcher_agent.
    instruction="""You are a business analyst backed by Gemini's code_execution sandbox. Use Python (not mental math) for every numeric result.

# Mandatory behaviour
1. Never mental-math. Every numeric result comes from `print(...)` inside executed Python.
2. You MUST produce at least one matplotlib chart whenever the input is a time series, a comparison across categories, or more than two numeric rows. Skip the chart only if the input is a single scalar or the user explicitly says "no chart".
3. Print numeric summaries with f-strings so the coordinator can quote them verbatim.

# Charting rules — follow EXACTLY
The chart MUST be produced in ONE single code cell. Splitting figure construction and `plt.show()` across cells is the #1 cause of blank PNGs in this sandbox: the figure created in cell A is closed by the time `plt.show()` runs in cell B, so the framework auto-saves an empty canvas as the artifact (AGENTS.md gotcha #20).

Required structure of the chart cell — every step in the SAME executable_code block, in this order:

    fig, ax = plt.subplots(figsize=(8, 5))     # explicit fig + ax — never bare plt.figure()
    # ... plot data INTO `ax` (ax.plot, ax.bar, df.plot(ax=ax), ...) ...
    ax.set_title(...); ax.set_xlabel(...); ax.set_ylabel(...)
    # Defensive sanity check — proves the figure has artists before show:
    n_artists = len(ax.lines) + len(ax.patches) + len(ax.collections)
    print("chart artists:", n_artists)        # MUST be > 0; if 0, fix the data wrangling above and retry
    plt.tight_layout()
    plt.show()

Hard rules for the chart cell:
- Label axes and title every chart.
- For time series, sort by the time axis before plotting.
- For pandas: build the pivot/grouped frame, `print(frame)` to verify it has non-NaN rows, THEN plot. A pivot that silently collapses to NaN is the second-most-common cause of blank charts.
- NEVER use `plt.figure()` without immediately drawing into it. Use `fig, ax = plt.subplots(...)` and plot via `ax.*` so the figure you show is the figure you populated.
- NEVER call `plt.savefig(...)` / write to `BytesIO` / write to any file path. The sandbox is sealed; bytes you write to disk never escape it. Only `plt.show()` produces an image Part the framework can intercept.
- NEVER call `plt.show()` more than once; never re-open the figure after show.
- NEVER call `save_artifact` or reference `tool_context` — they don't exist inside the sandbox (the runner injects `ToolContext` into Python function tools, not into Gemini's hosted code-execution environment).
- After the chart cell, run NO further `plt.*` calls.
- If `n_artists` printed 0, the chart cell failed: re-do the data wrangling, then re-run the FULL chart cell (figure + plot + show) as one block. Do NOT add a follow-up `plt.show()`-only cell.

# Environment
- Stateful across turns: do not re-initialise variables or re-load data.
- Pre-imported: io, math, re, matplotlib.pyplot as plt, numpy as np, pandas as pd, scipy.
- Do NOT run `pip install`.
""",
    code_executor=BuiltInCodeExecutor(),
    # Mandatory for the same reason as data_fetcher_agent — except here
    # the conflicting built-in is BuiltInCodeExecutor (auto-injected as
    # the executable_code tool surface) rather than google_search.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


report_writer_agent = Agent(
    name="report_writer_agent",
    model="gemini-2.5-flash",
    description=(
        "Formats accumulated findings into a structured BI brief."
        " Output is the final answer — do not re-paraphrase."
    ),
    mode="single_turn",
    input_schema=WriterInput,
    # SAFE to set output_schema here: this agent has no built-in tools,
    # so set_model_response can be injected without conflict. Demonstrates
    # the v2 typed-output contract for terminal nodes.
    output_schema=Brief,
    instruction=(
        "Synthesise a Brief from the fetcher_findings (raw facts) and"
        " analyst_findings (numeric summaries) for the topic. Quote"
        " analyst numbers VERBATIM — do not round, re-format, or"
        " re-interpret them. Weave fetcher source domains inline in the"
        " analysis. Lead with the most important finding in"
        " executive_summary."
    ),
    # No built-in tools, so transfer_to_agent injection wouldn't
    # conflict — but consistency: terminal sub-agents shouldn't
    # transfer; they should auto-return via request_task.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Agent creator — builds new specialists on demand. Uses mode='task'
# for multi-turn confirmation (the v2 way of doing in-agent HITL: it
# can ask the user "Shall I proceed?" and wait for the answer, then
# auto-return on finish_task). v1 used default chat + manual
# transfer-back; v2 mode='task' is structurally cleaner.
# ---------------------------------------------------------------------------


agent_creator = Agent(
    name="agent_creator",
    # Gemini 3.1 Pro for the creator specifically — the multi-turn
    # HITL flow (draft → wait for confirmation → call
    # create_specialist → call finish_task) is the kind of *chained*
    # tool-decision where Flash exhibits empty-STOP responses
    # (observed: "yes proceed" → 0 output tokens with
    # finish_reason=STOP). Gemini 3.1 Pro is described in the model
    # card (https://ai.google.dev/gemini-api/docs/gemini-3) as "our
    # most intelligent model… state-of-the-art reasoning" and it
    # supports compositional function calling (parallel + chained),
    # which maps directly onto this agent's create_specialist →
    # finish_task sequence. Cost is higher than Flash but creator is
    # invoked rarely (only when the team has a capability gap), so
    # the absolute spend stays small. Do NOT use Pro on
    # `analyst_agent` — gotcha #21 (Pro + BuiltInCodeExecutor under
    # AFC can hang 6+ min); Flash + BuiltInPlanner is the proven
    # combination there.
    model="gemini-3.1-pro-preview",
    description=(
        "Synthesises a new specialist agent when the BI team lacks a"
        " capability. Use when the user's request cannot be served by"
        " data_fetcher_agent, analyst_agent, or report_writer_agent."
    ),
    mode="task",
    # NO input_schema. v2 enforces input_schema on EVERY user turn within
    # a task-mode conversation (not just the initial coordinator-
    # delegated call). For multi-turn HITL like "draft spec → ask 'Shall
    # I proceed?' → user types 'yes'", the user's free-form replies
    # would fail input_schema validation. Drop the schema and let ADK
    # use the default flexible input — the creator's instruction
    # already documents the expected mission shape.
    # No output_schema — the creator's "result" is conveyed through the
    # create_specialist tool's return string (and the side effect on
    # state.capabilities). Adding output_schema would force the LLM
    # into a structured response that doesn't match the conversational
    # flow.
    instruction="""You are the Agent Creator. You build new specialists to fill capability gaps in the BI team.

# Available tools (the safety allowlist)
A new specialist may use any combination of these and only these tools — pass the short names verbatim in tool_set:

General tools (inherited from Level 4):
  - "google_search"     — Web search for current information.
  - "get_current_date"  — Returns today's date (YYYY-MM-DD). Pair with google_search for time-sensitive lookups like "current season", "this week's earnings", "recent IPOs".
  - "calculator"        — Safe math expression evaluator. Use for arithmetic in specialists that don't need full Python (currency conversion, percentages, basic stats). Supports + - * / // % **, sqrt/log/exp/sin/cos/floor/ceil, min/max/round/sum, constants pi/e.
  - "load_web_page"     — Fetch text content of a specific URL. Use for specialists that need to read a particular page rather than search.

Singapore Government data (MCP, level 4a only):
  - "datagovsg_list_collections"   — List all data.gov.sg dataset collections.
  - "datagovsg_get_collection"     — Get details of a specific data.gov.sg collection by ID.
  - "datagovsg_list_datasets"      — List datasets within a data.gov.sg collection.
  - "datagovsg_get_dataset_metadata" — Read schema / columns / freshness for a specific data.gov.sg dataset.
  - "datagovsg_search_dataset"     — Free-text search across data.gov.sg datasets.
  - "singstat_search_resources"    — Search SingStat (Department of Statistics SG) resources.
  - "singstat_get_metadata"        — Read metadata for a specific SingStat resource.
  - "singstat_get_table_data"      — Pull data rows from a specific SingStat table.

Use the gahmen-mcp tools ONLY for specialists answering Singapore-specific questions. The MCP server is rate-limited and slow to cold-start, so don't add these tools to a generalist spec.

A spec with `tool_set: []` is allowed too — that creates a pure-LLM specialist with no tools.

# Workflow (multi-turn — you may chat with the user during this)
1. Read the mission from the input. Draft a spec out loud:
   - name: snake_case identifier (e.g., "f1_data_agent")
   - description: one-sentence role summary used by the coordinator for routing
   - instruction: the system prompt the new agent will follow
   - tool_set: list of allowlisted tool names from the table above
2. Pick the MINIMAL set of tools the specialist needs. More tools = more decisions for its LLM. If google_search alone is enough, use just that.
3. **Ask the user to confirm**: "I propose creating a specialist called X that does Y using tools Z. Shall I proceed?" Wait for an affirmative reply.
4. On confirmation, call `create_specialist` with the spec.
5. Call `finish_task` to hand control back to the coordinator with the new specialist's name and description as the result.

# Rules
- Use ONLY tool names from the allowlist above (case-sensitive). Violations are rejected by `create_specialist`.
- Keep specialist instructions narrow — one capability per specialist.
- Never create a duplicate (the tool auto-dedupes by name).
- Never create a specialist that duplicates the existing fixed team.
- If the user says "no" or wants changes, revise the spec and ask again. Do not proceed without confirmation.
""",
    tools=[create_specialist],
    # mode='task' AND tools=[create_specialist] — the framework also
    # injects FinishTaskTool (because mode='task'), so this agent has
    # at least 2 function tools. Both are function tools, no built-in
    # conflict.
    #
    # Setting disallow_transfer_to_{parent,peers}=True suppresses the
    # auto-injection of `transfer_to_agent` from
    # `agent_transfer.py:152-188`. Without this, agent_creator's tool
    # surface is `[transfer_to_agent, create_specialist, finish_task]`
    # — three tools competing for the same "I'm done" semantics, which
    # empirically causes tool-choice paralysis on Gemini-2.5-flash
    # (the model returns finish_reason=STOP with zero output tokens
    # after "yes proceed"). With these flags set, the surface narrows
    # to `[create_specialist, finish_task]` — clean separation, no
    # ambiguity. The creator returns to the coordinator via
    # `finish_task` (the v2 task-mode auto-return path), not via
    # `transfer_to_agent` — so we lose nothing functional.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Coordinator — meta-reasoning router with runtime tool hydration.
# ---------------------------------------------------------------------------


def _rehydrate_runtime_tools(callback_context: CallbackContext):
  """Rebuild `root_agent.tools` from runtime capabilities on every turn.

  Level 4 feature: persistence across turns within a session AND across
  server restarts (the disk YAML library is loaded by
  `hydrate_capabilities`). v2 note: the auto-derived
  `_SingleTurnAgentTool` entries from `sub_agents` are added at
  `model_post_init` and are preserved in `_INITIAL_TOOLS`; this
  callback rebuilds `tools` as `fixed + runtime` on every turn.

  We mutate the module-global `root_agent.tools`, NOT
  `callback_context.agent.tools` — `CallbackContext` in v2 does not
  expose `.agent` (use `get_invocation_context().agent` if you need
  it, but here the module-global reference is the simpler equivalent
  and matches the v1 pattern).
  """
  runtime_tools = hydrate_capabilities(callback_context.state)
  root_agent.tools = list(_INITIAL_TOOLS) + runtime_tools


root_agent = Agent(
    name="level_4a_agent",
    model="gemini-2.5-flash",
    description=(
        "Self-evolving BI coordinator with Singapore Government data"
        " access via gahmen-mcp. Routes analytical questions to a"
        " fixed team (data_fetcher with web + URL + SG-MCP sources,"
        " analyst, report_writer) and synthesises a new specialist"
        " via agent_creator when the team lacks a needed capability."
    ),
    # PlanReActPlanner forces the LLM to emit explicit
    # PLANNING / REASONING / ACTION / FINAL_ANSWER blocks before
    # calling tools. For Level 4 the coordinator's plan IS the
    # meta-reasoning ("can my fixed team handle this? does a runtime
    # specialist match? do I need agent_creator?"), so making it
    # visible inline is the half of L4's teaching value that the
    # rewrite was missing. Same choice as Level 3 (where the
    # coordinator's routing decisions also benefit from inline
    # visibility). Different from Level 2 (deterministic graph) and
    # Level 1 (no orchestration to plan).
    planner=PlanReActPlanner(),
    instruction="""You are a BI coordinator. You delegate all work — you do not compute, search, or format yourself.

# Your fixed team (always available)
- `data_fetcher_agent` — three sources: web search (general), URL fetch (specific page), and gahmen-mcp (Singapore Government data via data.gov.sg + SingStat). The fetcher's LLM picks the right source per query.
- `analyst_agent` — all arithmetic, pandas exploration, matplotlib charts
- `report_writer_agent` — formats the final BI brief (call this LAST)
- `agent_creator` — builds new specialists when the team has a capability gap (multi-turn dialogue)

# Runtime specialists (PERSISTENT — may appear in your tool list)
Your tool list also contains runtime specialists previously created by `agent_creator`. These persist across server restarts via the `runtime_agents/` library, so a specialist created on Monday is still callable today. Each has a `description` you can read to decide if it matches the current task. **Always check this list FIRST — if a runtime specialist's description matches the user's request, call it directly. Only delegate to `agent_creator` when no fixed-team specialist AND no runtime specialist covers the request.**

# Greetings and meta questions ("hi", "hello", "what can you do?")
Respond directly yourself — do NOT delegate. Reply with two sentences: one describing the team, one inviting a question.

# Meta-reasoning — what to do for each request
Walk through this checklist in order:
1. Is the request a greeting / meta-question? → answer directly.
2. Does a runtime specialist's description match the request (e.g., F1 questions and there's an `f1_*_agent` in your tools)? → call THAT specialist. Do NOT recreate.
3. Is the request analytical (data lookup + math + brief)? → fixed-team flow: `data_fetcher_agent` → `analyst_agent` (one or more times) → `report_writer_agent`.
4. Is the request a true capability gap (none of the above match — e.g., "query our live Salesforce pipeline")? → delegate to `agent_creator` with a clear mission.

# Hard rules
1. NEVER compute a number yourself. Always call analyst_agent.
2. NEVER write the final BI brief yourself. Always call report_writer_agent.
3. Quote analyst_agent's printed numbers VERBATIM. Do not round, re-format, or re-interpret them.
4. For simple factual questions, one data_fetcher_agent call may be enough — skip the analyst and writer in that case.
5. Don't delegate to `agent_creator` if a runtime specialist already exists for the topic. The dedupe check will reject it anyway, but checking first saves a round trip.
""",
    sub_agents=[
        data_fetcher_agent,
        analyst_agent,
        report_writer_agent,
        agent_creator,
    ],
    before_agent_callback=_rehydrate_runtime_tools,
)


# Capture the framework's auto-derived `_SingleTurnAgentTool` /
# `_TaskAgentTool` entries created during `model_post_init` from
# `sub_agents=[...]`. The callback rebuilds tools as
# `_INITIAL_TOOLS + runtime_tools` each turn so runtime specialists
# can be added/removed without disturbing the fixed team's auto-tools.
_INITIAL_TOOLS = list(root_agent.tools)
