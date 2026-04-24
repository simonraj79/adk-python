"""Level 4 — The Self-Evolving System: Business Intelligence Agent Team.

Extends the Level 3 pattern with meta-reasoning + dynamic agent creation.

Fixed team:
  - data_fetcher_agent : web search via `google_search` built-in
  - analyst_agent      : math + pandas + matplotlib via `BuiltInCodeExecutor`
  - report_writer_agent: formats the final BI brief

Level 4 additions:
  - agent_creator      : sub-agent that synthesizes new specialists on demand
  - bi_coordinator     : meta-reasons about capability gaps, delegates to the
                         fixed team OR to agent_creator when a gap is detected

Every specialist is a leaf LlmAgent wrapped as an `AgentTool` (rule R8 from
LEVEL_4_PLAN.md §3). `agent_creator` is the only `sub_agent` on the coordinator
because creation is a conversational flow, not a single tool call.

Reasoning mode: explicit `BuiltInPlanner` on `analyst_agent` only — Flash with
native thinking enabled to plan code-cell layout before execution. See
AGENTS.md gotcha #21 for why Pro is NOT used here despite the analyst's
multi-step workload, and gotcha #20 for the chart blank-version failure
mode the planner + hardened prompt below are designed to prevent.
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.genai import types

from .creator_tools import create_specialist
from .registry import hydrate_capabilities


# ---------------------------------------------------------------------------
# Fixed team — three specialists. Every one is a leaf agent with NO sub_agents
# so it can be safely wrapped as an AgentTool (LEVEL_4_PLAN.md §3.3 R6).
# ---------------------------------------------------------------------------

data_fetcher_agent = Agent(
    name="data_fetcher_agent",
    model="gemini-2.5-flash",
    description=(
        "Fetches public business data via Google Search: company revenues, "
        "financial filings, market data, industry benchmarks, news. Returns "
        "text facts with source URLs."
    ),
    instruction=(
        "You are a business-data research specialist. Use google_search to "
        "answer the query. Return the raw facts with source URLs. Do not "
        "compute or interpret — that is the analyst's job."
    ),
    tools=[google_search],
    output_key="last_search_result",
)


analyst_agent = Agent(
    name="analyst_agent",
    # Flash + BuiltInPlanner per AGENTS.md gotcha #21: Pro on a
    # BuiltInCodeExecutor leaf can hang 6+ min under AFC. The planner
    # turns Gemini's native thinking on for Flash so the model plans the
    # cell layout before writing code — the directly addresses gotcha #20
    # ("a later code cell closes/re-saves the figure → blank Version 1
    # overwrite hides the real chart at Version 0").
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True)
    ),
    description=(
        "Business analyst. Performs arithmetic (CAGR, percent change, "
        "weighted averages, ratios, arbitrary expressions), explores tabular "
        "data with pandas, and generates matplotlib charts. Returns a text "
        "summary; charts surface in adk web via the framework's auto-save of "
        "BuiltInCodeExecutor image output into the artifact service."
    ),
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
    print(f"chart artists: {n_artists}")        # MUST be > 0; if 0, fix the data wrangling above and retry
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
- If the user reports a blank chart in adk web's Artifacts tab, ask them to open the Version dropdown on the artifact and select `Version: 0` — when two image Parts are emitted in one turn (e.g. a real figure plus an empty `plt.show()` afterwards), the framework auto-saves both and the UI defaults to the latest (blank) version while the real chart hides under v0. Following the single-cell rule above prevents this from happening in the first place.

# How charts reach adk web (informational — you do not call this code)
`plt.show()` makes Gemini's code-execution return the figure as an image Part on the model response. ADK's `_code_execution.py` post-processor detects any `inline_data` image, calls `artifact_service.save_artifact(...)` with an auto-generated `YYYYMMDD_HHMMSS.png` filename, strips the bytes from the Part, and replaces them with text `"Saved as artifact: <name>"` plus an `artifact_delta` event. adk web watches that delta and renders the chart from the artifact store. This requires the Runner to have an `artifact_service` configured (adk web wires `InMemoryArtifactService` by default). If you ever run this agent through a custom Runner without one, the first chart will raise `ValueError('Artifact service is not initialized.')`.

# Environment
- Stateful across turns: do not re-initialise variables or re-load data.
- Pre-imported: io, math, re, matplotlib.pyplot as plt, numpy as np, pandas as pd, scipy.
- Do NOT run `pip install`.
""",
    code_executor=BuiltInCodeExecutor(),
    output_key="last_analysis",
)


report_writer_agent = Agent(
    name="report_writer_agent",
    model="gemini-2.5-flash",
    description=(
        "Formats accumulated findings into a structured BI brief. Output is "
        "the final answer — do not re-paraphrase."
    ),
    instruction="""Format the findings below into a structured BI brief.

Latest search result:
{last_search_result?}

Latest analyst output:
{last_analysis?}

Output format:
## BI Brief: [Topic]
### Executive Summary
### Key Metrics
### Analysis
### Sources
### Confidence & Gaps
""",
)


# ---------------------------------------------------------------------------
# Agent creator — the Level 4 delta. Not an AgentTool (creation is a
# conversational flow); it is the coordinator's only sub_agent.
# ---------------------------------------------------------------------------

agent_creator = Agent(
    name="agent_creator",
    model="gemini-2.5-flash",
    description=(
        "Synthesizes a new specialist agent when the BI team lacks a "
        "capability. Use when the user's request cannot be served by "
        "data_fetcher_agent, analyst_agent, or report_writer_agent."
    ),
    instruction="""You are the Agent Creator. You build new specialists to fill capability gaps.

# Workflow
1. Read the mission the coordinator gave you (e.g., "Build a specialist that
   pulls Formula 1 race results").
2. Draft a spec out loud:
   - name: snake_case identifier (e.g., "f1_data_agent")
   - description: one-sentence role summary used by the coordinator for routing
   - instruction: the system prompt the new agent will follow
   - tool_set: a list of ALLOWED tool names (see safety.py allowlist)
3. **Ask the user to confirm**: "I propose creating a specialist called X that
   does Y using tools Z. Shall I proceed?" Wait for an affirmative reply.
4. On confirmation, call `create_specialist` with the spec.
5. Report success or failure back with the new specialist's name and description.

# Rules
- Use ONLY tool names from the safety allowlist. Violations are rejected by
  the tool.
- Keep specialist instructions narrow — one capability per specialist.
- Never create a duplicate (the tool auto-dedupes by name).
- Never create a specialist that duplicates the existing fixed team.
""",
    tools=[create_specialist],
)


# ---------------------------------------------------------------------------
# Coordinator — meta-reasoning router. Wraps fixed specialists as AgentTools;
# agent_creator is a sub_agent for conversational delegation during creation.
# ---------------------------------------------------------------------------


# Fixed team tools. Kept in a module-level list so the hydration callback can
# rebuild `root_agent.tools` from this base + runtime specialists every turn,
# instead of accumulating across sessions.
_FIXED_TOOLS = [
    AgentTool(agent=data_fetcher_agent, propagate_grounding_metadata=True),
    AgentTool(agent=analyst_agent),
    AgentTool(agent=report_writer_agent, skip_summarization=True),
]


def _rehydrate_runtime_tools(callback_context: CallbackContext):
  """On each turn, rebuild tools = fixed + session-specific runtime tools.

  Level 4 feature F6 — persistence across turns within a session. Rebuilding
  from scratch each turn (rather than appending) keeps sessions isolated: a
  runtime specialist created in session A never leaks into session B.
  """
  runtime_tools = hydrate_capabilities(callback_context.state)
  root_agent.tools = list(_FIXED_TOOLS) + runtime_tools


root_agent = Agent(
    name="level_4_agent",
    model="gemini-2.5-flash",
    description=(
        "Self-evolving Business Intelligence coordinator. Routes analytical "
        "business questions to fixed specialists (data_fetcher, analyst, "
        "report_writer) and creates new specialists on demand when a "
        "capability gap is detected."
    ),
    instruction="""You are a BI coordinator. You delegate all work — you do not compute, search, or format yourself.

# Your team (call as AgentTools)
- `data_fetcher_agent` — web searches for business data
- `analyst_agent` — all arithmetic, pandas exploration, matplotlib charts
- `report_writer_agent` — formats the final BI brief (call this LAST)

Additional runtime specialists may appear in your tool list if created by
`agent_creator` earlier in the session. Use them when their description
matches the task.

# Meta-reasoning — capability-gap detection
Before answering, decide: can my team handle this?
- If YES: orchestrate calls. Typical order: data_fetcher_agent → analyst_agent
  (one or more times) → report_writer_agent.
- If NO (user needs a capability none of my specialists cover — e.g., "query
  our live Salesforce pipeline", "pull Formula 1 race data"): transfer to
  `agent_creator` with a clear mission. After creation, resume the original
  task using the new specialist on the next turn.

# Hard rules
1. NEVER compute a number yourself. Always call analyst_agent.
2. NEVER write the final BI brief yourself. Always call report_writer_agent.
3. Quote analyst_agent's printed numbers VERBATIM. Do not round, re-format, or
   re-interpret them.
4. For simple factual questions one data_fetcher_agent call may be enough —
   skip the analyst and writer in that case.
""",
    tools=list(_FIXED_TOOLS),
    sub_agents=[agent_creator],
    before_agent_callback=_rehydrate_runtime_tools,
)
