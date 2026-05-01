"""A2A Orchestrator — Pro-tier coordinator that consults Levels 1-4 via A2A,
delegates chart generation to an in-package code-executor sub-agent, and
finalizes via a writer sub-agent that produces a structured Brief.

This agent demonstrates Anthropic's "Orchestrator-workers" pattern at
deployment scale plus an internal multi-agent workflow:
- One LLM-driven coordinator that picks 1+ specialists per query
- Five remote A2A peers (deployed Vertex Agent Engines in asia-southeast1)
- Two in-package sub-agents (chart_agent + writer_agent) for finalization

ADK 2.0 primitives in use
-------------------------
- `Agent` (alias for LlmAgent) — three of them: orchestrator, chart, writer.
- `BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True))` —
  Pro's native thinking on the orchestrator (routing decisions); Flash's
  native thinking on chart_agent (chart-cell layout reasoning).
- `BuiltInCodeExecutor` on chart_agent — pandas/matplotlib in a sandboxed
  Python environment. Output PNGs become artifacts via the framework.
- 5 `FunctionTool` wrappers for the consult_level_* A2A tools.
- `mode='single_turn'` on every agent — no HITL, single-pass orchestration.
- `input_schema` + (where compatible) `output_schema` Pydantic contracts.
- v2 sub_agents auto-derive `request_task_<name>` tools on the
  orchestrator's tool surface — alongside the 5 consult_* function tools.

Why three agents (not one)
--------------------------
Gemini's API forbids combining BuiltInCodeExecutor (built-in) with
FunctionTools on the same `LlmAgent`. The orchestrator MUST have the 5
consult_* function tools, so chart-making MUST live on a separate agent.
Writer also lives separately because it has output_schema=Brief — which
injects set_model_response, also incompatible with built-ins. Same
factor-of-concerns Level 4 uses.

Why Pro on the orchestrator, Flash on the workers
--------------------------------------------------
Compositional function calling (decide which 2-5 of 5 specialists, then
optionally chain consult → chart → write) needs Pro. Chart generation is
a tool-heavy code-execution task — Flash + BuiltInPlanner is the proven
combination there (Level 4's analyst_agent comment line ~252). Writer is
template-shaped formatting — Flash is more than enough.

Cost: Pro on orchestrator (~$5/1M input tokens) is invoked rarely (per
query); Flash on chart + writer is cheap. Total per query: ~$0.10–0.30.
"""
from __future__ import annotations

from google.adk import Agent
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import BaseModel, Field

from .remote_tools import (
    consult_level_1,
    consult_level_2,
    consult_level_2b,
    consult_level_3,
    consult_level_4,
)


# ---------------------------------------------------------------------------
# Pydantic schemas — typed contracts at every coordinator → sub-agent boundary.
# ---------------------------------------------------------------------------


class ChartInput(BaseModel):
    """Argument schema the orchestrator passes to chart_agent."""

    task: str = Field(
        description=(
            "What chart to make, with the raw numbers / table / structured "
            "data inline. Describe the data and the chart type. The chart "
            "agent will run Python (pandas, matplotlib); never compute the "
            "chart in this argument — just describe it precisely."
        )
    )


class Brief(BaseModel):
    """Final structured report from writer_agent."""

    title: str = Field(description="Short title summarising the question.")
    executive_summary: str = Field(
        description="1–2 paragraph plain-English answer to the question."
    )
    key_findings: str = Field(
        description=(
            "Markdown bullet list of the 3–5 most important findings, each "
            "attributed to the consulted Level (e.g., 'Per Level 4 (gahmen "
            "SingStat), ...')."
        )
    )
    detailed_analysis: str = Field(
        description=(
            "2–3 paragraph synthesis weaving together the consulted Levels' "
            "replies. Inline source attribution. Reference any chart that "
            "was produced (the chart artifact is rendered separately by the "
            "framework — refer to it by description)."
        )
    )
    sources: list[str] = Field(
        description=(
            "Deduplicated list of source domains / Levels referenced. "
            "Include both the consulted Level names AND any external "
            "domains the Levels cited (e.g., 'Level 4 (gahmen SingStat)', "
            "'Level 1 (bloomberg.com)')."
        )
    )
    confidence_and_gaps: str = Field(
        description=(
            "Where the report is most/least confident. Call out any "
            "consult that returned [error]/[skip]/[empty] explicitly."
        )
    )


class WriterInput(BaseModel):
    """Argument schema the orchestrator passes to writer_agent."""

    topic: str = Field(description="The original user question / topic.")
    consulted_findings: list[str] = Field(
        description=(
            "All raw replies from consult_level_* tool calls. One entry per "
            "consult, including the Level name in the text. Pass an empty "
            "list only if no consults were made (which should not happen)."
        )
    )
    chart_description: str = Field(
        default="",
        description=(
            "If chart_agent was invoked, paste its text description of the "
            "chart here so the writer can reference it. Empty string if no "
            "chart was made."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-agent: chart_agent — Flash + BuiltInCodeExecutor for matplotlib charts.
# Same hardened charting prompt pattern as Level 4's analyst_agent (mostly
# verbatim — gotcha #20 / blank-PNG-version applies identically here).
# ---------------------------------------------------------------------------


chart_agent = Agent(
    name="chart_agent",
    # Flash + BuiltInPlanner per Level 4 line ~252 ("the proven combination"
    # for Flash on tool-heavy code-execution tasks). Pro on a
    # BuiltInCodeExecutor leaf can hang 6+ min under AFC — Level 4 gotcha #21.
    model="gemini-2.5-flash",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    ),
    description=(
        "Generates matplotlib charts from raw data passed by the "
        "orchestrator. Uses Python in a sandboxed code executor; the chart "
        "image is auto-saved as an artifact by the framework."
    ),
    mode="single_turn",
    input_schema=ChartInput,
    # NO output_schema: BuiltInCodeExecutor is a built-in tool, so
    # _OutputSchemaRequestProcessor's set_model_response injection
    # would conflict (Level 4 docstring §4).
    instruction="""You are a chart-making specialist backed by Gemini's
code_execution sandbox. Your job: turn the structured data in the input
into ONE matplotlib chart, plus a one-paragraph text description of what
the chart shows.

# Mandatory behaviour
1. Use Python (not mental math) for any number that appears in the chart.
2. The chart MUST be produced in ONE single code cell. Splitting figure
   construction and `plt.show()` across cells is the #1 cause of blank
   PNG artifacts (the figure created in cell A is closed by the time
   `plt.show()` runs in cell B, so the framework auto-saves an empty
   canvas — see Level 4's gotcha #20).
3. After the chart cell, print a one-paragraph text description of the
   chart (what it shows, the key insight) so the writer agent can
   reference it.

# Required structure of the chart cell — every step in the SAME executable_code block, in this order:

    fig, ax = plt.subplots(figsize=(8, 5))     # explicit fig + ax — never bare plt.figure()
    # ... plot data INTO `ax` (ax.plot, ax.bar, df.plot(ax=ax), ...) ...
    ax.set_title(...); ax.set_xlabel(...); ax.set_ylabel(...)
    # Defensive sanity check — proves the figure has artists before show:
    n_artists = len(ax.lines) + len(ax.patches) + len(ax.collections)
    print("chart artists:", n_artists)        # MUST be > 0
    plt.tight_layout()
    plt.show()

# Hard rules for the chart cell
- Label axes and title every chart.
- For time series, sort by the time axis before plotting.
- For pandas: build the pivot/grouped frame, `print(frame)` to verify it
  has non-NaN rows, THEN plot.
- NEVER use `plt.figure()` without immediately drawing into it.
- NEVER call `plt.savefig(...)` / write to `BytesIO` / write to any file
  path. The sandbox is sealed; only `plt.show()` produces an image Part
  the framework intercepts as an artifact.
- NEVER call `plt.show()` more than once; never re-open the figure.
- If `n_artists` printed 0, the chart cell failed: re-do the data
  wrangling, then re-run the FULL chart cell as one block.

# Environment
- Stateful across turns: do not re-initialise variables.
- Pre-imported: io, math, re, matplotlib.pyplot as plt, numpy as np,
  pandas as pd, scipy.
- Do NOT run `pip install`.

# After the chart
Output a one-paragraph plain-text description of what the chart shows.
This goes in your final response (the chart itself is already a Part
the framework saved as an artifact).
""",
    code_executor=BuiltInCodeExecutor(),
    # Mandatory: prevents the auto-injected transfer_to_agent function
    # tool that would conflict with BuiltInCodeExecutor's built-in tool
    # surface (Level 4 v2 migration gotcha #2 / docstring §3).
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Sub-agent: writer_agent — pure LLM, output_schema=Brief, no tools.
# Final node in the orchestration; produces the formatted report.
# ---------------------------------------------------------------------------


writer_agent = Agent(
    name="writer_agent",
    model="gemini-2.5-flash",
    description=(
        "Synthesises consulted findings (and optionally a chart "
        "description) into a Markdown-formatted report. Final node — "
        "its output is the user-visible report."
    ),
    mode="single_turn",
    input_schema=WriterInput,
    # NO output_schema. Earlier version used `output_schema=Brief`, which
    # forced the framework to serialize the writer's output as JSON — the
    # user saw raw `{"title": "...", "executive_summary": "..."}` in the
    # chat panel. For a terminal node meant to be HUMAN-READ, we want
    # free Markdown text, NOT a Pydantic schema. The Brief class above
    # is kept as a documentation contract for what fields the writer
    # produces; the actual enforcement is via the instruction's
    # "use these EXACT Markdown headings" rule below.
    #
    # Trade-off: we lose typed downstream consumption (no programmatic
    # caller can rely on a parsed Brief). For an orchestrator output
    # that goes directly to a Telegram user / adk web chat panel, that
    # trade-off is correct — readability beats parseability.
    instruction="""Synthesise a Markdown-formatted report from the
consulted_findings (raw replies from Level 1-4 specialists) and the
chart_description (if provided) for the topic.

# Output structure — use these EXACT Markdown headings, in this order

# {Concise Title — describes the question's topic}

## Executive Summary

[1–2 paragraph plain-English answer to the question. Lead with the most
important finding. If data is incomplete, state that upfront.]

## Key Findings

- [Finding 1] (Per Level X, ...)
- [Finding 2] (Per Level Y, ...)
- [3–5 findings total, each attributed inline to its source Level]

## Detailed Analysis

[2–3 paragraphs weaving together the consulted Levels' replies. Inline
source attribution throughout. If chart_description is non-empty,
reference the chart explicitly here ("The chart above shows ...") — the
chart image itself is rendered separately by the framework as an
artifact.]

## Sources

- Level X — [what it contributed]
- Level Y — [what it contributed]
- [Include external domains the Levels cited if relevant, e.g.
  "bloomberg.com (via Level 1)"]

## Confidence and Gaps

[Where the report is most/least confident. CALL OUT any consult that
returned `[error]`, `[skip]`, or `[empty]` explicitly — do not pretend
those consults succeeded.]

# Rules

1. Quote numbers / specific facts VERBATIM from the consulted findings.
   Do not round or paraphrase numeric citations.
2. EVERY claim must have a Level attribution inline. "Per Level 3, ..."
   or "According to Level 4 (gahmen SingStat), ..." — the user should
   be able to trace any fact back to its source Level.
3. If the chart was made, reference it by what it shows in Detailed
   Analysis. Do not embed the image (the framework handles rendering).
4. If a consulted finding starts with `[error]`, `[skip]`, or `[empty]`,
   acknowledge the failure plainly in Confidence and Gaps. Do not omit
   the failure from the report — it's part of the orchestration story.
5. Output ONLY the Markdown report. No preamble ("Here is your
   report:"), no commentary, no JSON. The Markdown above is your
   complete final response.
""",
    # No built-in tools, but consistency with terminal sub-agents — they
    # auto-return via request_task; they should never transfer.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Root orchestrator — Pro + thinking, 5 consult_* function tools, plus the
# two sub_agents above (which auto-derive chart_agent and
# writer_agent as additional function tools).
# ---------------------------------------------------------------------------


root_agent = Agent(
    name="a2a_orchestrator",
    # gemini-2.5-pro — GA, universally available in us-central1 with no
    # per-project allowlisting. We initially tried gemini-3.1-pro-preview
    # (which `models.list()` showed as registered in us-central1), but
    # actual inference returned `404 NOT_FOUND. Publisher Model ... was
    # not found or your project does not have access to it.` Preview
    # models often gate behind per-project allowlists separate from
    # regional registration. Lesson: for production deploys, prefer GA
    # over preview unless you've explicitly verified inference (not just
    # listing) works for THIS project.
    # If access to a Gemini 3 preview opens up later, the candidates to
    # try are `gemini-3-pro-preview` (3.0) or `gemini-3.1-pro-preview`
    # (3.1) — both showed in the listing for us-central1. For now, 2.5
    # Pro is the right choice; it's still excellent at compositional
    # function calling and the orchestrator's reasoning load.
    model="gemini-2.5-pro",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
    ),
    description=(
        "Pro-tier multi-agent orchestrator. Consults Levels 1-4 over A2A "
        "to gather facts, optionally delegates to chart_agent for "
        "matplotlib visualization, and finalizes via writer_agent into a "
        "structured Brief. Demonstrates Anthropic's Orchestrator-workers "
        "pattern at Vertex Agent Engine scale + an internal multi-agent "
        "workflow."
    ),
    # No `mode=` here — defaults to 'chat'. ADK 2.0 requires the root agent
    # to be `mode='chat'`; only sub-agents can be `mode='single_turn'` or
    # `mode='task'`. The orchestrator is user-facing (Telegram chat) so
    # 'chat' is correct. The semantic difference from `single_turn`: the
    # framework keeps dialogue state across turns, so a follow-up like
    # "now also show me the EU rate" would let the orchestrator extend
    # the previous answer rather than starting fresh. For one-shot queries
    # the behaviour is identical.
    instruction="""You are an A2A orchestrator. You answer user questions
through a fixed three-stage workflow:

  STAGE 1 — Consult specialists (always at least 2 distinct Levels)
  STAGE 2 — Optionally make a chart (only when the answer benefits from one)
  STAGE 3 — Always delegate to writer_agent for the final formatted report

# STAGE 1: Specialists (call AT LEAST TWO different consult_* tools)

  - consult_level_1(query) — Single-shot Google Search specialist. Use
    for simple factual lookups answerable with one web search.
  - consult_level_2(query) — Day Planner / Strategic Problem-Solver. Use
    for planning tasks (study schedule, multi-step decomposition).
  - consult_level_2b(query) — Customer-support classifier (returns JSON
    category labels). Use for triage-style messages.
  - consult_level_3(query) — Research Coordinator with internal
    search/analyst/writer sub-agents. Use for multi-aspect research
    where you want a structured brief.
  - consult_level_4(query) — Self-Evolving BI agent. THE pick for any
    Singapore-specific data query (gahmen MCP for SingStat / data.gov.sg)
    AND for BI questions needing computation. Internally also consults
    Level 1 for non-SG facts.

  Rules:
  - SG-specific data → consult_level_4 (NOT consult_level_1).
  - Multi-aspect research → consult_level_3 + (often) consult_level_1.
  - Planning + facts → consult_level_2 (planning) + consult_level_1 or
    consult_level_4 (facts).
  - If a tool returns text starting with [skip]/[error]/[empty],
    acknowledge the failure — do not pretend the consult succeeded.

# STAGE 2: Chart (CONDITIONAL — only when the question warrants one)

After all consults, ASK YOURSELF: would a chart help answer the question?
Yes IF:
  - The user asked for a comparison across categories or time.
  - The findings include numeric series (multiple data points).
  - A chart would communicate the answer faster than prose.
No IF:
  - The user asked a single-number factual question.
  - The findings are qualitative (no numbers).
  - The user explicitly said "no chart" or asked for text only.

If yes: call chart_agent with `task=<a precise description
of what to chart, including the raw data inline>`. Pass the actual
numbers — do not say "the analyst can compute". The chart_agent does
NOT have access to the consult replies; YOU must include the data in the
task argument. Example:

  chart_agent(task=(
    "Bar chart comparing Singapore unemployment rate Q1-Q4 2020 vs "
    "United States Q1-Q4 2020. Data:\n"
    "Singapore: Q1=2.4, Q2=2.8, Q3=3.4, Q4=3.3 (per Level 4 — gahmen)\n"
    "United States: Q1=3.6, Q2=13.0, Q3=8.8, Q4=6.7 (per Level 1)\n"
    "Title 'SG vs US Unemployment 2020'. Y-axis 'Rate (%)'."
  ))

The chart will appear as an image artifact (auto-saved by the framework).
The chart_agent will also return a one-paragraph text description.

# STAGE 3: Writer (ALWAYS — this is the final step)

After consults (and chart if applicable), call writer_agent
with:
  - topic: the original user question (verbatim if short, paraphrased if
    very long).
  - consulted_findings: a list of all the raw text replies from the
    consult_level_* calls. Preserve Level attribution in each entry
    (e.g., the entry might start "From Level 4: ...").
  - chart_description: the text description chart_agent returned (or
    empty string "" if no chart was made).

The writer returns a Brief (structured Pydantic model). Whatever the
writer returns IS your final answer to the user — DO NOT re-paraphrase
it, DO NOT add commentary. The framework will surface the Brief plus the
chart artifact (if any) to the user.

# Anti-empty-STOP guard (DO NOT skip)

After EVERY tool call returns, you MUST do exactly one of:
  (a) call the next consult_* tool (still in STAGE 1),
  (b) call chart_agent (transitioning to STAGE 2),
  (c) call writer_agent (transitioning to STAGE 3),
  (d) wait for the writer's response and return it.
Never return with empty content and no tool call. Your native thinking
compute is enabled — use it to reason about which stage you're in.
""",
    tools=[
        FunctionTool(consult_level_1),
        FunctionTool(consult_level_2),
        FunctionTool(consult_level_2b),
        FunctionTool(consult_level_3),
        FunctionTool(consult_level_4),
    ],
    # The two in-package sub-agents. v2 framework auto-derives
    # `chart_agent` and `writer_agent` function
    # tools on this orchestrator's tool surface (because both have
    # mode='single_turn'). So the orchestrator's effective tool list is
    # 5 consult_* + 2 request_task_* = 7 function tools, all uniform.
    sub_agents=[chart_agent, writer_agent],
    # Cleanest tool surface — no auto-injected transfer_to_agent. The
    # orchestrator IS the root; nothing to transfer to.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
