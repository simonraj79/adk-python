"""Level 2 — Strategic Problem-Solver (ADK 2.0 rewrite).

Taxonomy: "Level 2 — The Strategic Problem-Solver". The agent handles
complex, multi-part questions by **planning a strategy upfront**,
**executing multiple research steps**, and **synthesising** a
structured brief.

What changed v1 → v2 (and why)
------------------------------
v1 (`V1_level_2_agent/agent.py`) shipped this as a single `LlmAgent`
with three tools (`google_search`, `save_research_note`,
`get_research_notes`) and a long prompt that *asked* the LLM to follow a
PLAN → EXECUTE → SYNTHESISE workflow. The structure was prompt-driven —
nothing in the framework enforced it. If the LLM skipped planning and
went straight to a search, the demo still ran; the lesson was implicit.

v2 expresses the exact same workflow as a **graph**:

    START → process_input → classify → ┬── greeting → greeter → END
                                       └── research → planner →
                                           fan_out_research → writer → END

Concrete v2 features used and the v1 surface they replace:

| v2 feature                      | Replaces in v1                       |
|---------------------------------|--------------------------------------|
| `Workflow(edges=[...])`         | The prompt's "Workflow" comment      |
| `output_schema` (Pydantic)      | "decompose into 2-3 sub-questions" prompt |
| `Event(route=...)` + dict edges | "if greeting, respond directly" prompt branch |
| `ctx.run_node()` dynamic fan-out| Serial "for each sub-question" loop in prompt |
| `Event(state={...})` + `{key?}` instruction injection | `save_research_note` / `get_research_notes` custom tools |
| `output_key` + state delta on event | `tool_context.state` scratchpad reads/writes |

**The two custom tools are gone.** `save_research_note` and
`get_research_notes` were just a prompt-driven scratchpad over
`tool_context.state`. In v2 the orchestrator node writes parallel
findings to `state["findings"]` directly, and the writer reads them via
`{findings?}` instruction injection — less surface, no chance for the
LLM to "forget" to call save_note.

**Searches now happen in parallel.** v1 ran searches one at a time in a
prompt-driven loop; v2 spawns them concurrently via `ctx.run_node()`
(the `dynamic_fan_out_fan_in` upstream sample's pattern). For a 3-sub-
question research turn, wall-clock latency drops by ~3x.

**The graph is visible.** The web UI's graph viz panel renders the node
chain explicitly; learners can see "PLAN → fan-out → JOIN → WRITE"
during execution, with the active node highlighted live (a v2 feature
called out in `CHANGELOG-v2.md` as "Workflow graph visualization in web
UI… active node rendering in event graph").

Sample queries to try
---------------------
- "hi"                                          → greeting branch
- "what's the population of Tokyo?"             → simple, 1 sub-question
- "compare quantum computing vs neuromorphic"   → 2-3 sub-questions, parallel
- "how is solid-state battery research progressing across LFP, sulfide, and oxide chemistries?"
                                                → 3 sub-questions, parallel
"""

from __future__ import annotations

import asyncio
from typing import Literal

from google.adk import Agent
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.tools.google_search_tool import google_search
from google.adk.workflow import node
from google.genai import types
from pydantic import BaseModel
from pydantic import Field


# ---------------------------------------------------------------------------
# Schemas — the structured contracts the framework enforces between nodes.
# In v1 these were "expected" via prompt; in v2 they're typed boundaries.
# ---------------------------------------------------------------------------


class InputCategory(BaseModel):
  category: Literal["greeting", "research"] = Field(
      description=(
          "'greeting' for hi/hello/'what can you do?'/empty/unclear input."
          " 'research' for any actual question."
      )
  )


class ResearchPlan(BaseModel):
  sub_questions: list[str] = Field(
      description=(
          "1 to 3 focused sub-questions that, answered together, fully"
          " answer the user's question. For simple factual questions"
          " return exactly one sub-question equal to the original."
      )
  )


# ---------------------------------------------------------------------------
# Function nodes — small, deterministic glue between LLM agents.
# ---------------------------------------------------------------------------


def process_input(node_input: types.Content):
  """Stash the original user question in state so every downstream agent
  can read it via `{question}` instruction injection."""
  text = node_input.parts[0].text if node_input.parts else ""
  return Event(state={"question": text})


def route_input(node_input: dict):
  """Route on the classifier's structured decision."""
  return Event(route=node_input["category"])


# ---------------------------------------------------------------------------
# Worker LLM agents — each does one focused job. They're wired into the
# Workflow via `edges`, not via `sub_agents` (which is a v1 multi-agent
# transfer pattern; v2 uses the graph instead).
# ---------------------------------------------------------------------------


classify = Agent(
    name="classify",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the user's input as 'greeting' or 'research'. Greeting"
        " covers 'hi', 'hello', 'what can you do?', empty, unclear, or"
        " meta-questions about your capabilities. Research covers any"
        " concrete question that needs information gathering."
    ),
    output_schema=InputCategory,
    output_key="category",
)


greeter = Agent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction=(
        "Reply with exactly two sentences: one describing what you do (a"
        " strategic researcher that decomposes questions, searches in"
        " parallel, and writes a structured brief with sources), and one"
        " inviting a research topic."
    ),
    output_key="last_brief",
)


planner = Agent(
    name="planner",
    model="gemini-2.5-flash",
    instruction=(
        'Decompose the question "{question}" into 1 to 3 focused'
        " sub-questions. Each sub-question should be self-contained — a"
        " researcher answering it should not need the others. For simple"
        " factual questions, return exactly one sub-question equal to"
        " the original. For complex multi-aspect questions, return 2 or"
        " 3 sub-questions covering distinct facets."
    ),
    output_schema=ResearchPlan,
    output_key="plan",
)


# Researcher worker — one instance is spawned per sub-question by
# `fan_out_research` below. Receives the sub-question as its
# `node_input` (which becomes the user message it sees) and can use
# `google_search` to ground the answer. Plain text output; the
# orchestrator collects them into `state["findings"]`.
researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=(
        "Answer the question in the user message using google_search."
        " Return a concise 2-3 sentence finding with the source domains"
        " you cited inline (e.g., 'According to nature.com, …')."
    ),
    tools=[google_search],
)


writer = Agent(
    name="writer",
    model="gemini-2.5-flash",
    instruction=(
        'Synthesise a structured research brief for the question'
        ' "{question}" using the findings below.\n\n'
        "Findings (one per sub-question):\n"
        "{findings?}\n\n"
        "Output format — always use this structure:\n"
        "## Research Brief: [Topic]\n"
        "### Key Findings\n"
        "### Detailed Analysis\n"
        "### Sources\n"
        "### Confidence & Gaps\n\n"
        "Distinguish facts from opinions. Flag uncertainty. If a finding"
        " was thin, say so in 'Confidence & Gaps'."
    ),
    output_key="last_brief",
)


# ---------------------------------------------------------------------------
# Dynamic fan-out orchestrator — the v2 graph idiom for "spawn N
# workers in parallel where N is decided at runtime by the LLM."
# Pattern lifted from `contributing/workflow_samples/dynamic_fan_out_fan_in/`.
# ---------------------------------------------------------------------------


@node(rerun_on_resume=True)
async def fan_out_research(ctx: Context, node_input: dict):
  """Spawn one `researcher` per sub-question in parallel via
  `ctx.run_node()`, then aggregate results into `state["findings"]`.

  `node_input` here is the planner's structured output, which v2
  delivers as a `dict` (because the planner declared
  `output_schema=ResearchPlan`).
  """
  sub_questions = node_input["sub_questions"]
  yield Event(
      message=f"Researching {len(sub_questions)} sub-question(s) in parallel."
  )

  # Fan-out: kick off all researchers concurrently. `use_sub_branch=True`
  # tags each worker's events with a distinct branch so the trace UI can
  # render them as parallel lanes (CHANGELOG-v2: "branch isolation").
  tasks = [
      ctx.run_node(researcher, node_input=q, use_sub_branch=True)
      for q in sub_questions
  ]
  answers = await asyncio.gather(*tasks)

  # Fan-in: format findings as a markdown bullet list for the writer's
  # instruction template. Pairing each finding with its sub-question
  # makes the writer's job cross-referencing rather than re-summarising.
  findings_md = "\n\n".join(
      f"**Sub-question:** {q}\n**Finding:** {a}"
      for q, a in zip(sub_questions, answers)
  )
  yield Event(state={"findings": findings_md})


# ---------------------------------------------------------------------------
# The Workflow — explicit graph, not a prompt instruction list.
# Sequence-shorthand tuples and dict routing are the idiomatic v2 style
# (see `adk-workflow` skill, "Edge Patterns").
# ---------------------------------------------------------------------------


root_agent = Workflow(
    name="level_2_agent",
    description=(
        "A strategic web researcher that decomposes complex questions,"
        " runs sub-question searches in parallel, and delivers a"
        " structured brief with sources, analysis, and gaps."
    ),
    edges=[
        # Main pipeline: capture input → classify → route → branch.
        ("START", process_input, classify, route_input),
        # Two routes: greeting short-circuits to greeter and ends; research
        # goes through plan → fan-out → write.
        (route_input, {"greeting": greeter, "research": planner}),
        # Research arm: planner produces a ResearchPlan dict; the
        # orchestrator dynamically fans out researcher instances; the
        # writer synthesises a brief from state["findings"].
        (planner, fan_out_research, writer),
    ],
)
