"""Level 2 — Strategic Problem-Solver: Student Day Planner (ADK 2.0).

Taxonomy: "Level 2 — The Strategic Problem-Solver". The agent handles
complex multi-part requests by **planning a strategy upfront**,
**executing multiple lookup steps in parallel**, and **synthesising**
a structured output. The use case is a **student day / week planner**:
the user describes their commitments and goals, the agent decomposes
them into study blocks, looks up background info per topic, and
produces a markdown timetable.

Why a day-planner (not a research assistant)
--------------------------------------------
The taxonomy is "strategic planning + context engineering." Day
planning is the same shape as research (decompose → look up → assemble)
but the output is concretely useful: students get a usable schedule.
The previous "research a topic" framing was abstract and produced
walls of text most learners never read. A day plan is short, scannable,
and the value is immediate.

ADK 2.0 features in use (genuine — not feature theatre)
-------------------------------------------------------
- `Workflow(BaseNode)` graph orchestration
- 3-way `Literal` classifier with `__DEFAULT__` fallback in dict routing
  (fixes the "[NO DEFAULT]" warning the Workflow runtime emits when
  unknown classifications would dead-end silently)
- Function node (`anchor_today`) — deterministic glue, demonstrates v2's
  function-as-node pattern
- LLM agent nodes (greeter, quick_answerer, task_planner, schedule_writer)
  embedded directly in `edges=[...]` — auto-wrapped via the v2 graph
  runtime
- `ctx.run_node()` dynamic fan-out — the planner LLM decides how many
  study topics need research at runtime; one researcher per topic spawns
  concurrently. Pattern from
  `contributing/workflow_samples/dynamic_fan_out_fan_in/`.
- `Event(state={...})` + `{key?}` instruction injection — replaces v1's
  `save_research_note` / `get_research_notes` custom tools with pure
  framework state.
- `@node(rerun_on_resume=True)` on the dynamic-fan-out orchestrator so
  it's HITL-resumable (`CHANGELOG-v2.md`: "lazy scan deduplication and
  resume for dynamic nodes").
- `output_key` on terminal LLM agents for state-delta observability.

Graph
-----

    START
      ↓
    process_input
      ↓
    classify (2-way: quick / plan)
      ↓
    route_input
      ├── quick → quick_answerer → END
      └── plan:
          ↓
        anchor_today (function node — stashes today's date in state)
          ↓
        task_planner (LLM — extracts commitments, drafts study blocks)
          ↓
        fan_out_research (dynamic — google_search per study topic, in parallel)
          ↓
        schedule_writer (LLM — assembles markdown timetable)
          ↓
          END

Why two classifier routes (not three)
-------------------------------------
Earlier drafts had a separate `greeter` node on a `greeting` route.
That looked like a dead-end in the graph viz and added structural
noise — Levels 3 and 4 don't have it (their lead agent handles
greetings inline via prompt). For consistency with the higher
levels, this Level 2 collapses greeting + quick-question into a
single `quick` route. The `quick_answerer`'s instruction handles
greetings as a special case: if the input is a greeting, it
describes the agent's capabilities + suggests two example queries;
otherwise it answers the factual question. Same idiom as L3/L4,
just inside a Workflow node instead of an LlmAgent's instruction.

Sample queries
--------------
- "hi" → greeter introduces capabilities + suggests example queries
- "what's the Pomodoro technique?" → quick_answerer (single search)
- "I have an exam Friday on linear algebra and a club meeting
  Wednesday 6pm. Plan my week to study." → full pipeline:
  anchor_today → task_planner → 2-3 parallel topic researches →
  schedule_writer
"""

from __future__ import annotations

import asyncio
import datetime
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
# Schemas — typed contracts the framework enforces between nodes.
# ---------------------------------------------------------------------------


class InputCategory(BaseModel):
  category: Literal["quick", "plan"] = Field(
      description=(
          "'quick' for greetings ('hi', 'hello', 'what can you do?',"
          " empty, unclear) AND for any single-step factual question"
          " that does not require multi-step planning (e.g., 'what's"
          " the Pomodoro technique?', 'how long should a study session"
          " be?'). The quick_answerer's instruction handles greeting"
          " inputs inline — same pattern as Level 3 and Level 4 where"
          " the lead agent handles greetings directly without a"
          " separate greeter node.\n\n  'plan' for a request that"
          " describes commitments / goals and asks for a structured day"
          " or week schedule."
      )
  )


class StudyPlan(BaseModel):
  study_topics: list[str] = Field(
      description=(
          "1 to 3 study topics extracted from the user's commitments that"
          " would benefit from a quick web lookup before scheduling (e.g.,"
          " exam subjects, project areas). Empty list if the request is"
          " purely calendar-shaped (no topics to research)."
      )
  )
  commitments: str = Field(
      description=(
          "Plain-text summary of the user's commitments and goals,"
          " preserving stated times/dates so the writer can build the"
          " timetable from them."
      )
  )


# ---------------------------------------------------------------------------
# Function nodes — small, deterministic glue between LLM agents.
# ---------------------------------------------------------------------------


def process_input(node_input: types.Content):
  """Stash the user's request in state for downstream agents."""
  text = node_input.parts[0].text if node_input.parts else ""
  return Event(state={"request": text})


def route_input(node_input: dict):
  """Route on the classifier's structured decision."""
  return Event(route=node_input["category"])


def anchor_today(ctx: Context):
  """Stash today's date in state so the task_planner can plan against
  concrete days. Pure local function — no LLM call, no tool call.

  Demonstrates v2's function-as-node pattern: the planner reads the
  date via `{today?}` instruction injection, no `get_current_date`
  tool needed.
  """
  today = datetime.date.today()
  return Event(
      state={
          "today": today.isoformat(),
          "today_human": today.strftime("%A, %B %d %Y"),
      }
  )


# ---------------------------------------------------------------------------
# LLM agents — each does one focused job, embedded in the graph as nodes.
# ---------------------------------------------------------------------------


classify = Agent(
    name="classify",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the user's input as 'quick' or 'plan'."
        "\n\n  QUICK: greetings ('hi', 'hello', 'what can you do?',"
        " empty, unclear) AND single-step factual questions that don't"
        " describe commitments or ask for scheduling — e.g., 'what's"
        " the Pomodoro technique?', 'how long should I study before a"
        " break?', 'tips for memorizing formulas'. The quick_answerer's"
        " instruction handles greeting inputs inline — same pattern as"
        " Level 3 and Level 4 where the lead agent handles greetings"
        " without a separate greeter node."
        "\n  PLAN: the user describes commitments (deadlines, classes,"
        " meetings, exam dates) AND asks for a schedule/plan/timetable."
    ),
    output_schema=InputCategory,
    output_key="category",
)


# `quick_answerer` is the agent the user actually talks to in the
# greeting/quick-question case. Its instruction handles BOTH:
#   - greetings → describe what the agent does + give 2 example queries
#     (same pattern as L3 / L4's lead agent — a dedicated greeter node
#     would be redundant)
#   - factual questions → answer in 2-3 sentences, using google_search
#     only when current information is needed.
quick_answerer = Agent(
    name="quick_answerer",
    model="gemini-2.5-flash",
    description=(
        "Greets the user and answers single-step factual questions"
        " without going through the full planning pipeline."
    ),
    instruction=(
        'The user said: "{request}"\n\n'
        "Decide which case applies and respond accordingly:\n"
        "\n"
        "1. GREETING / META QUESTION ('hi', 'hello', 'what can you"
        " do?', empty, or unclear input).\n"
        "   Respond in two short paragraphs:\n"
        "   - One sentence describing what you do: you help students"
        " turn a list of commitments (classes, exams, deadlines,"
        " meetings) into a structured study schedule, looking up"
        " background info on study topics in parallel as needed.\n"
        "   - Two example queries the user could try, as a short"
        " bulleted list:\n"
        "     - A planning example: 'I have an exam Friday on linear"
        " algebra and a club meeting Wednesday 6pm — plan my week.'\n"
        "     - A quick-question example: 'What's the Pomodoro"
        " technique?'\n"
        "   Under 100 words total. Don't explain HOW you work"
        " internally; just what the user can ask. Do NOT call"
        " google_search for greetings.\n"
        "\n"
        "2. FACTUAL QUESTION (everything else routed to you — study"
        " techniques, productivity methods, general knowledge).\n"
        "   Answer in 2-3 sentences. Use google_search ONLY if the"
        " answer needs current information (e.g., 'is the campus"
        " library open today?'). For evergreen questions, answer"
        " from your own knowledge — no search needed."
    ),
    tools=[google_search],
    output_key="last_response",
)


task_planner = Agent(
    name="task_planner",
    model="gemini-2.5-flash",
    instruction=(
        "Today is {today_human?} ({today?}).\n\n"
        'The user request: "{request}"\n\n'
        "Extract the user's commitments (classes, exams, deadlines,"
        " meetings) preserving stated times/dates, then identify 1-3"
        " study topics from those commitments that would benefit from"
        " a quick background lookup (e.g., the exam subject, the"
        " project area). If the request is purely calendar-shaped"
        " (just times and meetings, no study topics), return an"
        " empty study_topics list."
    ),
    output_schema=StudyPlan,
    output_key="study_plan",
)


# Researcher — one instance is spawned per study topic by
# `fan_out_research` below.
researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=(
        "Use google_search to gather a 2-3 sentence study brief on the"
        " topic in the user message. Focus on: key concepts to review,"
        " common pitfalls, and one good study resource. Concise prose,"
        " no bullet points."
    ),
    tools=[google_search],
)


schedule_writer = Agent(
    name="schedule_writer",
    model="gemini-2.5-flash",
    instruction=(
        "Today is {today_human?} ({today?}). Produce a markdown"
        " timetable for the user.\n\n"
        'User request: "{request}"\n\n'
        "Commitments and goals (extracted by the planner):\n"
        "{commitments?}\n\n"
        "Study briefs (one per topic the user mentioned, from"
        " parallel web research):\n"
        "{topic_briefs?}\n\n"
        "Output format:\n"
        "## Schedule for {today_human?}\n"
        "(Or '## Week of {today_human?}' if the user described a"
        " multi-day plan.)\n\n"
        "For each scheduled block, use this row format:\n"
        "- **HH:MM-HH:MM** — *Activity*. (1-2 sentence rationale or"
        " study tip drawn from the briefs.)\n\n"
        "Rules:\n"
        "- Anchor every block to a real time. Use the user's stated"
        " times verbatim; fill in study/break/meal blocks around them.\n"
        "- Pomodoro pacing (25 min focus / 5 min break) is a sensible"
        " default for study blocks but use your judgment based on the"
        " briefs.\n"
        "- Include short breaks and at least one meal block — students"
        " forget to eat. Studies rarely benefit from blocks longer than"
        " 90 minutes.\n"
        "- End with a 1-sentence 'Confidence and gaps' note: anything"
        " the user didn't tell you that affects the plan (e.g., 'I"
        " assumed dinner at 18:30 since you didn't specify.')"
    ),
    output_key="last_response",
)


# ---------------------------------------------------------------------------
# Dynamic fan-out — spawn one researcher per study topic in parallel.
# Pattern from `contributing/workflow_samples/dynamic_fan_out_fan_in/`.
# ---------------------------------------------------------------------------


@node(rerun_on_resume=True)
async def fan_out_research(ctx: Context, node_input: dict):
  """Spawn one `researcher` per study topic via `ctx.run_node()`,
  collect results into `state["topic_briefs"]` (markdown bullets),
  and stash `commitments` for the writer.

  `node_input` is the planner's structured output (delivered as
  `dict` because the planner declared `output_schema=StudyPlan`).
  """
  study_topics = node_input.get("study_topics", []) or []
  commitments = node_input.get("commitments", "")

  if not study_topics:
    # Pure calendar request — no topics to research. Skip straight to
    # the writer with an empty briefs section.
    yield Event(
        state={
            "commitments": commitments,
            "topic_briefs": "(no study topics to research — pure calendar request)",
        }
    )
    return

  yield Event(
      message=(
          f"Researching {len(study_topics)} study topic(s) in parallel:"
          f" {', '.join(study_topics)}."
      )
  )

  tasks = [
      ctx.run_node(researcher, node_input=topic, use_sub_branch=True)
      for topic in study_topics
  ]
  briefs = await asyncio.gather(*tasks)

  briefs_md = "\n\n".join(
      f"**{topic}**: {brief}" for topic, brief in zip(study_topics, briefs)
  )
  yield Event(state={"commitments": commitments, "topic_briefs": briefs_md})


# ---------------------------------------------------------------------------
# The Workflow — explicit graph, three legitimate terminal paths.
# ---------------------------------------------------------------------------


root_agent = Workflow(
    name="level_2_agent",
    description=(
        "Student day-planner: turns a list of commitments into a"
        " structured study schedule with parallel topic research."
        " Demonstrates Level 2's strategic-planning + context-"
        "engineering capability via the v2 Workflow(BaseNode) graph."
    ),
    edges=[
        # Capture input → classify → route on the classifier's decision.
        ("START", process_input, classify, route_input),
        # Two terminal routes. The classifier's output_schema is a
        # Pydantic `Literal["quick", "plan"]` — Pydantic rejects any
        # other value at the schema layer before reaching
        # `route_input`. Greetings are handled INSIDE the
        # `quick_answerer`'s instruction (same pattern as Level 3 and
        # Level 4 where the lead agent handles greetings inline,
        # without a dedicated greeter node).
        (
            route_input,
            {
                "quick": quick_answerer,
                "plan": anchor_today,
            },
        ),
        # Plan arm: anchor today → planner → dynamic fan-out → writer.
        (anchor_today, task_planner, fan_out_research, schedule_writer),
    ],
)
