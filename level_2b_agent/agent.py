"""Level 2b — Minimal Graph Router (ADK 2.0).

A pared-down sibling to `level_2_agent`. Same capability tier (Level 2 —
Strategic Problem-Solver / context engineering via a graph), but the
shape is the **smallest useful demo of the v2 graph primitive**:
classify → route → handler. No fan-out, no parallel research, no
synthesis stage. The point is to show the v2 routing primitive in
isolation.

Use case
--------
Customer support triage. The user sends a message; the agent classifies
it as a bug, billing question, feature request, or greeting, and routes
to the matching handler. Each handler is a small focused `LlmAgent`
that responds in-character for its category (asks for repro on bugs,
recites pricing rules on billing, etc.).

Inspired by the `graph_router` demo in the *ADK 2.0 launch* video. The
video's pedagogical message: in v1 you'd cram "first classify, then
call the right handler, never skip steps" into one giant prompt and
**hope** the LLM followed; in v2 the routing logic moves out of the
prompt and **into the graph**, so it's deterministic. This file is the
project's canonical illustration of that lesson.

Why this is `level_2b_agent` and not `level_5_agent`
----------------------------------------------------
The capability ladder in `AGENT_LEVELS.md` maps to Google's published
taxonomy (L0–L4) and stops there. New ADK 2.0 *primitives* are absorbed
as `Na/Nb/Nc` variants of the closest capability tier — they don't
inflate the ladder. This demo's primitive (`Workflow(BaseNode)` + dict
routing on a `Literal` classifier) sits squarely inside L2's
"strategic / context-engineering" tier; it's a different *shape* of L2,
not a new tier. Same rationale as L4a (MCP variant of L4) and L1a
(voice variant of L1).

Why the "lead agent handles greetings inline" idiom uses a 4th route
--------------------------------------------------------------------
`AGENT_LEVELS.md:721-731` documents the project-wide convention that
the agent the user talks to handles greetings inline rather than via a
dedicated greeter sub-agent. In a coordinator-LlmAgent shape (L3, L4),
that means a prompt branch on the coordinator. In a Workflow shape,
the equivalent is **a route on the classifier**: the classifier is the
"lead" the user's first message hits, and `GREETING` is one of its
output categories. A small `greet_user` agent then handles only that
route. Same idiom, expressed in graph form. The alternative — squeezing
"greet AND classify" into the same agent's instruction — fights with
`output_schema=Literal[...]` (greetings have no category to emit) and
muddies the graph viz.

ADK 2.0 features in use
-----------------------
- `Workflow(BaseNode)` graph orchestration
- `output_schema=Literal[...]` classifier with dict routing map
- `Event(state={...})` to stash the user's request for handlers'
  `{request}` template injection
- LLM agents embedded directly in `edges=[...]` — auto-wrapped (per
  `adk-workflow` skill: "Place `Agent` instances directly in workflow
  edges. They are auto-run as nodes")
- `output_key` on each handler so the final response shows up as a
  `state_delta` on the yielded event (observability)

Graph
-----

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

Sample queries
--------------
- "hi" / "what can you do?"
    → GREETING route → greet_user introduces capabilities + suggests
      example queries
- "the dashboard shows 500 errors every time I open analytics"
    → BUG route → bug_handler asks for repro steps, severity, recent
      changes
- "how much does the Pro plan cost and what's included?"
    → BILLING route → billing_handler answers pricing/plan questions
- "it would be great if you could add dark mode to the dashboard"
    → FEATURE route → feature_handler thanks the user, captures the
      use-case, sets expectations on triage

Run
---
    adk web .
    # → pick `level_2b_agent` in the picker
"""

from __future__ import annotations

from typing import Literal

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.genai import types
from pydantic import BaseModel
from pydantic import Field


# ---------------------------------------------------------------------------
# Schemas — typed contracts the framework enforces between nodes.
# ---------------------------------------------------------------------------


class TicketCategory(BaseModel):
  """Classifier output — a strict 4-way enum the routing dict matches on.

  Pydantic's `Literal` constraint rejects any other value at the
  schema layer, so `route_input` is guaranteed to receive one of these
  four strings. No `__DEFAULT__` fallback edge needed.
  """

  category: Literal["GREETING", "BUG", "BILLING", "FEATURE"] = Field(
      description=(
          "The category of the user's message.\n\n"
          "  GREETING: 'hi', 'hello', 'what can you do?', empty input,"
          " or any conversational opener with no actionable content.\n"
          "  BUG: a report of broken/unexpected behavior — error"
          " messages, crashes, things 'not working', wrong output.\n"
          "  BILLING: questions about price, plans, charges, invoices,"
          " payment methods, refunds, subscriptions.\n"
          "  FEATURE: a request for new capability, an enhancement"
          " suggestion, or 'it would be nice if...' framing."
      )
  )


# ---------------------------------------------------------------------------
# Function nodes — small deterministic glue between LLM agents.
# ---------------------------------------------------------------------------


def process_input(node_input: types.Content):
  """Stash the user's raw message in state so handlers can read it via
  `{request}` instruction injection. The classifier itself doesn't
  need state — it just classifies `node_input` directly — but every
  handler downstream needs the original text.
  """
  text = node_input.parts[0].text if node_input.parts else ""
  return Event(state={"request": text})


def route_input(node_input: dict):
  """Branch on the classifier's structured decision. `node_input` is a
  dict because `classify` declared `output_schema=TicketCategory`.
  """
  return Event(route=node_input["category"])


# ---------------------------------------------------------------------------
# LLM agents — one classifier and four handlers.
# ---------------------------------------------------------------------------


classify = Agent(
    name="classify",
    # us-central1 + Pro 2.5 (W9.2 — 3.x preview gated per audit 2026-05-01).
    model="gemini-2.5-pro",
    description=(
        "Classifies an inbound support message into one of four"
        " categories. Pure routing logic — does not respond to the"
        " user."
    ),
    instruction=(
        "Classify the user's message into exactly one category."
        " Return only the structured output — do not write any natural"
        " language reply. The downstream handler will respond to the"
        " user.\n\n"
        "Conventions:\n"
        "- Conversational openers with no concrete request"
        ' ("hi", "hello there", "what can you do?", "help")'
        " → GREETING.\n"
        "- Anything describing broken behavior, errors, crashes, or"
        ' "X is not working" → BUG.\n'
        "- Anything about price, plan tiers, billing/invoice issues,"
        " or payment → BILLING.\n"
        "- Anything proposing a new capability or enhancement"
        " → FEATURE."
    ),
    output_schema=TicketCategory,
    output_key="category",
)


# Each handler reads the raw user message from state via `{request}`.
# Single-purpose prompts; each one is the agent the user talks to on
# its branch.

greet_user = Agent(
    name="greet_user",
    # us-central1 + Pro 2.5 (W9.2 — 3.x preview gated per audit 2026-05-01).
    model="gemini-2.5-pro",
    description=(
        "Handles greeting / capability-question routes by introducing"
        " the agent and suggesting example queries."
    ),
    instruction=(
        'The user said: "{request}"\n\n'
        "Respond in two short paragraphs:\n"
        "1. One sentence: you are a customer support triage agent."
        " You classify inbound messages and route them to the right"
        " specialist handler — bugs, billing questions, or feature"
        " requests.\n"
        "2. Three example queries the user could try, as a short"
        " bulleted list — one per non-greeting route:\n"
        '   - "the dashboard shows 500 errors when I open analytics"'
        "  (bug report)\n"
        '   - "how much does the Pro plan cost?"  (billing)\n'
        '   - "could you add dark mode to the dashboard?"'
        "  (feature request)\n\n"
        "Under 100 words total. Don't explain the internal graph;"
        " just what the user can ask."
    ),
    output_key="last_response",
)


bug_handler = Agent(
    name="bug_handler",
    # us-central1 + Pro 2.5 (W9.2 — 3.x preview gated per audit 2026-05-01).
    model="gemini-2.5-pro",
    description=(
        "Handles bug reports — captures repro steps, severity, and"
        " recent changes."
    ),
    instruction=(
        'The user reported a bug: "{request}"\n\n'
        "Respond as a frontline support engineer. Your job is to"
        " gather enough information for an engineer to investigate.\n\n"
        "1. Acknowledge the issue in one sentence.\n"
        "2. Ask for the missing pieces among: exact reproduction"
        " steps, browser / OS / app version, when it started, whether"
        " a recent change correlates, severity (blocker / major /"
        " minor).\n"
        "3. Set expectations: a ticket has been logged and an engineer"
        " will follow up.\n\n"
        "Tone: calm, specific, no fluff. Under 120 words. No markdown"
        " headers; plain prose with at most one short bulleted list"
        " for the missing-info questions."
    ),
    output_key="last_response",
)


billing_handler = Agent(
    name="billing_handler",
    # us-central1 + Pro 2.5 (W9.2 — 3.x preview gated per audit 2026-05-01).
    model="gemini-2.5-pro",
    description=(
        "Handles billing and pricing questions for the (mock) product"
        " plans."
    ),
    instruction=(
        'The user asked about billing: "{request}"\n\n'
        "Respond as a billing specialist. Use these (illustrative)"
        " plan facts and answer the user's question with whichever"
        " ones are relevant — do not dump the whole table:\n"
        "- Free: $0/month — 1 user, community support, 1 GB storage.\n"
        "- Pro: $29/user/month — unlimited projects, email support,"
        " 100 GB storage, SSO.\n"
        "- Enterprise: custom pricing — SAML SSO, audit logs, SLA,"
        " dedicated CSM.\n"
        "- Annual billing: 20% discount on Pro and Enterprise.\n\n"
        "If the user's question is about an existing charge or"
        " invoice (not pricing), say a billing specialist will follow"
        " up by email and ask them to confirm the email on file.\n\n"
        "Under 120 words. Plain prose, no markdown headers."
    ),
    output_key="last_response",
)


feature_handler = Agent(
    name="feature_handler",
    # us-central1 + Pro 2.5 (W9.2 — 3.x preview gated per audit 2026-05-01).
    model="gemini-2.5-pro",
    description=(
        "Handles feature requests — captures use case, logs to the"
        " product backlog, sets expectations."
    ),
    instruction=(
        'The user proposed a feature: "{request}"\n\n'
        "Respond as a product-feedback contact. Your job is to capture"
        " the request well, not to commit to building it.\n\n"
        "1. Thank the user for the suggestion in one sentence.\n"
        "2. Ask one clarifying question about the underlying use case"
        " — what problem is this solving for them, who else on their"
        " team would use it, what they do today instead.\n"
        "3. Set expectations honestly: the request will be logged in"
        " the product backlog and reviewed at the next triage. No"
        " timeline commitments.\n\n"
        "Under 120 words. Warm but professional tone. Plain prose."
    ),
    output_key="last_response",
)


# ---------------------------------------------------------------------------
# The Workflow — explicit graph, four legitimate terminal routes.
# ---------------------------------------------------------------------------


root_agent = Workflow(
    name="level_2b_agent",
    description=(
        "Customer support triage: classify an inbound message and"
        " route to a specialist handler (bug / billing / feature) or"
        " a greeting handler. Demonstrates the v2 graph routing"
        " primitive in isolation — same capability tier as Level 2,"
        " minimal shape (no fan-out, no synthesis)."
    ),
    edges=[
        # Capture user input → classify → route on the classifier's
        # decision. Sequence-shorthand tuple expands to three edges.
        ("START", process_input, classify, route_input),
        # Four terminal routes — Pydantic's Literal constraint
        # guarantees `route_input` always emits one of these four
        # strings, so no `__DEFAULT__` fallback is required.
        (
            route_input,
            {
                "GREETING": greet_user,
                "BUG": bug_handler,
                "BILLING": billing_handler,
                "FEATURE": feature_handler,
            },
        ),
    ],
)
