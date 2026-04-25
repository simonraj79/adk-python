"""Level 3b — Dual-Mode Coordinator: Travel Planner (ADK 2.0).

A pared-down sibling to `level_3_agent`. Same capability tier (Level 3
— Collaborative Multi-Agent System), but the new axis is the
**`mode='task'` peer specialist**: one sub-agent runs autonomously
(`single_turn`) and a second one is allowed to **clarify with the
user** mid-task. Our canonical L3 only uses `single_turn`; only L4's
`agent_creator` uses `mode='task'`. L3b fills the gap by demonstrating
both modes side-by-side on peer specialists under one coordinator.

Use case
--------
Travel planning. The coordinator delegates to:
  - `weather_checker` — `mode='single_turn'`. Autonomous, one shot.
    The coordinator passes a city, the checker returns a structured
    forecast, control auto-returns to the coordinator. No user
    interaction.
  - `flight_booker` — `mode='task'`. Multi-turn — can ask the user
    clarifying questions. The coordinator passes origin + destination
    only (no date); the booker MUST ask the user for the date before
    it can call `search_flights`. After booking, it returns a
    structured confirmation via `finish_task`.

Inspired by the `travel_planner` demo in the *ADK 2.0 launch* video
("the moment the task completes, it will again auto return"). Same
shape as the canonical mixed-mode example in
`.agents/skills/adk-workflow/references/task-mode.md:188-232`.

Why this is `level_3b_agent` and not a new tier
-----------------------------------------------
Same capability tier as L3 (delegation to specialists). The new axis
is which **mode** each peer specialist uses, not what the team does.
Variant naming convention matches L4a (MCP variant of L4) and L1a
(voice variant of L1). New ADK 2.0 *primitives* are absorbed as `Na`/
`Nb`/`Nc` variants of the closest capability tier; they don't inflate
the L0–L4 ladder.

Key design choice — `date` deliberately omitted from FlightBookingInput
----------------------------------------------------------------------
`FlightBookingInput` exposes only `origin` and `destination`. The date
is NOT in the schema. Why: if it were, the coordinator's LLM might
guess one (or the user-facing instruction would have to fight that
tendency). Omitting it forces the booker to **have nothing to call
`search_flights` with** until it asks the user. That's the
pedagogical hook for `mode='task'`: the framework gives the agent a
legitimate reason to interrupt and ask. Same trick the video uses
("What is the exact date?").

The mock `search_flights` tool DOES accept `date` — that's where the
booker uses the user's answer. Schema and tool signatures intentionally
diverge: schema = coordinator → booker contract; tool signature =
booker's internal capability surface.

Tool/mode compatibility — no Gemini built-in conflict here
----------------------------------------------------------
Both specialists use only **function tools** (`get_weather`,
`search_flights`, `book_flight`) — no built-ins like `google_search`.
That avoids gotcha #24 (the v1/v2 "Built-in tools and Function Calling
cannot be combined" issue) entirely. With `mode='task'`, the framework
also injects `FinishTaskTool`; combined with `[search_flights,
book_flight]` that's three function tools on `flight_booker`, which is
fine — the limit only triggers on built-in + function mixes.

`disallow_transfer_to_*` — same v2 hygiene as L3
------------------------------------------------
Both specialists set `disallow_transfer_to_parent=True` and
`disallow_transfer_to_peers=True`. The framework otherwise auto-injects
`transfer_to_agent` on any sub-agent whose disallow flags are False
(`agent_transfer.py:152-188`), and that injection causes the gotcha
#24 conflict in many configurations. `single_turn` and `task` agents
don't need transfer anyway — they auto-return via `request_task` /
`finish_task`. Same flags as L3.

Sample queries
--------------
- "hi" / "what can you do?"
    → coordinator handles inline; no delegation
- "what's the weather in Paris?"
    → coordinator: request_task_weather_checker(city="Paris") → done
- "book me a flight from SFO to CDG"
    → coordinator: request_task_flight_booker(origin="SFO",
       destination="CDG") → booker asks "what date?" → user replies
       → booker calls search_flights, presents options, books, returns
- "I'm going to Tokyo next Friday — check the weather and book me a
   flight from SFO"
    → coordinator chains both: weather_checker first (autonomous),
       then flight_booker (which asks for the exact date because it
       can't trust "next Friday" without confirmation).

Run
---
    adk web .
    # → pick `level_3b_agent` in the picker
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from pydantic import BaseModel
from pydantic import Field

from .tools import book_flight
from .tools import get_weather
from .tools import search_flights


# ---------------------------------------------------------------------------
# Pydantic schemas — the typed contracts at every coordinator/specialist
# boundary. Each request_task_<name> tool exposes the input_schema as
# the function signature; finish_task validates against output_schema.
# ---------------------------------------------------------------------------


class WeatherInput(BaseModel):
  """Argument schema for `request_task_weather_checker`."""

  city: str = Field(
      description=(
          "The city to look up weather for. Pass the city name as the"
          " user wrote it (e.g., 'Paris', 'San Francisco', 'Tokyo')."
      )
  )


class WeatherForecast(BaseModel):
  """Structured forecast returned from weather_checker."""

  city: str = Field(description="Echo of the requested city.")
  current_summary: str = Field(
      description=(
          "One-sentence summary of current conditions, including"
          " temperature in Celsius and a short description"
          " (e.g., '14°C, partly cloudy, 72% humidity')."
      )
  )
  outlook_summary: str = Field(
      description=(
          "2-3 sentence summary of the next 3 days' outlook, calling"
          " out any notable weather (rain, heatwave, etc.)."
      )
  )


class FlightBookingInput(BaseModel):
  """Argument schema for `request_task_flight_booker`.

  IMPORTANT: `date` is intentionally NOT in this schema. The booker
  must ask the user. See module docstring for rationale.
  """

  origin: str = Field(
      description=(
          "Departure city or IATA airport code (e.g., 'SFO' or"
          " 'San Francisco')."
      )
  )
  destination: str = Field(
      description=(
          "Arrival city or IATA airport code (e.g., 'CDG' or 'Paris')."
      )
  )


class FlightBooking(BaseModel):
  """Structured booking confirmation returned from flight_booker."""

  origin: str = Field(description="Echo of origin.")
  destination: str = Field(description="Echo of destination.")
  date_iso: str = Field(
      description="The departure date the user confirmed, in YYYY-MM-DD."
  )
  flight_id: str = Field(description="The booked flight's identifier.")
  airline: str = Field(description="The booking airline.")
  pnr: str = Field(description="The reservation reference / PNR.")
  price_usd: float = Field(description="Final price in USD.")
  summary: str = Field(
      description=(
          "One-sentence human-readable summary of the booking — what"
          " was booked, when, and the price. The coordinator will pass"
          " this through to the user."
      )
  )


# ---------------------------------------------------------------------------
# Specialists — one autonomous (single_turn), one interactive (task).
# ---------------------------------------------------------------------------


weather_checker = Agent(
    name="weather_checker",
    model="gemini-2.5-flash",
    description=(
        "Checks current weather and a 3-day outlook for a city."
        " Autonomous — does not interact with the user."
    ),
    # mode='single_turn': autonomous one-shot. Auto-injects
    # FinishTaskTool but no user-facing prompts.
    mode="single_turn",
    input_schema=WeatherInput,
    output_schema=WeatherForecast,
    instruction=(
        "Call `get_weather` with the city in the structured input."
        " The tool returns real data from Open-Meteo (no API key)."
        " Tool response shape:\n"
        "  - `resolved_location`: the city the geocoder matched (e.g.,"
        ' "Paris, Île-de-France, France"). Use this in your summary so'
        " the user can confirm the right place was found, especially"
        " for ambiguous names.\n"
        "  - `current`: {temp_c, conditions, humidity}\n"
        "  - `outlook`: list of up to 3 days, each with `date`,"
        " `conditions`, `temp_c_max`, `temp_c_min`.\n"
        "  - `error` (only on failure): pass this verbatim to the user"
        " and do NOT fabricate weather.\n\n"
        "On success, call `finish_task` with a WeatherForecast:\n"
        "  - `city`: echo the resolved_location (not the raw input).\n"
        "  - `current_summary`: one sentence with current conditions"
        " (e.g., '12°C, light rain, 78% humidity in Paris,"
        " Île-de-France, France').\n"
        "  - `outlook_summary`: 2-3 sentences covering the 3 outlook"
        " days. Use min/max ranges (e.g., 'tomorrow 8-15°C, partly"
        " cloudy').\n\n"
        "Do not ask the user anything — you are autonomous."
    ),
    tools=[get_weather],
    # Suppress transfer_to_agent injection (gotcha #24 hygiene). The
    # `single_turn` mode auto-returns the result via `request_task`
    # without needing transfer.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


flight_booker = Agent(
    name="flight_booker",
    model="gemini-2.5-flash",
    description=(
        "Books flights for the user. Will ask the user for clarifying"
        " details (such as departure date) when needed before searching."
    ),
    # mode='task': can chat with the user mid-task to clarify. Auto-
    # injects FinishTaskTool. Combined with [search_flights,
    # book_flight] that's 3 function tools — under the limit, no
    # built-ins → no gotcha #24 conflict.
    mode="task",
    input_schema=FlightBookingInput,
    output_schema=FlightBooking,
    instruction=(
        "You book flights for the user. The coordinator gave you an"
        " origin and a destination but NOT a date — you MUST ask the"
        " user for the departure date before calling `search_flights`."
        " Do not guess or invent a date.\n\n"
        "Lifecycle:\n"
        "  1. ASK the user for the departure date in plain natural"
        " language. Accept any reasonable form ('April 28th', 'next"
        " Friday', '2026-04-28') and resolve to ISO YYYY-MM-DD. If"
        " the user gives a relative date that's ambiguous, ask once"
        " for an exact date.\n"
        "  2. Call `search_flights(origin, destination, date)` with the"
        " ISO date and present the 3 options to the user concisely"
        " (airline, depart, arrive, stops, price). Ask which one to"
        " book — or pick the lowest-stops/lowest-price option if the"
        " user says 'cheapest' or 'fastest'.\n"
        "  3. Call `book_flight(flight_id)` once the user confirms.\n"
        "  4. Call `finish_task` with the FlightBooking summary. The"
        " `summary` field should be one human-readable sentence the"
        " coordinator can pass through verbatim, e.g.:\n"
        "       \"Booked Air France AF83, SFO → CDG on 2026-04-28,\n"
        "        nonstop, $925. PNR: ABCXYZ.\"\n\n"
        "Do not respond to the user with a final confirmation directly"
        " — the coordinator will. Your job ends at `finish_task`."
    ),
    tools=[search_flights, book_flight],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Coordinator — the only agent the user talks to. Delegates via the
# auto-generated `request_task_<name>` tools the framework wires from
# `sub_agents=[...]` (one per non-chat sub-agent).
# ---------------------------------------------------------------------------


root_agent = Agent(
    name="level_3b_agent",
    model="gemini-2.5-flash",
    description=(
        "Travel coordinator that delegates weather lookups to a"
        " single_turn specialist and flight booking to a task specialist"
        " that can ask clarifying questions. Routing decisions are made"
        " at runtime and visible inline via PlanReActPlanner."
    ),
    # PlanReActPlanner — same rationale as L3: the coordinator's
    # delegation order IS its reasoning, and surfacing it inline
    # teaches the multi-agent pattern.
    planner=PlanReActPlanner(),
    instruction=(
        "You are a travel coordinator with two specialists you can"
        " delegate to via the request_task_<name> tools the framework"
        " has wired in for you:\n"
        "  - request_task_weather_checker(city): autonomous weather"
        " lookup for a single city. Returns a structured forecast.\n"
        "  - request_task_flight_booker(origin, destination):"
        " interactive flight booking. The booker WILL ask the user"
        " for the departure date and which option to book — that's"
        " expected. Pass through their messages verbatim while the"
        " booker is working.\n\n"
        "GREETINGS, META QUESTIONS, AND EMPTY OR UNCLEAR INPUT (e.g."
        " 'hi', 'hello', 'what can you do?'): respond directly"
        " yourself — do NOT delegate. Reply with exactly two short"
        " paragraphs: one describing what you do, and one with two"
        " example queries.\n"
        "  Example:\n"
        '    "I help plan trips. I can check the weather for a'
        " destination and book flights — I'll ask you for any details"
        ' my booking specialist needs (like the exact departure date)."'
        "\n\n"
        "    Two examples to try:\n"
        '    - "What\'s the weather in Paris?"\n'
        '    - "Book me a flight from SFO to CDG."'
        "\n\n"
        "FOR WEATHER QUESTIONS: call request_task_weather_checker(city)"
        " once and return the forecast to the user in 1-2 sentences.\n\n"
        "FOR FLIGHT BOOKING: call request_task_flight_booker(origin,"
        " destination). DO NOT pass a date — the booker needs to ask"
        " the user. After it returns, give the user the FlightBooking's"
        " `summary` verbatim plus any extra context the user asked"
        " about.\n\n"
        "FOR COMBINED REQUESTS (weather AND book): chain the calls in"
        " the order that makes sense — usually weather first (fast,"
        " autonomous), then booking (interactive). Return both"
        " results to the user in one consolidated reply.\n\n"
        "Do not call the tools yourself by name (e.g. don't call"
        " `get_weather` directly) — only the request_task_<name>"
        " specialists. They own the tools."
    ),
    sub_agents=[weather_checker, flight_booker],
)
