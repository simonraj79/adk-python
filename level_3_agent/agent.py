"""Level 3 — Collaborative Multi-Agent System.

Coordinator delegates to three specialists via ADK's `sub_agents`. The
coordinator has no tools of its own. Data flows between specialists through
ADK-native mechanisms: `output_key` (auto-writes a sub-agent's final text to
session state) and `{state_key}` instruction injection (auto-reads it).
"""

from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.tool_context import ToolContext


# ---------------------------------------------------------------------------
# Tools — only legitimate business logic. Search and state-reads are handled
# by ADK built-ins (`google_search`) and instruction state injection.
# ---------------------------------------------------------------------------


def save_research_note(
    topic: str,
    finding: str,
    source: str,
    tool_context: ToolContext,
) -> str:
  """Append a research finding to the shared `research_notes` scratchpad.

  Args:
    topic: The sub-topic or sub-question this finding relates to.
    finding: The key insight or fact discovered.
    source: The URL or source name where this was found.

  Returns:
    Confirmation including the current note count.
  """
  notes = tool_context.state.get("research_notes", [])
  notes = notes + [{"topic": topic, "finding": finding, "source": source}]
  tool_context.state["research_notes"] = notes
  return f"Saved. Scratchpad now has {len(notes)} note(s)."


# ---------------------------------------------------------------------------
# Specialist sub-agents — each owns a distinct, non-overlapping role.
# ---------------------------------------------------------------------------

search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches the web for information on a given query.",
    instruction=(
        "You are a web search specialist. Use google_search to answer the"
        " query you receive. Return the facts and cite the source URLs."
    ),
    # bypass_multi_tools_limit=True is REQUIRED here because search_agent
    # is registered as a sub_agent of root_agent, which causes ADK to
    # auto-inject the `transfer_to_agent` function tool at request time
    # (see src/google/adk/flows/llm_flows/agent_transfer.py). Without
    # bypass, Gemini rejects the call with HTTP 400: "Built-in tools
    # ({google_search}) and Function Calling cannot be combined in the
    # same request." Same flag, same reason as level_2_agent's
    # GoogleSearchTool — there it mixes with explicit user-defined
    # function tools; here it mixes with the framework's auto-injected
    # transfer tool.
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    output_key="last_search_result",
)

analyst_agent = Agent(
    name="analyst_agent",
    model="gemini-2.5-flash",
    description=(
        "Saves findings to the scratchpad and reviews accumulated notes for"
        " patterns, contradictions, and gaps."
    ),
    instruction="""You manage the research scratchpad.

Most recent search result (from search_agent):
{last_search_result?}

All notes saved so far:
{research_notes?}

Two modes:
- SAVE: extract key findings from the latest search result and call
  save_research_note(topic, finding, source) once per finding.
- REVIEW: when asked to review, summarise patterns, contradictions, and gaps
  across all saved notes. Do not call save_research_note in this mode.
""",
    tools=[save_research_note],
)

writer_agent = Agent(
    name="writer_agent",
    model="gemini-2.5-flash",
    description="Formats accumulated research notes into a structured brief.",
    instruction="""Format the research notes below into a structured brief.

Notes:
{research_notes?}

Output format:
## Research Brief: [Topic]
### Key Findings
### Detailed Analysis
### Sources
### Confidence & Gaps
""",
)


# ---------------------------------------------------------------------------
# Coordinator — delegates only. No tools of its own.
#
# Reasoning mode: NONE on any of the four agents (coordinator + three
# specialists). No `planner` is configured anywhere. Gemini 2.5's native
# thinking runs implicitly on each LLM call but thoughts are not surfaced,
# and there is no framework-enforced plan-then-act scaffold.
#
# To make any specialist an explicit reasoning agent — e.g. analyst_agent in
# REVIEW mode, where synthesis benefits most — add a `planner=` kwarg:
#   from google.adk.planners.built_in_planner import BuiltInPlanner
#   from google.genai import types
#   analyst_agent = Agent(
#       ...,
#       planner=BuiltInPlanner(
#           thinking_config=types.ThinkingConfig(include_thoughts=True),
#       ),
#   )
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="level_3_agent",
    model="gemini-2.5-flash",
    description=(
        "Research coordinator. Delegates to search, analyst, and writer"
        " specialists."
    ),
    instruction=(
        "You are a research coordinator with three specialists:"
        " search_agent (web search via google_search), analyst_agent"
        " (saves findings to a shared scratchpad and reviews them for"
        " patterns/gaps), and writer_agent (formats accumulated notes"
        " into a structured brief).\n\n"
        "GREETINGS, META QUESTIONS, AND EMPTY OR UNCLEAR INPUT (e.g."
        " 'hi', 'hello', 'what can you do?', 'help'): respond directly"
        " yourself — do NOT delegate. Introduce the team in one sentence"
        " and give 2–3 concrete example questions the user can try, e.g."
        " 'Compare lab-grown vs natural diamonds — pricing, environmental"
        " impact, and resale value' or 'What are the latest advances in"
        " solid-state batteries?'. Then invite the user to ask anything"
        " research-shaped.\n\n"
        "FOR EVERY OTHER TASK: delegate to the specialist whose"
        " description matches. Do not do the work yourself. Complex"
        " question → one or more search → analyst(save) cycles, then"
        " analyst(review), then writer. Simple factual question → one"
        " search → direct answer."
    ),
    sub_agents=[search_agent, analyst_agent, writer_agent],
)
