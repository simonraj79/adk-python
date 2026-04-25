"""Level 3 — Collaborative Multi-Agent System.

Coordinator delegates to three specialists via ADK's `AgentTool` pattern,
with an explicit `PlanReActPlanner` to make the delegation reasoning
visible. Data flows between specialists through ADK-native mechanisms:
`output_key` (auto-writes a specialist's final text to session state) and
`{state_key}` instruction injection (auto-reads it).

Why `AgentTool` and not `sub_agents`?

Google's taxonomy for Level 3 describes the new capability as "delegation
to sub-agents". The ADK feature map (`AGENTS.md:267`) exposes two
primitives for multi-agent systems:

  - `sub_agents=[...]` — conversation TRANSFER semantics. The user's
    conversation moves to the specialist; the specialist can transfer
    back to the parent or to peers. Good for escalation patterns
    ("general support → billing specialist").
  - `AgentTool(agent=X)` — coordinator CALLS the specialist as a typed
    tool, gets results back, stays in control. Good for delegation
    patterns like a research coordinator calling search/analyst/writer.

Both are "multi-agent" and both satisfy the taxonomy. AgentTool is the
better match for THIS coordinator because we want the coordinator to
orchestrate multiple specialists in sequence (search → analyst →
writer) and assemble a brief, not hand over the conversation to one of
them. `level_4_agent` and `level_4a_agent` use the same AgentTool
pattern — so L3 → L4 becomes "same delegation foundation, L4 adds
meta-reasoning + dynamic agent creation" rather than "L3 uses one API
primitive, L4 uses a different one for no particular reason".

Bonus: `AgentTool`-wrapped agents don't get `transfer_to_agent`
auto-injected into their requests (unlike `sub_agents`), so `search_agent`
can use the plain `google_search` built-in without the `GoogleSearchAgentTool`
nested-wrap workaround that AGENTS.md gotcha #24 describes. Cleaner,
fewer layers, lower latency.

Reasoning mode: `PlanReActPlanner()` on the coordinator. This is a
PROMPT-level ReAct scaffold (not Gemini's native thinking) that forces
the coordinator to structure its output as explicit PLANNING / REASONING
/ ACTION / FINAL_ANSWER sections. The plan text is visible in the chat
stream, making the "which specialist do I call next?" decision
transparent to the learner. Leaf specialists stay unplanned — their
jobs (search, save, format) are mechanical and don't benefit from
plan-then-act scaffolding.
"""

from google.adk.agents import Agent
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
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
# Specialist agents — each owns a distinct, non-overlapping role. Wrapped as
# `AgentTool`s by the coordinator (see root_agent.tools below).
# ---------------------------------------------------------------------------

search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches the web for information on a given query.",
    instruction=(
        "You are a web search specialist. Use google_search to answer the"
        " query you receive. Return the facts and cite the source URLs."
    ),
    # Plain `google_search` built-in works here because this agent is
    # wrapped as an `AgentTool` by the coordinator (not a `sub_agent`), so
    # ADK does NOT auto-inject `transfer_to_agent` into its requests.
    # Without the auto-injection, there's no built-in + function_calling
    # conflict with Gemini — see AGENTS.md gotcha #24 for the full
    # situation matrix.
    tools=[google_search],
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
# Coordinator — orchestrates the three specialists via AgentTool calls,
# with PlanReActPlanner making the delegation reasoning visible.
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="level_3_agent",
    model="gemini-2.5-flash",
    description=(
        "Research coordinator. Delegates to search, analyst, and writer"
        " specialists."
    ),
    # PlanReActPlanner forces an explicit Plan → Reason → Act → Final
    # scaffold. The coordinator must write out its plan (which specialist
    # to call next, in what order) before calling tools. Makes the
    # multi-agent orchestration visible in the chat stream — learners can
    # see the coordinator's reasoning rather than just the end result.
    #
    # Why PlanReAct instead of BuiltInPlanner (native thinking)?
    # BuiltInPlanner surfaces Gemini's internal thoughts as separate
    # THOUGHT events in the trace panel. PlanReActPlanner injects the
    # structure directly into the main response text, so the plan appears
    # in the chat bubble itself. For a coordinator whose job IS planning,
    # having the plan visible inline teaches the pattern better. Leaf
    # specialists stay unplanned.
    planner=PlanReActPlanner(),
    instruction=(
        "You are a research coordinator with three specialists:"
        " search_agent (web search via google_search), analyst_agent"
        " (saves findings to a shared scratchpad and reviews them for"
        " patterns/gaps), and writer_agent (formats accumulated notes"
        " into a structured brief).\n\n"
        "GREETINGS, META QUESTIONS, AND EMPTY OR UNCLEAR INPUT (e.g."
        " 'hi', 'hello', 'what can you do?'): respond directly yourself —"
        " do NOT delegate. Reply with exactly two sentences: one sentence"
        " describing the team, and one question inviting a research topic."
        " Example: \"I'm a research coordinator — I delegate to a search"
        " agent, analyst, and writer to produce structured briefs. What"
        " complex question would you like me to research?\"\n\n"
        "FOR EVERY OTHER TASK: call the specialist whose description"
        " matches (as a tool). Do not do the work yourself. Complex"
        " question → one or more search_agent → analyst_agent(save)"
        " cycles, then analyst_agent(review), then writer_agent. Simple"
        " factual question → one search_agent call → direct answer."
    ),
    tools=[
        AgentTool(agent=search_agent),
        AgentTool(agent=analyst_agent),
        AgentTool(agent=writer_agent),
    ],
)
