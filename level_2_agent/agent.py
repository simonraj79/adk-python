from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.tool_context import ToolContext


def save_research_note(
    topic: str,
    finding: str,
    source: str,
    tool_context: ToolContext,
) -> str:
    """Save a research finding to the scratchpad for later synthesis.

    Use this after each search to capture key findings.

    Args:
        topic: The sub-topic or sub-question this finding relates to.
        finding: The key insight or fact discovered.
        source: The URL or source name where this was found.

    Returns:
        A confirmation that the note was saved.
    """
    if "research_notes" not in tool_context.state:
        tool_context.state["research_notes"] = []

    note = {
        "topic": topic,
        "finding": finding,
        "source": source,
    }
    tool_context.state["research_notes"] = (
        tool_context.state["research_notes"] + [note]
    )
    count = len(tool_context.state["research_notes"])
    return f"Note saved. You now have {count} research note(s) on your scratchpad."


def get_research_notes(tool_context: ToolContext) -> str:
    """Retrieve all saved research notes from the scratchpad.

    Call this when ready to review accumulated findings before synthesis.

    Returns:
        All research notes collected so far, formatted for review.
    """
    notes = tool_context.state.get("research_notes", [])
    if not notes:
        return "No research notes saved yet. Use save_research_note after each search."

    output_lines = []
    for i, note in enumerate(notes, 1):
        output_lines.append(
            f"[{i}] Topic: {note['topic']}\n"
            f"    Finding: {note['finding']}\n"
            f"    Source: {note['source']}"
        )
    return "\n\n".join(output_lines)


# Reasoning mode: NONE (no `planner` configured).
# The PLAN / EXECUTE / SYNTHESISE structure below is prompt-driven, not
# framework-enforced. Gemini 2.5's native thinking runs implicitly but
# thoughts are not surfaced. To turn this into an explicit ADK reasoning
# agent, add one of:
#   from google.adk.planners.built_in_planner import BuiltInPlanner
#   from google.genai import types
#   planner=BuiltInPlanner(
#       thinking_config=types.ThinkingConfig(include_thoughts=True),
#   )
# or for a ReAct-style prompt scaffold:
#   from google.adk.planners.plan_re_act_planner import PlanReActPlanner
#   planner=PlanReActPlanner()
root_agent = Agent(
    name="level_2_agent",
    model="gemini-2.5-flash",
    description=(
        "A strategic web researcher that decomposes complex questions, "
        "plans multi-step research, and delivers structured reports."
    ),
    instruction="""You are a strategic research agent. You plan before you search.

Workflow:
1. PLAN: Decompose the question into 2-3 sub-questions. Tell the user your plan.
2. EXECUTE: For each sub-question, call google_search then save_research_note. Max 2-3 searches total.
3. SYNTHESISE: Call get_research_notes, cross-reference findings, then write your report.

Output format — always use this structure:
## Research Brief: [Topic]
### Key Findings
### Detailed Analysis
### Sources
### Confidence & Gaps

Rules:
- Save a note after every search.
- Combine related aspects into broad queries to minimise search calls.
- Distinguish facts from opinions. Flag uncertainty.
- For simple factual questions, skip the full workflow and answer with a single search.
""",
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        save_research_note,
        get_research_notes,
    ],
)
