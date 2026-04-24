from google.adk.agents import Agent
from google.adk.tools.google_search_tool import google_search

# Reasoning mode: NONE by design. Level 1 is a one-shot "search and answer"
# agent — no planning, no multi-hop synthesis — so an explicit reasoning
# scaffold (`planner=BuiltInPlanner(...)` / `PlanReActPlanner()`) would add
# cost without quality gain. Explicit reasoning is introduced in Level 2+
# as the taxonomy's next capability jump.
root_agent = Agent(
    name="level_1_agent",
    model="gemini-2.5-flash",
    description=(
        "A connected problem-solver that uses Google Search to answer "
        "questions requiring real-time information."
    ),
    instruction="""You answer questions using Google Search. One search per question.
Cite your sources. If the question is ambiguous, ask for clarification first.
""",
    tools=[google_search],
)
