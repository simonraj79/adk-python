from google.adk.agents import Agent

# Reasoning mode: NONE by design. Level 0 is an LLM-in-isolation sanity
# check with a single greet tool — no multi-step reasoning is possible or
# needed. Adding a `planner=` here would be pure overhead.


def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The person's name to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}! Welcome!"


root_agent = Agent(
    name="my_first_agent",
    model="gemini-2.5-flash",
    instruction="You are a friendly assistant. Use the greet tool when a user gives their name.",
    description="A friendly greeter agent.",
    tools=[greet],
)
