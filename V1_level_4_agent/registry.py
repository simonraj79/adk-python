"""Capability registry — Level 4 feature F6 (persistence across turns).

Runtime-created specialists live in `state.capabilities` as plain dicts, not
as live Agent objects. On every turn, the coordinator's `before_agent_callback`
calls `hydrate_capabilities` to re-construct AgentTool instances from the
stored specs. This survives across user turns within a session because the
SessionService persists `state` between invocations.

Storage format (one entry per runtime specialist):

    {
        "name": "crm_data_agent",
        "description": "Pulls live Salesforce opportunity data.",
        "instruction": "You fetch and summarize CRM records...",
        "tool_set": ["google_search"],
        "created_at": "2026-04-23T14:12:00Z",
        "smoke_test_passed": true,
    }
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from .safety import resolve_tools


STATE_KEY = "capabilities"


def get_capabilities(state: Any) -> list[dict]:
  """Return the list of runtime capability specs from state."""
  return list(state.get(STATE_KEY, []))


def has_capability(state: Any, name: str) -> bool:
  """Check whether a capability with this name is already registered."""
  return any(cap.get("name") == name for cap in get_capabilities(state))


def add_capability(state: Any, spec: dict) -> None:
  """Append a spec to state.capabilities. Writes back the full list.

  Direct list mutation may be lost depending on state backend; writing back
  the full list is the safe pattern (same approach used by Level 2/3 agents
  for their scratchpad).
  """
  caps = get_capabilities(state)
  caps.append(spec)
  state[STATE_KEY] = caps


def build_agent_from_spec(spec: dict) -> Agent:
  """Materialize a stored spec into a live LlmAgent."""
  return Agent(
      name=spec["name"],
      model=spec.get("model", "gemini-2.5-flash"),
      description=spec["description"],
      instruction=spec["instruction"],
      tools=resolve_tools(spec.get("tool_set", [])),
  )


def hydrate_capabilities(state: Any) -> list[AgentTool]:
  """Rebuild AgentTool wrappers for every stored capability.

  Called from `before_agent_callback` on the coordinator. Tools are appended
  to the coordinator's existing tool list (see agent.py).
  """
  tools: list[AgentTool] = []
  for spec in get_capabilities(state):
    try:
      agent = build_agent_from_spec(spec)
      tools.append(AgentTool(agent=agent))
    except Exception:  # pylint: disable=broad-except
      # A bad spec should not brick the coordinator. Skip and continue;
      # the creator will re-attempt next time if the gap is still present.
      continue
  return tools
