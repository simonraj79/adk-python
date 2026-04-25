"""Capability registry — runtime specialist persistence across turns AND restarts.

Runtime-created specialists live in two places:
  1. `state.capabilities` — a list of plain dicts on the session state
     (in-memory, lost on server restart).
  2. `runtime_agents/<name>.yaml` — a YAML audit copy on disk
     (survives restarts).

On every turn, the coordinator's `before_agent_callback` calls
`hydrate_capabilities` to rebuild `AgentTool` instances from BOTH
sources, deduped by name (state wins on conflict — it's the freshest
version of any spec the current session has interacted with).

This means the `runtime_agents/` directory becomes a **persistent
capability library**: a specialist created in session A on Monday is
still callable from session B on Tuesday after `adk web` was killed
and restarted in between. Trade-off: cross-session isolation is
weaker — any session can see any disk-persisted spec. For a teaching
demo this is the intended behaviour; for multi-user production you'd
namespace `runtime_agents/<user_id>/`.

Why `AgentTool` for runtime specialists (not `sub_agents` + `mode`)
-------------------------------------------------------------------
v2's `sub_agents → _SingleTurnAgentTool` auto-derivation runs once at
`model_post_init` time (`src/google/adk/agents/llm_agent.py:982-994`),
not on every turn. Mutating `sub_agents` from the callback won't
trigger re-derivation. So runtime specialists go into `tools=[...]` via
the v1-style `AgentTool` wrap. Fixed (compile-time) specialists use the
v2 `sub_agents=[...]` + `mode='single_turn'` pattern; runtime
specialists use `AgentTool`. Both end up as named function tools the
coordinator's LLM can call — picked by lifetime (static vs dynamic),
not by preference.

Storage format (one entry per runtime specialist):

    {
        "name": "crm_data_agent",
        "description": "Pulls live Salesforce opportunity data.",
        "instruction": "You fetch and summarize CRM records...",
        "tool_set": ["google_search"],
        "created_at": "2026-04-25T14:12:00Z",
        "smoke_test_passed": True,
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool

from .safety import resolve_tools


STATE_KEY = "capabilities"

# On-disk YAML audit directory. Single source of truth across the
# package — `creator_tools.py` imports this constant for write-side
# operations.
RUNTIME_DIR = Path(__file__).parent / "runtime_agents"

logger = logging.getLogger(__name__)


def get_capabilities(state: Any) -> list[dict]:
  """Return the list of runtime capability specs from state."""
  return list(state.get(STATE_KEY, []))


def has_capability(state: Any, name: str) -> bool:
  """Check whether a capability with this name is already registered.

  Checks BOTH session state (in-memory) AND the on-disk YAML library.
  This is what makes restart persistence + dedupe work together: a
  fresh session asked to recreate a previously-persisted spec will
  see the disk copy via this check and short-circuit rather than
  silently overwriting the YAML.
  """
  if any(cap.get("name") == name for cap in get_capabilities(state)):
    return True
  return (RUNTIME_DIR / f"{name}.yaml").exists()


def add_capability(state: Any, spec: dict) -> None:
  """Append a spec to state.capabilities. Writes back the full list.

  Direct list mutation may be lost depending on state backend; writing
  back the full list is the safe pattern (same approach used by Level 2
  for its scratchpad pre-v2-rewrite).
  """
  caps = get_capabilities(state)
  caps.append(spec)
  state[STATE_KEY] = caps


def build_agent_from_spec(spec: dict) -> Agent:
  """Materialize a stored spec into a live `Agent` (LlmAgent).

  Note: no `mode` is set. Runtime specialists are wrapped via
  `AgentTool` (see `hydrate_capabilities` below), so they don't need
  v2's `mode='single_turn'` — `AgentTool` provides the call-and-return
  semantics directly.
  """
  return Agent(
      name=spec["name"],
      model=spec.get("model", "gemini-2.5-flash"),
      description=spec["description"],
      instruction=spec["instruction"],
      tools=resolve_tools(spec.get("tool_set", [])),
  )


def _load_yaml_specs_from_disk() -> list[dict]:
  """Scan `runtime_agents/` for YAML files and return parsed specs.

  YAMLs that fail to parse, or that reference tools no longer in the
  allowlist, are skipped (with a log line) rather than aborting the
  whole hydration — one bad file should not brick the coordinator.
  """
  if not RUNTIME_DIR.exists():
    return []
  specs: list[dict] = []
  for yaml_file in sorted(RUNTIME_DIR.glob("*.yaml")):
    try:
      with open(yaml_file, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
      specs.append({
          "name": doc["name"],
          "description": doc["description"],
          "instruction": doc["instruction"],
          "tool_set": doc.get("tool_set", []),
          "model": doc.get("model", "gemini-2.5-flash"),
      })
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning(
          "Failed to load runtime-agent YAML %s: %s", yaml_file, exc
      )
      continue
  return specs


def hydrate_capabilities(state: Any) -> list[AgentTool]:
  """Rebuild AgentTool wrappers for every stored capability.

  Reads from BOTH session state and the on-disk YAML library, deduped
  by name (state wins on conflict). Called from
  `before_agent_callback` on the coordinator; the returned tools are
  appended to the coordinator's existing tool list (see `agent.py`).
  """
  state_specs = get_capabilities(state)
  state_names = {s.get("name") for s in state_specs}

  # Disk-persisted specs fill in anything the current session hasn't
  # already created or refreshed. State-side specs always win (they're
  # the most recently validated version).
  disk_specs = _load_yaml_specs_from_disk()
  merged_specs: list[dict] = list(state_specs)
  for spec in disk_specs:
    if spec.get("name") not in state_names:
      merged_specs.append(spec)

  tools: list[AgentTool] = []
  for spec in merged_specs:
    try:
      agent = build_agent_from_spec(spec)
      tools.append(AgentTool(agent=agent))
    except Exception as exc:  # pylint: disable=broad-except
      # A bad spec should not brick the coordinator. Skip and continue;
      # the creator will re-attempt next time if the gap is still
      # present.
      logger.warning(
          "Skipping runtime spec %r — build failed: %s",
          spec.get("name"),
          exc,
      )
      continue
  return tools
