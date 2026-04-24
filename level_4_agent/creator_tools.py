"""Creator glue — the `create_specialist` tool used by `agent_creator`.

This is the single place where a runtime specialist is brought into existence.
The flow: validate → optionally smoke-test → persist spec → register in state.

The YAML path described in the plan is kept as an optional side-effect (we
write a YAML next to the spec for audit/manual replay), but the live agent is
built directly in Python via registry.build_agent_from_spec. This avoids the
@experimental `config_agent_utils.from_config` dependency and keeps the demo
self-contained.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import yaml

from google.adk.tools.tool_context import ToolContext

from .registry import add_capability
from .registry import build_agent_from_spec
from .registry import has_capability
from .safety import SpecValidationError
from .safety import validate_spec


RUNTIME_DIR = Path(__file__).parent / "runtime_agents"


def _write_yaml_audit_copy(spec: dict) -> str:
  """Write a YAML copy of the spec under runtime_agents/ for audit.

  Returns the absolute path. This is an audit trail, not the source of truth;
  the live agent is rebuilt from state.capabilities each turn, not from YAML.
  """
  RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
  yaml_path = RUNTIME_DIR / f"{spec['name']}.yaml"
  yaml_doc = {
      "name": spec["name"],
      "agent_class": "LlmAgent",
      "model": spec.get("model", "gemini-2.5-flash"),
      "description": spec["description"],
      "instruction": spec["instruction"],
      # Tools in YAML need full module paths; we stash the short names in a
      # comment-like custom key. This YAML is for audit; live hydration uses
      # registry.build_agent_from_spec, which reads tool_set from state.
      "tool_set": spec.get("tool_set", []),
  }
  with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(yaml_doc, f, sort_keys=False)
  return str(yaml_path.resolve())


def _smoke_test(spec: dict) -> tuple[bool, str]:
  """Best-effort smoke test — build the agent and confirm construction.

  A full runtime smoke test (actually invoking the new agent against a canned
  query) requires an async Runner and is wired in a later pass. For now we
  treat "construction succeeded" as the pass criterion. This still catches
  most failure modes: bad tool names, invalid instruction, missing fields.
  """
  try:
    build_agent_from_spec(spec)
    return True, "Agent construction succeeded."
  except Exception as exc:  # pylint: disable=broad-except
    return False, f"Construction failed: {exc!r}"


def create_specialist(
    name: str,
    description: str,
    instruction: str,
    tool_set: list[str],
    tool_context: ToolContext,
) -> str:
  """Create a new specialist agent and add it to the coordinator's team.

  Use when the user asks for a capability that no existing specialist covers.
  The new agent will be available for delegation starting on the NEXT user
  turn (it takes effect when the coordinator's `before_agent_callback` runs).

  Args:
    name: snake_case identifier, e.g. "f1_data_agent".
    description: one-sentence role summary. The coordinator uses this string
      for routing decisions — be specific about capabilities and inputs.
    instruction: the system prompt the new specialist will follow.
    tool_set: list of short tool names from the safety allowlist. Pass `[]`
      for a pure LLM specialist with no tools.

  Returns:
    A confirmation string describing success or failure. On failure, nothing
    is persisted to state and no file is written.
  """
  # 1. Validate against the safety allowlist. Raises on any rule violation.
  try:
    validate_spec(name, description, instruction, tool_set)
  except SpecValidationError as exc:
    return f"REJECTED: {exc}"

  # 2. Dedupe. A well-meaning coordinator may try to recreate on every turn.
  if has_capability(tool_context.state, name):
    return (
        f"Specialist {name!r} already exists in this session. No changes made."
    )

  spec = {
      "name": name,
      "description": description,
      "instruction": instruction,
      "tool_set": tool_set,
      "model": "gemini-2.5-flash",
      "created_at": datetime.datetime.utcnow().isoformat() + "Z",
  }

  # 3. Smoke test before we commit.
  passed, test_message = _smoke_test(spec)
  spec["smoke_test_passed"] = passed
  if not passed:
    return f"REJECTED: smoke test failed. {test_message}"

  # 4. Persist to session state (survives across turns).
  add_capability(tool_context.state, spec)

  # 5. Audit trail: write a YAML copy. Non-fatal if it fails.
  try:
    yaml_path = _write_yaml_audit_copy(spec)
    audit_note = f" Audit copy: {os.path.relpath(yaml_path)}."
  except Exception:  # pylint: disable=broad-except
    audit_note = " (audit copy skipped)"

  return (
      f"Specialist {name!r} created and registered. "
      f"It will be available as an AgentTool on the next turn.{audit_note} "
      f"Description: {description}"
  )
