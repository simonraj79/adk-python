"""Safety allowlist for dynamically created specialists.

Level 4 spawns new agents at runtime. Without a safety gate, the agent_creator
could assemble agents that do anything — arbitrary HTTP calls, local code
execution, credentials exfiltration. This module is the non-negotiable gate:

- `ALLOWED_TOOLS` maps a short tool name to the actual ADK tool object.
  Anything not in this mapping is rejected.
- `validate_spec` is the single entry point the creator tool must call before
  building an agent.

If you need to expand the allowlist, add the tool here deliberately. Do NOT
bypass this module from creator code.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.tools.google_search_tool import google_search


# Short name → ADK tool object. Only these are offered to runtime-created
# specialists. Every entry is a leaf tool with known, audited behavior.
ALLOWED_TOOLS: dict[str, Any] = {
    "google_search": google_search,
}


MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 500
MAX_INSTRUCTION_LEN = 4000
_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SpecValidationError(ValueError):
  """Raised when a dynamic specialist spec fails safety validation."""


def validate_spec(
    name: str,
    description: str,
    instruction: str,
    tool_set: list[str],
) -> None:
  """Validate a runtime specialist spec. Raises SpecValidationError on failure.

  Rules:
  - `name` must be snake_case, start with a letter, max 64 chars.
  - `description` must be a non-empty string under 500 chars.
  - `instruction` must be a non-empty string under 4000 chars.
  - `tool_set` must be a list of short names, each present in ALLOWED_TOOLS.
    Empty list is allowed (pure LLM specialist with no tools).
  """
  if not isinstance(name, str) or not _VALID_NAME_RE.match(name):
    raise SpecValidationError(
        f"Invalid name {name!r}: must be snake_case starting with a letter."
    )
  if len(name) > MAX_NAME_LEN:
    raise SpecValidationError(
        f"name too long ({len(name)} > {MAX_NAME_LEN})."
    )
  if not isinstance(description, str) or not description.strip():
    raise SpecValidationError("description must be a non-empty string.")
  if len(description) > MAX_DESCRIPTION_LEN:
    raise SpecValidationError(
        f"description too long ({len(description)} > {MAX_DESCRIPTION_LEN})."
    )
  if not isinstance(instruction, str) or not instruction.strip():
    raise SpecValidationError("instruction must be a non-empty string.")
  if len(instruction) > MAX_INSTRUCTION_LEN:
    raise SpecValidationError(
        f"instruction too long ({len(instruction)} > {MAX_INSTRUCTION_LEN})."
    )
  if not isinstance(tool_set, list):
    raise SpecValidationError("tool_set must be a list of tool names.")
  for tool_name in tool_set:
    if tool_name not in ALLOWED_TOOLS:
      raise SpecValidationError(
          f"Tool {tool_name!r} is not in the allowlist. "
          f"Allowed: {sorted(ALLOWED_TOOLS)}."
      )


def resolve_tools(tool_set: list[str]) -> list[Any]:
  """Map short tool names to actual ADK tool objects. Assumes pre-validated."""
  return [ALLOWED_TOOLS[name] for name in tool_set]
