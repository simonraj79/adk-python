"""Safety allowlist for dynamically created specialists.

Level 4 spawns new agents at runtime. Without a safety gate, the
agent_creator could assemble agents that do anything — arbitrary HTTP
calls, local code execution, credentials exfiltration. This module is
the non-negotiable gate:

- `ALLOWED_TOOLS` maps a short tool name to the actual ADK tool object.
  Anything not in this mapping is rejected.
- `validate_spec` is the single entry point the creator tool MUST call
  before building an agent.

If you need to expand the allowlist, add the tool here deliberately. Do
NOT bypass this module from creator code. (Same module as v1; safety
contracts don't need v2-specific changes.)
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.load_web_page import load_web_page

from .tools import calculator


def get_current_date() -> str:
  """Return today's date as an ISO string (YYYY-MM-DD).

  Useful for runtime specialists that need to anchor a search to "now"
  (e.g., "current F1 season schedule", "this week's earnings calls").
  """
  return datetime.date.today().isoformat()


# Short name → ADK tool object. Only these are offered to runtime-
# created specialists. Every entry is a leaf tool with known, audited
# behaviour. The catalog is intentionally small: each addition expands
# the action surface a runtime LLM-built specialist can take, so each
# tool here was chosen for general utility while resisting abuse.
#
# `google_search` uses `bypass_multi_tools_limit=True` so that when a
# runtime specialist declares multiple tools (e.g.,
# `["google_search", "get_current_date"]`), ADK's auto-swap path
# (`src/google/adk/agents/llm_agent.py:151-157`) replaces the
# `GoogleSearchTool` built-in with `GoogleSearchAgentTool` (a function-
# tool wrapper that runs search via a sub-agent). This avoids Gemini's
# "Built-in tools and Function Calling cannot be combined" error when
# the specialist mixes the built-in with custom function tools.
#
# The bypass flag is safe for single-tool specialists too: the auto-swap
# only fires when `len(self.tools) > 1`, so a runtime spec with just
# `["google_search"]` keeps the native built-in behaviour.
ALLOWED_TOOLS: dict[str, Any] = {
    # Web search — Gemini built-in, auto-swaps to GoogleSearchAgentTool
    # in multi-tool specs. See bypass note above.
    "google_search": GoogleSearchTool(bypass_multi_tools_limit=True),
    # Today's date — anchors searches to "now" for time-sensitive
    # questions ("current season", "this week's earnings", etc).
    "get_current_date": get_current_date,
    # Math expression evaluator — safe (AST-based, whitelisted ops),
    # NOT a code-execution surface. For full Python execution see the
    # fixed `analyst_agent`'s `BuiltInCodeExecutor`. See `tools.py` for
    # the complete operator/function whitelist.
    "calculator": calculator,
    # Fetch the text content of a URL — function tool from
    # `google.adk.tools.load_web_page`. SSRF-resistant
    # (`allow_redirects=False`). Use when a runtime specialist needs
    # full text of a specific page rather than a search summary.
    "load_web_page": load_web_page,
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
  - `tool_set` must be a list of short names, each present in
    ALLOWED_TOOLS. Empty list is allowed (pure LLM specialist with no
    tools).
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
          f"Tool {tool_name!r} is not in the allowlist."
          f" Allowed: {sorted(ALLOWED_TOOLS)}."
      )


def resolve_tools(tool_set: list[str]) -> list[Any]:
  """Map short tool names to actual ADK tool objects. Assumes pre-validated."""
  return [ALLOWED_TOOLS[name] for name in tool_set]
