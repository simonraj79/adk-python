"""Safety allowlist for dynamically created specialists — Level 4a.

Extends the Level 4 allowlist to include MCP tools from gahmen-mcp. MCP tools
are stored as a sentinel string because a single `McpToolset` cannot be split
into per-name `BaseTool` objects at spec-validation time (discovery is async).
`resolve_tools` turns requested MCP names into a fresh, narrowed `McpToolset`
per runtime specialist so each specialist only sees the tools it asked for.

Plain tools (`google_search`) resolve to their BaseTool object directly.
"""

from __future__ import annotations

import re
from typing import Any

from google.adk.tools.google_search_tool import google_search
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

from .mcp_toolset import MCP_TOOL_NAMES
from .mcp_toolset import _make_stdio_params


# Sentinel marking a tool name as MCP-backed. resolve_tools() swaps
# occurrences of this sentinel for a narrowed McpToolset.
_MCP_SENTINEL = "__mcp_gahmen__"


# Short name → ADK tool object or sentinel. Only these are offered to
# runtime-created specialists. Every entry has known, audited behavior.
ALLOWED_TOOLS: dict[str, Any] = {
    "google_search": google_search,
    **{name: _MCP_SENTINEL for name in MCP_TOOL_NAMES},
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
  """Map validated tool names to ADK tool/toolset objects.

  Plain tools resolve to their BaseTool object. MCP tool names are grouped
  into a single per-specialist `McpToolset` narrowed via `tool_filter` so
  each runtime specialist only has access to the MCP tools it declared —
  the allowlist is enforced at the ADK boundary, not just at spec time.
  """
  non_mcp: list[Any] = []
  mcp_names: list[str] = []
  for name in tool_set:
    resolved = ALLOWED_TOOLS[name]
    if resolved is _MCP_SENTINEL:
      mcp_names.append(name)
    else:
      non_mcp.append(resolved)

  if not mcp_names:
    return non_mcp

  # A fresh McpToolset per runtime specialist, narrowed to only the MCP
  # tools the spec requested. Same stdio connection params as the fixed
  # team's toolset; the extra subprocess cost is acceptable because the
  # server rate-limits to ~5 req/min regardless.
  per_spec_mcp = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=_make_stdio_params(),
          timeout=30.0,
      ),
      tool_filter=mcp_names,
  )
  return non_mcp + [per_spec_mcp]
