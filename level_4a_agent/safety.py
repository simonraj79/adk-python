"""Safety allowlist for dynamically created specialists — Level 4a.

Extends the Level 4 allowlist with the gahmen-mcp tools (Singapore
Government Data via data.gov.sg + SingStat). MCP tools are stored as a
sentinel string in the allowlist because a single `McpToolset` cannot
be split into per-name `BaseTool` objects at spec-validation time
(MCP tool discovery is async). `resolve_tools` swaps the sentinels for
a fresh, narrowed `McpToolset` per runtime specialist so each
specialist only sees the MCP tools it asked for — the allowlist is
enforced at the ADK boundary, not just at spec time.

Native v2 features in this module:
  - `McpToolset` + `StdioConnectionParams` (v2 MCP primitives).
  - `tool_filter` enforces per-specialist tool narrowing at the
    framework layer.
  - Plain function tools (`calculator`, `get_current_date`,
    `load_web_page`) and the auto-swap-friendly
    `GoogleSearchTool(bypass=True)` from L4 are unchanged.

Plain (non-MCP) tools resolve to their tool object directly.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.load_web_page import load_web_page
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

from .mcp_toolset import _make_stdio_params
from .mcp_toolset import MCP_TOOL_NAMES
from .tools import calculator


def get_current_date() -> str:
  """Return today's date as an ISO string (YYYY-MM-DD).

  Useful for runtime specialists that need to anchor a search to "now"
  (e.g., "current F1 season schedule", "this week's earnings calls").
  """
  return datetime.date.today().isoformat()


# Sentinel marking a tool name as MCP-backed. `resolve_tools()` swaps
# every sentinel for a fresh McpToolset narrowed to the specific MCP
# tools the runtime spec asked for.
_MCP_SENTINEL = "__mcp_gahmen__"


# Short name → ADK tool object or sentinel. Only these are offered to
# runtime-created specialists. Every entry is a leaf tool with known,
# audited behaviour. The catalog inherits the L4 four-tool baseline and
# adds the eight read-only gahmen-mcp tools.
ALLOWED_TOOLS: dict[str, Any] = {
    # — Inherited from Level 4 ————————————————————————————————
    "google_search": GoogleSearchTool(bypass_multi_tools_limit=True),
    "get_current_date": get_current_date,
    "calculator": calculator,
    "load_web_page": load_web_page,
    # — Added in Level 4a: gahmen-mcp tools (Singapore Government) ——
    # Each sentinel is swapped for a per-spec narrowed McpToolset by
    # `resolve_tools()`. Naming matches the upstream server exactly —
    # no double-prefix, no rename.
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

  Same rules as Level 4 — `name` snake_case, `description` /
  `instruction` length-bounded, `tool_set` items must each be in
  `ALLOWED_TOOLS`. Empty `tool_set` is allowed (pure-LLM specialist).
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
  """Map validated tool names to ADK tool / toolset objects.

  Plain tools resolve to their existing object. MCP tool names are
  grouped into a single per-specialist `McpToolset` narrowed via
  `tool_filter`, so each runtime specialist only has access to the MCP
  tools it asked for. The extra subprocess cost (one McpToolset per
  spec) is acceptable because the upstream server rate-limits to a few
  requests per minute regardless.
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

  per_spec_mcp = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=_make_stdio_params(),
          timeout=30.0,
      ),
      tool_filter=mcp_names,
  )
  return non_mcp + [per_spec_mcp]
