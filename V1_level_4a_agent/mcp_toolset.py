"""MCP wiring for Level 4a — gahmen-mcp (Singapore Government Data).

Connects to the MCP server via local stdio. See
`New Agents/LEVEL_4A_MCP_PLAN.md` §4 for why hosted Streamable HTTP is
deliberately out of scope (no Smithery API key available).

The upstream `gahmen-mcp` server exports a `createStatelessServer` factory
for Smithery's HTTP runtime; it has no stdio entrypoint of its own. We
vendored the repo under `vendor/gahmen-mcp/` and added a 15-line stdio
wrapper at `vendor/gahmen-mcp/src/stdio_entry.ts` that adapts the factory
to a `StdioServerTransport`. This module launches that wrapper via `npx tsx`.
"""

from __future__ import annotations

from pathlib import Path

from mcp import StdioServerParameters
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams


# Absolute path to the vendored MCP server. Resolved once at import so a
# change of working directory (e.g. ADK web server spawning subprocesses
# from its own cwd) does not break launch.
_VENDOR_DIR = (Path(__file__).parent / "vendor" / "gahmen-mcp").resolve()


# Tool names the server exposes, narrowed to read-only data access. The
# download-orchestration tools (`datagovsg_initiate_download`,
# `datagovsg_poll_download`) are intentionally excluded — they require a
# multi-turn poll loop that muddies the L4 pattern for the demo.
MCP_TOOL_NAMES: tuple[str, ...] = (
    "datagovsg_list_collections",
    "datagovsg_get_collection",
    "datagovsg_list_datasets",
    "datagovsg_get_dataset_metadata",
    "datagovsg_search_dataset",
    "singstat_search_resources",
    "singstat_get_metadata",
    "singstat_get_table_data",
)


def _make_stdio_params() -> StdioServerParameters:
  """Build the stdio launch parameters for the vendored gahmen-mcp server."""
  return StdioServerParameters(
      command="npx",
      args=["tsx", "src/stdio_entry.ts"],
      cwd=str(_VENDOR_DIR),
  )


# Single module-level instance: one MCP subprocess per ADK process, reused
# across every specialist that needs gahmen tools. McpToolset owns the
# session lifecycle; the agent framework closes it on runner shutdown.
# No tool_name_prefix: the server already namespaces tools with
# `datagovsg_` / `singstat_`; another prefix would double-prefix them.
gahmen_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=_make_stdio_params(),
        # tsx cold-start can take 2-5s on Windows. Give it headroom.
        timeout=30.0,
    ),
    tool_filter=list(MCP_TOOL_NAMES),
)
