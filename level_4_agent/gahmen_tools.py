"""FunctionTool wrappers for gahmen-mcp (Singapore government data).

W9.3 (2026-05-01): replaces the McpToolset+StreamableHTTPConnectionParams
approach used in W9.2. Root cause of W9.2 failure: Smithery's hosted MCP
endpoint uses `?api_key=<key>` query-param auth, NOT
`Authorization: Bearer <key>` header. The McpToolset configured the
header-based auth and silently received 401s on its lazy-fetch
`tools/list` call, so the agent's tools_dict ended up empty — the LLM
then tried to call gahmen_* tools the framework couldn't dispatch.

This file uses the documented Smithery endpoint shape directly via
`httpx`, wrapped as ADK 2.0 `FunctionTool`s. Standard ADK primitive,
predictable lifecycle, no MCP session-manager drama. Verified
2026-05-01 with a `tools/list` probe that returned all 8 expected
tools.

Endpoint shape:
    POST https://server.smithery.ai/aniruddha-adhikary/gahmen-mcp/mcp?api_key=<KEY>
    Content-Type: application/json
    Accept: application/json, text/event-stream
    Body: JSON-RPC 2.0:
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "<tool>", "arguments": {...}}}
    Response: SSE-formatted "event: message\\ndata: {json}\\n\\n".

Tool naming: matches the gahmen_<original> naming the prior McpToolset
emitted (via `tool_name_prefix='gahmen'`), so the data_fetcher
instruction needs no edits.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

from google.adk.tools import FunctionTool


_SMITHERY_BASE = os.environ.get(
    "SMITHERY_GAHMEN_URL",
    "https://server.smithery.ai/aniruddha-adhikary/gahmen-mcp",
)
_API_KEY = os.environ.get("SMITHERY_API_KEY", "")


def _endpoint() -> str:
    return f"{_SMITHERY_BASE}/mcp?api_key={_API_KEY}"


def _parse_sse_response(body: str) -> dict | None:
    """Smithery returns JSON-RPC results as SSE: `event: message\\ndata: {...}\\n\\n`.

    Iterate lines, find the first `data: {...}` payload, return the
    parsed JSON. Returns None if no parseable payload found.
    """
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                continue
    return None


async def _call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Call a gahmen MCP tool via Smithery's JSON-RPC-over-HTTP endpoint.

    Returns the tool's text content, or an `[error] ...` string on
    failure (caller's instruction must mention failures explicitly).
    """
    if not _API_KEY:
        return "[error] SMITHERY_API_KEY not set in this deployment."

    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
            r = await client.post(
                _endpoint(),
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
    except Exception as exc:  # noqa: BLE001
        return f"[error] gahmen {tool_name} HTTP failure: {exc}"

    if r.status_code != 200:
        return (
            f"[error] gahmen {tool_name} returned {r.status_code}: "
            f"{r.text[:300]}"
        )

    parsed = _parse_sse_response(r.text)
    if parsed is None:
        return f"[error] gahmen {tool_name} could not parse response: {r.text[:300]}"

    if "error" in parsed:
        err = parsed["error"]
        return f"[error] gahmen {tool_name} JSON-RPC error: {err.get('message', err)}"

    result = parsed.get("result", {})
    # MCP tool-call result: {"content": [{"type":"text","text":"..."}], "isError": bool}
    content = result.get("content", []) if isinstance(result, dict) else []
    chunks = [c.get("text", "") for c in content if c.get("type") == "text"]
    text = "\n".join(c for c in chunks if c)
    if not text:
        # Fall back to raw result if no text parts (some tools return structured-only).
        text = json.dumps(result)[:8000]
    # Cap response so a huge table doesn't blow the agent's context.
    return text[:12000]


# ---------------------------------------------------------------------------
# Public functions — exact gahmen tool names with the gahmen_ prefix. Each
# is wrapped as a FunctionTool below. Docstrings flow into Gemini's tool
# schema, so write them as the LLM-facing description.
# ---------------------------------------------------------------------------


async def gahmen_singstat_search_resources(keyword: str) -> str:
    """Search SingStat (Singapore Department of Statistics) for tables matching `keyword`.

    Returns a list of resource_ids and titles. ALWAYS use this FIRST when
    looking for a specific Singapore statistic — unemployment, GDP,
    demographics, trade, manpower, housing, transport. Then call
    gahmen_singstat_get_table_data with the chosen resource_id.

    Args:
        keyword: Search term, e.g., "unemployment", "GDP", "labour force".
    """
    return await _call("singstat_search_resources", {"keyword": keyword})


async def gahmen_singstat_get_metadata(resource_id: str) -> str:
    """Get metadata (column names, available time range, units) for a SingStat table.

    Use this BEFORE fetching rows to understand the table's structure,
    especially when you need to know which `seriesNoOrRowNo` values exist
    or what time format the table uses.

    Args:
        resource_id: SingStat table id (e.g., "M015711"). Get from
            gahmen_singstat_search_resources first.
    """
    return await _call("singstat_get_metadata", {"resourceId": resource_id})


async def gahmen_singstat_get_table_data(
    resource_id: str,
    series_no_or_row_no: str = "",
    time_filter: str = "",
) -> str:
    """Fetch the actual data rows of a SingStat table.

    Use this AFTER you know the resource_id (from search) and ideally the
    structure (from metadata). Returns the table data (often a list of
    series with values).

    Args:
        resource_id: SingStat table id.
        series_no_or_row_no: Optional row filter as a string (1-indexed),
            e.g., "1" or "1,2,3". Empty string means "all rows".
        time_filter: Optional time filter, e.g., "2018,2019,2020,2021,2022,2023,2024"
            or "2018-2024" (table-specific format — check metadata).
            Empty string means "all available periods".
    """
    args: dict[str, Any] = {"resourceId": resource_id}
    if series_no_or_row_no:
        args["seriesNoOrRowNo"] = series_no_or_row_no
    if time_filter:
        args["timeFilter"] = time_filter
    return await _call("singstat_get_table_data", args)


async def gahmen_datagovsg_search_dataset(query: str) -> str:
    """Search data.gov.sg for datasets matching `query`.

    Use for granular operational/transactional Singapore-government data
    (HDB resale prices, LTA traffic, MAS exchange rates, etc.). For
    macro/labor/demographic time series, prefer SingStat instead.

    Args:
        query: Free-text search term.
    """
    return await _call("datagovsg_search_dataset", {"query": query})


async def gahmen_datagovsg_get_dataset_metadata(dataset_id: str) -> str:
    """Get metadata for a data.gov.sg dataset by id.

    Args:
        dataset_id: data.gov.sg dataset id (UUID-like or slug).
    """
    return await _call("datagovsg_get_dataset_metadata", {"datasetId": dataset_id})


async def gahmen_datagovsg_list_collections(page: int = 1) -> str:
    """List all data.gov.sg collection topics (paginated).

    Use to browse what's available when you don't know the right keywords.

    Args:
        page: Page number (1-indexed). Default 1.
    """
    return await _call("datagovsg_list_collections", {"page": page})


async def gahmen_datagovsg_get_collection(collection_id: str) -> str:
    """Get a data.gov.sg collection (datasets within that topic).

    Args:
        collection_id: Collection id from list_collections.
    """
    return await _call("datagovsg_get_collection", {"collectionId": collection_id})


async def gahmen_datagovsg_list_datasets(
    collection_id: str = "",
    page: int = 1,
) -> str:
    """List data.gov.sg datasets, optionally filtered by collection_id.

    Args:
        collection_id: Optional collection id to filter datasets. Empty
            string lists all.
        page: Page number (1-indexed). Default 1.
    """
    args: dict[str, Any] = {"page": page}
    if collection_id:
        args["collectionId"] = collection_id
    return await _call("datagovsg_list_datasets", args)


# ---------------------------------------------------------------------------
# Module export — the 8 FunctionTool instances. Empty list when SMITHERY_API_KEY
# is unset (the deploy will run consult_level_1-only, same as W9.2 fallback).
# ---------------------------------------------------------------------------


if _API_KEY:
    GAHMEN_TOOLS: list[FunctionTool] = [
        FunctionTool(gahmen_singstat_search_resources),
        FunctionTool(gahmen_singstat_get_metadata),
        FunctionTool(gahmen_singstat_get_table_data),
        FunctionTool(gahmen_datagovsg_search_dataset),
        FunctionTool(gahmen_datagovsg_get_dataset_metadata),
        FunctionTool(gahmen_datagovsg_list_collections),
        FunctionTool(gahmen_datagovsg_get_collection),
        FunctionTool(gahmen_datagovsg_list_datasets),
    ]
else:
    GAHMEN_TOOLS = []
