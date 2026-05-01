"""A2A peer-agent tools for Level 4.

The single function below — `consult_level_1` — wraps an `on_message_send`
call against the A2A-enabled Level 1 reasoning engine. It is exposed to
`data_fetcher_agent` as a `FunctionTool`, alongside `google_search`
(Gemini built-in, swapped to function-tool by `bypass_multi_tools_limit`),
`load_web_page` (function), and the gahmen MCP toolset.

Why this lives in Level 4 (not in a shared library)
---------------------------------------------------
Each Vertex Agent Engine deploy is a self-contained source bundle (the
`extra_packages=[<dir>]` upload). Importing from a sibling package would
fail at runtime in the deployed container — the sibling isn't bundled.
So the consult helper lives in `level_4_agent/`, gets uploaded with the
rest of the agent, and reads its peer's resource ID from an env var
that `deploy_a2a.py` auto-forwards.

ADK 2.0 design patterns demonstrated by Level 4
-----------------------------------------------
1. **MCP integration** (gahmen-mcp via Smithery) — `data_fetcher_agent`
   gets vertical SG-government data tools as `gahmen_*` function tools.
2. **A2A peer consultation** — `data_fetcher_agent` can delegate
   well-formed search-and-summarise tasks to Level 1 (a separately-
   deployed reasoning engine that fronts `google_search`) over the A2A
   protocol's `on_message_send` operation.

The two patterns are independent — both can be active at once, neither
depends on the other, and either can be removed without breaking the
agent. That independence is the point: in production multi-agent
systems, MCP and A2A are routinely both present in the same agent.
"""
from __future__ import annotations

import logging
import os

import vertexai

logger = logging.getLogger(__name__)

# Where Level 1's A2A engine lives. Defaults to the resource ID minted
# by Phase 7 (the level_1_agent A2A redeploy with `gemini-2.5-flash`).
# Override via env var for re-deploys without rebuilding Level 4.
_LEVEL_1_REGION = os.environ.get("LEVEL_1_A2A_REGION", "asia-southeast1")
_LEVEL_1_RESOURCE_ID = os.environ.get(
    "LEVEL_1_A2A_ENGINE_ID",
    "2134899737420103680",
)
_PROJECT_NUMBER = os.environ.get(
    "LEVEL_1_A2A_PROJECT_NUMBER",
    "888142536377",
)

_FULL_RESOURCE_NAME = (
    f"projects/{_PROJECT_NUMBER}/locations/{_LEVEL_1_REGION}/"
    f"reasoningEngines/{_LEVEL_1_RESOURCE_ID}"
)


# Cache one Vertex client per region. A2aAgent URLs are region-bound, so
# the SDK builds a different base_url per region.
_clients: dict[str, vertexai.Client] = {}


def _client(region: str) -> vertexai.Client:
    if region not in _clients:
        _clients[region] = vertexai.Client(location=region)
    return _clients[region]


def _extract_a2a_text(response) -> str:
    """Pull the agent's text reply out of an `on_message_send` response.

    A2A `on_message_send` returns `list[tuple[Task | Message, str | None]]`.
    Text lives at `Task.artifacts[*].parts[*].root.text` (preferred) or
    `Task.history[-1].parts[*].root.text` (fallback). Mirrors the helper
    in the swarm repo's `tests/integration/a2a_engines.py`.
    """
    chunks: list[str] = []

    def _from_parts(parts) -> None:
        for p in parts or []:
            root = getattr(p, "root", p)
            text = getattr(root, "text", None)
            if text:
                chunks.append(text)

    if response is None:
        return ""

    if hasattr(response, "parts"):
        _from_parts(response.parts)
        return "\n".join(chunks)

    items = response if isinstance(response, list) else [response]
    for entry in items:
        item = entry[0] if (isinstance(entry, tuple) and len(entry) >= 1) else entry
        if item is None:
            continue

        artifacts = getattr(item, "artifacts", None)
        if artifacts:
            for art in artifacts:
                _from_parts(getattr(art, "parts", None))
            if chunks:
                continue

        history = getattr(item, "history", None)
        if history:
            for msg in reversed(history):
                role = getattr(msg, "role", None)
                role_value = getattr(role, "value", role)
                if str(role_value) == "agent":
                    _from_parts(getattr(msg, "parts", None))
                    break
            if chunks:
                continue

        _from_parts(getattr(item, "parts", None))

    return "\n".join(chunks)


async def consult_level_1(query: str) -> str:
    """Delegate a search-and-summarise task to the Level 1 agent over A2A.

    Level 1 is a single LlmAgent with `google_search` as its only tool — a
    minimalist "ask a question, get a sourced answer" specialist. Use this
    when:
      - the question is well-formed enough to hand off as-is to a peer
        (i.e., a complete question, not a raw search keyword),
      - you want the peer to do the searching AND the answering (Level 1
        replies in natural language with inline source attribution),
      - the question is NOT Singapore-specific (use `gahmen_*` for SG),
      - you do NOT need raw search hits to feed into further analysis
        (use `google_search` directly for that).

    Args:
      query: A complete, self-contained question. Examples: "What was
        Apple's Q4 2025 revenue?", "When was the Suez Canal expansion
        completed?". Avoid bare keywords.

    Returns:
      Level 1's full text reply with inline citations. Empty string if
      the call succeeded but Level 1 returned no text. Error string
      starting with "[error]" or "[empty]" if the wire-protocol call
      failed.
    """
    try:
        remote = _client(_LEVEL_1_REGION).agent_engines.get(
            name=_FULL_RESOURCE_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to bind Level 1 A2A engine")
        return f"[error] Could not reach Level 1 engine: {exc}"

    if not hasattr(remote, "on_message_send"):
        return (
            "[error] Level 1 engine has no on_message_send — its deploy "
            "may have used the legacy AdkApp template instead of A2aAgent. "
            "Check check_a2a.py output."
        )

    try:
        response = await remote.on_message_send(
            messageId="level_4_consult",
            role="user",
            parts=[{"kind": "text", "text": query}],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("on_message_send to Level 1 failed")
        return f"[error] A2A call to Level 1 failed: {exc}"

    text = _extract_a2a_text(response)
    if not text:
        return "[empty] Level 1 returned no text parts."
    return text
