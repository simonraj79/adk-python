"""A2A tools for the orchestrator — five `consult_level_*` FunctionTool wrappers.

Each function calls `on_message_send` against a separately deployed Vertex
Agent Engine over the A2A protocol. The Levels live in `asia-southeast1`;
this orchestrator lives in `us-central1` (Pro is regional). So every consult
is a cross-region A2A call: ~150-200ms RTT added to the inference time, all
authenticated by the appspot SA's `roles/aiplatform.user` (no per-resource
IAM grants needed — see CLAUDE.md "Why A2A on Vertex sidesteps NTU's
setIamPolicy deny").

Resource IDs are baked in as defaults BUT can be overridden at deploy time
via env vars (deploy_a2a.py auto-forwards anything starting with LEVEL_).
That way Phase A redeploys of any Level (which mint new IDs) don't require
a code edit here — set `LEVEL_X_A2A_ENGINE_ID` in the deploy shell.

Why these IDs are correct for THIS deploy: pulled from
`adk/tests/integration/a2a_engines.py` after Phase A finished 2026-04-28.
"""
from __future__ import annotations

import logging
import os

import vertexai

logger = logging.getLogger(__name__)

# All five Level engines live in asia-southeast1.
_LEVEL_REGION = os.environ.get("LEVEL_REGION", "asia-southeast1")
_PROJECT_NUMBER = os.environ.get("LEVEL_PROJECT_NUMBER", "888142536377")

# Defaults are the post-Phase-A engine IDs (verified 2026-04-28). Override
# any of these by setting the corresponding env var in the deploy shell.
_LEVEL_IDS = {
    "level_1":  os.environ.get("LEVEL_1_A2A_ENGINE_ID",  "2134899737420103680"),
    "level_2":  os.environ.get("LEVEL_2_A2A_ENGINE_ID",  "2181061633600651264"),
    "level_2b": os.environ.get("LEVEL_2B_A2A_ENGINE_ID", "1635000178781978624"),
    "level_3":  os.environ.get("LEVEL_3_A2A_ENGINE_ID",  "1988532749530562560"),
    "level_4":  os.environ.get("LEVEL_4_A2A_ENGINE_ID",  "4048929579052564480"),
}


# One Vertex client per region; cached to avoid rebuilding auth state on every
# call. Region-bound: A2aAgent constructs a region-specific URL at set_up.
_clients: dict[str, vertexai.Client] = {}


def _client(region: str) -> vertexai.Client:
    if region not in _clients:
        _clients[region] = vertexai.Client(location=region)
    return _clients[region]


def _full_name(slug: str) -> str:
    rid = _LEVEL_IDS[slug]
    return (
        f"projects/{_PROJECT_NUMBER}/locations/{_LEVEL_REGION}/"
        f"reasoningEngines/{rid}"
    )


def _extract_a2a_text(response) -> str:
    """Pull the agent's text reply out of an `on_message_send` response.

    A2A `on_message_send` returns `list[tuple[Task | Message, str | None]]`.
    Text lives at `Task.artifacts[*].parts[*].root.text` (preferred) or
    `Task.history[-1].parts[*].root.text` (fallback for the last agent
    message). Mirrors the helper in the swarm repo's
    `tests/integration/a2a_engines.py`.
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


async def _consult(slug: str, query: str) -> str:
    """Generic A2A consult helper. The five public consult_level_* functions
    below are thin wrappers — each binds a slug so Gemini sees them as
    distinct, well-named tools (with their own docstrings driving routing).
    """
    try:
        remote = _client(_LEVEL_REGION).agent_engines.get(name=_full_name(slug))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to bind %s engine", slug)
        return f"[error] Could not reach {slug} engine: {exc}"

    if not hasattr(remote, "on_message_send"):
        return (
            f"[error] {slug} engine has no on_message_send — its deploy may "
            "have used the legacy AdkApp template instead of A2aAgent. "
            "Check check_a2a.py output."
        )

    try:
        response = await remote.on_message_send(
            messageId=f"orchestrator-consult-{slug}",
            role="user",
            parts=[{"kind": "text", "text": query}],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("on_message_send to %s failed", slug)
        return f"[error] A2A call to {slug} failed: {exc}"

    text = _extract_a2a_text(response)
    if not text:
        return f"[empty] {slug} returned no text parts."
    return text


# ----------------------------------------------------------------------------
# Public consult tools — one per Level. Docstrings drive Gemini's routing
# decisions; keep them precise and prescriptive.
# ----------------------------------------------------------------------------


async def consult_level_1(query: str) -> str:
    """Consult the Level 1 agent (single LlmAgent + google_search) over A2A.

    Use when:
      - the question is a simple factual lookup answerable with one web search,
      - you want a direct answer (Level 1 returns natural-language reply with
        inline source attribution),
      - the question is NOT Singapore-specific (use consult_level_4 for SG —
        it has authoritative gahmen MCP data).

    Examples of well-suited queries:
      - "What was Apple's Q4 2025 revenue?"
      - "When did the EU AI Act take effect?"
      - "Who is the current CEO of Tesla?"
    """
    return await _consult("level_1", query)


async def consult_level_2(query: str) -> str:
    """Consult the Level 2 agent (Day Planner / Strategic Problem-Solver) over A2A.

    Use when:
      - the question is a planning task (study schedule, project breakdown,
        multi-step task decomposition),
      - the user wants a structured timetable or sequenced action list,
      - the question requires "decompose → look up → assemble" workflow.

    Examples of well-suited queries:
      - "Plan a focused 2-hour study block on solid-state batteries."
      - "Break down 'launch a podcast' into the next 4 weeks of milestones."
    """
    return await _consult("level_2", query)


async def consult_level_2b(query: str) -> str:
    """Consult the Level 2b agent (Graph Router / classify-then-route) over A2A.

    Use when:
      - the question is a short customer-support style message that needs
        triage (bug / billing / feature-request / greeting),
      - you want a category label + the routing decision visible.

    Examples of well-suited queries:
      - "My credit card was charged twice — what should I do?"
      - "Hey can the dashboard support dark mode?"

    Note: Level 2b returns JSON-shaped routing output (e.g.,
    `{"category": "BILLING"}`). For chat-style answers, prefer other levels.
    """
    return await _consult("level_2b", query)


async def consult_level_3(query: str) -> str:
    """Consult the Level 3 agent (Research Coordinator with sub-agents) over A2A.

    Use when:
      - the question is a multi-aspect research topic where you want a
        STRUCTURED brief (key findings, patterns, contradictions, gaps),
      - the question benefits from a coordinator that delegates to a
        search agent + analyst + writer (Level 3's internal pipeline),
      - you want explicit confidence/uncertainty surfaced in the answer.

    Examples of well-suited queries:
      - "Compare mRNA vs viral-vector vaccine platforms — efficacy, safety,
        manufacturing scale."
      - "What's the latest in solid-state battery commercialisation?"
    """
    return await _consult("level_3", query)


async def consult_level_4(query: str) -> str:
    """Consult the Level 4 agent (Self-Evolving BI + gahmen MCP + A2A) over A2A.

    Use when:
      - the question involves SINGAPORE-specific data (any SG ministry,
        agency, dataset, indicator — Level 4 has gahmen MCP tools that
        access SingStat and data.gov.sg directly),
      - the question is a business-intelligence query needing computation
        + charts (Level 4 has a code-executor sub-agent for pandas /
        matplotlib),
      - the question might need a NEW specialist agent created at runtime
        (Level 4 has the agent_creator with native thinking).

    Examples of well-suited queries:
      - "What's Singapore's resident unemployment rate over the last 4
        quarters? Compute the QoQ delta."
      - "Compare HDB resale prices vs private property prices in 2025."
      - "Build a specialist that pulls Singapore weather observations."

    Note: Level 4 internally consults Level 1 via A2A for non-SG web
    queries and uses gahmen MCP for SG data. So this consult can result
    in TWO inter-system calls under the hood — you pay for that
    transitively in latency.
    """
    return await _consult("level_4", query)
