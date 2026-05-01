"""Level 1 — Connected Problem-Solver (ADK 2.0 rewrite).

Taxonomy: "Level 1 — The Connected Problem-Solver". A single agent that
connects to one external tool (`google_search`) to answer questions
requiring real-time information. Identify need → search → answer. No
planning, no loops, no multi-agent topology.

Why this is still a single `LlmAgent` in ADK 2.0
------------------------------------------------
ADK 2.0's headline feature is the `Workflow(BaseNode)` graph runtime, but
wrapping a one-shot search-and-answer turn in a graph node would be cargo-
cult v2 — it adds a node boundary that teaches nothing at this level. The
CHANGELOG explicitly calls out the *opposite* direction for leaf cases:

  > Optimized execution by bypassing the Mesh for leaf single-turn
  > `LlmAgent` instances.  — `CHANGELOG-v2.md`

So in v2, a leaf `LlmAgent` is the *fast path* — the runtime detects that
no graph orchestration is needed and skips the per-node scheduler. Level 1
is the canonical case for that fast path. Workflow graphs are introduced
when the lesson actually requires them (Level 3+ delegation, Level 4
dynamic-node spawning).

What is genuinely "v2" here
---------------------------
1. `Agent` (alias for `LlmAgent`) — same primitive, but in v2 the leaf
   execution path bypasses the Mesh, so this code now runs on the
   optimized fast path with no source change required.
2. `output_key="last_answer"` — v2 formalized state-delta flushing onto
   yielded events (CHANGELOG: "Supported flushing state/artifact deltas
   onto yielded events"). With `output_key` set, the agent's final text
   appears as a `state_delta` on the same event that carries the
   response, so downstream consumers (eval harnesses, the web UI trace
   panel) can observe state changes turn-by-turn without polling.

Kept identical to V1
--------------------
- One tool: the `google_search` built-in. No custom search wrap.
- No `planner`. Reasoning is introduced at Level 2+ as the next
  taxonomy jump; at L1 an explicit Plan→Reason→Act scaffold adds cost
  without quality gain.
- No state read at L1. (`output_key` writes state for observability,
  but the agent does not read state via instruction injection — that
  pattern starts at L2.)
"""

from google.adk import Agent
from google.adk.tools.google_search_tool import google_search

root_agent = Agent(
    name="level_1_agent",
    # Was `gemini-3.1-flash-lite-preview` (preview alias resolves only via
    # GOOGLE_CLOUD_LOCATION=global). Switched to `gemini-2.5-flash` because
    # Vertex Agent Engine deploys force-overwrite the location to the
    # engine's region (templates/a2a.py:241-245) and the preview alias 404s
    # in regional endpoints like asia-southeast1. `gemini-2.5-flash` works
    # in both `global` and `asia-southeast1`, so the local `adk run` path
    # is unaffected. See DEPLOYMENT_NOTES.md "Phase 7" for context.
    model="gemini-2.5-flash",
    description=(
        "A connected problem-solver that uses Google Search to answer"
        " questions requiring real-time information."
    ),
    instruction=(
        "You answer questions using Google Search. One search per question."
        " Cite your sources. If the question is ambiguous, ask for"
        " clarification first."
    ),
    tools=[google_search],
    # v2: final response auto-writes to state['last_answer'] AND flushes
    # as a state_delta on the yielded event. Visible in the web UI's
    # State panel and in `Event.actions.state_delta` for programmatic
    # consumers. No effect on the LLM's behaviour.
    output_key="last_answer",
)
