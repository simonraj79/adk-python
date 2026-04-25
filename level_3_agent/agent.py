"""Level 3 — Collaborative Multi-Agent System (ADK 2.0 rewrite).

Taxonomy: "Level 3 — The Collaborative Multi-Agent System". A
coordinator delegates work to a team of specialist sub-agents, each with
a distinct role and non-overlapping tools, and synthesises their typed
results into a single user-facing brief.

What changed v1 → v2 (and why)
------------------------------
v1 (`V1_level_3_agent/agent.py`) wrapped each specialist as an
`AgentTool(agent=X)` and put them on the coordinator's `tools=[...]`
list. That worked, but it was a v1 *workaround*: the only way to get
"call-and-return" semantics (instead of `sub_agents`' "transfer-and-
maybe-return" semantics) was to manually wrap each specialist as a tool.

v2 makes call-and-return delegation a first-class primitive. The
coordinator declares `sub_agents=[...]` and the framework auto-creates a
`request_task_<name>` tool **for each sub-agent that has a non-chat
mode** (`src/google/adk/agents/llm_agent.py:982-994`). Concretely:

    sub_agent.mode = 'single_turn'  # or 'task'
    # → framework auto-appends `_SingleTurnAgentTool(sub_agent)` to the
    #   coordinator's `tools` at agent-construction time. The coordinator
    #   delegates by calling `request_task_<name>(<input_schema args>)`.
    #   Sub-agent runs to completion and the result auto-returns to the
    #   coordinator's LLM context.

YouTube ADK 2.0 walkthrough framing: *"the moment the task completes,
it will again auto return."*

Why each specialist uses `mode='single_turn'`
---------------------------------------------
v2 offers three modes (`llm_agent.py:312`):

| Mode             | Sub-agent multi-turn? | Auto-injected tool         |
|------------------|-----------------------|----------------------------|
| `'chat'` (default) | yes (full transfer)   | (transfer_to_agent path)   |
| `'task'`           | yes (can clarify)     | `finish_task` (function)   |
| `'single_turn'`    | no (autonomous)       | none                       |

For research specialists in our setup, the **coordinator** is the
"user" — it gives full instructions in the structured input and expects
a result, no clarification round-trip. That's `single_turn`.

There's also a concrete tool-compatibility reason: `mode='task'` causes
the framework to inject `FinishTaskTool` into the sub-agent's `tools`
(`llm_agent.py:977-980`). Combined with `tools=[google_search]`, that
makes `multiple_tools = len(self.tools) > 1` (`llm_agent.py:692`)
trigger Gemini's "built-in + function tool" conflict — the same v1
gotcha #24 that required `bypass_multi_tools_limit=True`. With
`mode='single_turn'`, no `FinishTaskTool` is injected, so a sub-agent
with `tools=[google_search]` has exactly one tool — no conflict, no
workaround. **Mode choice gates tool compatibility in v2.**

If you wanted a clarifying-questions specialist (e.g., a follow-up
analyst that asks the user for more context), set `mode='task'` AND
either drop the built-in tool or set `bypass_multi_tools_limit=True`.

Structured I/O via Pydantic
---------------------------
Each specialist declares `input_schema` and `output_schema` Pydantic
models. The coordinator's auto-generated `request_task_<name>` tool
exposes the input schema as the function signature, and the sub-agent's
last response is parsed against the output schema before being returned
to the coordinator. This replaces v1's prompt-driven contracts ("save a
note in this format", "format the brief like this") with framework-
enforced typed boundaries.

What stays from v1
------------------
- **Three-specialist team**: search / analyst / writer. Same role
  decomposition as v1 — the v1 design was sound; only the wiring
  changes.
- **`PlanReActPlanner()` on the coordinator**. The coordinator's
  routing decision IS its reasoning ("which specialist next, with what
  input?"), and making that visible inline teaches the multi-agent
  pattern. v2 didn't deprecate this — it's still the right choice for
  *runtime LLM-decided* routing (different from Level 2, whose routing
  is deterministic and graph-encoded).
- **Greeting carve-out**: handled in the coordinator's instruction
  (same prompt-level pattern as v1). For L3 we don't need the
  classifier-router pattern from L2 because the coordinator is already
  an LLM that decides what to do — adding a classifier just adds a
  hop without teaching anything new.

What this is NOT
----------------
This is **not** a `Workflow(BaseNode)` graph. The coordinator's
delegation order is decided by the LLM at runtime based on the
question's content, not encoded as static edges. Wrapping multi-agent
delegation in a `Workflow` would force a fixed graph and waste the
coordinator's runtime reasoning — the AGENTS.md migration pattern #6
("Leaf `LlmAgent` is the fast path — don't force-fit `Workflow`")
generalises here: don't graph-wrap LLM-decided routing either.

Sample queries to try
---------------------
- "hi"  →  coordinator handles directly, no delegation
- "what's the latest in solid-state battery research?"
        → coordinator: search_agent (one or two times) → analyst_agent
          (review findings) → writer_agent (final brief)
- "compare mRNA vs viral-vector vaccine platforms"
        → coordinator: search_agent × 2 (parallel-ish via separate
          delegate calls) → analyst_agent → writer_agent
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.tools.google_search_tool import google_search
from pydantic import BaseModel
from pydantic import Field


# ---------------------------------------------------------------------------
# Pydantic schemas — the typed contracts at every coordinator/specialist
# boundary. These replace v1's prompt-level instructions like "save a
# note with topic, finding, source" with framework-enforced types.
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
  """Argument schema the coordinator passes to `request_task_search_agent`."""

  query: str = Field(
      description=(
          "A focused, self-contained search query (one sub-question)."
          " Avoid compound queries; call this multiple times for"
          " multi-aspect research."
      )
  )


# NOTE: search_agent has no `output_schema` — see search_agent below for
# the rationale (Gemini API limitation: output_schema + built-in tools
# is incompatible because the framework injects `set_model_response`
# which conflicts with `google_search`). Findings flow as plain text
# strings instead of typed `Finding` objects.


class AnalystInput(BaseModel):
  """Argument schema the coordinator passes to `request_task_analyst_agent`."""

  question: str = Field(description="The original user question.")
  findings: list[str] = Field(
      description=(
          "All search findings collected so far, as plain text strings"
          " (one per search_agent call). The strings should include"
          " inline source attribution like 'According to nature.com,...'."
      )
  )


class Analysis(BaseModel):
  """Structured cross-finding review returned from analyst_agent."""

  patterns: str = Field(
      description="Recurring themes or convergent claims across findings."
  )
  contradictions: str = Field(
      description=(
          "Any disagreements between sources, or 'None observed' if findings"
          " converge."
      )
  )
  gaps: str = Field(
      description=(
          "What the findings do NOT cover but the question implies — areas"
          " where the writer should flag uncertainty."
      )
  )


class WriterInput(BaseModel):
  """Argument schema the coordinator passes to `request_task_writer_agent`."""

  question: str = Field(description="The original user question.")
  findings: list[str] = Field(
      description="All search findings as plain text strings."
  )
  analysis: Analysis = Field(
      description="Analyst's cross-finding review (patterns/contradictions/gaps)."
  )


class Brief(BaseModel):
  """Final structured research brief returned from writer_agent."""

  title: str = Field(description="Short title summarising the question.")
  key_findings: str = Field(
      description="Markdown bullet list of 3-5 most important findings."
  )
  detailed_analysis: str = Field(
      description="2-3 paragraphs synthesising findings with inline source attribution."
  )
  sources: list[str] = Field(description="Deduplicated list of all source domains cited.")
  confidence_and_gaps: str = Field(
      description=(
          "Where the brief is most/least confident, drawing on the analyst's"
          " gaps section."
      )
  )


# ---------------------------------------------------------------------------
# Specialists — each declares its mode, tools, and typed I/O contract.
# All three use `mode='single_turn'` because the coordinator (not the
# human user) is the caller and gives full instructions in the
# structured input — no clarification round-trip needed.
# ---------------------------------------------------------------------------


search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description=(
        "Searches the web for one focused sub-question and returns a"
        " plain-text finding with source domains cited inline."
    ),
    # mode='single_turn': autonomous one-shot.
    mode="single_turn",
    input_schema=SearchInput,
    # NO `output_schema=...` ON THIS AGENT — see commentary above.
    instruction=(
        "Use google_search to answer the query in the structured input."
        " Return a 2-3 sentence finding with the source domains you"
        " cited inline (e.g., 'According to nature.com, ...'). Be"
        " concise — one search per call. The coordinator will call"
        " you again for additional sub-questions if needed."
    ),
    tools=[google_search],
    # CRITICAL for `google_search` + sub_agent compatibility:
    # by default ADK's `agent_transfer.py:188` auto-injects a
    # `transfer_to_agent` function tool on any sub-agent whose
    # `disallow_transfer_to_parent` is False, so that the sub-agent can
    # transfer the conversation back to its parent. That function tool
    # then conflicts with the `google_search` built-in (Gemini's
    # "Built-in tools and Function Calling cannot be combined" — the v1
    # gotcha #24 situation, very much still alive in v2). Setting both
    # flags True suppresses the injection. We don't need transfer
    # anyway — `single_turn` agents auto-return their result via
    # `request_task` without needing a transfer call.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


analyst_agent = Agent(
    name="analyst_agent",
    model="gemini-2.5-flash",
    description=(
        "Reviews accumulated search findings for patterns, contradictions,"
        " and gaps. Pure LLM reasoning — no tools."
    ),
    mode="single_turn",
    input_schema=AnalystInput,
    output_schema=Analysis,
    instruction=(
        "Review the findings below for the original question. Identify"
        " recurring themes (patterns), source disagreements"
        " (contradictions), and what the findings DO NOT cover (gaps)."
        " Be specific — cite which findings support each claim. If"
        " findings converge cleanly, say 'None observed' for"
        " contradictions; do not invent disagreement."
    ),
    # Suppress transfer_to_agent injection (consistent with search_agent)
    # — single_turn sub-agents auto-return via request_task; they should
    # never transfer.
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


writer_agent = Agent(
    name="writer_agent",
    model="gemini-2.5-flash",
    description=(
        "Synthesises findings + analysis into the final structured Brief"
        " for the user. Pure LLM reasoning — no tools."
    ),
    mode="single_turn",
    input_schema=WriterInput,
    output_schema=Brief,
    instruction=(
        "Produce a Brief that answers the original question using the"
        " findings and the analyst's review. Lead with the most"
        " important finding. Weave source attribution inline. Use the"
        " analyst's `gaps` section to populate `confidence_and_gaps` —"
        " do not hide uncertainty. Distinguish facts from opinions."
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ---------------------------------------------------------------------------
# Coordinator — the only agent the user talks to. Delegates via the
# auto-generated `request_task_<name>` tools (created by the framework
# from `sub_agents=[...]` because each specialist has a non-chat mode).
# `PlanReActPlanner()` makes the coordinator's routing decisions
# visible inline in the chat panel.
# ---------------------------------------------------------------------------


root_agent = Agent(
    name="level_3_agent",
    model="gemini-2.5-flash",
    description=(
        "Research coordinator that delegates to search, analyst, and"
        " writer specialists and returns a structured brief. Routing"
        " decisions are made by the coordinator at runtime and are"
        " visible in the chat stream via PlanReActPlanner."
    ),
    # PlanReActPlanner is a prompt-level scaffold that forces the LLM to
    # emit explicit PLANNING / REASONING / ACTION / FINAL_ANSWER
    # sections. For L3 the coordinator's plan IS the routing
    # decision — making it visible inline teaches the multi-agent
    # pattern better than hiding it in side-channel thoughts (which is
    # what BuiltInPlanner with thinking_config does). Different choice
    # from L2, where routing is graph-deterministic and
    # plan-then-act adds nothing.
    planner=PlanReActPlanner(),
    instruction=(
        "You are a research coordinator with three specialists you can"
        " delegate to via the request_task_<name> tools the framework"
        " has wired in for you:\n"
        "  - request_task_search_agent(query): one focused web search."
        "    Call multiple times for multi-aspect questions.\n"
        "  - request_task_analyst_agent(question, findings): review"
        " accumulated findings for patterns, contradictions, and gaps."
        " Call once after all searches are done.\n"
        "  - request_task_writer_agent(question, findings, analysis):"
        " produce the final structured Brief. Call once at the end."
        " The Brief is your final answer to the user.\n\n"
        "GREETINGS, META QUESTIONS, AND EMPTY OR UNCLEAR INPUT (e.g."
        " 'hi', 'hello', 'what can you do?'): respond directly yourself —"
        " do NOT delegate. Reply with exactly two sentences: one"
        " describing the team, and one inviting a research topic."
        ' Example: "I\'m a research coordinator — I delegate to a'
        " search agent, an analyst, and a writer to produce structured"
        ' briefs with sources. What would you like me to research?"'
        "\n\n"
        "FOR EVERY OTHER QUESTION, follow this orchestration pattern:\n"
        "  1. PLAN: decide which sub-questions to search. Simple"
        " factual question → 1 search. Multi-aspect question → 2 or 3"
        " searches.\n"
        "  2. EXECUTE: call request_task_search_agent once per"
        " sub-question. Collect each Finding from the tool response.\n"
        "  3. REVIEW: call request_task_analyst_agent ONCE with the"
        " original question and ALL collected findings.\n"
        "  4. WRITE: call request_task_writer_agent ONCE with the"
        " question, findings, and analysis. Return the resulting Brief"
        " to the user.\n\n"
        "Do not do the research yourself — delegate. Do not write the"
        " brief yourself — delegate."
    ),
    sub_agents=[search_agent, analyst_agent, writer_agent],
)
