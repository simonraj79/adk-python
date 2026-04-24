# AI Coding Assistant Context

This document provides context for AI coding assistants (Claude Code, Gemini
CLI, GitHub Copilot, Cursor, etc.) to understand the ADK Python project and
assist with development.

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for
building, evaluating, and deploying sophisticated AI agents with flexibility and
control. While optimized for Gemini and the Google ecosystem, ADK is
model-agnostic, deployment-agnostic, and is built for compatibility with other
frameworks. ADK was designed to make agent development feel more like software
development, to make it easier for developers to create, deploy, and orchestrate
agentic architectures that range from simple tasks to complex workflows.

### Key Components

-   **Agent** - Blueprint defining identity, instructions, and tools
    (`LlmAgent`, `LoopAgent`, `ParallelAgent`, `SequentialAgent`, etc.)
-   **Runner** - Execution engine that orchestrates agent execution. It manages
    the 'Reason-Act' loop, processes messages within a session, generates
    events, calls LLMs, executes tools, and handles multi-agent coordination. It
    interacts with various services like session management, artifact storage,
    and memory, and integrates with application-wide plugins. The runner
    provides different execution modes: `run_async` for asynchronous execution
    in production, `run_live` for bidirectional streaming interaction, and
    `run` for synchronous execution suitable for local testing and debugging. At
    the end of each invocation, it can perform event compaction to manage
    session history size.
-   **Tool** - Functions/capabilities agents can call (Python functions, OpenAPI
    specs, MCP tools, Google API tools)
-   **Session** - Conversation state management (in-memory, Vertex AI,
    Spanner-backed)
-   **Memory** - Long-term recall across sessions

### How the Runner Works

The Runner is the stateless orchestration engine that manages agent execution.
It does not hold conversation history in memory; instead, it relies on services
like `SessionService`, `ArtifactService`, and `MemoryService` for persistence.

**Invocation Lifecycle:**

Each call to `runner.run_async()` or `runner.run()` processes a single user
turn, known as an **invocation**.

1.  **Session Retrieval:** When `run_async()` is called with a `session_id`, the
    runner fetches the session state, including all conversation events, from
    the `SessionService`.
2.  **Context Creation:** It creates an `InvocationContext` containing the
    session, the new user message, and references to persistence services.
3.  **Agent Execution:** The runner calls `agent.run_async()` with this context.
    The agent then enters its reason-act loop, which may involve:
    *   Calling an LLM for reasoning.
    *   Executing tools (function calling).
    *   Generating text or audio responses.
    *   Transferring control to sub-agents.
4.  **Event Streaming & Persistence:** Each step in the agent's execution (LLM
    call, tool call, tool response, model response) generates `Event` objects.
    The runner streams these events back to the caller and simultaneously
    appends them to the session via `SessionService`.
5.  **Invocation Completion:** Once the agent has produced its final response
    for the turn (e.g., a text response to the user), the agent's execution loop
    finishes.
6.  **Event Compaction:** If event compaction is configured, the runner may
    summarize older events in the session to manage context window limits,
    appending a `CompactedEvent` to the session.
7.  **Next Turn:** When the user sends another message, a new `run_async()`
    invocation begins, repeating the cycle by loading the session, which now
    includes the events from all prior turns.

## Project Architecture

Please refer to
[ADK Project Overview and Architecture](https://github.com/google/adk-python/blob/main/contributing/adk_project_overview_and_architecture.md)
for details.

### Source Structure

```
src/google/adk/
├── agents/          # Agent implementations (LlmAgent, LoopAgent, ParallelAgent, etc.)
├── runners.py       # Core Runner orchestration class
├── tools/           # Tool ecosystem (50+ files)
│   ├── google_api_tool/
│   ├── bigtable/, bigquery/, spanner/
│   ├── openapi_tool/
│   └── mcp_tool/    # Model Context Protocol
├── models/          # LLM integrations (Gemini, Anthropic, LiteLLM)
├── sessions/        # Session management (in-memory, Vertex AI, Spanner)
├── memory/          # Long-term memory services
├── evaluation/      # Evaluation framework (47 files)
├── cli/             # CLI tools and web UI
├── flows/           # Execution flow orchestration
├── a2a/             # Agent-to-Agent protocol
├── telemetry/       # Observability and tracing
└── utils/           # Utility functions
```

### Test Structure

```
tests/
├── unittests/       # 2600+ unit tests across 236+ files
│   ├── agents/
│   ├── tools/
│   ├── models/
│   ├── evaluation/
│   ├── a2a/
│   └── ...
└── integration/     # Integration tests
```

### ADK Live (Bidi-streaming)

-   ADK live feature can be accessed from runner.run_live(...) and corresponding
    FAST api endpoint.
-   ADK live feature is built on top of
    [Gemini Live API](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api).
    We integrate Gemini Live API through
    [GenAI SDK](https://github.com/googleapis/python-genai).
-   ADK live related configs are in
    [run_config.py](https://github.com/google/adk-python/blob/main/src/google/adk/agents/run_config.py).
-   ADK live under multi-agent scenario: we convert the audio into text. This
    text will be passed to next agent as context.
-   Most logics are in
    [base_llm_flow.py](https://github.com/google/adk-python/blob/main/src/google/adk/flows/llm_flows/base_llm_flow.py)
    and
    [gemini_llm_connection.py](https://github.com/google/adk-python/blob/main/src/google/adk/models/gemini_llm_connection.py).
-   Input transcription and output transcription should be added to session as
    Event.
-   User audio or model audio should be saved into artifacts with a reference in
    Event to it.
-   Tests are in
    [tests/unittests/streaming](https://github.com/google/adk-python/tree/main/tests/unittests/streaming).

### Agent Structure Convention (Required)

**All agent directories must follow this structure:** `my_agent/ ├──
__init__.py # MUST contain: from . import agent └── agent.py # MUST define:
root_agent = Agent(...) OR app = App(...)`

**Choose one pattern based on your needs:**

**Option 1 - Simple Agent (for basic agents without plugins):** ```python from
google.adk.agents import Agent from google.adk.tools import google_search

root_agent = Agent( name="search_assistant", model="gemini-2.5-flash",
instruction="You are a helpful assistant.", description="An assistant that can
search the web.", tools=[google_search] ) ```

**Option 2 - App Pattern (when you need plugins, event compaction, custom
configuration):** ```python from google.adk import Agent from google.adk.apps
import App from google.adk.plugins import ContextFilterPlugin

root_agent = Agent( name="my_agent", model="gemini-2.5-flash", instruction="You
are a helpful assistant.", tools=[...], )

app = App( name="my_app", root_agent=root_agent, plugins=[
ContextFilterPlugin(num_invocations_to_keep=3), ], ) ```

**Rationale:** This structure allows the ADK CLI (`adk web`, `adk run`, etc.) to
automatically discover and load agents without additional configuration.

### Agent Instruction Best Practices (Required)

**ALWAYS refer to this section and the official ADK samples in
`contributing/samples/` when writing agent instructions.**

#### Keep Instructions Concise and Action-Oriented

Agent instructions should be **direct, minimal, and focused on constraints** —
not verbose methodology explanations. The LLM already understands common agentic
patterns (ReAct, planning, etc.). Your job is to **trigger** the right behaviour
with the fewest tokens, not to teach the LLM what ReAct is.

**Good instruction pattern:**
```
You are a ReAct web research agent. You answer questions by looping:
REASON what you need → ACT by calling web_search → OBSERVE the results → decide if sufficient.
Stop as soon as you have a complete answer. Use at most 3 searches.
```

**Bad instruction pattern (too verbose):**
```
You are a ReAct (Reason + Act) Web Research Agent. You answer questions by
iterating through a Reason → Act → Observe loop until you have enough information.

## Your ReAct Loop
For every user question, follow this cycle:

### REASON (think before you act)
Before each search, think step-by-step out loud:
- What specific information do I still need?
- What is the best search query to find it?
...
(40+ more lines explaining what ReAct is)
```

#### Let Tool Docstrings Do the Heavy Lifting

Tool docstrings are part of the LLM's context. They should explain **how and
when** to use each tool. The agent instruction should focus on **workflow,
constraints, and output format** — not repeat what the tool docstrings already
say.

#### Insights and Gotchas

1.  **Don't teach the LLM patterns it already knows.** Writing "A ReAct agent
    loops: Reason → Act → Observe → Reason → Act → ..." wastes tokens. The LLM
    knows ReAct. Just say "You are a ReAct agent" and specify constraints.

2.  **Avoid meta-commentary in instructions.** Sections like "How You Differ
    from a ReAct Agent" or "Your Core Capability: Context Engineering" are
    documentation for humans, not actionable instructions for the LLM. Put these
    in your README or design docs, not in the prompt.

3.  **Emoji and rich formatting in instructions are noise.** Using ✅ ❌ markers,
    elaborate markdown headers, and worked examples bloat the system prompt.
    Use plain text with minimal structure.

4.  **The official ADK `google_search_agent` instruction is one sentence.**
    This is the gold standard for simplicity. Start simple, add constraints only
    when the agent misbehaves during testing.

5.  **Verbose prompts waste context window and can confuse the LLM.** Every
    token in your instruction competes with the user's conversation history
    and tool outputs for context window space. Shorter instructions leave more
    room for the actual work.

6.  **Workflow steps should be terse.** Instead of multi-paragraph step
    descriptions, use a numbered list:
    `1. PLAN: Decompose into 2-3 sub-questions.`
    `2. EXECUTE: Search then save_research_note for each.`
    `3. SYNTHESISE: get_research_notes, cross-reference, write report.`

7.  **Test and iterate.** Start with a minimal instruction, run the agent, and
    only add constraints when you observe unwanted behaviour. This is more
    effective than front-loading a verbose prompt.

### ADK-First Development (Required)

**ALWAYS use ADK's built-in features before writing custom code.** The whole
point of ADK is declarative agent configuration — not reimplementing primitives
that the framework already provides.

#### ADK Feature Map

Before writing custom code, check if ADK already provides it:

| You need... | ADK provides... | Don't... |
|---|---|---|
| Web search | `google_search` built-in tool | Write a custom `web_search()` with `genai.Client` |
| State between tool calls | `ToolContext.state` | Build your own state store |
| Conversation history | `SessionService` | Manage chat history manually |
| Multi-agent orchestration | `sub_agents` parameter | Build a custom router |
| Serial execution | `SequentialAgent` | Enforce ordering via instructions alone |
| Parallel execution | `ParallelAgent` | Run agents manually in threads |
| Iterative loops | `LoopAgent` | Write custom loop logic |
| Code execution + math + charts | `code_executor=BuiltInCodeExecutor()` | Write `calculator.py` with hand-coded math functions |
| Agent callable as a typed tool | `AgentTool(agent=X, input_schema=..., output_schema=...)` | Wrap an agent with a custom dispatcher |
| Passing data between agents | `output_key="foo"` + `{foo?}` injection in the next agent's instruction | Manually copy state with custom helpers |
| Returning an image/file from an agent | `tool_context.save_artifact(name, Part.from_bytes(...))` | `print()` raw bytes or base64 strings |
| Loading an agent from YAML at runtime | `config_agent_utils.from_config(path)` (experimental — see gotchas) | Build your own YAML loader |
| Building a new agent from a generated spec | Either `from_config` (YAML path) or `Agent(**spec)` in Python | Generate + `exec()` Python source |
| Explicit plan-then-act reasoning | `planner=PlanReActPlanner()` or `BuiltInPlanner(thinking_config=...)` | Embed multi-paragraph "think step by step" blocks in instructions |

#### Use Built-in Tools

ADK provides ready-to-use tools in `google.adk.tools`:
- `google_search` — Google Search grounding (auto-invoked by Gemini)
- `BuiltInCodeExecutor` — sandboxed code execution

**Correct pattern (ADK built-in):**
```python
from google.adk.agents import Agent
from google.adk.tools.google_search_tool import google_search

root_agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    instruction="You answer questions using Google Search.",
    tools=[google_search],
)
```

**Anti-pattern (custom reimplementation):**
```python
import os
from google import genai
from google.genai import types

async def web_search(query: str) -> str:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Search for: {query}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return response.text
```

The built-in tool handles API keys, model routing, and search grounding
internally. The custom version adds ~20 lines of code that duplicate what ADK
already does.

#### Mixing Built-in Tools with Custom Tools

By default, ADK built-in tools like `google_search` cannot be combined with
custom function tools in the same agent. To mix them, use
`bypass_multi_tools_limit=True`:

```python
from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.tool_context import ToolContext

def save_note(topic: str, finding: str, tool_context: ToolContext) -> str:
    """Save a finding to the scratchpad."""
    if "notes" not in tool_context.state:
        tool_context.state["notes"] = []
    tool_context.state["notes"] = tool_context.state["notes"] + [
        {"topic": topic, "finding": finding}
    ]
    return "Saved."

root_agent = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Search the web and save key findings to your scratchpad.",
    tools=[
        GoogleSearchTool(bypass_multi_tools_limit=True),
        save_note,
    ],
)
```

This converts the built-in search into a function-calling tool that coexists
with custom tools. See `contributing/samples/built_in_multi_tools/` for the
official reference.

#### Use Workflow Agents for Flow Control

ADK provides workflow agents that enforce execution patterns architecturally
— don't rely solely on instructions to control flow:

- **`SequentialAgent`** — runs sub-agents one after another, in order
- **`ParallelAgent`** — runs sub-agents simultaneously
- **`LoopAgent`** — runs sub-agents in a loop until a condition is met

These are more reliable than instructing a single agent to "do step 1, then
step 2, then step 3" because the flow is enforced by the framework, not by
the LLM's interpretation of the instruction.

#### Insights and Gotchas

1.  **The official `google_search_agent` sample is 15 lines.** If your agent is
    significantly longer and does the same thing, you're probably reimplementing
    ADK features. Check `contributing/samples/` for reference implementations.

2.  **Built-in tools handle infrastructure you don't see.** `google_search`
    handles API key management, model version routing (Gemini 1.x vs 2.x), and
    search grounding type selection automatically. Custom reimplementations
    miss these edge cases.

3.  **`bypass_multi_tools_limit=True` unlocks composition.** When you need an
    agent to use BOTH built-in tools (like `google_search`) AND custom function
    tools (like a scratchpad), use `GoogleSearchTool(bypass_multi_tools_limit=
    True)`. Without this flag, Gemini restricts the agent to only one tool type.
    See `contributing/samples/built_in_multi_tools/` for the pattern.

4.  **`ToolContext.state` is your scratchpad.** For any agent that needs to
    accumulate context across tool calls, use `ToolContext.state` — don't build
    custom state management.

5.  **`sub_agents` is your orchestrator.** For multi-agent systems, use the
    `sub_agents` parameter on `Agent`. ADK handles routing based on agent
    `description` fields. Don't build custom routing logic.

6.  **Align agent levels with ADK features.** Each level in Google's agent
    taxonomy maps to specific ADK capabilities:
    - Level 1 (Connected): `Agent` + built-in tools
    - Level 2 (Strategic): `Agent` + `GoogleSearchTool(bypass_multi_tools_limit
      =True)` + custom tools + `ToolContext.state`
    - Level 3 (Multi-Agent): `Agent` + `sub_agents` (or `AgentTool` when the
      coordinator needs typed returns to compose)
    - Level 4 (Self-Evolving): L3 + a creator sub-agent that writes specs into
      `state.capabilities`, re-hydrated as `AgentTool`s by a
      `before_agent_callback` each turn

7.  **Custom tool functions are for business logic only.** Write custom tools
    when you need domain-specific logic (e.g., `save_research_note`). Don't
    write custom tools that wrap ADK built-ins.

8.  **Check `contributing/samples/` BEFORE writing any agent.** There are 100+
    official samples covering common patterns. Find the closest one and adapt
    it — don't start from scratch.

9.  **ADK is declarative — work with it.** The framework's power comes from
    configuration over code. If you're writing plumbing (API clients, state
    stores, routers, execution loops), you're fighting the framework. Step back
    and look for the ADK feature that does it for you.

10. **`AgentTool` returns text only — charts reach the user via the
    Artifacts sidebar, not inline chat.** When a wrapped agent uses
    `BuiltInCodeExecutor` + matplotlib to draw a chart,
    `AgentTool._run_async_impl` merges only `p.text` from the child's last
    content (`agent_tool.py:272-274`). `inline_data` image parts are
    **dropped** from the tool result — so `plt.show()` inside the sandbox
    does not produce an inline chart bubble when the producing agent is
    wrapped as an AgentTool. The compensating mechanism is Gemini's
    code_execution auto-save: any matplotlib figure produced in the
    sandbox is saved with a timestamp name (e.g. `YYYYMMDD_HHMMSS.png`)
    and flows through `ForwardingArtifactService` (`agent_tool.py:233`)
    into the parent session's artifact service, where `adk web` renders
    it in the **Artifacts** sidebar. The parent LLM never sees the chart.
    Note: `tool_context` is **not** accessible inside the sandbox —
    `tool_context.save_artifact(...)` in Python code executed by
    `BuiltInCodeExecutor` fails because the sandbox is server-side in
    Gemini and has no ADK namespace. To see the chart inline in chat,
    don't wrap the chart-producing agent as an AgentTool; make it a
    `sub_agent` so its events (including `inline_data`) land directly on
    the top-level event stream — but you lose coordinator orchestration
    after the transfer.

11. **`AgentTool` creates a fresh session per call — don't rely on child
    history.** Each invocation instantiates a new `InMemorySessionService`
    (`agent_tool.py:234`). The child sees only the args you pass and a
    *snapshot* of the parent's state (`agent_tool.py:240-249`). If the child
    needs context from prior turns, pass it explicitly in args or via shared
    state — don't assume the child can "remember" a previous conversation.
    State writes from the child stream back through `state_delta` events, so
    round-trip communication works.

12. **Don't wrap an agent that has `sub_agents` as an `AgentTool`.** An agent
    with both a parent routing to it via `AgentTool` *and* its own `sub_agents`
    has two valid delegation paths; the LLM silently picks one under pressure,
    usually the wrong one. Rule: wrap **leaf** agents (no `sub_agents`) as
    tools. Use `sub_agents` OR `AgentTool` on the same agent, not both.

13. **`output_key` + `{state_key?}` injection replaces custom save/get tools.**
    Setting `output_key="foo"` on an `LlmAgent` auto-writes its final text to
    `state.foo` (`llm_agent.py:859`). Reading it is `{foo?}` inside the next
    agent's instruction string — the `?` marks it optional so a missing key
    renders empty instead of raising. This pair replaces custom
    `save_finding`/`get_findings` tool pairs for single-value hand-offs. Use
    `ToolContext.state` + a custom tool only for list accumulation (multi-
    call append).

14. **`BuiltInCodeExecutor` is stateful and sandboxed — optimize around that.**
    Variables, DataFrames, and figures persist across turns within a session,
    so follow-up queries can reuse them. But the sandbox is Gemini-server-
    side: no `pip install`, no internet from inside, fixed library set (io,
    math, re, matplotlib, numpy, pandas, scipy). If the agent needs a package
    outside that set, the code won't run — either pick a different tool or
    switch to `ContainerCodeExecutor` (local, requires Docker) or a cloud
    executor. Requires Gemini 2.0+ (`built_in_code_executor.py:46-47`).

15. **Never use `UnsafeLocalCodeExecutor` for LLM-generated code.** It does
    `exec(code, globals_, globals_)` in a subprocess (`unsafe_local_code_
    executor.py:45`) with no filesystem, network, or import sandbox. Process
    isolation ≠ security isolation. For local-safe execution, use
    `ContainerCodeExecutor`; for zero-setup, use `BuiltInCodeExecutor`.

16. **`config_agent_utils.from_config` is `@experimental`.** The YAML-to-live-
    agent loader (`config_agent_utils.py:34`) is gated by
    `FeatureName.AGENT_CONFIG`. For production, consider building `Agent(**spec)`
    directly in Python from a validated dict — it's non-experimental, produces
    the same live object, and lets you keep the YAML as an audit-only artifact.
    The `level_4_agent` takes this route.

17. **Dynamic tools must rebuild from a fixed base, not accumulate.** When you
    add tools to a coordinator via `before_agent_callback` (e.g., re-hydrating
    runtime-created specialists from `state.capabilities`), do:
    `agent.tools = list(FIXED_TOOLS) + hydrate(state)` on every call — don't
    `agent.tools.append(...)`. Appending leaks tools from one session into the
    next because the `Agent` object is module-level. Rebuilding ensures session
    isolation. See `level_4_agent/agent.py:_rehydrate_runtime_tools`.

18. **Runtime agent creation has non-negotiable safety rules.** If a system
    creates agents at runtime (Level 4+), enforce: (a) a tool allowlist
    validated before `Agent(...)` construction; (b) never
    `UnsafeLocalCodeExecutor` — only sandboxed executors; (c) a HITL
    confirmation gate (prompt-level works; `tool_human_in_the_loop_config/`
    is the framework-level pattern); (d) name-based dedupe so a looping
    coordinator can't spawn unbounded copies. All four must be in place
    before demoing Level 4 behavior publicly.

19. **`AgentTool(skip_summarization=True)` suppresses final text emission
    — use it only for programmatic consumers.** With this flag, the
    coordinator's turn ends at the function_response; the tool's output
    does **not** become a visible text event in `adk web`. The output is
    still accessible via the Function Responses panel but it never
    renders as a chat bubble. Symptom: a wrapped agent whose purpose is
    to produce the final user-facing reply (e.g. a report writer) runs
    successfully, its output is present in the trace, but the chat
    stream ends with no assistant bubble. Fix: drop
    `skip_summarization=True` and add a prompt rule on the coordinator
    telling it to reproduce the writer's output verbatim. Costs one
    extra coordinator LLM call per turn; worth it when the user needs to
    see the text. Keep the flag when the tool's output is consumed
    programmatically (passed to another tool, parsed by the coordinator,
    etc.) — the original purpose of the flag is to prevent the
    coordinator from rewriting already-finalised text.

20. **Inside `BuiltInCodeExecutor`, `plt.show()` vs `plt.savefig(name)`
    have different UX semantics — pick one per chart, never mix.**
    `plt.show()` makes Gemini emit the figure as an `inline_data` Part,
    which renders inline in `adk web`'s chat stream — provided the
    chart-producing agent is **not** wrapped as an AgentTool (see gotcha
    #10). `plt.savefig("chart.png")` writes a named artifact visible in
    the Artifacts sidebar regardless of AgentTool wrapping, via
    `ForwardingArtifactService`. If you call both, or if a later code
    cell closes/re-saves the figure, Gemini will save a second blank
    version of the same artifact — the Artifacts tab then defaults to
    showing the blank version and the real chart hides under "Version:
    0". Instruction-level fix: require exactly one chart block ending in
    `plt.tight_layout(); plt.show()` with no subsequent `plt.*` calls.

21. **Gemini 2.5 Pro + `BuiltInCodeExecutor` + AFC can hang indefinitely
    on multi-step numeric+chart prompts.** Observed a 6-minute no-return
    with `AFC is enabled with max remote calls: 10` on a three-step
    prompt (calculate totals, compute percentages, produce a stacked bar
    chart). Gemini 2.5 Flash handles the same prompt in ~30s. **Rule:**
    put code-exec agents on Flash. Reserve Pro for meta-reasoning roles
    (coordinator routing, agent-creator spec design) where the model
    isn't going through AFC + code_execution rounds. If you must use Pro
    on a code-exec leaf, simplify the prompt to single-step and monitor
    wall time aggressively.

22. **MCP tool invocations appear in ADK traces as `build <tool_name>`
    spans — use the timing pattern to diagnose cold start vs upstream
    latency vs rate limiting.** Each `build` span is one
    `McpTool._run_async_impl()` → `session.call_tool()` round trip
    (`mcp_tool.py:373, 414`). The first call per session pays the
    stdio-subprocess cold-start cost (Node.js/`tsx` boot + MCP
    `initialize` handshake), typically 5–15s. Subsequent calls reuse the
    `MCPSessionManager`-pooled session and are usually <200ms. Three
    patterns to read from a trace: (a) first slow, rest fast → healthy
    cold start; (b) all slow → upstream server is slow (or the MCP
    server implements a rate limiter — check its source); (c) fast then
    slow → rate limiter kicked in between calls, not a bug in ADK. If
    you need to eliminate the cold-start visible delay, pre-warm the
    toolset from a `before_agent_callback` on the root agent before the
    user's first turn.

23. **`InMemoryMemoryService` / `InMemorySessionService` / `ForwardingArtifactService`
    are fresh per `AgentTool` call.** `AgentTool` constructs a new
    `Runner` with these services on every invocation (`agent_tool.py:230-235`).
    The child agent's artifact writes DO propagate to the parent session
    via `ForwardingArtifactService`, but its session events do **not** —
    the parent sees only the text summary via `p.text` merging (#10
    above). Consequence: you cannot rely on the child's event history
    across multiple `AgentTool` invocations in the same parent turn.
    For multi-turn continuity between a coordinator and a specialist,
    use `output_key` + `{state_key?}` injection (#13), not the child's
    session.

24. **A built-in tool (`google_search`, `BuiltInCodeExecutor`) on an
    agent that is registered as a `sub_agent` will fail with HTTP 400
    "Built-in tools and Function Calling cannot be combined in the same
    request."** When ADK transfers control into a `sub_agent`,
    `agent_transfer.py:44-66` auto-injects a `transfer_to_agent`
    function tool (so the sub-agent can transfer back to the parent or
    to peer sub-agents — see `_get_transfer_targets`,
    `agent_transfer.py:156-174`). At request time the sub-agent
    therefore has *both* a built-in tool and a function tool in its
    `tools=[...]` list, which Gemini's API forbids. There is no warning
    at agent-construction time; the failure surfaces only on the first
    real LLM call. **Two correct fixes, in order of preference:**
    (a) **`bypass_multi_tools_limit=True`** on the built-in tool's
    constructor — for `google_search` use
    `tools=[GoogleSearchTool(bypass_multi_tools_limit=True)]` instead
    of the default `google_search` singleton. The flag is documented at
    `src/google/adk/tools/google_search_tool.py:42-49` and exists for
    exactly this scenario. Same flag, same reason as `level_2_agent`'s
    use (which mixes `google_search` with explicit user-defined function
    tools). (b) **Wrap the built-in-using agent as `AgentTool`, not
    `sub_agent`.** `AgentTool`-wrapped agents do NOT get the
    `transfer_to_agent` auto-injection because they are not in the
    parent's `sub_agents` list; they appear in `tools` instead. This is
    the pattern `level_4_agent` and `level_4a_agent` use for their
    `data_fetcher_agent`. Pick (a) when the sub-agent genuinely needs
    bidirectional transfer affordances (peer/parent transfer); pick
    (b) when one-way "call and return text" is enough. **Do not** try
    to silence this by removing the built-in tool and re-implementing
    its behavior in a custom function tool — that's the anti-pattern
    flagged in the ADK Feature Map at `:267`. The whole point of
    `bypass_multi_tools_limit` is that this combination is supported,
    just gated behind an explicit opt-in.


## Development Setup

### Requirements

**Minimum requirements:**

- Python 3.10+ (**Python 3.11+ strongly recommended** for best performance)
- `uv` package manager (**required** - faster than pip/venv)

**Install uv if not already installed:** `bash curl -LsSf
https://astral.sh/uv/install.sh | sh`

### Setup Instructions

**Standard setup for development:** ```bash

# Create virtual environment with Python 3.11

uv venv --python "python3.11" ".venv" source .venv/bin/activate

# Install all dependencies for development

uv sync --all-extras ```

**Minimal setup for testing only (matches CI):** `bash uv sync --extra test`

**Virtual Environment Usage (Required):** - **Always use** `.venv/bin/python` or
`.venv/bin/pytest` directly - **Or activate** with `source .venv/bin/activate`
before running commands - **Never use** `python -m venv` - always create with
`uv venv` if missing

**Rationale:** `uv` is significantly faster and ensures consistent dependency
resolution across the team.

### Building

```bash
# Build wheel
uv build

# Install local build for testing
pip install dist/google_adk-<version>-py3-none-any.whl
```

### Running Agents Locally

**For interactive development and debugging:** ```bash

# Launch web UI (recommended for development)

adk web path/to/agents_dir ```

**For CLI-based testing:** ```bash

# Interactive CLI (prompts for user input)

adk run path/to/my_agent ```

**For API/production mode:** ```bash

# Start FastAPI server

adk api_server path/to/agents_dir ```

**For running evaluations:** ```bash

# Run evaluation set against agent

adk eval path/to/my_agent path/to/eval_set.json ```

## ADK: Style Guides

### Python Style Guide

The project follows the Google Python Style Guide. Key conventions are enforced
using `pylint` with the provided `pylintrc` configuration file. Here are some of
the key style points:

*   **Indentation**: 2 spaces.
*   **Line Length**: Maximum 80 characters.
*   **Naming Conventions**:
    *   `function_and_variable_names`: `snake_case`
    *   `ClassNames`: `CamelCase`
    *   `CONSTANTS`: `UPPERCASE_SNAKE_CASE`
*   **Docstrings**: Required for all public modules, functions, classes, and
    methods.
*   **Imports**: Organized and sorted.
*   **Error Handling**: Specific exceptions should be caught, not general ones
    like `Exception`.

### Autoformat (Required Before Committing)

**Always run** before committing code: `bash ./autoformat.sh`

**Manual formatting** (if needed): ```bash

# Format imports

isort src/ tests/ contributing/

# Format code style

pyink --config pyproject.toml src/ tests/ contributing/ ```

**Check formatting** without making changes: `bash pyink --check --diff --config
pyproject.toml src/ isort --check src/`

**Formatting Standards (Enforced by CI):** - **Formatter:** `pyink`
(Google-style Python formatter) - **Line length:** 80 characters maximum -
**Indentation:** 2 spaces (never tabs) - **Import sorter:** `isort` with Google
profile - **Linter:** `pylint` with Google Python Style Guide

**Rationale:** Consistent formatting eliminates style debates and makes code
reviews focus on logic rather than style.

### In ADK source

Below styles applies to the ADK source code (under `src/` folder of the GitHub
repo).

#### Use relative imports (Required)

```python
# DO - Use relative imports
from ..agents.llm_agent import LlmAgent

# DON'T - No absolute imports
from google.adk.agents.llm_agent import LlmAgent
```

**Rationale:** Relative imports make the code more maintainable and avoid
circular import issues in large codebases.

#### Import from module, not from `__init__.py` (Required)

```python
# DO - Import directly from module
from ..agents.llm_agent import LlmAgent

# DON'T - Import from __init__.py
from ..agents import LlmAgent
```

**Rationale:** Direct module imports make dependencies explicit and improve IDE
navigation and refactoring.

#### Always do `from __future__ import annotations` (Required)

**Rule:** Every source file must include `from __future__ import annotations`
immediately after the license header, before any other imports.

```python
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations  # REQUIRED - Always include this

# ... rest of imports ...
```

**Rationale:** This enables forward-referencing classes without quotes,
improving code readability and type hint support (PEP 563).

### In ADK tests

#### Use absolute imports (Required)

**Rule:** Test code must use absolute imports (`google.adk.*`) to match how
users import ADK.

```python
# DO - Use absolute imports
from google.adk.agents.llm_agent import LlmAgent

# DON'T - No relative imports in tests
from ..agents.llm_agent import LlmAgent
```

**Rationale:** Tests should exercise the same import paths that users will use,
catching issues with the public API.

## ADK: Local Testing

### Unit Tests

**Quick start:** Run all tests with: `bash pytest tests/unittests`

**Recommended:** Match CI configuration before submitting PRs: `bash uv sync
--extra test && pytest tests/unittests`

**Additional options:** ```bash

# Run tests in parallel for faster execution

pytest tests/unittests -n auto

# Run a specific test file during development

pytest tests/unittests/agents/test_llm_agent.py

```

### Testing Philosophy

**Use real code over mocks:** ADK tests should use real implementations as much
as possible instead of mocking. Only mock external dependencies like network
calls or cloud services.

**Test interface behavior, not implementation details:** Tests should verify
that the public API behaves correctly, not how it's implemented internally. This
makes tests resilient to refactoring and ensures the contract with users remains
intact.

**Test Requirements:** - Fast and isolated tests where possible - Use real ADK
components; mock only external dependencies (LLM APIs, cloud services, etc.) -
Focus on testing public interfaces and behavior, not internal implementation -
Descriptive test names that explain what behavior is being tested - High
coverage for new features, edge cases, and error conditions - Location:
`tests/unittests/` following source structure

## Docstring and comments

### Comments - Explaining the Why, Not the What

Philosophy: Well-written code should be largely self-documenting. Comments serve
a different purpose: they should explain the complex algorithms, non-obvious
business logic, or the rationale behind a particular implementation choice—the
things the code cannot express on its own. Avoid comments that merely restate
what the code does (e.g., # increment i above i += 1).

Style: Comments should be written as complete sentences. Block comments must
begin with a # followed by a single space.

## Versioning

ADK adherence to Semantic Versioning 2.0.0

Core Principle: The adk-python project strictly adheres to the Semantic
Versioning 2.0.0 specification. All release versions will follow the
MAJOR.MINOR.PATCH format.

### Breaking Change

A breaking change is any modification that introduces backward-incompatible
changes to the public API. In the context of the ADK, this means a change that
could force a developer using the framework to alter their existing code to
upgrade to the new version. The public API is not limited to just the Python
function and class signatures; it also encompasses data schemas for stored
information (like evaluation datasets), the command-line interface (CLI), and
the data format used for server communications.

### Public API Surface Definition

The "public API" of ADK is a broad contract that extends beyond its Python
function signatures. A breaking change in any of the following areas can disrupt
user workflows and the wider ecosystem of agents and tools built with ADK. The
analysis of the breaking changes introduced in v1.0.0 demonstrates the expansive
nature of this contract. For the purposes of versioning, the ADK Public API
Surface is defined as:

-   All public classes, methods, and functions in the google.adk namespace.

-   The names, required parameters, and expected behavior of all built-in Tools
    (e.g., google_search, BuiltInCodeExecutor).

-   The structure and schema of persisted data, including Session data, Memory,
    and Evaluation datasets.

-   The JSON request/response format of the ADK API server(FastAPI server) used
    by adk web, including field casing conventions.

-   The command-line interface (CLI) commands, arguments, and flags (e.g., adk
    deploy).

-   The expected file structure for agent definitions that are loaded by the
    framework (e.g., the agent.py convention).

#### Checklist for Breaking Changes:

The following changes are considered breaking and necessitate a MAJOR version
bump.

-   API Signature Change: Renaming, removing, or altering the required
    parameters of any public class, method, or function (e.g., the removal of
    the list_events method from BaseSessionService).

-   Architectural Shift: A fundamental change to a core component's behavior
    (e.g., making all service methods async, which requires consumers to use
    await).

-   Data Schema Change: A non-additive change to a persisted data schema that
    renders old data unreadable or invalid (e.g., the redesign of the
    MemoryService and evaluation dataset schemas).

-   Tool Interface Change: Renaming a built-in tool, changing its required
    parameters, or altering its fundamental purpose (e.g., replacing
    BuiltInCodeExecutionTool with BuiltInCodeExecutor and moving it from the
    tools parameter to the code_executor parameter of an Agent).

-   Configuration Change: Altering the required structure of configuration files
    or agent definition files that the framework loads (e.g., the simplification
    of the agent.py structure for MCPToolset).

-   Wire Format Change: Modifying the data format for API server interactions
    (e.g., the switch from snake_case to camelCase for all JSON payloads).

-   Dependency Removal: Removing support for a previously integrated third-party
    library or tool type.

## Commit Message Format (Required)

**All commits must** follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format.

**Format:** ``` <type>(<scope>): <description>

[optional body]

[optional footer] ```

**Common types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

**Examples:** ``` feat(agents): Add support for App pattern with plugins

fix(sessions): Prevent memory leak in session cleanup

refactor(tools): Unify environment variable enabled checks ```

**Rationale:** Conventional commits enable automated changelog generation and
version management.

## Key Files and Locations

Quick reference to important project files:

-   **Main config:** `pyproject.toml` (uses `flit_core` build backend)
-   **Dependencies:** `uv.lock` (managed by `uv`)
-   **Linting:** `pylintrc` (Google Python Style Guide)
-   **Auto-format:** `autoformat.sh` (runs isort + pyink)
-   **CLI entry point:** `src/google/adk/cli/cli_tools_click.py`
-   **Web UI backend:** `src/google/adk/cli/adk_web_server.py`
-   **Main exports:** `src/google/adk/__init__.py` (exports Agent, Runner)
-   **Examples:** `contributing/samples/` (100+ agent implementations)

## Additional Resources

- **Documentation:** https://google.github.io/adk-docs
- **Samples:** https://github.com/google/adk-samples
- **Architecture Details:** `contributing/adk_project_overview_and_architecture.md`
- **Contributing Guide:** `CONTRIBUTING.md`
- **LLM Context:** `llms.txt` (summarized), `llms-full.txt` (comprehensive)

## Python Tips

### General Python Best Practices

*   **Constants:** Use immutable global constant collections (tuple, frozenset, immutabledict) to avoid hard-to-find bugs. Prefer constants over wild string/int literals, especially for dictionary keys, pathnames, and enums.
*   **Naming:** Name mappings like `value_by_key` to enhance readability in lookups (e.g., `item = item_by_id[id]`).
*   **Readability:** Use f-strings for concise string formatting, but use lazy-evaluated `%`-based templates for logging. Use `repr()` or `pprint.pformat()` for human-readable debug messages. Use `_` as a separator in numeric literals to improve readability.
*   **Comprehensions:** Use list, set, and dict comprehensions for building collections concisely.
*   **Iteration:** Iterate directly over containers without indices. Use `enumerate()` when you need the index, `dict.items()` for keys and values, and `zip()` for parallel iteration.
*   **Built-ins:** Leverage built-in functions like `all()`, `any()`, `reversed()`, `sum()`, etc., to write more concise and efficient code.
*   **Flattening Lists:** Use `itertools.chain.from_iterable()` to flatten a list of lists efficiently without unnecessary copying.
*   **String Methods:** Use `startswith()` and `endswith()` with a tuple of strings to check for multiple prefixes or suffixes at once.
*   **Decorators:** Use decorators to add common functionality (like logging, timing, caching) to functions without modifying their core logic. Use `functools.wraps()` to preserve the original function's metadata.
*   **Context Managers:** Use `with` statements and context managers (from `contextlib` or custom classes with `__enter__`/`__exit__`) to ensure resources are properly initialized and torn down, even in the presence of exceptions.
*   **Else Clauses:** Utilize the `else` clause in `try/except` blocks (runs if no exception), and in `for/while` loops (runs if the loop completes without a `break`) to write more expressive and less error-prone code.
*   **Single Assignment:** Prefer single-assignment form (assign to a variable once) over assign-and-mutate to reduce bugs and improve readability. Use conditional expressions where appropriate.
*   **Equality vs. Identity:** Use `is` or `is not` for singleton comparisons (e.g., `None`, `True`, `False`). Use `==` for value comparison.
*   **Object Comparisons:** When implementing custom classes, be careful with `__eq__`. Return `NotImplemented` for unhandled types. Consider edge cases like subclasses and hashing. Prefer using `attrs` or `dataclasses` to handle this automatically.
*   **Hashing:** If objects are equal, their hashes must be equal. Ensure attributes used in `__hash__` are immutable. Disable hashing with `__hash__ = None` if custom `__eq__` is implemented without a proper `__hash__`.
*   **`__init__()` vs. `__new__()`:** `__new__()` creates the object, `__init__()` initializes it. For immutable types, modifications must happen in `__new__()`.
*   **Default Arguments:** NEVER use mutable default arguments. Use `None` as a sentinel value instead.
*   **`__add__()` vs. `__iadd__()`:** `x += y` (in-place add) can modify the object in-place if `__iadd__` is implemented (like for lists), while `x = x + y` creates a new object. This matters when multiple variables reference the same object.
*   **Properties:** Use `@property` to create getters and setters only when needed, maintaining a simple attribute access syntax. Avoid properties for computationally expensive operations or those that can fail.
*   **Modules for Namespacing:** Use modules as the primary mechanism for grouping and namespacing code elements, not classes. Avoid `@staticmethod` and methods that don't use `self`.
*   **Argument Passing:** Python is call-by-value, where the values are object references (pointers). Assignment binds a name to an object. Modifying a mutable object through one name affects all names bound to it.
*   **Keyword/Positional Arguments:** Use `*` to force keyword-only arguments and `/` to force positional-only arguments. This can prevent argument transposition errors and make APIs clearer, especially for functions with multiple arguments of the same type.
*   **Type Hinting:** Annotate code with types to improve readability, debuggability, and maintainability. Use abstract types from `collections.abc` for container annotations (e.g., `Sequence`, `Mapping`, `Iterable`). Annotate return values, including `None`. Choose the most appropriate abstract type for function arguments and return types.
*   **`NewType`:** Use `typing.NewType` to create distinct types from primitives (like `int` or `str`) to prevent argument transposition and improve type safety.
*   **`__repr__()` vs. `__str__()`:** Implement `__repr__()` for unambiguous, developer-focused string representations, ideally evaluable. Implement `__str__()` for human-readable output. `__str__()` defaults to `__repr__()`.
*   **F-string Debug:** Use `f"{expr=}"` for concise debug printing, showing both the expression and its value.

### Libraries and Tools

*   **`collections.Counter`:** Use for efficiently counting hashable objects in an iterable.
*   **`collections.defaultdict`:** Useful for avoiding key checks when initializing dictionary values, e.g., appending to lists.
*   **`heapq`:** Use `heapq.nlargest()` and `heapq.nsmallest()` for efficiently finding the top/bottom N items. Use `heapq.merge()` to merge multiple sorted iterables.
*   **`attrs` / `dataclasses`:** Use these libraries to easily define simple classes with boilerplate methods like `__init__`, `__repr__`, `__eq__`, etc., automatically generated.
*   **NumPy:** Use NumPy for efficient array computing, element-wise operations, math functions, filtering, and aggregations on numerical data.
*   **Pandas:** When constructing DataFrames row by row, append to a list of dicts and call `pd.DataFrame()` once to avoid inefficient copying. Use `TypedDict` or `dataclasses` for intermediate row data.
*   **Flags:** Use libraries like `argparse` or `click` for command-line flag parsing. Access flag values in a type-safe manner.
*   **Serialization:** For cross-language serialization, consider JSON (built-in), Protocol Buffers, or msgpack. For Python serialization with validation, use `pydantic` for runtime validation and automatic (de)serialization, or `cattrs` for performance-focused (de)serialization with `dataclasses` or `attrs`.
*   **Regular Expressions:** Use `re.VERBOSE` to make complex regexes more readable with whitespace and comments. Choose the right method (`re.search`, `re.fullmatch`). Avoid regexes for simple string checks (`in`, `startswith`, `endswith`). Compile regexes used multiple times with `re.compile()`.
*   **Caching:** Use `functools.lru_cache` with care. Prefer immutable return types. Be cautious when memoizing methods, as it can lead to memory leaks if the instance is part of the cache key; consider `functools.cached_property`.
*   **Pickle:** Avoid using `pickle` due to security risks and compatibility issues. Prefer JSON, Protocol Buffers, or msgpack for serialization.
*   **Multiprocessing:** Be aware of potential issues with `multiprocessing` on some platforms, especially concerning `fork`. Consider alternatives like threads (`concurrent.futures.ThreadPoolExecutor`) or `asyncio` for I/O-bound tasks.
*   **Debugging:** Use `IPython.embed()` or `pdb.set_trace()` to drop into an interactive shell for debugging. Use visual debuggers if available. Log with context, including inputs and exception info using `logging.exception()` or `exc_info=True`.
*   **Property-Based Testing & Fuzzing:** Use `hypothesis` for property-based testing that generates test cases automatically. For coverage-guided fuzzing, consider `atheris` or `python-afl`.

### Testing

*   **Assertions:** Use pytest's native `assert` statements with informative expressions. Pytest automatically provides detailed failure messages showing the values involved. Add custom messages with `assert condition, "helpful message"` when the expression alone isn't clear.
*   **Custom Assertions:** Write reusable helper functions (not methods) for repeated complex checks. Use `pytest.fail("message")` to explicitly fail a test with a custom message.
*   **Parameterized Tests:** Use `@pytest.mark.parametrize` to reduce duplication when running the same test logic with different inputs. This is more idiomatic than the `parameterized` library.
*   **Fixtures:** Use pytest fixtures (with `@pytest.fixture`) for test setup, teardown, and dependency injection. Fixtures are cleaner than class-based setup methods and can be easily shared across tests.
*   **Mocking:** Use `mock.create_autospec()` with `spec_set=True` to create mocks that match the original object's interface, preventing typos and API mismatch issues. Use context managers (`with mock.patch(...)`) to manage mock lifecycles and ensure patches are stopped. Prefer injecting dependencies via fixtures over patching.
*   **Asserting Mock Calls:** Use `mock.ANY` and other matchers for partial argument matching when asserting mock calls (e.g., `assert_called_once_with`).
*   **Temporary Files:** Use pytest's `tmp_path` and `tmp_path_factory` fixtures for creating isolated and automatically cleaned-up temporary files/directories. These are preferred over the `tempfile` module in pytest tests.
*   **Avoid Randomness:** Do not use random number generators to create inputs for unit tests. This leads to flaky, hard-to-debug tests. Instead, use deterministic, easy-to-reason-about inputs that cover specific behaviors.
*   **Test Invariants:** Focus tests on the invariant behaviors of public APIs, not implementation details.
*   **Test Organization:** Prefer simple test functions over class-based tests unless you need to share fixtures across multiple test methods in a class. Use descriptive test names that explain the behavior being tested.

### Error Handling

*   **Re-raising Exceptions:** Use a bare `raise` to re-raise the current exception, preserving the original stack trace. Use `raise NewException from original_exception` to chain exceptions, providing context. Use `raise NewException from None` to suppress the original exception's context.
*   **Exception Messages:** Always include a descriptive message when raising exceptions.
*   **Converting Exceptions to Strings:** `str(e)` can be uninformative. `repr(e)` is often better. For full details including tracebacks and chained exceptions, use functions from the `traceback` module (e.g., `traceback.format_exception(e)`, `traceback.format_exc()`).
*   **Terminating Programs:** Use `sys.exit()` for expected terminations. Uncaught non-`SystemExit` exceptions should signal bugs. Avoid functions that cause immediate, unclean exits like `os.abort()`.
*   **Returning None:** Be consistent. If a function can return a value, all paths should return a value (use `return None` explicitly). Bare `return` is only for early exit in conceptually void functions (annotated with `-> None`).
