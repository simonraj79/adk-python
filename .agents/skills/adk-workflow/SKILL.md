---
name: adk-workflow
description: Build ADK agents and workflows. Triggers on "build/create agent", "create workflow", "add nodes/edges", "conditional routing", "human-in-the-loop", "parallel workers", "fan-out/fan-in", "retry logic", "test agent", "task mode", "single-turn agent", "add tools", "MCP tools", "session state", or mentions Workflow, FunctionNode, LlmAgentWrapper, JoinNode, ParallelWorker, LlmAgent, Edge, RequestInput, mode='task', mode='single_turn'. Covers LLM agents with tools, graph-based workflows, state management, routing, parallel processing, HITL, and task delegation.
---

# ADK Agent Development

## Getting Started

For environment setup, API key configuration, basic LLM agent creation, tool definitions, running agents, and sample projects, consult **`references/getting-started.md`**.

Quick setup:

```bash
pip install google-adk        # Install
adk create my_agent           # Scaffold project
# Edit my_agent/.env with GOOGLE_API_KEY=...
# Edit my_agent/agent.py with agent definition
adk web my_agent/             # Run web UI at localhost:8000
```

Agent directory structure (required for CLI discovery):

```
my_agent/
├── __init__.py    # from . import agent
├── agent.py       # Must define root_agent
└── .env           # GOOGLE_API_KEY=... (not committed to git)
```

## Basic LLM Agent with Tools

```python
from google.adk import Agent

def get_weather(city: str) -> dict:
  """Returns current weather for a city."""
  return {"city": city, "weather": "sunny", "temp": "72F"}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction="You are a helpful assistant. Use get_weather for weather queries.",
    tools=[get_weather],
)
```

Tools are Python functions -- the name, docstring, and type hints become the tool schema the LLM sees. For all tool types (MCP, OpenAPI, Google API, built-in, BaseTool, BaseToolset), consult **`references/tool-catalog.md`**.

## Agent Modes: Chat, Task, and Single-Turn

Agents support three delegation modes via the `mode` parameter:

| Mode | Delegation Tool | User Interaction | Use Case |
|------|----------------|------------------|----------|
| `chat` (default) | `transfer_to_agent` | Full chat | General assistants |
| `task` | `request_task_{name}` | Multi-turn (can chat) | Structured I/O tasks |
| `single_turn` | `request_task_{name}` | None | Autonomous tasks |

### Task Mode (Structured Delegation)

```python
from google.adk import Agent
from pydantic import BaseModel

class ResearchInput(BaseModel):
  topic: str
  depth: str = 'standard'

class ResearchOutput(BaseModel):
  summary: str
  key_findings: str

researcher = Agent(
    name='researcher',
    mode='task',
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    instruction='Research the topic, then call finish_task with results.',
    description='Researches topics.',
    tools=[search_web],
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[researcher],
    instruction='Delegate research to the researcher using request_task_researcher.',
)
```

### Single-Turn Mode (Autonomous)

```python
summarizer = Agent(
    name='summarizer',
    mode='single_turn',
    output_schema=SummaryOutput,
    instruction='Summarize the content and call finish_task. No user interaction.',
    description='Summarizes documents autonomously.',
    tools=[extract_text],
)
```

For full task mode details, schemas, mixed-mode patterns, and the delegation lifecycle, consult **`references/task-mode.md`**.

## Workflow Agents

A `Workflow` extends the basic agent with graph-based execution. Instead of a single LLM deciding what to do, define explicit nodes and edges:

```python
from google.adk import Workflow

def greet(node_input: str) -> str:
  return f"Hello, {node_input}!"

root_agent = Workflow(
    name="greeter",
    edges=[('START', greet)],
)
```

### Workflow Schemas
You can define `input_schema` and `output_schema` on a `Workflow` (using Pydantic models).
- **`input_schema`**: The workflow will automatically parse the initial user message into this schema. The `START` node will output this model instance instead of `types.Content`.
- **`output_schema`**: Enforces that the final output of the workflow matches this schema.

### Core Concepts

A workflow has three building blocks:

1. **Nodes** -- units of work (functions, LLM agents, tools)
2. **Edges** -- connections between nodes, optionally with route conditions
3. **START** -- the built-in entry point that receives user input

## Node Types

Any "NodeLike" is accepted in edges and auto-wrapped:

| Python Object | Wrapped As | Default rerun_on_resume |
|--------------|-----------|------------------------|
| Function/callable | `FunctionNode` | `False` |
| `LlmAgent` (core) | Auto-wrapped | `True` |
| Other `BaseAgent` | `AgentNode` | `False` |
| `BaseTool` | `ToolNode` | `False` |
| `BaseNode` subclass | Used as-is | Per subclass |

## Function Nodes

Functions are the most common node type. Parameter resolution:

| Parameter | Source |
|-----------|--------|
| `ctx` | Workflow `Context` object |
| `node_input` | Output from predecessor node |
| Any other name | `ctx.state[param_name]` |

```python
from google.adk import Context

def process(ctx: Context, node_input: Any, user_name: str) -> str:
  # node_input = predecessor output; user_name = ctx.state['user_name']
  # NOTE: START node outputs types.Content (not str) unless input_schema is set
  return f"{user_name}: {node_input}"

### Dynamic Node Execution
You can run nodes dynamically using `await ctx.run_node(node, node_input=...)`. To treat the output of the dynamically run node as the output of the current node, set `use_as_output=True`.

> [!NOTE]
> **Alternative Construction Style**: You can also use dynamic nodes to build workflows in an imperative style (using standard Python control flow) as an alternative to static graph edges. See [Dynamic Nodes Reference](file:///Users/deanchen/Desktop/adk-workflow/.agents/skills/adk-workflow/references/dynamic-nodes.md) for details.

```

Return `None` to suppress downstream triggering. Return an `Event` for routing or state updates:

```python
from google.adk import Event

def classify(node_input: str):
  if "urgent" in node_input:
    return Event(output=node_input, route="urgent")
  return Event(output=node_input, route="normal", state={"processed": True})
```

## Edge Patterns

**Prefer sequence shorthand tuples and dict routing** — these are the idiomatic patterns used in all samples:

```python
# Sequence shorthand (PREFERRED — tuple creates chain)
edges = [("START", step_a, step_b, step_c)]
# Equivalent to: [("START", step_a), (step_a, step_b), (step_b, step_c)]

# Routing map (PREFERRED — dict maps routes to targets)
edges = [
    ("START", process_input, classify_input, route_on_category),
    (route_on_category, {"question": answer, "statement": comment, "other": handle_other}),
]

# Fan-out + JoinNode (all in one tuple)
from google.adk.workflow import JoinNode
join = JoinNode(name="merge")
edges = [("START", (branch_a, branch_b, branch_c), join, aggregate)]

# Fan-out + multi-trigger (no JoinNode — downstream fires per branch)
edges = [("START", (branch_a, branch_b, branch_c), aggregate)]

# Self-loop (node routes back to itself)
edges = [
    ("START", validate, guess_number),
    (guess_number, {"guessed_wrong": guess_number}),
]

# Loop with revision (combine shorthand + dict routing)
edges = [
    ("START", process_input, draft_email, human_review),
    (human_review, {"revise": draft_email, "approved": send, "rejected": discard}),
]

# Fallback route
(classifier, {"success": handler_a, '__DEFAULT__': fallback_handler})
```

## Data Flow: State vs node_input

**For workflows with LLM agents, prefer state-based data flow.** Store data in state via `Event(state={...})` and reference it in LLM instructions via `{var}` templates. This is more robust than threading data through `node_input`, especially when multiple branches or loops need the same data.

```python
from google.adk import Agent, Event, Workflow

def process_input(node_input: str):
  """Store user input in state. LLM agents downstream read from state, not node_input."""
  yield Event(state={"complaint": node_input, "feedback": ""})

draft_email = Agent(
    name="draft_email",
    instruction='Write a response to: "{complaint}". Feedback: {feedback?}',
    output_key="draft",  # Stores LLM output in state["draft"]
)

def send_email(draft: str):
  """Read draft from state via parameter name resolution."""
  yield Event(message="Email sent!")
```

**Key patterns:**
- **`Event(state={...})` without output** — updates state but downstream nodes receive `None` as `node_input`. If you need both state update AND data flow, yield two events: `Event(state={...})` then `Event(output=value)`
- **`output_key="draft"`** — stores LLM text output in `state["draft"]`; downstream reads via param name `draft`
- **`{var}` / `{var?}`** — instruction templates resolved from state (`?` = optional, empty string if missing)
- **`def func(draft: str)`** — parameter name `draft` auto-resolves from `ctx.state["draft"]`

## LLM Agent Nodes

Place `Agent` instances directly in workflow edges. They are auto-run as nodes (via `run_llm_agent_as_node`). The wrapper defaults to `single_turn` mode (isolated, no session history); set `mode="task"` on the Agent for multi-turn HITL within the node. LLM agents receive `node_input` from predecessors and pass output to downstream nodes — they work like any other node in the graph:

```python
from google.adk import Agent, Workflow
from typing import Literal
from pydantic import BaseModel, Field

class InputCategory(BaseModel):
  category: Literal["question", "statement", "other"]

# Agent receives node_input as user message, outputs structured dict downstream
classify_input = Agent(
    name="classify_input",
    instruction='Classify the user input into one of the categories.',
    output_schema=InputCategory,  # Structured output for routing
    output_key="category",        # Also store in state
)

# Agent without output_schema — outputs types.Content downstream
summarizer = Agent(
    name="summarizer",
    instruction="Summarize the following text in one sentence.",
    output_key="summary",         # Stores raw text in state
)
```

**LLM agent output types:**

| Config | `node_input` for next node | `state[output_key]` |
|--------|---------------------------|---------------------|
| No `output_schema` | `types.Content` | raw text string |
| With `output_schema` | `dict` | dict |

**How LLM agents receive input — `node_input` vs state:**

LLM agents get input two ways: `node_input` (predecessor output, becomes the user message the LLM sees) and state (`{var}` templates in the instruction resolve **only** from `ctx.state`, never from `node_input`). To use predecessor data in instruction templates, first store it in state via `Event(state={...})` or `output_key`:

| Scenario | Use `node_input` | Use state `{var}` |
|----------|-----------------|-------------------|
| Simple pipeline (A → B → C) | ✅ Each agent processes predecessor's output | Optional for extra context |
| Fan-out / routing | ✅ All branches receive same input | ✅ Store shared data early |
| Revision loops | Predecessor output changes each iteration | ✅ Stable context (feedback, original request) |
| Multiple data sources needed | Only one predecessor's output | ✅ Read multiple state keys |

```python
# node_input only: simple pipeline, agent processes what predecessor outputs
summarizer = Agent(
    name="summarizer",
    instruction="Summarize the following text in one sentence.",
)

# State only: agent reads all context from state, ignores node_input
draft_email = Agent(
    name="draft_email",
    instruction='Write a response to: "{complaint}". Feedback: {feedback?}',
    output_key="draft",
)

# Both: node_input for primary data, state for context
reviewer = Agent(
    name="reviewer",
    instruction='Review this draft. Original request was: "{original_request}".',
    output_schema=ReviewResult,
)
```

**When to use `output_schema`:**
- **Required** when downstream function nodes need structured dict data, or when feeding into JoinNode (serialization)
- **Required** for classification/routing schemas — use `Literal` types to constrain LLM output
- **Not needed** for terminal agents (user-facing text output) or text-passing agents — use `output_key` alone

**When to use `output_key`:**
- Stores the agent's output in `state[key]` for downstream access via `{key}` templates or param injection
- With `output_schema`: stores a dict. Without: stores the raw text string.

For tools, callbacks, and advanced LLM configuration, consult **`references/llm-agent-nodes.md`**.

## Parallel Processing

**Two distinct patterns — do not confuse them:**

**Fan-out (edge syntax)** — run different nodes in parallel on the same input:
```python
# Each reviewer runs in parallel on the same draft
edges = [("START", store_draft, (reviewer_a, reviewer_b, reviewer_c), join)]
```

**ParallelWorker (node flag)** — split a LIST into items, process each concurrently. The predecessor must output a `list`; each item is processed by a cloned worker. Output is a `list` of results:
```python
from google.adk.workflow import node

@node(parallel_worker=True)
def process_item(node_input: int) -> int:
  return node_input * 2
# Input: [1, 2, 3] -> Output: [2, 4, 6]

# On Agent: set parallel_worker=True directly
# Predecessor must output list (e.g., output_schema=list[str])
analyze_topic = Agent(
    name="analyze_topic",
    instruction="Analyze this sub-topic in depth.",
    output_schema=TopicAnalysis,
    parallel_worker=True,
)
# Input: ["solar", "wind", "hydro"] -> Output: [TopicAnalysis, TopicAnalysis, TopicAnalysis]
```

**Do NOT use `parallel_worker=True` on fan-out nodes.** Fan-out edges already run nodes in parallel. Adding `parallel_worker=True` makes the node expect a list input and iterate over it — if it receives a single value or None, it produces no output and the JoinNode gets nothing.

## Workflow Branching

ADK tracks execution branches in workflows to manage context and history separation, especially during parallel execution.

**Key Rules:**
1. **Sequential Propagation**: When node A completes and triggers node B, node B inherits node A's branch.
2. **Conditional Parallel Segments**: A segment `.node_name@run_id` is appended to the branch ONLY when parallelism occurs (i.e., when a node triggers multiple downstream nodes in parallel).
3. **JoinNode Common Prefix**: When a `JoinNode` aggregates multiple parallel paths, its final output event uses a branch that is the common dot-separated prefix of all incoming nodes' branches. If there is no common prefix, it uses an empty string `""`.

This ensures that events are tagged with the correct branch, allowing UI and logs to separate parallel execution paths.

## Human-in-the-Loop

Pause execution and request user input:

```python
from google.adk.events import RequestInput

# Yield from a generator
async def approval_gate(ctx: Context, node_input: str):
  yield RequestInput(
      message="Approve this action?",
      response_schema={"type": "string"},
  )

# Or return directly from a regular function
def evaluate_request(node_input: dict):
  if node_input["auto_approve"]:
    return "Approved automatically"
  return RequestInput(
      message="Please review this request.",
      response_schema=ApprovalDecision,  # Pydantic class
  )
```

HITL works in two modes:

**Resumable mode** (recommended for multi-step HITL): Export an `App` with resumability. The workflow checkpoints state and resumes at the interrupted node.

```python
from google.adk.apps.app import App, ResumabilityConfig

app = App(
    name="my_app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

**Non-resumable mode** (simpler, no App needed): The workflow replays from START on each user response, reconstructing state from session events. Works automatically for simple HITL but replays all nodes up to the interrupt point.

When `rerun_on_resume=False` (default for FunctionNode), the user's response becomes the node's output. When `rerun_on_resume=True`, the node reruns with `ctx.resume_inputs` populated.

**Node-Level Auth:** You can also configure `AuthConfig` on a `@node` (FunctionNode) to pause the workflow and request specific credentials (like an API key) from the user. For details, consult **`references/human-in-the-loop.md`**.

## Retry Configuration

```python
from google.adk.workflow import RetryConfig
from google.adk.workflow import FunctionNode

node = FunctionNode(
    flaky_call,
    retry_config=RetryConfig(max_attempts=3, initial_delay=1.0, backoff_factor=2.0),
)

Inside the node, you can access `ctx.attempt_count` to know the current attempt number (starts at 1).
```

## Agent Directory Convention

For CLI discovery (`adk web`, `adk run`):

```
my_workflow/
  __init__.py    # from . import agent
  agent.py       # root_agent = Workflow(...)
```

Every `agent.py` should include a module docstring describing what the agent does and sample queries for testing:

```python
"""Smart Briefing Generator.

Generates executive briefings by researching multiple angles of a topic
in parallel, writing a synthesis, and iterating with a reviewer.

Sample queries:
  - "quantum computing"
  - "the future of remote work"
  - "CRISPR gene editing"
"""
```

## Testing Agents

Test agents interactively or via automated queries with `adk run`, or use the web UI:

```bash
# Interactive CLI (reads from stdin)
adk run path/to/my_agent

# Automated query mode (single-turn)
adk run path/to/my_agent "my test input"

# Machine-consumable JSONL output (strips noise)
adk run --jsonl path/to/my_agent "my test input"

# Pipe input for non-interactive testing (legacy)
echo "my test input" | adk run path/to/my_agent

# Web UI at localhost:8000
adk web path/to/agents_dir
```

> [!TIP]
> **The "Write and Test" Workflow**: You can use automated query mode to test your agent immediately after editing code. For structured inputs, pass a JSON string:
> `adk run path/to/agent '{"field": "value"}'`
> This allows you to verify logic and catch bugs (like loop state issues) instantly without human intervention.



### When to use automated query mode

- **No Server Needed**: Fast testing without starting `adk web` server.
- **CI/CD & Automation**: Perfect for non-interactive regression tests.
- **Noise Reduction**: `--jsonl` strips empty payloads for machine parsing.
- **Concurrency**: Use with `--in_memory` for multi-threaded testing (isolates session state).

> [!TIP]
> Always read the sample's `README.md` first to understand expected inputs and behaviors!

### Exit Codes & Details

- **Exit Code 0**: Success.
- **Exit Code 1**: Error (e.g., API key missing, agent load failure).
- **Exit Code 2**: Paused (Workflow is waiting for human input/HITL).

For more options and flags, run:
```bash
adk run --help
```

Use `adk run` to verify agents work end-to-end before deploying. It requires a configured API key (via `.env` or environment variables).

## Best Practices (MUST FOLLOW)

Detailed best practices and critical rules are documented in a separate file to keep this summary concise.

Key rules include:
- **Use Pydantic Models**: Always define `BaseModel` for schemas.
- **Emit Content Events**: Use `message=` for UI rendering.
- **Prefer State-Based Data Flow**: Use `{var}` templates and `output_key`.
- **One Output Event Per Node**: Do not yield multiple outputs.
- **Don't Mix yield and return**: Use one style per function.

For the full detailed rules and examples, consult **`references/best-practices.md`**.

## Additional Resources

### Reference Files

For detailed patterns and techniques, consult:

- **`references/getting-started.md`** -- Environment setup, API keys, basic LLM agents with tools, running agents, complete sample projects
- **`references/tool-catalog.md`** -- All tool types: function tools, MCP, OpenAPI, Google API, built-in tools, BaseTool, BaseToolset, ToolContext, LongRunningFunctionTool
- **`references/task-mode.md`** -- Task delegation: mode='task', mode='single_turn', input/output schemas, request_task, finish_task, mixed-mode patterns
- **`references/multi-agent.md`** -- Multi-agent patterns: chat transfer, SequentialAgent, ParallelAgent, LoopAgent, model configuration
- **`references/session-and-state.md`** -- Session state, artifacts, memory services, state key conventions
- **`references/callbacks-and-plugins.md`** -- Callback types and signatures (note: callbacks and plugins are not well supported in Workflow agents yet; they work with standard LlmAgent)
- **`references/function-nodes.md`** -- FunctionNode details, @node decorator, generators, auto type conversion
- **`references/routing-and-conditions.md`** -- Conditional branching, dynamic routing, loops, multi-route fan-out
- **`references/state-and-events.md`** -- Context API, shared state, Event fields, intermediate content
- **`references/llm-agent-nodes.md`** -- LlmAgentWrapper, instructions, tools, all callback types, output schemas
- **`references/human-in-the-loop.md`** -- RequestInput, resume behavior, multi-step HITL, resumability config
- **`references/parallel-and-fanout.md`** -- ParallelWorker, JoinNode, fan-out/fan-in, diamond pattern, SequentialAgent/ParallelAgent
- **`references/advanced-patterns.md`** -- Nested workflows, retry config, custom BaseNode, ToolNode, AgentNode, graph validation
- **`references/dynamic-nodes.md`** -- Dynamic node scheduling at runtime via `ctx.run_node()`, rules and constraints
- **`references/testing.md`** -- pytest patterns, MockModel, InMemoryRunner, testing utilities
- **`references/import-paths.md`** -- Quick-reference import table for all ADK components
- **`references/best-practices.md`** -- Critical rules, Pydantic use, Event rules, and data flow guidelines
