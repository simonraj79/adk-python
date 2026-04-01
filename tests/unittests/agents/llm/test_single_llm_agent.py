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

"""End-to-end tests for _SingleLlmAgent pipeline behavior.

Tests exercise _SingleLlmAgent as a complete workflow agent via
InMemoryRunner backed by a MockModel. Covers construction, multi-turn
conversations, instruction resolution, generate_content_config
propagation, tool callbacks, resumability, and full pipeline behavior
for both CallLlmNode and ExecuteToolsNode together.

Note: canonical_* property tests are in test_single_llm_agent_fields.py.
Note: Isolated node-level unit tests are in test_call_llm_node.py and
      test_execute_tools_node.py.
"""

from __future__ import annotations

from typing import cast

from google.adk.agents.llm._call_llm_node import call_llm
from google.adk.agents.llm._execute_tools_node import execute_tools
from google.adk.agents.llm._single_llm_agent import _SingleLlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._dynamic_node_registry import dynamic_node_registry
from google.adk.workflow._node import node
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

from tests.unittests.workflow import testing_utils
from tests.unittests.workflow import workflow_testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(mock_model, tools=None, **kwargs):
  """Create a _SingleLlmAgent wired to a MockModel."""
  return _SingleLlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **kwargs,
  )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
  """Tests for _SingleLlmAgent graph construction."""

  def test_graph_edges_wired_correctly(self):
    """model_post_init creates call_llm <-> execute_tools edges."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model)

    assert len(agent.edges) == 3

    # Edge 0: START -> call_llm
    assert agent.edges[0][0] is START
    assert agent.edges[0][1] is call_llm

    # Edge 1: call_llm -> {execute_tools: execute_tools_node}
    assert agent.edges[1][0] is call_llm
    routing_map = agent.edges[1][1]
    assert isinstance(routing_map, dict)
    assert 'execute_tools' in routing_map
    execute_tools_node = routing_map['execute_tools']
    assert isinstance(execute_tools_node, FunctionNode)
    assert execute_tools_node._func is execute_tools
    assert execute_tools_node.rerun_on_resume is True

    # Edge 2: execute_tools_node -> {continue: call_llm}
    assert agent.edges[2][0] is execute_tools_node
    routing_map2 = agent.edges[2][1]
    assert isinstance(routing_map2, dict)
    assert call_llm in routing_map2.values()

  def test_agent_name_propagated(self):
    """Agent name is accessible on the constructed agent."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model)
    assert agent.name == 'test_agent'

  def test_tools_stored_on_agent(self):
    """Tools passed at construction are stored on the agent."""

    def my_tool() -> str:
      """A tool."""
      return 'result'

    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, tools=[my_tool])
    assert len(agent.tools) == 1


# ---------------------------------------------------------------------------
# Multi-turn conversation tests
# ---------------------------------------------------------------------------


class TestMultiTurn:
  """Tests for multi-turn conversation behavior."""

  @pytest.mark.asyncio
  async def test_multi_turn_session_persists(self):
    """Session history from turn 1 is fed to LLM in turn 2."""
    mock_model = testing_utils.MockModel.create(
        responses=['Reply to turn 1.', 'Reply to turn 2.']
    )
    agent = _make_agent(mock_model)
    runner = testing_utils.InMemoryRunner(agent)

    # Turn 1
    events1 = runner.run('Hello')
    simplified1 = testing_utils.simplify_events(events1)
    assert simplified1 == [('test_agent', 'Reply to turn 1.')]

    # Turn 2 (same runner/session)
    events2 = runner.run('Follow up')
    simplified2 = testing_utils.simplify_events(events2)
    assert simplified2 == [('test_agent', 'Reply to turn 2.')]

    # The second LLM request should have history from turn 1
    assert len(mock_model.requests) == 2
    second_request = mock_model.requests[1]
    assert len(second_request.contents) > 1

  @pytest.mark.asyncio
  async def test_multi_turn_with_tools(self):
    """Tool calls in turn 1 don't break turn 2."""

    def greet(name: str) -> str:
      """Greet someone."""
      return f'Hello, {name}!'

    fc = types.Part.from_function_call(name='greet', args={'name': 'Alice'})
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'Greeted Alice.', 'Turn 2 response.']
    )
    agent = _make_agent(mock_model, tools=[greet])
    runner = testing_utils.InMemoryRunner(agent)

    # Turn 1: tool call
    events1 = runner.run('Greet Alice')
    simplified1 = testing_utils.simplify_events(events1)
    assert len(simplified1) == 3  # FC, FR, Text

    # Turn 2: plain text
    events2 = runner.run('What did you do?')
    simplified2 = testing_utils.simplify_events(events2)
    assert simplified2 == [('test_agent', 'Turn 2 response.')]


# ---------------------------------------------------------------------------
# Instruction and config tests
# ---------------------------------------------------------------------------


class TestInstructionAndConfig:
  """Tests for instruction resolution and config propagation."""

  @pytest.mark.asyncio
  async def test_static_instruction_in_llm_request(self):
    """Static instruction string is included in the LLM request."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(
        mock_model,
        instruction='You are a helpful assistant.',
    )
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('test')

    assert len(mock_model.requests) == 1
    si = mock_model.requests[0].config.system_instruction
    assert si is not None
    # The system instruction should contain our instruction text
    instruction_text = si if isinstance(si, str) else si.parts[0].text
    assert 'helpful assistant' in instruction_text

  @pytest.mark.asyncio
  async def test_dynamic_instruction_callable(self):
    """Callable instruction is resolved and included in LLM request."""

    def instruction_fn(ctx):
      return 'Dynamic instruction from callable.'

    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(
        mock_model,
        instruction=instruction_fn,
    )
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('test')

    assert len(mock_model.requests) == 1
    si = mock_model.requests[0].config.system_instruction
    assert si is not None
    instruction_text = si if isinstance(si, str) else si.parts[0].text
    assert 'Dynamic instruction from callable' in instruction_text

  @pytest.mark.asyncio
  async def test_generate_content_config_temperature(self):
    """generate_content_config temperature is propagated to LLM request."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(
        mock_model,
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('test')

    assert len(mock_model.requests) == 1
    config = mock_model.requests[0].config
    assert config.temperature == 0.1


class TestParallelWorker:
  """Tests for the parallel_worker support."""

  @pytest.mark.asyncio
  async def test_single_llm_agent_with_parallel_worker(
      self, request: pytest.FixtureRequest
  ):
    """Tests that a _SingleLlmAgent can act as a parallel worker."""
    dynamic_node_registry.clear()

    async def producer_func() -> list[str]:
      return ['item1', 'item2']

    async def process_item(item: str) -> str:
      return f'{item}_processed'

    # Provide enough responses since both workers will poll from the same MockModel
    # and due to asyncio concurrency, one might consume multiple before the other finishes
    mock_model = testing_utils.MockModel.create(
        responses=[
            'processed',
            'processed',
        ]
    )

    nested_agent = node(
        _SingleLlmAgent(
            name='llm_agent',
            model=mock_model,
            tools=[process_item],
        ),
        parallel_worker=True,
    )

    outer_agent = Workflow(
        name='outer_agent',
        edges=[
            (START, producer_func),
            (producer_func, nested_agent),
        ],
    )

    runner = testing_utils.InMemoryRunner(outer_agent)
    events = runner.run('start')

    simplified_events = workflow_testing_utils.simplify_events_with_node(events)

    assert simplified_events == [
        (
            'outer_agent',
            {
                'node_name': 'producer_func',
                'output': ['item1', 'item2'],
            },
        ),
        # Children outputs
        (
            'llm_agent__0',
            'processed',
        ),
        (
            'llm_agent__1',
            'processed',
        ),
        # Parent output
        # Since the outputs are text & no output_schema was specified,
        # the LLM Agent node output data is None.
        (
            'outer_agent',
            {
                'node_name': 'llm_agent',
                'output': [
                    None,
                    None,
                ],
            },
        ),
    ]

  @pytest.mark.asyncio
  async def test_single_llm_agent_with_parallel_worker_as_flag(
      self, request: pytest.FixtureRequest
  ):
    """Tests that a _SingleLlmAgent can be configured with flag parallel_worker=True."""

    async def producer_func() -> list[str]:
      return ['item1', 'item2']

    # Provide enough responses since both workers will poll from the same MockModel
    # and due to asyncio concurrency, one might consume multiple before the other finishes
    mock_model = testing_utils.MockModel.create(
        responses=[
            'processed',
            'processed',
        ]
    )

    dynamic_node_registry.clear()

    nested_agent = _SingleLlmAgent(
        name='llm_agent',
        model=mock_model,
        parallel_worker=True,
    )

    outer_agent = Workflow(
        name='outer_agent',
        edges=[
            (START, producer_func),
            (producer_func, nested_agent),
        ],
    )

    runner = testing_utils.InMemoryRunner(outer_agent)
    events = runner.run('start')
    simplified_events = workflow_testing_utils.simplify_events_with_node(events)

    assert simplified_events == [
        (
            'outer_agent',
            {
                'node_name': 'producer_func',
                'output': ['item1', 'item2'],
            },
        ),
        # Children outputs
        (
            'llm_agent__0',
            'processed',
        ),
        (
            'llm_agent__1',
            'processed',
        ),
        # Parent output
        # Since the outputs are text & no output_schema was specified,
        # the LLM Agent node output data is None.
        (
            'outer_agent',
            {
                'node_name': 'llm_agent',
                'output': [
                    None,
                    None,
                ],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool callback tests
# ---------------------------------------------------------------------------


class TestToolCallbacks:
  """Tests for before_tool_callback and after_tool_callback."""

  @pytest.mark.asyncio
  async def test_before_tool_callback_modifies_args(self):
    """before_tool_callback can modify tool arguments."""

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    def before_tool_cb(tool, args, tool_context):
      # Override y to always be 10
      args['y'] = 10
      return None

    fc = types.Part.from_function_call(name='multiply', args={'x': 3, 'y': 5})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Result.'])
    agent = _make_agent(
        mock_model,
        tools=[multiply],
        before_tool_callback=before_tool_cb,
    )
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Multiply')
    simplified = testing_utils.simplify_events(events)

    # The tool should have been called with y=10 (not y=5)
    assert simplified[1][1].function_response.response['result'] == 30

  @pytest.mark.asyncio
  async def test_before_tool_callback_short_circuits(self):
    """before_tool_callback returning a dict short-circuits the tool."""
    call_count = 0

    def my_tool(x: int) -> str:
      """A tool."""
      nonlocal call_count
      call_count += 1
      return f'real result {x}'

    def before_tool_cb(tool, args, tool_context):
      return {'result': 'intercepted'}

    fc = types.Part.from_function_call(name='my_tool', args={'x': 42})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Done.'])
    agent = _make_agent(
        mock_model,
        tools=[my_tool],
        before_tool_callback=before_tool_cb,
    )
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('test')
    simplified = testing_utils.simplify_events(events)

    # Tool was NOT actually called
    assert call_count == 0
    # But a function response was still produced
    assert simplified[1][1].function_response.response == {
        'result': 'intercepted'
    }

  @pytest.mark.asyncio
  async def test_after_tool_callback_modifies_result(self):
    """after_tool_callback can modify the tool result."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def after_tool_cb(tool, args, tool_context, tool_response):
      # tool_response is the raw return value from the tool
      return {'result': tool_response * 100}

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Done.'])
    agent = _make_agent(
        mock_model,
        tools=[add],
        after_tool_callback=after_tool_cb,
    )
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('test')
    simplified = testing_utils.simplify_events(events)

    # The result should be modified by after_tool_callback
    assert simplified[1][1].function_response.response == {'result': 300}


# ---------------------------------------------------------------------------
# Resumability tests
# ---------------------------------------------------------------------------


class TestResumability:
  """Tests for state checkpointing with resumability."""

  @pytest.mark.asyncio
  async def test_resumable_app_text_response(self):
    """Text-only response works with resumable App."""
    mock_model = testing_utils.MockModel.create(
        responses=['Hello from resumable app.']
    )
    agent = _make_agent(mock_model)
    app = App(
        name='resumable_test',
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=app)

    events = runner.run('Hi')
    simplified = testing_utils.simplify_events(events)

    assert len(simplified) == 1
    assert simplified[0] == ('test_agent', 'Hello from resumable app.')

  @pytest.mark.asyncio
  async def test_resumable_app_with_tool_call(self):
    """Tool call + response works with resumable App (no serialization error)."""

    def square(x: int) -> int:
      """Square a number."""
      return x * x

    fc = types.Part.from_function_call(name='square', args={'x': 7})
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'The square of 7 is 49.']
    )
    agent = _make_agent(mock_model, tools=[square])
    app = App(
        name='resumable_tool_test',
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=app)

    # Should not raise PydanticSerializationError
    events = runner.run('Square 7')
    simplified = testing_utils.simplify_events(events)

    assert len(simplified) == 3  # FC, FR, Text
    assert simplified[2] == ('test_agent', 'The square of 7 is 49.')


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
  """Tests for edge cases and error handling."""

  @pytest.mark.asyncio
  async def test_empty_tools_list(self):
    """Agent with no tools works for text-only responses."""
    mock_model = testing_utils.MockModel.create(responses=['No tools needed.'])
    agent = _make_agent(mock_model)
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Hello')
    simplified = testing_utils.simplify_events(events)

    assert simplified == [('test_agent', 'No tools needed.')]

  @pytest.mark.asyncio
  async def test_tool_returning_dict(self):
    """Tool returning a dict is properly serialized as function response."""

    def get_info() -> dict:
      """Get info."""
      return {'name': 'Alice', 'age': 30}

    fc = types.Part.from_function_call(name='get_info', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Got the info.'])
    agent = _make_agent(mock_model, tools=[get_info])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Get info')
    simplified = testing_utils.simplify_events(events)

    fr_response = simplified[1][1].function_response.response
    assert fr_response == {'name': 'Alice', 'age': 30}

  @pytest.mark.asyncio
  async def test_tool_returning_string(self):
    """Tool returning a string is properly serialized."""

    def say_hello() -> str:
      """Say hello."""
      return 'Hello!'

    fc = types.Part.from_function_call(name='say_hello', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Said hello.'])
    agent = _make_agent(mock_model, tools=[say_hello])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Say hello')
    simplified = testing_utils.simplify_events(events)

    assert simplified[1][1].function_response.response == {'result': 'Hello!'}

  @pytest.mark.asyncio
  async def test_tool_returning_none(self):
    """Tool returning None still produces a function response."""

    def do_nothing() -> None:
      """Do nothing."""
      return None

    fc = types.Part.from_function_call(name='do_nothing', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Done.'])
    agent = _make_agent(mock_model, tools=[do_nothing])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Do nothing')
    simplified = testing_utils.simplify_events(events)

    # Should still have FC, FR, Text
    assert len(simplified) == 3
    assert simplified[1][1].function_response.name == 'do_nothing'


# ---------------------------------------------------------------------------
# Pipeline: CallLlmNode behavior (e2e via _SingleLlmAgent)
# ---------------------------------------------------------------------------


class TestPipelineCallLlm:
  """E2E tests for CallLlmNode behavior through the full pipeline."""

  @pytest.mark.asyncio
  async def test_text_only_response(self):
    """LLM returns plain text -> no route to execute_tools."""
    mock_model = testing_utils.MockModel.create(responses=['Hello, world!'])
    agent = _make_agent(mock_model)
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Hi')
    simplified = testing_utils.simplify_events(events)

    assert len(simplified) == 1
    assert simplified[0] == ('test_agent', 'Hello, world!')

  @pytest.mark.asyncio
  async def test_single_function_call_and_response(self):
    """LLM returns function call -> tool executes -> LLM returns text."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'The answer is 3.']
    )
    agent = _make_agent(mock_model, tools=[add])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('What is 1+2?')
    simplified = testing_utils.simplify_events(events)

    # Expect: FunctionCall, FunctionResponse, Text
    assert len(simplified) == 3
    # FunctionCall event
    assert simplified[0][0] == 'test_agent'
    assert simplified[0][1].function_call.name == 'add'
    # FunctionResponse event
    assert simplified[1][0] == 'test_agent'
    assert simplified[1][1].function_response.name == 'add'
    assert simplified[1][1].function_response.response['result'] == 3
    # Text event
    assert simplified[2] == ('test_agent', 'The answer is 3.')

  @pytest.mark.asyncio
  async def test_multi_step_function_calls(self):
    """LLM calls tool twice before producing final text."""

    def double(x: int) -> int:
      """Double the input."""
      return x * 2

    fc1 = types.Part.from_function_call(name='double', args={'x': 3})
    fc2 = types.Part.from_function_call(name='double', args={'x': 6})
    mock_model = testing_utils.MockModel.create(
        responses=[fc1, fc2, 'The final answer is 12.']
    )
    agent = _make_agent(mock_model, tools=[double])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Double 3 twice')
    simplified = testing_utils.simplify_events(events)

    # Expect: FC, FR, FC, FR, Text
    assert len(simplified) == 5
    # First call
    assert simplified[0][1].function_call.name == 'double'
    assert simplified[1][1].function_response.response['result'] == 6
    # Second call
    assert simplified[2][1].function_call.name == 'double'
    assert simplified[3][1].function_response.response['result'] == 12
    # Final text
    assert simplified[4] == ('test_agent', 'The final answer is 12.')

  @pytest.mark.asyncio
  async def test_resumable_function_calls_json_serializable(self):
    """function_calls passed to ExecuteToolsNode are JSON-safe.

    CallLlmNode serializes each FunctionCall via model_dump(mode='json')
    so they can be stored in NodeState.input without serialization
    errors.
    """

    def echo(msg: str) -> str:
      """Echo the message."""
      return msg

    fc = types.Part.from_function_call(name='echo', args={'msg': 'hello'})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Done.'])
    agent = _make_agent(mock_model, tools=[echo])

    app = App(
        name='serialization_test',
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=app)

    # This would raise PydanticSerializationError if the event
    # wasn't properly serialized.
    events = runner.run('echo hello')
    simplified = testing_utils.simplify_events(events)

    assert len(simplified) == 3
    assert simplified[2] == ('test_agent', 'Done.')

  @pytest.mark.asyncio
  async def test_model_request_has_agent_name_label(self):
    """CallLlmNode sets adk_agent_name label on the LLM request."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model)
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('test')

    assert len(mock_model.requests) == 1
    labels = mock_model.requests[0].config.labels
    assert labels.get('adk_agent_name') == 'test_agent'

  @pytest.mark.asyncio
  async def test_before_model_callback_prevents_llm_call(self):
    """before_model_callback can short-circuit the LLM call."""

    def before_cb(callback_context, llm_request):
      return LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Intercepted!')],
          )
      )

    mock_model = testing_utils.MockModel.create(
        responses=['Should not reach here.']
    )
    agent = _make_agent(
        mock_model,
        before_model_callback=before_cb,
    )
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('test')

    # The key assertion: MockModel should not have been called
    # because before_model_callback intercepted the request.
    assert len(mock_model.requests) == 0

  @pytest.mark.asyncio
  async def test_after_model_callback(self):
    """after_model_callback can modify the LLM response."""

    def after_cb(callback_context, llm_response):
      return LlmResponse(
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text='Modified response!')],
          )
      )

    mock_model = testing_utils.MockModel.create(
        responses=['Original response.']
    )
    agent = _make_agent(
        mock_model,
        after_model_callback=after_cb,
    )
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('test')
    simplified = testing_utils.simplify_events(events)

    assert len(simplified) == 1
    assert simplified[0] == ('test_agent', 'Modified response!')


# ---------------------------------------------------------------------------
# Pipeline: ExecuteToolsNode behavior (e2e via _SingleLlmAgent)
# ---------------------------------------------------------------------------


class TestPipelineExecuteTools:
  """E2E tests for ExecuteToolsNode behavior through the full pipeline."""

  @pytest.mark.asyncio
  async def test_tool_result_fed_back_to_llm(self):
    """Tool response is appended to session and fed back to LLM."""
    call_count = 0

    def counter() -> int:
      """Return an incrementing count."""
      nonlocal call_count
      call_count += 1
      return call_count

    fc = types.Part.from_function_call(name='counter', args={})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Count is 1.'])
    agent = _make_agent(mock_model, tools=[counter])
    runner = testing_utils.InMemoryRunner(agent)

    runner.run('Count')

    # Second LLM request should contain the function response
    assert len(mock_model.requests) == 2
    second_request_contents = mock_model.requests[1].contents
    # Find the function response in the request
    has_fr = False
    for content in second_request_contents:
      for part in content.parts:
        if part.function_response and part.function_response.name == 'counter':
          has_fr = True
          assert part.function_response.response['result'] == 1
    assert has_fr, 'Function response not found in second LLM request'

  @pytest.mark.asyncio
  async def test_tools_dict_reconstructed_from_agent(self):
    """tools_dict is reconstructed in ExecuteToolsNode, not via data.

    This verifies that the tool function is actually callable even
    though it was never serialized through workflow event data.
    """

    def greet(name: str) -> str:
      """Greet someone."""
      return f'Hello, {name}!'

    fc = types.Part.from_function_call(name='greet', args={'name': 'Alice'})
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'Greeted Alice.']
    )
    agent = _make_agent(mock_model, tools=[greet])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Greet Alice')
    simplified = testing_utils.simplify_events(events)

    # FunctionResponse should contain the actual result
    assert simplified[1][1].function_response.response['result'] == (
        'Hello, Alice!'
    )

  @pytest.mark.asyncio
  async def test_tool_confirmation_require_confirmation_true(self):
    """Tool with require_confirmation=True triggers confirmation."""

    def dangerous_action(x: int) -> str:
      """A dangerous action."""
      return 'done'

    tool = FunctionTool(dangerous_action, require_confirmation=True)
    fc = types.Part.from_function_call(name='dangerous_action', args={'x': 42})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[tool])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Do dangerous action')

    # Should have a confirmation request in the events
    has_confirmation = False
    has_interrupt = False
    for event in events:
      if event.actions.requested_tool_confirmations:
        has_confirmation = True
      if event.long_running_tool_ids:
        has_interrupt = True

    assert has_confirmation, 'No tool confirmation request found'
    assert has_interrupt, 'No interrupt (long_running_tool_ids) found'

  @pytest.mark.asyncio
  async def test_tool_confirmation_callback(self):
    """Tool with require_confirmation callback triggers conditionally."""

    async def should_confirm(x: int) -> bool:
      return x > 100

    def action(x: int) -> str:
      """An action."""
      return f'processed {x}'

    tool = FunctionTool(action, require_confirmation=should_confirm)

    # Case 1: x=50, should NOT trigger confirmation
    fc_low = types.Part.from_function_call(name='action', args={'x': 50})
    mock_model = testing_utils.MockModel.create(
        responses=[fc_low, 'Done with 50.']
    )
    agent = _make_agent(mock_model, tools=[tool])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Do action with 50')
    has_confirmation = any(
        event.actions.requested_tool_confirmations for event in events
    )
    assert not has_confirmation, 'Should NOT request confirmation for x=50'

    # Case 2: x=200, SHOULD trigger confirmation
    fc_high = types.Part.from_function_call(name='action', args={'x': 200})
    mock_model2 = testing_utils.MockModel.create(responses=[fc_high])
    agent2 = _make_agent(mock_model2, tools=[tool])
    runner2 = testing_utils.InMemoryRunner(agent2)

    events2 = runner2.run('Do action with 200')
    has_confirmation = any(
        event.actions.requested_tool_confirmations for event in events2
    )
    assert has_confirmation, 'Should request confirmation for x=200'

  @pytest.mark.asyncio
  async def test_tool_request_confirmation_custom(self):
    """Tool using request_confirmation inside the function body."""

    def review_action(amount: int, tool_context: ToolContext) -> dict:
      """Action requiring review for large amounts."""
      if amount > 1000:
        tool_confirmation = tool_context.tool_confirmation
        if not tool_confirmation:
          tool_context.request_confirmation(
              hint='Please approve this large amount.',
              payload={'amount': amount},
          )
          return {'status': 'pending approval'}
        approved = tool_confirmation.payload.get('approved', False)
        if not approved:
          return {'status': 'rejected'}
      return {'status': 'ok', 'amount': amount}

    fc = types.Part.from_function_call(
        name='review_action', args={'amount': 5000}
    )
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[review_action])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Process $5000')

    # Should have confirmation request
    has_confirmation = False
    for event in events:
      if event.actions.requested_tool_confirmations:
        has_confirmation = True
        # Verify the confirmation has the correct payload
        for _, conf in event.actions.requested_tool_confirmations.items():
          assert conf.hint == 'Please approve this large amount.'
          assert conf.payload == {'amount': 5000}
    assert has_confirmation

  @pytest.mark.asyncio
  async def test_no_function_calls_skips_execute_tools(self):
    """When LLM returns text, execute_tools node is never triggered."""
    call_count = 0

    def spy_tool() -> str:
      """Should never be called."""
      nonlocal call_count
      call_count += 1
      return 'called'

    mock_model = testing_utils.MockModel.create(
        responses=['Just text, no tools.']
    )
    agent = _make_agent(mock_model, tools=[spy_tool])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Hello')

    assert call_count == 0, 'Tool should not have been called'
    simplified = testing_utils.simplify_events(events)
    assert simplified == [('test_agent', 'Just text, no tools.')]

  @pytest.mark.asyncio
  async def test_multiple_function_calls_in_single_response(self):
    """LLM returns multiple function calls in a single response."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    # LLM returns both function calls in one response
    fc_add = types.Part.from_function_call(name='add', args={'x': 2, 'y': 3})
    fc_mul = types.Part.from_function_call(
        name='multiply', args={'x': 4, 'y': 5}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[
            [fc_add, fc_mul],
            'Sum is 5 and product is 20.',
        ]
    )
    agent = _make_agent(mock_model, tools=[add, multiply])
    runner = testing_utils.InMemoryRunner(agent)

    events = runner.run('Add 2+3 and multiply 4*5')
    simplified = testing_utils.simplify_events(events)

    # Expect: [fc_add, fc_mul], [fr_add, fr_mul], text
    assert len(simplified) == 3
    # First event has two function calls
    fc_parts = simplified[0][1]
    assert isinstance(fc_parts, list)
    assert len(fc_parts) == 2
    # Second event has two function responses
    fr_parts = simplified[1][1]
    assert isinstance(fr_parts, list)
    assert len(fr_parts) == 2
    # Final text
    assert simplified[2] == ('test_agent', 'Sum is 5 and product is 20.')


# ---------------------------------------------------------------------------
# Duplicate node_input regression tests
# ---------------------------------------------------------------------------


class TestDuplicateNodeInput:
  """Tests that node_input is not appended twice to the session."""

  @pytest.mark.asyncio
  async def test_first_node_in_workflow_no_duplicate_user_message(self):
    """When _SingleLlmAgent is the first node in a Workflow, the user
    message should appear exactly once in the LLM request contents.

    Regression test: the runner appends user_content to the session,
    and _SingleLlmAgent.run() used to append it again as node_input.
    """
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    inner_agent = _make_agent(mock_model)
    outer_workflow = Workflow(
        name='outer',
        edges=[
            (START, inner_agent),
        ],
    )
    runner = testing_utils.InMemoryRunner(outer_workflow)

    runner.run('Hello')

    # The LLM should see the user message exactly once.
    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1
    assert user_contents[0].parts[0].text == 'Hello'


# ---------------------------------------------------------------------------
# input_schema validation tests
# ---------------------------------------------------------------------------


class _TaskInput(BaseModel):
  goal: str
  priority: int = 1


class TestInputSchema:
  """Tests that input_schema validates node_input in _SingleLlmAgent.run()."""

  @pytest.mark.asyncio
  async def test_dict_node_input_validated_against_input_schema(self):
    """Dict node_input is validated and converted to BaseModel."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    events = []
    async for event in agent.run(
        ctx=ctx, node_input={'goal': 'test goal', 'priority': 5}
    ):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1
    text = user_contents[0].parts[0].text
    # Validated BaseModel is serialized via model_dump_json
    assert 'test goal' in text
    assert '5' in text

  @pytest.mark.asyncio
  async def test_json_string_node_input_validated_against_input_schema(self):
    """JSON string node_input is parsed and validated."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    events = []
    async for event in agent.run(
        ctx=ctx,
        node_input='{"goal": "from json", "priority": 3}',
    ):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1
    text = user_contents[0].parts[0].text
    # Validated BaseModel is serialized via model_dump_json
    assert 'from json' in text
    assert '3' in text

  @pytest.mark.asyncio
  async def test_already_typed_node_input_passes_through(self):
    """Node input that is already the correct BaseModel type passes."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    task_input = _TaskInput(goal='typed input', priority=10)
    events = []
    async for event in agent.run(ctx=ctx, node_input=task_input):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1
    text = user_contents[0].parts[0].text
    assert 'typed input' in text

  @pytest.mark.asyncio
  async def test_invalid_node_input_raises_validation_error(self):
    """Invalid node_input raises ValidationError."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    with pytest.raises(ValidationError):
      async for _ in agent.run(
          ctx=ctx,
          # Missing required 'goal' field
          node_input={'priority': 5},
      ):
        pass

  @pytest.mark.asyncio
  async def test_no_input_schema_skips_validation(self):
    """When input_schema is None, dict node_input passes through."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model)  # No input_schema
    ctx = await testing_utils.create_workflow_context(agent)

    events = []
    async for event in agent.run(ctx=ctx, node_input={'any': 'data'}):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1

  @pytest.mark.asyncio
  async def test_content_node_input_skips_validation(self):
    """types.Content node_input bypasses input_schema validation."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    content = types.Content(
        role='user',
        parts=[types.Part(text='raw content')],
    )
    events = []
    async for event in agent.run(ctx=ctx, node_input=content):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    assert len(user_contents) == 1
    assert user_contents[0].parts[0].text == 'raw content'

  @pytest.mark.asyncio
  async def test_default_values_applied_during_validation(self):
    """Default field values from input_schema are applied."""
    mock_model = testing_utils.MockModel.create(responses=['ok'])
    agent = _make_agent(mock_model, input_schema=_TaskInput)
    ctx = await testing_utils.create_workflow_context(agent)

    events = []
    # Only provide 'goal', 'priority' should default to 1
    async for event in agent.run(ctx=ctx, node_input={'goal': 'minimal'}):
      events.append(event)

    assert len(mock_model.requests) == 1
    user_contents = [
        c for c in mock_model.requests[0].contents if c.role == 'user'
    ]
    text = user_contents[0].parts[0].text
    assert 'minimal' in text
    # Default priority=1 should be present
    assert '1' in text
