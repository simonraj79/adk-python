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

"""Tests for SingleAgentReactNode.

Runs SingleAgentReactNode directly via Runner(node=...) — no Workflow
wrapper needed because SingleAgentReactNode creates its own
schedule_dynamic_node and sets ic.agent from its agent field.
"""

from __future__ import annotations

from google.adk.agents.llm._single_agent_react_node import SingleAgentReactNode
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from tests.unittests.agents.llm.event_utils import function_call_names
from tests.unittests.agents.llm.event_utils import function_response_dicts
from tests.unittests.agents.llm.event_utils import function_responses_by_name
from tests.unittests.agents.llm.event_utils import text_parts
from ... import testing_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = 'test_user'
_SESSION_ID = 'test_session'


async def _run(mock_model, user_message, tools=None, **agent_kwargs):
  """Run SingleAgentReactNode via Runner and return events."""
  llm_agent = LlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **agent_kwargs,
  )
  react_node = SingleAgentReactNode(name='react', agent=llm_agent)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
  )
  runner = Runner(
      app_name='test',
      node=react_node,
      session_service=session_service,
  )

  return [
      e async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          new_message=types.Content(
              role='user', parts=[types.Part(text=user_message)]
          ),
      )
  ]


# ---------------------------------------------------------------------------
# Tests: text response
# ---------------------------------------------------------------------------


class TestBasicTextResponse:

  async def test_pure_text_response(self):
    """LLM returns plain text — loop ends after one iteration."""
    mock_model = testing_utils.MockModel.create(responses=['Hello!'])
    events = await _run(mock_model, 'Hi')

    assert 'Hello!' in text_parts(events)


# ---------------------------------------------------------------------------
# Tests: tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:

  async def test_single_tool_call(self):
    """LLM calls a tool, gets response, then returns text."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'Result is 3.']
    )
    events = await _run(mock_model, 'Add 1 and 2', tools=[add])

    assert {'result': 3} in function_response_dicts(events)
    assert any('Result is 3.' in t for t in text_parts(events))

  async def test_no_function_calls(self):
    """LLM returns text with no function calls — tool not invoked."""
    mock_model = testing_utils.MockModel.create(
        responses=['Just a text reply.']
    )

    def unused_tool() -> str:
      """A tool that should not be called."""
      raise AssertionError('Tool should not be called')

    events = await _run(mock_model, 'Hello', tools=[unused_tool])

    assert 'Just a text reply.' in text_parts(events)
    assert function_response_dicts(events) == []

  async def test_parallel_tool_calls(self):
    """LLM calls multiple tools in parallel."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    fc_add = types.Part.from_function_call(
        name='add', args={'x': 1, 'y': 2}
    )
    fc_mul = types.Part.from_function_call(
        name='multiply', args={'x': 3, 'y': 4}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[[fc_add, fc_mul], 'Done.']
    )
    events = await _run(mock_model, 'Compute', tools=[add, multiply])

    fr = function_responses_by_name(events)
    assert fr['add'] == {'result': 3}
    assert fr['multiply'] == {'result': 12}

  async def test_multiple_sequential_tool_calls(self):
    """LLM calls tools in sequence across ReAct turns."""
    call_count = 0

    def increment(x: int) -> int:
      """Increment a number."""
      nonlocal call_count
      call_count += 1
      return x + 1

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name='increment', args={'x': 1}
            ),
            types.Part.from_function_call(
                name='increment', args={'x': 2}
            ),
            'done',
        ]
    )
    events = await _run(mock_model, 'Go', tools=[increment])

    assert call_count == 2
    assert function_call_names(events).count('increment') == 2
    assert 'done' in text_parts(events)


# ---------------------------------------------------------------------------
# Tests: termination
# ---------------------------------------------------------------------------


class TestTermination:

  async def test_transfer_to_agent_ends_loop(self):
    """Tool that calls transfer_to_agent — ReAct loop terminates."""

    def transfer_tool(tool_context: ToolContext) -> str:
      tool_context.actions.transfer_to_agent = 'other_agent'
      return 'transferring'

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='transfer_tool', args={}),
        ]
    )
    events = await _run(
        mock_model,
        'Go',
        tools=[transfer_tool],
    )

    # Loop should end after transfer (no second LLM call).
    assert len(mock_model.requests) == 1
    assert 'transfer_tool' in function_call_names(events)
