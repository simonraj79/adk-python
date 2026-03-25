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

"""Unit tests for LlmCallNode (BaseNode version) in isolation.

Tests call ``LlmCallNode.run()`` directly with a hand-built Context,
verifying the output contract: text responses yield no output,
function call responses yield ``LlmCallResult`` as output.
"""

from __future__ import annotations

from google.adk.agents.llm._llm_call_node import LlmCallNode
from google.adk.agents.llm._llm_call_node import LlmCallResult
from google.adk.agents.llm_agent import LlmAgent
from google.genai import types

from tests.unittests.agents.llm.event_utils import collect_events
from tests.unittests.agents.llm.event_utils import function_call_names
from tests.unittests.agents.llm.event_utils import output_events
from tests.unittests.agents.llm.event_utils import text_parts
from tests.unittests.workflow import testing_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(mock_model, tools=None, **kwargs):
  """Create an LlmAgent wired to a MockModel."""
  return LlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **kwargs,
  )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLlmCallNode:

  async def test_text_response_no_output(self):
    """Text-only LLM response — no output event, text in content."""
    mock_model = testing_utils.MockModel.create(responses=['Hello!'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    node = LlmCallNode(agent=agent)
    events = await collect_events(node, ctx)

    assert 'Hello!' in text_parts(events)
    assert output_events(events) == []

  async def test_function_call_yields_output(self):
    """Function call response — yields LlmCallResult as output."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[add])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Add 1+2'
    )

    node = LlmCallNode(agent=agent)
    events = await collect_events(node, ctx)

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, LlmCallResult)
    assert len(result.function_calls) == 1
    assert result.function_calls[0].name == 'add'

  async def test_function_call_content_event(self):
    """Function call — also yields the model response as content event."""

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    fc = types.Part.from_function_call(
        name='multiply', args={'x': 3, 'y': 4}
    )
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[multiply])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Multiply 3*4'
    )

    node = LlmCallNode(agent=agent)
    events = await collect_events(node, ctx)

    assert 'multiply' in function_call_names(events)

  async def test_tools_dict_in_output(self):
    """Function call response — LlmCallResult includes tools_dict."""

    def greet(name: str) -> str:
      """Greet someone."""
      return f'Hi {name}'

    fc = types.Part.from_function_call(name='greet', args={'name': 'Ada'})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[greet])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Hi'
    )

    node = LlmCallNode(agent=agent)
    events = await collect_events(node, ctx)

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, LlmCallResult)
    assert 'greet' in result.tools_dict

  async def test_multiple_tools_in_output(self):
    """Multiple tools — all included in LlmCallResult.tools_dict."""

    def add(x: int, y: int) -> int:
      """Add."""
      return x + y

    def sub(x: int, y: int) -> int:
      """Subtract."""
      return x - y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc])
    agent = _make_agent(mock_model, tools=[add, sub])
    ctx = await testing_utils.create_workflow_context(
        agent, user_content='Go'
    )

    node = LlmCallNode(agent=agent)
    events = await collect_events(node, ctx)

    outputs = output_events(events)
    result = outputs[0].output
    assert isinstance(result, LlmCallResult)
    assert 'add' in result.tools_dict
    assert 'sub' in result.tools_dict
