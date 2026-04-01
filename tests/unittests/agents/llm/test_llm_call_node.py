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
function call responses yield the model response content event.
"""

from __future__ import annotations

from google.adk.agents.llm._llm_call_node import LlmCallNode
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
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

  async def test_text_response_yields_text_output(self):
    """Text-only LLM response — yields event with message_as_output=True."""
    mock_model = testing_utils.MockModel.create(responses=['Hello!'])
    agent = _make_agent(mock_model)
    ctx = await testing_utils.create_workflow_context(agent, user_content='Hi')

    node = LlmCallNode(agent=agent)
    llm_request = LlmRequest(model='mock', tools_dict={})
    events = await collect_events(node, ctx, node_input=llm_request)

    assert 'Hello!' in text_parts(events)
    # The node should yield a finalized event with message_as_output=True
    final_event = events[-1]
    assert final_event.node_info is not None
    assert final_event.node_info.message_as_output is True

  async def test_function_call_yields_content(self):
    """Function call response — yields content event with message_as_output=True."""

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

    tools = await agent.canonical_tools(ctx)
    tools_dict = {t.name: t for t in tools}
    llm_request = LlmRequest(model='mock', tools_dict=tools_dict)

    events = await collect_events(node, ctx, node_input=llm_request)

    assert 'add' in function_call_names(events)

    final_event = events[-1]
    assert final_event.node_info is not None
    assert final_event.node_info.message_as_output is True
    assert final_event.content.parts[0].function_call.name == 'add'
