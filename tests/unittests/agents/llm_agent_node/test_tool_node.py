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

"""Unit tests for ToolNode in isolation.

Tests call ``ToolNode.run()`` directly with a hand-built Context,
verifying that a single tool call produces the expected function response
output.
"""

from __future__ import annotations

from google.adk.agents.llm_agent_node._tool_node import ToolNode
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from tests.unittests.agents.llm.event_utils import collect_events
from tests.unittests.agents.llm.event_utils import output_events
from tests.unittests.workflow import testing_utils


class TestToolNode:

  async def test_simple_tool_call(self):
    """Tool call returns a value — yields normalized response dict."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    tool = FunctionTool(add)
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)

    fc = types.FunctionCall(name='add', args={'x': 2, 'y': 3}, id='fc-1')
    node = ToolNode(name='tool_call__fc-1', tool=tool)
    events = await collect_events(node, ctx, fc)

    outputs = output_events(events)
    assert len(outputs) == 1
    assert outputs[0].output.response == {'result': 5}

  async def test_tool_returning_dict(self):
    """Tool returning a dict — output preserved as-is."""

    def lookup(key: str) -> dict:
      """Look up a value."""
      return {'found': True, 'value': 42}

    tool = FunctionTool(lookup)
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)

    fc = types.FunctionCall(name='lookup', args={'key': 'x'}, id='fc-2')
    node = ToolNode(name='tool_call__fc-2', tool=tool)
    events = await collect_events(node, ctx, fc)

    outputs = output_events(events)
    assert len(outputs) == 1
    assert outputs[0].output.response == {'found': True, 'value': 42}

  async def test_tool_with_no_args(self):
    """Tool with no args — empty args dict passed."""

    def get_time() -> str:
      """Get current time."""
      return '12:00'

    tool = FunctionTool(get_time)
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)

    fc = types.FunctionCall(name='get_time', args={}, id='fc-3')
    node = ToolNode(name='tool_call__fc-3', tool=tool)
    events = await collect_events(node, ctx, fc)

    outputs = output_events(events)
    assert len(outputs) == 1
    assert outputs[0].output.response == {'result': '12:00'}

  async def test_function_call_id_auto_generated(self):
    """FunctionCall without id — ToolNode generates one."""

    def noop() -> str:
      """No-op."""
      return 'ok'

    tool = FunctionTool(noop)
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)

    fc = types.FunctionCall(name='noop', args={})  # no id
    node = ToolNode(name='tool_call__0', tool=tool)
    events = await collect_events(node, ctx, fc)

    outputs = output_events(events)
    assert len(outputs) == 1
    # The output is the ToolNodeOutput; response is the response dict.
    assert outputs[0].output.response == {'result': 'ok'}

  async def test_tool_sets_transfer_action(self):
    """Tool that sets transfer_to_agent — action on context, not output."""

    def transfer_tool(tool_context) -> str:
      """Transfer."""
      tool_context.actions.transfer_to_agent = 'other_agent'
      return 'transferring'

    tool = FunctionTool(transfer_tool)
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)

    fc = types.FunctionCall(name='transfer_tool', args={}, id='fc-t')
    node = ToolNode(name='tool_call__fc-t', tool=tool)
    events = await collect_events(node, ctx, fc)

    outputs = output_events(events)
    assert len(outputs) == 1
    assert outputs[0].output.response == {'result': 'transferring'}
    # Actions are bubbled up to ToolNodeOutput.
    assert outputs[0].output.actions.transfer_to_agent == 'other_agent'
