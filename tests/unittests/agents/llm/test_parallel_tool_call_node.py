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

"""Unit tests for ParallelToolCallNode in isolation.

Tests call ``ParallelToolCallNode.run()`` directly with a hand-built
Context and a scheduler wired up via NodeRunner, verifying that tool
calls are executed in parallel and results are merged correctly.
"""

from __future__ import annotations

import asyncio

from google.adk.agents.llm._parallel_tool_call_node import ParallelToolCallNode
from google.adk.agents.llm._parallel_tool_call_node import ParallelToolCallResult
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow._node_runner_class import NodeRunner
from google.genai import types

from tests.unittests.agents.llm.event_utils import function_responses_by_name
from tests.unittests.agents.llm.event_utils import output_events
from tests.unittests.workflow import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tools_dict(funcs):
  """Build a tools_dict from a list of plain functions."""
  return {f.__name__: FunctionTool(f) for f in funcs}


async def _drain_queue(queue):
  """Background task that drains the event queue, unblocking enqueue_event."""
  while True:
    event, processed = await queue.get()
    if processed is not None:
      processed.set()


def _wire_scheduler(ctx):
  """Wire up _schedule_dynamic_node_internal and event_queue on the context."""
  ic = ctx._invocation_context
  ic.event_queue = asyncio.Queue()

  async def _schedule(
      current_ctx,
      node,
      run_id,
      node_input,
      *,
      node_name=None,  # noqa: ARG001
  ):
    runner = NodeRunner(
        node=node,
        parent_ctx=current_ctx,
        run_id=run_id,
    )
    return await runner.run(node_input=node_input)

  ctx._schedule_dynamic_node_internal = _schedule


async def _collect_events(node, ctx, node_input):
  """Collect events from node.run() with a background queue drain."""
  drain_task = asyncio.create_task(
      _drain_queue(ctx._invocation_context.event_queue)
  )
  try:
    events = []
    async for event in node.run(ctx=ctx, node_input=node_input):
      events.append(event)
    return events
  finally:
    drain_task.cancel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelToolCallNode:

  async def test_single_tool_call(self):
    """Single function call — executes and returns merged result."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    tools_dict = _make_tools_dict([add])
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)
    _wire_scheduler(ctx)

    fc = types.FunctionCall(name='add', args={'x': 1, 'y': 2}, id='fc-1')
    node = ParallelToolCallNode(tools_dict=tools_dict)

    content = types.Content(role='model', parts=[types.Part(function_call=fc)])
    events = await _collect_events(node, ctx, content)

    fr = function_responses_by_name(events)
    assert fr['add'] == {'result': 3}

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, ParallelToolCallResult)
    assert 'fc-1' in result.tool_results

  async def test_parallel_tool_calls(self):
    """Multiple function calls — executed in parallel, results merged."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    tools_dict = _make_tools_dict([add, multiply])
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)
    _wire_scheduler(ctx)

    fc_add = types.FunctionCall(name='add', args={'x': 1, 'y': 2}, id='fc-add')
    fc_mul = types.FunctionCall(
        name='multiply', args={'x': 3, 'y': 4}, id='fc-mul'
    )
    node = ParallelToolCallNode(tools_dict=tools_dict)

    content = types.Content(
        role='model',
        parts=[
            types.Part(function_call=fc_add),
            types.Part(function_call=fc_mul),
        ],
    )
    events = await _collect_events(node, ctx, content)

    fr = function_responses_by_name(events)
    assert fr['add'] == {'result': 3}
    assert fr['multiply'] == {'result': 12}

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, ParallelToolCallResult)
    assert 'fc-add' in result.tool_results
    assert 'fc-mul' in result.tool_results

  async def test_transfer_to_agent_propagated(self):
    """Tool sets transfer_to_agent — propagated in ParallelToolCallResult."""

    def transfer_tool(tool_context) -> str:
      """Transfer."""
      tool_context.actions.transfer_to_agent = 'target_agent'
      return 'done'

    tools_dict = _make_tools_dict([transfer_tool])
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)
    _wire_scheduler(ctx)

    fc = types.FunctionCall(name='transfer_tool', args={}, id='fc-transfer')
    node = ParallelToolCallNode(tools_dict=tools_dict)

    content = types.Content(role='model', parts=[types.Part(function_call=fc)])
    events = await _collect_events(node, ctx, content)

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, ParallelToolCallResult)
    assert result.transfer_to_agent == 'target_agent'

  async def test_same_tool_called_twice(self):
    """Same tool called twice with different args — both execute."""
    call_log = []

    def echo(msg: str) -> str:
      """Echo."""
      call_log.append(msg)
      return msg

    tools_dict = _make_tools_dict([echo])
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_workflow_context(agent)
    _wire_scheduler(ctx)

    fc1 = types.FunctionCall(name='echo', args={'msg': 'a'}, id='fc-1')
    fc2 = types.FunctionCall(name='echo', args={'msg': 'b'}, id='fc-2')
    node = ParallelToolCallNode(tools_dict=tools_dict)

    content = types.Content(
        role='model',
        parts=[types.Part(function_call=fc1), types.Part(function_call=fc2)],
    )
    events = await _collect_events(node, ctx, content)

    assert sorted(call_log) == ['a', 'b']

    outputs = output_events(events)
    assert len(outputs) == 1
    result = outputs[0].output
    assert isinstance(result, ParallelToolCallResult)
    assert len(result.tool_results) == 2
