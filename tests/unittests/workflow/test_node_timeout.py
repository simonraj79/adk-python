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

"""Tests for node timeout support."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._errors import NodeTimeoutError
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow_graph import WorkflowGraph
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import TestingNode


class _SlowNode(BaseNode):
  """A node that sleeps for a configurable duration before yielding."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  delay: float = Field(default=0.0)
  message: str = Field(default='done')
  tracker: dict[str, Any] = Field(default_factory=dict)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    count = self.tracker.get('run_count', 0) + 1
    self.tracker['run_count'] = count
    await asyncio.sleep(self.delay)
    yield Event(output=self.message)


class _SlowNodeThatSucceedsOnRetry(BaseNode):
  """Times out on first attempt, succeeds on subsequent attempts."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  slow_delay: float = Field(default=10.0)
  fast_delay: float = Field(default=0.01)
  tracker: dict[str, Any] = Field(default_factory=dict)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    count = self.tracker.get('run_count', 0) + 1
    self.tracker['run_count'] = count

    if count == 1:
      await asyncio.sleep(self.slow_delay)
    else:
      await asyncio.sleep(self.fast_delay)

    yield Event(output='success')


@pytest.mark.asyncio
async def test_node_completes_within_timeout(request: pytest.FixtureRequest):
  """A node that finishes before the timeout should succeed normally."""
  node_a = _SlowNode(name='NodeA', delay=0.01, timeout=5.0)
  graph = WorkflowGraph(
      edges=[Edge(START, node_a)],
  )
  agent = Workflow(name='test_workflow', graph=graph)
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('test_workflow', {'node_name': 'NodeA', 'output': 'done'}),
  ]


@pytest.mark.asyncio
async def test_node_exceeds_timeout(request: pytest.FixtureRequest):
  """A node that exceeds its timeout should raise NodeTimeoutError."""
  node_a = _SlowNode(name='NodeA', delay=10.0, timeout=0.05)
  node_b = TestingNode(name='NodeB', output='should not run')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  agent = Workflow(name='test_workflow', graph=graph)
  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  with pytest.raises(NodeTimeoutError) as exc_info:
    [e async for e in agent.run_async(ctx)]

  assert exc_info.value.node_name == 'NodeA'
  assert exc_info.value.timeout == 0.05
  assert 'NodeA' in str(exc_info.value)


@pytest.mark.asyncio
async def test_node_no_timeout(request: pytest.FixtureRequest):
  """A node with timeout=None should run without any time limit."""
  node_a = _SlowNode(name='NodeA', delay=0.01, timeout=None)
  graph = WorkflowGraph(
      edges=[Edge(START, node_a)],
  )
  agent = Workflow(name='test_workflow', graph=graph)
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('test_workflow', {'node_name': 'NodeA', 'output': 'done'}),
  ]


@pytest.mark.asyncio
async def test_timeout_with_retry(request: pytest.FixtureRequest):
  """A timed-out node should be retried if retry_config is set."""
  tracker: dict[str, Any] = {}

  # First run: delay=10s exceeds timeout → times out
  # After retry the tracker is updated, so we use a node that succeeds
  # on the second attempt.
  node_a = _SlowNodeThatSucceedsOnRetry(
      name='NodeA',
      slow_delay=10.0,
      fast_delay=0.01,
      timeout=0.05,
      tracker=tracker,
      retry_config=RetryConfig(
          max_attempts=3,
          initial_delay=0.0,
          jitter=0.0,
      ),
  )
  graph = WorkflowGraph(
      edges=[Edge(START, node_a)],
  )
  agent = Workflow(name='test_workflow', graph=graph)
  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('test_workflow', {'node_name': 'NodeA', 'output': 'success'}),
  ]

  # Verify retry_count in agent state events to confirm retry occurred.
  state_events = [e for e in events if e.actions.agent_state]
  assert state_events, 'Expected at least one agent state checkpoint event'
  last_state = state_events[-1].actions.agent_state
  node_a_state = last_state['nodes']['NodeA']
  assert (
      node_a_state['retry_count'] >= 1
  ), f'Expected retry_count >= 1, got {node_a_state["retry_count"]}'
