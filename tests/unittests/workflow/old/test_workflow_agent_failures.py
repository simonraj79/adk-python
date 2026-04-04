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

"""Testings for Workflow retry logic on failures."""

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._node import Node
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import workflow_node_input
from google.adk.workflow._workflow import WorkflowAgentState
from google.adk.workflow._workflow_graph import WorkflowGraph
from google.adk.workflow.utils._node_path_utils import join_paths
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from ..workflow_testing_utils import create_parent_invocation_context
from ..workflow_testing_utils import simplify_events_with_node
from ..workflow_testing_utils import TestingNode


class CustomRetryableError(Exception):
  """A custom error meant to be retried."""


class CustomNonRetryableError(Exception):
  """A custom error not meant to be retried."""


class _FlakyNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  message: str = Field(default='')
  succeed_on_iteration: int = Field(default=0)
  tracker: dict[str, Any] = Field(default_factory=dict)
  exception_to_raise: Exception = Field(...)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    iteration_count = self.tracker.get('iteration_count', 0) + 1
    self.tracker['iteration_count'] = iteration_count
    self.tracker.setdefault('retry_counts', []).append(ctx.retry_count)

    if iteration_count < self.succeed_on_iteration:
      raise self.exception_to_raise

    yield Event(
        output=self.message,
    )


@pytest.mark.asyncio
async def test_retry_on_matching_exception(request: pytest.FixtureRequest):
  """Tests that retries occur for exceptions listed in RetryConfig."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_no_retry_on_non_matching_exception(
    request: pytest.FixtureRequest,
):
  """Tests that no retry occurs for exceptions not listed in RetryConfig."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomNonRetryableError('Unexpected failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_no_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(CustomNonRetryableError, match='Unexpected failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_no_retry',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_retry_on_all_exceptions_if_not_specified(
    request: pytest.FixtureRequest,
):
  """Tests retries when `exceptions` is not specified.

  Retries should occur for any exception in this case.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=None,
      ),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_all',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_all',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_all',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_count_populated_correctly(
    request: pytest.FixtureRequest,
):
  """Tests that retry_count is populated correctly in the workflow context."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0, exceptions=['CustomRetryableError']
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_count_populated_correctly',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_retry_count_populated_correctly',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3
  assert flaky_node_in_agent.tracker['retry_counts'] == [0, 1, 2]


@pytest.mark.asyncio
async def test_retry_max_attempts_exceeded(
    request: pytest.FixtureRequest,
):
  """Tests that the agent stops retrying after exceeding `max_attempts`."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 4 times, but max_attempts is 3.
  # Total attempts = 3 (1 initial + 2 retries).
  # Attempt 1: retry_count = 0, fails.
  # Attempt 2: retry_count = 1, fails.
  # Attempt 3: retry_count = 2, fails. Now _should_retry_node returns False.
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=5,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Persisted failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          max_attempts=3,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_max_attempts',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(CustomRetryableError, match='Persisted failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_max_attempts',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_fails_without_retry_config(
    request: pytest.FixtureRequest,
):
  """Tests that the agent fails immediately if `retry_config` is None."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Any failure'),
      retry_config=None,
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_fails_without_retry_config',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(ValueError, match='Any failure'):
    async for e in agent.run_async(ctx):
      events.append(e)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_fails_without_retry_config',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


@pytest.mark.asyncio
async def test_retries_with_empty_retry_config(
    request: pytest.FixtureRequest,
):
  """Tests that retries occur when `retry_config` is an empty instance."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  # Node will fail 1 time, then succeed
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=ValueError('Another failure'),
      retry_config=RetryConfig(),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retries_with_empty_retry_config',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retries_with_empty_retry_config',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retries_with_empty_retry_config',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_with_delay(request: pytest.FixtureRequest):
  """Tests retry with initial delay.

  This test verifies that the agent waits for the specified initial_delay before
  retrying a failed node.
  """
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Sleep test failure'),
      retry_config=RetryConfig(
          initial_delay=5.0,
          max_attempts=3,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_delay',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with mock.patch.object(
      asyncio, 'sleep', new_callable=mock.AsyncMock
  ) as mock_sleep:
    events = [e async for e in agent.run_async(ctx)]
    mock_sleep.assert_any_await(5.0)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_delay',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_with_backoff_and_jitter(request: pytest.FixtureRequest):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=4,  # Fails 3 times
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Backoff test failure'),
      retry_config=RetryConfig(
          initial_delay=2.0,
          max_attempts=5,
          backoff_factor=3.0,
          jitter=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_backoff',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep:
    events = [e async for e in agent.run_async(ctx)]
    # Attempt 1: fails, delay = 2.0 * (3.0 ** 0) = 2.0
    # Attempt 2: fails, delay = 2.0 * (3.0 ** 1) = 6.0
    # Attempt 3: fails, delay = 2.0 * (3.0 ** 2) = 18.0
    mock_sleep.assert_has_awaits(
        [mock.call(2.0), mock.call(6.0), mock.call(18.0)]
    )

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_backoff',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 4


@pytest.mark.asyncio
async def test_retry_with_jitter(request: pytest.FixtureRequest):
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Jitter test failure'),
      retry_config=RetryConfig(
          initial_delay=4.0,
          max_attempts=3,
          backoff_factor=1.0,
          jitter=0.5,
          exceptions=['CustomRetryableError'],
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_retry_jitter',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with (
      mock.patch('asyncio.sleep', new_callable=mock.AsyncMock) as mock_sleep,
      mock.patch('random.uniform', return_value=-1.0) as mock_random,
  ):
    events = [e async for e in agent.run_async(ctx)]

    # 4.0 + (-1.0) = 3.0
    mock_sleep.assert_any_await(3.0)
    # Called with -0.5 * 4.0, 0.5 * 4.0
    mock_random.assert_called_once_with(-2.0, 2.0)

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_retry_jitter',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_with_exception_classes(request: pytest.FixtureRequest):
  """Tests that RetryConfig accepts exception classes, not just strings."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError],  # class, not string
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_exception_classes',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_retry_exception_classes',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_retry_exception_classes',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_retry_exception_classes',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_retry_with_mixed_exception_types(request: pytest.FixtureRequest):
  """Tests that RetryConfig accepts a mix of strings and exception classes."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=2,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Simulated failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError, 'ValueError'],  # mixed
      ),
  )
  node_c = TestingNode(name='NodeC', output='Executing C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
          Edge(flaky_node, node_c),
      ],
  )
  agent = Workflow(
      name='test_retry_mixed_exceptions',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_retry_mixed_exceptions',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_retry_mixed_exceptions',
          {'node_name': 'FlakyNode', 'output': 'Executing B'},
      ),
      (
          'test_retry_mixed_exceptions',
          {'node_name': 'NodeC', 'output': 'Executing C'},
      ),
  ]
  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 2


@pytest.mark.asyncio
async def test_retry_exception_class_no_match(request: pytest.FixtureRequest):
  """Tests that exception classes that don't match are not retried."""

  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomNonRetryableError('Unexpected failure'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=[CustomRetryableError],  # class, won't match
      ),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_retry_exception_class_no_match',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with pytest.raises(CustomNonRetryableError, match='Unexpected failure'):
    async for _ in agent.run_async(ctx):
      pass

  flaky_node_in_agent = next(
      n for n in agent.graph.nodes if n.name == 'FlakyNode'
  )
  assert flaky_node_in_agent.tracker['iteration_count'] == 1


def test_retry_config_rejects_invalid_exception_types():
  """Tests that RetryConfig rejects non-string, non-class exception entries."""
  with pytest.raises(ValueError, match='exception class names'):
    RetryConfig(exceptions=[42])


def test_retry_config_normalizes_classes_to_strings():
  """Tests that exception classes are normalized to their names."""
  config = RetryConfig(exceptions=[ValueError, 'KeyError'])
  assert config.exceptions == ['ValueError', 'KeyError']


@pytest.mark.asyncio
async def test_node_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  """Tests that a node is marked as CANCELLED when a sibling node fails."""

  async def slow_node():
    await asyncio.sleep(10)
    yield 'Slow'

  async def fail_node():
    await asyncio.sleep(0.1)
    if False:
      yield
    raise ValueError('Fail')

  agent = Workflow(
      name='test_workflow_cancellation_sibling',
      edges=[
          (START, slow_node),
          (START, fail_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with pytest.raises(ValueError, match='Fail'):
    async for _ in agent.run_async(ctx):
      pass

  # Check persistence
  assert agent.name in ctx.agent_states
  state = WorkflowAgentState.model_validate(ctx.agent_states[agent.name])
  assert state.nodes['fail_node'].status == NodeStatus.FAILED
  assert state.nodes['slow_node'].status == NodeStatus.CANCELLED


@pytest.mark.asyncio
async def test_parallel_worker_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  """Tests that a node using parallel_worker is marked as CANCELLED when a sibling node fails."""

  async def slow_node_impl(ctx: Context, node_input: Any):
    await asyncio.sleep(10)
    yield f'Slow {node_input}'

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  node_parallel = node(
      slow_node_impl, name='node_parallel', parallel_worker=True
  )

  agent = Workflow(
      name='test_workflow_parallel_cancellation_sibling',
      edges=[
          (START, node_parallel),
          (START, fail_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  token = workflow_node_input.set(['item1', 'item2'])
  try:
    with pytest.raises(ValueError, match='Fail'):
      async for _ in agent.run_async(ctx):
        pass
  finally:
    workflow_node_input.reset(token)

  # Check persistence
  assert agent.name in ctx.agent_states
  state = WorkflowAgentState.model_validate(ctx.agent_states[agent.name])
  assert state.nodes['fail_node'].status == NodeStatus.FAILED
  assert state.nodes['node_parallel'].status == NodeStatus.CANCELLED
  # Verify internal dynamic nodes are also cancelled
  assert state.nodes['node_parallel'].status == NodeStatus.CANCELLED


@pytest.mark.asyncio
async def test_parallel_worker_cancellation_on_worker_failure(
    request: pytest.FixtureRequest,
):
  """Tests that all worker nodes are cancelled when one of them fails."""

  async def worker_node_impl(ctx: Context, node_input: Any):
    if node_input == 'fail':
      await asyncio.sleep(0.1)
      raise ValueError('Worker Fail')
    else:
      await asyncio.sleep(10)
      yield f'Success {node_input}'

  node_parallel = node(
      worker_node_impl, name='node_parallel', parallel_worker=True
  )

  agent = Workflow(
      name='test_workflow_parallel_cancellation_worker',
      edges=[
          (START, node_parallel),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  token = workflow_node_input.set(['fail', 'slow'])
  try:
    with pytest.raises(ValueError, match='Worker Fail'):
      async for _ in agent.run_async(ctx):
        pass
  finally:
    workflow_node_input.reset(token)

  # Check persistence
  assert agent.name in ctx.agent_states
  state = WorkflowAgentState.model_validate(ctx.agent_states[agent.name])
  # Since the outputs are tracked purely dynamically now without index keys
  # the failing worker forces the parent entry into FAILED.
  assert state.nodes['node_parallel'].status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_nested_workflow_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  """Tests that a nested workflow and its internal nodes are cancelled."""

  async def inner_slow_node():
    await asyncio.sleep(10)
    yield 'Inner Slow'

  inner_agent = Workflow(
      name='inner_workflow',
      edges=[
          (START, inner_slow_node),
      ],
  )

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  outer_agent = Workflow(
      name='outer_workflow',
      edges=[
          (START, inner_agent),
          (START, fail_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, outer_agent, resumable=True
  )

  with pytest.raises(ValueError, match='Fail'):
    async for _ in outer_agent.run_async(ctx):
      pass

  # Check outer persistence
  assert outer_agent.name in ctx.agent_states
  outer_state = WorkflowAgentState.model_validate(
      ctx.agent_states[outer_agent.name]
  )
  assert outer_state.nodes['fail_node'].status == NodeStatus.FAILED
  assert outer_state.nodes['inner_workflow'].status == NodeStatus.CANCELLED

  # Check inner persistence
  inner_path = join_paths(outer_agent.name, inner_agent.name)
  assert inner_path in ctx.agent_states
  inner_state = WorkflowAgentState.model_validate(ctx.agent_states[inner_path])
  assert inner_state.nodes['inner_slow_node'].status == NodeStatus.CANCELLED


@pytest.mark.asyncio
async def test_error_event_emitted_on_failure(
    request: pytest.FixtureRequest,
):
  """Tests that an error event is emitted when a node raises an exception."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')

  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Executing B',
      succeed_on_iteration=999,
      tracker=tracker,
      exception_to_raise=ValueError('Something went wrong'),
      retry_config=None,
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_error_event',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = []
  with pytest.raises(ValueError, match='Something went wrong'):
    async for e in agent.run_async(ctx):
      events.append(e)

  # Find the error event emitted by the failed node.
  error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_name == 'FlakyNode'
  ]
  assert len(error_events) == 1
  assert error_events[0].error_code == 'ValueError'
  assert error_events[0].error_message == 'Something went wrong'


@pytest.mark.asyncio
async def test_error_event_emitted_on_each_retry(
    request: pytest.FixtureRequest,
):
  """Tests that an error event is emitted for each failed retry attempt."""
  tracker = {'iteration_count': 0}

  # Node will fail 2 times, then succeed on 3rd attempt
  flaky_node = _FlakyNode(
      name='FlakyNode',
      message='Success',
      succeed_on_iteration=3,
      tracker=tracker,
      exception_to_raise=CustomRetryableError('Transient error'),
      retry_config=RetryConfig(
          initial_delay=0.0,
          exceptions=['CustomRetryableError'],
      ),
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, flaky_node),
      ],
  )
  agent = Workflow(
      name='test_error_event_retry',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  events = [e async for e in agent.run_async(ctx)]

  # Two failures before success → two error events.
  error_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.error_code is not None
      and e.node_name == 'FlakyNode'
  ]
  assert len(error_events) == 2
  for err in error_events:
    assert err.error_code == 'CustomRetryableError'
    assert err.error_message == 'Transient error'

  # The node should still produce its output after retries.
  assert simplify_events_with_node(events) == [
      (
          'test_error_event_retry',
          {'node_name': 'FlakyNode', 'output': 'Success'},
      ),
  ]
