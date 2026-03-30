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

import asyncio
from typing import Any
from typing import AsyncGenerator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.apps import app
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._node_runner import _check_and_schedule_nodes
from google.adk.workflow._node_runner import _execute_node
from google.adk.workflow._node_runner import _NodeCompletion
from google.adk.workflow._node_runner import _WorkflowRunState
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._trigger import Trigger
from google.adk.workflow._trigger_processor import _cleanup_child_runs
from google.adk.workflow._workflow import WorkflowAgentState
from google.adk.workflow._workflow_graph import WorkflowGraph
from pydantic import ConfigDict
from pydantic import Field
import pytest
import pytest_asyncio
from typing_extensions import override

from . import testing_utils


class SimpleNode(BaseNode):
  """A simple node for testing that yields a single event."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  output_data: Any = None

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if self.output_data is not None:
      yield Event(output=self.output_data)


class MultiEventNode(BaseNode):
  """A node that yields multiple events for testing."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  events: list[Any] = Field(default_factory=list)

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for event in self.events:
      yield event


class TestCleanupChildExecutions:
  """Tests for _cleanup_child_runs function."""

  def test_removes_only_terminal_children(self):
    """Test that terminal children are removed while non-terminal are kept."""
    agent_state = WorkflowAgentState()
    parent_run_id = 'parent-exec-123'

    agent_state.nodes['parent_node'] = NodeState(
        status=NodeStatus.COMPLETED,
        run_id=parent_run_id,
    )
    agent_state.nodes['child_completed'] = NodeState(
        status=NodeStatus.COMPLETED,
        run_id='child-exec-1',
        parent_run_id=parent_run_id,
    )
    agent_state.nodes['child_failed'] = NodeState(
        status=NodeStatus.FAILED,
        run_id='child-exec-2',
        parent_run_id=parent_run_id,
    )
    agent_state.nodes['child_cancelled'] = NodeState(
        status=NodeStatus.CANCELLED,
        run_id='child-exec-3',
        parent_run_id=parent_run_id,
    )
    agent_state.nodes['child_running'] = NodeState(
        status=NodeStatus.RUNNING,
        run_id='child-exec-4',
        parent_run_id=parent_run_id,
    )

    _cleanup_child_runs(parent_run_id, agent_state)

    assert 'child_completed' not in agent_state.nodes
    assert 'child_failed' not in agent_state.nodes
    assert 'child_cancelled' not in agent_state.nodes
    assert 'child_running' in agent_state.nodes
    assert 'parent_node' in agent_state.nodes

  def test_preserves_non_terminal_children(self):
    """Test that running/pending children are not removed."""
    agent_state = WorkflowAgentState()
    parent_run_id = 'parent-exec-123'

    agent_state.nodes['child_running'] = NodeState(
        status=NodeStatus.RUNNING,
        run_id='child-exec-1',
        parent_run_id=parent_run_id,
    )
    agent_state.nodes['child_pending'] = NodeState(
        status=NodeStatus.PENDING,
        run_id='child-exec-2',
        parent_run_id=parent_run_id,
    )
    agent_state.nodes['child_waiting'] = NodeState(
        status=NodeStatus.WAITING,
        run_id='child-exec-3',
        parent_run_id=parent_run_id,
    )

    _cleanup_child_runs(parent_run_id, agent_state)

    assert 'child_running' in agent_state.nodes
    assert 'child_pending' in agent_state.nodes
    assert 'child_waiting' in agent_state.nodes

  def test_does_not_remove_unrelated_nodes(self):
    """Test that nodes with different parent_run_id are preserved."""
    agent_state = WorkflowAgentState()

    agent_state.nodes['child_of_other'] = NodeState(
        status=NodeStatus.COMPLETED,
        run_id='other-exec',
        parent_run_id='other-parent',
    )
    agent_state.nodes['no_parent'] = NodeState(
        status=NodeStatus.COMPLETED,
        run_id='solo-exec',
    )

    _cleanup_child_runs('parent-exec-123', agent_state)

    assert 'child_of_other' in agent_state.nodes
    assert 'no_parent' in agent_state.nodes

  def test_handles_empty_agent_state(self):
    """Test cleanup with empty agent state does not raise."""
    agent_state = WorkflowAgentState()

    _cleanup_child_runs('any-parent-id', agent_state)

    assert len(agent_state.nodes) == 0


def _get_final_agent_state_nodes(events: list[Event]) -> dict:
  """Extract the nodes dict from the last checkpoint event."""
  for event in reversed(events):
    if event.actions.agent_state is not None and isinstance(
        event.actions.agent_state, dict
    ):
      return event.actions.agent_state.get('nodes', {})
  return {}


class TestChildExecutionCleanupE2E:
  """End-to-end tests verifying dynamic child state cleanup."""

  @pytest.mark.asyncio
  async def test_child_states_cleaned_after_workflow_completes(self):
    """Dynamic children are removed from agent_state after completion."""

    @node
    def func_a() -> str:
      return 'A'

    @node
    def func_b() -> str:
      return 'B'

    @node(rerun_on_resume=True)
    async def parent_fn(ctx: Context) -> str:
      output_a = await ctx.run_node(func_a)
      output_b = await ctx.run_node(func_b)
      return f'{output_a} & {output_b}'

    agent = Workflow(
        name='test_cleanup_wf',
        edges=[(START, parent_fn)],
    )
    test_app = app.App(
        name='test_child_cleanup',
        root_agent=agent,
        resumability_config=app.ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=test_app)
    events = await runner.run_async('start')

    final_nodes = _get_final_agent_state_nodes(events)
    for node_name, node_state in final_nodes.items():
      assert (
          node_state.get('parent_run_id') is None
      ), f'Dynamic child {node_name!r} was not cleaned up'

  @pytest.mark.asyncio
  async def test_fire_and_forget_child_cleaned_after_parent(self):
    """A child completing after its parent is cleaned via self-cleanup."""

    @node
    async def slow_child() -> str:
      await asyncio.sleep(0.1)
      return 'child done'

    @node(rerun_on_resume=True)
    async def parent_fn(ctx: Context) -> str:
      asyncio.ensure_future(ctx.run_node(slow_child))
      # Yield control so the background task schedules the child.
      await asyncio.sleep(0.05)
      return 'parent done'

    agent = Workflow(
        name='test_fire_forget_wf',
        edges=[(START, parent_fn)],
    )
    test_app = app.App(
        name='test_fire_forget',
        root_agent=agent,
        resumability_config=app.ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=test_app)
    events = await runner.run_async('start')

    final_nodes = _get_final_agent_state_nodes(events)
    for node_name, node_state in final_nodes.items():
      assert (
          node_state.get('parent_run_id') is None
      ), f'Dynamic child {node_name!r} was not cleaned up'


class TestCheckAndScheduleNodes:
  """Tests for _check_and_schedule_nodes function."""

  @pytest.fixture
  def simple_graph(self):
    """Create a simple graph with START -> node_a -> node_b."""
    node_a = SimpleNode(name='node_a')
    node_b = SimpleNode(name='node_b')
    return WorkflowGraph(
        edges=[
            Edge(from_node=START, to_node=node_a),
            Edge(from_node=node_a, to_node=node_b),
        ]
    )

  @pytest.fixture
  def nodes_map(self, simple_graph):
    """Create a nodes map from the graph."""
    return {node.name: node for node in simple_graph.nodes}

  @pytest_asyncio.fixture
  async def mock_run_state(self, simple_graph, nodes_map):
    """Create a mock WorkflowRunState for testing."""
    agent = testing_utils.create_test_agent()
    ctx = await testing_utils.create_invocation_context(agent)

    agent_state = WorkflowAgentState()

    return _WorkflowRunState(
        ctx=ctx,
        event_queue=asyncio.Queue(),
        graph=simple_graph,
        node_path='test_workflow',
        agent_state=agent_state,
        nodes_map=nodes_map,
        running_tasks={},
        dynamic_futures={},
        local_output_events=[],
        static_node_names=set(nodes_map.keys()),
        transfer_targets=[],
    )

  @pytest.mark.asyncio
  async def test_schedules_pending_nodes(self, mock_run_state):
    """Test that PENDING nodes are scheduled."""
    # Set node_a to PENDING
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.PENDING,
        input='test_input',
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should now have a running task
    assert 'node_a' in mock_run_state.running_tasks
    # Clean up the task
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_schedules_running_nodes_on_resume(self, mock_run_state):
    """Test that RUNNING nodes without tasks are scheduled (resume case)."""
    # Set node_a to RUNNING but not in running_tasks (simulating resume)
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.RUNNING,
        run_id='exec-123',
        input='test_input',
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should now have a running task
    assert 'node_a' in mock_run_state.running_tasks
    # Clean up the task
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_does_not_reschedule_running_nodes_with_tasks(
      self, mock_run_state
  ):
    """Test that RUNNING nodes with existing tasks are not rescheduled."""
    # Set node_a to RUNNING with an existing task
    mock_run_state.agent_state.nodes['node_a'] = NodeState(
        status=NodeStatus.RUNNING,
        run_id='exec-123',
        input='test_input',
    )
    existing_task = mock.MagicMock()
    mock_run_state.running_tasks['node_a'] = existing_task

    _check_and_schedule_nodes(mock_run_state)

    # Task should remain the same
    assert mock_run_state.running_tasks['node_a'] is existing_task

  @pytest.mark.asyncio
  async def test_processes_trigger_buffer(self, mock_run_state):
    """Test that buffered triggers are processed and nodes scheduled."""
    # Add a trigger to the buffer
    mock_run_state.agent_state.trigger_buffer.setdefault('node_a', []).append(
        Trigger(input='buffered_input', triggered_by='__START__')
    )

    _check_and_schedule_nodes(mock_run_state)

    # node_a should be scheduled
    assert 'node_a' in mock_run_state.running_tasks
    # Trigger buffer should be empty
    assert 'node_a' not in mock_run_state.agent_state.trigger_buffer
    # Node state should have the input from the trigger
    node_state = mock_run_state.agent_state.nodes['node_a']
    assert node_state.input == 'buffered_input'
    assert node_state.triggered_by == '__START__'
    # Status is RUNNING because _schedule_node sets it to RUNNING
    assert node_state.status == NodeStatus.RUNNING
    # Clean up
    mock_run_state.running_tasks['node_a'].cancel()

  @pytest.mark.asyncio
  async def test_cleans_up_removed_dynamic_nodes(self, mock_run_state):
    """Test that dynamic nodes removed from state are cleaned from nodes_map."""
    # Add a dynamic node to nodes_map but not to agent_state.nodes
    dynamic_node = SimpleNode(name='dynamic_node')
    mock_run_state.nodes_map['dynamic_node'] = dynamic_node
    # It's not in static_node_names and not in agent_state.nodes

    _check_and_schedule_nodes(mock_run_state)

    # Dynamic node should be removed from nodes_map
    assert 'dynamic_node' not in mock_run_state.nodes_map


class TestExecuteNode:
  """Tests for _execute_node function."""

  @pytest_asyncio.fixture
  async def mock_ctx(self):
    """Create a mock InvocationContext."""
    agent = testing_utils.create_test_agent()
    return await testing_utils.create_invocation_context(agent)

  @pytest.fixture
  def mock_schedule_dynamic_node(self):
    """Create a mock schedule_dynamic_node function."""
    return mock.MagicMock(return_value=asyncio.Future())

  @pytest.mark.asyncio
  async def test_execute_node_yields_events(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node yields events from node.run."""
    node = SimpleNode(name='test_node', output_data={'key': 'value'})

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input='test_input',
        triggered_by='__START__',
        in_nodes={'__START__'},
        run_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].output == {'key': 'value'}

  @pytest.mark.asyncio
  async def test_execute_node_assigns_author(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns author to events."""
    node = SimpleNode(name='test_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='exec-123',
        current_node_path='my_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].author == 'my_workflow'

  @pytest.mark.asyncio
  async def test_execute_node_assigns_node_name(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns node_name to events."""
    node = SimpleNode(name='my_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].node_name == 'my_node'

  @pytest.mark.asyncio
  async def test_execute_node_assigns_run_id(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node assigns run_id to events."""
    node = SimpleNode(name='test_node', output_data='output')

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='my-exec-id',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].node_info.run_id == 'my-exec-id'

  @pytest.mark.asyncio
  async def test_execute_node_with_multiple_events(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that _execute_node handles multiple events from a node."""
    node = MultiEventNode(
        name='multi_node',
        events=[
            Event(output='first'),
            Event(output='second'),
            Event(output='third'),
        ],
    )

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 3
    assert events[0].output == 'first'
    assert events[1].output == 'second'
    assert events[2].output == 'third'

  @pytest.mark.asyncio
  async def test_execute_node_converts_non_event_to_event(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that non-Event yields are wrapped in Event."""
    node = MultiEventNode(name='test_node', events=['raw_data'])

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], Event)
    assert events[0].output == 'raw_data'

  @pytest.mark.asyncio
  async def test_execute_node_skips_none_yields(
      self, mock_ctx, mock_schedule_dynamic_node
  ):
    """Test that None yields are skipped."""
    node = MultiEventNode(
        name='test_node', events=[None, Event(output='valid'), None]
    )

    events = []
    async for event in _execute_node(
        node=node,
        ctx=mock_ctx,
        node_input=None,
        triggered_by='',
        in_nodes=set(),
        run_id='exec-123',
        current_node_path='test_workflow',
        schedule_dynamic_node=mock_schedule_dynamic_node,
    ):
      events.append(event)

    assert len(events) == 1
    assert events[0].output == 'valid'


class TestNodeCompletion:
  """Tests for _NodeCompletion dataclass."""

  def test_default_values(self):
    """Test default values of _NodeCompletion."""
    completion = _NodeCompletion(node_name='test_node')

    assert completion.node_name == 'test_node'
    assert completion.run_id is None
    assert completion.node_interrupted is False
    assert completion.interrupt_ids == []
    assert completion.has_output is False
    assert completion.exception is None

  def test_with_interrupt(self):
    """Test _NodeCompletion with interrupt."""
    completion = _NodeCompletion(
        node_name='test_node',
        run_id='exec-123',
        node_interrupted=True,
        interrupt_ids=['int-1', 'int-2'],
    )

    assert completion.node_interrupted is True
    assert completion.interrupt_ids == ['int-1', 'int-2']

  def test_with_exception(self):
    """Test _NodeCompletion with exception."""
    error = ValueError('test error')
    completion = _NodeCompletion(
        node_name='test_node',
        exception=error,
    )

    assert completion.exception is error
