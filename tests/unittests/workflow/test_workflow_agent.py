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

"""Testings for the Workflow."""

import asyncio
from collections import Counter
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import unittest

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event import Event as AdkEvent
from google.adk.events.event_actions import EventActions
from google.adk.workflow import BaseNode
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._trigger import Trigger
from google.adk.workflow._workflow import NodeState
from google.adk.workflow._workflow import WorkflowAgentState
from google.adk.workflow._workflow_graph import WorkflowGraph
from google.adk.workflow.utils._node_path_utils import is_descendant
from google.adk.workflow.utils._node_path_utils import is_direct_child
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import simplify_events_with_node_and_agent_state
from .workflow_testing_utils import TestingNode
from .workflow_testing_utils import TestingNodeWithIntermediateContent


@pytest.mark.asyncio
async def test_run_async(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='Hello')
  node_b = TestingNode(name='NodeB', output='World')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent',
      graph=graph,
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('test_workflow_agent', {'node_name': 'NodeA', 'output': 'Hello'}),
      ('test_workflow_agent', {'node_name': 'NodeB', 'output': 'World'}),
  ]


@pytest.mark.asyncio
async def test_run_async_with_intermediate_content(
    request: pytest.FixtureRequest,
):
  node_a = TestingNodeWithIntermediateContent(
      name='NodeA',
      intermediate_content=[
          types.Content(parts=[types.Part(text='A message')]),
          types.Content(parts=[types.Part(text='Another message')]),
      ],
      output='A output',
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent',
      graph=graph,
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('NodeA', 'A message'),
      ('NodeA', 'Another message'),
      ('test_workflow_agent', {'node_name': 'NodeA', 'output': 'A output'}),
  ]


class IncrementingNode(BaseNode):
  """A node that increments a value in the tracker."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  message: str = Field(default='')
  tracker: dict[str, Any] = Field(default_factory=dict)

  def __init__(
      self,
      *,
      name: str,
      message: str,
      tracker: dict[str, Any],
  ):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'message', message)
    object.__setattr__(self, 'tracker', tracker)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    current_value = self.tracker.get('iteration_count', 0)
    new_value = current_value + 1
    self.tracker['iteration_count'] = new_value
    yield Event(
        output=self.message,
        route='continue_loop' if new_value < 3 else 'exit_loop',
    )


@pytest.mark.asyncio
async def test_run_async_with_loop_and_break(request: pytest.FixtureRequest):
  """Tests a simple loop with a break condition."""
  tracker = {'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Looping')
  check_node = IncrementingNode(
      name='CheckNode',
      message='Checking',
      tracker=tracker,
  )
  node_b = TestingNode(name='NodeB', output='Finished')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          # TODO: b/474675592 - this requires matching check node to string in
          # edge definition, bug prone. consider fix.
          Edge(node_a, check_node),
          Edge(
              check_node,
              node_a,
              route='continue_loop',
          ),
          Edge(
              check_node,
              node_b,
              route='exit_loop',
          ),
      ],
  )
  agent = Workflow(
      name='loop_agent_with_conditional_break',
      graph=graph,
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'NodeA', 'output': 'Looping'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'CheckNode', 'output': 'Checking'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'NodeA', 'output': 'Looping'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'CheckNode', 'output': 'Checking'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'NodeA', 'output': 'Looping'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'CheckNode', 'output': 'Checking'},
      ),
      (
          'loop_agent_with_conditional_break',
          {'node_name': 'NodeB', 'output': 'Finished'},
      ),
  ]
  assert tracker['iteration_count'] == 3


class _FailableNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  message: str = Field(default='')
  fail_on_iteration: int = Field(default=0)
  tracker: dict[str, Any] = Field(default_factory=dict)

  def __init__(
      self,
      *,
      name: str,
      message: str,
      fail_on_iteration: int,
      tracker: dict[str, Any],
  ):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'message', message)
    object.__setattr__(self, 'fail_on_iteration', fail_on_iteration)
    object.__setattr__(self, 'tracker', tracker)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    iteration_count = self.tracker.get('iteration_count', 0)

    if (
        not self.tracker.get('has_failed', False)
        and iteration_count == self.fail_on_iteration
    ):
      self.tracker['has_failed'] = True
      raise ValueError('Artificial failure')

    yield Event(
        output=self.message,
    )


@pytest.mark.asyncio
async def test_resume_behavior(request: pytest.FixtureRequest):
  tracker = {'has_failed': False, 'iteration_count': 0}
  node_a = TestingNode(name='NodeA', output='Executing A')
  fail_node = _FailableNode(
      name='FailNode',
      message='Executing B',
      fail_on_iteration=1,
      tracker=tracker,
  )
  check_node = IncrementingNode(
      name='CheckNode',
      message='Executing C',
      tracker=tracker,
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, fail_node),
          Edge(fail_node, check_node),
          Edge(
              check_node,
              node_a,
              route='continue_loop',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_resume',
      graph=graph,
  )

  # Run 1: fails
  ctx1 = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  ctx1.user_content = testing_utils.UserContent('start it')

  events1 = []
  with pytest.raises(ValueError, match='Artificial failure'):
    async for e in agent.run_async(ctx1):
      events1.append(e)

  # Iteration 0: A, B, C. count becomes 1.
  # Iteration 1: A runs. B fails.
  assert simplify_events_with_node(events1) == [
      (
          'test_workflow_agent_resume',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'FailNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'CheckNode', 'output': 'Executing C'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
  ]
  assert tracker['iteration_count'] == 1

  # Constructing agent state to simulate resume. Marking Failed node as PENDING.
  agent_state = WorkflowAgentState(
      nodes={
          node_a.name: NodeState(status=NodeStatus.COMPLETED),
          fail_node.name: NodeState(status=NodeStatus.PENDING),
          check_node.name: NodeState(status=NodeStatus.COMPLETED),
      }
  ).model_dump(mode='json')

  # Run 2: resume
  ctx2 = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  ctx2.agent_states[agent.name] = agent_state
  ctx2.invocation_id = events1[0].invocation_id

  events2 = [e async for e in agent.run_async(ctx2)]

  # Resumes from B in iteration 1. count is 1.
  # Iteration 1 continued: B, C. count becomes 2.
  # Iteration 2: A, B, C. count becomes 3. Loop terminates.
  assert simplify_events_with_node(events2) == [
      (
          'test_workflow_agent_resume',
          {'node_name': 'FailNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'CheckNode', 'output': 'Executing C'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'NodeA', 'output': 'Executing A'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'FailNode', 'output': 'Executing B'},
      ),
      (
          'test_workflow_agent_resume',
          {'node_name': 'CheckNode', 'output': 'Executing C'},
      ),
  ]
  assert tracker['iteration_count'] == 3


@pytest.mark.asyncio
async def test_agent_state_event_recorded(request: pytest.FixtureRequest):
  """Verifies that agent_state events are correctly recorded."""
  node_a = TestingNode(name='NodeA', output='Hello A')
  node_b = TestingNode(name='NodeB', output='Hello B')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  agent_name = 'test_workflow_agent_state'
  agent = Workflow(
      name=agent_name,
      graph=graph,
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  events = [e async for e in agent.run_async(ctx)]
  simplified_events = simplify_events_with_node_and_agent_state(
      events, include_inputs_and_triggers=True
  )

  assert simplified_events == [
      (
          agent_name,
          {
              'nodes': {
                  node_a.name: {
                      'status': NodeStatus.RUNNING.value,
                      'triggered_by': START.name,
                  },
              }
          },
      ),
      (agent_name, {'node_name': 'NodeA', 'output': 'Hello A'}),
      (
          agent_name,
          {
              'nodes': {
                  node_a.name: {
                      'status': NodeStatus.COMPLETED.value,
                  },
                  node_b.name: {
                      'status': NodeStatus.RUNNING.value,
                      'triggered_by': node_a.name,
                      'input': 'Hello A',
                  },
              }
          },
      ),
      (agent_name, {'node_name': 'NodeB', 'output': 'Hello B'}),
      (
          agent_name,
          {
              'nodes': {
                  node_a.name: {
                      'status': NodeStatus.COMPLETED.value,
                  },
                  node_b.name: {
                      'status': NodeStatus.COMPLETED.value,
                  },
              }
          },
      ),
      (agent_name, testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='Hello')
  node_b = TestingNode(name='NodeB', output='World')
  agent = Workflow(
      name='test_workflow_agent_implicit_graph',
      edges=[
          (START, node_a),
          (node_a, node_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_implicit_graph',
          {'node_name': 'NodeA', 'output': 'Hello'},
      ),
      (
          'test_workflow_agent_implicit_graph',
          {'node_name': 'NodeB', 'output': 'World'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_string_start(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='Hello')
  node_b = TestingNode(name='NodeB', output='World')
  agent = Workflow(
      name='test_workflow_agent_string_start',
      edges=[
          ('START', node_a),
          (node_a, node_b),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_string_start',
          {'node_name': 'NodeA', 'output': 'Hello'},
      ),
      (
          'test_workflow_agent_string_start',
          {'node_name': 'NodeB', 'output': 'World'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_with_edge_combinations(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_workflow_agent_implicit_complex',
      edges=[
          (START, node_a),
          Edge(node_a, node_b),  # Edge object
          (node_b, node_c),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_implicit_complex',
          {'node_name': 'NodeA', 'output': 'A'},
      ),
      (
          'test_workflow_agent_implicit_complex',
          {'node_name': 'NodeB', 'output': 'B'},
      ),
      (
          'test_workflow_agent_implicit_complex',
          {'node_name': 'NodeC', 'output': 'C'},
      ),
  ]


class _StateUpdatingNode(BaseNode):
  """A node that yields an UpdateStateEvent."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  state_delta: dict[str, Any] = Field(default_factory=dict)

  def __init__(self, *, name: str, state_delta: dict[str, Any]):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'state_delta', state_delta)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        state=self.state_delta,
    )


@pytest.mark.asyncio
async def test_run_async_with_update_state_event(
    request: pytest.FixtureRequest,
):
  node_a = _StateUpdatingNode(name='NodeA', state_delta={'key1': 'value1'})
  agent = Workflow(
      name='test_workflow_agent_update_state',
      edges=[
          (START, node_a),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))
  assert runner.session.state['key1'] == 'value1'


class _RawOutputNode(BaseNode):
  """A node that yields raw data."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  output: Any = None

  def __init__(self, *, name: str, output: Any):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'output', output)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield self.output


class _EventNode(BaseNode):
  """A node that yields an Event."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  output: Any = None
  route: Optional[str] = None

  def __init__(self, *, name: str, output: Any, route: str | None = None):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'output', output)
    object.__setattr__(self, 'route', route)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        output=self.output,
        route=self.route,
    )


@pytest.mark.asyncio
async def test_run_async_with_event(
    request: pytest.FixtureRequest,
):
  """Tests that yielding Event with data and route works."""
  node_a = _EventNode(name='NodeA', output='Hello', route='route_b')
  node_b = InputCapturingNode(name='NodeB')
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_event',
      edges=[
          (START, node_a),
          Edge(node_a, node_b, route='route_b'),
          Edge(node_a, node_c, route='route_c'),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert simplify_events_with_node(events) == [
      (
          'test_event',
          {'node_name': 'NodeA', 'output': 'Hello'},
      ),
      (
          'test_event',
          {
              'node_name': 'NodeB',
              'output': {'received': 'Hello'},
          },
      ),
  ]
  assert node_b.received_inputs == ['Hello']
  assert not node_c.received_inputs


@pytest.mark.asyncio
async def test_run_async_with_raw_output_node(
    request: pytest.FixtureRequest,
):
  node_a = _RawOutputNode(name='NodeA', output='Hello')
  agent = Workflow(
      name='test_workflow_agent_raw_output',
      edges=[
          (START, node_a),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_raw_output',
          {'node_name': 'NodeA', 'output': 'Hello'},
      ),
  ]


@pytest.mark.asyncio
async def test_node_output_event_with_content_data(
    request: pytest.FixtureRequest,
):
  """Tests Event with data being types.Content."""

  def content_producing_node_fn() -> types.Content:
    return types.Content(parts=[types.Part(text='hello')])

  agent = Workflow(
      name='test_content_output',
      edges=[
          (START, content_producing_node_fn),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  node_output_events = [
      e
      for e in events
      if isinstance(e, Event) and e.node_name == 'content_producing_node_fn'
  ]
  assert len(node_output_events) == 1
  event = node_output_events[0]
  assert event.output is None
  assert event.content == types.Content(parts=[types.Part(text='hello')])


@pytest.mark.asyncio
async def test_input_propagation_linear(request: pytest.FixtureRequest):
  """Tests if the output of one node is passed as input to the next."""
  node_a = TestingNode(name='NodeA', output={'message': 'from A'})
  node_b = InputCapturingNode(name='NodeB')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  agent = Workflow(
      name='test_input_linear',
      graph=graph,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_b.received_inputs == [{'message': 'from A'}]


@pytest.mark.asyncio
async def test_input_propagation_fan_in_sequential(
    request: pytest.FixtureRequest,
):
  """Tests inputs from different branches arriving in different cycles."""
  node_a = TestingNode(name='NodeA', output={'message': 'from A'})
  node_b = TestingNode(name='NodeB', output={'message': 'from B'})
  node_b2 = TestingNode(name='NodeB2', output={'message': 'from B2'})
  node_b3 = TestingNode(name='NodeB3', output={'message': 'from B3'})
  node_c = InputCapturingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, node_c),
          Edge(node_b, node_b2),
          Edge(node_b2, node_b3),
          Edge(node_b3, node_c),
      ],
  )
  agent = Workflow(
      name='test_input_fan_in_sequential',
      graph=graph,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  unittest.TestCase().assertCountEqual(
      node_c.received_inputs,
      [
          {'message': 'from A'},
          {'message': 'from B3'},
      ],
  )


@pytest.mark.asyncio
async def test_start_node_receives_user_content(request: pytest.FixtureRequest):
  """Tests if the user_content is passed as input to the START node."""
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_input',
      graph=graph,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(
      testing_utils.get_user_content('test message')
  )

  assert node_a.received_inputs == [testing_utils.UserContent('test message')]
  assert simplify_events_with_node(events) == [
      (
          'test_start_node_input',
          {
              'node_name': 'NodeA',
              'output': {'received': testing_utils.UserContent('test message')},
          },
      ),
  ]


class _TriggerCapturingNode(BaseNode):
  """A node that captures the trigger it receives."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  received_triggers: list[str] = Field(default_factory=list)

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'received_triggers', [])

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    self.received_triggers.append(ctx.triggered_by)
    yield Event(
        output={'triggered_by': ctx.triggered_by},
    )


class _InNodesCapturingNode(BaseNode):
  """A node that captures the in_nodes it receives."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  received_in_nodes: list[set[str]] = Field(default_factory=list)

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'received_in_nodes', [])

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    self.received_in_nodes.append(ctx.in_nodes)
    yield Event(
        output={'in_nodes': list(ctx.in_nodes)},
    )


@pytest.mark.asyncio
async def test_triggered_by_fan_in(request: pytest.FixtureRequest):
  """Tests triggered_by() in Context."""
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_x = _TriggerCapturingNode(name='NodeX')
  agent = Workflow(
      name='test_triggered_by',
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, node_x),
          Edge(node_b, node_c),
          Edge(node_c, node_x),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert Counter(node_x.received_triggers) == Counter(['NodeA', 'NodeC'])


@pytest.mark.asyncio
async def test_in_nodes_fan_in_sequential(
    request: pytest.FixtureRequest,
):
  """Tests in_nodes in Context."""
  node_a = TestingNode(name='NodeA', output={'message': 'from A'})
  node_b = TestingNode(name='NodeB', output={'message': 'from B'})
  node_b2 = TestingNode(name='NodeB2', output={'message': 'from B2'})
  node_b3 = TestingNode(name='NodeB3', output={'message': 'from B3'})
  node_c = _InNodesCapturingNode(name='NodeC')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, node_c),
          Edge(node_b, node_b2),
          Edge(node_b2, node_b3),
          Edge(node_b3, node_c),
      ],
  )
  agent = Workflow(
      name='test_in_nodes_fan_in_sequential',
      graph=graph,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  unittest.TestCase().assertCountEqual(
      node_c.received_in_nodes,
      [
          {'NodeA', 'NodeB3'},
          {'NodeA', 'NodeB3'},
      ],
  )


class _NoTriggerNode(BaseNode):
  """A node that yields NoTriggerEvent."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  name: str = Field(default='')

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  wait_for_output: bool = True

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    # Yields a state-update event but no data — with wait_for_output=True,
    # the workflow won't trigger downstream nodes.
    yield Event(state={'seen': True})


@pytest.mark.asyncio
async def test_wait_for_output_suppresses_trigger(
    request: pytest.FixtureRequest,
):
  """Tests that wait_for_output=True prevents downstream when no data is yielded."""
  node_a = _NoTriggerNode(name='NodeA')
  node_b = TestingNode(name='NodeB', output='B')
  agent = Workflow(
      name='test_no_trigger',
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # NodeB should not run because NodeA has wait_for_output=True and
  # yielded no data events.
  assert simplify_events_with_node(events) == []


class _GateNode(BaseNode):
  """A node with wait_for_output that only yields data on the Nth trigger."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  triggers_needed: int = 2

  wait_for_output: bool = True

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    count = (ctx.state.get(f'{self.name}_count') or 0) + 1
    if count >= self.triggers_needed:
      yield Event(output='gate_open', state={f'{self.name}_count': None})
    else:
      yield Event(state={f'{self.name}_count': count})


@pytest.mark.asyncio
async def test_wait_for_output_retrigger_then_complete(
    request: pytest.FixtureRequest,
):
  """Tests WAITING -> re-trigger -> COMPLETED -> downstream runs."""
  gate = _GateNode(name='Gate', triggers_needed=2)
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  downstream = TestingNode(name='Downstream', output='done')
  agent = Workflow(
      name='test_retrigger',
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, gate),
          Edge(node_b, gate),
          Edge(gate, downstream),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Gate receives two triggers (from NodeA and NodeB). On the first it stays
  # WAITING; on the second it produces data and triggers Downstream.
  simplified = simplify_events_with_node(events)
  # NodeA and NodeB run in parallel so order may vary; Gate and Downstream
  # must come after both.
  assert len(simplified) == 4
  first_two = {simplified[0][1]['node_name'], simplified[1][1]['node_name']}
  assert first_two == {'NodeA', 'NodeB'}
  assert simplified[2] == (
      'test_retrigger',
      {'node_name': 'Gate', 'output': 'gate_open'},
  )
  assert simplified[3] == (
      'test_retrigger',
      {'node_name': 'Downstream', 'output': 'done'},
  )


class _TestInput(BaseModel):
  foo: str
  bar: int


@pytest.mark.asyncio
async def test_start_node_with_str_input_schema(request: pytest.FixtureRequest):
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_with_str_input_schema',
      graph=graph,
      input_schema=str,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_input_str = 'some raw text'
  await runner.run_async(testing_utils.get_user_content(user_input_str))

  assert node_a.received_inputs == ['some raw text']


@pytest.mark.asyncio
async def test_start_node_with_int_input_schema(request: pytest.FixtureRequest):
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_with_int_input_schema',
      graph=graph,
      input_schema=int,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_input_str = '123'
  await runner.run_async(testing_utils.get_user_content(user_input_str))

  assert node_a.received_inputs == [123]


@pytest.mark.asyncio
async def test_start_node_with_int_list_input_schema(
    request: pytest.FixtureRequest,
):
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_with_int_input_schema',
      graph=graph,
      input_schema=list[int],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_input_str = '[123, 256]'
  await runner.run_async(testing_utils.get_user_content(user_input_str))

  assert node_a.received_inputs == [[123, 256]]


@pytest.mark.asyncio
async def test_start_node_with_invalid_input_schema(
    request: pytest.FixtureRequest,
):
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_with_invalid_input_schema',
      graph=graph,
      input_schema=_TestInput,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_input_str = '{"foo": "val", "bar": "invalid"}'
  with pytest.raises(
      ValueError, match='Failed to parse input content into schema'
  ):
    await runner.run_async(testing_utils.get_user_content(user_input_str))


@pytest.mark.asyncio
async def test_start_node_receives_parsed_user_content_with_schema(
    request: pytest.FixtureRequest,
):
  """Tests if user_content is parsed with schema and passed to START node."""
  node_a = InputCapturingNode(name='NodeA')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
      ],
  )
  agent = Workflow(
      name='test_start_node_parsed_input',
      graph=graph,
      input_schema=_TestInput,
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_input_str = '{"foo": "val", "bar": 123}'
  events = await runner.run_async(
      testing_utils.get_user_content(user_input_str)
  )

  parsed_input = _TestInput(foo='val', bar=123)
  assert node_a.received_inputs == [parsed_input]
  assert simplify_events_with_node(events) == [
      (
          'test_start_node_parsed_input',
          {
              'node_name': 'NodeA',
              'output': {'received': parsed_input},
          },
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_chain(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_chain',
      edges=[
          (START, node_a, node_b, node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  assert simplify_events_with_node(events) == [
      ('test_chain', {'node_name': 'NodeA', 'output': 'A'}),
      ('test_chain', {'node_name': 'NodeB', 'output': 'B'}),
      ('test_chain', {'node_name': 'NodeC', 'output': 'C'}),
  ]


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_out(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A')
  node_b = InputCapturingNode(name='NodeB')
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_fan_out',
      edges=[
          (START, node_a),
          (node_a, (node_b, node_c)),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_b.received_inputs == ['A']
  assert node_c.received_inputs == ['A']


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_in(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_fan_in',
      edges=[
          (START, (node_a, node_b)),
          ((node_a, node_b), node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert Counter(node_c.received_inputs) == Counter(['A', 'B'])


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_out_fan_in(
    request: pytest.FixtureRequest,
):
  node_s = TestingNode(name='NodeS', output='S')
  node_a = TestingNode(name='NodeA', output='A')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_fan_out_fan_in',
      edges=[
          (START, node_s),
          (node_s, (node_a, node_b), node_c),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_a.received_inputs == ['S']
  assert node_b.received_inputs == ['S']
  assert Counter(node_c.received_inputs) == Counter(['A', 'B'])


class _DelayedMultiEventNode(BaseNode):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  events: list[tuple[str, float]] = Field(default_factory=list)

  def __init__(
      self,
      *,
      name: str,
      events: list[tuple[str, float]],
  ):
    """Events are (message, delay_after_message)."""
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'events', events)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for message, delay in self.events:
      yield AdkEvent(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=types.Content(parts=[types.Part(text=message)]),
      )
      if delay > 0:
        await asyncio.sleep(delay)


@pytest.mark.asyncio
async def test_run_async_parallel_nodes_interleaved_events(
    request: pytest.FixtureRequest,
):
  node_a = _DelayedMultiEventNode(
      name='NodeA',
      events=[('A1', 0.2), ('A2', 0)],
  )
  node_b = _DelayedMultiEventNode(
      name='NodeB',
      events=[('B1', 0.1), ('B2', 0)],
  )
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_parallel',
      graph=graph,
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  simplified_events = testing_utils.simplify_events(events)
  assert len(simplified_events) == 4
  # The first two events are concurrently generated.
  assert set(simplified_events[0:2]) == {('NodeA', 'A1'), ('NodeB', 'B1')}
  assert simplified_events[2:] == [('NodeB', 'B2'), ('NodeA', 'A2')]


@pytest.mark.asyncio
async def test_buffers_events_from_parallel_nodes(
    request: pytest.FixtureRequest,
):
  """Tests that events from parallel nodes are buffered with fan-in."""
  node_a = TestingNode(name='NodeA', output={'a': 1})
  node_b = TestingNode(name='NodeB', output={'b': 2})
  node_capture = InputCapturingNode(name='NodeCapture')

  agent = Workflow(
      name='test_join_node',
      edges=[
          Edge(START, node_a),
          Edge(START, node_b),
          Edge(node_a, node_capture),
          Edge(node_b, node_capture),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  unittest.TestCase().assertCountEqual(
      node_capture.received_inputs, [{'a': 1}, {'b': 2}]
  )


@pytest.mark.asyncio
async def test_execution_id_uniqueness(request: pytest.FixtureRequest):
  """Tests that execution_id is unique per node execution."""
  node_a = TestingNode(name='NodeA', output='A')
  # Loop NodeA 3 times: A -> A -> A (exit)
  tracker = {'count': 0}

  async def loop_controller(ctx: Context, node_input: Any):
    tracker['count'] += 1
    if tracker['count'] < 3:
      return 'continue'
    return 'stop'

  node_a.route = loop_controller

  agent = Workflow(
      name='test_execution_id',
      edges=[
          (START, node_a),
          Edge(node_a, node_a, route='continue'),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  node_a_events = [
      e for e in events if isinstance(e, Event) and e.node_name == 'NodeA'
  ]
  assert len(node_a_events) == 3

  # Check that all have execution_ids
  execution_ids = [e.node_info.execution_id for e in node_a_events]
  assert all(execution_ids)

  # Check that they are all different (unique per execution)
  assert len(set(execution_ids)) == 3


@pytest.mark.asyncio
async def test_execution_id_uniqueness_nested(request: pytest.FixtureRequest):
  """Tests that execution_id is unique in nested workflows."""
  inner_node = TestingNode(name='InnerNode', output='Inner')
  inner_agent = Workflow(
      name='inner_agent',
      edges=[
          (START, inner_node),
      ],
  )

  outer_node_a = TestingNode(name='OuterNodeA', output='OuterA')
  outer_node_b = TestingNode(name='OuterNodeB', output='OuterB')

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          (START, outer_node_a),
          (outer_node_a, inner_agent),
          (inner_agent, outer_node_b),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, outer_agent
  )
  ctx.user_content = testing_utils.UserContent('start outer')
  events = [e async for e in outer_agent.run_async(ctx)]

  node_output_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and is_descendant(outer_agent.name, e.node_info.path)
  ]
  execution_ids = []
  for e in node_output_events:
    if e.node_info.execution_id:
      execution_ids.append(e.node_info.execution_id)

  # We expect 3 node output events:
  # - OuterNodeA (direct child)
  # - InnerNode (descendant via inner_agent)
  # - OuterNodeB (direct child)
  assert len(node_output_events) == 3
  assert len(set(execution_ids)) == 3


@pytest.mark.asyncio
async def test_resume_with_manual_state_verifies_input_persistence(
    request: pytest.FixtureRequest,
):
  """Tests that node inputs are read from state on resume."""
  node_a = TestingNode(name='NodeA', output='original_output')
  node_b = InputCapturingNode(name='NodeB')

  agent = Workflow(
      name='test_manual_state_resume',
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )

  # Manually construct state where NodeA is done and NodeB is pending with input
  agent_state = WorkflowAgentState(
      nodes={
          node_a.name: NodeState(status=NodeStatus.COMPLETED),
          node_b.name: NodeState(
              status=NodeStatus.PENDING,
              input='injected_input_from_state',
              triggered_by=node_a.name,
          ),
      },
  ).model_dump(mode='json')

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  # Inject the state
  ctx.agent_states[agent.name] = agent_state

  # Run agent
  events = [e async for e in agent.run_async(ctx)]

  # NodeB should have run and captured the input from the state
  assert node_b.received_inputs == ['injected_input_from_state']

  # Verify NodeB output event
  simplified = simplify_events_with_node(events)
  assert (
      'test_manual_state_resume',
      {
          'node_name': 'NodeB',
          'output': {'received': 'injected_input_from_state'},
      },
  ) in simplified


@pytest.mark.asyncio
async def test_run_async_with_multiple_node_outputs_fails(
    request: pytest.FixtureRequest,
):
  """Tests that a node producing multiple data events raises ValueError."""

  def multi_output():
    yield 'Output1'
    yield 'Output2'

  agent = Workflow(
      name='test_workflow_agent_multi_output',
      edges=[
          (START, multi_output),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(
      ValueError,
      match='multiple outputs or mixed output/interrupt',
  ):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_in_with_route(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A', route=['route1'])
  node_b = TestingNode(name='NodeB', output='B', route=['route1'])
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_fan_in_route',
      edges=[
          (START, node_a),
          (START, node_b),
          ((node_a, node_b), {'route1': node_c}),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert Counter(node_c.received_inputs) == Counter(['A', 'B'])


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_out_with_route(
    request: pytest.FixtureRequest,
):
  node_r = TestingNode(name='R', output='R', route=['route1'])
  node_b = InputCapturingNode(name='NodeB')
  node_c = InputCapturingNode(name='NodeC')
  agent = Workflow(
      name='test_fan_out_route',
      edges=[
          (START, node_r),
          (node_r, {'route1': (node_b, node_c)}),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_b.received_inputs == ['R']
  assert node_c.received_inputs == ['R']


@pytest.mark.asyncio
async def test_run_async_with_implicit_graph_fan_in_out_with_route(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A', route=['route1'])
  node_b = TestingNode(name='NodeB', output='B', route=['route1'])
  node_c = InputCapturingNode(name='NodeC')
  node_d = InputCapturingNode(name='NodeD')
  agent = Workflow(
      name='test_fan_in_out_route',
      edges=[
          (START, node_a),
          (START, node_b),
          ((node_a, node_b), {'route1': (node_c, node_d)}),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert Counter(node_c.received_inputs) == Counter(['A', 'B'])
  assert Counter(node_d.received_inputs) == Counter(['A', 'B'])


class _SleepyNode(BaseNode):
  """A node that yields an event, sleeps, and yields another event."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='')
  sleep_started: bool = Field(default=False)

  def __init__(self, *, name: str):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'sleep_started', False)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    yield Event(content=types.Content(parts=[types.Part(text='first')]))
    self.sleep_started = True
    await asyncio.sleep(0.5)
    yield Event(output='second')


@pytest.mark.asyncio
async def test_run_async_streaming_behavior(request: pytest.FixtureRequest):
  """Tests that content events stream to a client before sleep completes."""

  node = _SleepyNode(name='SleepyNode')
  agent = Workflow(
      name='test_streaming',
      edges=[(START, node)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  events = []
  sleep_started_values = []
  async for e in agent.run_async(ctx):
    events.append(e)
    if (
        isinstance(e, Event)
        and (e.content or e.output is not None)
        and is_direct_child(e.node_info.path, agent.name)
    ):
      sleep_started_values.append(node.sleep_started)

  # Content event arrives before sleep, data event arrives after.
  assert sleep_started_values == [False, True], (
      "Content event 'first' is yielded before sleep_started is set to True."
      " Data event 'second' is yielded after sleep_started is set to True."
  )


@pytest.mark.parametrize(
    'field, value, error_message',
    [
        (
            'sub_agents',
            [BaseAgent(name='sub')],
            'sub_agents is not supported in Workflow.',
        ),
    ],
)
def test_workflow_agent_unsupported_base_agent_fields(
    field, value, error_message
):
  """Tests that unsupported BaseAgent fields raise ValueError."""
  kwargs = {
      'name': 'test_agent',
      'graph': WorkflowGraph(edges=[]),
      field: value,
  }
  with pytest.raises(ValueError, match=error_message):
    Workflow(**kwargs)


@pytest.mark.asyncio
async def test_node_path_generation(request: pytest.FixtureRequest):
  """Verifies that node_path is correctly generated on events."""
  node_a = TestingNode(name='NodeA', output='Hello')
  agent = Workflow(
      name='test_workflow_agent_path',
      graph=WorkflowGraph(edges=[Edge(START, node_a)]),
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  # Filter for node events (excluding user input echo if any)
  node_events = [e for e in events if e.node_name == 'NodeA']
  assert len(node_events) == 1
  event = node_events[0]

  # For root agent, author_path is likely empty/None, so node_path is just
  # NodeName or it might vary depending on how ctx.node_path is initialized.
  # We check that it ends with NodeA.
  assert event.node_info.path.endswith('NodeA')
  # If we want to be strict about no "None":
  assert 'None' not in event.node_info.path


@pytest.mark.asyncio
async def test_bytes_in_content_output_e2e(request: pytest.FixtureRequest):
  """Tests E2E workflow with a node outputting Content with bytes."""
  content = types.Content(
      parts=[
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG\r\n')
          )
      ]
  )

  node_a = _RawOutputNode(name='NodeA', output=content)
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_bytes_in_content_output_e2e',
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert len(node_b.received_inputs) == 1
  received = node_b.received_inputs[0]
  assert isinstance(received, types.Content)
  assert received.parts[0].inline_data.data == b'\x89PNG\r\n'


@pytest.mark.asyncio
async def test_raw_bytes_output_e2e(request: pytest.FixtureRequest):
  """Tests E2E workflow with a node outputting raw bytes."""
  raw_bytes = b'\x89PNG\r\n'
  node_a = _RawOutputNode(name='NodeA', output=raw_bytes)
  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_raw_bytes_output_e2e',
      edges=[
          Edge(START, node_a),
          Edge(node_a, node_b),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_b.received_inputs == [raw_bytes]


@pytest.mark.asyncio
async def test_bytes_in_node_input_serialization_round_trip(
    request: pytest.FixtureRequest,
):
  """Tests that bytes in NodeState.input survive agent state serialization."""
  raw_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'

  # Build state with bytes in node input, serialize, and deserialize.
  agent_state = WorkflowAgentState(
      nodes={
          'NodeB': NodeState(
              status=NodeStatus.PENDING,
              input=raw_bytes,
              triggered_by='NodeA',
          ),
      },
  )
  dumped = agent_state.model_dump(mode='json')

  # Bytes should be base64-encoded strings in the JSON-safe dict.
  assert isinstance(dumped['nodes']['NodeB']['input'], str)

  # Round-trip through model_validate (simulates loading from session).
  restored = WorkflowAgentState.model_validate(dumped)
  # After round-trip through Any field, bytes become base64 strings.
  # This is expected — nodes should use typed wrappers for binary data.
  assert isinstance(restored.nodes['NodeB'].input, str)


@pytest.mark.asyncio
async def test_bytes_in_typed_model_input_round_trip(
    request: pytest.FixtureRequest,
):
  """Tests that bytes inside typed Pydantic models survive round-trip."""
  content = types.Content(
      parts=[
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG\r\n')
          )
      ]
  )

  agent_state = WorkflowAgentState(
      nodes={
          'NodeB': NodeState(
              status=NodeStatus.PENDING,
              input=content,
              triggered_by='NodeA',
          ),
      },
  )
  dumped = agent_state.model_dump(mode='json')

  # The Content model is serialized to a dict with base64 inline_data.
  node_b_input = dumped['nodes']['NodeB']['input']
  assert isinstance(node_b_input, dict)

  # Reconstruct typed Content from the dict (as FunctionNode would do).
  restored_content = types.Content.model_validate(node_b_input)
  assert restored_content.parts[0].inline_data.data == b'\x89PNG\r\n'


@pytest.mark.asyncio
async def test_bytes_in_trigger_buffer_serialization(
    request: pytest.FixtureRequest,
):
  """Tests that bytes in trigger_buffer survive serialization."""
  raw_bytes = b'\x89PNG\r\n'

  agent_state = WorkflowAgentState(
      nodes={
          'NodeA': NodeState(status=NodeStatus.RUNNING),
      },
      trigger_buffer={
          'NodeB': [Trigger(input=raw_bytes, triggered_by='NodeA')]
      },
  )
  dumped = agent_state.model_dump(mode='json')

  # trigger_buffer should be serialized with base64 string, not raw bytes.
  trigger_input = dumped['trigger_buffer']['NodeB'][0]['input']
  assert isinstance(trigger_input, str)


@pytest.mark.asyncio
async def test_bytes_input_full_workflow_resume(
    request: pytest.FixtureRequest,
):
  """Tests full workflow resume with Content bytes in node input."""
  content = types.Content(
      parts=[
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG\r\n')
          )
      ]
  )

  node_b = InputCapturingNode(name='NodeB')
  agent = Workflow(
      name='test_bytes_resume',
      edges=[
          Edge(START, node_b),
      ],
  )

  # Simulate a persisted state with Content as input.
  agent_state = WorkflowAgentState(
      nodes={
          node_b.name: NodeState(
              status=NodeStatus.PENDING,
              input=content,
              triggered_by=START.name,
          ),
      },
  ).model_dump(mode='json')

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )
  ctx.agent_states[agent.name] = agent_state

  events = [e async for e in agent.run_async(ctx)]

  # NodeB receives a dict (Content was serialized). It can reconstruct
  # the Content object if it has a type hint.
  assert len(node_b.received_inputs) == 1
  received = node_b.received_inputs[0]
  assert isinstance(received, dict)

  # Verify the dict can be reconstructed back to Content with bytes.
  restored_content = types.Content.model_validate(received)
  assert restored_content.parts[0].inline_data.data == b'\x89PNG\r\n'
