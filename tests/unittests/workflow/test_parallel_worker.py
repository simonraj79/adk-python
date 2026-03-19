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

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._dynamic_node_registry import dynamic_node_registry
from google.adk.workflow._node import node
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import testing_utils
from .workflow_testing_utils import simplify_events_with_node


class _ProducerNode(BaseNode):
  """A node that produces a list of items."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  items: list[Any] = Field(default_factory=list)
  name: str = Field(default='Producer')

  def __init__(self, items: list[Any], name: str = 'Producer'):
    super().__init__()
    object.__setattr__(self, 'items', items)
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.items)


class _SingleItemProducerNode(BaseNode):
  """A node that produces a single item."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  item: Any = None
  name: str = Field(default='Producer')

  def __init__(self, item: Any, name: str = 'Producer'):
    super().__init__()
    object.__setattr__(self, 'item', item)
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.item)


class _WorkerNode(BaseNode):
  """A node that processes an item, with an optional delay."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='Worker')

  def __init__(self, name: str = 'Worker'):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if isinstance(node_input, dict) and 'delay' in node_input:
      await asyncio.sleep(node_input['delay'])
      val = node_input['val']
    else:
      val = node_input
    yield Event(output=f'{val}_processed')


@pytest.mark.asyncio
async def test_parallel_worker_simple(request: pytest.FixtureRequest):
  """Tests that ParallelWorker processes items in parallel."""
  # Use delays to ensure deterministic output order of children
  items = [{'val': 'item1', 'delay': 0}, {'val': 'item2', 'delay': 0.1}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent',
          {
              'node_name': 'NodeA',
              'output': [
                  {'val': 'item1', 'delay': 0},
                  {'val': 'item2', 'delay': 0.1},
              ],
          },
      ),
      # Children outputs
      ('test_agent', {'node_name': 'Worker__0', 'output': 'item1_processed'}),
      ('test_agent', {'node_name': 'Worker__1', 'output': 'item2_processed'}),
      # Parent output
      (
          'test_agent',
          {
              'node_name': 'Worker',
              'output': ['item1_processed', 'item2_processed'],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_empty_input(request: pytest.FixtureRequest):
  """Tests that ParallelWorker handles empty list input."""
  items = []
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_empty_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_empty_agent',
          {
              'node_name': 'NodeA',
              'output': [],
          },
      ),
      # Parent output
      (
          'test_empty_agent',
          {
              'node_name': 'Worker',
              'output': [],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_single_item_input(
    request: pytest.FixtureRequest,
):
  """Tests that ParallelWorker handles single item input."""
  item = {'val': 'item1', 'delay': 0}
  node_a = _SingleItemProducerNode(item=item, name='NodeA')
  worker = ParallelWorker(_WorkerNode(name='Worker'))

  agent = Workflow(
      name='test_single_item_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_single_item_agent',
          {
              'node_name': 'NodeA',
              'output': {'val': 'item1', 'delay': 0},
          },
      ),
      # Children outputs
      (
          'test_single_item_agent',
          {'node_name': 'Worker__0', 'output': 'item1_processed'},
      ),
      # Parent output
      (
          'test_single_item_agent',
          {
              'node_name': 'Worker',
              'output': ['item1_processed'],
          },
      ),
  ]


async def _worker_func(node_input: dict[str, Any]) -> AsyncGenerator[Any, None]:
  if isinstance(node_input, dict) and 'delay' in node_input:
    await asyncio.sleep(node_input['delay'])
    val = node_input['val']
  else:
    val = node_input
  yield f'{val}_processed'


@pytest.mark.asyncio
async def test_parallel_worker_with_function(request: pytest.FixtureRequest):
  """Tests that ParallelWorker works with a function."""
  items = [{'val': 'item1', 'delay': 0}, {'val': 'item2', 'delay': 0.1}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(_worker_func)

  agent = Workflow(
      name='test_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent',
          {
              'node_name': 'NodeA',
              'output': [
                  {'val': 'item1', 'delay': 0},
                  {'val': 'item2', 'delay': 0.1},
              ],
          },
      ),
      # Children outputs
      (
          'test_agent',
          {'node_name': '_worker_func__0', 'output': 'item1_processed'},
      ),
      (
          'test_agent',
          {'node_name': '_worker_func__1', 'output': 'item2_processed'},
      ),
      # Parent output
      (
          'test_agent',
          {
              'node_name': '_worker_func',
              'output': ['item1_processed', 'item2_processed'],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_with_failure(request: pytest.FixtureRequest):
  """Tests that ParallelWorker raises exception if one worker fails.

  Basically asyncio.gather() by default raises the exception immediately and
  lets other tasks to run on their own. Currently workflow agent cancels any
  running nodes, so it cancels the any other worker node.

  In the testcase, we verify that:

  - task-1 finishes before the failure.
  - task-2 fails.
  - task-3 does not finish.
  - task-3 is cancelled by the workflow agent.

  Args:
    request: The pytest fixture request.
  """
  items = ['task-1', 'task-2', 'task-3']
  node_a = _ProducerNode(items=items, name='NodeA')

  tracker = {}
  task_3_done_cancelled = False

  async def _worker_failable_func(node_input: str) -> AsyncGenerator[Any, None]:
    if node_input == 'task-1':
      yield f'{node_input}_processed'
    elif node_input == 'task-2':
      await asyncio.sleep(0.05)
      raise ValueError(f'{node_input} failed')
    elif node_input == 'task-3':
      try:
        await asyncio.sleep(0.1)
      except asyncio.CancelledError:
        nonlocal task_3_done_cancelled
        task_3_done_cancelled = True
        raise
      yield f'{node_input}_processed'

    tracker[node_input] = True

  worker = ParallelWorker(_worker_failable_func)

  agent = Workflow(
      name='test_agent_fail',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  with pytest.raises(ValueError, match='task-2 failed'):
    async for event in runner.runner.run_async(
        user_id=runner.session.user_id,
        session_id=runner.session.id,
        new_message=testing_utils.get_user_content('start'),
    ):
      events.append(event)

  # task-1 finishes before the failure.
  # task-2 fails.
  # task-3 does not finish.
  assert tracker == {'task-1': True}

  # task-3 should be cancelled.
  assert task_3_done_cancelled

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_agent_fail',
          {
              'node_name': 'NodeA',
              'output': ['task-1', 'task-2', 'task-3'],
          },
      ),
      # Children outputs: Only task-1 finishes before the failure.
      (
          'test_agent_fail',
          {
              'node_name': '_worker_failable_func__0',
              'output': 'task-1_processed',
          },
      ),
  ]


class _HitlWorkerNode(BaseNode):
  """A worker node that can request human input."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = Field(default=True)
  name: str = Field(default='Worker')

  def __init__(self, name: str = 'Worker'):
    super().__init__()
    object.__setattr__(self, 'name', name)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    val = node_input['val']
    ask = node_input['ask']

    if ask:
      interrupt_id = f'req_{val}'
      if resume_input := ctx.resume_inputs.get(interrupt_id):
        yield Event(output=f"{val}_{resume_input['text']}")
      else:
        yield RequestInput(
            interrupt_id=interrupt_id, message=f'Input for {val}'
        )
    else:
      yield Event(output=f'{val}_processed')


@pytest.mark.asyncio
async def test_parallel_worker_hitl(request: pytest.FixtureRequest):
  """Tests ParallelWorker with HITL in one of the workers."""
  # Use ordered items to ensure event order
  # The ask flag controls whether the worker requests input.
  items = [{'val': 'item1', 'ask': False}, {'val': 'item2', 'ask': True}]
  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(_HitlWorkerNode(name='Worker'))

  agent = Workflow(
      name='parallel_worker_hitl_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1
  events1 = await runner.run_async(testing_utils.get_user_content('start'))

  # Check interruption
  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id == 'req_item2'
  invocation_id = events1[0].invocation_id

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'parallel_worker_hitl_agent',
          {
              'node_name': 'NodeA',
              'output': [
                  {'val': 'item1', 'ask': False},
                  {'val': 'item2', 'ask': True},
              ],
          },
      ),
      # item1 finishes and emits output
      (
          'parallel_worker_hitl_agent',
          {'node_name': 'Worker__0', 'output': 'item1_processed'},
      ),
      (
          'parallel_worker_hitl_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
  ]

  # Run 2: Resume
  user_input = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id,
          name='user_input',
          response={'text': 'resumed'},
      )
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )
  simplified_events2 = simplify_events_with_node(events2)

  # Upon resume, ParallelWorker reruns tasks.
  # item2 runs and emits output.
  # Parent emits list.

  assert simplified_events2 == [
      (
          'parallel_worker_hitl_agent',
          {'node_name': 'Worker__1', 'output': 'item2_resumed'},
      ),
      (
          'parallel_worker_hitl_agent',
          {
              'node_name': 'Worker',
              'output': ['item1_processed', 'item2_resumed'],
          },
      ),
  ]


class _AsyncWorkerNode(BaseNode):
  """A worker node that waits for an asyncio event before processing an item."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default='Worker')
  events: dict[str, asyncio.Event] = Field(default_factory=dict)

  def __init__(
      self,
      name: str = 'Worker',
      events: dict[str, asyncio.Event] | None = None,
  ):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'events', events or {})

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    if node_input in self.events:
      await self.events[node_input].wait()
    yield Event(output=f'{node_input}_res')


@pytest.mark.asyncio
async def test_parallel_worker_out_of_order(request: pytest.FixtureRequest):
  """Tests that ParallelWorker preserves order even if workers finish out of order."""
  item1 = 'item1'
  item2 = 'item2'
  events_map = {
      item1: asyncio.Event(),
      item2: asyncio.Event(),
  }

  node_a = _ProducerNode(items=[item1, item2], name='NodeA')
  worker = ParallelWorker(_AsyncWorkerNode(name='Worker', events=events_map))

  agent = Workflow(
      name='out_of_order_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Start the runner in a background task so we can control events
  run_task = asyncio.create_task(
      runner.run_async(testing_utils.get_user_content('start'))
  )

  # Allow the runner to reach the wait point.
  await asyncio.sleep(0.1)

  # Finish item2 first
  events_map[item2].set()
  await asyncio.sleep(0.1)

  # Finish item1 second
  events_map[item1].set()

  events = await run_task

  simplified_events = simplify_events_with_node(events)

  # Output should be:
  # item2 output (finished first)
  # item1 output (finished second)
  # parent output (input order)

  assert simplified_events == [
      ('out_of_order_agent', {'node_name': 'NodeA', 'output': [item1, item2]}),
      ('out_of_order_agent', {'node_name': 'Worker__1', 'output': 'item2_res'}),
      ('out_of_order_agent', {'node_name': 'Worker__0', 'output': 'item1_res'}),
      (
          'out_of_order_agent',
          {'node_name': 'Worker', 'output': ['item1_res', 'item2_res']},
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_nested_agent(request: pytest.FixtureRequest):
  """Tests that a nested Workflow can be used as a ParallelWorker."""
  items = ['item1', 'item2']
  node_a = _ProducerNode(items=items, name='NodeA')

  async def worker_func(node_input: Any):
    return f'{node_input}_processed'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[(START, worker_func)],
  )

  parallel_nested = node(nested_agent, parallel_worker=True)

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          (START, node_a),
          (node_a, parallel_nested),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(testing_utils.get_user_content('start'))
  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'outer_agent',
          {
              'node_name': 'NodeA',
              'output': ['item1', 'item2'],
          },
      ),
      # Children outputs (nested workflow finalize events are deduplicated)
      (
          'nested_agent__0',
          {'node_name': 'worker_func', 'output': 'item1_processed'},
      ),
      (
          'nested_agent__1',
          {'node_name': 'worker_func', 'output': 'item2_processed'},
      ),
      # Parent output
      (
          'outer_agent',
          {
              'node_name': 'nested_agent',
              'output': [
                  'item1_processed',
                  'item2_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_workflow_agent_with_parallel_worker_flag(
    request: pytest.FixtureRequest,
):
  """Tests that a Workflow can be configured with parallel_worker=True."""
  dynamic_node_registry.clear()

  async def producer_func():
    # Produces a list of items to be processed in parallel
    return ['item1', 'item2']

  async def worker_func(node_input: Any):
    # Simulates work on a single item.
    return f'{node_input}_processed'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', worker_func)],
      parallel_worker=True,
  )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', producer_func),
          (producer_func, nested_agent),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'outer_agent',
          {
              'node_name': 'producer_func',
              'output': ['item1', 'item2'],
          },
      ),
      # Children outputs (nested workflow finalize events are deduplicated)
      (
          'nested_agent__0',
          {'node_name': 'worker_func', 'output': 'item1_processed'},
      ),
      (
          'nested_agent__1',
          {'node_name': 'worker_func', 'output': 'item2_processed'},
      ),
      # Parent output
      (
          'outer_agent',
          {
              'node_name': 'nested_agent',
              'output': [
                  'item1_processed',
                  'item2_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_max_concurrency(request: pytest.FixtureRequest):
  """Tests that ParallelWorker honors max_concurrency."""
  items = ['item1', 'item2', 'item3', 'item4']
  started_events = {item: asyncio.Event() for item in items}
  finish_events = {item: asyncio.Event() for item in items}

  async def _concurrency_worker_func(
      node_input: str,
  ) -> AsyncGenerator[Any, None]:
    started_events[node_input].set()
    await finish_events[node_input].wait()
    yield f'{node_input}_processed'

  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(_concurrency_worker_func, max_concurrency=2)

  agent = Workflow(
      name='max_concurrency_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  async def _run_agent():
    async for event in runner.runner.run_async(
        user_id=runner.session.user_id,
        session_id=runner.session.id,
        new_message=testing_utils.get_user_content('start'),
    ):
      events.append(event)

  run_task = asyncio.create_task(_run_agent())

  # Verify that initially only two workers are started.
  await asyncio.gather(
      started_events['item1'].wait(),
      started_events['item2'].wait(),
  )

  assert started_events['item1'].is_set()
  assert started_events['item2'].is_set()
  assert not started_events['item3'].is_set()
  assert not started_events['item4'].is_set()

  # Then signal second worker to finish.
  finish_events['item2'].set()

  # Check that the next worker (third one) is scheduled.
  await started_events['item3'].wait()
  assert started_events['item3'].is_set()
  assert not started_events['item4'].is_set()

  # Then signal the third worker to be finished. Check 4th worker is scheduled.
  finish_events['item3'].set()
  await started_events['item4'].wait()
  assert started_events['item4'].is_set()

  # Then finish all workers, and assert the workflow finishes.
  finish_events['item1'].set()
  finish_events['item4'].set()

  await run_task

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'max_concurrency_agent',
          {'node_name': 'NodeA', 'output': items},
      ),
      # item2 finishes first
      (
          'max_concurrency_agent',
          {
              'node_name': '_concurrency_worker_func__1',
              'output': 'item2_processed',
          },
      ),
      # then item3 finishes
      (
          'max_concurrency_agent',
          {
              'node_name': '_concurrency_worker_func__2',
              'output': 'item3_processed',
          },
      ),
      # then item1 and item4 finish.
      # order is deterministic due to creation order.
      (
          'max_concurrency_agent',
          {
              'node_name': '_concurrency_worker_func__0',
              'output': 'item1_processed',
          },
      ),
      (
          'max_concurrency_agent',
          {
              'node_name': '_concurrency_worker_func__3',
              'output': 'item4_processed',
          },
      ),
      (
          'max_concurrency_agent',
          {
              'node_name': '_concurrency_worker_func',
              'output': [
                  'item1_processed',
                  'item2_processed',
                  'item3_processed',
                  'item4_processed',
              ],
          },
      ),
  ]


@pytest.mark.asyncio
async def test_parallel_worker_max_concurrency_with_hitl(
    request: pytest.FixtureRequest,
):
  """Tests ParallelWorker with both max_concurrency and HITL.

  We have 3 nodes with max_concurrency=2. The second node goes to HITL. We
  resume, and that node finishes, which should then schedule the 3rd node.
  3rd node also does HITL. Then we signal the first node to finish, and resume
  node 3 to finish.
  """
  items = [
      {'val': 'item1', 'ask': False},
      {'val': 'item2', 'ask': True},
      {'val': 'item3', 'ask': True},
  ]
  started_events = {item['val']: asyncio.Event() for item in items}
  finish_events = {item['val']: asyncio.Event() for item in items}

  @node(name='Worker', rerun_on_resume=True)
  async def hitl_concurrency_worker(
      ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    val = node_input['val']
    ask = node_input['ask']

    started_events[val].set()

    if ask:
      interrupt_id = f'req_{val}'
      if resume_input := ctx.resume_inputs.get(interrupt_id):
        yield Event(output=f"{val}_{resume_input['text']}")
      else:
        yield RequestInput(
            interrupt_id=interrupt_id, message=f'Input for {val}'
        )
    else:
      await finish_events[val].wait()
      yield Event(output=f'{val}_processed')

  node_a = _ProducerNode(items=items, name='NodeA')
  worker = ParallelWorker(hitl_concurrency_worker, max_concurrency=2)

  agent = Workflow(
      name='max_concurrency_hitl_agent',
      edges=[
          (START, node_a),
          (node_a, worker),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events = []

  # Run 1: Start
  run_task = asyncio.create_task(
      runner.run_async(testing_utils.get_user_content('start'))
  )

  # Node 1 and Node 2 start. Node 2 is HITL and will yield RequestInput.
  await asyncio.gather(
      started_events['item1'].wait(),
      started_events['item2'].wait(),
  )

  assert started_events['item1'].is_set()
  assert started_events['item2'].is_set()
  # Node 3 is not started because max_concurrency is 2 and nodes 1 and 2 are active
  # (node 2 yields RequestInput shutting down the run, but node 3 was not yet scheduled).
  assert not started_events['item3'].is_set()

  # Signal node 1 to finish. This is needed to let runner finish. Otherwise,
  # the runner will never finish because there is still a pending task (node 1).
  finish_events['item1'].set()

  # Output should contain a generic RequestInput for 2.
  events1 = await run_task
  events.extend(events1)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert len(req_events) == 1
  interrupt_id_2 = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id_2 == 'req_item2'
  invocation_id_1 = events1[0].invocation_id

  simplified_events1 = simplify_events_with_node(events1)
  assert simplified_events1 == [
      (
          'max_concurrency_hitl_agent',
          {
              'node_name': 'NodeA',
              'output': items,
          },
      ),
      (
          'max_concurrency_hitl_agent',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'max_concurrency_hitl_agent',
          {'node_name': 'Worker__0', 'output': 'item1_processed'},
      ),
  ]

  # Run 2: Resume Node 2.
  user_input_2 = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id_2,
          name='user_input',
          response={'text': 'resumed'},
      )
  )

  run_task_2 = asyncio.create_task(
      runner.run_async(
          new_message=testing_utils.UserContent(user_input_2),
          invocation_id=invocation_id_1,
      )
  )

  # wait for node 3 to start and yield RequestInput
  await asyncio.sleep(0.1)
  await started_events['item3'].wait()
  assert started_events['item3'].is_set()

  events2 = await run_task_2
  events.extend(events2)

  req_events_2 = [e for e in events2 if has_request_input_function_call(e)]
  assert len(req_events_2) == 1
  interrupt_id_3 = get_request_input_interrupt_ids(req_events_2[0])[0]
  assert interrupt_id_3 == 'req_item3'
  invocation_id_2 = events2[0].invocation_id

  simplified_events2 = simplify_events_with_node(events2)
  assert simplified_events2 == [
      (
          'max_concurrency_hitl_agent',
          {'node_name': 'Worker__1', 'output': 'item2_resumed'},
      ),
      (
          'max_concurrency_hitl_agent',
          testing_utils.simplify_content(req_events_2[0].content),
      ),
  ]

  # Run 3: Signal Node 1 to finish, and Resume Node 3.
  # We should use create_task for run_async and finish events like before
  # so that asyncio doesn't hang.

  user_input_3 = types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id_3,
          name='user_input',
          response={'text': 'resumed'},
      )
  )

  run_task_3 = asyncio.create_task(
      runner.run_async(
          new_message=testing_utils.UserContent(user_input_3),
          invocation_id=invocation_id_2,
      )
  )

  events3 = await run_task_3
  events.extend(events3)

  simplified_events3 = simplify_events_with_node(events3)

  # The parent output should be the last event.
  assert simplified_events3 == [
      (
          'max_concurrency_hitl_agent',
          {'node_name': 'Worker__2', 'output': 'item3_resumed'},
      ),
      (
          'max_concurrency_hitl_agent',
          {
              'node_name': 'Worker',
              'output': ['item1_processed', 'item2_resumed', 'item3_resumed'],
          },
      ),
  ]
