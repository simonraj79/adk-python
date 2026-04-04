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
from google.adk.events.event import Event
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from pydantic import Field
import pytest
from typing_extensions import override

from .. import testing_utils
from ..workflow_testing_utils import create_parent_invocation_context


class ConcurrencyWorkerNode(BaseNode):
  """A node that signals when it starts and waits for a signal to finish."""

  def __init__(
      self, name: str, started_event: asyncio.Event, finish_event: asyncio.Event
  ):
    super().__init__(name=name)
    object.__setattr__(self, 'started_event', started_event)
    object.__setattr__(self, 'finish_event', finish_event)

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
    self.started_event.set()
    await self.finish_event.wait()
    yield f'{self.name}_done'


class ProducerNode(BaseNode):
  """A node that produces a list of items."""

  items: list[Any] = Field(default_factory=list)

  def __init__(self, items: list[Any], name: str = 'Producer'):
    super().__init__(name=name)
    object.__setattr__(self, 'items', items)

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.items)


@pytest.mark.asyncio
async def test_workflow_max_concurrency(request: pytest.FixtureRequest):
  """Tests that max_concurrency limits the number of running nodes."""
  num_nodes = 4
  max_concurrency = 2
  started_events = [asyncio.Event() for _ in range(num_nodes)]
  finish_events = [asyncio.Event() for _ in range(num_nodes)]
  nodes = [
      ConcurrencyWorkerNode(f'Node{i}', started_events[i], finish_events[i])
      for i in range(num_nodes)
  ]

  agent = Workflow(
      name='concurrency_agent',
      max_concurrency=max_concurrency,
      edges=[(START, node) for node in nodes],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  # Run the agent in a background task
  async def run_agent():
    return [e async for e in agent.run_async(ctx)]

  run_task = asyncio.create_task(run_agent())

  # Wait a bit for nodes to start.
  await asyncio.sleep(0.1)

  # Check how many nodes have started
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency

  # Release one node
  for i in range(num_nodes):
    if started_events[i].is_set():
      finish_events[i].set()
      break

  # Wait for another node to start
  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency + 1

  # Release remaining nodes
  for e in finish_events:
    e.set()

  await run_task
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == num_nodes


@pytest.mark.asyncio
async def test_parallel_worker_with_workflow_concurrency(
    request: pytest.FixtureRequest,
):
  """Tests that ParallelWorker respects workflow-level max_concurrency."""
  num_items = 5
  max_concurrency = 3
  started_events = [asyncio.Event() for _ in range(num_items)]
  finish_events = [asyncio.Event() for _ in range(num_items)]

  # Create a function that uses the shared events
  async def worker_fn(node_input: int) -> AsyncGenerator[Any, None]:
    started_events[node_input].set()
    await finish_events[node_input].wait()
    yield f'done_{node_input}'

  producer = ProducerNode(items=list(range(num_items)))
  worker = ParallelWorker(worker_fn, max_concurrency=None)  # No limit in worker

  agent = Workflow(
      name='parallel_concurrency_agent',
      max_concurrency=max_concurrency,
      edges=[
          (START, producer),
          (producer, worker),
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  async def run_agent():
    return [e async for e in agent.run_async(ctx)]

  run_task = asyncio.create_task(run_agent())

  await asyncio.sleep(0.1)

  # Check how many workers have started
  started_count = sum(1 for e in started_events if e.is_set())
  # Total ACTIVE = 1 (ParallelWorker) + children_started
  # so children_started <= max_concurrency - 1
  assert started_count == max_concurrency - 1

  # Release one child
  for i in range(num_items):
    if started_events[i].is_set():
      finish_events[i].set()
      break

  await asyncio.sleep(0.1)
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == max_concurrency

  # Release all
  for e in finish_events:
    e.set()

  await run_task
  started_count = sum(1 for e in started_events if e.is_set())
  assert started_count == num_items
