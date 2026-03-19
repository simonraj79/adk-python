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

"""Tests for nested workflow output event deduplication.

Verifies that nested workflows produce only leaf terminal events and
resolve output via terminal path resolution from the graph structure.
"""

from __future__ import annotations

from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.workflow import Workflow
from google.genai import types
import pytest

from . import testing_utils


def _is_checkpoint(event: Event) -> bool:
  """Returns True if event is an agent state checkpoint or end_of_agent."""
  if event.actions and event.actions.agent_state is not None:
    return True
  if event.actions and event.actions.end_of_agent:
    return True
  return False


def _output_events(events: list[Event]) -> list[Event]:
  """Returns only events with output data (not checkpoint/state)."""
  return [
      e for e in events if not _is_checkpoint(e) and e.output is not None
  ]


# ─── 2-level nesting ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_level_nesting_deduplicates(
    request: pytest.FixtureRequest,
):
  """outer → inner → leaf. Only 1 output event (the leaf)."""

  async def leaf(node_input: types.Content):
    return 'leaf_data'

  inner = Workflow(name='inner', edges=[('START', leaf)])
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  # Should be exactly 1: just the leaf event. No finalize events.
  assert len(out_events) == 1

  leaf_event = out_events[0]
  assert leaf_event.output == 'leaf_data'
  assert leaf_event.node_info.path == 'outer/inner/leaf'


# ─── Nested workflow with output_schema ──────────────────────────


@pytest.mark.asyncio
async def test_nested_with_output_schema_validates_at_read_time(
    request: pytest.FixtureRequest,
):
  """output_schema validates at read time — no extra finalize event."""

  async def leaf(node_input: types.Content):
    return 'raw_data'

  inner = Workflow(
      name='inner',
      edges=[('START', leaf)],
      output_schema=str,
  )
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  # 1 event: leaf, validated at read time.
  assert len(out_events) == 1
  assert out_events[0].output == 'raw_data'


# ─── Route propagation through nested workflow ────────────────────


@pytest.mark.asyncio
async def test_route_propagates_through_nested_workflow(
    request: pytest.FixtureRequest,
):
  """Route from a deep leaf propagates correctly to parent routing."""

  async def classify(node_input: types.Content):
    return Event(output='classified', route='path_a')

  inner = Workflow(name='inner', edges=[('START', classify)])

  async def handler_a(node_input: str):
    return f'handled: {node_input}'

  async def handler_b(node_input: str):
    return 'wrong path'

  outer = Workflow(
      name='outer',
      edges=[
          ('START', inner),
          (inner, {'path_a': handler_a, 'path_b': handler_b}),
      ],
  )

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  # handler_a should run, handler_b should NOT.
  handler_outputs = [
      e for e in out_events if e.node_info.name == 'handler_a'
  ]
  assert len(handler_outputs) == 1
  assert handler_outputs[0].output == 'handled: classified'
  assert not any(e.node_info.name == 'handler_b' for e in out_events)


# ─── Multiple terminal nodes in nested workflow ────────────────────


@pytest.mark.asyncio
async def test_multiple_terminals_in_nested_workflow(
    request: pytest.FixtureRequest,
):
  """Fan-out with no join: both terminals produce output events."""

  async def branch_a(node_input: types.Content):
    return 'a_out'

  async def branch_b(node_input: types.Content):
    return 'b_out'

  inner = Workflow(
      name='inner',
      edges=[('START', (branch_a, branch_b))],
  )
  outer = Workflow(name='outer', edges=[('START', inner)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  # Both branch events should appear. No finalize events.
  branch_events = [
      e
      for e in out_events
      if e.node_info.name in ('branch_a', 'branch_b')
  ]
  assert len(branch_events) == 2

  # No root finalize events.
  root_events = [e for e in out_events if e.node_info.path == 'outer']
  assert len(root_events) == 0


# ─── Non-terminal node output stays internal ────────────────────────


@pytest.mark.asyncio
async def test_non_terminal_output_not_exposed_as_workflow_output(
    request: pytest.FixtureRequest,
):
  """Only terminal node output is used as workflow output."""

  async def step_a(node_input: types.Content):
    return 'intermediate'

  async def step_b(node_input: str):
    return 'final'

  inner = Workflow(name='inner', edges=[('START', step_a, step_b)])

  async def consume(node_input: str):
    return f'got: {node_input}'

  outer = Workflow(name='outer', edges=[('START', inner, consume)])

  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hi'))
  out_events = _output_events(events)

  # consume should receive step_b's output (terminal), not step_a's.
  consume_events = [e for e in out_events if e.node_info.name == 'consume']
  assert len(consume_events) == 1
  assert consume_events[0].output == 'got: final'
