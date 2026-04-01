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

"""Tests for DynamicNodeScheduler.

Verifies the three scheduling cases (fresh, dedup, resume) and the
lazy event scan that reconstructs dynamic node state.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._dynamic_node_scheduler import DefaultNodeScheduler
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeRun
from google.adk.workflow._dynamic_node_scheduler import DynamicNodeScheduler
from google.adk.workflow._node_state import NodeState
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow_class import _LoopState
import pytest

# --- Fixtures ---


def _make_parent_ctx(events=None):
  """Create a minimal parent Context with mock IC."""
  ic = MagicMock()
  ic.invocation_id = 'inv-1'
  ic.session = MagicMock()
  ic.session.state = {}
  ic.session.events = events or []
  ic.run_config = None

  collected = []

  async def _enqueue(event):
    collected.append(event)

  ic.enqueue_event = AsyncMock(side_effect=_enqueue)

  ctx = MagicMock(spec=Context)
  ctx._invocation_context = ic
  ctx.node_path = 'wf/parent'
  ctx.run_id = 'run-parent'
  ctx.event_author = 'wf'
  ctx._schedule_dynamic_node_internal = None
  ctx._output_for_ancestors = []
  ctx._output_delegated = False
  return ctx, collected


def _make_event(
    path='',
    output=None,
    interrupt_ids=None,
    run_id=None,
    author='node',
    invocation_id='inv-1',
    output_for=None,
):
  """Create a minimal Event for session event lists."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = author
  event.output = output
  event.partial = False
  event.node_info = MagicMock(spec=NodeInfo)
  event.node_info.path = path
  event.node_info.run_id = run_id
  event.node_info.output_for = output_for
  event.long_running_tool_ids = set(interrupt_ids) if interrupt_ids else None
  event.content = None
  event.actions = None
  return event


def _make_fr_event(fc_id, response, invocation_id='inv-1'):
  """Create a user FR event."""
  event = MagicMock(spec=Event)
  event.invocation_id = invocation_id
  event.author = 'user'
  event.output = None
  event.node_info = MagicMock(spec=NodeInfo)
  event.node_info.path = ''
  event.long_running_tool_ids = None

  fr = MagicMock()
  fr.id = fc_id
  fr.response = response

  part = MagicMock()
  part.function_response = fr

  content = MagicMock()
  content.parts = [part]
  event.content = content
  return event


# =========================================================================
# _rehydrate_from_events — lazy scan
# =========================================================================


@pytest.mark.asyncio
async def test_rehydrate_finds_completed_node():
  """Scan finds output event → node marked COMPLETED."""
  events = [
      _make_event(
          path='wf/parent/child',
          output='result',
          run_id='r-1',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  assert 'wf/parent/child' in ls.runs
  assert ls.runs['wf/parent/child']['r-1'].state.status == NodeStatus.COMPLETED
  assert ls.runs['wf/parent/child']['r-1'].output == 'result'


@pytest.mark.asyncio
async def test_rehydrate_finds_interrupted_node():
  """Scan finds interrupt event → node marked WAITING."""
  events = [
      _make_event(
          path='wf/parent/child',
          interrupt_ids=['fc-1'],
          run_id='r-1',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  state = ls.runs['wf/parent/child']['r-1'].state
  assert state.status == NodeStatus.WAITING
  assert 'fc-1' in state.interrupts


@pytest.mark.asyncio
async def test_rehydrate_resolves_interrupt_with_fr():
  """Scan finds interrupt + FR → all resolved, ready to re-run."""
  events = [
      _make_event(
          path='wf/parent/child',
          interrupt_ids=['fc-1'],
          run_id='r-1',
      ),
      _make_fr_event('fc-1', {'approved': True}),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  state = ls.runs['wf/parent/child']['r-1'].state
  assert state.status == NodeStatus.WAITING
  assert state.interrupts == []  # all resolved
  assert 'fc-1' in state.resume_inputs


@pytest.mark.asyncio
async def test_rehydrate_no_events_does_nothing():
  """Scan with no matching events does not populate dynamic_nodes."""
  events = [
      _make_event(path='wf/other/node', output='x'),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  assert 'wf/parent/child' not in ls.runs


@pytest.mark.asyncio
async def test_rehydrate_subtree_interrupt():
  """Interrupts from nested descendants are collected."""
  events = [
      _make_event(
          path='wf/parent/child/inner',
          interrupt_ids=['fc-deep'],
          run_id='r-1',
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  state = ls.runs['wf/parent/child']['r-1'].state
  assert 'fc-deep' in state.interrupts


@pytest.mark.asyncio
async def test_rehydrate_output_for_delegation():
  """Output via output_for delegation is recognized."""
  events = [
      _make_event(
          path='wf/parent/child/inner',
          output='delegated',
          run_id='r-1',
          output_for=['wf/parent/child/inner', 'wf/parent/child'],
      ),
  ]
  ctx, _ = _make_parent_ctx(events=events)
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  scheduler._rehydrate_from_events(ctx, 'wf/parent/child')

  assert ls.runs['wf/parent/child']['r-1'].output == 'delegated'


# =========================================================================
# __call__ — dispatch logic
# =========================================================================


@pytest.mark.asyncio
async def test_fresh_execution_runs_node():
  """No prior state → node executes and output is returned."""

  class _Child(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'hello: {node_input}'

  ctx, events = _make_parent_ctx()
  ls = _LoopState()
  scheduler = DynamicNodeScheduler(ls)

  child_ctx = await scheduler(
      ctx,
      _Child(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='1',
  )

  assert child_ctx.output == 'hello: input'
  assert 'wf/parent/child' in ls.runs


@pytest.mark.asyncio
async def test_completed_dedup_returns_cached():
  """Pre-populated COMPLETED state → returns cached output."""

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  ls.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(status=NodeStatus.COMPLETED, run_id='r-1'),
          output='cached',
      )
  }

  scheduler = DynamicNodeScheduler(ls)

  child_ctx = await scheduler(
      ctx,
      BaseNode(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'cached'


@pytest.mark.asyncio
async def test_waiting_unresolved_propagates_interrupts():
  """WAITING with unresolved interrupts → propagated to loop_state."""

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  ls.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(
              status=NodeStatus.WAITING, interrupts=['fc-1'], run_id='r-1'
          )
      )
  }

  scheduler = DynamicNodeScheduler(ls)

  child_ctx = await scheduler(
      ctx,
      BaseNode(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.interrupt_ids == {'fc-1'}
  assert 'fc-1' in ls.interrupt_ids


@pytest.mark.asyncio
async def test_waiting_resolved_resumes_node():
  """WAITING with all resolved → node re-runs with resume_inputs."""

  class _Resumable(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-1"]}'
        return
      yield 'should not reach here'

  ctx, _ = _make_parent_ctx()
  ls = _LoopState()
  ls.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(
              status=NodeStatus.WAITING,
              interrupts=[],
              run_id='r-1',
              resume_inputs={'fc-1': 'response'},
          )
      )
  }

  scheduler = DynamicNodeScheduler(ls)

  child_ctx = await scheduler(
      ctx,
      _Resumable(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'resumed: response'


# =========================================================================
# DefaultNodeScheduler — standalone scheduler
# =========================================================================


@pytest.mark.asyncio
async def test_default_scheduler_fresh_execution():
  """DefaultNodeScheduler runs a fresh node just like DynamicNodeScheduler."""

  class _Child(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'ct: {node_input}'

  ctx, _ = _make_parent_ctx()
  tracker = DefaultNodeScheduler()

  child_ctx = await tracker(
      ctx,
      _Child(name='child'),
      'unused',
      'data',
      node_name='child',
      run_id='1',
  )

  assert child_ctx.output == 'ct: data'


@pytest.mark.asyncio
async def test_default_scheduler_dedup_returns_cached():
  """DefaultNodeScheduler returns cached output for completed nodes."""
  ctx, _ = _make_parent_ctx()
  tracker = DefaultNodeScheduler()

  # Pre-populate state as if node already completed.
  tracker._state.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(status=NodeStatus.COMPLETED, run_id='r-1'),
          output='cached',
      )
  }

  child_ctx = await tracker(
      ctx,
      BaseNode(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'cached'


@pytest.mark.asyncio
async def test_default_scheduler_resume_with_resolved_interrupts():
  """DefaultNodeScheduler re-runs nodes with resolved interrupts."""

  class _Resumable(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-1"]}'
        return
      yield 'should not reach here'

  ctx, _ = _make_parent_ctx()
  tracker = DefaultNodeScheduler()

  # Pre-populate state as if node interrupted and was resolved.
  tracker._state.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(
              status=NodeStatus.WAITING,
              interrupts=[],
              run_id='r-1',
              resume_inputs={'fc-1': 'approved'},
          )
      )
  }

  child_ctx = await tracker(
      ctx,
      _Resumable(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.output == 'resumed: approved'


@pytest.mark.asyncio
async def test_default_scheduler_propagates_unresolved_interrupts():
  """DefaultNodeScheduler propagates unresolved interrupts."""
  ctx, _ = _make_parent_ctx()
  tracker = DefaultNodeScheduler()

  tracker._state.runs['wf/parent/child'] = {
      'r-1': DynamicNodeRun(
          state=NodeState(
              status=NodeStatus.WAITING, interrupts=['fc-1'], run_id='r-1'
          )
      )
  }

  child_ctx = await tracker(
      ctx,
      BaseNode(name='child'),
      'unused',
      'input',
      node_name='child',
      run_id='r-1',
  )

  assert child_ctx.interrupt_ids == {'fc-1'}
  assert 'fc-1' in tracker._state.interrupt_ids
