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

"""Unit tests for _Mesh orchestrator node."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm._mesh import _Mesh
from google.adk.agents.llm._transfer_target_info import _TransferTargetInfo
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import BaseNode
from google.genai import types
import pytest
from typing_extensions import override

from tests.unittests.workflow import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubNode(BaseNode):
  """A minimal BaseNode stub for unit testing _Mesh.

  Yields pre-configured events when run. Supports optional
  ``disallow_transfer_to_parent`` and ``disallow_transfer_to_peers``
  attributes for transfer target computation testing.
  """

  description: str = ''
  events_to_yield: list[Event] = []
  disallow_transfer_to_parent: bool = False
  disallow_transfer_to_peers: bool = False

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    for event in self.events_to_yield:
      yield event


def _make_text_event(
    author: str,
    text: str,
    *,
    node_path: str = '',
    transfer_to_agent: str | None = None,
) -> Event:
  """Create a workflow Event with text content."""
  actions = EventActions()
  if transfer_to_agent:
    actions.transfer_to_agent = transfer_to_agent
  return Event(
      author=author,
      content=types.Content(
          role='model',
          parts=[types.Part(text=text)],
      ),
      actions=actions,
      node_path=node_path,
  )


def _make_session_event(
    author: str,
    text: str = '',
    *,
    node_path: str = '',
    transfer_to_agent: str | None = None,
    agent_state: dict | None = None,
    end_of_agent: bool = False,
) -> Event:
  """Create a workflow Event for session history."""
  actions = EventActions()
  if transfer_to_agent:
    actions.transfer_to_agent = transfer_to_agent
  if agent_state is not None:
    actions.agent_state = agent_state
  if end_of_agent:
    actions.end_of_agent = True

  content = None
  if text:
    content = types.Content(
        role='model',
        parts=[types.Part(text=text)],
    )

  return Event(
      invocation_id='test_id',
      author=author,
      content=content,
      actions=actions,
      node_path=node_path,
  )


async def _create_context(
    node_path: str = 'mesh',
    transfer_targets: list[_TransferTargetInfo] | None = None,
    session_events: list[Event] | None = None,
) -> Context:
  """Create a Context suitable for _Mesh tests."""
  agent = testing_utils.create_test_agent()
  ic = await testing_utils.create_invocation_context(agent)

  if session_events:
    for event in session_events:
      ic.session.events.append(event)

  return Context(
      invocation_context=ic,
      node_path=node_path,
      run_id='test-exec',
      local_events=[],
      transfer_targets=transfer_targets,
  )


async def _collect_events(mesh: _Mesh, ctx: Context) -> list:
  """Run mesh and collect all yielded events."""
  events = []
  async for event in mesh.run(ctx=ctx, node_input=None):
    events.append(event)
  return events


# ===================================================================
# Tests: _get_coordinator
# ===================================================================


class TestGetCoordinator:

  def test_coordinator_by_name_match(self):
    """Coordinator is the node whose name matches the mesh name."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    result = mesh._get_coordinator()

    assert result is coord

  def test_coordinator_fallback_to_first(self):
    """Falls back to first node if no name match."""
    a = StubNode(name='a')
    b = StubNode(name='b')
    mesh = _Mesh(name='no_match', nodes=[a, b])

    result = mesh._get_coordinator()

    assert result is a

  def test_coordinator_empty_nodes_rejected(self):
    """_Mesh with empty nodes raises ValueError."""
    with pytest.raises(ValueError, match='must have at least one node'):
      _Mesh(name='parent', nodes=[])

  def test_coordinator_name_match_not_first(self):
    """Name match takes priority even if not first in list."""
    a = StubNode(name='a')
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[a, coord])

    result = mesh._get_coordinator()

    assert result is coord


# ===================================================================
# Tests: _is_coordinator_name
# ===================================================================


class TestIsCoordinatorName:

  def test_true_for_coordinator(self):
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])
    assert mesh._is_coordinator_name('parent') is True

  def test_false_for_non_coordinator(self):
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])
    assert mesh._is_coordinator_name('child') is False

  def test_false_for_unknown_name(self):
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])
    assert mesh._is_coordinator_name('unknown') is False

  def test_empty_nodes_rejected(self):
    with pytest.raises(ValueError, match='must have at least one node'):
      _Mesh(name='parent', nodes=[])


# ===================================================================
# Tests: _get_node_context_path
# ===================================================================


class TestGetNodeContextPath:

  def test_coordinator_shares_mesh_path(self):
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    path = mesh._get_node_context_path('root/parent', 'parent')

    assert path == 'root/parent'

  def test_non_coordinator_appends_node_name(self):
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    path = mesh._get_node_context_path('root/parent', 'child')

    assert path == 'root/parent/child'

  def test_empty_mesh_path(self):
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    path = mesh._get_node_context_path('', 'child')

    assert path == 'child'


# ===================================================================
# Tests: _build_workflow_context
# ===================================================================


class TestBuildWorkflowContext:

  @pytest.mark.asyncio
  async def test_coordinator_gets_external_plus_local_targets(self):
    """Coordinator targets: external first, then local non-coordinator."""
    coord = StubNode(name='parent', description='coordinator')
    child_a = StubNode(name='a', description='agent a')
    child_b = StubNode(name='b', description='agent b')
    mesh = _Mesh(name='parent', nodes=[coord, child_a, child_b])

    external = [_TransferTargetInfo(name='ext', description='external')]
    ctx = await _create_context(
        node_path='root/parent',
        transfer_targets=external,
    )

    result = mesh._build_workflow_context(ctx, coord)

    target_names = [t.name for t in result.transfer_targets]
    assert target_names == ['ext', 'a', 'b']

  @pytest.mark.asyncio
  async def test_coordinator_no_external_targets(self):
    """Coordinator with no external targets gets only local nodes."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    target_names = [t.name for t in result.transfer_targets]
    assert target_names == ['child']

  @pytest.mark.asyncio
  async def test_non_coordinator_gets_coordinator_plus_peers(self):
    """Non-coordinator targets: coordinator first, then peers."""
    coord = StubNode(name='parent', description='coordinator')
    child_a = StubNode(name='a')
    child_b = StubNode(name='b')
    mesh = _Mesh(name='parent', nodes=[coord, child_a, child_b])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child_a)

    target_names = [t.name for t in result.transfer_targets]
    assert target_names == ['parent', 'b']

  @pytest.mark.asyncio
  async def test_non_coordinator_disallow_parent(self):
    """disallow_transfer_to_parent excludes coordinator from targets."""
    coord = StubNode(name='parent')
    child = StubNode(name='child', disallow_transfer_to_parent=True)
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child)

    target_names = [t.name for t in result.transfer_targets]
    assert 'parent' not in target_names

  @pytest.mark.asyncio
  async def test_non_coordinator_disallow_peers(self):
    """disallow_transfer_to_peers excludes peers from targets."""
    coord = StubNode(name='parent')
    child_a = StubNode(name='a', disallow_transfer_to_peers=True)
    child_b = StubNode(name='b')
    mesh = _Mesh(name='parent', nodes=[coord, child_a, child_b])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child_a)

    target_names = [t.name for t in result.transfer_targets]
    # Only coordinator, no peers
    assert target_names == ['parent']

  @pytest.mark.asyncio
  async def test_non_coordinator_disallow_both(self):
    """disallow both parent and peers -> empty targets."""
    coord = StubNode(name='parent')
    child = StubNode(
        name='child',
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child)

    assert result.transfer_targets == []

  @pytest.mark.asyncio
  async def test_coordinator_node_path(self):
    """Coordinator's context gets the mesh's own node_path."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    assert result.node_path == 'root/parent'

  @pytest.mark.asyncio
  async def test_non_coordinator_node_path(self):
    """Non-coordinator gets mesh_path/node_name as node_path."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child)

    assert result.node_path == 'root/parent/child'

  @pytest.mark.asyncio
  async def test_node_path_propagated(self):
    """node_path on IC is set to the node's path."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, child)

    ic = result._invocation_context
    assert ic.node_path == 'root/parent/child'

  @pytest.mark.asyncio
  async def test_node_path_for_coordinator(self):
    """Coordinator's IC gets the mesh's own path as node_path."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    ic = result._invocation_context
    assert ic.node_path == 'root/parent'

  @pytest.mark.asyncio
  async def test_ic_copy_shares_session(self):
    """IC model_copy is shallow — session dict is shared."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    ctx = await _create_context(node_path='root/parent')
    original_session = ctx._invocation_context.session

    result = mesh._build_workflow_context(ctx, coord)

    # Session object is the same (shallow copy).
    assert result._invocation_context.session is original_session

  @pytest.mark.asyncio
  async def test_local_events_always_empty(self):
    """Each child context gets a fresh empty local_events list."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    assert result._local_events == []

  @pytest.mark.asyncio
  async def test_run_id_inherited(self):
    """Child context inherits run_id from parent."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    assert result.run_id == ctx.run_id

  @pytest.mark.asyncio
  async def test_target_descriptions_propagated(self):
    """Transfer target descriptions come from node descriptions."""
    coord = StubNode(name='parent')
    child = StubNode(name='child', description='A helpful child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')

    result = mesh._build_workflow_context(ctx, coord)

    child_target = next(t for t in result.transfer_targets if t.name == 'child')
    assert child_target.description == 'A helpful child'


# ===================================================================
# Tests: _find_agent_to_run
# ===================================================================


class TestFindAgentToRun:

  @pytest.mark.asyncio
  async def test_defaults_to_coordinator(self):
    """With no session events, returns coordinator."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    ctx = await _create_context(node_path='root/parent')
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_finds_last_active_node(self):
    """Returns the last node that generated events."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    events = [
        _make_session_event('child', 'hello', node_path='root/parent/child'),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is child

  @pytest.mark.asyncio
  async def test_skips_user_events(self):
    """User events are ignored when finding last active node."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    events = [
        _make_session_event('child', 'hello', node_path='root/parent/child'),
        _make_session_event('user', 'question'),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is child

  @pytest.mark.asyncio
  async def test_skips_agent_state_events(self):
    """Events with agent_state are skipped."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    events = [
        _make_session_event(
            'child',
            node_path='root/parent/child',
            agent_state={'nodes': {}},
        ),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    # Skipped agent_state event, falls back to coordinator.
    assert result is coord

  @pytest.mark.asyncio
  async def test_skips_end_of_agent_events(self):
    """Events with end_of_agent are skipped."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    events = [
        _make_session_event(
            'child',
            node_path='root/parent/child',
            end_of_agent=True,
        ),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_skips_events_from_different_mesh(self):
    """Events scoped to a different mesh are ignored."""
    coord = StubNode(name='parent')
    child = StubNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])

    # Event from a different mesh path.
    events = [
        _make_session_event(
            'child',
            'hello',
            node_path='other_root/other_parent/child',
        ),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_skips_unknown_authors(self):
    """Events from authors not in nodes are ignored."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    events = [
        _make_session_event(
            'unknown_agent',
            'hello',
            node_path='root/parent/unknown_agent',
        ),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_disallow_transfer_to_parent_auto_returns(self):
    """Node with disallow_transfer_to_parent skips to coordinator."""
    coord = StubNode(name='parent')
    child = StubNode(name='child', disallow_transfer_to_parent=True)
    mesh = _Mesh(name='parent', nodes=[coord, child])

    events = [
        _make_session_event('child', 'hello', node_path='root/parent/child'),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_skips_loop_agent_node(self):
    """LoopAgent nodes are skipped — exit_loop means the loop completed."""
    from google.adk.agents.loop_agent import LoopAgent

    coord = StubNode(name='parent')
    loop = LoopAgent(name='loop_child', sub_agents=[])
    mesh = _Mesh(name='parent', nodes=[coord, loop])

    events = [
        _make_session_event(
            'loop_child',
            'hello',
            node_path='root/parent/loop_child',
        ),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_finds_most_recent_active_node(self):
    """Returns the most recent non-skipped node (reverse scan)."""
    coord = StubNode(name='parent')
    child_a = StubNode(name='a')
    child_b = StubNode(name='b')
    mesh = _Mesh(name='parent', nodes=[coord, child_a, child_b])

    events = [
        _make_session_event('a', 'first', node_path='root/parent/a'),
        _make_session_event('b', 'second', node_path='root/parent/b'),
    ]
    ctx = await _create_context(node_path='root/parent', session_events=events)
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is child_b


# ===================================================================
# Tests: run() orchestration loop
# ===================================================================


class TestMeshRun:

  @pytest.mark.asyncio
  async def test_single_node_no_transfer(self):
    """Single node finishes without transfer -> mesh ends."""
    event = _make_text_event('coord', 'Hello')
    coord = StubNode(name='parent', events_to_yield=[event])
    mesh = _Mesh(name='parent', nodes=[coord])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    text_events = [e for e in events if isinstance(e, Event) and e.content]
    assert len(text_events) >= 1
    assert text_events[0].content.parts[0].text == 'Hello'

  @pytest.mark.asyncio
  async def test_internal_transfer(self):
    """Transfer between internal nodes within the mesh."""
    transfer_event = _make_text_event(
        'parent',
        'delegating',
        transfer_to_agent='child',
    )
    coord = StubNode(
        name='parent',
        events_to_yield=[transfer_event],
    )
    child_event = _make_text_event('child', 'child response')
    child = StubNode(
        name='child',
        events_to_yield=[child_event],
    )
    mesh = _Mesh(name='parent', nodes=[coord, child])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    text_events = [e for e in events if isinstance(e, Event) and e.content]
    texts = [e.content.parts[0].text for e in text_events]
    assert 'delegating' in texts
    assert 'child response' in texts

  @pytest.mark.asyncio
  async def test_cross_mesh_transfer(self):
    """Transfer to unknown name exits the mesh."""
    transfer_event = _make_text_event(
        'parent',
        'escalating',
        transfer_to_agent='external_agent',
    )
    coord = StubNode(
        name='parent',
        events_to_yield=[transfer_event],
    )
    mesh = _Mesh(name='parent', nodes=[coord])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    # _Mesh should yield the transfer event and exit.
    text_events = [e for e in events if isinstance(e, Event) and e.content]
    assert len(text_events) == 1
    assert text_events[0].content.parts[0].text == 'escalating'

  @pytest.mark.asyncio
  async def test_data_output_reyielded_at_end(self):
    """Mesh re-yields the last data event as its own output."""
    data_event = Event(output={'result': 42}, author='parent')
    coord = StubNode(
        name='parent',
        events_to_yield=[data_event],
    )
    mesh = _Mesh(name='parent', nodes=[coord])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    output_events = [
        e for e in events if isinstance(e, Event) and e.output is not None
    ]
    # Original output event + re-yielded output event.
    assert len(output_events) == 2

  @pytest.mark.asyncio
  async def test_no_data_output_means_no_reyield(self):
    """When no data output event, no extra event at the end."""
    event = _make_text_event('parent', 'Hello')
    coord = StubNode(name='parent', events_to_yield=[event])
    mesh = _Mesh(name='parent', nodes=[coord])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    output_events = [
        e for e in events if isinstance(e, Event) and e.output is not None
    ]
    assert len(output_events) == 0

  @pytest.mark.asyncio
  async def test_multiple_internal_transfers(self):
    """Chain of internal transfers: coord -> a -> b."""
    coord_event = _make_text_event(
        'parent',
        'go to a',
        transfer_to_agent='a',
    )
    a_event = _make_text_event(
        'a',
        'go to b',
        transfer_to_agent='b',
    )
    b_event = _make_text_event('b', 'done')

    coord = StubNode(name='parent', events_to_yield=[coord_event])
    a = StubNode(name='a', events_to_yield=[a_event])
    b = StubNode(name='b', events_to_yield=[b_event])
    mesh = _Mesh(name='parent', nodes=[coord, a, b])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    text_events = [e for e in events if isinstance(e, Event) and e.content]
    texts = [e.content.parts[0].text for e in text_events]
    assert texts == ['go to a', 'go to b', 'done']

  @pytest.mark.asyncio
  async def test_node_input_only_for_first_node(self):
    """node_input is passed only to the first node, None after."""
    received_inputs = []

    class CapturingNode(BaseNode):

      @override
      async def run(
          self,
          *,
          ctx: Context,
          node_input: Any,
      ) -> AsyncGenerator[Any, None]:
        received_inputs.append(node_input)
        if self.name == 'parent':
          yield _make_text_event(
              'parent',
              'transfer',
              transfer_to_agent='child',
          )
        else:
          yield _make_text_event('child', 'done')

    coord = CapturingNode(name='parent')
    child = CapturingNode(name='child')
    mesh = _Mesh(name='parent', nodes=[coord, child])
    ctx = await _create_context(node_path='root/parent')

    # Pass specific node_input.
    events = []
    async for event in mesh.run(ctx=ctx, node_input='initial_data'):
      events.append(event)

    assert received_inputs[0] == 'initial_data'
    assert received_inputs[1] is None

  @pytest.mark.asyncio
  async def test_empty_node_no_events(self):
    """Node that yields nothing -> mesh ends gracefully."""
    coord = StubNode(name='parent', events_to_yield=[])
    mesh = _Mesh(name='parent', nodes=[coord])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_events(mesh, ctx)

    assert events == []


# ===================================================================
# Tests: get_name
# ===================================================================


class TestGetName:

  def test_returns_mesh_name(self):
    mesh = _Mesh(name='my_mesh', nodes=[StubNode(name='my_mesh')])
    assert mesh.name == 'my_mesh'

  def test_empty_name(self):
    mesh = _Mesh(name='', nodes=[StubNode(name='node')])
    assert mesh.name == ''


# ===================================================================
# Helpers: Task delegation stubs
# ===================================================================


class TaskStubNode(BaseNode):
  """A BaseNode stub with mode support for task delegation tests.

  Each call to run() pops the first event list from ``runs``
  (a list of lists), so the coordinator can yield different events
  on subsequent calls.

  Note: Events are NOT appended to the session here. The test
  helper ``_collect_task_events`` appends yielded events to the
  original IC's session (simulating what the Runner does in
  production).
  """

  description: str = ''
  mode: str = 'chat'
  runs: list[list[Event]] = []

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    events_to_yield = self.runs.pop(0) if self.runs else []
    for event in events_to_yield:
      yield event


async def _collect_task_events(mesh: _Mesh, ctx: Context) -> list:
  """Run mesh and collect events, appending to session.

  Simulates what the Runner does: events yielded by the mesh are
  appended to the actual session's events list so that the
  completion gate can scan them. The Context wraps the session in
  a ``_SessionProxy``, so we access the underlying session directly.
  """
  ic = ctx.get_invocation_context()
  session_proxy = ic.session
  # Get the underlying session (not the proxy).
  actual_session = object.__getattribute__(session_proxy, '_session')
  events = []
  async for event in mesh.run(ctx=ctx, node_input=None):
    if isinstance(event, Event):
      actual_session.events.append(event)
    events.append(event)
  return events


def _make_request_task_event(
    author: str,
    request_task: dict,
    *,
    node_path: str = '',
) -> Event:
  """Create an event with request_task actions."""
  return Event(
      invocation_id='test_id',
      author=author,
      content=types.Content(
          role='model',
          parts=[types.Part(text='Delegating tasks')],
      ),
      actions=EventActions(request_task=request_task),
      node_path=node_path,
  )


def _make_finish_task_event(
    author: str,
    finish_task: dict,
    *,
    branch: str = '',
    node_path: str = '',
) -> Event:
  """Create an event with finish_task actions."""
  return Event(
      invocation_id='test_id',
      author=author,
      content=types.Content(
          role='model',
          parts=[types.Part(text='Task completed.')],
      ),
      actions=EventActions(finish_task=finish_task),
      branch=branch,
      node_path=node_path,
  )


# ===================================================================
# Tests: _get_unfulfilled_fc_ids
# ===================================================================


class TestGetUnfulfilledFcIds:

  def test_all_unfulfilled_when_no_finish_events(self):
    """All fc_ids are unfulfilled when no finish_task events exist."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    req_event = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
    )

    result = mesh._get_unfulfilled_fc_ids(req_event, [req_event])

    assert result == {'fc_A', 'fc_B'}

  def test_one_fulfilled(self):
    """One finish_task reduces unfulfilled set."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    req_event = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
    )
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:mesh.agent_a.fc_A',
    )

    result = mesh._get_unfulfilled_fc_ids(req_event, [req_event, finish_a])

    assert result == {'fc_B'}

  def test_all_fulfilled(self):
    """All finish_task events -> empty unfulfilled set."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    req_event = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
    )
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:mesh.agent_a.fc_A',
    )
    finish_b = _make_finish_task_event(
        'agent_b',
        {'output': {'result': 'done B'}},
        branch='task:mesh.agent_b.fc_B',
    )

    result = mesh._get_unfulfilled_fc_ids(
        req_event, [req_event, finish_a, finish_b]
    )

    assert result == set()

  def test_single_delegation(self):
    """Single delegation -- one fc_id unfulfilled, then fulfilled."""
    coord = StubNode(name='parent')
    mesh = _Mesh(name='parent', nodes=[coord])

    req_event = _make_request_task_event(
        'parent',
        {'fc_X': {'agentName': 'agent_x', 'input': {}}},
    )

    # Before finish.
    result = mesh._get_unfulfilled_fc_ids(req_event, [req_event])
    assert result == {'fc_X'}

    # After finish.
    finish_x = _make_finish_task_event(
        'agent_x',
        {'output': {'result': 'done'}},
        branch='task:mesh.agent_x.fc_X',
    )
    result = mesh._get_unfulfilled_fc_ids(req_event, [req_event, finish_x])
    assert result == set()


# ===================================================================
# Tests: Completion gate in run() and _find_agent_to_run
# ===================================================================


class TestCompletionGate:

  @pytest.mark.asyncio
  async def test_two_delegations_run_sequentially(self):
    """Two request_task entries -> both agents run sequentially."""
    # Agent A yields a finish_task event.
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:root/parent/agent_a.agent_a.fc_A',
        node_path='root/parent/agent_a',
    )
    # Agent B yields a finish_task event.
    finish_b = _make_finish_task_event(
        'agent_b',
        {'output': {'result': 'done B'}},
        branch='task:root/parent/agent_b.agent_b.fc_B',
        node_path='root/parent/agent_b',
    )

    # Coordinator yields a request_task event with 2 entries.
    request = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
        node_path='root/parent',
    )

    coord = TaskStubNode(
        name='parent',
        runs=[[request], []],  # First call: request_task; second: done.
    )
    agent_a = TaskStubNode(
        name='agent_a',
        mode='single_turn',
        runs=[[finish_a]],
    )
    agent_b = TaskStubNode(
        name='agent_b',
        mode='single_turn',
        runs=[[finish_b]],
    )
    mesh = _Mesh(name='parent', nodes=[coord, agent_a, agent_b])
    ctx = await _create_context(node_path='root/parent')

    events = await _collect_task_events(mesh, ctx)

    # Verify both agents ran: request_task event, finish_a, finish_b,
    # then coordinator resumes (empty yield).
    authors = [
        e.author
        for e in events
        if isinstance(e, Event)
        and (e.actions.request_task or e.actions.finish_task)
    ]
    assert 'parent' in authors  # request_task
    assert 'agent_a' in authors  # finish_task A
    assert 'agent_b' in authors  # finish_task B

  @pytest.mark.asyncio
  async def test_find_agent_to_run_resumes_unfulfilled(self):
    """On resume, _find_agent_to_run returns the unfulfilled agent."""
    coord = StubNode(name='parent')
    agent_a = StubNode(name='agent_a')
    agent_b = StubNode(name='agent_b')
    mesh = _Mesh(name='parent', nodes=[coord, agent_a, agent_b])

    # Simulate: request_task with 2 entries, agent_a finished.
    request = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
        node_path='root/parent',
    )
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:root/parent/agent_a.agent_a.fc_A',
        node_path='root/parent/agent_a',
    )

    ctx = await _create_context(
        node_path='root/parent',
        session_events=[request, finish_a],
    )
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is agent_b

  @pytest.mark.asyncio
  async def test_find_agent_to_run_all_fulfilled_returns_coordinator(
      self,
  ):
    """When all delegations fulfilled, _find_agent_to_run returns
    coordinator."""
    coord = StubNode(name='parent')
    agent_a = StubNode(name='agent_a')
    mesh = _Mesh(name='parent', nodes=[coord, agent_a])

    request = _make_request_task_event(
        'parent',
        {'fc_A': {'agentName': 'agent_a', 'input': {}}},
        node_path='root/parent',
    )
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:root/parent/agent_a.agent_a.fc_A',
        node_path='root/parent/agent_a',
    )

    ctx = await _create_context(
        node_path='root/parent',
        session_events=[request, finish_a],
    )
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    assert result is coord

  @pytest.mark.asyncio
  async def test_find_agent_to_run_finish_then_unfulfilled(self):
    """finish_task as most recent event, but sibling unfulfilled."""
    coord = StubNode(name='parent')
    agent_a = StubNode(name='agent_a')
    agent_b = StubNode(name='agent_b')
    mesh = _Mesh(name='parent', nodes=[coord, agent_a, agent_b])

    request = _make_request_task_event(
        'parent',
        {
            'fc_A': {'agentName': 'agent_a', 'input': {}},
            'fc_B': {'agentName': 'agent_b', 'input': {}},
        },
        node_path='root/parent',
    )
    finish_a = _make_finish_task_event(
        'agent_a',
        {'output': {'result': 'done A'}},
        branch='task:root/parent/agent_a.agent_a.fc_A',
        node_path='root/parent/agent_a',
    )

    # finish_a is the most recent event.
    ctx = await _create_context(
        node_path='root/parent',
        session_events=[request, finish_a],
    )
    ic = ctx.get_invocation_context()

    result, _ = mesh._find_agent_to_run(ic, mesh_path='root/parent')

    # Should find agent_b (unfulfilled), not coordinator.
    assert result is agent_b
