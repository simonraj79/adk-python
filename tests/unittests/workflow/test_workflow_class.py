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

"""Tests for the new Workflow(BaseNode) implementation.

Migrated from test_workflow_agent.py — each test validates the same
workflow behavior through Runner(node=...) instead of agent.run_async().
"""

from collections import Counter
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow_class import Workflow
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest

# ---------------------------------------------------------------------------
# Shared helper nodes (used by multiple tests)
# ---------------------------------------------------------------------------


class _OutputNode(BaseNode):
  """Yields a fixed output value."""

  value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield self.value


class _PassthroughNode(BaseNode):
  """Passes node_input through as output."""

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield node_input


class _RouteNode(BaseNode):
  """Yields output and sets a route."""

  value: Any = None
  route_value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    yield Event(output=self.value, route=self.route_value)


class _InputCapturingNode(BaseNode):
  """Captures node_input for later assertion."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  received_inputs: list[Any] = Field(default_factory=list)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.received_inputs.append(node_input)
    yield {'received': node_input}


class _ContextCapturingNode(BaseNode):
  """Captures ctx fields for later assertion."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  captured_triggered_by: list[str] = Field(default_factory=list)
  captured_in_nodes: list[set] = Field(default_factory=list)

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    self.captured_triggered_by.append(ctx.triggered_by)
    self.captured_in_nodes.append(set(ctx.in_nodes))
    yield node_input


class _IntermediateContentNode(BaseNode):
  """Yields intermediate content events before output."""

  contents: list[types.Content] = Field(default_factory=list)
  value: Any = None

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    for content in self.contents:
      yield Event(content=content)
    if self.value is not None:
      yield self.value


class _BytesOutputNode(BaseNode):
  """Yields bytes content or raw bytes."""

  raw_bytes: bool = False

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    data = b'\x89PNG\r\n\x1a\n'
    if self.raw_bytes:
      yield data
    else:
      yield Event(
          content=types.Content(
              parts=[types.Part.from_bytes(data=data, mime_type='image/png')]
          ),
          output='bytes_sent',
      )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_workflow(wf, message='start'):
  """Run a Workflow through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _outputs(events):
  """Extract non-None outputs from events."""
  return [e.output for e in events if e.output is not None]


def _output_by_node(events):
  """Extract (node_name_from_path, output) for child node events."""
  results = []
  for e in events:
    if e.output is not None and e.node_info.path and '/' in e.node_info.path:
      node_name = e.node_info.path.rsplit('/', 1)[-1]
      results.append((node_name, e.output))
  return results


# ---------------------------------------------------------------------------
# Tests — 1:1 mapping from test_workflow_agent.py
# ---------------------------------------------------------------------------


# 1. test_run_async → sequential A→B
@pytest.mark.asyncio
async def test_sequential_two_nodes():
  """Sequential A->B produces both outputs in order.

  Maps to: test_run_async in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  b = _OutputNode(name='NodeB', value='World')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('NodeA', 'Hello') in by_node
  assert ('NodeB', 'World') in by_node
  assert by_node.index(('NodeA', 'Hello')) < by_node.index(('NodeB', 'World'))


# 2. test_run_async_with_intermediate_content
@pytest.mark.asyncio
async def test_intermediate_content_before_output():
  """Node yields content events before output.

  Maps to: test_run_async_with_intermediate_content in test_workflow_agent.py.
  """
  a = _IntermediateContentNode(
      name='NodeA',
      contents=[
          types.Content(parts=[types.Part(text='msg1')]),
          types.Content(parts=[types.Part(text='msg2')]),
      ],
      value='A output',
  )
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf)

  texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if p.text
  ]
  assert texts.index('msg1') < texts.index('msg2')  # also asserts presence
  assert 'A output' in _outputs(events)


# 3. test_run_async_with_loop_and_break
@pytest.mark.asyncio
async def test_loop_with_conditional_break():
  """Loop iterates 3 times: A,Check repeats, then exits to B.

  Maps to: test_run_async_with_loop_and_break in test_workflow_agent.py.
  """

  class _IncrementingNode(BaseNode):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    message: str = ''
    _tracker_ref: list = []

    def __init__(self, *, name: str, message: str, tracker: dict):
      super().__init__(name=name)
      object.__setattr__(self, 'message', message)
      object.__setattr__(self, '_tracker_ref', [tracker])

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      tracker = self._tracker_ref[0]
      count = tracker.get('iteration_count', 0) + 1
      tracker['iteration_count'] = count
      yield Event(
          output=self.message,
          route='continue_loop' if count < 3 else 'exit_loop',
      )

  tracker = {'iteration_count': 0}
  node_a = _OutputNode(name='NodeA', value='Looping')
  check = _IncrementingNode(
      name='CheckNode', message='Checking', tracker=tracker
  )
  node_b = _OutputNode(name='NodeB', value='Finished')
  wf = Workflow(
      name='wf',
      edges=[
          (START, node_a, check),
          (check, {'continue_loop': node_a, 'exit_loop': node_b}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert tracker['iteration_count'] == 3
  by_node = _output_by_node(events)
  node_names = [name for name, _ in by_node]
  # 3 iterations of (NodeA, CheckNode) then NodeB
  assert node_names.count('NodeA') == 3
  assert node_names.count('CheckNode') == 3
  assert 'NodeB' in node_names


# 4. test_resume_behavior
@pytest.mark.xfail(reason='Resume not yet implemented in new Workflow.')
@pytest.mark.asyncio
async def test_resume_after_failure():
  """Workflow resumes from checkpoint after failure.

  Maps to: test_resume_behavior in test_workflow_agent.py.
  """
  assert False, 'TODO: implement resume'


# 5. test_agent_state_event_recorded
@pytest.mark.xfail(reason='Checkpoint not yet implemented in new Workflow.')
@pytest.mark.asyncio
async def test_checkpoint_events_emitted():
  """Agent state checkpoint events are emitted.

  Maps to: test_agent_state_event_recorded in test_workflow_agent.py.
  """
  assert False, 'TODO: implement checkpoint'


# 6. test_run_async_with_implicit_graph
@pytest.mark.asyncio
async def test_edge_tuple_syntax():
  """Edges defined via tuples work.

  Maps to: test_run_async_with_implicit_graph in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  b = _OutputNode(name='NodeB', value='World')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (a, b),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  outputs = _outputs(events)
  assert outputs.index('Hello') < outputs.index(
      'World'
  )  # also asserts presence


# 7. test_run_async_with_string_start
@pytest.mark.asyncio
async def test_string_start_in_edges():
  """'START' string works as edge source.

  Maps to: test_run_async_with_string_start in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='Hello')
  wf = Workflow(name='wf', edges=[('START', a)])

  events, _, _ = await _run_workflow(wf)

  assert 'Hello' in _outputs(events)


# 8. test_run_async_with_implicit_graph_with_edge_combinations
@pytest.mark.asyncio
async def test_mixed_edge_and_tuple_syntax():
  """Mix of Edge objects and tuple syntax in a 3-node chain.

  Maps to: test_run_async_with_implicit_graph_with_edge_combinations
  in test_workflow_agent.py.
  """
  from google.adk.workflow._workflow_graph import Edge as GraphEdge

  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  c = _OutputNode(name='NodeC', value='C')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          GraphEdge(from_node=a, to_node=b),
          (b, c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert by_node.index(('NodeA', 'A')) < by_node.index(
      ('NodeB', 'B')
  )  # also asserts presence
  assert by_node.index(('NodeB', 'B')) < by_node.index(('NodeC', 'C'))


# 9. test_run_async_with_update_state_event
@pytest.mark.asyncio
async def test_state_update_via_event_persisted():
  """State updates via Event(state=...) are persisted to session.

  Maps to: test_run_async_with_update_state_event in test_workflow_agent.py.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'key1': 'value1'})

  wf = Workflow(name='wf', edges=[(START, _Node(name='NodeA'))])

  events, ss, session = await _run_workflow(wf)
  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )

  assert updated.state.get('key1') == 'value1'


# 10. test_run_async_with_event
@pytest.mark.asyncio
async def test_event_with_output_and_route():
  """Node yields Event with output and route — only matching target runs.

  Maps to: test_run_async_with_event in test_workflow_agent.py.
  """
  router = _RouteNode(name='NodeA', value='Hello', route_value='route_b')
  node_b = _InputCapturingNode(name='NodeB')
  node_c = _InputCapturingNode(name='NodeC')
  wf = Workflow(
      name='wf',
      edges=[
          (START, router),
          (router, {'route_b': node_b, 'route_c': node_c}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert node_b.received_inputs == ['Hello']
  assert not node_c.received_inputs


# 11. test_run_async_with_raw_output_node
@pytest.mark.asyncio
async def test_raw_output_auto_wrapped():
  """Node yields raw value, auto-wrapped to Event(output=...).

  Maps to: test_run_async_with_raw_output_node in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='raw_data')
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf)

  assert 'raw_data' in _outputs(events)


# 12. test_node_output_event_with_content_data
@pytest.mark.asyncio
async def test_content_return_produces_content_event():
  """FunctionNode returning Content yields content event, not output.

  Maps to: test_node_output_event_with_content_data
  in test_workflow_agent.py.
  """

  def content_fn() -> types.Content:
    return types.Content(parts=[types.Part(text='hello')])

  wf = Workflow(name='wf', edges=[(START, content_fn)])

  events, _, _ = await _run_workflow(wf)

  fn_events = [
      e for e in events if e.node_info.path and 'content_fn' in e.node_info.path
  ]
  assert len(fn_events) == 1
  assert fn_events[0].output is None
  assert fn_events[0].content == types.Content(parts=[types.Part(text='hello')])


# 13. test_input_propagation_linear
@pytest.mark.asyncio
async def test_input_propagation_linear():
  """Output of A becomes input to B.

  Maps to: test_input_propagation_linear in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='from_a')
  b = _InputCapturingNode(name='NodeB')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ['from_a']


# 14. test_input_propagation_fan_in_sequential
@pytest.mark.asyncio
async def test_input_propagation_fan_in():
  """Fan-in with asymmetric branches: C receives from A and B3.

  Maps to: test_input_propagation_fan_in_sequential
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='from_a')
  b = _OutputNode(name='NodeB', value='from_b')
  b2 = _PassthroughNode(name='NodeB2')
  b3 = _PassthroughNode(name='NodeB3')
  c = _InputCapturingNode(name='NodeC')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b, b2, b3),
          (a, c),
          (b3, c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['from_a', 'from_b'])


# 15. test_start_node_receives_user_content
@pytest.mark.asyncio
async def test_start_node_receives_user_content():
  """First node receives user message as input.

  Maps to: test_start_node_receives_user_content
  in test_workflow_agent.py.
  """
  a = _InputCapturingNode(name='NodeA')
  wf = Workflow(name='wf', edges=[(START, a)])

  events, _, _ = await _run_workflow(wf, message='hello')

  assert len(a.received_inputs) == 1
  assert a.received_inputs[0].parts[0].text == 'hello'


# 16. test_triggered_by_fan_in
@pytest.mark.asyncio
async def test_triggered_by_set_correctly():
  """ctx.triggered_by reflects the predecessor (asymmetric fan-in).

  Maps to: test_triggered_by_fan_in in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  c = _OutputNode(name='NodeC', value='C')
  x = _ContextCapturingNode(name='NodeX')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          (a, x),
          (b, c, x),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(x.captured_triggered_by) == Counter(['NodeA', 'NodeC'])


# 17. test_in_nodes_fan_in_sequential
@pytest.mark.asyncio
async def test_in_nodes_lists_predecessors():
  """ctx.in_nodes lists all predecessor node names (asymmetric fan-in).

  Maps to: test_in_nodes_fan_in_sequential in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='from_a')
  b = _OutputNode(name='NodeB', value='from_b')
  b2 = _PassthroughNode(name='NodeB2')
  b3 = _PassthroughNode(name='NodeB3')
  c = _ContextCapturingNode(name='NodeC')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b, b2, b3),
          (a, c),
          (b3, c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  for in_nodes in c.captured_in_nodes:
    assert in_nodes == {'NodeA', 'NodeB3'}


# 18. test_wait_for_output_suppresses_trigger
@pytest.mark.asyncio
async def test_wait_for_output_suppresses_downstream():
  """wait_for_output=True with no output prevents downstream from running.

  Maps to: test_wait_for_output_suppresses_trigger
  in test_workflow_agent.py.
  """

  class _NoOutputNode(BaseNode):
    wait_for_output: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'seen': True})

  blocker = _NoOutputNode(name='blocker')
  downstream = _OutputNode(name='downstream', value='should_not_run')
  wf = Workflow(
      name='wf',
      edges=[
          (START, blocker),
          (blocker, downstream),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert 'should_not_run' not in _outputs(events)


# 19. test_wait_for_output_retrigger_then_complete
@pytest.mark.asyncio
async def test_wait_for_output_completes_after_all():
  """Gate with wait_for_output opens after 2 triggers, fires downstream.

  Maps to: test_wait_for_output_retrigger_then_complete
  in test_workflow_agent.py.
  """

  class _GateNode(BaseNode):
    triggers_needed: int = 2
    wait_for_output: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      count = (ctx.state.get(f'{self.name}_count') or 0) + 1
      if count >= self.triggers_needed:
        yield Event(
            output='gate_open',
            state={f'{self.name}_count': None},
        )
      else:
        yield Event(state={f'{self.name}_count': count})

  gate = _GateNode(name='Gate', triggers_needed=2)
  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  downstream = _OutputNode(name='Downstream', value='done')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          (a, gate),
          (b, gate),
          (gate, downstream),
      ],
  )

  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  # A and B run first (parallel), then Gate opens, then Downstream
  assert len(by_node) == 4
  first_two = {by_node[0][0], by_node[1][0]}
  assert first_two == {'NodeA', 'NodeB'}
  assert by_node[2] == ('Gate', 'gate_open')
  assert by_node[3] == ('Downstream', 'done')


# 20-24. Input schema tests
@pytest.mark.xfail(reason='Input schema parsing not yet in new Workflow.')
@pytest.mark.asyncio
async def test_start_node_with_str_input_schema():
  """input_schema=str parses user text.

  Maps to: test_start_node_with_str_input_schema
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Input schema parsing not yet in new Workflow.')
@pytest.mark.asyncio
async def test_start_node_with_int_input_schema():
  """input_schema=int parses user text to int.

  Maps to: test_start_node_with_int_input_schema
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Input schema parsing not yet in new Workflow.')
@pytest.mark.asyncio
async def test_start_node_with_int_list_input_schema():
  """input_schema=list[int] parses JSON list.

  Maps to: test_start_node_with_int_list_input_schema
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Input schema parsing not yet in new Workflow.')
@pytest.mark.asyncio
async def test_start_node_with_invalid_input_schema():
  """Invalid input against schema raises error.

  Maps to: test_start_node_with_invalid_input_schema
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Input schema parsing not yet in new Workflow.')
@pytest.mark.asyncio
async def test_start_node_receives_parsed_user_content_with_schema():
  """Parsed input replaces raw Content for first node.

  Maps to: test_start_node_receives_parsed_user_content_with_schema
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


# 25. test_run_async_with_implicit_graph_chain
@pytest.mark.asyncio
async def test_chain_syntax():
  """(START, a, b, c) creates sequential chain producing all outputs.

  Maps to: test_run_async_with_implicit_graph_chain
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  c = _OutputNode(name='c', value='C')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])

  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)

  assert [name for name, _ in by_node] == ['a', 'b', 'c']
  assert [val for _, val in by_node] == ['A', 'B', 'C']


# 26. test_run_async_with_implicit_graph_fan_out
@pytest.mark.asyncio
async def test_fan_out():
  """Fan-out from A to (B, C) — both receive A's output.

  Maps to: test_run_async_with_implicit_graph_fan_out
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _InputCapturingNode(name='b')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (a, (b, c)),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ['A']
  assert c.received_inputs == ['A']


# 27. test_run_async_with_implicit_graph_fan_in
@pytest.mark.asyncio
async def test_fan_in():
  """((a, b), c) — c triggered by both a and b.

  Maps to: test_run_async_with_implicit_graph_fan_in
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, (a, b)),
          ((a, b), c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['A', 'B'])


# 28. test_run_async_with_implicit_graph_fan_out_fan_in
@pytest.mark.asyncio
async def test_fan_out_fan_in():
  """S fans out to (A, B), both feed into C.

  Maps to: test_run_async_with_implicit_graph_fan_out_fan_in
  in test_workflow_agent.py.
  """
  s = _OutputNode(name='s', value='S')
  a = _PassthroughNode(name='a')
  b = _PassthroughNode(name='b')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, s),
          (s, (a, b), c),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  # Both A and B receive S's output, C receives from both
  assert Counter(c.received_inputs) == Counter(['S', 'S'])


# 29. test_run_async_parallel_nodes_interleaved_events
@pytest.mark.asyncio
async def test_parallel_events_interleaved():
  """Parallel nodes both produce output events.

  Maps to: test_run_async_parallel_nodes_interleaved_events
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert 'A' in _outputs(events)
  assert 'B' in _outputs(events)


# 30. test_buffers_events_from_parallel_nodes
@pytest.mark.asyncio
async def test_parallel_events_all_delivered():
  """All events from parallel nodes fan-in to capture node.

  Maps to: test_buffers_events_from_parallel_nodes
  in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  capture = _InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          (a, capture),
          (b, capture),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(capture.received_inputs) == Counter(['A', 'B'])


# 31. test_execution_id_uniqueness
@pytest.mark.asyncio
async def test_execution_id_unique_per_node():
  """Each node run gets a unique execution_id.

  Maps to: test_execution_id_uniqueness in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)
  exec_ids = [
      e.node_info.execution_id for e in events if e.node_info.execution_id
  ]

  assert len(exec_ids) >= 2
  assert len(set(exec_ids)) >= 2


# 32. test_execution_id_uniqueness_nested
@pytest.mark.asyncio
async def test_execution_id_unique_nested():
  """Nested workflow nodes also get unique IDs.

  Maps to: test_execution_id_uniqueness_nested
  in test_workflow_agent.py.
  """
  inner_a = _OutputNode(name='inner_a', value='IA')
  inner = Workflow(name='inner', edges=[(START, inner_a)])
  outer_a = _OutputNode(name='outer_a', value='OA')
  wf = Workflow(name='wf', edges=[(START, outer_a, inner)])

  events, _, _ = await _run_workflow(wf)
  exec_ids = [
      e.node_info.execution_id for e in events if e.node_info.execution_id
  ]

  assert len(set(exec_ids)) >= 2


# 33. test_resume_with_manual_state_verifies_input_persistence
@pytest.mark.xfail(reason='Resume not yet implemented in new Workflow.')
@pytest.mark.asyncio
async def test_resume_preserves_inputs():
  """Resume preserves original node inputs.

  Maps to: test_resume_with_manual_state_verifies_input_persistence
  in test_workflow_agent.py.
  """
  assert False, 'TODO: implement resume'


# 34. test_run_async_with_multiple_node_outputs_fails
@pytest.mark.xfail(reason='Runner does not propagate background task errors.')
@pytest.mark.asyncio
async def test_multiple_outputs_rejected():
  """Node yielding two outputs raises error.

  Maps to: test_run_async_with_multiple_node_outputs_fails
  in test_workflow_agent.py.

  NodeRunner raises ValueError but Runner swallows it in the
  background task. This test is xfail until Runner error propagation
  is implemented.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(output='first')
      yield Event(output='second')

  wf = Workflow(name='wf', edges=[(START, _Node(name='a'))])

  with pytest.raises(ValueError, match='at most one output'):
    await _run_workflow(wf)


# 35. test_run_async_with_implicit_graph_fan_in_with_route
@pytest.mark.asyncio
async def test_fan_in_with_route():
  """Fan-in with conditional routes — both route to same target.

  Maps to: test_run_async_with_implicit_graph_fan_in_with_route
  in test_workflow_agent.py.
  """
  a = _RouteNode(name='a', value='A', route_value='route1')
  b = _RouteNode(name='b', value='B', route_value='route1')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          ((a, b), {'route1': c}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['A', 'B'])


# 36. test_run_async_with_implicit_graph_fan_out_with_route
@pytest.mark.asyncio
async def test_fan_out_with_route():
  """Fan-out via route — both targets receive router's output.

  Maps to: test_run_async_with_implicit_graph_fan_out_with_route
  in test_workflow_agent.py.
  """
  router = _RouteNode(name='r', value='R', route_value='route1')
  b = _InputCapturingNode(name='b')
  c = _InputCapturingNode(name='c')
  wf = Workflow(
      name='wf',
      edges=[
          (START, router),
          (router, {'route1': (b, c)}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ['R']
  assert c.received_inputs == ['R']


# 37. test_run_async_with_implicit_graph_fan_in_out_with_route
@pytest.mark.asyncio
async def test_fan_in_out_with_route():
  """Fan-in/out with routes — both routers feed both targets.

  Maps to: test_run_async_with_implicit_graph_fan_in_out_with_route
  in test_workflow_agent.py.
  """
  a = _RouteNode(name='a', value='A', route_value='route1')
  b = _RouteNode(name='b', value='B', route_value='route1')
  c = _InputCapturingNode(name='c')
  d = _InputCapturingNode(name='d')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a),
          (START, b),
          ((a, b), {'route1': (c, d)}),
      ],
  )

  events, _, _ = await _run_workflow(wf)

  assert Counter(c.received_inputs) == Counter(['A', 'B'])
  assert Counter(d.received_inputs) == Counter(['A', 'B'])


# 38. test_run_async_streaming_behavior
@pytest.mark.asyncio
async def test_streaming_partial_events():
  """Partial events are streamed before final output.

  Maps to: test_run_async_streaming_behavior
  in test_workflow_agent.py.
  """

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(
          content=types.Content(parts=[types.Part(text='partial1')]),
          partial=True,
      )
      yield Event(
          content=types.Content(parts=[types.Part(text='partial2')]),
          partial=True,
      )
      yield Event(output='final')

  wf = Workflow(name='wf', edges=[(START, _Node(name='a'))])

  events, _, _ = await _run_workflow(wf)
  partials = [e for e in events if e.partial]

  assert len(partials) == 2
  assert 'final' in _outputs(events)


# 39. test_workflow_agent_unsupported_base_agent_fields
# SKIP — BaseAgent-specific, not applicable to Workflow(BaseNode).


# 40. test_node_path_generation
@pytest.mark.asyncio
async def test_node_path_correct():
  """Events have correct node_info.path.

  Maps to: test_node_path_generation in test_workflow_agent.py.
  """
  a = _OutputNode(name='NodeA', value='A')
  b = _OutputNode(name='NodeB', value='B')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)
  paths = [e.node_info.path for e in events if e.node_info.path]

  assert any(p.endswith('NodeA') for p in paths)
  assert any(p.endswith('NodeB') for p in paths)
  assert all('None' not in p for p in paths)


# 41. test_bytes_in_content_output_e2e
@pytest.mark.asyncio
async def test_bytes_in_content_output():
  """Content with bytes propagates to downstream node.

  Maps to: test_bytes_in_content_output_e2e
  in test_workflow_agent.py.
  """
  a = _BytesOutputNode(name='a', raw_bytes=False)
  b = _InputCapturingNode(name='b')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert b.received_inputs == ['bytes_sent']


# 42. test_raw_bytes_output_e2e
@pytest.mark.asyncio
async def test_raw_bytes_output():
  """Raw bytes output propagates to downstream node.

  Maps to: test_raw_bytes_output_e2e in test_workflow_agent.py.
  """
  a = _BytesOutputNode(name='a', raw_bytes=True)
  b = _InputCapturingNode(name='b')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)

  assert len(b.received_inputs) == 1
  assert isinstance(b.received_inputs[0], bytes)


# 43-46. Bytes serialization round-trip tests
@pytest.mark.xfail(reason='Checkpoint/resume not yet in new Workflow.')
@pytest.mark.asyncio
async def test_bytes_in_node_input_serialization():
  """Bytes in node input survive checkpoint/resume.

  Maps to: test_bytes_in_node_input_serialization_round_trip
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Checkpoint/resume not yet in new Workflow.')
@pytest.mark.asyncio
async def test_bytes_in_typed_model_input():
  """Bytes in Pydantic model input survive round-trip.

  Maps to: test_bytes_in_typed_model_input_round_trip
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Checkpoint/resume not yet in new Workflow.')
@pytest.mark.asyncio
async def test_bytes_in_trigger_buffer():
  """Bytes in trigger buffer survive serialization.

  Maps to: test_bytes_in_trigger_buffer_serialization
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


@pytest.mark.xfail(reason='Checkpoint/resume not yet in new Workflow.')
@pytest.mark.asyncio
async def test_bytes_full_workflow_resume():
  """Full resume with bytes data end-to-end.

  Maps to: test_bytes_input_full_workflow_resume
  in test_workflow_agent.py.
  """
  assert False, 'TODO'


# --- Additional: function node auto-wrap ---


@pytest.mark.asyncio
async def test_function_node_auto_wrap():
  """Callable in edge is auto-wrapped to FunctionNode."""

  def greet(node_input):
    return 'hi'

  wf = Workflow(name='wf', edges=[(START, greet)])

  events, _, _ = await _run_workflow(wf)

  assert 'hi' in _outputs(events)


# --- Additional: empty workflow ---


@pytest.mark.asyncio
async def test_empty_workflow():
  """Workflow with no edges produces no output."""
  wf = Workflow(name='wf', edges=[])

  events, _, _ = await _run_workflow(wf)

  assert _outputs(events) == []
