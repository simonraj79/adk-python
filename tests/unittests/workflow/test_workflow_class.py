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
import uuid

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._workflow_class import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
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
@pytest.mark.asyncio
async def test_resume_after_interrupt():
  """Workflow resumes from HITL interrupt and continues execution.

  Maps to: test_resume_behavior in test_workflow_agent.py.
  Old test used crash/checkpoint resume. New test uses HITL interrupt/FR.
  """

  class _InterruptOnce(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'resumed'
        return

      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  a = _OutputNode(name='NodeA', value='A')
  b = _InterruptOnce(name='NodeB')
  c = _OutputNode(name='NodeC', value='C')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # A completed, B interrupted
  assert 'A' in [e.output for e in events1 if e.output is not None]
  assert any(e.long_running_tool_ids for e in events1)
  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  # Run 2: resume B, then C runs
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id=fc_id, response={'ok': True}
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'resumed' in outputs
  assert 'C' in outputs


# 5. test_agent_state_event_recorded
@pytest.mark.asyncio
async def test_internal_interrupt_event_not_persisted():
  """Workflow interrupt event is _adk_internal — not in session.

  Maps to: test_agent_state_event_recorded in test_workflow_agent.py.
  Old test verified checkpoint events. New test verifies the workflow's
  interrupt event is NOT persisted (child's event is sufficient).
  """

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  wf = Workflow(name='wf', edges=[(START, _InterruptNode(name='ask'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='go')], role='user')
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  # Caller should see the child's interrupt event
  assert any(e.long_running_tool_ids for e in events)

  # Session should NOT have the workflow-level interrupt event
  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  wf_interrupt_events = [
      e
      for e in updated.events
      if e.long_running_tool_ids and e.node_info.path == 'wf'
  ]
  assert wf_interrupt_events == []
  # But child's interrupt event SHOULD be in session
  child_interrupt_events = [
      e
      for e in updated.events
      if e.long_running_tool_ids and e.node_info.path == 'wf/ask'
  ]
  assert len(child_interrupt_events) == 1


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
          (START, a, gate, downstream),
          (START, b, gate),
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


# 31. test_run_id_uniqueness
@pytest.mark.asyncio
async def test_run_id_unique_per_node():
  """Each node run gets a unique run_id.

  Maps to: test_run_id_uniqueness in test_workflow_agent.py.
  """
  a = _OutputNode(name='a', value='A')
  b = _OutputNode(name='b', value='B')
  wf = Workflow(name='wf', edges=[(START, a, b)])

  events, _, _ = await _run_workflow(wf)
  run_ids = [
      e.node_info.run_id for e in events if e.node_info.run_id
  ]

  assert len(run_ids) >= 2
  assert len(set(run_ids)) >= 2


# 32. test_run_id_uniqueness_nested
@pytest.mark.asyncio
async def test_run_id_unique_nested():
  """Nested workflow nodes also get unique IDs.

  Maps to: test_run_id_uniqueness_nested
  in test_workflow_agent.py.
  """
  inner_a = _OutputNode(name='inner_a', value='IA')
  inner = Workflow(name='inner', edges=[(START, inner_a)])
  outer_a = _OutputNode(name='outer_a', value='OA')
  wf = Workflow(name='wf', edges=[(START, outer_a, inner)])

  events, _, _ = await _run_workflow(wf)
  run_ids = [
      e.node_info.run_id for e in events if e.node_info.run_id
  ]

  assert len(set(run_ids)) >= 2


# 33. test_resume_with_manual_state_verifies_input_persistence
@pytest.mark.asyncio
async def test_resume_downstream_receives_output():
  """After resume, downstream node receives the resumed node's output.

  Maps to: test_resume_with_manual_state_verifies_input_persistence
  in test_workflow_agent.py. Old test verified input from checkpoint.
  New test verifies output flows to downstream after HITL resume.
  """

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:

      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield f'response:{ctx.resume_inputs[fc_id]["value"]}'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='get', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  a = _OutputNode(name='NodeA', value='from_a')
  b = _InterruptNode(name='NodeB')
  c = _InputCapturingNode(name='NodeC')
  wf = Workflow(name='wf', edges=[(START, a, b, c)])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # A completed, B interrupted
  assert 'from_a' in [e.output for e in events1 if e.output is not None]
  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]
  assert fc_id

  # Run 2: resume B with value, C receives B's output
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='get', id=fc_id, response={'value': 42}
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # NodeC should have captured NodeB's resume output
  assert c.received_inputs == ['response:42']


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


# ---------------------------------------------------------------------------
# use_as_output=True (dynamic output delegation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_as_output_function_to_function():
  """Node A delegates output to dynamic child B via use_as_output=True.

  B's output event appears; A's duplicate is suppressed.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(func_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # func_b emits output; func_a's duplicate is suppressed.
  assert ('func_b', 'from_b') in by_node
  assert not any(name == 'func_a' for name, _ in by_node)

  # output_for includes child, parent, and workflow paths.
  # func_a is terminal, so its output also represents the workflow's.
  output_event = [e for e in events if e.node_info.name == 'func_b'][0]
  assert output_event.node_info.output_for == [
      'wf/func_a/func_b',
      'wf/func_a',
      'wf',
  ]


@pytest.mark.asyncio
async def test_use_as_output_function_to_workflow():
  """Node A delegates output to a nested Workflow via use_as_output=True.

  The inner Workflow's terminal node output is used as A's output.
  """
  from google.adk.workflow._function_node import FunctionNode

  def step_1() -> str:
    return 'step_1_done'

  def step_2(node_input: str) -> str:
    return f'final:{node_input}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, step_1, step_2)],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # step_2 is the terminal; func_a's duplicate is suppressed.
  assert ('step_2', 'final:step_1_done') in by_node
  assert not any(name == 'func_a' for name, _ in by_node)


@pytest.mark.asyncio
async def test_use_as_output_custom_node():
  """Custom BaseNode delegates output via use_as_output."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'delegated_output'

  class _Delegator(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      await ctx.run_node(func_b, use_as_output=True)
      return
      yield

  wf = Workflow(name='wf', edges=[(START, _Delegator(name='delegator'))])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('func_b', 'delegated_output') in by_node
  assert not any(name == 'delegator' for name, _ in by_node)


@pytest.mark.asyncio
async def test_use_as_output_nested_delegation():
  """Chained delegation: A → B → C, all use_as_output=True.

  Only C's output event should appear; A and B are suppressed.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('func_c', 'from_c') in by_node
  assert not any(name in ('func_a', 'func_b') for name, _ in by_node)

  # output_for includes full ancestor chain plus workflow path.
  # func_a is terminal, so the chain extends to the workflow.
  output_event = [e for e in events if e.node_info.name == 'func_c'][0]
  assert output_event.node_info.output_for == [
      'wf/func_a/func_b/func_c',
      'wf/func_a/func_b',
      'wf/func_a',
      'wf',
  ]


@pytest.mark.asyncio
async def test_use_as_output_with_downstream():
  """Delegated output flows to downstream node via graph edge.

  A delegates to B via use_as_output. downstream is after A and
  receives B's output as node_input.
  """
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(func_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  downstream = _InputCapturingNode(name='downstream')

  wf = Workflow(name='wf', edges=[(START, node_a, downstream)])
  events, _, _ = await _run_workflow(wf)

  assert downstream.received_inputs == ['from_b']


@pytest.mark.asyncio
async def test_use_as_output_duplicate_raises():
  """Calling use_as_output=True twice in the same node raises ValueError."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  def func_c() -> str:
    return 'from_c'

  async def func_a(ctx: Context):
    await ctx.run_node(func_b, use_as_output=True)
    await ctx.run_node(func_c, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  # The second use_as_output=True call causes func_a to fail.
  # func_a never completes, so no output from func_a appears.
  by_node = _output_by_node(events)
  assert not any(name == 'func_a' for name, _ in by_node)


@pytest.mark.asyncio
async def test_without_use_as_output_parent_emits_duplicate():
  """Without use_as_output, parent re-emits child's output (duplicate)."""
  from google.adk.workflow._function_node import FunctionNode

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf = Workflow(name='wf', edges=[(START, node_a)])
  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  # Both func_b AND func_a emit output (duplicate).
  assert ('func_b', 'from_b') in by_node
  assert ('func_a', 'from_b') in by_node

  # func_b's output_for is just its own path (no delegation).
  b_event = [e for e in events if e.node_info.name == 'func_b'][0]
  assert b_event.node_info.output_for == ['wf/func_a/func_b']
  # func_a is terminal, so its output_for includes the workflow path.
  a_event = [e for e in events if e.node_info.name == 'func_a'][0]
  assert a_event.node_info.output_for == ['wf/func_a', 'wf']


@pytest.mark.asyncio
async def test_terminal_node_output_dedup():
  """Terminal node output is not duplicated by the workflow.

  The terminal node's output event includes the workflow path in
  output_for, and the workflow does not emit a separate output event.
  """

  def step_a(node_input: str) -> str:
    return node_input.upper()

  def step_b(node_input: str) -> str:
    return f'final: {node_input}'

  wf = Workflow(name='wf', edges=[(START, step_a, step_b)])
  events, _, _ = await _run_workflow(wf, 'hello')

  output_events = [e for e in events if e.output is not None]

  # step_a is not terminal — its output_for is just its own path.
  a_events = [e for e in output_events if e.node_info.name == 'step_a']
  assert len(a_events) == 1
  assert a_events[0].node_info.output_for == ['wf/step_a']

  # step_b is terminal — its output_for includes the workflow path.
  b_events = [e for e in output_events if e.node_info.name == 'step_b']
  assert len(b_events) == 1
  assert b_events[0].node_info.output_for == ['wf/step_b', 'wf']
  assert b_events[0].output == 'final: HELLO'

  # No duplicate output event from the workflow itself.
  wf_events = [e for e in output_events if e.node_info.path == 'wf']
  assert len(wf_events) == 0


@pytest.mark.asyncio
async def test_terminal_node_output_dedup_nested():
  """Terminal output dedup works for nested workflows.

  Inner workflow's terminal node output propagates to the outer
  workflow without duplication.
  """

  def inner_node(node_input: str) -> str:
    return node_input.upper()

  inner_wf = Workflow(name='inner', edges=[(START, inner_node)])

  def outer_node(node_input: str) -> str:
    return f'wrapped: {node_input}'

  outer_wf = Workflow(name='outer', edges=[(START, inner_wf, outer_node)])
  events, _, _ = await _run_workflow(outer_wf, 'test')

  output_events = [e for e in events if e.output is not None]

  # inner_node is terminal in inner_wf — includes inner_wf path.
  inner_events = [e for e in output_events if e.node_info.name == 'inner_node']
  assert len(inner_events) == 1
  assert inner_events[0].node_info.output_for == [
      'outer/inner/inner_node',
      'outer/inner',
  ]

  # outer_node is terminal in outer_wf — includes outer_wf path.
  outer_events = [e for e in output_events if e.node_info.name == 'outer_node']
  assert len(outer_events) == 1
  assert outer_events[0].node_info.output_for == [
      'outer/outer_node',
      'outer',
  ]
  assert outer_events[0].output == 'wrapped: TEST'

  # No duplicate output events from the workflows themselves.
  wf_output_events = [
      e for e in output_events if e.node_info.path in ('outer', 'outer/inner')
  ]
  assert len(wf_output_events) == 0


# --- Nested workflow tests ---


@pytest.mark.asyncio
async def test_nested_workflow_completes():
  """Inner workflow runs to completion, outer continues downstream."""
  inner_node = _OutputNode(name='inner_node', value='inner_result')
  inner_wf = Workflow(name='inner_wf', edges=[(START, inner_node)])
  before = _OutputNode(name='before', value='before_result')
  after = _InputCapturingNode(name='after')
  wf = Workflow(name='wf', edges=[(START, before, inner_wf, after)])

  events, _, _ = await _run_workflow(wf)

  by_node = _output_by_node(events)
  assert ('before', 'before_result') in by_node
  assert ('inner_node', 'inner_result') in by_node
  assert after.received_inputs == ['inner_result']


@pytest.mark.asyncio
async def test_nested_workflow_event_author():
  """Events are authored by the nearest orchestrator (workflow/agent).

  Setup: outer_wf → inner_wf → inner_node.
  Assert:
    - inner_node's events are authored by inner_wf (nearest).
    - outer_wf's direct children are authored by outer_wf.
    - inner_wf overrides the author for its subtree.
  """
  inner_node = _OutputNode(name='inner_node', value='inner_result')
  inner_wf = Workflow(name='inner_wf', edges=[(START, inner_node)])
  outer_node = _OutputNode(name='outer_node', value='outer_result')
  wf = Workflow(
      name='outer_wf',
      edges=[(START, outer_node, inner_wf)],
  )

  events, _, _ = await _run_workflow(wf)

  # outer_node's events authored by outer_wf (nearest orchestrator).
  outer_events = [
      e for e in events if e.node_info.path == 'outer_wf/outer_node'
  ]
  assert outer_events
  assert all(e.author == 'outer_wf' for e in outer_events)

  # inner_node's events authored by inner_wf (nearest orchestrator),
  # NOT outer_wf.
  inner_events = [
      e for e in events if e.node_info.path == 'outer_wf/inner_wf/inner_node'
  ]
  assert inner_events
  assert all(e.author == 'inner_wf' for e in inner_events)


@pytest.mark.asyncio
async def test_nested_workflow_interrupt_and_resume():
  """Inner workflow child interrupts, outer resumes on FR."""

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_nested_fc')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_nested_fc'] = None
        response = ctx.resume_inputs[fc_id]['answer']
        yield f'approved:{response}'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_nested_fc'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='inner_tool', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, _InterruptNode(name='approval'))],
  )
  before = _OutputNode(name='before', value='before_result')
  after = _InputCapturingNode(name='after')
  wf = Workflow(
      name='wf',
      edges=[(START, before, inner_wf, after)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: before completes, inner_wf/approval interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # Should have interrupt from inner's child
  interrupt_events = [e for e in events1 if e.long_running_tool_ids]
  assert len(interrupt_events) == 1
  assert interrupt_events[0].long_running_tool_ids is not None
  fc_id = list(interrupt_events[0].long_running_tool_ids)[0]

  # Workflow-level interrupt events should NOT be persisted
  # (they're _adk_internal). Only the leaf child's event at
  # 'wf/inner_wf/approval' should have interrupt ids in session.
  updated_session = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  assert updated_session is not None
  wf_interrupt_events = [
      e
      for e in updated_session.events
      if e.long_running_tool_ids and e.node_info.path in ('wf', 'wf/inner_wf')
  ]
  assert wf_interrupt_events == []

  # Run 2: resume
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='inner_tool',
                  id=fc_id,
                  response={'answer': 'yes'},
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # Inner resumed, after should receive inner's output
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'approved:yes' in outputs
  assert after.received_inputs == ['approved:yes']


# --- wait_for_output + HITL resume ---


@pytest.mark.asyncio
async def test_wait_for_output_node_preserved_across_resume():
  """wait_for_output node is marked WAITING on resume, not re-triggered.

  Scenario:
    START → A (completes), B (interrupts)
    A → Gate(wait_for_output, needs 2 triggers)
    B → Gate
    Gate → Downstream

  Run 1: A completes → triggers Gate (1/2, no output). B interrupts.
  Run 2: Resume B → B completes → triggers Gate (2/2, outputs).
         Gate opens → Downstream runs.

  Without _add_wait_for_output_nodes, Gate would be treated as fresh
  on resume and only receive 1 trigger (from B), never opening.
  """

  class _InterruptOnce(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'B_done'
        return
      fc_id = f'fc-{uuid.uuid4().hex[:8]}'
      ctx.state['_fc_id'] = fc_id
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  a = _OutputNode(name='NodeA', value='A')
  b = _InterruptOnce(name='NodeB')
  gate = JoinNode(name='Gate')
  downstream = _OutputNode(name='Downstream', value='done')
  wf = Workflow(
      name='wf',
      edges=[
          (START, a, gate, downstream),
          (START, b, gate),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: A completes, B interrupts, Gate triggered once (no output)
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  outputs1 = [e.output for e in events1 if e.output is not None]
  assert 'A' in outputs1
  assert any(e.long_running_tool_ids for e in events1)
  # Gate should NOT have opened (only 1 of 2 triggers received)
  assert not any(isinstance(o, dict) and 'NodeA' in o for o in outputs1)

  fc_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]

  # Run 2: resume B → Gate gets 2nd trigger → opens → Downstream runs
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='approve', id=fc_id, response={'ok': True}
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'B_done' in outputs
  # Gate opened with both inputs collected
  gate_output = [o for o in outputs if isinstance(o, dict)]
  assert len(gate_output) == 1
  assert 'NodeA' in gate_output[0]
  assert 'NodeB' in gate_output[0]
  assert 'done' in outputs


# --- run_id reuse on resume ---


@pytest.mark.asyncio
async def test_run_id_reused_on_resume():
  """Resumed node reuses run_id from original interrupted run."""

  class _InterruptOnce(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      fc_id = ctx.state.get('_fc_id')
      if fc_id and ctx.resume_inputs and fc_id in ctx.resume_inputs:
        ctx.state['_fc_id'] = None
        yield 'resumed'
        return
      fc_id = str(uuid.uuid4())
      ctx.state['_fc_id'] = fc_id
      yield RequestInput(interrupt_id=fc_id)

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptOnce(name='ask'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  fc_id = None
  original_run_id = None
  for e in events1:
    if e.long_running_tool_ids:
      fc_id = list(e.long_running_tool_ids)[0]
      original_run_id = e.node_info.run_id

  assert fc_id is not None
  assert original_run_id is not None

  # Run 2: resume
  msg2 = types.Content(
      parts=[create_request_input_response(fc_id, {'ok': True})],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  # Resumed node should have the same run_id
  resumed_events = [
      e
      for e in events2
      if e.node_info.path == 'wf/ask' and e.output is not None
  ]
  assert len(resumed_events) == 1
  assert resumed_events[0].node_info.run_id == original_run_id
