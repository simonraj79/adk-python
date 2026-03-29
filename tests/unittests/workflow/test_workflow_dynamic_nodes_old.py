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

"""Tests for dynamic node scheduling in old Workflow (_workflow.py).

Covers the three cases from doc 18 (Dynamic Node Resume via Lazy Scan):
- Case 1: Fresh execution (no prior events)
- Case 2: Completed dedup (return cached output on rerun)
- Case 3: Interrupted resume (rerun with resume_inputs)

Plus edge cases for multiple dynamic nodes, nested dynamic nodes,
and use_as_output delegation.

TODO: this is only for comparison with test_workflow_class_dynamic_nodes.py.
This will be deleted after migration.
"""

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._dynamic_node_registry import dynamic_node_registry
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest


@pytest.fixture(autouse=True)
def _clear_registry():
  """Clear dynamic_node_registry between tests."""
  dynamic_node_registry.clear()
  yield
  dynamic_node_registry.clear()


# --- Helpers ---


async def _run(
    runner: Runner,
    ss: InMemorySessionService,
    session: Any,
    message: str,
) -> list[Event]:
  """Send a text message and collect events."""
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events


async def _resume(
    runner: Runner,
    ss: InMemorySessionService,
    session: Any,
    fc_id: str,
    response: Any,
) -> list[Event]:
  """Send a function response and collect events."""
  if not isinstance(response, dict):
    response = {'value': response}
  msg = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id=fc_id, response=response
              )
          )
      ],
      role='user',
  )
  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events


def _outputs(events: list[Event]) -> list[Any]:
  """Extract non-None outputs."""
  return [e.output for e in events if e.output is not None]


def _interrupt_ids(events: list[Event]) -> set[str]:
  """Extract interrupt IDs from events."""
  ids: set[str] = set()
  for e in events:
    if e.long_running_tool_ids:
      ids.update(e.long_running_tool_ids)
  return ids


# =========================================================================
# Fresh execution (no resume)
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_fresh_execution():
  """Dynamic node runs normally on first invocation.

  Setup: Parent (rerun_on_resume=True) calls ctx.run_node(Child).
  Action: Send a text message to trigger the workflow.
  Assert: Parent receives Child's output and yields the combined result.
  """

  class _Child(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'child got: {node_input}'

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'), node_input='hello')
      yield f'parent got: {result}'

  wf = Workflow(name='wf', edges=[(START, _Parent(name='parent'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'go')

  outputs = _outputs(events)
  assert 'parent got: child got: hello' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_with_downstream_static():
  """Dynamic child's output flows to a downstream static node.

  Setup: Parent calls ctx.run_node(Child). Parent is followed by
    a static After node in the graph.
  Action: Send a text message.
  Assert: After node receives Parent's output (which includes
    Child's result) as node_input.
  """

  class _Child(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'child: {node_input}'

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'), node_input='forwarded')
      yield result

  class _After(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'after: {node_input}'

  wf = Workflow(
      name='wf',
      edges=[(START, _Parent(name='parent'), _After(name='after'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  events = await _run(runner, ss, session, 'hello')
  outputs = _outputs(events)
  assert 'after: child: forwarded' in outputs


# =========================================================================
# Single-level resume (interrupt → FR → resume)
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_node_interrupted_resume():
  """Dynamic child interrupts, then resumes with FR on parent rerun.

  Setup: Parent calls ctx.run_node(Approver). Approver interrupts
    with fc-1.
  Action: Send FR for fc-1 with {answer: 'yes'}.
  Assert:
    - Approver resumes and outputs 'approved: yes'.
    - Parent yields the final combined result.
  """

  class _Approver(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]["answer"]}'
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='approve', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Approver(name='approver'))
      yield f'final: {result}'

  wf = Workflow(name='wf', edges=[(START, _Parent(name='parent'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', {'answer': 'yes'})
  outputs = _outputs(events2)
  assert 'final: approved: yes' in outputs


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_node_completed_dedup_on_resume():
  """Completed dynamic child returns cached output when parent reruns.

  Setup: Parent calls ctx.run_node(Completer) then
    ctx.run_node(Interrupter). Completer completes, Interrupter
    interrupts with fc-1.
  Action: Send FR for fc-1 to resume.
  Assert:
    - Parent reruns (call_count increments to 2).
    - Completer is NOT re-executed (dedup returns cached output).
    - Interrupter resumes with the FR response.
    - Final output combines both results.
  """

  class _Completer(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield 'completed_result'

  class _Interrupter(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-1"]["value"]}'
        return
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

  call_count = [0]

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      call_count[0] += 1

      r1 = await ctx.run_node(_Completer(name='completer'))
      r2 = await ctx.run_node(_Interrupter(name='interrupter'))
      yield f'{r1} + {r2}'

  wf = Workflow(name='wf', edges=[(START, _Parent(name='parent'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: completer completes, interrupter interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert _interrupt_ids(events1)
  assert call_count[0] == 1

  # Run 2: resume interrupter
  events2 = await _resume(runner, ss, session, 'fc-1', 'done')

  # Parent should have rerun (call_count=2).
  # Completer should NOT re-execute (dedup returns cached).
  # Interrupter resumes with FR.
  assert call_count[0] == 2
  outputs = _outputs(events2)
  assert 'completed_result + resumed: done' in outputs


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_node_sequential_interrupts():
  """Sequential dynamic children interrupt one at a time.

  Setup: Parent calls ctx.run_node(a) then ctx.run_node(b).
    Both children interrupt, but sequentially — a interrupts first,
    parent never reaches b until a is resolved.
  Action: Resume a, then b.
  Assert:
    - Run 1: only fc-a interrupts (parent didn't reach b).
    - Run 2 (resume fc-a): a completes, parent continues to b,
      b interrupts with fc-b.
    - Run 3 (resume fc-b): b completes, parent yields combined
      output.
  """

  class _Interrupter(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      fc_id = f'fc-{self.name}'
      if ctx.resume_inputs and fc_id in ctx.resume_inputs:
        yield f'{self.name}: {ctx.resume_inputs[fc_id]["value"]}'
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      r1 = await ctx.run_node(_Interrupter(name='a'), node_input=node_input)
      r2 = await ctx.run_node(_Interrupter(name='b'), node_input=node_input)
      yield f'{r1} + {r2}'

  wf = Workflow(name='wf', edges=[(START, _Parent(name='parent'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: a interrupts, parent never reaches b
  events1 = await _run(runner, ss, session, 'go')
  ids1 = _interrupt_ids(events1)
  assert 'fc-a' in ids1
  assert 'fc-b' not in ids1

  # Run 2: resume a → a completes, parent reaches b → b interrupts
  events2 = await _resume(runner, ss, session, 'fc-a', 'done-a')
  ids2 = _interrupt_ids(events2)
  assert 'fc-b' in ids2

  # Run 3: resume b → b completes, parent yields combined output
  events3 = await _resume(runner, ss, session, 'fc-b', 'done-b')
  outputs = _outputs(events3)
  assert 'a: done-a + b: done-b' in outputs


@pytest.mark.asyncio
async def test_dynamic_node_execution_id_reused_on_resume():
  """Resumed dynamic child reuses execution_id from original run.

  Setup: Parent calls ctx.run_node(Interrupter). Interrupter
    interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert: The resumed Interrupter's output event has the same
    execution_id as the original interrupt event.
  """

  class _Interrupter(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield 'resumed'
        return
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

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Interrupter(name='interrupter'))
      yield result

  wf = Workflow(name='wf', edges=[(START, _Parent(name='parent'))])
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts
  events1 = await _run(runner, ss, session, 'go')
  interrupt_event = [e for e in events1 if e.long_running_tool_ids][0]
  original_exec_id = interrupt_event.node_info.execution_id

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'ok')
  resumed_output_events = [
      e
      for e in events2
      if e.output == 'resumed' and 'interrupter' in (e.node_info.path or '')
  ]
  assert len(resumed_output_events) == 1
  assert resumed_output_events[0].node_info.execution_id == original_exec_id


# =========================================================================
# Nested workflow + dynamic node combinations
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_nested_static_workflow_with_dynamic_interrupt():
  """Static sub-workflow's node schedules a dynamic node that interrupts.

  Setup: outer_wf → inner_wf (static). Inside inner_wf, a static
    parent node calls ctx.run_node(Approver). Approver interrupts.
  Action: Send FR to resume.
  Assert: Approver resumes, parent completes, inner_wf completes,
    outer_wf completes.
  """

  class _Approver(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]["value"]}'
        return
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

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Approver(name='approver'))
      yield f'parent: {result}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, _Parent(name='parent'))],
  )
  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, inner_wf)],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: dynamic approver interrupts inside inner_wf
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'yes')
  outputs = _outputs(events2)
  assert 'parent: approved: yes' in outputs


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_workflow_with_static_interrupt():
  """Dynamic child is a Workflow whose static node interrupts.

  Setup: Parent calls ctx.run_node(inner_wf). inner_wf has a
    static Interrupter node that interrupts with fc-1.
  Action: Send FR to resume.
  Assert: Interrupter resumes, inner_wf completes, parent
    receives inner_wf's output.
  """

  class _Interrupter(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'done: {ctx.resume_inputs["fc-1"]["value"]}'
        return
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

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, _Interrupter(name='step'))],
  )

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(inner_wf)
      yield f'parent: {result}'

  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, _Parent(name='parent'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: inner_wf's static node interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'ok')
  outputs = _outputs(events2)
  assert 'parent: done: ok' in outputs


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_workflow_with_nested_dynamic_interrupt():
  """Dynamic Workflow's inner node schedules another dynamic node.

  Setup: Parent calls ctx.run_node(inner_wf). inner_wf has a
    static Orchestrator node that calls ctx.run_node(Approver).
    Approver interrupts with fc-1.
  Action: Send FR to resume.
  Assert: Approver resumes → Orchestrator completes → inner_wf
    completes → Parent receives output. Three levels of dynamic
    nesting resolved correctly.
  """

  class _Approver(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]["value"]}'
        return
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

  class _Orchestrator(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Approver(name='approver'))
      yield f'orch: {result}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[(START, _Orchestrator(name='orch'))],
  )

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(inner_wf)
      yield f'parent: {result}'

  outer_wf = Workflow(
      name='outer_wf',
      edges=[(START, _Parent(name='parent'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=outer_wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: deeply nested approver interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'granted')
  outputs = _outputs(events2)
  assert 'parent: orch: approved: granted' in outputs


# =========================================================================
# Scoping: parallel parents with same-named dynamic children
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_parallel_parents_same_named_dynamic_children():
  """Two static parents schedule dynamic children with the same name.

  Setup: outer_wf fans out to parent_a and parent_b (parallel).
    Both call ctx.run_node(Child(name='child')). parent_a's child
    completes, parent_b's child interrupts.
  Action: Send FR to resume parent_b's child.
  Assert:
    - parent_a's child and parent_b's child are distinct (scoped
      by parent_path: wf/parent_a/child vs wf/parent_b/child).
    - On resume, parent_a's child returns cached output (dedup),
      parent_b's child resumes with FR.
    - Both parents complete.
  """

  class _Child(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-b' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-b"]["value"]}'
        return
      if node_input == 'interrupt':
        yield Event(
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name='tool', args={}, id='fc-b'
                        )
                    )
                ]
            ),
            long_running_tool_ids={'fc-b'},
        )
      else:
        yield f'child: {node_input}'

  class _ParentA(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'), node_input='complete')
      yield f'a: {result}'

  class _ParentB(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'), node_input='interrupt')
      yield f'b: {result}'

  class _Join(BaseNode):

    async def _run_impl(self, *, ctx, node_input):
      yield f'joined: {node_input}'

  from google.adk.workflow._join_node import JoinNode

  join = JoinNode(name='join')
  wf = Workflow(
      name='wf',
      edges=[
          (START, _ParentA(name='parent_a'), join),
          (START, _ParentB(name='parent_b'), join),
      ],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: parent_a completes, parent_b's child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-b' in _interrupt_ids(events1)
  # parent_a should have completed
  a_outputs = [
      e.output for e in events1 if e.output and 'a: child' in str(e.output)
  ]
  assert len(a_outputs) == 1

  # Run 2: resume parent_b's child
  events2 = await _resume(runner, ss, session, 'fc-b', 'done')
  outputs = _outputs(events2)
  # Both parents completed, join has both results
  assert any('b: resumed: done' in str(o) for o in outputs)


# =========================================================================
# use_as_output + interrupt
# =========================================================================


@pytest.mark.asyncio
async def test_dynamic_node_use_as_output_with_interrupt():
  """Dynamic child with use_as_output=True interrupts then resumes.

  Setup: Parent calls ctx.run_node(child, use_as_output=True).
    Child interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert:
    - On resume, child resumes and its output becomes the parent's
      output (use_as_output delegation).
    - The parent's own output event is suppressed.
  """

  class _Child(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'child: {ctx.resume_inputs["fc-1"]["value"]}'
        return
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

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'), use_as_output=True)
      ctx.output = result
      yield  # keep as async generator

  wf = Workflow(
      name='wf',
      edges=[(START, _Parent(name='parent'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', 'approved')
  outputs = _outputs(events2)
  assert 'child: approved' in outputs
  # Parent's output should be suppressed (use_as_output)
  parent_outputs = [
      e.output
      for e in events2
      if e.node_info.path == 'wf/parent' and e.output is not None
  ]
  assert len(parent_outputs) == 0


# =========================================================================
# rerun_on_resume=False for dynamic node
# =========================================================================


@pytest.mark.asyncio
@pytest.mark.xfail(reason='Old Workflow does not support this')
async def test_dynamic_node_rerun_on_resume_false():
  """Dynamic child with rerun_on_resume=False auto-completes on resume.

  Setup: Parent calls ctx.run_node(child). Child has
    rerun_on_resume=False and interrupts with fc-1.
  Action: Send FR for fc-1.
  Assert:
    - Child does NOT re-execute _run_impl.
    - Child auto-completes with the FR response as output.
    - Parent receives the auto-completed output.
  """
  run_count = [0]

  class _Child(BaseNode):
    rerun_on_resume: bool = False

    async def _run_impl(self, *, ctx, node_input):
      run_count[0] += 1
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

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
      result = await ctx.run_node(_Child(name='child'))
      yield f'parent: {result}'

  wf = Workflow(
      name='wf',
      edges=[(START, _Parent(name='parent'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', agent=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: child interrupts
  events1 = await _run(runner, ss, session, 'go')
  assert 'fc-1' in _interrupt_ids(events1)
  assert run_count[0] == 1

  # Run 2: resume
  events2 = await _resume(runner, ss, session, 'fc-1', {'answer': 42})

  # Child should NOT have re-executed (run_count stays 1).
  assert run_count[0] == 1
  # Parent receives the FR response as child's output.
  outputs = _outputs(events2)
  assert any('answer' in str(o) for o in outputs)
