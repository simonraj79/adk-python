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

"""Tests for Runner(node=...).

Verifies that Runner can execute standalone BaseNode instances,
persist events to session, handle resume (HITL), and yield events correctly.
"""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow_class import Workflow
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    text = node_input.parts[0].text if node_input else 'empty'
    yield f'Echo: {text}'


async def _run_node(node, message='hello'):
  """Run a BaseNode via Runner(node=...) and return (events, ss, session)."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _make_interrupt_event(fc_name='get_input', fc_id='fc-1'):
  """Create an interrupt Event with a long-running function call."""
  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=fc_name, args={}, id=fc_id
                  )
              )
          ]
      ),
      long_running_tool_ids={fc_id},
  )


def _make_resume_message(fc_name='get_input', fc_id='fc-1', response=None):
  """Create a user message with a function response for resuming."""
  return types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=fc_name,
                  id=fc_id,
                  response=response or {},
              )
          )
      ],
      role='user',
  )


async def _run_two_turns(node, msg1_text, resume_msg):
  """Run a node for two turns: initial message then resume."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg1 = types.Content(parts=[types.Part(text=msg1_text)], role='user')
  events1 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  events2 = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=resume_msg
  ):
    events2.append(event)

  return events1, events2, runner, ss, session


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_node_output():
  """Runner yields output from a simple BaseNode."""
  events, _, _ = await _run_node(_EchoNode(name='echo'), message='hi')

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Echo: hi']


@pytest.mark.asyncio
async def test_intermediate_events_yielded():
  """Runner yields intermediate events (e.g. state), not just output."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'step': 'processing'})
      yield 'final_result'

  events, _, _ = await _run_node(_Node(name='steps'))

  state_events = [e for e in events if e.actions.state_delta]
  assert len(state_events) >= 1
  assert [e.output for e in events if e.output is not None] == ['final_result']


@pytest.mark.asyncio
async def test_event_author_defaults_to_node_name():
  """Events are attributed to the node's name by default."""
  events, _, _ = await _run_node(_EchoNode(name='my_node'), message='hi')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'


@pytest.mark.asyncio
async def test_node_error_propagates():
  """A node that raises propagates the exception to the caller."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      raise RuntimeError('node failure')
      yield  # pylint: disable=unreachable

  with pytest.raises(RuntimeError, match='node failure'):
    await _run_node(_Node(name='error'))


@pytest.mark.asyncio
async def test_node_yielding_none_produces_no_output():
  """A node that yields None produces no output event."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield None

  events, _, _ = await _run_node(_Node(name='nil'))

  assert [e.output for e in events if e.output is not None] == []


@pytest.mark.asyncio
async def test_workflow_node_output():
  """Runner drives a Workflow and yields its terminal output."""

  def upper(node_input: str) -> str:
    return node_input.upper()

  wf = Workflow(name='wf', edges=[(START, upper)])
  events, _, _ = await _run_node(wf, message='hi')

  outputs = [e.output for e in events if e.output is not None]
  assert 'HI' in outputs


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_persisted_to_session():
  """Non-partial events are persisted to the session."""
  _, ss, session = await _run_node(_EchoNode(name='echo'), message='hi')

  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  session_outputs = [e.output for e in updated.events if e.output is not None]
  assert 'Echo: hi' in session_outputs


@pytest.mark.asyncio
async def test_multiple_invocations_accumulate_events():
  """Each invocation appends events; session accumulates across runs."""
  node = _EchoNode(name='echo')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  for msg_text in ['first', 'second', 'third']:
    async for _ in runner.run_async(
        user_id='u',
        session_id=session.id,
        new_message=types.Content(
            parts=[types.Part(text=msg_text)], role='user'
        ),
    ):
      pass

  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  outputs = [e.output for e in updated.events if e.output is not None]
  assert outputs == ['Echo: first', 'Echo: second', 'Echo: third']


# ---------------------------------------------------------------------------
# yield_user_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_yield_user_message_true():
  """When yield_user_message=True, user event is yielded before node events."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='hi')], role='user')

  events: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=msg,
      yield_user_message=True,
  ):
    events.append(event)

  user_events = [e for e in events if e.author == 'user']
  assert len(user_events) == 1
  assert user_events[0].content.parts[0].text == 'hi'
  assert events[0].author == 'user'


@pytest.mark.asyncio
async def test_yield_user_message_false_by_default():
  """By default, user event is not yielded to the caller."""
  events, _, _ = await _run_node(_EchoNode(name='echo'), message='hi')

  user_events = [e for e in events if e.author == 'user']
  assert user_events == []


# ---------------------------------------------------------------------------
# Resume (HITL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_node_resume():
  """A standalone node resumes with resume_inputs from function response."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'result: {ctx.resume_inputs["fc-1"]["value"]}'
        return
      yield _make_interrupt_event()

  events1, events2, _, _, _ = await _run_two_turns(
      _Node(name='standalone'),
      'go',
      _make_resume_message(response={'value': 42}),
  )

  assert any(e.long_running_tool_ids for e in events1)
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'result: 42' in outputs


@pytest.mark.asyncio
async def test_resume_preserves_original_user_content():
  """On resume, Runner passes the original text as node_input, not the FR."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        text = (
            node_input.parts[0].text
            if node_input and hasattr(node_input, 'parts')
            else str(node_input)
        )
        yield f'original:{text}'
        return
      yield _make_interrupt_event(fc_name='tool')

  events1, events2, _, _, _ = await _run_two_turns(
      _Node(name='node'),
      'my original input',
      _make_resume_message(fc_name='tool', response={'v': 1}),
  )

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'original:my original input' in outputs


@pytest.mark.asyncio
async def test_plain_text_does_not_trigger_resume():
  """Sending plain text (no FR) starts fresh, does not enter resume path."""
  node = _EchoNode(name='echo')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='first')], role='user'),
  ):
    pass

  # Run 2: plain text — should start fresh
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='second')], role='user'),
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert outputs == ['Echo: second']


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_raises_on_unmatched_fr():
  """Runner raises when function response has no matching FC in session."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = _make_resume_message(fc_name='unknown', fc_id='no-such-fc')

  with pytest.raises(ValueError, match='Function call not found'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_resume_raises_on_multi_invocation_fr():
  """Runner raises when FRs resolve to different invocations."""
  call_count = [0]

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_count[0] += 1
      fc_id = f'fc-{call_count[0]}'
      yield _make_interrupt_event(fc_name='tool', fc_id=fc_id)

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptNode(name='ask'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts with fc-1
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='go')], role='user'),
  ):
    pass

  # Run 2: interrupts with fc-2 (different invocation)
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text='go again')], role='user'
      ),
  ):
    pass

  # Run 3: send FRs for both fc-1 and fc-2 (different invocations)
  msg3 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'r': 1}
              )
          ),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-2', response={'r': 2}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='resolve to multiple invocations'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg3
    ):
      pass


@pytest.mark.asyncio
async def test_mixed_fr_and_text_raises():
  """Message with both function responses and text is rejected."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(
      parts=[
          types.Part(text='some text'),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'v': 1}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='cannot contain both'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


# ---------------------------------------------------------------------------
# Default scheduler & ctx.create_task cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_node_works_without_workflow():
  """ctx.run_node() works in a standalone BaseNode (default scheduler)."""

  class _ChildNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield f'child got: {node_input}'

  class _ParentNode(BaseNode):

    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      result = await ctx.run_node(_ChildNode(name='child'), 'hello')
      yield f'parent got: {result}'

  events, _, _ = await _run_node(_ParentNode(name='parent'), message='go')

  outputs = [e.output for e in events if e.output is not None]
  assert 'parent got: child got: hello' in outputs


@pytest.mark.asyncio
async def test_run_node_use_as_output_attributes_child_output_to_parent():
  """Child output with use_as_output=True is attributed to the parent node."""

  class _ChildNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'child result'

  class _ParentNode(BaseNode):

    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      result = await ctx.run_node(
          _ChildNode(name='child'), 'hello', use_as_output=True
      )
      yield f'parent got: {result}'

  events, _, _ = await _run_node(_ParentNode(name='parent'), message='go')

  # The child's output event should list the parent's path in output_for.
  # With use_as_output=True, the parent's own yield is suppressed —
  # only the child's output (attributed to the parent) is emitted.
  child_output = next(e for e in events if e.output == 'child result')
  assert 'parent/child' in child_output.node_info.path
  assert any(
      'parent' in p and 'child' not in p
      for p in child_output.node_info.output_for
  )


# ---------------------------------------------------------------------------
# DefaultNodeScheduler — dynamic child resume via ctx.run_node()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_node_child_resume_via_default_scheduler():
  """Completed children are cached on resume; interrupted child re-runs."""

  class _ChildA(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'child_a_output'

  class _ChildB(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'resumed: {ctx.resume_inputs["fc-1"]["answer"]}'
        return
      yield _make_interrupt_event(fc_name='ask', fc_id='fc-1')

  class _ParentNode(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      a = await ctx.run_node(_ChildA(name='a'), 'input_a')
      b = await ctx.run_node(_ChildB(name='b'), 'input_b')
      yield f'{a} + {b}'

  events1, events2, _, _, _ = await _run_two_turns(
      _ParentNode(name='parent'),
      'go',
      _make_resume_message(
          fc_name='ask', fc_id='fc-1', response={'answer': 42}
      ),
  )

  assert any(e.long_running_tool_ids for e in events1)
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'child_a_output + resumed: 42' in outputs


@pytest.mark.asyncio
async def test_run_node_default_scheduler_caches_by_call_count():
  """Only interrupted children re-run; completed children are skipped."""

  call_counts = {'a': 0, 'b': 0, 'c': 0}

  class _CountingChild(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_counts[self.name] += 1
      if self.name == 'c':
        if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
          yield 'c_resumed'
          return
        yield _make_interrupt_event(fc_name='tool', fc_id='fc-1')
        return
      yield f'{self.name}_out'

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      a = await ctx.run_node(_CountingChild(name='a'), 'x')
      b = await ctx.run_node(_CountingChild(name='b'), 'y')
      c = await ctx.run_node(_CountingChild(name='c'), 'z')
      yield f'{a},{b},{c}'

  events1, events2, _, _, _ = await _run_two_turns(
      _Parent(name='p'),
      'go',
      _make_resume_message(fc_name='tool', fc_id='fc-1', response={}),
  )

  assert any(e.long_running_tool_ids for e in events1)
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'a_out,b_out,c_resumed' in outputs
  assert call_counts == {'a': 1, 'b': 1, 'c': 2}


@pytest.mark.asyncio
async def test_run_node_use_as_output_with_resume():
  """use_as_output child resumes correctly; child output is attributed to parent."""

  class _Child(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]["ok"]}'
        return
      yield _make_interrupt_event(fc_name='approve', fc_id='fc-1')

  class _Parent(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      result = await ctx.run_node(
          _Child(name='child'), 'data', use_as_output=True
      )
      yield f'parent saw: {result}'

  events1, events2, _, _, _ = await _run_two_turns(
      _Parent(name='parent'),
      'go',
      _make_resume_message(
          fc_name='approve', fc_id='fc-1', response={'ok': True}
      ),
  )

  assert any(e.long_running_tool_ids for e in events1)
  outputs = [e.output for e in events2 if e.output is not None]
  assert any('approved: True' in o for o in outputs)


@pytest.mark.asyncio
async def test_run_node_nested_ctx_run_node_resume():
  """Nested ctx.run_node(): outer → middle → inner; inner interrupts and resumes."""

  call_counts = {'outer': 0, 'middle': 0, 'inner': 0}

  class _Inner(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_counts['inner'] += 1
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'inner_resumed:{ctx.resume_inputs["fc-1"]["v"]}'
        return
      yield _make_interrupt_event(fc_name='ask', fc_id='fc-1')

  class _Middle(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_counts['middle'] += 1
      inner_out = await ctx.run_node(_Inner(name='inner'), 'go')
      yield f'middle({inner_out})'

  class _Outer(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_counts['outer'] += 1
      mid_out = await ctx.run_node(_Middle(name='middle'), 'start')
      yield f'outer({mid_out})'

  events1, events2, _, _, _ = await _run_two_turns(
      _Outer(name='top'),
      'go',
      _make_resume_message(fc_name='ask', fc_id='fc-1', response={'v': 99}),
  )

  # Turn 1: inner interrupts, propagated through middle and outer.
  assert any(e.long_running_tool_ids for e in events1)

  # Turn 2: inner resumes, middle and outer produce final output.
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'outer(middle(inner_resumed:99))' in outputs

  # Outer and middle re-run on resume; inner runs twice (interrupt + resume).
  assert call_counts == {'outer': 2, 'middle': 2, 'inner': 2}


@pytest.mark.asyncio
async def test_run_node_use_as_output_nested_delegation():
  """Nested use_as_output delegates all the way up with run_ids."""

  class _Inner(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'inner_val'

  class _Middle(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      await ctx.run_node(_Inner(name='inner'), 'go', use_as_output=True)
      if False:
        yield

  class _Outer(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      await ctx.run_node(_Middle(name='middle'), 'start', use_as_output=True)
      if False:
        yield

  # When
  events, _, _ = await _run_node(_Outer(name='outer'), message='go')

  # Then
  inner_output = next(e for e in events if e.output == 'inner_val')
  output_for = inner_output.node_info.output_for
  paths = [t[0] for t in output_for]

  assert len(output_for) == 3
  assert any('middle' in p for p in paths)
  assert any('outer' in p for p in paths)
  assert any('inner' in p for p in paths)
  for _, run_id in output_for:
    assert run_id != ''
