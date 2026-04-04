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

"""Tests for ctx.run_node(use_as_output=True) dynamic terminal paths."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import ConfigDict
import pytest
from typing_extensions import override

from .. import testing_utils


def _make_app(name: str, agent: Workflow, resumable: bool) -> App:
  return App(
      name=name,
      root_agent=agent,
      resumability_config=(
          ResumabilityConfig(is_resumable=True) if resumable else None
      ),
  )


def _get_outputs(events: list[Event]) -> list[Any]:
  """Extracts output values from events, skipping non-output events."""
  return [
      e.output for e in events if isinstance(e, Event) and e.output is not None
  ]


# ---------------------------------------------------------------------------
# FunctionNode delegates to FunctionNode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_function_to_function(
    request: pytest.FixtureRequest, resumable: bool
):
  """Node A delegates output to dynamic child B via use_as_output=True.

  Only B's output event should appear; A should not emit a duplicate.
  """

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Only one output event (from B). A's output is suppressed.
  outputs = _get_outputs(events)
  assert outputs == ['from_b']


# ---------------------------------------------------------------------------
# FunctionNode delegates to Workflow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_function_to_workflow(
    request: pytest.FixtureRequest, resumable: bool
):
  """Node A delegates output to a Workflow B via use_as_output=True.

  The Workflow's terminal node output should be used as A's output.
  """

  def step_1() -> str:
    return 'step_1_done'

  def step_2(node_input: str) -> str:
    return f'final:{node_input}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, step_1),
          (step_1, step_2),
      ],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # step_2 is the terminal node; its output is func_a's output.
  # step_1's output is intermediate (not terminal), so only step_2 appears.
  outputs = _get_outputs(events)
  assert 'final:step_1_done' in outputs
  # func_a should NOT have a duplicate output event.
  assert outputs.count('final:step_1_done') == 1


# ---------------------------------------------------------------------------
# Custom BaseNode delegates to FunctionNode
# ---------------------------------------------------------------------------


class _DelegatingNode(BaseNode):
  """A custom BaseNode that delegates output via use_as_output."""

  model_config = ConfigDict(arbitrary_types_allowed=True)
  delegate: BaseNode

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    await ctx.run_node(self.delegate, use_as_output=True)
    # Intentionally yield nothing — output comes from delegate.
    return
    yield  # Make this an async generator


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_custom_node_to_function(
    request: pytest.FixtureRequest, resumable: bool
):
  """Custom BaseNode delegates output to dynamic child via use_as_output."""

  def func_b() -> str:
    return 'delegated_output'

  node_b = FunctionNode(func=func_b)
  node_a = _DelegatingNode(
      name='delegator', delegate=node_b, rerun_on_resume=True
  )

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert outputs == ['delegated_output']


# ---------------------------------------------------------------------------
# Multiple use_as_output=True calls (fan-out delegation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_last_wins(
    request: pytest.FixtureRequest, resumable: bool
):
  """When use_as_output=True is called multiple times, last one wins.

  The parent's output is resolved from the last delegate. Earlier
  delegates' output events still appear in the event stream.
  """

  def func_b() -> str:
    return 'from_b'

  def func_c() -> str:
    return 'from_c'

  node_b = FunctionNode(func=func_b)
  node_c = FunctionNode(func=func_c)

  async def func_a(ctx: Context):
    b_result = await ctx.run_node(node_b, use_as_output=True)
    c_result = await ctx.run_node(node_c, use_as_output=True)
    return f'{b_result}+{c_result}'

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  def func_d(node_input: str) -> str:
    return f'received:{node_input}'

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(
      name=wf_name,
      edges=[(START, node_a), (node_a, func_d)],
  )
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  # Last use_as_output wins: func_d receives func_c's output.
  assert 'received:from_c' in outputs


# ---------------------------------------------------------------------------
# Nested dynamic delegation (A → B → C, all use_as_output=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_nested_delegation(
    request: pytest.FixtureRequest, resumable: bool
):
  """Chained delegation: A delegates to B, B delegates to C.

  Only C's output should appear; both A's and B's outputs are suppressed.
  """

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert outputs == ['from_c']


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_nested_delegation_with_downstream(
    request: pytest.FixtureRequest, resumable: bool
):
  """Chained delegation with a downstream node that consumes the output.

  A delegates to B, B delegates to C. D is downstream of A and should
  receive C's output as its node_input.
  """

  def func_c() -> str:
    return 'from_c'

  node_c = FunctionNode(func=func_c)

  async def func_b(ctx: Context) -> str:
    return await ctx.run_node(node_c, use_as_output=True)

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  def func_d(node_input: str) -> str:
    return f'received:{node_input}'

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(
      name=wf_name,
      edges=[(START, node_a), (node_a, func_d)],
  )
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  outputs = _get_outputs(events)
  assert 'received:from_c' in outputs


# ---------------------------------------------------------------------------
# use_as_output=False (default) still emits both events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_without_use_as_output_emits_both(
    request: pytest.FixtureRequest, resumable: bool
):
  """Without use_as_output, both parent and child emit output events."""

  def func_b() -> str:
    return 'from_b'

  node_b = FunctionNode(func=func_b)

  async def func_a(ctx: Context) -> str:
    return await ctx.run_node(node_b)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Both B and A emit output events (no dedup).
  outputs = _get_outputs(events)
  assert outputs == ['from_b', 'from_b']


# ---------------------------------------------------------------------------
# HITL with use_as_output (interrupt + resume)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_use_as_output_with_hitl_resume(
    request: pytest.FixtureRequest, resumable: bool
):
  """Dynamic child with HITL pauses the workflow; on resume, output dedup works.

  Flow:
    1. func_a calls ctx.run_node(hitl_node, use_as_output=True)
    2. hitl_node yields RequestInput → workflow pauses
    3. User provides response → workflow resumes
    4. func_a re-runs, hitl_node returns cached result
    5. Only hitl_node's output event appears (func_a's is suppressed)
  """

  async def hitl_node(ctx: Context):
    if resume := ctx.resume_inputs.get('req1'):
      yield Event(output=f"user said: {resume['text']}")
      return
    yield RequestInput(
        interrupt_id='req1',
        message='what is your name?',
        response_schema={'type': 'string'},
    )

  hitl = FunctionNode(func=hitl_node, rerun_on_resume=True)

  async def func_a(ctx: Context):
    return await ctx.run_node(hitl, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should pause at hitl_node.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  outputs1 = _get_outputs(events1)
  assert outputs1 == []  # No output yet — paused.

  # Resume with user response.
  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req1',
              name='user_input',
              response={'text': 'Alice'},
          )
      )
  )
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  outputs2 = _get_outputs(events2)
  assert outputs2 == ['user said: Alice']


# ---------------------------------------------------------------------------
# use_as_output with Workflow containing HITL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_use_as_output_workflow_with_hitl(
    request: pytest.FixtureRequest,
):
  """Dynamic Workflow child with HITL node pauses and resumes correctly.

  Requires resumable=True because the inner Workflow's internal node state
  (hitl_step in WAITING) must be persisted across invocations. Without
  resumability, re-spawning the dynamic Workflow starts its graph fresh.

  Flow:
    1. func_a calls ctx.run_node(inner_wf, use_as_output=True)
    2. inner_wf's hitl_step yields RequestInput → workflow pauses
    3. User responds → workflow resumes
    4. inner_wf completes, func_a re-runs with cached result
    5. Only inner_wf's terminal node output appears
  """
  resumable = True

  async def hitl_step():
    yield RequestInput(
        interrupt_id='wf_req1',
        message='enter value',
        response_schema={'type': 'string'},
    )

  def final_step(node_input: Any) -> str:
    text = node_input['text'] if isinstance(node_input, dict) else node_input
    return f'final:{text}'

  inner_wf = Workflow(
      name='inner_wf',
      edges=[
          (START, hitl_step),
          (hitl_step, final_step),
      ],
  )

  async def func_a(ctx: Context):
    return await ctx.run_node(inner_wf, use_as_output=True)

  node_a = FunctionNode(func=func_a, rerun_on_resume=True)

  wf_name = request.node.name.replace('[', '_').replace(']', '')
  agent = Workflow(name=wf_name, edges=[(START, node_a)])
  app = _make_app(wf_name, agent, resumable)
  runner = testing_utils.InMemoryRunner(app=app)

  # Run 1: Should pause at hitl_step.
  events1 = await runner.run_async(testing_utils.get_user_content('start'))
  outputs1 = _get_outputs(events1)
  assert outputs1 == []

  # Resume with user response.
  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='wf_req1',
              name='user_input',
              response={'text': 'hello'},
          )
      )
  )
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  outputs2 = _get_outputs(events2)
  assert 'final:hello' in outputs2
  assert outputs2.count('final:hello') == 1
