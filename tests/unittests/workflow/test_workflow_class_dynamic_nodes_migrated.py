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

"""Tests for dynamic node execution (migrated from test_dynamic_nodes.py).

TODO: Merge into test_workflow_class_dynamic_nodes.py.
"""

import asyncio
import time

from google.adk.agents.context import Context
from google.adk.apps import app
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._workflow_class import Workflow
from google.adk.workflow.utils._node_path_utils import is_direct_child
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
import pytest

from . import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import simplify_event_with_node
from .workflow_testing_utils import simplify_events_with_node


@pytest.mark.asyncio
async def test_dynamically_run_nodes_return_outputs_and_emit_events(
    request: pytest.FixtureRequest,
):
  """Dynamic child outputs are returned to caller and emitted as events."""

  def func_a() -> str:
    return 'A'

  def func_b() -> str:
    return 'B'

  node_a = FunctionNode(func=func_a)
  node_b = FunctionNode(func=func_b)

  async def node_c(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C calls {output_a} & {output_b}'

  node_c = FunctionNode(func=node_c, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_dynamic_simple',
      edges=[
          (START, node_c),
      ],
  )
  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'func_a', 'output': 'A'},
      ),
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'func_b', 'output': 'B'},
      ),
      (
          'test_workflow_agent_dynamic_simple',
          {'node_name': 'node_c', 'output': 'C calls A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_with_custom_name(
    request: pytest.FixtureRequest,
):
  """ctx.run_node(name='custom') uses the custom name in event paths."""

  def func_a() -> str:
    return 'A'

  node_a = FunctionNode(func=func_a)

  async def func_b(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a, name='custom_node_a')
    return f'B calls {output_a}'

  node_b = FunctionNode(func=func_b, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_custom_name',
      edges=[
          (START, node_b),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  simplified_events = simplify_events_with_node(events)
  assert simplified_events == [
      (
          'test_agent_dynamic_custom_name',
          {'node_name': 'custom_node_a', 'output': 'A'},
      ),
      (
          'test_agent_dynamic_custom_name',
          {'node_name': 'func_b', 'output': 'B calls A'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_hitl_no_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """Dynamic child with rerun_on_resume=False auto-completes with FR response on resume.

  Setup: simple_caller → node_hitl(rerun_on_resume=False, RequestInput).
  Act: run 1 pauses, run 2 resumes with FR.
  Assert: node_hitl returns FR response without re-executing.
  """

  async def node_hitl():
    yield RequestInput(
        interrupt_id='req1',
        message='request 1',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def simple_caller(ctx: Context):
    result = await ctx.run_node(node_hitl)
    return {'child_result': result, 'status': 'parent done'}

  simple_caller = FunctionNode(func=simple_caller, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_hitl',
      edges=[(START, simple_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req1',
              name='user_input',
              response={'text': 'Hello'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # node_hitl auto-completed (rerun_on_resume=False) — no output event
  # emitted for it. Only simple_caller's output appears.
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_dynamic_hitl',
          {
              'node_name': 'simple_caller',
              'output': {
                  'child_result': {'req1': {'text': 'Hello'}},
                  'status': 'parent done',
              },
          },
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_hitl_with_rerun_on_resume(
    request: pytest.FixtureRequest,
):
  """Dynamic child with rerun_on_resume=True re-executes with resume_inputs on resume.

  Setup: simple_caller → node_hitl(rerun_on_resume=True, RequestInput).
  Act: run 1 pauses, run 2 resumes with FR.
  Assert: node_hitl re-runs, reads resume_inputs, yields FR text.
  """

  async def node_hitl(ctx: Context):
    if resume_input := ctx.resume_inputs.get('req1'):
      yield resume_input['text']
      return

    yield RequestInput(
        interrupt_id='req1',
        message='request 1',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl, rerun_on_resume=True)

  async def simple_caller(ctx: Context):
    result = await ctx.run_node(node_hitl)
    return {'child_result': result, 'status': 'parent done'}

  simple_caller = FunctionNode(func=simple_caller, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_dynamic_hitl_rerun',
      edges=[(START, simple_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req1',
              name='user_input',
              response={'text': 'Hello Rerun'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # Check result
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_dynamic_hitl_rerun',
          {'node_name': 'node_hitl', 'output': 'Hello Rerun'},
      ),
      (
          'test_agent_dynamic_hitl_rerun',
          {
              'node_name': 'simple_caller',
              'output': {
                  'child_result': 'Hello Rerun',
                  'status': 'parent done',
              },
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_dynamic_node_hitl(request: pytest.FixtureRequest):
  """Nested dynamic chain (top → middle → leaf) resumes through all levels.

  Setup: top_node → middle_node → leaf_node_hitl(RequestInput).
  Act: run 1 pauses at leaf, run 2 resumes with FR.
  Assert: all three nodes complete, propagating FR response as output.
  """

  async def leaf_node_hitl() -> str:
    yield RequestInput(
        interrupt_id='req2',
        message='request 2',
        response_schema={'type': 'string'},
    )

  async def middle_node(ctx: Context) -> str:
    return await ctx.run_node(leaf_node_hitl)

  middle_node = FunctionNode(func=middle_node, rerun_on_resume=True)
  leaf_node_hitl = FunctionNode(func=leaf_node_hitl)

  async def top_node(ctx: Context) -> str:
    return await ctx.run_node(middle_node)

  top_node = FunctionNode(func=top_node, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_nested_hitl',
      edges=[(START, top_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause at leaf_node_hitl
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req2',
              name='user_input',
              response={'text': 'World'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  # leaf_node_hitl auto-completed (rerun_on_resume=False) — no output
  # event. middle_node and top_node re-run and emit output.
  assert simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_agent_nested_hitl',
          {'node_name': 'middle_node', 'output': {'req2': {'text': 'World'}}},
      ),
      (
          'test_agent_nested_hitl',
          {'node_name': 'top_node', 'output': {'req2': {'text': 'World'}}},
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        'New Workflow has no agent_state events for dynamic node name mapping'
    )
)
async def test_dynamic_node_parallel_execution(request: pytest.FixtureRequest):
  """Three parallel ctx.run_node calls via asyncio.gather return ordered results."""

  def echo_node(node_input: str) -> str:
    return node_input

  echo_node = FunctionNode(func=echo_node)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(echo_node, node_input=f'call_{i}') for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='dynamic_parallel',
      edges=[
          (START, parent_node),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  simplified = simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  )

  # Sort child events for deterministic assertion.
  child_events = sorted(simplified[:-1], key=lambda x: x[1]['output'])
  assert child_events == [
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_0'},
      ),
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_1'},
      ),
      (
          'dynamic_parallel',
          {'node_name': 'echo_node', 'output': 'call_2'},
      ),
  ]
  # And then assert the parent event separately.
  assert simplified[-1] == (
      'dynamic_parallel',
      {'node_name': 'parent_node', 'output': ['call_0', 'call_1', 'call_2']},
  )


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        'New Workflow has no agent_state events for dynamic node name mapping'
    )
)
async def test_dynamic_node_parallel_mixed_hitl(request: pytest.FixtureRequest):
  """Parallel mix: two simple nodes complete, one HITL pauses then resumes.

  Setup: parent gathers simple_1, simple_2, node_hitl(RequestInput).
  Act: run 1 — simples complete, HITL pauses. Run 2 — resume HITL.
  Assert: run 1 emits simple outputs + RequestInput. Run 2 emits HITL
    output + parent aggregated list.
  """

  def simple_node(node_input: str) -> str:
    return f'simple_{node_input}'

  async def node_hitl() -> str:
    yield RequestInput(
        interrupt_id='req_hitl',
        message='request hitl',
        response_schema={'type': 'string'},
    )

  simple_node = FunctionNode(func=simple_node)
  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    t1 = ctx.run_node(simple_node, node_input='1')
    t2 = ctx.run_node(simple_node, node_input='2')
    t3 = ctx.run_node(node_hitl)
    return await asyncio.gather(t1, t2, t3)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='test_agent_parallel_mixed',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause. simple_nodes should run.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  simplified1 = simplify_events_with_node(
      events1, map_dynamic_node_to_the_source=True
  )

  assert simplified1 == [
      (
          'test_agent_parallel_mixed',
          {'node_name': 'simple_node', 'output': 'simple_1'},
      ),
      (
          'test_agent_parallel_mixed',
          {'node_name': 'simple_node', 'output': 'simple_2'},
      ),
      (
          'test_agent_parallel_mixed',
          types.Part(
              function_call=types.FunctionCall(
                  args={
                      'interrupt_id': 'req_hitl',
                      'message': 'request hitl',
                      'payload': None,
                      'response_schema': {'type': 'string'},
                  },
                  name=REQUEST_INPUT_FUNCTION_CALL_NAME,
              )
          ),
      ),
  ], 'Check that simple nodes ran in the first run.'

  invocation_id = events1[0].invocation_id
  resume_payload = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req_hitl',
              name='user_input',
              response={'text': 'HitlResponse'},
          )
      )
  )

  # Run 2: Resume. Simple nodes should NOT rerun.
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'test_agent_parallel_mixed',
          {'node_name': 'node_hitl', 'output': {'text': 'HitlResponse'}},
      ),
      (
          'test_agent_parallel_mixed',
          {
              'node_name': 'parent_node',
              'output': ['simple_1', 'simple_2', {'text': 'HitlResponse'}],
          },
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        'New Workflow has no agent_state events for dynamic node name mapping'
    )
)
async def test_dynamic_node_parallel_hitl_all_resume(
    request: pytest.FixtureRequest,
):
  """All parallel HITL nodes resumed at once complete together."""

  async def node_hitl(node_input: str):
    yield RequestInput(
        interrupt_id=f'req_{node_input}',
        message=f'request {node_input}',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(node_hitl, node_input=str(i)) for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='parallel_hitl_all',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)

  invocation_id = events1[0].invocation_id
  parts = []
  for i in range(3):
    parts.append(
        types.Part(
            function_response=types.FunctionResponse(
                id=f'req_{i}',
                name='user_input',
                response={'text': f'resp_{i}'},
            )
        )
    )
  resume_payload = types.Content(role='user', parts=parts)

  # Run 2: Resume all.
  events2 = await runner.run_async(
      new_message=resume_payload, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_0'}},
      ),
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_1'}},
      ),
      (
          'parallel_hitl_all',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_2'}},
      ),
      (
          'parallel_hitl_all',
          {
              'node_name': 'parent_node',
              'output': [
                  {'text': 'resp_0'},
                  {'text': 'resp_1'},
                  {'text': 'resp_2'},
              ],
          },
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        'New Workflow has no agent_state events for dynamic node name mapping'
    )
)
async def test_dynamic_node_parallel_hitl_partial_resume(
    request: pytest.FixtureRequest,
):
  """Parallel HITL nodes resumed one-at-a-time; parent waits for all.

  Act: run 1 — all 3 pause. Run 2 — resume req_0 only. Run 3 — resume
    req_1 and req_2.
  Assert: run 2 emits only req_0's output. Run 3 emits req_1, req_2,
    and parent aggregated list.
  """

  async def node_hitl(node_input: str):
    yield RequestInput(
        interrupt_id=f'req_{node_input}',
        message=f'request {node_input}',
        response_schema={'type': 'string'},
    )

  node_hitl = FunctionNode(func=node_hitl)

  async def parent_node(ctx: Context) -> list[str]:
    tasks = [ctx.run_node(node_hitl, node_input=str(i)) for i in range(3)]
    return await asyncio.gather(*tasks)

  parent_node = FunctionNode(func=parent_node, rerun_on_resume=True)

  agent = Workflow(
      name='parallel_hitl_partial',
      edges=[(START, parent_node)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  # Run 1: Should pause.
  user_event = testing_utils.get_user_content('start')
  events1 = await runner.run_async(user_event)
  invocation_id = events1[0].invocation_id

  # Resume 1: Respond to req_0 only.
  resume_payload1 = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id='req_0',
              name='user_input',
              response={'text': 'resp_0'},
          )
      )
  )

  events2 = await runner.run_async(
      new_message=resume_payload1, invocation_id=invocation_id
  )

  simplified2 = simplify_events_with_node(
      events2, map_dynamic_node_to_the_source=True
  )
  assert simplified2 == [
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_0'}},
      ),
  ], 'Check that parent is NOT finished (not in events).'

  # Resume 2: Respond to req_1 and req_2.
  parts = []
  for i in [1, 2]:
    parts.append(
        types.Part(
            function_response=types.FunctionResponse(
                id=f'req_{i}',
                name='user_input',
                response={'text': f'resp_{i}'},
            )
        )
    )
  resume_payload2 = types.Content(role='user', parts=parts)

  events3 = await runner.run_async(
      new_message=resume_payload2, invocation_id=invocation_id
  )

  simplified3 = simplify_events_with_node(
      events3, map_dynamic_node_to_the_source=True
  )

  # Sort the events for deterministic assertion since parallel nodes may
  # finish in any order. We separate the parent event (last one)
  # from the child events.
  parent_event = simplified3[-1]
  child_events = sorted(simplified3[:-1], key=lambda x: x[1]['output']['text'])

  assert child_events == [
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_1'}},
      ),
      (
          'parallel_hitl_partial',
          {'node_name': 'node_hitl', 'output': {'text': 'resp_2'}},
      ),
  ]
  assert parent_event == (
      'parallel_hitl_partial',
      {
          'node_name': 'parent_node',
          'output': [
              {'text': 'resp_0'},
              {'text': 'resp_1'},
              {'text': 'resp_2'},
          ],
      },
  )


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason='Uses create_parent_invocation_context which requires BaseAgent'
)
async def test_dynamic_node_with_multiple_events(
    request: pytest.FixtureRequest,
):  # pylint: disable=redefined-outer-name
  """Content events stream before blocking calls; each node yields one output.

  Setup: parent streams content, calls dynamic child (streams content
    then sleeps), then yields final output.
  Assert: both content events arrive before sleep starts.
  """
  sleep_started = False

  async def node_dynamic():
    nonlocal sleep_started

    yield Event(content=types.Content(parts=[types.Part(text='child-1')]))
    sleep_started = True
    time.sleep(0.5)
    yield 'child-done'

  node_d = FunctionNode(func=node_dynamic)

  async def node_parent(ctx: Context):
    yield Event(content=types.Content(parts=[types.Part(text='parent-1')]))
    output_d = await ctx.run_node(node_d)
    yield f'parent-2 with {output_d}'

  node_parent = FunctionNode(func=node_parent, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_dynamic_multi_events',
      edges=[
          (START, node_parent),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)

  events = []
  sleep_started_values = []
  async for e in agent.run_async(ctx):
    events.append(e)
    # Track sleep_started for content events (streaming)
    if isinstance(e, Event) and e.content:
      sleep_started_values.append(sleep_started)

  # Content events before sleep arrive before sleep starts.
  assert len(sleep_started_values) >= 2
  assert (
      sleep_started_values[0] is False
  ), 'parent-1 content event should arrive before sleep'
  assert (
      sleep_started_values[1] is False
  ), 'child-1 content event should arrive before sleep'

  # Filter to data-bearing events only (exclude workflow output events).
  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and is_direct_child(e.node_info.path, agent.name)
  ]

  assert len(data_events) == 2

  # D data event from node_dynamic
  assert data_events[0].output == 'child-done'
  assert data_events[0].node_name.startswith('node_dynamic_')

  # P2 data event from node_parent
  assert data_events[1].output == 'parent-2 with child-done'
  assert data_events[1].node_name == 'node_parent'


@pytest.mark.asyncio
async def test_node_like_simple(request: pytest.FixtureRequest):
  """Plain functions passed to ctx.run_node are auto-wrapped and executed."""

  def node_a() -> str:
    return 'A'

  def node_b() -> str:
    return 'B'

  async def node_c(ctx: Context) -> str:
    # Pass functions directly instead of FunctionNode objects
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C calls {output_a} & {output_b}'

  node_c = FunctionNode(func=node_c, rerun_on_resume=True)

  agent = Workflow(
      name='test_workflow_agent_node_like',
      edges=[
          (START, node_c),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_a', 'output': 'A'},
      ),
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_b', 'output': 'B'},
      ),
      (
          'test_workflow_agent_node_like',
          {'node_name': 'node_c', 'output': 'C calls A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_node_like_nested(request: pytest.FixtureRequest):
  """Nested ctx.run_node with plain functions propagates outputs through chain."""

  def node_a() -> str:
    return 'A'

  def node_b() -> str:
    return 'B'

  async def func_c(ctx: Context) -> str:
    output_a = await ctx.run_node(node_a)
    output_b = await ctx.run_node(node_b)
    return f'C -> {output_a} & {output_b}'

  node_c = FunctionNode(func=func_c, rerun_on_resume=True)

  async def func_d(ctx: Context) -> str:
    # Pass functions directly instead of FunctionNode objects
    output_c = await ctx.run_node(node_c)
    return f'D -> {output_c}'

  node_d = FunctionNode(func=func_d, rerun_on_resume=True)

  agent = Workflow(
      name='node_like_nested',
      edges=[
          (START, node_d),
      ],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
      resumability_config=app.ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert simplify_events_with_node(
      events, map_dynamic_node_to_the_source=True
  ) == [
      (
          'node_like_nested',
          {'node_name': 'node_a', 'output': 'A'},
      ),
      (
          'node_like_nested',
          {'node_name': 'node_b', 'output': 'B'},
      ),
      (
          'node_like_nested',
          {'node_name': 'func_c', 'output': 'C -> A & B'},
      ),
      (
          'node_like_nested',
          {'node_name': 'func_d', 'output': 'D -> C -> A & B'},
      ),
  ]


@pytest.mark.asyncio
async def test_dynamic_node_fails_if_caller_no_rerun(
    request: pytest.FixtureRequest,
):
  """ctx.run_node raises ValueError when caller has rerun_on_resume=False."""

  def node_a() -> str:
    return 'A'

  node_a = FunctionNode(func=node_a)

  async def node_caller(ctx: Context) -> str:
    return await ctx.run_node(node_a)

  # Caller has rerun_on_resume=False (default is False for FunctionNode)
  node_caller = FunctionNode(func=node_caller, rerun_on_resume=False)

  agent = Workflow(
      name='test_agent_fail_no_rerun',
      edges=[(START, node_caller)],
  )

  test_app = app.App(
      name=request.function.__name__,
      root_node=agent,
  )
  runner = testing_utils.InMemoryRunner(app=test_app)

  user_event = testing_utils.get_user_content('start')

  with pytest.raises(ValueError, match='A node must have rerun_on_resume=True'):
    await runner.run_async(user_event)
