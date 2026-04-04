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

"""Tests for Workflow schema validation (input_schema, output_schema)."""

from __future__ import annotations

from google.adk.events.event import Event
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from pydantic import BaseModel
import pytest

from ..workflow_testing_utils import create_parent_invocation_context


class _OutputModel(BaseModel):
  name: str
  value: int


class _OtherModel(BaseModel):
  name: str
  value: int
  extra: str = 'default'


@pytest.mark.asyncio
async def test_workflow_output_schema_validates_terminal(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema validates when downstream reads the output."""

  def produce() -> dict:
    return {'name': 'result', 'value': 10}

  def consume(node_input: dict) -> str:
    return f"got {node_input['name']}"

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=_OutputModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'got result' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_output_schema_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema rejects invalid output when downstream reads."""

  def produce_bad() -> dict:
    return {'wrong_field': 'oops'}

  def consume(node_input: dict) -> str:
    return 'should not reach'

  inner = Workflow(
      name='wf',
      edges=[(START, produce_bad)],
      output_schema=_OutputModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  with pytest.raises(ValueError):
    [e async for e in outer.run_async(ctx)]


@pytest.mark.asyncio
async def test_workflow_output_schema_coerces_defaults(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema coerces terminal output (fills defaults)."""

  def produce() -> dict:
    return {'name': 'x', 'value': 1}

  def consume(node_input: dict) -> dict:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=_OtherModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  consume_events = [e for e in data_events if e.node_info.name == 'consume']
  assert len(consume_events) == 1
  assert consume_events[0].output == {
      'name': 'x',
      'value': 1,
      'extra': 'default',
  }


@pytest.mark.asyncio
async def test_nested_workflow_output_schema(
    request: pytest.FixtureRequest,
):
  """Nested workflow's output_schema validates before passing to parent."""

  def inner_produce() -> dict:
    return {'name': 'inner', 'value': 3}

  inner_workflow = Workflow(
      name='inner_wf',
      edges=[(START, inner_produce)],
      output_schema=_OutputModel,
  )

  def outer_consume(node_input: dict) -> str:
    return f"got {node_input['name']}"

  outer_workflow = Workflow(
      name='outer_wf',
      edges=[
          (START, inner_workflow),
          (inner_workflow, outer_consume),
      ],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer_workflow
  )
  events = [e async for e in outer_workflow.run_async(ctx)]

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'got inner' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_output_schema_validates_multiple_terminals(
    request: pytest.FixtureRequest,
):
  """Each terminal output is validated when downstream reads."""

  def branch_a() -> dict:
    return {'name': 'from_a', 'value': 1}

  def branch_b() -> dict:
    return {'name': 'from_b', 'value': 2}

  inner = Workflow(
      name='wf',
      edges=[
          (START, branch_a),
          (START, branch_b),
      ],
      output_schema=_OtherModel,
  )

  def consume(node_input: list) -> list:
    return node_input

  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  consume_events = [
      e
      for e in events
      if isinstance(e, Event) and e.output and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  # Both terminal outputs should have 'extra' filled by _OtherModel default.
  terminal_data = sorted(consume_events[0].output, key=lambda d: d['name'])
  assert terminal_data == [
      {'name': 'from_a', 'value': 1, 'extra': 'default'},
      {'name': 'from_b', 'value': 2, 'extra': 'default'},
  ]


@pytest.mark.asyncio
async def test_workflow_output_schema_rejects_invalid_among_multiple_terminals(
    request: pytest.FixtureRequest,
):
  """One invalid terminal among multiple raises validation error."""

  def branch_good() -> dict:
    return {'name': 'ok', 'value': 1}

  def branch_bad() -> dict:
    return {'wrong_field': 'oops'}

  inner = Workflow(
      name='wf',
      edges=[
          (START, branch_good),
          (START, branch_bad),
      ],
      output_schema=_OutputModel,
  )

  def consume(node_input: list) -> str:
    return 'should not reach'

  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  with pytest.raises(ValueError):
    [e async for e in outer.run_async(ctx)]


# ── Primitive and generic type output_schema ─────────────────────────


@pytest.mark.asyncio
async def test_workflow_output_schema_int_coerces(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=int coerces string to int at read time."""

  def produce() -> str:
    return '42'

  def consume(node_input: int) -> int:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=int,
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  consume_events = [
      e
      for e in events
      if isinstance(e, Event) and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == 42


@pytest.mark.asyncio
async def test_workflow_output_schema_int_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=int rejects non-coercible value at read time."""

  def produce() -> dict:
    return {'key': 'value'}

  def consume(node_input: int) -> int:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=int,
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  with pytest.raises(ValueError):
    [e async for e in outer.run_async(ctx)]


@pytest.mark.asyncio
async def test_workflow_output_schema_list_of_str(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=list[str] validates list output at read time."""

  def produce() -> list:
    return ['a', 'b', 'c']

  def consume(node_input: list) -> list:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=list[str],
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  consume_events = [
      e
      for e in events
      if isinstance(e, Event) and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == ['a', 'b', 'c']


@pytest.mark.asyncio
async def test_workflow_output_schema_list_of_basemodel(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=list[BaseModel] validates and serializes."""

  def produce() -> list:
    return [
        {'name': 'x', 'value': 1},
        {'name': 'y', 'value': 2},
    ]

  def consume(node_input: list) -> list:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=list[_OutputModel],
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  ctx = await create_parent_invocation_context(
      request.function.__name__, outer
  )
  events = [e async for e in outer.run_async(ctx)]
  consume_events = [
      e
      for e in events
      if isinstance(e, Event) and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == [
      {'name': 'x', 'value': 1},
      {'name': 'y', 'value': 2},
  ]


# ── End-to-end: input_schema + output_schema combined ──────────────


class _TaskOutput(BaseModel):
  result: str
  score: int


class _ReviewResult(BaseModel):
  result: str
  score: int
  reviewer: str = 'auto'


@pytest.mark.asyncio
async def test_e2e_input_and_output_schema_pipeline(
    request: pytest.FixtureRequest,
):
  """output_schema on producer + input_schema on consumer validates both."""

  def produce() -> _TaskOutput:
    return {'result': 'done', 'score': 88}

  def consume(node_input: _TaskOutput) -> str:
    return f'result={node_input.result}, score={node_input.score}'

  agent = Workflow(
      name='wf',
      edges=[(START, produce), (produce, consume)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'result=done, score=88' for e in data_events)


@pytest.mark.asyncio
async def test_e2e_output_schema_fails_before_input_schema(
    request: pytest.FixtureRequest,
):
  """Producer output_schema failure prevents consumer from running."""

  def produce_bad() -> _TaskOutput:
    return {'wrong': 'shape'}

  def consume(node_input: _TaskOutput) -> str:
    return 'should not reach'

  agent = Workflow(
      name='wf',
      edges=[(START, produce_bad), (produce_bad, consume)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  with pytest.raises(ValueError):
    [e async for e in agent.run_async(ctx)]


@pytest.mark.asyncio
async def test_e2e_fan_out_join_with_schemas(
    request: pytest.FixtureRequest,
):
  """Fan-out -> join -> consume with input/output schemas."""

  class _JoinedResult(BaseModel):
    analyzer: dict
    summarizer: dict

  def analyzer() -> _TaskOutput:
    return {'result': 'analysis complete', 'score': 92}

  def summarizer() -> _TaskOutput:
    return {'result': 'summary complete', 'score': 78}

  join = JoinNode(name='join')

  def reviewer(node_input: dict) -> _ReviewResult:
    a = node_input['analyzer']
    s = node_input['summarizer']
    return {
        'result': f"{a['result']} + {s['result']}",
        'score': (a['score'] + s['score']) // 2,
    }

  agent = Workflow(
      name='wf',
      edges=[
          (START, analyzer),
          (START, summarizer),
          (analyzer, join),
          (summarizer, join),
          (join, reviewer),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  terminal = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and e.node_info.name == 'reviewer'
  ]
  assert len(terminal) == 1
  assert terminal[0].output == {
      'result': 'analysis complete + summary complete',
      'score': 85,
      'reviewer': 'auto',
  }
  assert terminal[0].node_info.path == 'wf/reviewer'
