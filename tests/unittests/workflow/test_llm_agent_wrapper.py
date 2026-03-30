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

"""Tests for LlmAgentWrapper.

Verifies that LlmAgentWrapper correctly adapts LlmAgent for use as a
workflow graph node, covering mode validation, input conversion,
content isolation, output extraction, and both old/new workflow paths.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.context import Context
from google.adk.agents.llm.task._task_models import TaskResult
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper
from google.adk.workflow.utils._workflow_graph_utils import build_node
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import TestingNode

# --- Fixtures ---


class StoryOutput(BaseModel):
  title: str
  content: str


class StoryInput(BaseModel):
  topic: str
  style: str = 'narrative'


def _make_agent(
    name: str = 'test_agent',
    mode: str = 'task',
    **kwargs,
) -> LlmAgent:
  return LlmAgent(
      name=name,
      model='gemini-2.5-flash',
      instruction='Test agent.',
      mode=mode,
      **kwargs,
  )


def _mock_agent_run(agent, finish_output=None, content_text=None):
  """Mocks agent.run to yield events. Returns a context manager."""

  async def fake_run(*, ctx, node_input):
    if content_text:
      yield Event(
          invocation_id='inv',
          author=agent.name,
          content=types.Content(parts=[types.Part(text=content_text)]),
      )
    if finish_output is not None:
      yield Event(
          invocation_id='inv',
          author=agent.name,
          actions=EventActions(
              finish_task=TaskResult(output=finish_output).model_dump(),
          ),
      )

  original = agent.run
  object.__setattr__(agent, 'run', fake_run)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(agent, 'run', original)

  return _Ctx()


def _mock_leaf_run(wrapper, content_text=None):
  """Mocks the old-workflow leaf path (_single.run). Returns a context manager."""
  target = wrapper._single if wrapper._single is not None else wrapper.agent

  async def fake_run(*, ctx, node_input):
    if content_text:
      yield Event(output=content_text)

  original = target.run
  object.__setattr__(target, 'run', fake_run)

  class _Ctx:

    def __enter__(self):
      return self

    def __exit__(self, *args):
      object.__setattr__(target, 'run', original)

  return _Ctx()


def _new_workflow_runner(wf, test_name):
  """Creates an InMemoryRunner for the new Workflow (root_node path)."""
  from google.adk.apps.app import App

  from . import testing_utils

  app = App(name=test_name, root_node=wf)
  return testing_utils.InMemoryRunner(app=app)


# --- Validation ---


class TestValidation:

  def test_task_mode_accepted(self):
    """Wrapping a task-mode agent succeeds."""
    wrapper = _LlmAgentWrapper(agent=_make_agent(mode='task'))
    assert wrapper.name == 'test_agent'

  def test_single_turn_mode_accepted(self):
    """Wrapping a single_turn-mode agent succeeds."""
    wrapper = _LlmAgentWrapper(agent=_make_agent(mode='single_turn'))
    assert wrapper.name == 'test_agent'

  def test_chat_mode_rejected(self):
    """Wrapping a chat-mode agent raises ValueError."""
    with pytest.raises(ValueError, match='task and single_turn'):
      _LlmAgentWrapper(agent=_make_agent(mode='chat'))

  def test_name_defaults_to_agent_name(self):
    """Wrapper name defaults to the inner agent's name."""
    wrapper = _LlmAgentWrapper(agent=_make_agent(name='my_agent'))
    assert wrapper.name == 'my_agent'

  def test_name_can_be_overridden(self):
    """Explicit name overrides the agent's name."""
    wrapper = _LlmAgentWrapper(
        agent=_make_agent(name='my_agent'), name='custom'
    )
    assert wrapper.name == 'custom'

  def test_task_mode_waits_for_output(self):
    """Task mode sets wait_for_output=True."""
    wrapper = _LlmAgentWrapper(agent=_make_agent(mode='task'))
    assert wrapper.wait_for_output is True

  def test_single_turn_does_not_wait_for_output(self):
    """Single_turn mode does not set wait_for_output."""
    wrapper = _LlmAgentWrapper(agent=_make_agent(mode='single_turn'))
    assert wrapper.wait_for_output is False

  def test_rerun_on_resume_defaults_true(self):
    """Wrapper defaults to rerun_on_resume=True."""
    wrapper = _LlmAgentWrapper(agent=_make_agent())
    assert wrapper.rerun_on_resume is True

  def test_workflow_as_sub_agent_rejected(self):
    """Using a Workflow as a sub_agent of LlmAgent raises ValueError."""
    wf = Workflow(name='wf', edges=[(START, lambda: 'done')])
    with pytest.raises(
        ValueError, match='Workflow.*cannot be used as a sub_agent'
    ):
      LlmAgent(
          name='parent',
          model='gemini-2.5-flash',
          instruction='Test.',
          sub_agents=[wf],
      )


# --- build_node auto-wrapping ---


class TestBuildNode:

  def test_task_mode_wrapped(self):
    """build_node wraps a task-mode LlmAgent."""
    agent = _make_agent(mode='task')
    node = build_node(agent)
    assert isinstance(node, _LlmAgentWrapper)
    assert node.agent is agent

  def test_single_turn_mode_wrapped(self):
    """build_node wraps a single_turn-mode LlmAgent."""
    node = build_node(_make_agent(mode='single_turn'))
    assert isinstance(node, _LlmAgentWrapper)

  def test_default_mode_auto_set_to_single_turn(self):
    """LlmAgent with default mode is auto-converted to single_turn."""
    agent = LlmAgent(
        name='agent', model='gemini-2.5-flash', instruction='Test.'
    )

    build_node(agent)

    assert agent.mode == 'single_turn'

  def test_explicit_chat_mode_rejected(self):
    """build_node rejects LlmAgent with explicit chat mode."""
    with pytest.raises(ValueError, match="mode='chat'.*not supported"):
      build_node(_make_agent(mode='chat'))

  def test_mode_synced_when_used_in_workflow_edges(self):
    """LlmAgent mode is set to single_turn when placed in Workflow edges."""
    agent = LlmAgent(
        name='agent', model='gemini-2.5-flash', instruction='Test.'
    )
    assert agent.mode == 'chat'

    Workflow(name='wf', edges=[(START, agent)])

    assert agent.mode == 'single_turn'

  def test_name_override(self):
    """build_node respects explicit name override."""
    node = build_node(_make_agent(mode='task'), name='override')
    assert node.name == 'override'


# --- Old workflow path (uses _SingleLlmAgent via _single) ---


@pytest.mark.asyncio
async def test_task_finish_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Task mode extracts finish_task output for downstream nodes."""
  agent = _make_agent(mode='task')
  wrapper = _LlmAgentWrapper(agent=agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[(START, wrapper), (wrapper, capture)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(
      agent,
      finish_output={'title': 'Story', 'content': 'Once upon a time'},
      content_text='Writing...',
  ):
    [e async for e in wf.run_async(ctx)]

  assert capture.received_inputs == [
      {'title': 'Story', 'content': 'Once upon a time'}
  ]


@pytest.mark.asyncio
async def test_single_turn_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """Single_turn output flows to downstream nodes."""
  agent = _make_agent(mode='single_turn')
  wrapper = _LlmAgentWrapper(agent=agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[(START, wrapper), (wrapper, capture)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_leaf_run(wrapper, content_text='Done.'):
    [e async for e in wf.run_async(ctx)]

  assert capture.received_inputs == ['Done.']


@pytest.mark.asyncio
async def test_valid_input_schema_accepted(
    request: pytest.FixtureRequest,
):
  """Valid dict matching input_schema passes through without error."""
  agent = _make_agent(mode='task', input_schema=StoryInput)
  wrapper = _LlmAgentWrapper(agent=agent)
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[(START, wrapper), (wrapper, capture)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(agent, finish_output={'result': 'ok'}):
    [e async for e in wf.run_async(ctx)]

  assert capture.received_inputs == [{'result': 'ok'}]


@pytest.mark.asyncio
async def test_invalid_input_schema_raises(
    request: pytest.FixtureRequest,
):
  """Invalid input not matching input_schema raises ValidationError."""
  agent = _make_agent(mode='task', input_schema=StoryInput)
  wrapper = _LlmAgentWrapper(agent=agent)
  wf = Workflow(name='wf', edges=[(START, wrapper)])
  ctx = await create_parent_invocation_context(request.function.__name__, wf)
  ic = ctx.model_copy(update={'branch': None})
  agent_ctx = Context(
      invocation_context=ic, node_path='wf', run_id='exec'
  )

  with _mock_agent_run(agent, finish_output={'result': 'ok'}):
    with pytest.raises(ValidationError):
      async for _ in wrapper.run(
          ctx=agent_ctx, node_input={'style': 'comedy'}
      ):
        pass


@pytest.mark.asyncio
async def test_auto_wrap_in_workflow_edges(request: pytest.FixtureRequest):
  """LlmAgent placed directly in edges is auto-wrapped and works."""
  agent = _make_agent(mode='task')
  capture = InputCapturingNode(name='capture')
  wf = Workflow(
      name='wf',
      edges=[(START, agent), (agent, capture)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  with _mock_agent_run(agent, finish_output={'result': 'auto'}):
    [e async for e in wf.run_async(ctx)]

  assert capture.received_inputs == [{'result': 'auto'}]


@pytest.mark.asyncio
async def test_single_turn_isolates_content_via_branch(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper sets a branch for content isolation."""
  agent = _make_agent(mode='single_turn')
  wrapper = _LlmAgentWrapper(agent=agent)
  captured_branches = []

  async def fake_run(*, ctx, node_input):
    captured_branches.append(ctx._invocation_context.branch)
    yield Event(output='response')

  wf = Workflow(name='wf', edges=[(START, wrapper)])
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  original = wrapper._single.run
  object.__setattr__(wrapper._single, 'run', fake_run)
  try:
    [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(wrapper._single, 'run', original)

  assert len(captured_branches) == 1
  assert captured_branches[0].startswith('node:')
  assert agent.name in captured_branches[0]


@pytest.mark.asyncio
async def test_task_mode_does_not_set_branch(
    request: pytest.FixtureRequest,
):
  """Task mode preserves None branch for HITL visibility."""
  agent = _make_agent(mode='task')
  wrapper = _LlmAgentWrapper(agent=agent)
  captured_branches = []

  async def fake_run(*, ctx, node_input):
    captured_branches.append(ctx._invocation_context.branch)
    yield Event(
        invocation_id='inv',
        author=agent.name,
        actions=EventActions(finish_task={'output': {'result': 'done'}}),
    )

  wf = Workflow(name='wf', edges=[(START, wrapper)])
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  object.__setattr__(agent, 'run', fake_run)
  try:
    [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(agent, 'run', agent.__class__.run)

  assert captured_branches == [None]


@pytest.mark.asyncio
async def test_single_turn_converts_input_to_content(
    request: pytest.FixtureRequest,
):
  """Single_turn wrapper converts string node_input to types.Content."""
  agent = _make_agent(mode='single_turn')
  wrapper = _LlmAgentWrapper(agent=agent)
  captured_inputs = []

  async def fake_run(*, ctx, node_input):
    captured_inputs.append(node_input)
    yield Event(output='response')

  predecessor = TestingNode(name='pred', output='hello world')
  wf = Workflow(
      name='wf',
      edges=[(START, predecessor), (predecessor, wrapper)],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, wf)

  original = wrapper._single.run
  object.__setattr__(wrapper._single, 'run', fake_run)
  try:
    [e async for e in wf.run_async(ctx)]
  finally:
    object.__setattr__(wrapper._single, 'run', original)

  assert len(captured_inputs) == 1
  assert isinstance(captured_inputs[0], types.Content)
  assert captured_inputs[0].parts[0].text == 'hello world'


# --- New workflow path (uses SingleAgentReactNode via _react) ---


def _get_user_content():
  from . import testing_utils

  return testing_utils.get_user_content


@pytest.mark.asyncio
async def test_react_path_user_content_visible_to_llm(
    request: pytest.FixtureRequest,
):
  """First-node LLM agent sees the user message in the new Workflow."""
  from google.adk.workflow._workflow_class import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['extracted output'])
  agent = LlmAgent(
      name='process_request',
      model=mock_model,
      instruction='Extract info from the user message.',
  )
  wf = NewWorkflow(name='wf', edges=[('START', agent)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(
      testing_utils.get_user_content('I want 3 days off for vacation')
  )

  assert len(mock_model.requests) == 1
  user_texts = [
      p.text
      for c in mock_model.requests[0].contents
      if c.role == 'user'
      for p in c.parts or []
      if p.text
  ]
  assert any('3 days' in t for t in user_texts)


@pytest.mark.asyncio
async def test_react_path_output_reaches_downstream(
    request: pytest.FixtureRequest,
):
  """LLM output flows to the next node in the new Workflow."""
  from google.adk.workflow._workflow_class import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['hello world'])
  agent = LlmAgent(
      name='greeter', model=mock_model, instruction='Greet.',
  )
  captured = []

  def capture(node_input: str):
    captured.append(node_input)

  wf = NewWorkflow(name='wf', edges=[('START', agent, capture)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('hi'))

  assert captured == ['hello world']


@pytest.mark.asyncio
async def test_react_path_output_key_stored_in_state(
    request: pytest.FixtureRequest,
):
  """output_key stores LLM output in state in the new Workflow."""
  from google.adk.workflow._workflow_class import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['summary text'])
  agent = LlmAgent(
      name='summarizer',
      model=mock_model,
      instruction='Summarize.',
      output_key='summary',
  )
  captured_state = []

  def check_state(ctx: Context):
    captured_state.append(ctx.state.get('summary'))

  wf = NewWorkflow(name='wf', edges=[('START', agent, check_state)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('some text'))

  assert captured_state == ['summary text']


@pytest.mark.asyncio
async def test_react_path_output_schema_validated(
    request: pytest.FixtureRequest,
):
  """output_schema is validated and parsed in the new Workflow."""
  from google.adk.workflow._workflow_class import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(
      responses=['{"title": "My Story", "content": "Once upon a time"}']
  )
  agent = LlmAgent(
      name='writer',
      model=mock_model,
      instruction='Write a story.',
      output_schema=StoryOutput,
      output_key='story',
  )
  captured = []

  def check_output(node_input: dict):
    captured.append(node_input)

  wf = NewWorkflow(name='wf', edges=[('START', agent, check_output)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('write'))

  assert len(captured) == 1
  assert captured[0]['title'] == 'My Story'
  assert captured[0]['content'] == 'Once upon a time'


@pytest.mark.asyncio
async def test_react_path_predecessor_input_visible_to_llm(
    request: pytest.FixtureRequest,
):
  """Predecessor output is injected as user content for the LLM."""
  from google.adk.workflow._workflow_class import Workflow as NewWorkflow

  from . import testing_utils

  mock_model = testing_utils.MockModel.create(responses=['processed'])
  agent = LlmAgent(
      name='processor', model=mock_model, instruction='Process.',
  )

  def step_one(node_input: str) -> str:
    return 'transformed data'

  wf = NewWorkflow(name='wf', edges=[('START', step_one, agent)])

  runner = _new_workflow_runner(wf, request.function.__name__)
  await runner.run_async(testing_utils.get_user_content('raw input'))

  assert len(mock_model.requests) == 1
  user_texts = [
      p.text
      for c in mock_model.requests[0].contents
      if c.role == 'user'
      for p in c.parts or []
      if p.text
  ]
  assert any('transformed data' in t for t in user_texts)
