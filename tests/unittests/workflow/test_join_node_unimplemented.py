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

"""Testings for the JoinNode."""

from google.adk.apps import app
from google.adk.workflow import _base_node as base_node
from google.adk.workflow import _join_node as join_node
from google.adk.workflow import _workflow as workflow
from google.adk.workflow import _workflow_graph as workflow_graph
from google.adk.workflow import START
from google.adk.workflow import Workflow
from pydantic import BaseModel
import pytest

from .. import testing_utils
from . import workflow_testing_utils

pytest.skip(
    'Skipping since not yet migrated to use .',
    allow_module_level=True,
)


def _build_join_node_workflow(
    request: pytest.FixtureRequest,
) -> tuple[
    workflow_testing_utils.InputCapturingNode, testing_utils.InMemoryRunner
]:
  """Builds a workflow with a JoinNode."""
  node_a = workflow_testing_utils.TestingNode(
      name='NodeA', output={'a': 1, 'b': 1}
  )
  node_b = workflow_testing_utils.TestingNode(name='NodeB', output={'b': 2})
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  agent = workflow.Workflow(
      name='test_join_node',
      edges=[
          workflow_graph.Edge(base_node.START, node_a),
          workflow_graph.Edge(base_node.START, node_b),
          workflow_graph.Edge(node_a, node_join),
          workflow_graph.Edge(node_b, node_join),
          workflow_graph.Edge(node_join, node_capture),
      ],
  )
  app_instance = app.App(
      name=request.function.__name__,
      root_agent=agent,
  )
  return node_capture, testing_utils.InMemoryRunner(app=app_instance)


def test_get_state_key():
  """Tests _get_state_key."""
  node = join_node.JoinNode(name='NodeJoin')
  assert node._get_state_key('path/to/node') == 'path/to/node_join_state'


@pytest.mark.asyncio
async def test_join_node_waits_for_all_inputs(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in."""
  node_capture, runner = _build_join_node_workflow(request)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [{
      'NodeA': {'a': 1, 'b': 1},
      'NodeB': {'b': 2},
  }]

  # assert that there is a state event to save the state with agent path as key
  assert any(
      e.actions.state_delta
      and 'test_join_node/NodeJoin_join_state' in e.actions.state_delta
      for e in events
  )


@pytest.mark.asyncio
async def test_join_node_with_none_state(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in when node state is None."""
  node_capture, runner = _build_join_node_workflow(request)
  # Run once to set state to None
  await runner.run_async(testing_utils.get_user_content('start'))
  # Run again to trigger join_node with state=None
  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [
      {'NodeA': {'a': 1, 'b': 1}, 'NodeB': {'b': 2}},
      {'NodeA': {'a': 1, 'b': 1}, 'NodeB': {'b': 2}},
  ]


@pytest.mark.asyncio
async def test_join_node_with_none_inputs(request: pytest.FixtureRequest):
  """Tests JoinNode with fan-in when incoming edges have None output."""
  node_a = workflow_testing_utils.TestingNode(
      name='NodeA', output=None, route='NodeJoin'
  )
  node_b = workflow_testing_utils.TestingNode(
      name='NodeB', output=None, route='NodeJoin'
  )
  node_join = join_node.JoinNode(name='NodeJoin')
  node_capture = workflow_testing_utils.InputCapturingNode(name='NodeCapture')
  agent = workflow.Workflow(
      name='test_join_node_none_inputs',
      edges=[
          workflow_graph.Edge(base_node.START, node_a),
          workflow_graph.Edge(base_node.START, node_b),
          workflow_graph.Edge(node_a, node_join),
          workflow_graph.Edge(node_b, node_join),
          workflow_graph.Edge(node_join, node_capture),
      ],
  )
  app_instance = app.App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app_instance)

  await runner.run_async(testing_utils.get_user_content('start'))

  assert node_capture.received_inputs == [
      {'NodeA': None, 'NodeB': None},
  ]


# ── JoinNode input_schema ──────────────────────────────────────
# input_schema on JoinNode validates each trigger input individually
# (each predecessor's output), not the joined dict.


class _TriggerInput(BaseModel):
  key: str
  value: int


@pytest.mark.asyncio
async def test_join_node_input_schema_validates_per_trigger(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema validates each trigger input individually."""

  def node_a() -> dict:
    return {'key': 'a', 'value': 1}

  def node_b() -> dict:
    return {'key': 'b', 'value': 2}

  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a),
          (START, node_b),
          (node_a, join),
          (node_b, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == [{
      'node_a': {'key': 'a', 'value': 1},
      'node_b': {'key': 'b', 'value': 2},
  }]


@pytest.mark.asyncio
async def test_join_node_input_schema_rejects_invalid_trigger(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema rejects invalid trigger input early."""

  def node_a() -> dict:
    return {'key': 'a', 'value': 1}

  def node_b() -> dict:
    return {'wrong': 'shape'}  # missing required fields

  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a),
          (START, node_b),
          (node_a, join),
          (node_b, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  with pytest.raises(Exception):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_join_node_input_schema_none_trigger_passes(
    request: pytest.FixtureRequest,
):
  """JoinNode input_schema skips validation for None trigger input."""
  node_a_fn = workflow_testing_utils.TestingNode(
      name='NodeA', output=None, route='join'
  )
  node_b_fn = workflow_testing_utils.TestingNode(
      name='NodeB', output={'key': 'b', 'value': 2}
  )
  join = join_node.JoinNode(name='join', input_schema=_TriggerInput)
  capture = workflow_testing_utils.InputCapturingNode(name='capture')

  agent = Workflow(
      name='wf',
      edges=[
          (START, node_a_fn),
          (START, node_b_fn),
          (node_a_fn, join),
          (node_b_fn, join),
          (join, capture),
      ],
  )
  app_instance = app.App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  await runner.run_async(testing_utils.get_user_content('start'))

  assert capture.received_inputs == [{
      'NodeA': None,
      'NodeB': {'key': 'b', 'value': 2},
  }]
