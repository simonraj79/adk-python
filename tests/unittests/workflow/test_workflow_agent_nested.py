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

import copy
from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm._call_llm_node import CallLlmResult
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import JoinNode
from google.adk.workflow import Workflow
from google.adk.workflow._execution_state import NodeStatus
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
import pytest
from typing_extensions import override

from . import testing_utils
from .workflow_testing_utils import InputCapturingNode
from .workflow_testing_utils import RequestInputNode
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import simplify_events_with_node_and_agent_state
from .workflow_testing_utils import strip_checkpoint_events
from .workflow_testing_utils import TestingNode


def long_running_tool_func():
  """A test tool that simulates a long-running operation."""
  return None


@pytest.mark.asyncio
async def test_nested_workflow_agent_as_node(request: pytest.FixtureRequest):
  """Tests that a Workflow can be used as a node in another Workflow."""

  async def nested_func(node_input: types.Content):
    return 'I am nested'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_func)],
  )

  async def output_func(node_input: str):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/nested_func',
          {
              'node_name': 'nested_func',
              'output': 'I am nested',
          },
      ),
      # nested_agent output is resolved via terminal path resolution.
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_agent_as_node_resumable(
    request: pytest.FixtureRequest,
):
  """Tests that a Workflow can be used as a node in another Workflow with resumability."""

  async def nested_func(node_input: types.Content):
    return 'I am nested'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_func)],
  )

  async def output_func(node_input: str):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node_and_agent_state(
      events, use_node_path=True
  )
  assert simplified_events == [
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'nested_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent/nested_func',
          {'node_name': 'nested_func', 'output': 'I am nested'},
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'nested_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # nested_agent finalize event is deduplicated (non-root, no
      # output_schema).
      ('outer_agent/nested_agent', testing_utils.END_OF_AGENT),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/output_func',
          {'node_name': 'output_func', 'output': 'I am outer'},
      ),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('outer_agent', testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_agent_with_join_node(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow with JoinNode works correctly."""

  async def nested_node_a():
    return {'a': 1}

  async def nested_node_b():
    return {'b': 2}

  async def nested_node_c():
    return {'c': 3}

  nested_join_node = JoinNode(name='nested_join')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', nested_node_a),
          ('START', nested_node_c),
          (nested_node_a, nested_join_node),
          (nested_node_c, nested_node_b),
          (nested_node_b, nested_join_node),
      ],
  )

  async def output_func(node_input: dict):
    return (
        'Joined output:'
        f' a={node_input["nested_node_a"]["a"]},'
        f' b={node_input["nested_node_b"]["b"]}'
    )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert sorted(simplified_events[0:2], key=lambda x: x[1]['node_name']) == [
      (
          'outer_agent/nested_agent/nested_node_a',
          {'node_name': 'nested_node_a', 'output': {'a': 1}},
      ),
      (
          'outer_agent/nested_agent/nested_node_c',
          {'node_name': 'nested_node_c', 'output': {'c': 3}},
      ),
  ]
  assert simplified_events[2:] == [
      (
          'outer_agent/nested_agent/nested_node_b',
          {'node_name': 'nested_node_b', 'output': {'b': 2}},
      ),
      (
          'outer_agent/nested_agent/nested_join',
          {
              'node_name': 'nested_join',
              'output': {'nested_node_a': {'a': 1}, 'nested_node_b': {'b': 2}},
          },
      ),
      # nested_agent finalize event is deduplicated.
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'Joined output: a=1, b=2',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_agent_updates_state_outer_reads(
    request: pytest.FixtureRequest,
):
  """Tests that outer agent can read state updated by nested agent."""

  async def nested_state_updater(ctx: Context):
    yield Event(
        state={'my_key': 'my_value'},
    )
    yield 'nested agent finished'

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', nested_state_updater)],
  )

  def outer_state_reader(my_key: str, node_input: str):
    return f'Nested agent output: {node_input}, state value: {my_key}'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, outer_state_reader)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/nested_state_updater',
          {
              'node_name': 'nested_state_updater',
              'output': 'nested agent finished',
          },
      ),
      # nested_agent finalize event is deduplicated.
      (
          'outer_agent/outer_state_reader',
          {
              'node_name': 'outer_state_reader',
              'output': (
                  'Nested agent output: nested agent finished, state value:'
                  ' my_value'
              ),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_nested_workflow_agent_intermediate_nodes(
    request: pytest.FixtureRequest,
):
  """Tests that only the final output of a nested workflow is passed to the outer workflow."""

  # Nested workflow: NodeA -> NodeB
  # NodeA outputs 'Inner Intermediate', but it's not the final node.
  node_a = TestingNode(name='NodeA', output='Inner Intermediate')
  # NodeB outputs 'Inner Final', and it is the final node (leaf).
  node_b = TestingNode(name='NodeB', output='Inner Final')

  nested_agent = Workflow(
      name='nested_agent',
      edges=[
          ('START', node_a),
          (node_a, node_b),
      ],
  )

  # Outer workflow: NestedAgent -> OutputNode
  output_node = InputCapturingNode(name='OutputNode')

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_node),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Verify that OutputNode received 'Inner Final' and NOT 'Inner Intermediate'
  assert output_node.received_inputs == ['Inner Final']

  # Verify the event stream
  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/nested_agent/NodeA',
          {'node_name': 'NodeA', 'output': 'Inner Intermediate'},
      ),
      (
          'outer_agent/nested_agent/NodeB',
          {'node_name': 'NodeB', 'output': 'Inner Final'},
      ),
      # nested_agent finalize event is deduplicated.
      (
          'outer_agent/OutputNode',
          {'node_name': 'OutputNode', 'output': {'received': 'Inner Final'}},
      ),
  ]


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_nested_workflow_agent_with_hitl(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests that a nested Workflow with HITL works correctly."""
  llm_agent = LlmAgent(
      name='llm_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='long_running_tool_func',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tool'),
          ]
      ),
      tools=[LongRunningFunctionTool(func=long_running_tool_func)],
  )

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func(node_input: Any):
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on the long-running function call.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  function_call_id = None
  for ev in events1:
    if (
        ev.content
        and ev.content.parts
        and ev.content.parts[0].function_call
        and ev.content.parts[0].function_call.name == 'long_running_tool_func'
    ):
      function_call_id = ev.content.parts[0].function_call.id
      break
  assert function_call_id is not None

  expected_events1 = [
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'llm_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      # LlmAgent is now a Mesh, so it emits internal workflow state events.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {'status': NodeStatus.RUNNING.value},
              },
          },
      ),
      (
          'outer_agent/nested_agent/llm_agent/call_llm',
          {
              'node_name': 'call_llm',
              'output': CallLlmResult(
                  function_calls=[
                      types.FunctionCall(
                          name='long_running_tool_func',
                          args={},
                          id=function_call_id,
                      )
                  ],
              ),
          },
      ),
      # call_llm completes and routes to execute_tools.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {
                      'status': NodeStatus.COMPLETED.value,
                  },
                  'execute_tools': {
                      'status': NodeStatus.RUNNING.value,
                  },
              },
          },
      ),
      # execute_tools yields the function_call (with long_running_tool_ids)
      # as an interrupt event since call_llm suppresses its finalized event
      # for long-running tools.
      (
          'outer_agent/nested_agent/llm_agent/execute_tools',
          types.Part.from_function_call(name='long_running_tool_func', args={}),
      ),
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {
                      'status': NodeStatus.COMPLETED.value,
                  },
                  'execute_tools': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': [function_call_id],
                  },
              },
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'llm_agent': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': [function_call_id],
                  },
              },
          },
      ),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': [function_call_id],
                  },
              },
          },
      ),
  ]

  events1_simplified = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events1), use_node_path=True
  )
  assert events1_simplified == (
      expected_events1
      if resumable
      else strip_checkpoint_events(expected_events1)
  )

  tool_response = testing_utils.UserContent(
      types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name='long_running_tool_func',
              response={'result': 'Final tool output'},
          )
      )
  )

  # Resume with tool output
  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=tool_response,
      invocation_id=invocation_id,
  )

  expected_events2 = [
      # outer_agent schedules nested_agent for re-run.
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      # nested_agent schedules llm_agent for re-run.
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'llm_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      # llm_agent schedules execute_tools for re-run (rerun_on_resume).
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {'status': NodeStatus.COMPLETED.value},
                  'execute_tools': {'status': NodeStatus.RUNNING.value},
              },
          },
      ),
      # execute_tools processes the long-running resume → FR.
      (
          'outer_agent/nested_agent/llm_agent/execute_tools',
          types.Part(
              function_response=types.FunctionResponse(
                  name='long_running_tool_func',
                  response={'result': 'Final tool output'},
              )
          ),
      ),
      # execute_tools completes, routes back to call_llm.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {'status': NodeStatus.RUNNING.value},
                  'execute_tools': {'status': NodeStatus.COMPLETED.value},
              },
          },
      ),
      # call_llm produces text response (no more function calls).
      (
          'outer_agent/nested_agent/llm_agent/call_llm',
          {
              'node_name': 'call_llm',
              'output': 'LLM response after tool',
          },
      ),
      # llm_agent completes.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'nodes': {
                  'call_llm': {'status': NodeStatus.COMPLETED.value},
                  'execute_tools': {'status': NodeStatus.COMPLETED.value},
              },
          },
      ),
      # LlmAgent re-emits single_turn output before END_OF_AGENT.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'node_name': 'llm_agent',
              'output': 'LLM response after tool',
          },
      ),
      ('outer_agent/nested_agent/llm_agent', testing_utils.END_OF_AGENT),
      # nested_agent completes.
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'llm_agent': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # nested_agent output is resolved via terminal path resolution.
      ('outer_agent/nested_agent', testing_utils.END_OF_AGENT),
      # outer_agent runs output_func.
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/output_func',
          {'node_name': 'output_func', 'output': 'I am outer'},
      ),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('outer_agent', testing_utils.END_OF_AGENT),
  ]

  events2_simplified = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events2), use_node_path=True
  )
  assert events2_simplified == (
      expected_events2
      if resumable
      else strip_checkpoint_events(expected_events2)
  )


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_nested_workflow_agent_with_request_input_event_hitl(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests that a nested Workflow with RequestInputEvent HITL works correctly."""

  class NodeHitl(BaseNode):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerun_on_resume: bool = Field(default=True)
    name: str = Field(default='node_hitl')

    @override
    async def run(
        self,
        *,
        ctx: Context,
        node_input: Any,
    ) -> AsyncGenerator[Any, None]:
      if resume_input := ctx.resume_inputs.get('request_input'):
        yield f'Resumed with user input: {resume_input}'
      else:
        yield RequestInput(
            message='requesting input via RequestInputEvent',
            interrupt_id='request_input',
        )

  node_hitl_instance = NodeHitl()

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', node_hitl_instance)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', nested_agent), (nested_agent, output_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on RequestInputEvent.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id == 'request_input'

  simplified_events = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events1), use_node_path=True
  )

  expected_events1 = [
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'node_hitl': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent/node_hitl',
          testing_utils.simplify_content(req_events[0].content),
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'node_hitl': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': ['request_input'],
                  },
              },
          },
      ),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {
                      'status': NodeStatus.WAITING.value,
                      'interrupts': ['request_input'],
                  },
              },
          },
      ),
  ]

  events1_simplified = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events1), use_node_path=True
  )
  assert events1_simplified == (
      expected_events1
      if resumable
      else strip_checkpoint_events(expected_events1)
  )

  hitl_response = types.Content(
      parts=[
          create_request_input_response(
              interrupt_id, {'response': 'user input for hitl'}
          )
      ],
      role='user',
  )

  # Resume with tool output
  invocation_id = events1[0].invocation_id
  events2 = await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  expected_events2 = [
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'node_hitl': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/nested_agent/node_hitl',
          {
              'node_name': 'node_hitl',
              'output': (
                  "Resumed with user input: {'response': 'user input for hitl'}"
              ),
          },
      ),
      (
          'outer_agent/nested_agent',
          {
              'nodes': {
                  'node_hitl': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # nested_agent finalize event is deduplicated.
      ('outer_agent/nested_agent', testing_utils.END_OF_AGENT),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'outer_agent/output_func',
          {'node_name': 'output_func', 'output': 'I am outer'},
      ),
      (
          'outer_agent',
          {
              'nodes': {
                  'nested_agent': {'status': NodeStatus.COMPLETED.value},
                  'output_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('outer_agent', testing_utils.END_OF_AGENT),
  ]

  events2_simplified = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events2), use_node_path=True
  )
  assert events2_simplified == (
      expected_events2
      if resumable
      else strip_checkpoint_events(expected_events2)
  )


@pytest.mark.parametrize('resumable', [True, False])
@pytest.mark.asyncio
async def test_nested_agent_with_request_input_piped_to_next_node(
    request: pytest.FixtureRequest, resumable: bool
):
  """Tests that user response to RequestInput in nested agent is piped to next node."""
  ask_user = RequestInputNode(
      name='ask_user',
      message='Please provide input',
  )
  capture_node = InputCapturingNode(name='capture_node')

  sub_agent = Workflow(
      name='sub_agent',
      edges=[
          ('START', ask_user),
          (ask_user, capture_node),
      ],
  )

  root_agent = Workflow(
      name='root_agent',
      edges=[('START', sub_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause on RequestInputEvent.
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  req_events = [e for e in events1 if has_request_input_function_call(e)]
  assert req_events
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  assert interrupt_id
  hitl_response_payload = {'response': 'user input for hitl'}
  hitl_response = types.Content(
      parts=[
          create_request_input_response(interrupt_id, hitl_response_payload)
      ],
      role='user',
  )

  # Resume with tool output
  invocation_id = events1[0].invocation_id
  await runner.run_async(
      new_message=hitl_response,
      invocation_id=invocation_id,
  )

  assert capture_node.received_inputs == [hitl_response_payload]


@pytest.mark.asyncio
async def test_nested_workflow_agent_chain_input_propagation(
    request: pytest.FixtureRequest,
):

  async def create_output_a():
    return 'output of A'

  nested_agent_a = Workflow(
      name='nested_agent_a',
      edges=[('START', create_output_a)],
  )

  capture_node_b = InputCapturingNode(name='capture_node_b')
  nested_agent_b = Workflow(
      name='nested_agent_b',
      edges=[('START', capture_node_b)],
  )

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent_a),
          (nested_agent_a, nested_agent_b),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('hello'))

  # The output of nested_agent_a should be propagated as the input to
  # nested_agent_b.
  assert capture_node_b.received_inputs == ['output of A']


@pytest.mark.asyncio
async def test_nested_workflow_agent_with_tool_calls(
    request: pytest.FixtureRequest,
):
  """Tests that a nested Workflow works correctly with two tool calls."""
  tool_call_count = 0

  def simple_tool() -> str:
    nonlocal tool_call_count
    tool_call_count += 1
    return f'Tool output {tool_call_count}'

  llm_agent = LlmAgent(
      name='llm_agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(
                  name='simple_tool',
                  args={},
              ),
              types.Part.from_text(text='LLM response after tools'),
          ]
      ),
      tools=[simple_tool],
  )

  nested_agent = Workflow(
      name='nested_agent',
      edges=[('START', llm_agent)],
  )

  async def output_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[
          ('START', nested_agent),
          (nested_agent, output_func),
      ],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content('start workflow')
  events = await runner.run_async(user_event)

  assert tool_call_count == 1

  simplified_events = simplify_events_with_node(events, use_node_path=True)

  # Extract the dynamically generated function_call_id from call_llm output.
  call_llm_output = simplified_events[1][1]
  function_call_id = call_llm_output['output'].function_calls[0].id

  assert simplified_events == [
      (
          'outer_agent/nested_agent/llm_agent/call_llm',
          types.Part.from_function_call(name='simple_tool', args={}),
      ),
      (
          'outer_agent/nested_agent/llm_agent/call_llm',
          {
              'node_name': 'call_llm',
              'output': CallLlmResult(
                  function_calls=[
                      types.FunctionCall(
                          name='simple_tool',
                          args={},
                          id=function_call_id,
                      )
                  ],
              ),
          },
      ),
      (
          'outer_agent/nested_agent/llm_agent/execute_tools',
          types.Part.from_function_response(
              name='simple_tool',
              response={'result': 'Tool output 1'},
          ),
      ),
      (
          'outer_agent/nested_agent/llm_agent/call_llm',
          {
              'node_name': 'call_llm',
              'output': 'LLM response after tools',
          },
      ),
      # LlmAgentWrapper re-emits the output for downstream routing.
      (
          'outer_agent/nested_agent/llm_agent',
          {
              'node_name': 'llm_agent',
              'output': 'LLM response after tools',
          },
      ),
      # nested_agent finalize event is deduplicated.
      (
          'outer_agent/output_func',
          {
              'node_name': 'output_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_three_level_nested_workflow(request: pytest.FixtureRequest):
  """Tests a 3-level nested Workflow structure."""

  async def inner_func():
    return 'I am inner'

  inner_agent = Workflow(
      name='inner_agent',
      edges=[('START', inner_func)],
  )

  async def middle_func():
    return 'I am middle'

  middle_agent = Workflow(
      name='middle_agent',
      edges=[('START', inner_agent), (inner_agent, middle_func)],
  )

  async def outer_func():
    return 'I am outer'

  outer_agent = Workflow(
      name='outer_agent',
      edges=[('START', middle_agent), (middle_agent, outer_func)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=outer_agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('hello'))

  simplified_events = simplify_events_with_node(events, use_node_path=True)
  assert simplified_events == [
      (
          'outer_agent/middle_agent/inner_agent/inner_func',
          {
              'node_name': 'inner_func',
              'output': 'I am inner',
          },
      ),
      # inner_agent finalize event is deduplicated.
      (
          'outer_agent/middle_agent/middle_func',
          {
              'node_name': 'middle_func',
              'output': 'I am middle',
          },
      ),
      # middle_agent finalize event is deduplicated.
      (
          'outer_agent/outer_func',
          {
              'node_name': 'outer_func',
              'output': 'I am outer',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_duplicate_grandchild_workflow_agent_names(
    request: pytest.FixtureRequest,
):
  """Tests that grandchild workflow agents with same name can coexist.

  This test defines two workflow agents, child1 and child2, which both contain
  a nested workflow agent named 'grandchild'. This tests that the framework can
  distinguish between grandchild agents with the same name but different parent
  agents.
  """

  async def grandchild_func():
    return 'I am grandchild'

  grandchild_agent = Workflow(
      name='grandchild',
      edges=[('START', grandchild_func)],
  )

  # child1 workflow contains a grandchild agent named 'grandchild'.
  child1_agent = Workflow(
      name='child1',
      edges=[('START', grandchild_agent.clone())],
  )

  # child2 workflow also contains a grandchild agent named 'grandchild'.
  child2_agent = Workflow(
      name='child2',
      edges=[('START', grandchild_agent.clone())],
  )

  root_agent = Workflow(
      name='root',
      edges=[('START', child1_agent), (child1_agent, child2_agent)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  simplified_events = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events), use_node_path=True
  )
  assert simplified_events == [
      (
          'root',
          {
              'nodes': {
                  'child1': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child1',
          {
              'nodes': {
                  'grandchild': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child1/grandchild',
          {
              'nodes': {
                  'grandchild_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child1/grandchild/grandchild_func',
          {'node_name': 'grandchild_func', 'output': 'I am grandchild'},
      ),
      (
          'root/child1/grandchild',
          {
              'nodes': {
                  'grandchild_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # grandchild finalize event is deduplicated.
      ('root/child1/grandchild', testing_utils.END_OF_AGENT),
      (
          'root/child1',
          {
              'nodes': {
                  'grandchild': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # child1 finalize event is deduplicated.
      ('root/child1', testing_utils.END_OF_AGENT),
      (
          'root',
          {
              'nodes': {
                  'child1': {'status': NodeStatus.COMPLETED.value},
                  'child2': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child2',
          {
              'nodes': {
                  'grandchild': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child2/grandchild',
          {
              'nodes': {
                  'grandchild_func': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'root/child2/grandchild/grandchild_func',
          {'node_name': 'grandchild_func', 'output': 'I am grandchild'},
      ),
      (
          'root/child2/grandchild',
          {
              'nodes': {
                  'grandchild_func': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # grandchild finalize event is deduplicated.
      ('root/child2/grandchild', testing_utils.END_OF_AGENT),
      (
          'root/child2',
          {
              'nodes': {
                  'grandchild': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # child2 finalize event is deduplicated.
      ('root/child2', testing_utils.END_OF_AGENT),
      (
          'root',
          {
              'nodes': {
                  'child1': {'status': NodeStatus.COMPLETED.value},
                  'child2': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('root', testing_utils.END_OF_AGENT),
  ]


@pytest.mark.asyncio
async def test_duplicate_name_in_ancestral_path(
    request: pytest.FixtureRequest,
):
  """Tests that agent with same name can exist in ancestral path (A->B->A)."""

  async def func_a():
    return 'I am A'

  agent_a_child = Workflow(
      name='A',
      edges=[('START', func_a)],
  )

  agent_b = Workflow(
      name='B',
      edges=[('START', agent_a_child)],
  )

  agent_a_root = Workflow(
      name='A',
      edges=[('START', agent_b)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=agent_a_root,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_content = testing_utils.get_user_content('hello')
  events = await runner.run_async(user_content)

  simplified_events = simplify_events_with_node_and_agent_state(
      copy.deepcopy(events), use_node_path=True
  )
  assert simplified_events == [
      (
          'A',
          {
              'nodes': {
                  'B': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'A/B',
          {
              'nodes': {
                  'A': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'A/B/A',
          {
              'nodes': {
                  'func_a': {'status': NodeStatus.RUNNING.value},
              }
          },
      ),
      (
          'A/B/A/func_a',
          {'node_name': 'func_a', 'output': 'I am A'},
      ),
      (
          'A/B/A',
          {
              'nodes': {
                  'func_a': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # A/B/A finalize event is deduplicated.
      ('A/B/A', testing_utils.END_OF_AGENT),
      (
          'A/B',
          {
              'nodes': {
                  'A': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      # B finalize event is deduplicated.
      ('A/B', testing_utils.END_OF_AGENT),
      (
          'A',
          {
              'nodes': {
                  'B': {'status': NodeStatus.COMPLETED.value},
              }
          },
      ),
      ('A', testing_utils.END_OF_AGENT),
  ]
