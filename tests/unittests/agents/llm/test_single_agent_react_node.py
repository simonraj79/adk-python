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

"""Tests for SingleAgentReactNode.

Runs SingleAgentReactNode directly via Runner(node=...) — no Workflow
wrapper needed because SingleAgentReactNode creates its own
schedule_dynamic_node and sets ic.agent from its agent field.
"""

from __future__ import annotations

from google.adk.agents.llm._single_agent_react_node import SingleAgentReactNode
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from tests.unittests.agents.llm.event_utils import function_call_names
from tests.unittests.agents.llm.event_utils import function_response_dicts
from tests.unittests.agents.llm.event_utils import function_responses_by_name
from tests.unittests.agents.llm.event_utils import text_parts

from ... import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USER_ID = 'test_user'
_SESSION_ID = 'test_session'


async def _run(mock_model, user_message, tools=None, **agent_kwargs):
  """Run SingleAgentReactNode via Runner and return events."""
  llm_agent = LlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **agent_kwargs,
  )
  react_node = SingleAgentReactNode(name='react', agent=llm_agent)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
  )
  runner = Runner(
      app_name='test',
      node=react_node,
      session_service=session_service,
  )

  return [
      e
      async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          new_message=types.Content(
              role='user', parts=[types.Part(text=user_message)]
          ),
      )
  ]


async def _setup_runner(mock_model, tools=None, **agent_kwargs):
  """Create a Runner with SingleAgentReactNode for multi-turn tests."""
  llm_agent = LlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **agent_kwargs,
  )
  react_node = SingleAgentReactNode(name='react', agent=llm_agent)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name='test', user_id=_USER_ID, session_id=_SESSION_ID
  )
  runner = Runner(
      app_name='test',
      node=react_node,
      session_service=session_service,
  )
  return runner, mock_model


async def _run_turn(runner, user_message):
  """Run a single user turn and return events."""
  return [
      e
      async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          new_message=types.Content(
              role='user', parts=[types.Part(text=user_message)]
          ),
      )
  ]


async def _resume(runner, prev_events):
  """Resume after an interrupt, sending FRs for all pending tool IDs."""
  interrupt = next(e for e in prev_events if e.long_running_tool_ids)
  invocation_id = prev_events[0].invocation_id
  fc_ids = list(interrupt.long_running_tool_ids)

  # Build FR parts for each pending long-running tool.
  fr_parts = [
      types.Part(
          function_response=types.FunctionResponse(
              name=f'tool_{i}',
              id=fc_id,
              response={'result': f'done_{i}'},
          )
      )
      for i, fc_id in enumerate(fc_ids)
  ]
  resume_msg = types.Content(role='user', parts=fr_parts)

  return [
      e
      async for e in runner.run_async(
          user_id=_USER_ID,
          session_id=_SESSION_ID,
          invocation_id=invocation_id,
          new_message=resume_msg,
      )
  ]


# ---------------------------------------------------------------------------
# Tests: text response
# ---------------------------------------------------------------------------


class TestBasicTextResponse:

  async def test_pure_text_response(self):
    """LLM returns plain text — loop ends after one iteration."""
    mock_model = testing_utils.MockModel.create(responses=['Hello!'])
    events = await _run(mock_model, 'Hi')

    assert 'Hello!' in text_parts(events)


# ---------------------------------------------------------------------------
# Tests: tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:

  async def test_single_tool_call(self):
    """LLM calls a tool, gets response, then returns text."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    fc = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    mock_model = testing_utils.MockModel.create(responses=[fc, 'Result is 3.'])
    events = await _run(mock_model, 'Add 1 and 2', tools=[add])

    assert {'result': 3} in function_response_dicts(events)
    assert any('Result is 3.' in t for t in text_parts(events))

  async def test_no_function_calls(self):
    """LLM returns text with no function calls — tool not invoked."""
    mock_model = testing_utils.MockModel.create(
        responses=['Just a text reply.']
    )

    def unused_tool() -> str:
      """A tool that should not be called."""
      raise AssertionError('Tool should not be called')

    events = await _run(mock_model, 'Hello', tools=[unused_tool])

    assert 'Just a text reply.' in text_parts(events)
    assert function_response_dicts(events) == []

  async def test_parallel_tool_calls(self):
    """LLM calls multiple tools in parallel."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    fc_add = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    fc_mul = types.Part.from_function_call(
        name='multiply', args={'x': 3, 'y': 4}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[[fc_add, fc_mul], 'Done.']
    )
    events = await _run(mock_model, 'Compute', tools=[add, multiply])

    fr = function_responses_by_name(events)
    assert fr['add'] == {'result': 3}
    assert fr['multiply'] == {'result': 12}

  async def test_multiple_sequential_tool_calls(self):
    """LLM calls tools in sequence across ReAct turns."""
    call_count = 0

    def increment(x: int) -> int:
      """Increment a number."""
      nonlocal call_count
      call_count += 1
      return x + 1

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='increment', args={'x': 1}),
            types.Part.from_function_call(name='increment', args={'x': 2}),
            'done',
        ]
    )
    events = await _run(mock_model, 'Go', tools=[increment])

    assert call_count == 2
    assert function_call_names(events).count('increment') == 2
    assert 'done' in text_parts(events)


# ---------------------------------------------------------------------------
# Tests: termination
# ---------------------------------------------------------------------------


class TestTermination:

  async def test_transfer_to_agent_ends_loop(self):
    """Tool that calls transfer_to_agent — ReAct loop terminates."""

    def transfer_tool(tool_context: ToolContext) -> str:
      tool_context.actions.transfer_to_agent = 'other_agent'
      return 'transferring'

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='transfer_tool', args={}),
        ]
    )
    events = await _run(
        mock_model,
        'Go',
        tools=[transfer_tool],
    )

    # Loop should end after transfer (no second LLM call).
    assert len(mock_model.requests) == 1
    assert 'transfer_tool' in function_call_names(events)


# ---------------------------------------------------------------------------
# Tests: interrupt and resume
# ---------------------------------------------------------------------------


class TestInterruptAndResume:

  async def test_long_running_tool_stops_after_one_llm_call(self):
    """Long-running tool causes the agent to stop with one LLM call."""

    def approve(request: str) -> None:
      """Approve a request (long-running)."""
      return None

    fc = types.Part.from_function_call(
        name='approve', args={'request': 'deploy'}
    )
    mock_model = testing_utils.MockModel.create(responses=[fc])

    events = await _run(
        mock_model,
        'Deploy please',
        tools=[LongRunningFunctionTool(approve)],
    )

    # Only one LLM call — loop stopped on interrupt.
    assert len(mock_model.requests) == 1
    assert any(e.long_running_tool_ids for e in events)

  async def test_resume_after_interrupt_completes_with_one_additional_llm_call(
      self,
  ):
    """Resuming after interrupt calls the LLM once more to complete."""

    def approve(request: str) -> None:
      """Approve a request (long-running)."""
      return None

    fc = types.Part.from_function_call(
        name='approve', args={'request': 'deploy'}
    )
    # First run: FC → interrupt. Second run (resume): text response.
    mock_model = testing_utils.MockModel.create(
        responses=[fc, 'Approved and deployed.']
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[LongRunningFunctionTool(approve)],
    )

    # Run 1: LLM → FC → interrupt
    events1 = await _run_turn(runner, 'Deploy please')
    assert any(e.long_running_tool_ids for e in events1)

    # Run 2: Resume with FR
    events2 = await _resume(runner, events1)

    # Total LLM calls: 1 (first run) + 1 (resume) = 2.
    assert len(mock_model.requests) == 2
    assert any('Approved and deployed.' in t for t in text_parts(events2))

  async def test_multiple_sequential_interrupts(self):
    """Two interrupts in sequence each resume and complete correctly."""

    def step_one() -> None:
      """First long-running step."""
      return None

    def step_two() -> None:
      """Second long-running step."""
      return None

    fc1 = types.Part.from_function_call(name='step_one', args={})
    fc2 = types.Part.from_function_call(name='step_two', args={})
    # Run 1: FC1 → interrupt.
    # Run 2 (resume): LLM → FC2 → interrupt.
    # Run 3 (resume): LLM → text → done.
    mock_model = testing_utils.MockModel.create(
        responses=[fc1, fc2, 'All done.']
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[
            LongRunningFunctionTool(step_one),
            LongRunningFunctionTool(step_two),
        ],
    )

    # Run 1: LLM → FC1 → interrupt
    events1 = await _run_turn(runner, 'Start')
    assert any(e.long_running_tool_ids for e in events1)

    # Run 2: Resume FC1 → LLM → FC2 → interrupt again
    events2 = await _resume(runner, events1)
    assert any(e.long_running_tool_ids for e in events2)
    assert len(mock_model.requests) == 2

    # Run 3: Resume FC2 → LLM → text → done
    events3 = await _resume(runner, events2)

    # Total: 3 LLM calls (one per run).
    assert len(mock_model.requests) == 3
    assert any('All done.' in t for t in text_parts(events3))

  async def test_two_parallel_long_running_tools_interrupt_together(self):
    """LLM calls two long-running tools — both interrupt in one event."""

    def approve() -> None:
      """Approve (long-running)."""
      return None

    def review() -> None:
      """Review (long-running)."""
      return None

    fc_approve = types.Part.from_function_call(name='approve', args={})
    fc_review = types.Part.from_function_call(name='review', args={})
    mock_model = testing_utils.MockModel.create(
        responses=[[fc_approve, fc_review], 'Both approved.']
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[
            LongRunningFunctionTool(approve),
            LongRunningFunctionTool(review),
        ],
    )

    # Run 1: LLM → [approve, review] → interrupts with 1 pending ID each
    events1 = await _run_turn(runner, 'Approve and review')
    interrupts = [e for e in events1 if e.long_running_tool_ids]
    assert len(interrupts) == 2
    assert len(mock_model.requests) == 1

    # Run 2: Resume both FRs → LLM → text → done
    invocation_id = events1[0].invocation_id
    fc_ids = []
    for interrupt in interrupts:
      fc_ids.extend(list(interrupt.long_running_tool_ids))
    resume_msg = types.Content(
        role='user',
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name='approve',
                    id=fc_ids[0],
                    response={'result': 'approved'},
                )
            ),
            types.Part(
                function_response=types.FunctionResponse(
                    name='review',
                    id=fc_ids[1],
                    response={'result': 'reviewed'},
                )
            ),
        ],
    )
    events2 = [
        e
        async for e in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            invocation_id=invocation_id,
            new_message=resume_msg,
        )
    ]

    assert len(mock_model.requests) == 2
    assert any('Both approved.' in t for t in text_parts(events2))

  async def test_mixed_regular_and_long_running_tools(self):
    """Regular tool executes, long-running tool interrupts in same response."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def approve() -> None:
      """Approve (long-running)."""
      return None

    fc_add = types.Part.from_function_call(name='add', args={'x': 1, 'y': 2})
    fc_approve = types.Part.from_function_call(name='approve', args={})
    mock_model = testing_utils.MockModel.create(
        responses=[[fc_add, fc_approve], 'Sum is 3, approved.']
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[add, LongRunningFunctionTool(approve)],
    )

    # Run 1: LLM → [add, approve] → add executes, approve interrupts
    events1 = await _run_turn(runner, 'Add and approve')
    assert any(e.long_running_tool_ids for e in events1)
    # add should have been executed
    assert {'result': 3} in function_response_dicts(events1)
    assert len(mock_model.requests) == 1

    # Run 2: Resume approve → LLM → text → done
    events2 = await _resume(runner, events1)
    assert len(mock_model.requests) == 2
    assert any('Sum is 3, approved.' in t for t in text_parts(events2))

  async def test_regular_tool_call_after_resume(self):
    """After resume, LLM calls a regular tool then returns text."""

    def approve() -> None:
      """Approve (long-running)."""
      return None

    def summarize() -> str:
      """Summarize the results."""
      return 'summary'

    fc_approve = types.Part.from_function_call(name='approve', args={})
    fc_summarize = types.Part.from_function_call(name='summarize', args={})
    # Run 1: approve → interrupt.
    # Run 2 (resume): LLM → summarize → FR → LLM → text.
    mock_model = testing_utils.MockModel.create(
        responses=[fc_approve, fc_summarize, 'Final answer.']
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[LongRunningFunctionTool(approve), summarize],
    )

    # Run 1: LLM → approve → interrupt
    events1 = await _run_turn(runner, 'Go')
    assert any(e.long_running_tool_ids for e in events1)

    # Run 2: Resume → LLM → summarize (regular) → LLM → text
    events2 = await _resume(runner, events1)
    assert {'result': 'summary'} in function_response_dicts(events2)
    # 1 (first run) + 2 (resume: LLM→tool→LLM) = 3 total.
    assert len(mock_model.requests) == 3
    assert any('Final answer.' in t for t in text_parts(events2))

  async def test_transfer_after_resume_ends_loop(self):
    """After resume, a transfer_to_agent call terminates the loop."""

    def approve() -> None:
      """Approve (long-running)."""
      return None

    def hand_off(tool_context: ToolContext) -> str:
      """Transfer to another agent."""
      tool_context.actions.transfer_to_agent = 'other_agent'
      return 'handing off'

    fc_approve = types.Part.from_function_call(name='approve', args={})
    fc_handoff = types.Part.from_function_call(name='hand_off', args={})
    # Run 1: approve → interrupt.
    # Run 2 (resume): LLM → hand_off → transfer terminates loop.
    mock_model = testing_utils.MockModel.create(
        responses=[fc_approve, fc_handoff]
    )

    runner, mock_model = await _setup_runner(
        mock_model,
        tools=[LongRunningFunctionTool(approve), hand_off],
    )

    # Run 1: LLM → approve → interrupt
    events1 = await _run_turn(runner, 'Go')
    assert any(e.long_running_tool_ids for e in events1)

    # Run 2: Resume → LLM → hand_off → transfer ends loop
    events2 = await _resume(runner, events1)
    assert len(mock_model.requests) == 2
    assert 'hand_off' in function_call_names(events2)
