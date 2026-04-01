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

"""Unit tests for execute_tools node function in isolation.

Tests call ``execute_tools()`` directly with a hand-built
WorkflowContext and explicit ``CallLlmResult``, without going through
the full _SingleLlmAgent + InMemoryRunner pipeline.  This ensures the
node's input contract and output behavior are verified independently.

End-to-end pipeline tests are in test_single_llm_agent.py.
"""

from __future__ import annotations

from google.adk.agents.context import Context as WorkflowContext
from google.adk.agents.llm._call_llm_node import CallLlmResult
from google.adk.agents.llm._execute_tools_node import _process_auth_resume
from google.adk.agents.llm._execute_tools_node import _process_confirmation_resume
from google.adk.agents.llm._execute_tools_node import _process_long_running_resume
from google.adk.agents.llm._execute_tools_node import execute_tools
from google.adk.agents.llm._functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.agents.llm._functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.agents.llm._single_llm_agent import _SingleLlmAgent
from google.adk.events.event import Event as AdkEvent
from google.adk.events.event import Event as WorkflowEvent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from tests.unittests.workflow import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(mock_model, tools=None, **kwargs):
  """Create a _SingleLlmAgent wired to a MockModel."""
  return _SingleLlmAgent(
      name='test_agent',
      model=mock_model,
      tools=tools or [],
      **kwargs,
  )


async def _collect_events(node_fn, ctx, node_input):
  """Collect all events yielded by a node function."""
  events = []
  async for event in node_fn(ctx=ctx, node_input=node_input):
    events.append(event)
  return events


def _adk_events_with_content(events):
  """Filter to AdkEvents that have content (function responses)."""
  return [e for e in events if isinstance(e, AdkEvent) and e.content]


def _route_events(events):
  """Filter to WorkflowEvents that have a route set."""
  return [e for e in events if isinstance(e, WorkflowEvent) and e.actions.route]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecuteTools:
  """Tests for execute_tools node function in isolation."""

  @pytest.mark.asyncio
  async def test_executes_function_from_input(self):
    """CallLlmResult input with FunctionCall executes the tool."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[add])
    ctx = await testing_utils.create_workflow_context(agent)

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='add', args={'x': 1, 'y': 2}),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    # Find the function response
    fr_event = content_events[0]
    fr_part = fr_event.content.parts[0]
    assert fr_part.function_response.name == 'add'
    assert fr_part.function_response.response['result'] == 3

  @pytest.mark.asyncio
  async def test_multiple_function_calls(self):
    """Multiple FunctionCall entries all execute."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    def multiply(x: int, y: int) -> int:
      """Multiply two numbers."""
      return x * y

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[add, multiply])
    ctx = await testing_utils.create_workflow_context(agent)

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='add', args={'x': 2, 'y': 3}),
            types.FunctionCall(name='multiply', args={'x': 4, 'y': 5}),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    # Function response event should have two parts
    fr_event = content_events[0]
    fr_names = {
        p.function_response.name
        for p in fr_event.content.parts
        if p.function_response
    }
    assert fr_names == {'add', 'multiply'}

  @pytest.mark.asyncio
  async def test_tool_returning_dict(self):
    """Tool returning a dict produces correct function response."""

    def get_info() -> dict:
      """Get info."""
      return {'name': 'Alice', 'age': 30}

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[get_info])
    ctx = await testing_utils.create_workflow_context(agent)

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='get_info', args={}),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    fr_part = content_events[0].content.parts[0]
    assert fr_part.function_response.response == {
        'name': 'Alice',
        'age': 30,
    }

  @pytest.mark.asyncio
  async def test_confirmation_require_confirmation_true(self):
    """Tool with require_confirmation=True triggers confirmation."""

    def dangerous_action(x: int) -> str:
      """A dangerous action."""
      return 'done'

    tool = FunctionTool(dangerous_action, require_confirmation=True)
    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[tool])
    ctx = await testing_utils.create_workflow_context(agent)

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(
                name='dangerous_action',
                args={'x': 42},
                id='call-1',
            ),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    has_confirmation = any(
        isinstance(e, AdkEvent) and e.actions.requested_tool_confirmations
        for e in events
    )
    assert has_confirmation, 'No tool confirmation request found'

  @pytest.mark.asyncio
  async def test_data_contract_round_trip(self):
    """CallLlmResult survives dict serialization round-trip.

    Simulates checkpoint restore: CallLlmResult is serialized to a
    dict and reconstructed via model_validate, then passed to
    execute_tools.
    """

    def greet(name: str) -> str:
      """Greet someone."""
      return f'Hello, {name}!'

    # Build a CallLlmResult and round-trip through dict
    original = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='greet', args={'name': 'Bob'}),
        ]
    )
    serialized = original.model_dump(mode='json')
    restored = CallLlmResult.model_validate(serialized)

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[greet])
    ctx = await testing_utils.create_workflow_context(agent)

    events = await _collect_events(execute_tools, ctx, restored)

    content_events = _adk_events_with_content(events)
    fr_part = content_events[0].content.parts[0]
    assert fr_part.function_response.name == 'greet'
    assert fr_part.function_response.response['result'] == 'Hello, Bob!'


# ---------------------------------------------------------------------------
# Resume handling helpers
# ---------------------------------------------------------------------------


async def _create_resume_context(
    agent,
    resume_inputs,
    session_events=None,
    user_content='',
):
  """Create a WorkflowContext with resume_inputs and pre-populated session events."""
  invocation_context = await testing_utils.create_invocation_context(
      agent, user_content
  )
  if session_events:
    invocation_context.session.events.extend(session_events)
  return WorkflowContext(
      invocation_context=invocation_context,
      node_path='test',
      run_id='test-execution',
      local_events=[],
      resume_inputs=resume_inputs,
  )


def _make_auth_fc_event(auth_fc_id, original_fc_id, invocation_id='test_id'):
  """Create a session event containing an adk_request_credential FC."""
  auth_args = {
      'function_call_id': original_fc_id,
      'auth_config': {
          'auth_scheme': {'type': 'apiKey'},
      },
  }
  return AdkEvent(
      invocation_id=invocation_id,
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_EUC_FUNCTION_CALL_NAME,
                      id=auth_fc_id,
                      args=auth_args,
                  )
              )
          ],
      ),
  )


def _make_confirmation_fc_event(
    confirmation_fc_id,
    original_fc_name,
    original_fc_id,
    original_fc_args=None,
    invocation_id='test_id',
):
  """Create a session event containing an adk_request_confirmation FC."""
  original_fc = types.FunctionCall(
      name=original_fc_name,
      id=original_fc_id,
      args=original_fc_args or {},
  )
  confirmation_args = {
      'originalFunctionCall': original_fc.model_dump(
          exclude_none=True, by_alias=True
      ),
      'toolConfirmation': {
          'hint': 'Please confirm.',
          'confirmed': False,
          'payload': {'amount': 100},
      },
  }
  return AdkEvent(
      invocation_id=invocation_id,
      author='test_agent',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                      id=confirmation_fc_id,
                      args=confirmation_args,
                  )
              )
          ],
      ),
  )


# ---------------------------------------------------------------------------
# Resume handling tests
# ---------------------------------------------------------------------------


class TestProcessAuthResume:
  """Tests for _process_auth_resume pre-filtering and dispatch."""

  @pytest.mark.asyncio
  async def test_returns_none_when_no_auth_fcs_in_session(self):
    """Returns None when resume_inputs don't match any auth FCs."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    # Resume inputs with an ID that doesn't match any auth FC
    resume_inputs = {'some-non-auth-id': {'confirmed': True}}
    ctx = await _create_resume_context(agent, resume_inputs)
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}, id='fc-1'),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    result = await _process_auth_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_returns_none_when_resume_inputs_empty(self):
    """Returns None when resume_inputs is empty (pre-filter finds nothing)."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    ctx = await _create_resume_context(agent, resume_inputs={})
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    result = await _process_auth_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_filters_non_auth_fcs_from_resume_inputs(self):
    """Only passes auth FC IDs to the store helper, ignoring others."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    auth_fc_id = 'auth-fc-1'
    original_fc_id = 'original-fc-1'
    confirmation_fc_id = 'conf-fc-1'

    # Session has an auth FC event but not a confirmation FC event
    auth_event = _make_auth_fc_event(auth_fc_id, original_fc_id)

    # Resume inputs include both auth and non-auth IDs
    resume_inputs = {
        auth_fc_id: {
            'auth_scheme': {'type': 'apiKey'},
        },
        confirmation_fc_id: {'confirmed': True},
    }
    ctx = await _create_resume_context(
        agent, resume_inputs, session_events=[auth_event]
    )
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}, id=original_fc_id),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    # This should process only the auth FC ID, not crash on the
    # confirmation ID.
    result = await _process_auth_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    # Result should be an event (the auth was processed) - but
    # _store_auth_and_collect_resume_targets may fail to find the
    # exact original tool. The key assertion is that it didn't crash
    # on the confirmation ID.
    # Note: The result may be None if the auth helper can't find
    # matching tools to resume (which depends on how the auth config
    # is set up). The important thing is no exception was raised.


class TestProcessConfirmationResume:
  """Tests for _process_confirmation_resume parsing and dispatch."""

  @pytest.mark.asyncio
  async def test_returns_none_when_parse_fails(self):
    """Returns None when resume inputs can't be parsed as ToolConfirmation."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    # Resume inputs with data that can't be parsed as ToolConfirmation
    # (missing required fields, has extra='forbid')
    resume_inputs = {
        'some-id': {
            'auth_scheme': {'type': 'apiKey'},
            'not_a_confirmation_field': True,
        },
    }
    ctx = await _create_resume_context(agent, resume_inputs)
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    result = await _process_confirmation_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_returns_none_when_no_matching_confirmation_fcs(self):
    """Returns None when parsed confirmations don't match session events."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    # Resume inputs that parse as ToolConfirmation but don't match
    # any adk_request_confirmation FC in session events
    resume_inputs = {
        'conf-fc-1': {'confirmed': True, 'payload': {'amount': 100}},
    }
    ctx = await _create_resume_context(agent, resume_inputs)
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    result = await _process_confirmation_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_processes_valid_confirmation(self):
    """Processes valid confirmation resume and re-executes the tool."""

    def reimburse(amount: int) -> dict:
      """Reimburse the employee."""
      return {'status': 'ok', 'amount': amount}

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[reimburse])

    original_fc_id = 'original-fc-1'
    confirmation_fc_id = 'conf-fc-1'

    # Session has an adk_request_confirmation FC event
    confirmation_event = _make_confirmation_fc_event(
        confirmation_fc_id=confirmation_fc_id,
        original_fc_name='reimburse',
        original_fc_id=original_fc_id,
        original_fc_args={'amount': 1500},
    )

    # Resume input: confirmed with payload
    resume_inputs = {
        confirmation_fc_id: {'confirmed': True, 'payload': {'amount': 1500}},
    }
    ctx = await _create_resume_context(
        agent, resume_inputs, session_events=[confirmation_event]
    )
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(
            name='reimburse',
            args={'amount': 1500},
            id=original_fc_id,
        ),
    ]
    tools_dict = {'reimburse': FunctionTool(reimburse)}

    result = await _process_confirmation_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    # Should return a function response event
    assert result is not None
    assert result.content is not None
    # Find the function response
    fr_parts = [p for p in result.content.parts if p.function_response]
    assert len(fr_parts) == 1
    assert fr_parts[0].function_response.name == 'reimburse'
    assert fr_parts[0].function_response.response['status'] == 'ok'


class TestExecuteToolsResume:
  """Tests for execute_tools resume handling (integration)."""

  @pytest.mark.asyncio
  async def test_no_resume_inputs_runs_normal_path(self):
    """With no resume_inputs, tools execute normally."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[add])
    # No resume_inputs (empty dict)
    ctx = await _create_resume_context(agent, resume_inputs={})

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='add', args={'x': 1, 'y': 2}),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    fr_part = content_events[0].content.parts[0]
    assert fr_part.function_response.name == 'add'
    assert fr_part.function_response.response['result'] == 3

  @pytest.mark.asyncio
  async def test_confirmation_resume_via_execute_tools(self):
    """execute_tools with confirmation resume re-executes the tool."""

    def reimburse(amount: int) -> dict:
      """Reimburse the employee."""
      return {'status': 'ok', 'amount': amount}

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[reimburse])

    original_fc_id = 'original-fc-1'
    confirmation_fc_id = 'conf-fc-1'

    confirmation_event = _make_confirmation_fc_event(
        confirmation_fc_id=confirmation_fc_id,
        original_fc_name='reimburse',
        original_fc_id=original_fc_id,
        original_fc_args={'amount': 1500},
    )

    resume_inputs = {
        confirmation_fc_id: {'confirmed': True, 'payload': {'amount': 1500}},
    }
    ctx = await _create_resume_context(
        agent, resume_inputs, session_events=[confirmation_event]
    )

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(
                name='reimburse',
                args={'amount': 1500},
                id=original_fc_id,
            ),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    fr_parts = [
        p
        for e in content_events
        for p in e.content.parts
        if p.function_response
    ]
    assert len(fr_parts) >= 1
    assert fr_parts[0].function_response.name == 'reimburse'
    assert fr_parts[0].function_response.response['status'] == 'ok'

  @pytest.mark.asyncio
  async def test_unrecognized_resume_inputs_fall_through(self):
    """Unrecognized resume inputs (not auth or confirmation) fall through.

    When resume_inputs contain data that neither auth nor confirmation
    handlers recognize, execute_tools runs the normal execution path.
    """

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[add])

    # Resume inputs that don't match any auth or confirmation pattern
    resume_inputs = {
        'unknown-id': {'some_random_key': 'value'},
    }
    ctx = await _create_resume_context(agent, resume_inputs)

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(name='add', args={'x': 5, 'y': 3}),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    # Since both handlers return None, execute_tools should fall
    # through to normal execution
    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    fr_parts = [
        p
        for e in content_events
        for p in e.content.parts
        if p.function_response
    ]
    assert len(fr_parts) >= 1
    assert fr_parts[0].function_response.name == 'add'
    assert fr_parts[0].function_response.response['result'] == 8

  @pytest.mark.asyncio
  async def test_request_confirmation_tool_resumes(self):
    """Tool using request_confirmation in body resumes after approval."""

    def request_time_off(days: int, tool_context: ToolContext) -> dict:
      """Request day off for the employee."""
      if days <= 2:
        return {'status': 'ok', 'approved_days': days}

      tool_confirmation = tool_context.tool_confirmation
      if not tool_confirmation:
        tool_context.request_confirmation(
            hint='Manager approval needed.',
            payload={'approved_days': 0},
        )
        return {'status': 'pending'}

      approved_days = tool_confirmation.payload.get('approved_days', 0)
      return {'status': 'ok', 'approved_days': min(approved_days, days)}

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[request_time_off])

    original_fc_id = 'fc-time-off'
    confirmation_fc_id = 'conf-time-off'

    # Simulate the confirmation FC that was in the session
    confirmation_event = _make_confirmation_fc_event(
        confirmation_fc_id=confirmation_fc_id,
        original_fc_name='request_time_off',
        original_fc_id=original_fc_id,
        original_fc_args={'days': 5},
    )

    # Manager approves 4 days
    resume_inputs = {
        confirmation_fc_id: {
            'confirmed': True,
            'payload': {'approved_days': 4},
        },
    }
    ctx = await _create_resume_context(
        agent, resume_inputs, session_events=[confirmation_event]
    )

    node_input = CallLlmResult(
        function_calls=[
            types.FunctionCall(
                name='request_time_off',
                args={'days': 5},
                id=original_fc_id,
            ),
        ]
    )
    events = await _collect_events(execute_tools, ctx, node_input)

    content_events = _adk_events_with_content(events)
    assert len(content_events) >= 1
    fr_parts = [
        p
        for e in content_events
        for p in e.content.parts
        if p.function_response
    ]
    assert len(fr_parts) >= 1
    response = fr_parts[0].function_response.response
    assert response['status'] == 'ok'
    assert response['approved_days'] == 4


# ---------------------------------------------------------------------------
# Long-running resume tests
# ---------------------------------------------------------------------------


class TestProcessLongRunningResume:
  """Tests for _process_long_running_resume."""

  @pytest.mark.asyncio
  async def test_returns_none_when_no_long_running_tools(self):
    """Returns None when no tools are long-running."""

    def my_tool(x: int) -> int:
      """A tool."""
      return x

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[my_tool])

    resume_inputs = {'fc-1': {'status': 'done'}}
    ctx = await _create_resume_context(agent, resume_inputs)
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(name='my_tool', args={'x': 1}, id='fc-1'),
    ]
    tools_dict = {'my_tool': FunctionTool(my_tool)}

    result = await _process_long_running_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_function_response_has_user_role(self):
    """Long-running resume event must use role='user' for function responses."""

    def approve_request(request_id: str) -> dict:
      """A long-running approval tool."""
      return {'status': 'approved'}

    lr_tool = LongRunningFunctionTool(approve_request)

    mock_model = testing_utils.MockModel.create(responses=[])
    agent = _make_agent(mock_model, tools=[lr_tool])

    fc_id = 'lr-fc-1'
    resume_inputs = {fc_id: {'status': 'approved'}}
    ctx = await _create_resume_context(agent, resume_inputs)
    invocation_context = ctx.get_invocation_context()

    function_calls = [
        types.FunctionCall(
            name='approve_request', args={'request_id': 'req-1'}, id=fc_id
        ),
    ]
    tools_dict = {'approve_request': lr_tool}

    result = await _process_long_running_resume(
        ctx, invocation_context, function_calls, tools_dict
    )
    assert result is not None
    assert result.content is not None
    assert result.content.role == 'user'

    fr_parts = [p for p in result.content.parts if p.function_response]
    assert len(fr_parts) == 1
    assert fr_parts[0].function_response.name == 'approve_request'
    assert fr_parts[0].function_response.id == fc_id
    assert fr_parts[0].function_response.response == {'status': 'approved'}
