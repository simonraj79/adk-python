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

"""Tests for HITL flows with different agent structures."""

import copy
from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm._functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentResponse
from google.genai.types import Part
import pytest

from .. import testing_utils

HINT_TEXT = (
    "Please approve or reject the tool call _test_function() by"
    " responding with a FunctionResponse with an"
    " expected ToolConfirmation payload."
)

TOOL_CALL_ERROR_RESPONSE = {
    "error": "This tool call requires confirmation, please approve or reject."
}


def _find_confirmation_event(events):
  """Find the event containing the adk_request_confirmation function call."""
  return next(
      e
      for e in events
      if e.content
      and e.content.parts
      and e.content.parts[0].function_call
      and e.content.parts[0].function_call.name
      == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
  )


def _create_llm_response_from_tools(
    tools: list[FunctionTool],
) -> GenerateContentResponse:
  """Creates a mock LLM response containing a function call."""
  parts = [
      Part(function_call=FunctionCall(name=tool.name, args={}))
      for tool in tools
  ]
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=parts)
  )


def _create_llm_response_from_text(text: str) -> GenerateContentResponse:
  """Creates a mock LLM response containing text."""
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=[Part(text=text)])
  )


def _test_function(
    tool_context: ToolContext,
) -> dict[str, str]:
  return {"result": f"confirmed={tool_context.tool_confirmation.confirmed}"}


def _test_request_confirmation_function_with_custom_schema(
    tool_context: ToolContext,
) -> dict[str, str]:
  """A test tool function that requests confirmation, but with a custom payload schema."""
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(
        hint="test hint for request_confirmation with custom payload schema",
        payload={
            "test_custom_payload": {
                "int_field": 0,
                "str_field": "",
                "bool_field": False,
            }
        },
    )
    return TOOL_CALL_ERROR_RESPONSE
  return {
      "result": f"confirmed={tool_context.tool_confirmation.confirmed}",
      "custom_payload": tool_context.tool_confirmation.payload,
  }


class BaseHITLTest:
  """Base class for HITL tests with common fixtures."""

  @pytest.fixture
  def runner(self, agent: BaseAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    return testing_utils.InMemoryRunner(root_agent=agent)


class TestHITLConfirmationFlowWithSingleAgent(BaseHITLTest):
  """Tests the HITL confirmation flow with a single LlmAgent."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)
    tools = agent.tools

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    confirmation_event = _find_confirmation_event(events)
    ask_for_confirmation_function_call_id = confirmation_event.content.parts[
        0
    ].function_call.id
    invocation_id = confirmation_event.invocation_id
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": tool_call_confirmed},
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response={"result": f"confirmed={tool_call_confirmed}"}
                    if tool_call_confirmed
                    else {"error": "This tool call is rejected."},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]
    for event in events:
      assert event.invocation_id != invocation_id
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithCustomPayloadSchema(BaseHITLTest):
  """Tests the HITL confirmation flow with a single agent, for custom confirmation payload schema."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [
        FunctionTool(
            func=_test_request_confirmation_function_with_custom_schema
        )
    ]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text(
            "test llm response after final tool call"
        ),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.mark.asyncio
  @pytest.mark.parametrize("tool_call_confirmed", [True, False])
  async def test_confirmation_flow(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
      tool_call_confirmed: bool,
  ):
    """Tests HITL flow with custom payload schema."""
    tools = agent.tools
    user_query = testing_utils.UserContent("test user query")
    events = await runner.run_async(user_query)

    expected_parts = [
        (
            agent.name,
            Part(function_call=FunctionCall(name=tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": (
                                "test hint for request_confirmation with"
                                " custom payload schema"
                            ),
                            "confirmed": False,
                            "payload": {
                                "test_custom_payload": {
                                    "int_field": 0,
                                    "str_field": "",
                                    "bool_field": False,
                                }
                            },
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]

    simplified = testing_utils.simplify_events(copy.deepcopy(events))
    for i, (agent_name, part) in enumerate(expected_parts):
      assert simplified[i][0] == agent_name
      assert simplified[i][1] == part

    confirmation_event = _find_confirmation_event(events)
    ask_for_confirmation_function_call_id = confirmation_event.content.parts[
        0
    ].function_call.id
    invocation_id = confirmation_event.invocation_id
    custom_payload = {
        "test_custom_payload": {
            "int_field": 123,
            "str_field": "test_str",
            "bool_field": True,
        }
    }
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={
                    "confirmed": tool_call_confirmed,
                    "payload": custom_payload,
                },
            )
        )
    )
    events = await runner.run_async(user_confirmation)

    expected_response = {
        "result": f"confirmed={tool_call_confirmed}",
        "custom_payload": custom_payload,
    }
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=tools[0].name,
                    response=expected_response,
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
    ]
    for event in events:
      assert event.invocation_id != invocation_id
    assert (
        testing_utils.simplify_events(copy.deepcopy(events))
        == expected_parts_final
    )


class TestHITLConfirmationFlowWithResumableApp:
  """Tests the HITL confirmation flow with a resumable app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> LlmAgent:
    """Provides a single LlmAgent for the test."""
    return LlmAgent(name="root_agent", model=mock_model, tools=tools)

  @pytest.fixture
  def runner(self, agent: LlmAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = runner.run("test user query")

    # Verify that the invocation is paused when tool confirmation is requested.
    # The tool call returns error response, and summarization was skipped.
    behavioral = [
        e
        for e in testing_utils.simplify_resumable_app_events(
            copy.deepcopy(events)
        )
        if not isinstance(e[1], dict)
    ]
    assert behavioral == [
        (
            agent.name,
            Part(function_call=FunctionCall(name=agent.tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": agent.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]
    confirmation_event = _find_confirmation_event(events)
    ask_for_confirmation_function_call_id = confirmation_event.content.parts[
        0
    ].function_call.id
    invocation_id = confirmation_event.invocation_id
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events = await runner.run_async(
        user_confirmation, invocation_id=invocation_id
    )
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
    for event in events:
      assert event.invocation_id == invocation_id
    behavioral = [
        e
        for e in testing_utils.simplify_resumable_app_events(
            copy.deepcopy(events)
        )
        if not isinstance(e[1], dict)
    ]
    assert behavioral == expected_parts_final

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation_without_invocation_id(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: LlmAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = runner.run("test user query")

    # Verify that the invocation is paused when tool confirmation is requested.
    # The tool call returns error response, and summarization was skipped.
    behavioral = [
        e
        for e in testing_utils.simplify_resumable_app_events(
            copy.deepcopy(events)
        )
        if not isinstance(e[1], dict)
    ]
    assert behavioral == [
        (
            agent.name,
            Part(function_call=FunctionCall(name=agent.tools[0].name, args={})),
        ),
        (
            agent.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": agent.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name, response=TOOL_CALL_ERROR_RESPONSE
                )
            ),
        ),
    ]
    confirmation_event = _find_confirmation_event(events)
    ask_for_confirmation_function_call_id = confirmation_event.content.parts[
        0
    ].function_call.id
    invocation_id = confirmation_event.invocation_id
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events = await runner.run_async(user_confirmation)
    expected_parts_final = [
        (
            agent.name,
            Part(
                function_response=FunctionResponse(
                    name=agent.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        (agent.name, "test llm response after tool call"),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
    for event in events:
      assert event.invocation_id == invocation_id
    behavioral = [
        e
        for e in testing_utils.simplify_resumable_app_events(
            copy.deepcopy(events)
        )
        if not isinstance(e[1], dict)
    ]
    assert behavioral == expected_parts_final


class TestHITLConfirmationFlowWithSequentialAgentAndResumableApp:
  """Tests the HITL confirmation flow with a resumable sequential agent app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
        _create_llm_response_from_text("test llm response from second agent"),
    ]

  @pytest.fixture
  def mock_model(
      self, llm_responses: list[GenerateContentResponse]
  ) -> testing_utils.MockModel:
    """Provides a mock model with predefined responses."""
    return testing_utils.MockModel(responses=llm_responses)

  @pytest.fixture
  def agent(
      self, mock_model: testing_utils.MockModel, tools: list[FunctionTool]
  ) -> SequentialAgent:
    """Provides a single LlmAgent for the test."""
    return SequentialAgent(
        name="root_agent",
        sub_agents=[
            LlmAgent(name="agent1", model=mock_model, tools=tools),
            LlmAgent(name="agent2", model=mock_model, tools=[]),
        ],
    )

  @pytest.fixture
  def runner(self, agent: SequentialAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: SequentialAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""

    # Test setup:
    # - root_agent is a SequentialAgent with two sub-agents: sub_agent1 and
    #   sub_agent2.
    #   - sub_agent1 has a tool call that asks for HITL confirmation.
    #   - sub_agent2 does not have any tool calls.
    # - The test will:
    #   - Run the query and verify that the invocation is paused when tool
    #     confirmation is requested, at sub_agent1.
    #   - Resume the invocation and execute the tool call from sub_agent1.
    #   - Verify that root_agent continues to run sub_agent2.

    events = runner.run("test user query")
    sub_agent1 = agent.sub_agents[0]
    sub_agent2 = agent.sub_agents[1]

    # Step 1:
    # Verify that the invocation is paused when tool confirmation is requested.
    # So that no intermediate llm response is generated.
    # And the second sub agent is not started.
    actual = testing_utils.simplify_resumable_app_events(copy.deepcopy(events))
    behavioral = [e for e in actual if not isinstance(e[1], dict)]
    assert behavioral == [
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=sub_agent1.tools[0].name, args={}
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_call=FunctionCall(
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    args={
                        "originalFunctionCall": {
                            "name": sub_agent1.tools[0].name,
                            "id": mock.ANY,
                            "args": {},
                        },
                        "toolConfirmation": {
                            "hint": HINT_TEXT,
                            "confirmed": False,
                        },
                    },
                )
            ),
        ),
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response=TOOL_CALL_ERROR_RESPONSE,
                )
            ),
        ),
    ]
    confirmation_event = next(
        e
        for e in events
        if e.content
        and e.content.parts
        and e.content.parts[0].function_call
        and e.content.parts[0].function_call.name
        == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    )
    ask_for_confirmation_function_call_id = confirmation_event.content.parts[
        0
    ].function_call.id
    invocation_id = confirmation_event.invocation_id

    # Step 2:
    # Resume the invocation and confirm the tool call from sub_agent1, and
    # sub_agent2 will continue.
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )
    events = await runner.run_async(
        user_confirmation, invocation_id=invocation_id
    )
    expected_behavioral = [
        (
            sub_agent1.name,
            Part(
                function_response=FunctionResponse(
                    name=sub_agent1.tools[0].name,
                    response={"result": "confirmed=True"},
                )
            ),
        ),
        # Single_turn: content stripped, output set on event.
        (sub_agent1.name, "test llm response after tool call"),
        # LlmAgent re-emits output for node routing.
        (agent.name, "test llm response after tool call"),
        (sub_agent1.name, testing_utils.END_OF_AGENT),
        (sub_agent2.name, "test llm response from second agent"),
        (agent.name, "test llm response from second agent"),
        (sub_agent2.name, testing_utils.END_OF_AGENT),
        (agent.name, testing_utils.END_OF_AGENT),
    ]
    for event in events:
      assert event.invocation_id == invocation_id
    actual = testing_utils.simplify_resumable_app_events(copy.deepcopy(events))
    behavioral = [e for e in actual if not isinstance(e[1], dict)]
    assert behavioral == expected_behavioral


class TestHITLConfirmationFlowWithParallelAgentAndResumableApp:
  """Tests the HITL confirmation flow with a resumable sequential agent app."""

  @pytest.fixture
  def tools(self) -> list[FunctionTool]:
    """Provides the tools for the agent."""
    return [FunctionTool(func=_test_function, require_confirmation=True)]

  @pytest.fixture
  def llm_responses(
      self, tools: list[FunctionTool]
  ) -> list[GenerateContentResponse]:
    """Provides mock LLM responses for the tests."""
    return [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]

  @pytest.fixture
  def agent(
      self,
      tools: list[FunctionTool],
      llm_responses: list[GenerateContentResponse],
  ) -> ParallelAgent:
    """Provides a single ParallelAgent for the test."""
    return ParallelAgent(
        name="root_agent",
        sub_agents=[
            LlmAgent(
                name="agent1",
                model=testing_utils.MockModel(responses=llm_responses),
                tools=tools,
            ),
            LlmAgent(
                name="agent2",
                model=testing_utils.MockModel(responses=llm_responses),
                tools=tools,
            ),
        ],
    )

  @pytest.fixture
  def runner(self, agent: ParallelAgent) -> testing_utils.InMemoryRunner:
    """Provides an in-memory runner for the agent."""
    # Mark the app as resumable. So that the invocation will be paused when
    # tool confirmation is requested.
    app = App(
        name="test_app",
        resumability_config=ResumabilityConfig(is_resumable=True),
        root_agent=agent,
    )
    return testing_utils.InMemoryRunner(app=app)

  @pytest.mark.asyncio
  async def test_pause_and_resume_on_request_confirmation(
      self,
      runner: testing_utils.InMemoryRunner,
      agent: ParallelAgent,
  ):
    """Tests HITL flow where all tool calls are confirmed."""
    events = runner.run("test user query")

    sub_agent1 = agent.sub_agents[0]
    sub_agent2 = agent.sub_agents[1]

    # Step 1: Verify both sub-agents paused with confirmation requests.
    simplified = testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    )
    behavioral = [e for e in simplified if not isinstance(e[1], dict)]

    # Both agents should have emitted tool calls and error responses.
    tool_call_events = [
        e
        for e in behavioral
        if isinstance(e[1], Part)
        and e[1].function_call
        and e[1].function_call.name == sub_agent1.tools[0].name
    ]
    assert len(tool_call_events) == 2
    confirmation_events = [
        e
        for e in behavioral
        if isinstance(e[1], Part)
        and e[1].function_call
        and e[1].function_call.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    ]
    assert len(confirmation_events) == 2

    # Find confirmation function call IDs and invocation ID.
    confirmation_raw_events = [
        e
        for e in events
        if e.content
        and e.content.parts
        and e.content.parts[0].function_call
        and e.content.parts[0].function_call.name
        == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    ]
    assert len(confirmation_raw_events) == 2
    ask_for_confirmation_function_call_ids = [
        e.content.parts[0].function_call.id for e in confirmation_raw_events
    ]
    invocation_id = confirmation_raw_events[0].invocation_id

    user_confirmations = [
        testing_utils.UserContent(
            Part(
                function_response=FunctionResponse(
                    id=id,
                    name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                    response={"confirmed": True},
                )
            )
        )
        for id in ask_for_confirmation_function_call_ids
    ]

    # Step 2: Resume with first confirmation.
    events = await runner.run_async(
        user_confirmations[0], invocation_id=invocation_id
    )
    for event in events:
      assert event.invocation_id == invocation_id

    simplified = testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    )
    behavioral = [e for e in simplified if not isinstance(e[1], dict)]
    # First confirmed agent should produce tool response + text + END_OF_AGENT.
    confirmed_responses = [
        e
        for e in behavioral
        if isinstance(e[1], Part)
        and e[1].function_response
        and e[1].function_response.response == {"result": "confirmed=True"}
    ]
    assert len(confirmed_responses) == 1
    # Root agent should NOT be final yet.
    assert (agent.name, testing_utils.END_OF_AGENT) not in behavioral

    # Step 3: Resume with second confirmation.
    events = await runner.run_async(
        user_confirmations[1], invocation_id=invocation_id
    )
    for event in events:
      assert event.invocation_id == invocation_id

    simplified = testing_utils.simplify_resumable_app_events(
        copy.deepcopy(events)
    )
    behavioral = [e for e in simplified if not isinstance(e[1], dict)]
    # Second confirmed agent should produce tool response + text + END_OF_AGENT.
    confirmed_responses = [
        e
        for e in behavioral
        if isinstance(e[1], Part)
        and e[1].function_response
        and e[1].function_response.response == {"result": "confirmed=True"}
    ]
    assert len(confirmed_responses) == 1
    # Root agent should be final now.
    assert (agent.name, testing_utils.END_OF_AGENT) in behavioral
