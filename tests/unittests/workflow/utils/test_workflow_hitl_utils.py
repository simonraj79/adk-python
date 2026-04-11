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

from __future__ import annotations

from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.genai import types
from google.adk.workflow.utils._workflow_hitl_utils import _ChildScanState
from google.adk.workflow.utils._workflow_hitl_utils import create_auth_request_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from google.adk.workflow.utils._workflow_hitl_utils import _scan_node_events
from google.adk.workflow.utils._workflow_hitl_utils import unwrap_response
from google.adk.workflow.utils._workflow_hitl_utils import validate_resume_response
from google.adk.workflow.utils._workflow_hitl_utils import wrap_response

# --- wrap_response ---


class TestWrapResponse:

  def test_dict_returned_as_is(self):
    d = {"foo": "bar"}
    assert wrap_response(d) is d

  def test_string_wrapped(self):
    assert wrap_response("hello") == {"result": "hello"}

  def test_int_wrapped(self):
    assert wrap_response(42) == {"result": 42}

  def test_none_wrapped(self):
    assert wrap_response(None) == {"result": None}

  def test_list_wrapped(self):
    assert wrap_response([1, 2]) == {"result": [1, 2]}


# --- unwrap_response ---


class TestUnwrapResponse:

  def test_single_result_key_string(self):
    assert unwrap_response({"result": "hello"}) == "hello"

  def test_single_result_key_int(self):
    assert unwrap_response({"result": 42}) == 42

  def test_single_result_key_none(self):
    assert unwrap_response({"result": None}) is None

  def test_dict_without_result_key_unchanged(self):
    d = {"foo": "bar"}
    assert unwrap_response(d) == {"foo": "bar"}

  def test_dict_with_multiple_keys_unchanged(self):
    d = {"result": "x", "other": "y"}
    assert unwrap_response(d) == {"result": "x", "other": "y"}

  def test_non_dict_unchanged(self):
    assert unwrap_response("hello") == "hello"
    assert unwrap_response(42) == 42
    assert unwrap_response(None) is None

  def test_json_string_parsed_to_dict(self):
    """Web frontend sends {"result": '{"approved": false}'}."""
    assert unwrap_response({"result": '{"approved": false}'}) == {
        "approved": False
    }

  def test_json_string_parsed_to_list(self):
    assert unwrap_response({"result": "[1, 2, 3]"}) == [1, 2, 3]

  def test_json_string_parsed_to_number(self):
    assert unwrap_response({"result": "42"}) == 42

  def test_json_string_parsed_to_bool(self):
    assert unwrap_response({"result": "true"}) is True

  def test_non_json_string_stays_string(self):
    assert unwrap_response({"result": "plain text"}) == "plain text"

  def test_roundtrip_wrap_unwrap_string(self):
    assert unwrap_response(wrap_response("hello")) == "hello"

  def test_roundtrip_wrap_unwrap_dict(self):
    """Dicts are not wrapped, so unwrap is a no-op."""
    d = {"foo": "bar"}
    assert unwrap_response(wrap_response(d)) == d


# --- create_request_input_event ---


class TestCreateRequestInputEvent:

  def test_basic_event(self):
    ri = RequestInput(
        interrupt_id="test-id",
        message="Please approve",
    )
    event = create_request_input_event(ri)

    assert event.long_running_tool_ids == {"test-id"}
    assert event.content is not None
    fc = event.content.parts[0].function_call
    assert fc.name == "adk_request_input"
    assert fc.id == "test-id"
    assert fc.args["message"] == "Please approve"

  def test_with_payload(self):
    ri = RequestInput(
        interrupt_id="id-1",
        payload={"key": "value"},
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    assert fc.args["payload"] == {"key": "value"}

  def test_with_response_schema(self):
    from pydantic import BaseModel

    class MySchema(BaseModel):
      approved: bool

    ri = RequestInput(
        interrupt_id="id-2",
        response_schema=MySchema,
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    schema = fc.args["response_schema"]
    assert "approved" in schema["properties"]
    assert schema["properties"]["approved"]["type"] == "boolean"


# --- has_request_input_function_call ---


class TestHasRequestInputFunctionCall:

  def test_true_for_request_input_event(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert has_request_input_function_call(event) is True

  def test_false_for_empty_event(self):
    assert has_request_input_function_call(Event()) is False

  def test_false_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="other_tool", args={})
                )
            ]
        )
    )
    assert has_request_input_function_call(event) is False


# --- create_request_input_response ---


class TestCreateRequestInputResponse:

  def test_creates_function_response_part(self):
    part = create_request_input_response("id-1", {"approved": True})
    assert part.function_response.id == "id-1"
    assert part.function_response.name == "adk_request_input"
    assert part.function_response.response == {"approved": True}


# --- get_request_input_interrupt_ids ---


class TestGetRequestInputInterruptIds:

  def test_extracts_ids(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert get_request_input_interrupt_ids(event) == ["id-1"]

  def test_empty_for_no_function_calls(self):
    assert get_request_input_interrupt_ids(Event()) == []

  def test_empty_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="other_tool", args={}, id="id-1"
                    )
                )
            ]
        )
    )
    assert get_request_input_interrupt_ids(event) == []


# --- create_auth_request_event ---


class TestCreateAuthRequestEvent:

  def test_creates_credential_request(self):
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key="test_key",
        ),
        credential_key="test_cred",
    )
    event = create_auth_request_event(auth_config, "auth-id-1")

    assert event.long_running_tool_ids is not None
    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
    assert fc.id == "auth-id-1"
    assert "authConfig" in fc.args


# --- validate_resume_response ---


class TestValidateResumeResponse:

  def test_none_schema_returns_data(self):
    assert validate_resume_response("hello", None) == "hello"

  def test_str_to_int_coercion(self):
    assert validate_resume_response("42", {"type": "integer"}) == 42

  def test_str_to_float_coercion(self):
    assert validate_resume_response("42.5", {"type": "number"}) == 42.5

  def test_str_to_bool_true(self):
    assert validate_resume_response("true", {"type": "boolean"}) is True
    assert validate_resume_response("1", {"type": "boolean"}) is True

  def test_str_to_bool_false(self):
    assert validate_resume_response("false", {"type": "boolean"}) is False
    assert validate_resume_response("0", {"type": "boolean"}) is False

  def test_invalid_coercion_raises_value_error(self):
    import pytest

    with pytest.raises(ValueError):
      validate_resume_response("abc", {"type": "integer"})

  def test_object_schema_validates_dict_type(self):
    import pytest

    schema = {"type": "object"}
    assert validate_resume_response({"name": "Alice"}, schema) == {
        "name": "Alice"
    }

    with pytest.raises(ValueError, match="Failed to coerce data to object"):
      validate_resume_response("not a dict", schema)

  def test_array_schema_validates_list_type(self):
    import pytest

    schema = {"type": "array"}
    assert validate_resume_response([1, 2], schema) == [1, 2]

    with pytest.raises(ValueError, match="Failed to coerce data to array"):
      validate_resume_response("not a list", schema)

  def test_pydantic_type_validation(self):
    from pydantic import BaseModel

    class User(BaseModel):
      name: str
      age: int

    assert validate_resume_response({"name": "Alice", "age": 30}, User) == User(
        name="Alice", age=30
    )


# --- scan_node_events ---


class TestScanNodeEvents:

  def test_scan_empty_events(self):
    results = _scan_node_events([], "/wf@1")
    assert results == {}

  def test_scan_direct_child_output(self):
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"), output="node_a output"
    )
    results = _scan_node_events([event], "/wf@1", group_by_direct_child=True)

    assert "node_a" in results
    assert results["node_a"].output == "node_a output"
    assert results["node_a"].run_id == "1"

  def test_scan_message_as_output(self):
    content = types.Content(parts=[types.Part(text="hello")])
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        content=content,
    )
    event.node_info.message_as_output = True

    results = _scan_node_events([event], "/wf@1", group_by_direct_child=True)

    assert "node_a" in results
    assert results["node_a"].output == content

  def test_scan_descendant_interrupts(self):
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1/sub_node@1"),
        long_running_tool_ids={"interrupt-1"},
    )
    results = _scan_node_events([event], "/wf@1", group_by_direct_child=True)

    assert "node_a" in results
    assert "interrupt-1" in results["node_a"].interrupt_ids

  def test_scan_resolve_interrupts(self):
    event_int = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        long_running_tool_ids={"interrupt-1"},
    )
    event_fr = Event(
        author="user",
        content=types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="interrupt-1",
                        name="adk_request_input",
                        response={"result": "user answer"},
                    )
                )
            ]
        ),
    )

    # Act
    results = _scan_node_events(
        [event_int, event_fr], "/wf@1", group_by_direct_child=True
    )

    # Assert
    assert "node_a" in results
    assert "interrupt-1" in results["node_a"].resolved_ids
    assert results["node_a"].resolved_responses["interrupt-1"] == "user answer"

  def test_scan_matches_specific_node_path_without_child_grouping(self):
    """Scanning matches events for a specific node path when not grouping by direct child."""
    from google.adk.events.event import Event
    from google.adk.events.event import NodeInfo

    # Arrange
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"), output="node_a output"
    )

    # Act
    results = _scan_node_events(
        [event], "/wf@1/node_a@1", group_by_direct_child=False
    )

    # Assert
    assert "/wf@1/node_a@1" in results
    assert results["/wf@1/node_a@1"].output == "node_a output"

  def test_scan_validates_and_coerces_response_against_schema(self):
    """Scanning validates and coerces user response data against the provided schema."""
    from google.adk.events.event import Event
    from google.adk.events.event import NodeInfo
    from google.adk.events.request_input import RequestInput
    from google.genai import types
    from pydantic import BaseModel

    # Arrange
    class MySchema(BaseModel):
      count: int

    ri = RequestInput(
        interrupt_id="interrupt-1",
        response_schema=MySchema,
    )
    event_int = create_request_input_event(ri)
    event_int.node_info = NodeInfo(path="/wf@1/node_a@1")

    event_fr = Event(
        author="user",
        content=types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="interrupt-1",
                        name="adk_request_input",
                        response={"result": '{"count": "42"}'},
                    )
                )
            ]
        ),
    )

    # Act
    results = _scan_node_events(
        [event_int, event_fr], "/wf@1", group_by_direct_child=True
    )

    # Assert
    assert "node_a" in results
    assert results["node_a"].resolved_responses["interrupt-1"] == {"count": 42}
