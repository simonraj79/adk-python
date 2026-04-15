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

"""Utilities for ADK workflows."""

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import json
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import TypeAdapter
from pydantic import ValidationError

from ...auth.auth_credential import AuthCredentialTypes as _AuthCredentialTypes
from ...auth.auth_handler import AuthHandler
from ...auth.auth_tool import AuthConfig
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.request_input import RequestInput
from ...utils._schema_utils import schema_to_json_schema
from .._node_path_builder import _NodePathBuilder

if TYPE_CHECKING:
  from ...auth.auth_credential import AuthCredential
  from ...sessions.state import State

REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = 'adk_request_credential'

_RESULT_KEY = 'result'
"""Key used to wrap non-dict values in a FunctionResponse dict."""


def wrap_response(value: Any) -> dict[str, Any]:
  """Wraps a value into a dict suitable for FunctionResponse.response.

  If the value is already a dict, returns it as-is.
  Otherwise wraps as ``{"result": value}``.
  """
  if isinstance(value, dict):
    return value
  return {_RESULT_KEY: value}


def unwrap_response(data: Any) -> Any:
  """Unwraps a FunctionResponse dict to the original value.

  If ``data`` is a dict with exactly one key ``"result"``, extracts the
  value.  String values are JSON-parsed when possible (the web frontend
  wraps user text as ``{"result": text}`` without parsing).

  Otherwise returns ``data`` unchanged.
  """
  if isinstance(data, dict) and len(data) == 1 and _RESULT_KEY in data:
    value = data[_RESULT_KEY]
    if isinstance(value, str):
      try:
        value = json.loads(value)
      except (json.JSONDecodeError, ValueError):
        pass
    return value
  return data


def create_request_input_event(request_input: RequestInput) -> Event:
  """Creates a RequestInput event from a RequestInput object."""
  args = request_input.model_dump(exclude={'response_schema'})
  args['response_schema'] = (
      schema_to_json_schema(request_input.response_schema)
      if request_input.response_schema is not None
      else None
  )
  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                      args=args,
                      id=request_input.interrupt_id,
                  )
              )
          ]
      ),
      long_running_tool_ids=[request_input.interrupt_id],
  )


def has_request_input_function_call(event: Event) -> bool:
  """Checks if an event contains a `request_input` function call."""
  if not (event.content and event.content.parts):
    return False
  return any(
      p.function_call
      and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
      for p in event.content.parts
  )


def has_auth_request_function_call(event: Event) -> bool:
  """Checks if an event contains an `adk_request_credential` function call."""
  if not (event.content and event.content.parts):
    return False
  return any(
      p.function_call
      and p.function_call.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
      for p in event.content.parts
  )


def create_request_input_response(
    interrupt_id: str,
    response: Mapping[str, Any],
) -> types.Part:
  """Creates a FunctionResponse part in response to a `request_input` function call.

  Args:
    interrupt_id: The interrupt_id from an event containing a `request_input`
      function call.
    response: The response data to send back.

  Returns:
    A types.Part containing the FunctionResponse.
  """
  return types.Part(
      function_response=types.FunctionResponse(
          id=interrupt_id,
          name=REQUEST_INPUT_FUNCTION_CALL_NAME,
          response=response,
      )
  )


def get_request_input_interrupt_ids(event: Event) -> list[str]:
  """Extracts interrupt_ids from an event containing `request_input` function calls."""
  interrupt_ids: list[str] = []
  if not event.content or not event.content.parts:
    return interrupt_ids
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
    ):
      interrupt_ids.append(part.function_call.id)
  return interrupt_ids


# ---------------------------------------------------------------------------
# Auth credential utilities
# ---------------------------------------------------------------------------


def _build_auth_message(auth_config: AuthConfig) -> str:
  """Builds a human-readable message describing what credential is needed."""
  raw_cred = auth_config.raw_auth_credential
  if not raw_cred:
    return 'Please provide your authentication credentials.'

  auth_type = raw_cred.auth_type
  if auth_type == _AuthCredentialTypes.API_KEY:
    name = getattr(auth_config.auth_scheme, 'name', 'API key')
    return f'Please provide your API key for {name}.'
  elif auth_type in (
      _AuthCredentialTypes.OAUTH2,
      _AuthCredentialTypes.OPEN_ID_CONNECT,
  ):
    return 'Please complete the authentication flow.'

  return 'Please provide your authentication credentials.'


def create_auth_request_event(
    auth_config: AuthConfig,
    interrupt_id: str,
) -> Event:
  """Creates an event requesting user authentication credentials.

  Args:
    auth_config: The auth configuration for the node.
    interrupt_id: The interrupt ID for this auth request.

  Returns:
    An Event containing an ``adk_request_credential`` function call.
  """
  auth_handler = AuthHandler(auth_config)
  auth_request = auth_handler.generate_auth_request()
  args = AuthToolArguments(
      function_call_id=interrupt_id,
      auth_config=auth_request,
  ).model_dump(exclude_none=True, by_alias=True)

  # Add message so the UI / CLI knows what to display.
  args['message'] = _build_auth_message(auth_config)

  return Event(
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
                      id=interrupt_id,
                      args=args,
                  )
              )
          ]
      ),
      long_running_tool_ids=[interrupt_id],
  )


def _build_credential_from_value(
    auth_config: AuthConfig,
    value: Any,
) -> 'AuthCredential':
  """Builds an AuthCredential from a raw user-provided value.

  For API_KEY, the value is used as the key string directly.
  For all other types, the value is parsed as an AuthCredential dict.
  """
  from ...auth.auth_credential import AuthCredential

  raw_cred = auth_config.raw_auth_credential
  if raw_cred is None:
    return AuthCredential.model_validate(value)

  if raw_cred.auth_type == _AuthCredentialTypes.API_KEY:
    return AuthCredential(
        auth_type=_AuthCredentialTypes.API_KEY,
        api_key=str(value),
    )

  return AuthCredential.model_validate(value)


async def process_auth_resume(
    response_data: Any,
    auth_config: AuthConfig,
    state: State,
) -> None:
  """Stores credentials from an auth resume response into session state.

  Accepts multiple response formats (tried in order):
    1. A full ``AuthConfig`` dict (from web UI OAuth flow).
    2. An ``AuthCredential`` dict.
    3. A plain value (string for API key). The node's
       ``auth_config.raw_auth_credential.auth_type`` determines how the
       value is interpreted.

  The caller is responsible for unwrapping ``{"result": ...}`` wrappers
  before calling this function.

  Args:
    response_data: The unwrapped response from the client.
    auth_config: The original auth configuration for the node.
    state: The session state to store credentials in.
  """
  try:
    response_config = AuthConfig.model_validate(response_data)
  except (ValidationError, TypeError):
    response_config = auth_config.model_copy(deep=True)
    response_config.exchanged_auth_credential = _build_credential_from_value(
        auth_config, response_data
    )

  response_config.credential_key = auth_config.credential_key
  await AuthHandler(auth_config=response_config).parse_and_store_auth_response(
      state=state
  )


def has_auth_credential(
    auth_config: AuthConfig,
    state: State,
) -> bool:
  """Returns True if a credential for the given auth config exists in state."""
  return AuthHandler(auth_config).get_auth_response(state) is not None


def extract_schema_from_event(event: Event, interrupt_id: str) -> Any | None:
  """Extracts the response schema from an event if it's a RequestInput call.

  Args:
    event: The event to extract from.
    interrupt_id: The ID of the interrupt to match.

  Returns:
    The schema if found, or None.
  """
  if not event.content or not event.content.parts:
    return None

  for part in event.content.parts:
    fc = part.function_call
    if (
        fc
        and fc.name == REQUEST_INPUT_FUNCTION_CALL_NAME
        and fc.id == interrupt_id
    ):
      return fc.args.get('response_schema')

  return None


def validate_resume_response(response_data: Any, schema: Any) -> Any:
  """Validates and coerces resume response data against a schema.

  Args:
    response_data: The data to validate.
    schema: The schema to validate against (Python type, GenericAlias, or raw
      JSON Schema dict).

  Returns:
    The validated and coerced data.
  """
  if schema is None:
    return response_data

  # If it's a JSON Schema dict, map type to Python type for TypeAdapter
  if isinstance(schema, dict):
    type_str = schema.get('type')

    type_mapping = {
        'integer': int,
        'number': float,
        'string': str,
        'boolean': bool,
        'array': list,
        'object': dict,
    }

    # Special handling for object schemas with properties
    if type_str == 'object' and 'properties' in schema:
      from pydantic import create_model

      properties = schema['properties']
      required = schema.get('required', [])

      fields = {}
      for prop_name, prop_schema in properties.items():
        prop_type_str = prop_schema.get('type')
        prop_type = (
            type_mapping.get(prop_type_str, Any) if prop_type_str else Any
        )

        if prop_name in required:
          fields[prop_name] = (prop_type, ...)
        else:
          fields[prop_name] = (prop_type | None, None)  # type: ignore[assignment]

      try:
        DynamicModel = create_model('DynamicModel', **fields)
        # Validate and return as dict
        model_instance = TypeAdapter(DynamicModel).validate_python(
            response_data
        )
        return model_instance.model_dump()
      except Exception as e:
        raise ValueError(f'Validation failed for object schema: {e}') from e

    mapped_type = type_mapping.get(type_str) if type_str else None
    if mapped_type:
      try:
        return TypeAdapter(mapped_type).validate_python(response_data)
      except Exception as e:
        raise ValueError(f'Failed to coerce data to {type_str}: {e}') from e

    # Fallback: skip validation for complex schemas (similar to base node)
    return response_data

  # For Python types and Pydantic models, use TypeAdapter directly
  try:
    return TypeAdapter(schema).validate_python(response_data)
  except Exception as e:
    raise ValueError(f'Validation failed against schema: {e}') from e


@dataclass
class _ChildScanState:
  """Per-child state accumulated during event scanning for resume."""

  run_id: str | None = None
  output: Any = None
  route: str | None = None
  branch: str | None = None
  interrupt_ids: set[str] = field(default_factory=set)
  resolved_ids: set[str] = field(default_factory=set)
  resolved_responses: dict[str, Any] = field(default_factory=dict)


def _scan_node_events(
    events: list[Event],
    base_path: str,
    group_by_direct_child: bool = False,
) -> dict[str, _ChildScanState]:
  """Scans session events to reconstruct node states for resume.

  Args:
    events: List of session events.
    base_path: The path of the workflow or node to scan under.
    group_by_direct_child: If True, groups events by the direct child segment
      under base_path (preserving @run_id to isolate instances). If False,
      returns state for the base_path itself.

  Returns:
    A dict mapping node identifiers to _ChildScanState.
    If group_by_direct_child is True, keys are direct child segments (e.g.,
    "node_a@1").
    If group_by_direct_child is False, the key is base_path itself.
  """
  results: dict[str, _ChildScanState] = {}
  interrupt_owner: dict[str, str] = {}
  schemas_by_id: dict[str, Any] = {}

  base_path_builder = _NodePathBuilder.from_string(base_path)

  def get_owner_key(event_path_builder: _NodePathBuilder) -> str | None:
    if group_by_direct_child:
      if not event_path_builder.is_descendant_of(base_path_builder):
        return None
      child_path = base_path_builder.get_direct_child(event_path_builder)
      return child_path._segments[-1]  # Return raw segment with @run_id
    else:
      if event_path_builder == base_path_builder or event_path_builder.is_descendant_of(
          base_path_builder
      ):
        return base_path
      return None

  for event in events:
    # 1. Handle FR events (User responses)
    if event.author == 'user' and event.content and event.content.parts:
      for part in event.content.parts:
        fr = part.function_response
        if fr and fr.id and fr.id in interrupt_owner:
          owner = interrupt_owner[fr.id]
          if owner not in results:
            results[owner] = _ChildScanState()
          results[owner].resolved_ids.add(fr.id)
          response_data = unwrap_response(fr.response)

          schema = schemas_by_id.get(fr.id)
          if schema:
            try:
              response_data = validate_resume_response(response_data, schema)
            except Exception as e:
              raise ValueError(
                  f'Validation failed for interrupt {fr.id}: {e}'
              ) from e

          results[owner].resolved_responses[fr.id] = response_data
      continue

    # 2. Match events under base_path
    event_node_path = event.node_info.path or ''
    event_path_builder = _NodePathBuilder.from_string(event_node_path)
    owner_key = get_owner_key(event_path_builder)

    if not owner_key:
      continue

    if owner_key not in results:
      owner_path_builder = _NodePathBuilder.from_string(owner_key)
      results[owner_key] = _ChildScanState(run_id=owner_path_builder.run_id)

    child = results[owner_key]

    # 3. Extract output and route
    is_direct = False
    if group_by_direct_child:
      is_direct = event_path_builder.is_direct_child_of(base_path_builder)
    else:
      is_direct = event_path_builder == base_path_builder

    has_output = event.output is not None
    use_message_as_output = False
    if (
        not has_output
        and event.node_info
        and event.node_info.message_as_output
        and event.content is not None
    ):
      has_output = True
      use_message_as_output = True

    is_delegated = False
    if has_output and event.node_info.output_for:
      if not group_by_direct_child:
        is_delegated = base_path in event.node_info.output_for

    if is_direct or is_delegated:
      if event.output is not None:
        child.output = event.output
        child.branch = event.branch
      elif use_message_as_output:
        child.output = event.content
      if event.actions and event.actions.route is not None:
        child.route = event.actions.route

    # 5. Extract interrupts
    if event.long_running_tool_ids:
      for interrupt_id in event.long_running_tool_ids:
        child.interrupt_ids.add(interrupt_id)
        interrupt_owner[interrupt_id] = owner_key

        schema_json = extract_schema_from_event(event, interrupt_id)
        if schema_json:
          schemas_by_id[interrupt_id] = schema_json

  return results
