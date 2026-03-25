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

import asyncio
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...events.event import Event
from ...events.event_actions import EventActions
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ..invocation_context import InvocationContext
from . import _output_schema_processor
from ._execute_tools_node import _long_running_interrupt_event
from ._functions import deep_merge_dicts
from ._functions import _get_tool
from ._functions import generate_auth_event
from ._functions import generate_request_confirmation_event
from ._functions import get_long_running_function_calls
from ._tool_call_node import ToolCallNode
from ._tool_call_node import ToolCallResult


class ParallelToolCallResult(BaseModel):
  """Result of parallel tool execution."""

  tool_results: dict[str, ToolCallResult]
  """Mapping from function_call_id to ToolCallResult."""

  transfer_to_agent: str | None = None
  """Agent name to transfer control to, if any."""

  request_task: dict[str, Any] | None = None
  """Task request from a tool, if any."""

  finish_task: dict[str, Any] | None = None
  """Task completion signal from a tool, if any."""

  skip_summarization: bool | None = None
  """Whether to skip summarization."""


def _build_merged_event(
    tool_results: list[ToolCallResult],
    invocation_context: InvocationContext,
) -> Event:
  """Builds a merged function response Event from ToolCallResults."""
  merged_parts = [
      types.Part.from_function_response(
          name=tr.name,
          response=(
              tr.output
              if isinstance(tr.output, dict)
              else {'result': tr.output}
          ),
      )
      for tr in tool_results
  ]
  # Set function_response.id on each part.
  for part, tr in zip(merged_parts, tool_results):
    part.function_response.id = tr.function_call_id

  # Merge actions from all tool results.
  merged_actions_data: dict[str, Any] = {}
  for tr in tool_results:
    if tr.actions:
      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          tr.actions.model_dump(exclude_none=True),
      )
  merged_actions = EventActions.model_validate(merged_actions_data)

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,
  )


class ParallelToolCallNode(BaseNode):
  """Executes multiple tool calls in parallel via ``ctx.run_node``.

  For each ``FunctionCall``, creates a ``ToolCallNode`` and runs it
  via ``ctx._run_node_internal`` in parallel using ``asyncio.create_task``.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = 'parallel_tool_call'
  rerun_on_resume: bool = True
  tools_dict: dict[str, BaseTool] = Field(...)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    function_calls: list[types.FunctionCall] = list(node_input)
    invocation_context = ctx.get_invocation_context()

    # Detect long-running tools before execution.
    long_running_tool_ids = get_long_running_function_calls(
        function_calls, self.tools_dict
    )

    # Execute each function call in parallel via ToolCallNode.
    tasks = []
    for i, fc in enumerate(function_calls):
      tool = _get_tool(fc, self.tools_dict)
      tool_call_node = ToolCallNode(
          name=f'tool_call__{fc.id or i}',
          tool=tool,
      )
      task = asyncio.create_task(
          ctx._run_node_internal(
              tool_call_node, node_input=fc, name=tool_call_node.name
          )
      )
      tasks.append(task)

    run_results = await asyncio.gather(*tasks)

    # Collect ToolCallResults from completed tool calls.
    tool_results: list[ToolCallResult] = [
        r.output for r in run_results
        if r.output is not None and isinstance(r.output, ToolCallResult)
    ]

    if not tool_results:
      if long_running_tool_ids:
        yield _long_running_interrupt_event(
            invocation_context, function_calls, long_running_tool_ids
        )
      return

    # Build merged event from ToolCallResults for auth/confirmation checks
    # and session content.
    merged_event = _build_merged_event(tool_results, invocation_context)

    # Generate auth event if any tool requested credentials.
    # TODO: unify below auth / confirmation handling based on RFC 683.
    auth_event = generate_auth_event(invocation_context, merged_event)
    if auth_event:
      yield auth_event.model_copy()

    # Generate confirmation event if any tool requested confirmation.
    confirmation_event = generate_request_confirmation_event(
        invocation_context, function_calls, merged_event
    )
    if confirmation_event:
      yield confirmation_event.model_copy()

    # Auth/confirmation are interrupts — yield function response but
    # do not set output (no continuation).
    if auth_event or confirmation_event:
      yield merged_event.model_copy()
      return

    # Handle pending long-running tools (mixed case).
    if long_running_tool_ids:
      function_call_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=invocation_context.agent.name,
          content=types.Content(
              role='model',
              parts=[
                  types.Part(function_call=fc) for fc in function_calls
              ],
          ),
      )
      yield function_call_event

    # Yield the merged function response event.
    yield merged_event.model_copy()

    # Check for pending long-running tools that returned None.
    if long_running_tool_ids:
      responded_ids = {tr.function_call_id for tr in tool_results}
      pending_ids = long_running_tool_ids - responded_ids
      if pending_ids:
        yield _long_running_interrupt_event(
            invocation_context, function_calls, pending_ids
        )
        return

    # Check for structured output.
    json_response = _output_schema_processor.get_structured_model_response(
        merged_event
    )
    if json_response:
      final_event = (
          _output_schema_processor.create_final_model_response_event(
              invocation_context, json_response
          )
      )
      yield final_event.model_copy()

    # Build structured result for parent to read via ctx.run_node.
    actions = merged_event.actions
    result = ParallelToolCallResult(
        tool_results={tr.function_call_id: tr for tr in tool_results},
        transfer_to_agent=actions.transfer_to_agent if actions else None,
        request_task=actions.request_task if actions else None,
        finish_task=actions.finish_task if actions else None,
        skip_summarization=actions.skip_summarization if actions else None,
    )
    yield result
