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

from . import _output_schema_processor
from ...events.event import Event
from ...events.event_actions import EventActions
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ..invocation_context import InvocationContext
from ._functions import _get_tool
from ._functions import deep_merge_dicts
from ._tool_call_node import ToolCallNode


def _build_merged_event(
    completed: list[tuple[types.FunctionCall, Context]],
    invocation_context: InvocationContext,
) -> Event:
  """Builds a merged function response Event from completed tool contexts."""
  merged_parts = []
  for fc, child_ctx in completed:
    response = child_ctx.output
    part = types.Part.from_function_response(
        name=fc.name,
        response=response,
    )
    part.function_response.id = child_ctx.function_call_id
    merged_parts.append(part)

  # Merge actions from all child contexts.
  merged_actions_data: dict[str, Any] = {}
  for _, child_ctx in completed:
    if child_ctx.actions:
      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          child_ctx.actions.model_dump(exclude_none=True),
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
      node_input: types.Content,
  ) -> AsyncGenerator[Any, None]:
    function_calls: list[types.FunctionCall] = []
    if node_input and node_input.parts:
      function_calls = [
          part.function_call for part in node_input.parts if part.function_call
      ]
    invocation_context = ctx.get_invocation_context()

    # Execute each function call in parallel via ToolCallNode.
    tasks = []
    for i, fc in enumerate(function_calls):
      tool = _get_tool(fc, self.tools_dict)
      tool_call_node = ToolCallNode(
          name=fc.name,
          tool=tool,
      )
      task = asyncio.create_task(
          ctx._run_node_internal(
              tool_call_node,
              node_input=fc,
              name=tool_call_node.name,
              run_id=fc.id or str(i),
          )
      )
      tasks.append(task)

    run_results = await asyncio.gather(*tasks)

    # Pair each function call with its child context; keep only completed ones.
    completed: list[tuple[types.FunctionCall, Context]] = [
        (fc, child_ctx)
        for fc, child_ctx in zip(function_calls, run_results)
        if child_ctx.output is not None
    ]

    interrupted_ctxs = [
        child_ctx for child_ctx in run_results if child_ctx.interrupt_ids
    ]

    if not completed:
      return

    # Build merged event from child contexts for auth/confirmation checks
    # and session content.
    merged_event = _build_merged_event(completed, invocation_context)

    # Bubble up actions to parent context for checking termination conditions.
    actions = merged_event.actions
    if actions:
      ctx.actions.transfer_to_agent = (
          actions.transfer_to_agent or ctx.actions.transfer_to_agent
      )
      ctx.actions.request_task = (
          actions.request_task or ctx.actions.request_task
      )
      ctx.actions.finish_task = actions.finish_task or ctx.actions.finish_task
      ctx.actions.skip_summarization = (
          actions.skip_summarization or ctx.actions.skip_summarization
      )

    if interrupted_ctxs:
      yield merged_event.model_copy()
      return

    # Check for structured output.
    json_response = _output_schema_processor.get_structured_model_response(
        merged_event
    )
    if json_response:
      yield merged_event.model_copy()
      final_event = _output_schema_processor.create_final_model_response_event(
          invocation_context, json_response
      )
      if final_event.node_info is None:
        from ...events.event import NodeInfo

        final_event.node_info = NodeInfo()
      final_event.node_info.message_as_output = True
      yield final_event.model_copy()
    else:
      if merged_event.node_info is None:
        from ...events.event import NodeInfo

        merged_event.node_info = NodeInfo()
      merged_event.node_info.message_as_output = True
      yield merged_event.model_copy()
