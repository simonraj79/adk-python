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

from ..llm import _output_schema_processor
from ...events.event import Event
from ...events.event_actions import EventActions
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ..invocation_context import InvocationContext
from ..llm._functions import _get_tool
from ..llm._functions import deep_merge_dicts
from ._tool_node import ToolActions
from ._tool_node import ToolNode
from ._tool_node import ToolNodeOutput


def _build_merged_event(
    completed: list[tuple[types.FunctionCall, ToolNodeOutput]],
    invocation_context: InvocationContext,
) -> Event:
  """Builds a merged function response Event from completed tool outputs."""
  merged_parts = []
  for fc, tool_output in completed:
    response = tool_output.response
    part = types.Part.from_function_response(
        name=fc.name,
        response=response,
    )
    part.function_response.id = fc.id
    merged_parts.append(part)

  # Merge actions from all tool outputs.
  merged_actions_data: dict[str, Any] = {}
  for _, tool_output in completed:
    if tool_output.actions:
      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          tool_output.actions.model_dump(exclude_none=True),
      )
  merged_actions = EventActions.model_validate(merged_actions_data)

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,
  )


def _merge_tool_actions(tool_actions_list: list[ToolActions]) -> ToolActions:
  """Merges a list of ToolActions objects."""
  merged_tool_actions = ToolActions()
  if not tool_actions_list:
    return merged_tool_actions

  all_skip_true = True
  for tool_actions in tool_actions_list:
    if tool_actions.transfer_to_agent is not None:
      if merged_tool_actions.transfer_to_agent is not None:
        raise ValueError(
            'transfer_to_agent cannot be set by more than one tool.'
        )
      merged_tool_actions.transfer_to_agent = tool_actions.transfer_to_agent

    if tool_actions.set_model_response is not None:
      if merged_tool_actions.set_model_response is not None:
        raise ValueError(
            'set_model_response cannot be set by more than one tool.'
        )
      merged_tool_actions.set_model_response = tool_actions.set_model_response

    if tool_actions.skip_summarization is not True:
      all_skip_true = False

  if all_skip_true:
    merged_tool_actions.skip_summarization = True

  return merged_tool_actions


class RunToolsNode(BaseNode):
  """Executes multiple tool calls in parallel via ``ctx.run_node``.

  For each ``FunctionCall``, creates a ``ToolNode`` and runs it
  via ``ctx.run_node`` in parallel using ``asyncio.create_task``.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = 'run_tools'
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

    # Execute each function call in parallel via ToolNode.
    tasks = []
    for i, fc in enumerate(function_calls):
      tool = _get_tool(fc, self.tools_dict)
      tool_node = ToolNode(
          name=fc.name,
          tool=tool,
      )
      task = asyncio.create_task(
          ctx.run_node(
              tool_node,
              node_input=fc,
              run_id=fc.id or str(i),
          )
      )
      tasks.append(task)

    run_results = await asyncio.gather(*tasks)

    # Pair each function call with its output; keep only completed ones.
    completed: list[tuple[types.FunctionCall, ToolNodeOutput]] = [
        (fc, res)
        for fc, res in zip(function_calls, run_results)
        if res is not None
    ]

    # Build merged event from tool outputs.
    merged_event = _build_merged_event(completed, invocation_context)

    tool_actions_list = [tool_output.actions for _, tool_output in completed]
    merged_tool_actions = _merge_tool_actions(tool_actions_list)

    merged_event.output = merged_tool_actions
    yield merged_event.model_copy()
