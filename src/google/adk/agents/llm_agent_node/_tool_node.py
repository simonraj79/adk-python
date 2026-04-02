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

import copy
from typing import Any
from typing import AsyncGenerator
import uuid

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...events.event import Event
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ..llm._execute_tools_node import _long_running_interrupt_event
from ..llm._functions import generate_auth_event
from ..llm._functions import generate_request_confirmation_event


class ToolActions(BaseModel):
  skip_summarization: bool | None = None
  transfer_to_agent: Any = None
  set_model_response: Any = None


class ToolNodeOutput(BaseModel):
  response: Any
  actions: ToolActions = Field(default_factory=ToolActions)


class ToolNode(BaseNode):
  """Executes a single tool call.

  Calls ``tool.run_async(tool_context=ctx)`` and yields the function
  response value.  The parent ``RunToolsNode`` reads the child
  context's output and actions to build a merged function response event.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = 'tool_call'
  rerun_on_resume: bool = True
  tool: BaseTool = Field(...)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    function_call: types.FunctionCall = node_input

    function_args = (
        copy.deepcopy(function_call.args) if function_call.args else {}
    )

    ctx.function_call_id = function_call.id or str(uuid.uuid4())

    function_response = await self.tool.run_async(
        args=function_args,
        tool_context=ctx,
    )

    if self.tool.is_long_running and not function_response:
      yield _long_running_interrupt_event(
          ctx.get_invocation_context(), [function_call], {ctx.function_call_id}
      )
      return

    # Normalize to dict (Gemini API requires function responses to be dicts).
    if not isinstance(function_response, dict):
      function_response = {'result': function_response}

    invocation_context = ctx.get_invocation_context()
    part = types.Part.from_function_response(
        name=function_call.name,
        response=function_response,
    )
    part.function_response.id = ctx.function_call_id
    response_event = Event(
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        branch=invocation_context.branch,
        content=types.Content(role='user', parts=[part]),
        actions=ctx.actions,
    )

    auth_event = generate_auth_event(invocation_context, response_event)
    if auth_event:
      yield auth_event.model_copy()

    confirmation_event = generate_request_confirmation_event(
        invocation_context, [function_call], response_event
    )
    if confirmation_event:
      yield confirmation_event.model_copy()

    if auth_event or confirmation_event:
      yield response_event.model_copy()
      return

    tool_actions = ToolActions(
        skip_summarization=ctx.actions.skip_summarization,
        transfer_to_agent=ctx.actions.transfer_to_agent,
        set_model_response=ctx.actions.set_model_response,
    )
    yield ToolNodeOutput(response=function_response, actions=tool_actions)
