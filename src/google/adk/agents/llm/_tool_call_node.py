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
import uuid
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...events.event_actions import EventActions
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ._functions import __build_response_event as _build_response_event


class ToolCallResult(BaseModel):
  """Result of a single tool call."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str
  """The tool/function name that was called."""

  function_call_id: str
  """The unique id of the function call."""

  output: Any = None
  """The function response output."""

  error: str | None = None
  """Error message if the tool call failed."""

  actions: EventActions | None = None
  """Actions set by the tool (e.g. transfer_to_agent, auth requests)."""


class ToolCallNode(BaseNode):
  """Executes a single tool call.

  Calls ``tool.run_async(tool_context=ctx)`` and yields the function
  response event plus a structured ``ToolCallResult`` for the parent
  ``ParallelToolCallNode`` to collect.
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
    invocation_context = ctx.get_invocation_context()

    function_args = (
        copy.deepcopy(function_call.args) if function_call.args else {}
    )

    ctx.function_call_id = function_call.id or str(uuid.uuid4())

    function_response = await self.tool.run_async(
        args=function_args, tool_context=ctx,
    )

    if self.tool.is_long_running and not function_response:
      return

    # TODO: clean up to avoid setting meta ids directly.
    function_response_event = _build_response_event(
        self.tool, function_response, ctx, invocation_context
    )
    # Yield the content event (goes to session).
    yield function_response_event

    # Yield as a non-Event value so BaseNode.run() wraps it as
    # Event(output=ToolCallResult(...)).  This separate output-only event
    # lets ctx.run_node() return the result to ParallelToolCallNode.
    yield ToolCallResult(
        name=self.tool.name,
        function_call_id=ctx.function_call_id,
        output=(
            function_response
            if isinstance(function_response, dict)
            else {'result': function_response}
        ),
        actions=function_response_event.actions,
    )
