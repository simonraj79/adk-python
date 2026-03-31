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
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context


class ToolCallNode(BaseNode):
  """Executes a single tool call.

  Calls ``tool.run_async(tool_context=ctx)`` and yields the function
  response value.  The parent ``ParallelToolCallNode`` reads the child
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
      return

    # Normalize to dict (Gemini API requires function responses to be dicts).
    if not isinstance(function_response, dict):
      function_response = {'result': function_response}

    yield function_response
