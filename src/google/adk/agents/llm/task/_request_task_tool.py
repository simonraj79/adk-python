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

"""RequestTaskTool: delegates a task to a sub-agent."""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
from typing_extensions import override

from ....tools.base_tool import BaseTool
from ._task_models import _DefaultTaskInput
from ._task_models import TaskRequest

if TYPE_CHECKING:
  from ....models.llm_request import LlmRequest
  from ....tools.tool_context import ToolContext
  from ...llm_agent import LlmAgent


class RequestTaskTool(BaseTool):
  """Tool for delegating a task to a sub-agent.

  The coordinator calls this tool to send structured input to a task or
  single-turn sub-agent. On success it sets
  ``tool_context.actions.request_task`` with a ``TaskRequest``
  instance.
  """

  def __init__(
      self,
      task_agent: LlmAgent,
  ):
    """Initialize the request_task tool.

    Args:
      task_agent: The target sub-agent to delegate to. The agent's
        ``input_schema`` is used for validation. If None, the default
        schema (``goal`` + ``background``) is used.
    """
    self._agent_name = task_agent.name
    input_schema = task_agent.input_schema
    self.input_schema: type[BaseModel] = (
        input_schema if input_schema is not None else _DefaultTaskInput
    )

    agent_desc = task_agent.description or task_agent.name
    description = (
        f'Delegate a task to the "{task_agent.name}" agent. {agent_desc}'
    )

    super().__init__(
        name=task_agent.name,
        description=description,
    )

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Get the function declaration for this tool."""
    schema_json = self.input_schema.model_json_schema()

    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema=schema_json,
    )

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Add tool declaration and delegation instructions.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request.
    """
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    instruction = (
        f'To delegate work to "{self._agent_name}", call the'
        f' `{self.name}` tool with the required parameters.'
        ' The agent will execute the task and return results.'
    )
    llm_request.append_instructions([instruction])

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> str | dict[str, str]:
    """Execute the request_task tool.

    Validates args against the input schema and sets
    ``tool_context.actions.request_task`` on success.

    Args:
      args: The arguments passed to the tool.
      tool_context: The tool execution context.

    Returns:
      Confirmation message, or error dict if validation fails.
    """
    try:
      validated = self.input_schema.model_validate(args)
    except ValidationError as e:
      return {
          'error': (
              f'Invoking `{self.name}()` failed due to validation'
              f' errors:\n{e}\nYou could retry calling this tool, but'
              ' it is IMPORTANT for you to provide all the mandatory'
              ' parameters with correct types.'
          )
      }

    validated_dict = validated.model_dump()

    # Merge into existing dict to support parallel delegations.
    if tool_context.actions.request_task is None:
      tool_context.actions.request_task = {}
    tool_context.actions.request_task[tool_context.function_call_id] = (
        TaskRequest(
            agent_name=self._agent_name,
            input=validated_dict,
        )
    )

    return f'Delegating task to {self._agent_name}.'


def render_task_input(
    input_dict: dict[str, Any],
    is_single_turn: bool,
) -> str:
  """Render task input as a human-readable string for the task agent.

  Used by the _Mesh to set ``user_content`` on the InvocationContext
  so the content processor includes it in the task agent's LLM context.

  Args:
    input_dict: The validated input data for the task.
    is_single_turn: Whether the target agent is single-turn.

  Returns:
    Formatted task input string.
  """
  lines = ['[Delegated Task]']
  for key, value in input_dict.items():
    lines.append(f'{key}: {value}')
  if is_single_turn:
    lines.append('')
    lines.append(
        'Important: You will not receive any user replies or'
        ' clarifications. Complete the task using only the'
        ' information provided above.'
    )
  return '\n'.join(lines)
