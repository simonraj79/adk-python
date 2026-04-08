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

"""_SingleLlmAgent: single-agent LLM reasoning loop as a workflow graph."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import ClassVar
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from ...events.event import Event
from ...models.base_llm import BaseLlm
from ...models.registry import LLMRegistry
from ...tools.base_tool import BaseTool
from ...utils.context_utils import Aclosing
from ...workflow._base_node import START
from ...workflow._function_node import FunctionNode
from ...workflow._workflow import Workflow
from ..context import Context
from ._base_llm_agent import _SingleAfterModelCallback
from ._base_llm_agent import _SingleBeforeModelCallback
from ._base_llm_agent import _SingleOnModelErrorCallback
from ._base_llm_agent import AfterToolCallback
from ._base_llm_agent import BaseLlmAgent
from ._base_llm_agent import BeforeToolCallback
from ._base_llm_agent import OnToolErrorCallback
from ._call_llm_node import call_llm
from ._execute_tools_node import CONTINUE_ROUTE
from ._execute_tools_node import execute_tools
from ._tool_utils import _convert_tool_union_to_tools

if TYPE_CHECKING:
  from ..readonly_context import ReadonlyContext

logger = logging.getLogger('google_adk.' + __name__)


def _convert_node_input_to_json(node_input: Any) -> str:
  """Converts node_input to a JSON string for LLM consumption.

  Args:
    node_input: The input data to convert.

  Returns:
    A JSON string representation of the input.
  """
  if isinstance(node_input, str):
    return node_input
  if isinstance(node_input, (int, float, bool)):
    return json.dumps(node_input)
  if isinstance(node_input, BaseModel):
    return node_input.model_dump_json()
  if isinstance(node_input, (dict, list)):
    return json.dumps(node_input, default=str)
  raise TypeError(
      'Cannot convert node_input of type'
      f' {type(node_input).__name__} to JSON string.'
  )


class _SingleLlmAgent(Workflow, BaseLlmAgent):
  """Single-agent LLM reasoning loop as a workflow graph.

  Inherits all LLM-related fields and ``canonical_*`` properties from
  ``BaseLlmAgent`` mixin, and the workflow graph execution engine from
  ``Workflow``.

  Decomposes the LLM reason-act loop into two workflow nodes:
  ``call_llm`` and ``execute_tools``, connected in a graph.

  Graph structure::

      START -> call_llm -> (route: "execute_tools") -> execute_tools
                ^                                         |
                |___ (route: "continue") _________________|
                     (no route / final response) -> END

  When ``execute_tools`` detects a ``transfer_to_agent`` action or
  other terminal condition, it returns without emitting the
  ``'continue'`` route, so the workflow terminates. The parent
  ``_Mesh`` handles routing to the target agent.

  Transfer targets are NOT stored on the instance. They are read from
  ``Context`` at runtime, enabling instance reuse across
  different parents.
  """

  # All LLM fields (model, instruction, tools, callbacks, etc.)
  # are inherited from BaseLlmAgent mixin.
  #
  # Workflow graph fields (graph, edges, input_schema, rerun_on_resume)
  # are inherited from Workflow.

  DEFAULT_MODEL: ClassVar[str] = 'gemini-2.5-flash'
  """System default model used when no model is set on an agent."""

  _default_model: ClassVar[str | BaseLlm] = DEFAULT_MODEL
  """Overridable default model. Changed via ``set_default_model``."""

  # ------------------------------------------------------------------
  # canonical_* properties
  # ------------------------------------------------------------------

  @property
  def canonical_model(self) -> BaseLlm:
    """Resolves ``self.model`` to a ``BaseLlm`` instance."""
    if isinstance(self.model, BaseLlm):
      return self.model
    elif self.model:
      return LLMRegistry.new_llm(self.model)
    else:
      ancestor = self.parent_agent
      while ancestor is not None:
        if hasattr(ancestor, 'canonical_model'):
          return ancestor.canonical_model
        ancestor = ancestor.parent_agent
      return self._resolve_default_model()

  async def canonical_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """Resolves ``self.instruction`` for this agent."""
    if isinstance(self.instruction, str):
      return self.instruction, False
    instruction = self.instruction(ctx)
    if inspect.isawaitable(instruction):
      instruction = await instruction
    return cast(str, instruction), True

  async def canonical_global_instruction(
      self, ctx: ReadonlyContext
  ) -> tuple[str, bool]:
    """Resolves ``self.global_instruction`` for this agent."""
    if self.global_instruction:
      warnings.warn(
          'global_instruction field is deprecated and will be removed'
          ' in a future version. Use GlobalInstructionPlugin instead'
          ' for the same functionality at the App level.',
          DeprecationWarning,
          stacklevel=2,
      )
    if isinstance(self.global_instruction, str):
      return self.global_instruction, False
    global_instruction = self.global_instruction(ctx)
    if inspect.isawaitable(global_instruction):
      global_instruction = await global_instruction
    return cast(str, global_instruction), True

  async def canonical_tools(
      self, ctx: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Resolves ``self.tools`` to a list of ``BaseTool``."""
    resolved_tools = []
    multiple_tools = len(self.tools) > 1
    model = self.canonical_model
    for tool_union in self.tools:
      try:
        resolved_tools.extend(
            await _convert_tool_union_to_tools(
                tool_union, ctx, model, multiple_tools
            )
        )
      except Exception as e:
        logger.warning(
            'Failed to get tools from toolset %s: %s',
            type(tool_union).__name__,
            e,
        )
    return resolved_tools

  @property
  def canonical_before_model_callbacks(
      self,
  ) -> list[_SingleBeforeModelCallback]:
    """Resolves ``before_model_callback`` to a list."""
    if not self.before_model_callback:
      return []
    if isinstance(self.before_model_callback, list):
      return self.before_model_callback
    return [self.before_model_callback]

  @property
  def canonical_after_model_callbacks(
      self,
  ) -> list[_SingleAfterModelCallback]:
    """Resolves ``after_model_callback`` to a list."""
    if not self.after_model_callback:
      return []
    if isinstance(self.after_model_callback, list):
      return self.after_model_callback
    return [self.after_model_callback]

  @property
  def canonical_on_model_error_callbacks(
      self,
  ) -> list[_SingleOnModelErrorCallback]:
    """Resolves ``on_model_error_callback`` to a list."""
    if not self.on_model_error_callback:
      return []
    if isinstance(self.on_model_error_callback, list):
      return self.on_model_error_callback
    return [self.on_model_error_callback]

  @property
  def canonical_before_tool_callbacks(
      self,
  ) -> list[BeforeToolCallback]:
    """Resolves ``before_tool_callback`` to a list."""
    if not self.before_tool_callback:
      return []
    if isinstance(self.before_tool_callback, list):
      return self.before_tool_callback
    return [self.before_tool_callback]

  @property
  def canonical_after_tool_callbacks(
      self,
  ) -> list[AfterToolCallback]:
    """Resolves ``after_tool_callback`` to a list."""
    if not self.after_tool_callback:
      return []
    if isinstance(self.after_tool_callback, list):
      return self.after_tool_callback
    return [self.after_tool_callback]

  @property
  def canonical_on_tool_error_callbacks(
      self,
  ) -> list[OnToolErrorCallback]:
    """Resolves ``on_tool_error_callback`` to a list."""
    if not self.on_tool_error_callback:
      return []
    if isinstance(self.on_tool_error_callback, list):
      return self.on_tool_error_callback
    return [self.on_tool_error_callback]

  # ------------------------------------------------------------------
  # class method
  # ------------------------------------------------------------------

  @classmethod
  def set_default_model(cls, model: str | BaseLlm) -> None:
    """Overrides the default model for agents with no model set."""
    if not isinstance(model, (str, BaseLlm)):
      raise TypeError('Default model must be a model name or BaseLlm.')
    if isinstance(model, str) and not model:
      raise ValueError('Default model must be a non-empty string.')
    cls._default_model = model

  @classmethod
  def _resolve_default_model(cls) -> BaseLlm:
    """Resolves the current default model to a ``BaseLlm`` instance."""
    default_model = cls._default_model
    if isinstance(default_model, BaseLlm):
      return default_model
    return LLMRegistry.new_llm(default_model)

  @classmethod
  def from_base_llm_agent(cls, agent: BaseLlmAgent) -> _SingleLlmAgent:
    """Create a ``_SingleLlmAgent`` from a ``BaseLlmAgent``'s fields.

    Dynamically copies all ``BaseLlmAgent`` fields (except
    ``sub_agents`` and non-init fields like ``parent_agent``),
    so new fields added to ``BaseLlmAgent`` are automatically
    picked up.
    """
    # In task mode, input_schema and output_schema are handled by
    # FinishTaskTool, not by the LLM config. Exclude them to avoid
    # double-applying the schema. In single_turn mode, output_schema
    # stays so basic.py can set response_schema for controlled output.
    exclude = {'sub_agents'}
    if agent.mode == 'task':
      exclude.update(('input_schema', 'output_schema'))
    elif agent.mode == 'single_turn':
      exclude.add('input_schema')
    data = {
        k: getattr(agent, k)
        for k, info in BaseLlmAgent.model_fields.items()
        if info.init is not False and k not in exclude
    }
    return cls(**data)

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  @override
  def model_post_init(self, context: Any) -> None:
    """Build the call_llm <-> execute_tools workflow graph.

    Sets up ``self.edges`` to wire the two nodes, then delegates to
    ``Workflow.model_post_init`` for graph construction and
    validation.
    """
    execute_tools_node = FunctionNode(execute_tools, rerun_on_resume=True)
    self.edges = [
        (START, call_llm),
        (call_llm, {'execute_tools': execute_tools_node}),
        (execute_tools_node, {CONTINUE_ROUTE: call_llm}),
    ]
    super().model_post_init(context)

  # ------------------------------------------------------------------
  # Node interface
  # ------------------------------------------------------------------

  @override
  async def run_node_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the agent as a workflow node.

    Appends ``node_input`` as user content to the session so the LLM
    sees it in the conversation history built by the content request
    processor. If ``node_input`` is already a ``types.Content``, it is
    appended directly; otherwise it is converted to a JSON string.

    When ``input_schema`` is set (a ``BaseModel`` subclass), validates
    ``node_input`` against the schema before conversion. This applies
    when the agent is used as a node inside a workflow (e.g.
    ``SequentialAgent``, ``ParallelAgent``).

    Args:
      ctx: The workflow context.
      node_input: The input to the node. This will be added to the session
        history as user content. Can be a `types.Content` object, a string,
        a Pydantic BaseModel, or other JSON-serializable types.

    Yields:
      `Event` objects generated during the execution of the agent's workflow.
    """
    # Validate node_input against input_schema if it's a BaseModel type.
    if (
        node_input is not None
        and not isinstance(node_input, types.Content)
        and isinstance(self.input_schema, type)
        and issubclass(self.input_schema, BaseModel)
        and not isinstance(node_input, self.input_schema)
    ):
      if isinstance(node_input, str):
        node_input = self.input_schema.model_validate_json(node_input)
      else:
        node_input = self.input_schema.model_validate(node_input)

    if node_input is not None:
      ic = ctx._invocation_context
      # Append node_input as user content unless the runner already
      # appended it (same object, no branch). In a branched context
      # (e.g. single_turn via _LlmAgentWrapper), always re-append so
      # the content is visible under the branch.
      if node_input is not ic.user_content or ic.branch:
        if isinstance(node_input, types.Content):
          content = node_input
        else:
          json_str = _convert_node_input_to_json(node_input)
          content = types.Content(
              role='user',
              parts=[types.Part(text=json_str)],
          )
        ic.session.events.append(
            Event(
                invocation_id=ic.invocation_id,
                author='user',
                branch=ic.branch,
                content=content,
            )
        )
    async with Aclosing(
        super().run_node_impl(ctx=ctx, node_input=node_input)
    ) as run_gen:
      async for event in run_gen:
        yield event
