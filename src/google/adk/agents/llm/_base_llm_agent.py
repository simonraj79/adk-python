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

"""BaseLlmAgent: BaseAgent subclass providing LLM-related fields.

Inherits from BaseAgent and adds all LLM-specific fields (model,
instruction, tools, callbacks, etc.). Also defines callback and
tool type aliases. Serves as the base class for _SingleLlmAgent and
the new LlmAgent.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Literal
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import TypeAlias

from ...code_executors.base_code_executor import BaseCodeExecutor
from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...planners.base_planner import BasePlanner
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.tool_context import ToolContext
from ...utils._schema_utils import SchemaType
from ..base_agent import BaseAgent
from ..callback_context import CallbackContext
from ..readonly_context import ReadonlyContext

# ---------------------------------------------------------------------------
# Callback type aliases (previously defined in old LlmAgent)
# ---------------------------------------------------------------------------

_SingleBeforeModelCallback: TypeAlias = Callable[
    ['CallbackContext', LlmRequest],
    Awaitable[Optional[LlmResponse]] | Optional[LlmResponse],
]

BeforeModelCallback: TypeAlias = (
    _SingleBeforeModelCallback | list[_SingleBeforeModelCallback]
)

_SingleAfterModelCallback: TypeAlias = Callable[
    ['CallbackContext', LlmResponse],
    Awaitable[Optional[LlmResponse]] | Optional[LlmResponse],
]

AfterModelCallback: TypeAlias = (
    _SingleAfterModelCallback | list[_SingleAfterModelCallback]
)

_SingleOnModelErrorCallback: TypeAlias = Callable[
    ['CallbackContext', LlmRequest, Exception],
    Awaitable[Optional[LlmResponse]] | Optional[LlmResponse],
]

OnModelErrorCallback: TypeAlias = (
    _SingleOnModelErrorCallback | list[_SingleOnModelErrorCallback]
)

_SingleBeforeToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext],
    Awaitable[Optional[dict]] | Optional[dict],
]

BeforeToolCallback: TypeAlias = (
    _SingleBeforeToolCallback | list[_SingleBeforeToolCallback]
)

_SingleAfterToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext, dict],
    Awaitable[Optional[dict]] | Optional[dict],
]

AfterToolCallback: TypeAlias = (
    _SingleAfterToolCallback | list[_SingleAfterToolCallback]
)

_SingleOnToolErrorCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext, Exception],
    Awaitable[Optional[dict]] | Optional[dict],
]

OnToolErrorCallback: TypeAlias = (
    _SingleOnToolErrorCallback | list[_SingleOnToolErrorCallback]
)

InstructionProvider: TypeAlias = Callable[
    ['ReadonlyContext'], str | Awaitable[str]
]

ToolUnion: TypeAlias = Callable | BaseTool | BaseToolset

logger = logging.getLogger('google_adk.' + __name__)


class BaseLlmAgent(BaseAgent):
  """BaseAgent subclass providing LLM-related fields.

  Inherits from BaseAgent (name, description, sub_agents,
  parent_agent, run_async, state management, etc.) and adds all
  LLM-specific fields.

  Subclassed by:
    - LlmAgent
  """

  # --- Model ---

  model: str | BaseLlm = ''
  """The model to use for the agent.

  When not set, inherits from the parent agent or uses the default.
  """

  # --- Instructions ---

  instruction: str | InstructionProvider = ''
  """Dynamic instructions for the LLM model, guiding the agent's behavior.

  These instructions can contain placeholders like {variable_name} that
  will be resolved at runtime using session state and context.
  """

  static_instruction: Optional[types.ContentUnion] = None
  """Static instruction content sent literally as system instruction.

  This field is for content that never changes and doesn't contain
  placeholders. Primarily for context caching optimization.
  """

  global_instruction: str | InstructionProvider = ''
  """Instructions for all agents in the entire agent tree.

  ONLY the global_instruction in the root agent will take effect.

  .. deprecated::
      Use ``GlobalInstructionPlugin`` instead. This field will be
      removed in a future release.
  """

  # --- Tools ---

  tools: list[ToolUnion] = Field(default_factory=list)
  """Tools available to this agent."""

  # --- Configuration ---

  generate_content_config: Optional[types.GenerateContentConfig] = None
  """Additional content generation configurations."""

  output_schema: Optional[SchemaType] = None
  """The output schema when the agent replies.

  Supports all schema types that the underlying Google GenAI API supports:
    - type[BaseModel]: e.g., MySchema
    - list[type[BaseModel]]: e.g., list[MySchema]
    - list[primitive]: e.g., list[str], list[int]
    - dict: Raw dict schemas
    - Schema: Google's Schema type
  """

  output_key: Optional[str] = None
  """The key in session state to store the output of the agent."""

  input_schema: Optional[type[BaseModel]] = None
  """The input schema when the agent is used as a tool."""

  mode: Literal['chat', 'task', 'single_turn'] = 'chat'
  """The delegation mode for this agent.

  Options:
    chat: Standard chat agent reachable via transfer_to_agent (default).
    task: Task agent that receives structured input and returns structured
      output via finish_task. Supports multi-turn tool use.
    single_turn: Like task but completes in a single LLM turn with no
      user interaction.
  """

  include_contents: Literal['default', 'none'] = 'default'
  """Controls content inclusion in model requests.

  Options:
    default: Model receives relevant conversation history
    none: Model receives no prior history
  """

  # --- Advanced ---

  planner: Optional[BasePlanner] = None
  """Instructs the agent to make a plan and execute it step by step."""

  code_executor: Optional[BaseCodeExecutor] = None
  """Allow agent to execute code blocks from model responses."""

  # --- Transfer ---

  disallow_transfer_to_parent: bool = False
  """Disallows LLM-controlled transferring to the parent agent."""

  disallow_transfer_to_peers: bool = False
  """Disallows LLM-controlled transferring to peer agents."""

  # --- Callbacks ---

  before_model_callback: Optional[BeforeModelCallback] = None
  """Callback or list of callbacks to be called before calling the LLM."""

  after_model_callback: Optional[AfterModelCallback] = None
  """Callback or list of callbacks to be called after calling the LLM."""

  on_model_error_callback: Optional[OnModelErrorCallback] = None
  """Callback or list of callbacks when a model call encounters an error."""

  before_tool_callback: Optional[BeforeToolCallback] = None
  """Callback or list of callbacks to be called before calling a tool."""

  after_tool_callback: Optional[AfterToolCallback] = None
  """Callback or list of callbacks to be called after calling a tool."""

  on_tool_error_callback: Optional[OnToolErrorCallback] = None
  """Callback or list of callbacks when a tool call encounters an error."""
