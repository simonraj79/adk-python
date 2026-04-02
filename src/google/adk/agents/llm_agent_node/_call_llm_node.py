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

import datetime
import logging
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ..llm import _basic
from ..llm import _code_execution
from ..llm import _compaction
from ..llm import _context_cache_processor
from ..llm import _identity
from ..llm import _instructions
from ..llm import _interactions_processor
from ..llm import _nl_planning
from ..llm import _output_schema_processor
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.base_tool import BaseTool
from ...workflow._base_node import BaseNode
from ..context import Context
from ..llm_agent_workflow.llm_agent import LlmAgent
from ..run_config import StreamingMode
from ..llm._agent_transfer import inject_transfer_tools
from ..llm._reasoning import _create_response_processors
from ..llm._reasoning import _finalize_model_response_event
from ..llm._reasoning import _handle_after_model_callback
from ..llm._reasoning import _handle_before_model_callback
from ..llm._reasoning import _process_agent_tools
from ..llm._reasoning import _resolve_toolset_auth
from ..llm._reasoning import _run_and_handle_error
from ..llm.task import _task_contents_processor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'


def _create_request_processors():
  """Request processors for the CallLlmNode.

  Excludes ``auth_preprocessor`` and ``request_confirmation`` processors
  because auth and confirmation resume is handled natively by
  ``RunToolsNode`` via ``rerun_on_resume=True``.
  """
  return [
      _basic.request_processor,
      _instructions.request_processor,
      _identity.request_processor,
      _compaction.request_processor,
      _task_contents_processor.request_processor,
      _context_cache_processor.request_processor,
      _interactions_processor.request_processor,
      _nl_planning.request_processor,
      _code_execution.request_processor,
      _output_schema_processor.request_processor,
  ]


async def _build_llm_request(
    ctx: Context,
    agent: LlmAgent,
    llm_request: LlmRequest,
) -> AsyncGenerator[Any, None]:
  """Runs request processors and prepares the LlmRequest."""
  invocation_context = ctx.get_invocation_context()

  # --- Run request processors ---
  request_processors = _create_request_processors()
  for processor in request_processors:
    async for event in processor.run_async(invocation_context, llm_request):
      yield event

  # --- Resolve toolset authentication ---
  async for event in _resolve_toolset_auth(invocation_context, agent):
    yield event
  if invocation_context.end_invocation:
    return

  # --- Process tool unions ---
  await _process_agent_tools(invocation_context, llm_request)

  # --- Inject transfer targets ---
  transfer_targets = ctx.transfer_targets
  if transfer_targets:
    await inject_transfer_tools(
        invocation_context, llm_request, transfer_targets
    )

  # Config setup
  llm_request.config = llm_request.config or types.GenerateContentConfig()
  llm_request.config.labels = llm_request.config.labels or {}
  if _ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
    llm_request.config.labels[_ADK_AGENT_NAME_LABEL_KEY] = agent.name


class CallLlmNode(BaseNode):
  """Encapsulates a single LLM call cycle.

  It expects an ``LlmRequest`` as its input, calls the LLM, runs response
  processors, and yields the model response event.

  Response is yielded as content events with ``message_as_output=True`` so that
  node runners will automatically pass that content to the caller as the output.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = 'call_llm'

  agent: LlmAgent = Field(...)
  """The LlmAgent whose model, tools, and callbacks drive the LLM call."""

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    invocation_context = ctx.get_invocation_context()

    llm_request = node_input

    # --- Call the LLM ---
    model_response_event = Event(
        author=self.agent.name,
        branch=invocation_context.branch,
    )

    # Before-model callback
    # TODO: Move model callbacks to nodes.
    if response := await _handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    ):
      finalized = _finalize_model_response_event(
          llm_request, response, model_response_event
      )
      yield finalized
      return

    # LLM call
    llm = self.agent.canonical_model
    invocation_context.increment_llm_call_count()
    responses_generator = llm.generate_content_async(
        llm_request,
        stream=(
            invocation_context.run_config.streaming_mode == StreamingMode.SSE
        ),
    )

    response_processors = _create_response_processors()

    async for llm_response in _run_and_handle_error(
        responses_generator,
        invocation_context,
        llm_request,
        model_response_event,
    ):
      # After-model callback
      if altered := await _handle_after_model_callback(
          invocation_context, llm_response, model_response_event
      ):
        llm_response = altered

      # --- Run response processors ---
      for processor in response_processors:
        async for event in processor.run_async(
            invocation_context, llm_response
        ):
          yield event

      # Skip empty responses.
      if (
          not llm_response.content
          and not llm_response.error_code
          and not llm_response.interrupted
      ):
        continue

      # --- Finalize the model response event ---
      # Refresh id/timestamp so each streamed chunk gets unique values.
      model_response_event.id = Event.new_id()
      model_response_event.timestamp = datetime.datetime.now().timestamp()

      finalized_event = _finalize_model_response_event(
          llm_request, llm_response, model_response_event
      )
      has_long_running = bool(finalized_event.long_running_tool_ids)
      finalized_event.long_running_tool_ids = None

      # --- Route decision ---
      if not finalized_event.partial:
        if finalized_event.node_info is None:
          from ....events.event import NodeInfo

          finalized_event.node_info = NodeInfo()
        finalized_event.node_info.message_as_output = True

      yield finalized_event
