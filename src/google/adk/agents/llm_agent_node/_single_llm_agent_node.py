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

import logging
from typing import Any
from typing import AsyncGenerator

from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ..llm import _output_schema_processor
from ...models.llm_request import LlmRequest
from ...workflow._base_node import BaseNode
from ..context import Context
from ..llm_agent_workflow.llm_agent import LlmAgent
from ._call_llm_node import _build_llm_request
from ._call_llm_node import CallLlmNode
from ._run_tools_node import RunToolsNode

logger = logging.getLogger('google_adk.' + __name__)


class SingleLlmAgentNode(BaseNode):
  """Orchestrates the LLM reason-act loop for an individual LlmAgent.

  1. Call LLM via ``CallLlmNode``
  2. If function calls returned, execute tools via ``RunToolsNode``
  3. Check termination conditions, otherwise loop back to step 1

  The ``agent`` field must be set by the parent.  This avoids reading from
  ``invocation_context.agent`` at runtime, making the node
  self-contained and testable in any Workflow.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  rerun_on_resume: bool = True

  agent: LlmAgent = Field(...)
  """The LlmAgent whose model, tools, and callbacks drive the loop."""

  @staticmethod
  async def _schedule_node(
      current_ctx: Context,
      node: BaseNode,
      node_input: Any,
      *,
      node_name: str | None = None,  # noqa: ARG004
      use_as_output: bool = False,  # noqa: ARG004
      run_id: str | None = None,
  ):
    """Schedule a child node via NodeRunner.

    Matches the ``ScheduleDynamicNode`` protocol so that child nodes
    can use ``ctx._run_node_internal()``.
    """
    from ...workflow._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node,
        parent_ctx=current_ctx,
        run_id=run_id or _node_name,
    )
    return await runner.run(node_input=node_input)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    # Set IC agent — needed by downstream code
    # (e.g. _build_basic_request reads agent.canonical_model).
    # When run inside a Workflow, ic.agent may be the Workflow itself
    # which lacks canonical_model. Always override.
    # TODO: remove this dependency.
    ctx._invocation_context.agent = self.agent
    ctx.event_author = self.agent.name

    # Always provide our own scheduler so child nodes
    # (CallLlmNode, RunToolsNode) are managed by this node,
    # not by an outer Workflow.
    # TODO: make this a closure and track the created asyncio.Tasks
    ctx._schedule_dynamic_node_internal = self._schedule_node

    # TODO: On resume, resume the tool node first before calling
    # the LLM.  Currently the interrupted RunToolsNode is
    # never completed — on resume, the node restarts and calls the
    # LLM directly, which only works for long-running tools (the FR
    # is in session events).  For auth/confirmation interrupts, the
    # tool must be re-executed with credentials/confirmation before
    # the LLM can proceed.  The fix requires:
    #   1. Extract last FCs from session events
    #   2. Build tools_dict via _process_agent_tools
    #   3. Run RunToolsNode with resume_inputs (port resume
    #      handling from execute_tools: _process_auth_resume,
    #      _process_confirmation_resume, _process_long_running_resume)
    #   4. Then continue to the ReAct loop

    # --- ReAct loop ---
    while True:
      # 1. Build request

      llm_request = LlmRequest()
      async for event in _build_llm_request(ctx, self.agent, llm_request):
        yield event

      if ctx.get_invocation_context().end_invocation:
        break

      # 2. Call LLM
      llm_node = CallLlmNode(agent=self.agent)
      content = await ctx.run_node(llm_node, node_input=llm_request)

      # 3. Check for function calls.
      has_function_calls = False
      if content and getattr(content, 'parts', None):
        has_function_calls = any(
            part.function_call for part in content.parts if part.function_call
        )

      if not has_function_calls:
        break

      # 4. Execute tools
      tool_node = RunToolsNode(
          tools_dict=llm_request.tools_dict,
      )
      tool_actions = await ctx.run_node(tool_node, node_input=content)

      # 5. Check termination conditions
      if tool_actions.set_model_response is not None:
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                ctx.get_invocation_context(), tool_actions.set_model_response
            )
        )
        yield final_event.model_copy()
        break

      if tool_actions.transfer_to_agent or tool_actions.skip_summarization:
        break

    return
    yield  # noqa: unreachable — keeps this an async generator
