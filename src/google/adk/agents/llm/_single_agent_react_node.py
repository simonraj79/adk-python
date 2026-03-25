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

from ...workflow._base_node import BaseNode
from ..context import Context
from ..llm_agent import LlmAgent
from ._llm_call_node import LlmCallNode
from ._llm_call_node import LlmCallResult
from ._parallel_tool_call_node import ParallelToolCallNode
from ._parallel_tool_call_node import ParallelToolCallResult

logger = logging.getLogger('google_adk.' + __name__)


class SingleAgentReactNode(BaseNode):
  """Orchestrates the LLM reason-act loop for an individual LlmAgent.

  1. Call LLM via ``LlmCallNode``
  2. If function calls returned, execute tools via ``ParallelToolCallNode``
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
      execution_id: str,
      node_input: Any,
      *,
      node_name: str | None = None,
  ):
    """Schedule a child node via NodeRunner.

    Matches the ``ScheduleDynamicNode`` protocol so that child nodes
    can use ``ctx._run_node_internal()``.
    """
    from ...workflow._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node,
        parent_ctx=current_ctx,
        execution_id=execution_id,
    )
    return await runner.run(node_input=node_input)

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    # Ensure IC has the agent — needed by downstream code
    # (e.g. _build_response_event reads invocation_context.agent.name).
    # When run via Runner(node=...), ic.agent is None.
    # TODO: remove this dependency.
    if ctx._invocation_context.agent is None:
      ctx._invocation_context.agent = self.agent

    # Always provide our own scheduler so child nodes
    # (LlmCallNode, ParallelToolCallNode) are managed by this node,
    # not by an outer Workflow.
    # TODO: make this a closure and track the created asyncio.Tasks
    ctx._schedule_dynamic_node_internal = self._schedule_node

    # --- ReAct loop ---
    while True:
      # 1. Call LLM
      llm_node = LlmCallNode(agent=self.agent)
      llm_result = await ctx._run_node_internal(llm_node)

      if not llm_result.output or not isinstance(
          llm_result.output, LlmCallResult
      ):
        # LlmCallNode yields output only when the LLM returns function
        # calls.  A pure text response has output=None — the text content
        # event was already yielded (and enqueued) by LlmCallNode, so
        # there is nothing more to do here.  We simply exit the loop.
        break

      # 2. Execute tools
      tool_node = ParallelToolCallNode(
          tools_dict=llm_result.output.tools_dict,
      )
      tool_run_result = await ctx._run_node_internal(
          tool_node, node_input=llm_result.output.function_calls
      )

      # 3. Check termination conditions
      tool_result = tool_run_result.output
      if isinstance(tool_result, ParallelToolCallResult) and (
          tool_result.transfer_to_agent
          or tool_result.request_task
          or tool_result.finish_task
          or tool_result.skip_summarization
      ):
        break

    # AsyncGenerator requires at least one yield.
    # All meaningful events are yielded by child nodes via ctx.run_node.
    return
    yield  # noqa: unreachable — makes this an async generator
