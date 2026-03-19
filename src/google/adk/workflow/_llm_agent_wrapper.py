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

"""Wrapper that adapts an LlmAgent for use as a workflow graph node.

- Sets a branch for content isolation (single_turn mode only)
- Converts node_input to user content (single_turn mode only)
- Re-emits finish_task output so the outer node_runner can route it
"""

from __future__ import annotations

import json
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..agents.context import Context
from ..agents.llm_agent import LlmAgent
from ..events.event import Event
from ._base_node import BaseNode


def _node_input_to_content(node_input: Any) -> types.Content:
  """Converts node_input to a user Content for the LLM agent."""
  if isinstance(node_input, types.Content):
    return node_input
  if isinstance(node_input, str):
    text = node_input
  elif isinstance(node_input, BaseModel):
    text = node_input.model_dump_json()
  elif isinstance(node_input, (dict, list)):
    text = json.dumps(node_input)
  else:
    text = str(node_input)
  return types.Content(role='user', parts=[types.Part(text=text)])


class _LlmAgentWrapper(BaseNode):
  """Adapts a task/single_turn LlmAgent for use as a workflow graph node.

  Output handling by mode:
    single_turn: LlmAgent.run_node_impl() emits an agent-level output
      event (no path) that flows through the wrapper naturally. The
      outer node_runner enriches it with the wrapper's path.
    task: The wrapper intercepts finish_task actions and re-emits the
      output as a separate Event, since the original finish_task event
      carries the output inside actions, not in event.output.
  """

  agent: LlmAgent = Field(...)
  rerun_on_resume: bool = Field(default=True)

  @model_validator(mode='before')
  @classmethod
  def _set_defaults(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if data.get('name') is None and 'agent' in data:
        data['name'] = getattr(data['agent'], 'name', '')
    return data

  @model_validator(mode='after')
  def _validate_and_default_mode(self) -> _LlmAgentWrapper:
    """Defaults unset mode to single_turn; rejects unsupported modes."""
    if self.agent.mode not in ('task', 'single_turn'):
      if 'mode' not in self.agent.model_fields_set:
        self.agent._update_mode('single_turn')
      else:
        raise ValueError(
            f'LlmAgentWrapper only supports task and single_turn mode,'
            f" but agent '{self.agent.name}' has"
            f" mode='{self.agent.mode}'."
        )
    if self.agent.mode == 'task':
      self.wait_for_output = True
    return self

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> _LlmAgentWrapper:
    """Propagates name updates to the inner agent.

    When _ParallelWorker schedules dynamic nodes, each worker gets a
    unique name (e.g. 'agent__0'). The inner agent must also receive
    this name so that events carry the correct author.
    """
    copied = super().model_copy(update=update, deep=deep)
    if update and 'name' in update:
      copied.agent = copied.agent.model_copy(update={'name': update['name']})
    return copied

  def _validate_input(self, node_input: Any) -> None:
    """Validates node_input against the agent's input_schema if set."""
    if not self.agent.input_schema or node_input is None:
      return
    if isinstance(node_input, dict):
      self.agent.input_schema.model_validate(node_input)
    elif isinstance(node_input, BaseModel):
      self.agent.input_schema.model_validate(node_input.model_dump())

  def _prepare_input(
      self, ctx: Context, node_input: Any
  ) -> tuple[Context, Any]:
    """Prepares the agent context and input based on mode.

    Single_turn agents get content isolation via a branch so parallel
    agents don't see each other's events. Task agents skip branching
    because HITL user messages are appended without a branch.
    """
    if self.agent.mode != 'single_turn':
      return ctx, node_input

    node_path = ctx.node_path or ''
    branch = f'node:{node_path}.{self.agent.name}'
    agent_input = (
        _node_input_to_content(node_input) if node_input is not None else None
    )
    ic = ctx._invocation_context.model_copy(
        update={'branch': branch},
    )
    agent_ctx = Context(
        invocation_context=ic,
        node_path=ctx.node_path,
        execution_id=ctx.execution_id,
    )
    return agent_ctx, agent_input

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the wrapped agent and translates output for downstream nodes."""
    self._validate_input(node_input)
    agent_ctx, agent_input = self._prepare_input(ctx, node_input)

    # When the agent has parallel_worker=True, call run_node_impl()
    # directly to bypass Node.run()'s internal parallel logic.
    if self.agent.parallel_worker:
      run_iter = self.agent.run_node_impl(ctx=agent_ctx, node_input=agent_input)
    else:
      run_iter = self.agent.run(ctx=agent_ctx, node_input=agent_input)

    if self.agent.mode == 'single_turn':
      # Output flows through naturally: LlmAgent.run_node_impl() emits
      # a path-less Event(output=...) that the outer node_runner enriches.
      # Suppress output when interrupted (long-running tools, HITL) to
      # avoid mixed output/interrupt errors in node_runner.
      interrupted = False
      async for event in run_iter:
        if isinstance(event, Event) and event.long_running_tool_ids:
          interrupted = True
        if (
            interrupted
            and isinstance(event, Event)
            and event.output is not None
            and not event.node_info.path
        ):
          continue
        yield event
    else:
      # Task mode: finish_task output is inside event.actions, not
      # event.output. Intercept it and re-emit as a proper output event.
      finish_task_output = None
      async for event in run_iter:
        yield event
        if (
            isinstance(event, Event)
            and event.actions
            and event.actions.finish_task
        ):
          finish_task_output = Event(
              output=event.actions.finish_task.get('output')
          )

      if finish_task_output:
        yield finish_task_output
