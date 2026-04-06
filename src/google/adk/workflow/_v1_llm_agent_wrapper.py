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

import json
from typing import Any
from typing import AsyncGenerator

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..agents.context import Context
from ..agents.llm_agent_1x import LlmAgent as V1LlmAgent
from ..events.event import Event
from ..utils._schema_utils import validate_schema
from ._base_node import BaseNode


def _node_input_to_content(node_input: Any) -> types.Content:
  """Converts node_input to a user Content for the LLM agent."""
  if isinstance(node_input, types.Content):
    return types.Content(role='user', parts=node_input.parts)
  if isinstance(node_input, str):
    text = node_input
  elif isinstance(node_input, BaseModel):
    text = node_input.model_dump_json()
  elif isinstance(node_input, (dict, list)):
    text = json.dumps(node_input)
  else:
    text = str(node_input)
  return types.Content(role='user', parts=[types.Part(text=text)])


class _V1LlmAgentWrapper(BaseNode):
  """Adapts a V1 LlmAgent for use as a workflow graph node.

  This wrapper handles V1 agents specifically, isolation for single_turn mode,
  and ensuring output is correctly flagged for the workflow runner.
  """

  agent: Any = Field(...)
  rerun_on_resume: bool = Field(default=True)

  @model_validator(mode='before')
  @classmethod
  def _set_defaults(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if data.get('name') is None and 'agent' in data:
        data['name'] = getattr(data['agent'], 'name', '')
    return data

  @model_validator(mode='after')
  def _validate_mode(self) -> _V1LlmAgentWrapper:
    # As a node in a workflow, agent is by default single_turn.
    if self.agent.mode is None:
      self.agent.mode = 'single_turn'

    if self.agent.mode not in ('task', 'single_turn', 'chat'):
      raise ValueError(
          f'_V1LlmAgentWrapper only supports task, single_turn, and chat mode,'
          f" but agent '{self.agent.name}' has mode='{self.agent.mode}'."
      )

    if self.agent.mode == 'single_turn':
      self.agent.include_contents = 'none'
    if self.agent.mode in ('task', 'chat'):
      self.wait_for_output = True
    return self

  def _prepare_context(self, ctx: Context) -> Context:
    if self.agent.mode != 'single_turn':
      return ctx

    node_path = ctx.node_path or ''
    branch = f'node:{node_path}.{self.agent.name}'
    ic = ctx._invocation_context.model_copy(
        update={'branch': branch},
    )
    ic.event_queue = ctx._invocation_context.event_queue
    agent_ctx = Context(
        invocation_context=ic,
        node_path=ctx.node_path,
        run_id=ctx.run_id,
        resume_inputs=ctx.resume_inputs,
    )

    ic.session = ic.session.model_copy(deep=False)
    return agent_ctx

  def _prepare_input(self, ctx: Context, node_input: Any) -> None:
    if node_input is not None and self.agent.mode == 'single_turn':
      agent_input = _node_input_to_content(node_input)
      user_event = Event(author='user', message=agent_input)
      user_event.content.role = 'user'
      user_event.branch = ctx._invocation_context.branch
      ctx.session.events.append(user_event)

  def _process_output(self, ctx: Context, event: Event) -> None:
    if (
        event.get_function_calls()
        or event.partial
        or not event.content
        or event.content.role != 'model'
    ):
      return

    event.node_info.message_as_output = True

    output = None
    text = (
        ''.join(
            p.text
            for p in event.content.parts
            if getattr(p, 'text', None) and not getattr(p, 'thought', False)
        )
        if event.content.parts
        else ''
    )
    if self.agent.output_schema:
      if text.strip():
        output = validate_schema(self.agent.output_schema, text)
      else:
        output = None
    else:
      output = text

    if self.agent.output_key and output is not None:
      ctx.actions.state_delta[self.agent.output_key] = output

    event.output = output

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    agent_ctx = self._prepare_context(ctx)
    self._prepare_input(agent_ctx, node_input)

    run_iter = self.agent.run_async(agent_ctx._invocation_context)

    if self.agent.mode == 'single_turn':
      async for event in run_iter:
        self._process_output(ctx, event)
        yield event
    elif self.agent.mode == 'chat':
      async for event in run_iter:
        yield event
        if event.actions.transfer_to_agent:
          target_name = event.actions.transfer_to_agent
          if target_name != self.agent.name:
            target_agent = self.agent.root_agent.find_agent(target_name)
            if target_agent:
              wrapped_target = _V1LlmAgentWrapper(agent=target_agent)
              await ctx.run_node(wrapped_target, node_input=None)
              break
    else:
      # Task mode: finish_task output is inside event.actions, not
      # event.output. Intercept it and set as a proper output event.
      async for event in run_iter:
        if event.actions.finish_task:
          event.output = event.actions.finish_task.get('output')
          yield event
          break
        yield event

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> _V1LlmAgentWrapper:
    copied = super().model_copy(update=update, deep=deep)
    if update and 'name' in update:
      copied.agent = copied.agent.model_copy(update={'name': update['name']})
    return copied

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, _V1LlmAgentWrapper):
      return False
    return (
        self.name == other.name
        and self.description == other.description
        and self.rerun_on_resume == other.rerun_on_resume
        and self.wait_for_output == other.wait_for_output
        and self.retry_config == other.retry_config
        and self.timeout == other.timeout
        and self.input_schema == other.input_schema
        and self.output_schema == other.output_schema
        and self.agent == other.agent
    )
