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

"""Utility functions for running LlmAgent as a workflow node."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
import json
from typing import Any

from google.genai import types
from pydantic import BaseModel

from ..agents.context import Context
from ..events.event import Event
from ..utils._schema_utils import validate_schema


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


def prepare_llm_agent_context(agent: Any, ctx: Context) -> Context:
  """Prepares the context for running LlmAgent as a node."""
  if agent.mode != 'single_turn':
    return ctx

  ic = ctx._invocation_context.model_copy()
  ic.event_queue = ctx._invocation_context.event_queue
  agent_ctx = Context(
      invocation_context=ic,
      node_path=ctx.node_path,
      run_id=ctx.run_id,
      resume_inputs=ctx.resume_inputs,
  )

  ic.session = ic.session.model_copy(deep=False)
  return agent_ctx


def prepare_llm_agent_input(agent: Any, ctx: Context, node_input: Any) -> None:
  """Prepares the input for running LlmAgent as a node."""
  if node_input is not None and agent.mode == 'single_turn':
    agent_input = _node_input_to_content(node_input)
    user_event = Event(author='user', message=agent_input)
    if user_event.content is not None:
      user_event.content.role = 'user'
    user_event.branch = ctx._invocation_context.branch
    ctx.session.events.append(user_event)


def process_llm_agent_output(agent: Any, ctx: Context, event: Event) -> None:
  """Processes the output of LlmAgent run as a node."""
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
      ''.join(p.text for p in event.content.parts if p.text and not p.thought)
      if event.content.parts
      else ''
  )
  if agent.output_schema:
    if text.strip():
      output = validate_schema(agent.output_schema, text)
    else:
      output = None
  else:
    output = text

  if agent.output_key and output is not None:
    ctx.actions.state_delta[agent.output_key] = output

  event.output = output


async def run_llm_agent_as_node(
    agent: Any,
    *,
    ctx: Context,
    node_input: Any,
) -> AsyncGenerator[Any, None]:
  """Runs an LlmAgent as a workflow node."""
  # As a node in a workflow, agent is by default single_turn.
  if agent.mode is None:
    agent.mode = 'single_turn'

  if agent.mode not in ('task', 'single_turn', 'chat'):
    raise ValueError(
        f'LlmAgent as node only supports task, single_turn, and chat mode,'
        f" but agent '{agent.name}' has mode='{agent.mode}'."
    )

  if agent.mode == 'single_turn':
    agent.include_contents = 'none'

  agent_ctx = prepare_llm_agent_context(agent, ctx)
  prepare_llm_agent_input(agent, agent_ctx, node_input)

  ic = agent_ctx.get_invocation_context()
  ic = ic.model_copy(update={'agent': agent})
  run_iter = aclosing(agent.run_async(ic))

  async with run_iter as run_iter:
    if agent.mode == 'single_turn':
      async for event in run_iter:
        process_llm_agent_output(agent, ctx, event)
        yield event
    elif agent.mode == 'chat':
      async for event in run_iter:
        yield event
        if event.actions.request_task:
          for fc_id, task_req in event.actions.request_task.items():
            target_name = task_req.get('agentName')
            if target_name:
              target_agent = agent.root_agent.find_agent(target_name)
              if target_agent:
                from .utils._workflow_graph_utils import build_node

                wrapped_target = build_node(target_agent)
                wrapped_target.parent_agent = target_agent.parent_agent

                # TODO: decide branch format and have a util for it.
                override_branch = f'task:{fc_id}'
                await ctx.run_node(
                    wrapped_target,
                    node_input=None,
                    override_branch=override_branch,
                )
                if ctx._invocation_context.is_resumable:
                  ctx._invocation_context.set_agent_state(
                      agent.name, end_of_agent=True
                  )
                  yield agent._create_agent_state_event(ctx._invocation_context)
          break
        if event.actions.transfer_to_agent:
          target_name = event.actions.transfer_to_agent
          if target_name != agent.name:
            target_agent = agent.root_agent.find_agent(target_name)
            if target_agent:
              from .utils._workflow_graph_utils import build_node

              wrapped_target = build_node(target_agent)
              wrapped_target.parent_agent = target_agent.parent_agent
              await ctx.run_node(wrapped_target, node_input=None)
              if ctx._invocation_context.is_resumable:
                ctx._invocation_context.set_agent_state(
                    agent.name, end_of_agent=True
                )
                yield agent._create_agent_state_event(ctx._invocation_context)
              break
    else:
      # Task mode: finish_task output is inside event.actions, not
      # event.output. Intercept it and set as a proper output event.
      async for event in run_iter:
        if event.actions.finish_task:
          event.output = event.actions.finish_task.get('output')
          yield event

          if agent.parent_agent:
            from .utils._workflow_graph_utils import build_node

            wrapped_parent = build_node(agent.parent_agent)
            wrapped_parent.parent_agent = agent.parent_agent.parent_agent
            await ctx.run_node(wrapped_parent, node_input=None)

            if ctx._invocation_context.is_resumable:
              ctx._invocation_context.set_agent_state(
                  agent.name, end_of_agent=True
              )
              yield agent._create_agent_state_event(ctx._invocation_context)
          break
        yield event
