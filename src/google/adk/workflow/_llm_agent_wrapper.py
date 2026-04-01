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
- Runs leaf single_turn agents via SingleAgentReactNode (new Workflow)
  or _SingleLlmAgent (old Workflow)
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
from pydantic import PrivateAttr
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
    single_turn (leaf, no sub_agents, new Workflow): Runs the ReAct
      loop via SingleAgentReactNode. LlmCallNode events are enqueued
      internally; the wrapper post-processes the final text output
      (output_schema validation, output_key storage).
    single_turn (leaf, no sub_agents, old Workflow): Bypasses Mesh by
      running _SingleLlmAgent directly. Output is extracted via
      LlmAgent._maybe_save_output_to_state and emitted as a
      separate Event before END_OF_AGENT.
    single_turn (with sub_agents): Runs the full LlmAgent which
      handles output internally via run_node_impl(). The wrapper
      only suppresses output on interrupt.
    task: The wrapper intercepts finish_task actions and re-emits the
      output as a separate Event, since the original finish_task event
      carries the output inside actions, not in event.output.
  """

  agent: LlmAgent = Field(...)
  rerun_on_resume: bool = Field(default=True)
  _single: Any = PrivateAttr(default=None)
  _react: Any = PrivateAttr(default=None)

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

    # For leaf single_turn agents, prepare both execution paths.
    # The old Workflow (uses _node_runner.py, no event_queue) needs
    # _SingleLlmAgent. The new Workflow (_workflow_class.py, sets
    # event_queue) uses SingleAgentReactNode. The choice is made at
    # runtime in _run_impl based on event_queue presence.
    if self.agent.mode == 'single_turn' and not self.agent.sub_agents:
      from ..agents.llm._single_agent_react_node import SingleAgentReactNode
      from ..agents.llm._single_llm_agent import _SingleLlmAgent

      self._single = _SingleLlmAgent.from_base_llm_agent(self.agent)
      self._react = SingleAgentReactNode(name=self.agent.name, agent=self.agent)

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
      if copied._single is not None:
        copied._single = copied._single.model_copy(
            update={'name': update['name']}
        )
      if copied._react is not None:
        copied._react = copied._react.model_copy(
            update={'name': update['name'], 'agent': copied.agent}
        )
    return copied

  def __eq__(self, other: Any) -> bool:
    """Checks equality based on public fields, ignoring runtime caches."""
    if not isinstance(other, _LlmAgentWrapper):
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
    # event_queue is excluded from model_copy; propagate manually
    # so child NodeRunners can enqueue events.
    ic.event_queue = ctx._invocation_context.event_queue
    agent_ctx = Context(
        invocation_context=ic,
        node_path=ctx.node_path,
        run_id=ctx.run_id,
        resume_inputs=ctx.resume_inputs,
    )
    return agent_ctx, agent_input

  def _use_react_path(self, ctx: Context) -> bool:
    """Returns True if we should use SingleAgentReactNode (new Workflow).

    The new Workflow (_workflow_class.py) runs via Runner._run_node_async
    which sets ic.event_queue. The old Workflow (_workflow.py) runs via
    Runner.run_async (agent path) which does not set event_queue.
    """
    return (
        self._react is not None
        and ctx._invocation_context.event_queue is not None
    )

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

    if self._use_react_path(ctx):
      # New Workflow: leaf single_turn via SingleAgentReactNode.
      # Inject input as user content in session, then run the react
      # loop. LlmCallNode events are enqueued to event_queue internally;
      # only the final text output comes through the generator.
      if agent_input is not None and not ctx.resume_inputs:
        # Only inject on first run — on resume, content is already
        # in session from the first run.
        ic = agent_ctx._invocation_context
        ic.session.events.append(
            Event(
                invocation_id=ic.invocation_id,
                author='user',
                branch=ic.branch,
                content=agent_input,
            )
        )
      output = None
      async for event in self._react.run(ctx=agent_ctx, node_input=None):
        if isinstance(event, Event):
          if event.node_info and event.node_info.message_as_output:
            output = event.message
          elif event.output is not None:
            output = event.output

      if output is None:
        # If output was not yielded, check if it was enqueued to the session
        # via the internal NodeRunner logic.
        for e in reversed(agent_ctx._invocation_context.session.events):
          if getattr(e, 'node_info', None) and e.node_info.message_as_output:
            # We only extract final text output in the wrapper. Function calls
            # are intermediate or represent interrupts, so we ignore them here
            # to let the interrupt correctly propagate (by outputting None).
            has_fc = False
            msg = getattr(e, 'message', None)
            if msg and getattr(msg, 'parts', None):
              has_fc = any(p.function_call for p in msg.parts)
            if not has_fc:
              output = msg
            break

      if output is not None:
        if isinstance(output, types.Content):
          text = (
              ''.join(
                  p.text
                  for p in output.parts
                  if getattr(p, 'text', None)
                  and not getattr(p, 'thought', False)
              )
              if output.parts
              else ''
          )
          if self.agent.output_schema:
            if not text.strip():
              return
            from ..utils._schema_utils import validate_schema

            output = validate_schema(self.agent.output_schema, text)
          else:
            output = text
        elif isinstance(output, str) and self.agent.output_schema:
          if not output.strip():
            return
          from ..utils._schema_utils import validate_schema

          output = validate_schema(self.agent.output_schema, output)
        if self.agent.output_key:
          ctx.actions.state_delta[self.agent.output_key] = output
        # LlmCallNode's content event has message_as_output=True,
        # which auto-sets _output_delegated via NodeRunner. No
        # separate output event will be enqueued.
        yield output
      return
      yield  # noqa: unreachable — keeps this an async generator

    # Determine inner runner: _SingleLlmAgent for leaf, agent for others.
    inner = self._single if self._single is not None else self.agent

    # When the agent has parallel_worker=True, call run_node_impl()
    # directly to bypass Node.run()'s internal parallel logic.
    if self.agent.parallel_worker:
      run_iter = inner.run_node_impl(ctx=agent_ctx, node_input=agent_input)
    else:
      run_iter = inner.run(ctx=agent_ctx, node_input=agent_input)

    if self.agent.mode == 'single_turn':
      if self._single is not None:
        # Old Workflow: leaf agent bypass. Since we skip
        # LlmAgent.run_node_impl(), replicate its output handling.
        # _maybe_save_output_to_state applies output_schema/output_key
        # and clears content on the final response. We emit the output
        # as a pathless Event before END_OF_AGENT.
        node_path = agent_ctx.node_path or ''
        single_output = None
        async for event in run_iter:
          if isinstance(event, Event):
            output_before = event.output
            self.agent._maybe_save_output_to_state(event, node_path)
            if event.output is not None and output_before is None:
              single_output = event.output
          if (
              single_output is not None
              and isinstance(event, Event)
              and event.actions.end_of_agent
          ):
            yield Event(output=single_output)
            single_output = None
          yield event
        if single_output is not None:
          yield Event(output=single_output)
      else:
        # Agent with sub_agents: LlmAgent.run_node_impl() handles
        # output internally. Suppress output when interrupted to
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
        if isinstance(event, Event) and event.actions.finish_task:
          finish_task_output = Event(
              output=event.actions.finish_task.get('output')
          )

      if finish_task_output:
        yield finish_task_output
