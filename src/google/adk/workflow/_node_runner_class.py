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

"""NodeRunner — per-node executor class.

Converts BaseNode.run() (async generator) into an awaitable that returns
NodeRunResult. Used internally by orchestrators (WorkflowNode, MeshNode,
LlmAgent) that need output, route, and interrupt info.

User-facing ctx.run_node() wraps this and returns just the output.
"""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..agents.context import ScheduleDynamicNode
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ._base_node import BaseNode
  from ._node_run_result import NodeRunResult


class NodeRunner:
  """Per-node executor. Drives BaseNode.run(), enriches events, returns result.

  Creates child Context, iterates node.run(), enqueues events to
  ic.event_queue, detects interrupts, and returns NodeRunResult.
  """

  def __init__(
      self,
      *,
      node: BaseNode,
      parent_ctx: Context,
      execution_id: str | None = None,
      triggered_by: str = '',
      in_nodes: set[str] | None = None,
      additional_output_for_ancestor: str | None = None,
  ) -> None:
    from ..platform import uuid as platform_uuid

    self._node = node
    self._parent_ctx = parent_ctx
    self._execution_id = execution_id or platform_uuid.new_uuid()
    self._triggered_by = triggered_by
    self._in_nodes = in_nodes
    self._additional_output_for_ancestor = additional_output_for_ancestor

  @property
  def execution_id(self) -> str:
    """The execution ID assigned to this node run."""
    return self._execution_id

  async def run(
      self,
      node_input: Any = None,
      *,
      resume_inputs: dict[str, Any] | None = None,
  ) -> NodeRunResult:
    """Drive node.run(), enqueue events, return structured result."""
    from ._node_run_result import NodeRunResult

    ctx = self._create_child_context(resume_inputs)

    output, route, interrupt_ids = await self._execute_node(ctx, node_input)
    await self._emit_remaining_deltas(ctx)

    return NodeRunResult(
        output=output,
        route=route,
        interrupt_ids=interrupt_ids,
    )

  def _build_node_path(self) -> str:
    """Construct this node's path from parent context."""
    from .utils._node_path_utils import join_paths

    return join_paths(self._parent_ctx.node_path or None, self._node.name)

  def _create_child_context(
      self,
      resume_inputs: dict[str, Any] | None,
  ) -> Context:
    """Create a child Context for the node, inheriting from parent."""
    from ..agents.context import Context

    ancestors = list(self._parent_ctx._output_for_ancestors or [])
    if self._additional_output_for_ancestor:
      ancestors = [self._additional_output_for_ancestor] + ancestors

    return Context(
        self._parent_ctx._invocation_context,
        node_path=self._build_node_path(),
        execution_id=self._execution_id,
        resume_inputs=resume_inputs,
        schedule_dynamic_node_internal=self._parent_ctx._schedule_dynamic_node_internal,
        triggered_by=self._triggered_by,
        in_nodes=self._in_nodes,
        output_for_ancestors=ancestors,
    )

  async def _execute_node(
      self,
      ctx: Context,
      node_input: Any,
  ) -> tuple[Any, Any, list[str]]:
    """Iterate node.run(), enqueue events, capture output/route/interrupts."""
    output = None
    route = None
    interrupt_ids: list[str] = []
    async for event in self._node.run(ctx=ctx, node_input=node_input):
      # Skip the parent's output event when output is delegated via
      # use_as_output. The child already emitted this output. Check
      # before _flush_deltas so pending deltas stay in ctx and are
      # emitted by _emit_remaining_deltas.
      if event.output is not None and ctx._output_delegated:
        output = event.output
        continue

      if not event.partial:
        self._flush_deltas(event, ctx)
      self._enrich_event(event, ctx)

      await ctx._invocation_context.enqueue_event(event)

      if event.output is not None:
        if output is not None:
          raise ValueError(
              f'Node {self._node.name} (node_path='
              f'{ctx.node_path}): a node can yield at most one output.'
          )
        output = event.output
      if event.actions and event.actions.route is not None:
        route = event.actions.route
      if event.long_running_tool_ids is not None:
        interrupt_ids.extend(event.long_running_tool_ids)

    return output, route, interrupt_ids

  async def _emit_remaining_deltas(self, ctx: Context) -> None:
    """Emit any deltas that weren't flushed onto a yielded event."""
    from ..events.event import Event
    from ..events.event_actions import EventActions

    state = ctx.actions.state_delta
    artifact = ctx.actions.artifact_delta
    if not state and not artifact:
      return

    delta_event = Event(
        actions=EventActions(
            state_delta=dict(state),
            artifact_delta=dict(artifact),
        ),
    )
    state.clear()
    artifact.clear()
    self._enrich_event(delta_event, ctx)
    await ctx._invocation_context.enqueue_event(delta_event)

  def _flush_deltas(self, event: Event, ctx: Context) -> None:
    """Move pending state/artifact deltas from ctx onto the event."""
    from ..events.event_actions import EventActions

    state = ctx.actions.state_delta
    artifact = ctx.actions.artifact_delta
    if not state and not artifact:
      return

    if not event.actions:
      event.actions = EventActions()
    if state:
      event.actions.state_delta.update(state)
      state.clear()
    if artifact:
      event.actions.artifact_delta.update(artifact)
      artifact.clear()

  def _enrich_event(self, event: Event, ctx: Context) -> None:
    """Set author, node_info, invocation_id on the event."""
    if not event.author:
      # TODO: Use ctx.event_author when available (PR 6) so
      # orchestrators (WorkflowNode, MeshNode) can stamp their name
      # on all child node events.
      event.author = self._node.name
    event.invocation_id = ctx._invocation_context.invocation_id
    event.node_info.path = ctx.node_path
    event.node_info.execution_id = self._execution_id
    if event.output is not None:
      event.node_info.output_for = [ctx.node_path] + ctx._output_for_ancestors
