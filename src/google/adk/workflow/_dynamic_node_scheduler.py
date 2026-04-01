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

"""Dynamic node scheduler for Workflow.

Handles ctx.run_node() calls by tracking dynamic nodes in the
Workflow's _LoopState. Supports dedup (cached output), resume
(lazy event scan + re-run), and fresh execution.

Also provides ``DefaultNodeScheduler``, a lightweight scheduler
that any BaseNode can use to manage dynamic children via
``ctx.run_node()``. DefaultNodeScheduler reuses all
DynamicNodeScheduler logic but owns its own state instead of
sharing a Workflow's _LoopState.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

from ._node_state import NodeState
from ._node_status import NodeStatus
from .utils._node_path_utils import join_paths
from .utils._workflow_hitl_utils import unwrap_response as _unwrap_fr_response

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._base_node import BaseNode


@dataclass
class DynamicNodeState:
  """State for tracking dynamic nodes scheduled via ctx.run_node().

  Base class for both Workflow's ``_LoopState`` and standalone
  ``DefaultNodeScheduler``. DynamicNodeScheduler reads/writes
  these fields for dedup, resume, and interrupt propagation.
  """

  dynamic_nodes: dict[str, NodeState] = field(default_factory=dict)
  """Dynamic node states. Populated lazily by the schedule
  callback on first ctx.run_node() call per name."""

  dynamic_outputs: dict[str, Any] = field(default_factory=dict)
  """Cached dynamic node outputs."""

  dynamic_pending_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
  """Running dynamic node tasks."""

  # --- Shared (static + dynamic) ---

  interrupt_ids: set[str] = field(default_factory=set)
  """Union of all unresolved interrupt IDs across static and
  dynamic child nodes.

  Populated by:
  - _restore_static_nodes_from_events: from WAITING static nodes
  - _handle_completion: when a static node interrupts at runtime
  - schedule callback: when a dynamic node interrupts

  Read by _finalize to propagate to the Workflow's own ctx,
  which the parent orchestrator checks after this Workflow
  completes.
  """


class DynamicNodeScheduler:
  """Handles ctx.run_node() calls for a Workflow.

  Implements ScheduleDynamicNode protocol via __call__. Tracks
  dynamic nodes in loop_state, handles dedup via lazy event
  scanning, and manages resume/interrupt propagation.

  Three cases:
  1. Fresh: no prior events → execute normally.
  2. Completed: prior events show output → return cached.
  3. Waiting: prior events show interrupt → resolve or propagate.
  """

  def __init__(self, loop_state: DynamicNodeState) -> None:
    self._loop_state = loop_state

  async def __call__(
      self,
      ctx: Context,
      node: BaseNode,
      run_id: str,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
  ) -> Context:
    """Schedule a dynamic node: dedup, resume, or fresh run.

    Args:
      ctx: The calling node's Context.
      node: The BaseNode to execute (original, before renaming).
      run_id: Unused. Kept for protocol compat; NodeRunner
        generates its own.
      node_input: Input data for the node.
      node_name: Deterministic tracking name from ctx.run_node().
        Always provided (user-specified or auto-generated).
      use_as_output: If True, the child's output replaces the
        calling node's output.

    Returns:
      Child Context with output, route, and interrupt_ids set.
    """
    # node_name is always provided by ctx.run_node() (either
    # user-specified or auto-generated via _next_child_name).
    name = node_name or node.name  # fallback for safety
    node_path = join_paths(ctx.node_path, name)

    # Phase 1: Lazy rehydration from session events.
    if node_path not in self._loop_state.dynamic_nodes:
      self._rehydrate_from_events(ctx, node_path)

    # Phase 2: Dedup — return cached or handle waiting.
    if node_path in self._loop_state.dynamic_nodes:
      state = self._loop_state.dynamic_nodes[node_path]

      # Already completed → return cached output.
      if state.status == NodeStatus.COMPLETED:
        return self._make_cached_ctx(ctx, node_path, state)

      if state.status == NodeStatus.WAITING:
        # Unresolved interrupts remain → propagate to parent.
        if state.interrupts:
          self._loop_state.interrupt_ids.update(state.interrupts)
          return self._make_cached_ctx(
              ctx,
              node_path,
              state,
              interrupt_ids=set(state.interrupts),
          )
        # All resolved → re-run or auto-complete.
        if not node.rerun_on_resume:
          # All resolved, no rerun → auto-complete with resume_inputs.
          output = dict(state.resume_inputs)
          state.status = NodeStatus.COMPLETED
          self._loop_state.dynamic_outputs[node_path] = output
          return self._make_cached_ctx(ctx, node_path, state)
        # All resolved, rerun → re-execute with resume_inputs.
        return await self._resume_node(
            ctx,
            node,
            name,
            node_path,
            state,
            node_input,
            use_as_output,
        )

      # Running in this invocation — await existing task.
      if node_path in self._loop_state.dynamic_pending_tasks:
        return await self._loop_state.dynamic_pending_tasks[node_path]

    # Phase 3: Fresh execution.
    return await self._execute_fresh(
        ctx,
        node,
        name,
        node_path,
        node_input,
        use_as_output,
    )

  # --- Lazy scan ---

  def _rehydrate_from_events(self, ctx: Context, node_path: str) -> None:
    """Scan session events for a dynamic node's prior state."""
    from ._workflow_class import _ChildScanState

    ic = ctx._invocation_context
    invocation_id = ic.invocation_id
    dynamic_child = _ChildScanState()
    has_prior_events = False

    for event in ic.session.events:
      if event.invocation_id != invocation_id:
        continue

      # FR events resolve interrupts. FR events are user
      # messages with no node_path — can't filter by path,
      # so we match by interrupt ID instead.
      if event.author == 'user' and event.content and event.content.parts:
        for part in event.content.parts:
          fr = part.function_response
          if fr and fr.id and fr.id in dynamic_child.interrupt_ids:
            # Safe: interrupt events always precede their FRs
            # chronologically, so interrupt_ids is already
            # populated.
            dynamic_child.resolved_ids.add(fr.id)
            dynamic_child.resolved_responses[fr.id] = _unwrap_fr_response(
                fr.response
            )
        continue

      # Match events under this dynamic node's path.
      event_path = event.node_info.path or ''
      if not event_path.startswith(node_path):
        continue

      has_prior_events = True
      dynamic_child.run_id = event.node_info.run_id or dynamic_child.run_id

      # Output: direct path or output_for delegation.
      if event.output is not None:
        if event_path == node_path:
          dynamic_child.output = event.output
        elif (
            event.node_info.output_for
            and node_path in event.node_info.output_for
        ):
          dynamic_child.output = event.output

      # Interrupts from any descendant.
      if event.long_running_tool_ids:
        dynamic_child.interrupt_ids.update(event.long_running_tool_ids)

    if not has_prior_events:
      return  # No prior events → fresh execution.

    # Derive NodeState from scan — same logic as static nodes.
    unresolved = dynamic_child.interrupt_ids - dynamic_child.resolved_ids
    existing_run_id = dynamic_child.run_id

    if unresolved:
      # Node still has unresolved interrupts.
      self._loop_state.dynamic_nodes[node_path] = NodeState(
          status=NodeStatus.WAITING,
          interrupts=list(unresolved),
          run_id=existing_run_id,
      )
    elif dynamic_child.interrupt_ids:
      # Node had interrupts, all resolved → ready to re-run.
      # TODO: If the node already re-ran and completed with
      # None output, the scan can't tell — no events were
      # emitted. This causes an unnecessary re-run. Fix by
      # emitting a completion marker event from NodeRunner.
      self._loop_state.dynamic_nodes[node_path] = NodeState(
          status=NodeStatus.WAITING,
          interrupts=[],
          run_id=existing_run_id,
          resume_inputs=dynamic_child.resolved_responses,
      )
    elif dynamic_child.output is not None:
      # Node completed with output.
      self._loop_state.dynamic_nodes[node_path] = NodeState(
          status=NodeStatus.COMPLETED,
          run_id=existing_run_id,
      )
      self._loop_state.dynamic_outputs[node_path] = dynamic_child.output

  # --- Context construction ---

  def _make_cached_ctx(
      self,
      ctx: Context,
      node_path: str,
      state: NodeState,
      interrupt_ids: set[str] | None = None,
  ) -> Context:
    """Build a Context with cached results (no execution)."""
    from ..agents.context import Context as Ctx

    child_ctx = Ctx(
        ctx._invocation_context,
        node_path=node_path,
        run_id=state.run_id or '',
        event_author=ctx.event_author,
        schedule_dynamic_node_internal=(ctx._schedule_dynamic_node_internal),
    )
    if node_path in self._loop_state.dynamic_outputs:
      child_ctx._output_value = self._loop_state.dynamic_outputs[node_path]
      child_ctx._output_emitted = True
    if interrupt_ids:
      child_ctx._interrupt_ids = set(interrupt_ids)
    return child_ctx

  # --- Execution ---

  async def _execute_fresh(
      self,
      ctx: Context,
      node: BaseNode,
      name: str,
      node_path: str,
      node_input: Any,
      use_as_output: bool,
  ) -> Context:
    """Run a dynamic node for the first time."""
    from ._node_runner_class import NodeRunner

    state = NodeState(
        status=NodeStatus.RUNNING,
        input=node_input,
        # TODO: Right now, dynamic node with same node path is only run once.
        run_id='1',
        source_node_name=node.name,
        parent_run_id=ctx.run_id,
    )

    runner = NodeRunner(
        node=node.model_copy(update={'name': name}),
        parent_ctx=ctx,
        run_id=state.run_id,
        additional_output_for_ancestor=(
            ctx.node_path if use_as_output else None
        ),
    )
    self._loop_state.dynamic_nodes[node_path] = state
    task = asyncio.create_task(runner.run(node_input=node_input))
    self._loop_state.dynamic_pending_tasks[node_path] = task

    child_ctx = await task
    self._record_result(node_path, state, child_ctx)
    return child_ctx

  async def _resume_node(
      self,
      ctx: Context,
      node: BaseNode,
      name: str,
      node_path: str,
      state: NodeState,
      node_input: Any,
      use_as_output: bool,
  ) -> Context:
    """Re-run a dynamic node with resume_inputs."""
    from ._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node.model_copy(update={'name': name}),
        parent_ctx=ctx,
        run_id=state.run_id,
        additional_output_for_ancestor=(
            ctx.node_path if use_as_output else None
        ),
    )
    state.status = NodeStatus.RUNNING
    task = asyncio.create_task(
        runner.run(
            node_input=node_input,
            resume_inputs=dict(state.resume_inputs),
        )
    )
    self._loop_state.dynamic_pending_tasks[node_path] = task

    child_ctx = await task
    self._record_result(node_path, state, child_ctx)
    return child_ctx

  def _record_result(
      self,
      node_path: str,
      state: NodeState,
      child_ctx: Context,
  ) -> None:
    """Update dynamic node state after execution."""
    if child_ctx.interrupt_ids:
      state.status = NodeStatus.WAITING
      state.interrupts = list(child_ctx.interrupt_ids)
      self._loop_state.interrupt_ids.update(child_ctx.interrupt_ids)
    else:
      state.status = NodeStatus.COMPLETED
      self._loop_state.dynamic_outputs[node_path] = child_ctx.output


# ---------------------------------------------------------------------------
# DefaultNodeScheduler — default scheduler for standalone nodes
# ---------------------------------------------------------------------------


class DefaultNodeScheduler(DynamicNodeScheduler):
  """Default scheduler for nodes without a Workflow.

  Enables ``ctx.run_node()`` to work correctly on re-run
  (``rerun_on_resume=True``) by tracking dynamic children: caching
  completed outputs, forwarding resume_inputs for interrupted
  children, and propagating unresolved interrupts.

  Reuses all DynamicNodeScheduler logic but owns its own
  ``DynamicNodeState`` instead of sharing a Workflow's ``_LoopState``.

  State starts empty — on resume, ``_rehydrate_from_events()``
  lazily reconstructs child states from session events, so no
  external state injection is needed.
  """

  def __init__(self) -> None:
    super().__init__(DynamicNodeState())
