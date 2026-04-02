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
import logging
from typing import Any
from typing import TYPE_CHECKING

from ._node_state import NodeState
from ._node_status import NodeStatus
from .utils._node_path_utils import join_paths
from .utils._workflow_hitl_utils import unwrap_response as _unwrap_fr_response

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._base_node import BaseNode


logger = logging.getLogger('google_adk.' + __name__)


@dataclass
class DynamicNodeRun:
  """Combines state, output, and running task for a single node execution."""

  state: NodeState
  """The tracking state (status, interrupts, run_id)."""

  output: Any = None
  """The final output of the node once it completes."""

  task: Optional[asyncio.Task] = None
  """The running asyncio Task for this node execution."""


@dataclass
class DynamicNodeState:
  """State for tracking dynamic nodes scheduled via ctx.run_node().

  Base class for both Workflow's ``_LoopState`` and standalone
  ``DefaultNodeScheduler``. DynamicNodeScheduler reads/writes
  these fields for dedup, resume, and interrupt propagation.
  """

  runs: dict[str, dict[str, DynamicNodeRun]] = field(default_factory=dict)
  """Dynamic node runs. Outer key is node_path, inner key is run_id."""

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

  def get_dynamic_tasks(self) -> list[asyncio.Task]:
    """Get all active dynamic node tasks."""
    return [
        run.task
        for node_runs in self.runs.values()
        for run in node_runs.values()
        if run.task
    ]


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

  def __init__(self, state: DynamicNodeState) -> None:
    self._state = state

  async def __call__(
      self,
      ctx: Context,
      node: BaseNode,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
      run_id: str,
  ) -> Context:
    """Schedule a dynamic node: dedup, resume, or fresh run.

    Args:
      ctx: The calling node's Context.
      node: The BaseNode to execute (original, before renaming).
      _node_name: Deprecated positional. Use ``node_name`` kwarg
        instead. Will be removed in a future cleanup.
      node_input: Input data for the node.
      node_name: Deterministic tracking name from ctx.run_node().
        Always provided (user-specified or auto-generated).
      use_as_output: If True, the child's output replaces the
        calling node's output.
      run_id: Custom run ID for the child node execution.

    Returns:
      Child Context with output, route, and interrupt_ids set.
    """
    # node_name is always provided by ctx.run_node() (either
    # user-specified or auto-generated via _next_child_name).
    name = node_name or node.name
    node_path = join_paths(ctx.node_path, name)

    # Phase 1: Lazy rehydration from session events.
    if (
        node_path not in self._state.runs
        or run_id not in self._state.runs[node_path]
    ):
      self._rehydrate_from_events(ctx, node_path, target_run_id=run_id)

    # Phase 2: Dedup — return cached or handle waiting.
    active_run_id = run_id
    runs_by_run_id = self._state.runs.setdefault(node_path, {})

    if active_run_id not in runs_by_run_id:
      # Phase 3: Fresh execution.
      return await self._execute_fresh(
          ctx,
          node,
          name,
          node_path,
          node_input,
          use_as_output,
          run_id=active_run_id,
      )

    # Found an existing run for this node and run_id -> rerun or interrupt or auto-complete.
    run = runs_by_run_id[active_run_id]
    state = run.state
    if state.status == NodeStatus.COMPLETED:
      # Already completed → return cached output.
      return self._make_cached_ctx(ctx, node_path, active_run_id)

    if state.status == NodeStatus.WAITING:
      # Unresolved interrupts remain → propagate to parent.
      if state.interrupts:
        self._state.interrupt_ids.update(state.interrupts)
        return self._make_cached_ctx(
            ctx,
            node_path,
            active_run_id,
            interrupt_ids=set(state.interrupts),
        )

      # All resolved → re-run or auto-complete.
      if not node.rerun_on_resume:
        # All resolved, no rerun → auto-complete with resume_inputs.
        output = dict(state.resume_inputs)
        state.status = NodeStatus.COMPLETED
        run.output = output
        return self._make_cached_ctx(ctx, node_path, active_run_id)

      # All resolved, rerun → re-execute with resume_inputs.
      return await self._resume_node(
          ctx,
          node,
          name,
          node_path,
          active_run_id,
          node_input,
          use_as_output,
      )

    # Running in this invocation — await existing task.
    if run.task:
      return await run.task

  # --- Lazy scan ---

  def _rehydrate_from_events(
      self, ctx: Context, node_path: str, target_run_id: str
  ) -> None:
    """Scan session events for a dynamic node's prior state for a target run_id."""
    from ._workflow_class import _ChildScanState

    ic = ctx._invocation_context
    invocation_id = ic.invocation_id

    # The single state object for the target run_id.
    target_state = _ChildScanState(run_id=target_run_id)
    # Interrupt IDs belonging to target_run_id.
    interrupt_ids_for_target: set[str] = set()

    for event in ic.session.events:
      if event.invocation_id != invocation_id:
        continue

      # FR events resolve interrupts.
      if event.author == 'user' and event.content and event.content.parts:
        for part in event.content.parts:
          fr = part.function_response
          if fr and fr.id and fr.id in interrupt_ids_for_target:
            target_state.resolved_ids.add(fr.id)
            target_state.resolved_responses[fr.id] = _unwrap_fr_response(
                fr.response
            )
        continue

      # Match events under this dynamic node's path.
      event_node_path = event.node_info.path or ''
      if not event_node_path.startswith(node_path):
        continue

      event_run_id = event.node_info.run_id

      # Direct match or delegated output for target_run_id.
      is_run_id_match = event_run_id == target_run_id
      delegated_to_target = False
      if event.output is not None and event.node_info.output_for:
        for (
            output_for_node_path,
            output_for_run_id,
        ) in event.node_info.output_for:
          if (
              output_for_node_path == node_path
              and output_for_run_id == target_run_id
          ):
            delegated_to_target = True
            break

      is_descendant = (
          event_node_path != node_path and event_node_path.startswith(node_path)
      )
      has_interrupts = bool(event.long_running_tool_ids)

      if (
          not is_run_id_match
          and not delegated_to_target
          and not (is_descendant and has_interrupts)
      ):
        continue

      # Output: direct path or output_for delegation.
      if event.output is not None:
        if (
            event_node_path == node_path and is_run_id_match
        ) or delegated_to_target:
          target_state.output = event.output

      # Interrupts from any descendant.
      # TODO: This logic doesn't work for sub-nodes under parallel worker,
      # where they all have identical node_path.
      if event.long_running_tool_ids:
        if (event_node_path == node_path and is_run_id_match) or is_descendant:
          for interrupt_id in event.long_running_tool_ids:
            target_state.interrupt_ids.add(interrupt_id)
            interrupt_ids_for_target.add(interrupt_id)

    unresolved = target_state.interrupt_ids - target_state.resolved_ids
    state = None
    output = None

    if unresolved:
      state = NodeState(
          status=NodeStatus.WAITING,
          interrupts=list(unresolved),
          run_id=target_run_id,
      )
    elif target_state.interrupt_ids:
      # Node had interrupts, all resolved → ready to re-run.
      state = NodeState(
          status=NodeStatus.WAITING,
          interrupts=[],
          run_id=target_run_id,
          resume_inputs=target_state.resolved_responses,
      )
    elif target_state.output is not None:
      # Node completed with output.
      state = NodeState(
          status=NodeStatus.COMPLETED,
          run_id=target_run_id,
      )
      output = target_state.output

    if state:
      runs_for_path = self._state.runs.setdefault(node_path, {})
      runs_for_path[target_run_id] = DynamicNodeRun(state=state, output=output)

  # --- Context construction ---

  def _make_cached_ctx(
      self,
      ctx: Context,
      node_path: str,
      run_id: str,
      interrupt_ids: set[str] | None = None,
  ) -> Context:
    """Build a Context with cached results (no execution)."""
    from ..agents.context import Context as Ctx

    child_ctx = Ctx(
        ctx._invocation_context,
        node_path=node_path,
        run_id=run_id,
        event_author=ctx.event_author,
        schedule_dynamic_node_internal=(ctx._schedule_dynamic_node_internal),
    )
    run = self._state.runs[node_path][run_id]
    if run.output is not None:
      child_ctx._output_value = run.output
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
      *,
      run_id: str,
  ) -> Context:
    """Run a dynamic node for the first time."""
    from ._node_runner_class import NodeRunner

    state = NodeState(
        status=NodeStatus.RUNNING,
        input=node_input,
        run_id=run_id,
        source_node_name=node.name,
        parent_run_id=ctx.run_id,
    )

    runner = NodeRunner(
        node=node.model_copy(update={'name': name}),
        parent_ctx=ctx,
        run_id=run_id,
        additional_output_for_ancestor=(
            ctx.node_path if use_as_output else None
        ),
    )

    run = DynamicNodeRun(state=state)
    run_by_id = self._state.runs.setdefault(node_path, {})
    run_by_id[run_id] = run

    run.task = asyncio.create_task(runner.run(node_input=node_input))

    child_ctx = await run.task
    self._record_result(run, child_ctx)
    return child_ctx

  async def _resume_node(
      self,
      ctx: Context,
      node: BaseNode,
      name: str,
      node_path: str,
      run_id: str,
      node_input: Any,
      use_as_output: bool,
  ) -> Context:
    """Re-run a dynamic node with resume_inputs."""
    run = self._state.runs[node_path][run_id]
    from ._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node.model_copy(update={'name': name}),
        parent_ctx=ctx,
        run_id=run_id,
        additional_output_for_ancestor=(
            ctx.node_path if use_as_output else None
        ),
    )
    run.state.status = NodeStatus.RUNNING
    run.task = asyncio.create_task(
        runner.run(
            node_input=node_input,
            resume_inputs=dict(run.state.resume_inputs),
        )
    )
    child_ctx = await run.task

    self._record_result(run, child_ctx)
    return child_ctx

  def _record_result(
      self,
      run: DynamicNodeRun,
      child_ctx: Context,
  ) -> None:
    """Update dynamic node state after execution."""
    state = run.state
    if child_ctx.interrupt_ids:
      state.status = NodeStatus.WAITING
      state.interrupts = list(child_ctx.interrupt_ids)
      self._state.interrupt_ids.update(child_ctx.interrupt_ids)
    else:
      state.status = NodeStatus.COMPLETED
      run.output = child_ctx.output


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
