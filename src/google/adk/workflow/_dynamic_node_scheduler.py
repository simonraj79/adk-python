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
Workflow's _LoopState or a local DynamicNodeState. Supports dedup
(cached output), resume (lazy event scan + re-run), and fresh execution.
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
from ._node_path_builder import _NodePathBuilder
from .utils._rehydration_utils import _ChildScanState
from .utils._rehydration_utils import _scan_node_events

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._base_node import BaseNode


logger = logging.getLogger('google_adk.' + __name__)


@dataclass(kw_only=True)
class DynamicNodeRun:
  """Combines state, output, and running task for a single node execution."""

  state: NodeState
  """The tracking state (status, interrupts, run_id)."""

  output: Any = None
  """The final output of the node once it completes."""

  task: asyncio.Task[Context] | None = None
  """The running asyncio Task for this node execution."""


@dataclass(kw_only=True)
class DynamicNodeState:
  """State for tracking dynamic nodes scheduled via ctx.run_node().

  Base class for both Workflow's ``_LoopState`` and standalone
  ``DefaultNodeScheduler``. DynamicNodeScheduler reads/writes
  these fields for dedup, resume, and interrupt propagation.
  """

  runs: dict[str, DynamicNodeRun] = field(default_factory=dict)
  """Dynamic node runs keyed by unique node_path (e.g. /wf@1/node_a@1)."""

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

  def get_dynamic_tasks(self) -> list[asyncio.Task[Context]]:
    """Get all active dynamic node tasks."""
    return [run.task for run in self.runs.values() if run.task]


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

  def __init__(self, *, state: DynamicNodeState) -> None:
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
      is_parallel: bool = False,
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
      is_parallel: Whether the node is running in parallel.

    Returns:
      Child Context with output, route, and interrupt_ids set.
    """
    # node_name is always provided by ctx.run_node() (either
    # user-specified or auto-generated via _next_child_name).
    name = node_name or node.name
    base_path_builder = _NodePathBuilder.from_string(ctx.node_path) if ctx.node_path else _NodePathBuilder([])
    node_path = str(base_path_builder.append(name, run_id))

    # Runtime schema validation.
    if node_input is not None:
      try:
        node_input = node._validate_input_data(node_input)
      except Exception as e:
        raise ValueError(
            f"Runtime schema validation failed for dynamic node '{name}'."
            f' Input does not match input_schema: {e}'
        ) from e

    logger.info('node %s schedule start.', node_path)

    # Phase 1: Lazy rehydration from session events.
    if node_path not in self._state.runs:
      self._rehydrate_from_events(ctx, node_path)

    if node_path not in self._state.runs:
      # Phase 3: Fresh execution.
      logger.info('node %s schedule end: Fresh execution.', node_path)
      return await self._run_node_internal(
          ctx,
          node,
          name,
          node_path,
          run_id,
          node_input,
          use_as_output,
          is_fresh=True,
          is_parallel=is_parallel,
      )

    # Found an existing run for this node and run_id -> rerun, interrupt,
    # or auto-complete.
    run = self._state.runs[node_path]
    state = run.state
    if state.status == NodeStatus.COMPLETED:
      # Already completed → return cached output.
      logger.info('node %s schedule end: Already completed.', node_path)
      return self._make_cached_ctx(ctx, node_path, run_id)

    if state.status == NodeStatus.WAITING:
      # Unresolved interrupts remain → propagate to parent.
      if state.interrupts:
        self._state.interrupt_ids.update(state.interrupts)
        logger.info(
            'node %s schedule end: Unresolved interrupts remain.', node_path
        )
        return self._make_cached_ctx(
            ctx,
            node_path,
            run_id,
            interrupt_ids=set(state.interrupts),
        )

      # All resolved → re-run or auto-complete.
      if node.wait_for_output and not node.rerun_on_resume:
        raise ValueError(
            f'Node {node_path} is waiting for output but was called again with'
            ' rerun_on_resume=False. This would cause it to auto-complete with'
            ' empty output, which is likely a configuration error. Consider'
            ' setting rerun_on_resume=True.'
        )

      if not node.rerun_on_resume:
        # All resolved, no rerun → auto-complete with resume_inputs.
        if len(state.resume_inputs) == 1:
          output = list(state.resume_inputs.values())[0]
        else:
          output = dict(state.resume_inputs)
        state.status = NodeStatus.COMPLETED
        run.output = output
        logger.info(
            'node %s schedule end: Auto-complete with resume_inputs.', node_path
        )
        return self._make_cached_ctx(ctx, node_path, run_id)

      # All resolved, rerun → re-execute with resume_inputs.
      logger.info(
          'node %s schedule end: Re-execute with resume_inputs.', node_path
      )
      return await self._run_node_internal(
          ctx,
          node,
          name,
          node_path,
          run_id,
          node_input,
          use_as_output,
          is_fresh=False,
          is_parallel=is_parallel,
      )

    # Running in this invocation — await existing task.
    if run.task:
      logger.info('node %s schedule end: Awaiting existing task.', node_path)
      return await run.task

    raise RuntimeError(
        f'Dynamic node {node_path} is in state'
        f' {run.state.status} with no task.'
    )

  # --- Lazy scan ---

  def _rehydrate_from_events(self, ctx: Context, node_path: str) -> None:
    """Scan session events for a dynamic node's prior state."""
    logger.info('node %s rehydrate start.', node_path)
    ic = ctx._invocation_context

    results = _scan_node_events(
        events=ic.session.events,
        base_path=node_path,
        group_by_direct_child=False,
    )

    target_state = results.get(node_path)
    if not target_state:
      extracted_run_id = _NodePathBuilder.from_string(node_path).run_id
      target_state = _ChildScanState(run_id=extracted_run_id)

    unresolved = target_state.interrupt_ids - target_state.resolved_ids
    state = None
    output = None

    if unresolved:
      state = NodeState(
          status=NodeStatus.WAITING,
          interrupts=list(unresolved),
          run_id=target_state.run_id,
      )
    elif target_state.output is not None:
      # Node had all interrupts resolved and completed with output.
      # NOTE: We assume if a node has output, it is complete. If a node yields
      # output and then interrupts (Output -> Interrupt), it is an invalid/
      # unsupported use case in this engine.
      state = NodeState(
          status=NodeStatus.COMPLETED,
          run_id=target_state.run_id,
      )
      output = target_state.output
    elif target_state.interrupt_ids:
      # Node had interrupts, all resolved, and no output yet → ready to re-run.
      state = NodeState(
          status=NodeStatus.WAITING,
          interrupts=[],
          run_id=target_state.run_id,
          resume_inputs=target_state.resolved_responses,
      )

    if state:
      self._state.runs[node_path] = DynamicNodeRun(state=state, output=output)
    logger.info('node %s rehydrate end.', node_path)

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
    run = self._state.runs[node_path]
    if run.output is not None:
      child_ctx._output_value = run.output
      child_ctx._output_emitted = True
    if interrupt_ids:
      child_ctx._interrupt_ids = set(interrupt_ids)
    return child_ctx

  # --- Execution ---

  async def _run_node_internal(
      self,
      ctx: Context,
      node: BaseNode,
      name: str,
      node_path: str,
      run_id: str,
      node_input: Any,
      use_as_output: bool,
      is_fresh: bool,
      is_parallel: bool = False,
  ) -> Context:
    """Unified runner for both fresh and resume executions."""
    if is_fresh:
      state = NodeState(
          status=NodeStatus.RUNNING,
          input=node_input,
          run_id=run_id,
          parent_run_id=ctx.run_id,
      )
      run = DynamicNodeRun(state=state)
      self._state.runs[node_path] = run
      resume_inputs = None
    else:
      run = self._state.runs[node_path]
      run.state.status = NodeStatus.RUNNING
      resume_inputs = (
          dict(run.state.resume_inputs) if run.state.resume_inputs else None
      )

    from ._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node.model_copy(update={'name': name}),
        parent_ctx=ctx,
        run_id=run_id,
        additional_output_for_ancestor=(
            ctx.node_path if use_as_output else None
        ),
        is_parallel=is_parallel,
    )
    run.task = asyncio.create_task(
        runner.run(node_input=node_input, resume_inputs=resume_inputs)
    )
    try:
      child_ctx = await run.task
    except asyncio.CancelledError:
      if node_path in self._state.runs:
        del self._state.runs[node_path]
      raise
    self._record_result(run, child_ctx, node)
    return child_ctx

  def _record_result(
      self,
      run: DynamicNodeRun,
      child_ctx: Context,
      node: BaseNode,
  ) -> None:
    """Update dynamic node state after execution."""
    state = run.state
    if child_ctx.error:
      state.status = NodeStatus.FAILED
    elif child_ctx.interrupt_ids:
      state.status = NodeStatus.WAITING
      state.interrupts = list(child_ctx.interrupt_ids)
      self._state.interrupt_ids.update(child_ctx.interrupt_ids)
    elif node.wait_for_output and child_ctx.output is None:
      state.status = NodeStatus.WAITING
    else:
      state.status = NodeStatus.COMPLETED
      run.output = child_ctx.output
