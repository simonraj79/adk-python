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

"""New Workflow implementation — BaseNode with graph orchestration.

Combines user-facing graph definition with the execution engine.
Workflow(BaseNode) with _run_impl() as the orchestration loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PrivateAttr

from ._base_node import BaseNode
from ._base_node import START
from ._node_state import NodeState
from ._node_status import NodeStatus
from ._trigger import Trigger
from ._workflow_graph import Edge
from ._workflow_graph import EdgeItem
from ._workflow_graph import WorkflowGraph

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..agents.context import ScheduleDynamicNode
  from ._node_run_result import NodeRunResult

logger = logging.getLogger('google_adk.' + __name__)

# ---------------------------------------------------------------------------
# Loop state (mutable, not persisted)
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
  """Mutable state for the orchestration loop."""

  node_outputs: dict[str, Any] = field(default_factory=dict)
  """Collected outputs keyed by node name."""

  pending_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
  """Running asyncio tasks keyed by node name."""

  dynamic_pending_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
  """Tasks from ctx.run_node(), keyed by node name."""

  interrupt_ids: set[str] = field(default_factory=set)
  """Aggregated interrupt IDs from all WAITING nodes managed by this workflow and nested workflows."""

  trigger_buffer: dict[str, list[Trigger]] = field(default_factory=dict)
  """Pending triggers keyed by target node name."""

  nodes: dict[str, NodeState] = field(default_factory=dict)
  """Per-node state (status, input, interrupts, etc.) keyed by node name."""

  schedule_dynamic_node: ScheduleDynamicNode | None = None
  """schedule_dynamic_node closure for child Contexts."""


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow(BaseNode):
  """A graph-based workflow node.

  _run_impl() IS the graph orchestration loop:
  - SETUP: build graph, seed triggers
  - LOOP: schedule ready nodes via NodeRunner, handle completions
  - FINALIZE: collect terminal outputs
  """

  rerun_on_resume: bool = Field(default=True)

  edges: list[EdgeItem] = Field(
      description='Edges to build the workflow graph.',
      default_factory=list,
  )

  max_concurrency: int | None = None
  """Maximum parallel graph-scheduled nodes. None means unlimited.

  Only applies to nodes triggered by graph edges. Dynamic nodes
  (via ctx.run_node()) are excluded — they are awaited inline by
  their parent and throttling them would cause deadlock.
  """

  _graph: WorkflowGraph | None = PrivateAttr(default=None)

  # --- Construction ---

  def model_post_init(self, context: Any) -> None:
    super().model_post_init(context)
    if self.edges:
      self._graph = self._build_graph()

  def _build_graph(self) -> WorkflowGraph:
    """Convert edge definitions to a validated WorkflowGraph."""
    graph = WorkflowGraph.from_edge_items(self.edges)
    graph.validate_graph()
    return graph

  # --- _run_impl: the orchestration loop ---

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Orchestration loop: SETUP -> LOOP -> FINALIZE."""
    if self._graph is None:
      return

    loop_state = _LoopState()
    # TODO: Resume support:
    # - If checkpoint exists in session, restore _LoopState from it.
    # - If no checkpoint (checkpointing not enabled), reconstruct
    #   state from session event list (scan for completed/waiting nodes).
    # - Match ctx.resume_inputs against WAITING nodes.
    # - Set matched nodes to PENDING instead of seeding start triggers.
    self._seed_start_triggers(loop_state, node_input)

    # Create closure for dynamic node scheduling
    loop_state.schedule_dynamic_node = self._make_schedule_dynamic_node(
        loop_state
    )
    ctx._schedule_dynamic_node_internal = loop_state.schedule_dynamic_node

    # --- LOOP ---
    try:
      await self._run_loop(loop_state, ctx)
    finally:
      await self._cleanup_all_tasks(loop_state)

    # Collect remaining interrupts from WAITING nodes
    self._collect_remaining_interrupts(loop_state)

    # --- FINALIZE ---
    async for item in self._finalize(loop_state):
      yield item

  # --- LOOP ---

  async def _run_loop(self, loop_state: _LoopState, ctx: Context) -> None:
    """Schedule and execute nodes until no more work."""
    while True:
      self._schedule_ready_nodes(loop_state, ctx)

      if not loop_state.pending_tasks:
        break

      done, _ = await asyncio.wait(
          loop_state.pending_tasks.values(),
          return_when=asyncio.FIRST_COMPLETED,
      )

      for task in done:
        name = self._pop_completed_task(loop_state, task)
        node = self._get_static_node_by_name(name)
        result: NodeRunResult = task.result()
        self._handle_completion(loop_state, name, node, result)

    # Await fire-and-forget dynamic tasks.
    # TODO: Handle dynamic task failures and interrupts here.
    # Currently, dynamic node completion is handled inline in the
    # _schedule closure. But failures are not caught.
    if loop_state.dynamic_pending_tasks:
      await asyncio.wait(loop_state.dynamic_pending_tasks.values())

  # --- Scheduling ---

  def _seed_start_triggers(
      self, loop_state: _LoopState, node_input: Any
  ) -> None:
    """Seed triggers for START's direct successors."""
    for edge in self._graph.edges:
      if edge.from_node.name == START.name:
        loop_state.trigger_buffer.setdefault(edge.to_node.name, []).append(
            Trigger(input=node_input, triggered_by=START.name)
        )

  def _schedule_ready_nodes(self, loop_state: _LoopState, ctx: Context) -> None:
    """Pop triggers from buffer and schedule ready nodes."""
    for node_name in list(loop_state.trigger_buffer.keys()):
      if node_name in loop_state.pending_tasks:
        continue
      if self._at_concurrency_limit(loop_state):
        break

      trigger = self._pop_trigger(loop_state, node_name)
      if trigger is None:
        continue

      self._prepare_node_state(loop_state, node_name, trigger)
      self._start_node_task(loop_state, ctx, node_name, trigger)

  def _at_concurrency_limit(self, loop_state: _LoopState) -> bool:
    """Check if max_concurrency has been reached."""
    return (
        bool(self.max_concurrency)
        and len(loop_state.pending_tasks) >= self.max_concurrency
    )

  def _pop_trigger(
      self, loop_state: _LoopState, node_name: str
  ) -> Trigger | None:
    """Pop the next trigger for a node, or None if empty."""
    buffer = loop_state.trigger_buffer.get(node_name, [])
    if not buffer:
      return None
    trigger = buffer.pop(0)
    if not buffer:
      del loop_state.trigger_buffer[node_name]
    return trigger

  def _prepare_node_state(
      self, loop_state: _LoopState, node_name: str, trigger: Trigger
  ) -> None:
    """Create or reset NodeState for a node about to be scheduled."""
    node_state = loop_state.nodes.setdefault(node_name, NodeState())
    node_state.input = trigger.input
    node_state.triggered_by = trigger.triggered_by
    node_state.status = NodeStatus.RUNNING

  def _start_node_task(
      self,
      loop_state: _LoopState,
      ctx: Context,
      node_name: str,
      trigger: Trigger,
  ) -> None:
    """Create NodeRunner and start asyncio task for a node."""
    from ._node_runner_class import NodeRunner

    node = self._get_static_node_by_name(node_name)

    runner = NodeRunner(
        node=node,
        parent_ctx=ctx,
        triggered_by=trigger.triggered_by,
        in_nodes={  # TODO: move to WorkflowGraph and add tests.
            e.from_node.name
            for e in self._graph.edges
            if e.to_node.name == node_name
        },
    )
    loop_state.nodes[node_name].execution_id = runner.execution_id
    loop_state.pending_tasks[node_name] = asyncio.create_task(
        runner.run(node_input=trigger.input)
    )

  def _make_schedule_dynamic_node(
      self, loop_state: _LoopState
  ) -> ScheduleDynamicNode:
    """Create schedule_dynamic_node closure for child Contexts."""
    from ._node_runner_class import NodeRunner

    async def _schedule(
        ctx: Context,
        node: BaseNode,
        execution_id: str,
        node_input: Any,
        *,
        node_name: str | None = None,
    ) -> NodeRunResult:
      # TODO: consider unify this across all orchestration nodes - LlmAgent, Workflow, etc.
      name = node_name or f'{node.name}_{execution_id[:8]}'
      if name in loop_state.dynamic_pending_tasks:
        raise ValueError(f'Dynamic node {name} already exists.')

      node_state = NodeState(
          status=NodeStatus.RUNNING,
          input=node_input,
          execution_id=execution_id,
          source_node_name=node.name,
          parent_execution_id=ctx.execution_id,
      )
      loop_state.nodes[name] = node_state
      runner = NodeRunner(
          node=node,
          parent_ctx=ctx,
          execution_id=execution_id,
      )
      task = asyncio.create_task(runner.run(node_input=node_input))

      loop_state.dynamic_pending_tasks[name] = task

      result = await task
      if result.interrupt_ids:
        node_state.status = NodeStatus.WAITING
        node_state.interrupts = list(result.interrupt_ids)
        loop_state.interrupt_ids.update(result.interrupt_ids)
      else:
        node_state.status = NodeStatus.COMPLETED
      return result

    return _schedule

  # --- Completion handling ---

  def _handle_completion(
      self,
      loop_state: _LoopState,
      node_name: str,
      node: BaseNode,
      result: NodeRunResult,
  ) -> None:
    """Update state and trigger downstream after node completes."""
    node_state = loop_state.nodes[node_name]

    if result.interrupt_ids:
      node_state.status = NodeStatus.WAITING
      node_state.interrupts = list(result.interrupt_ids)
      loop_state.interrupt_ids.update(result.interrupt_ids)
      return

    if node.wait_for_output and result.output is None:
      node_state.status = NodeStatus.WAITING
      return

    node_state.status = NodeStatus.COMPLETED
    if result.output is not None:
      loop_state.node_outputs[node_name] = result.output

    # Buffer downstream triggers.
    self._buffer_downstream_triggers(
        loop_state, node_name, result.output, result.route
    )

  def _buffer_downstream_triggers(
      self,
      loop_state: _LoopState,
      node_name: str,
      output: Any,
      route: Any,
  ) -> None:
    """Find downstream edges and add triggers to the buffer."""
    from ._trigger_processor import _get_next_pending_nodes

    next_nodes = _get_next_pending_nodes(
        node_name=node_name,
        routes_to_match=route,
        graph=self._graph,
    )
    for target_name in next_nodes:
      loop_state.trigger_buffer.setdefault(target_name, []).append(
          Trigger(input=output, triggered_by=node_name)
      )
      # Re-trigger COMPLETED nodes (loop back-edges)
      node_state = loop_state.nodes.get(target_name)
      if node_state and node_state.status == NodeStatus.COMPLETED:
        node_state.status = NodeStatus.PENDING

  def _collect_remaining_interrupts(self, loop_state: _LoopState) -> None:
    """Gather interrupt_ids from nodes still WAITING after the loop."""
    for node_state in loop_state.nodes.values():
      if node_state.status == NodeStatus.WAITING and node_state.interrupts:
        loop_state.interrupt_ids.update(node_state.interrupts)

  # --- FINALIZE ---

  async def _finalize(
      self, loop_state: _LoopState
  ) -> AsyncGenerator[Any, None]:
    """Yield interrupt signal or terminal output for propagation."""
    from ..events.event import Event

    if loop_state.interrupt_ids:
      yield Event(long_running_tool_ids=loop_state.interrupt_ids)
      return

    # Yield terminal output so NodeRunResult.output is set.
    # Needed for nested workflows — parent sees child's output
    # via NodeRunResult. Terminal nodes = no outgoing edges.
    terminal_outputs = [
        loop_state.node_outputs[name]
        for name in self._graph._terminal_node_names
        if name in loop_state.node_outputs
    ]
    if len(terminal_outputs) == 1:
      yield terminal_outputs[0]
    elif terminal_outputs:
      raise ValueError(
          f'Workflow {self.name}: multiple terminal nodes produced'
          f' output ({len(terminal_outputs)}). A workflow must have'
          ' at most one terminal output.'
      )

  # --- Utilities ---

  def _get_static_node_by_name(self, name: str) -> BaseNode:
    """Find a node in the graph by name."""
    for node in self._graph.nodes:
      if node.name == name:
        return node
    raise ValueError(f'Node {name} not found in graph.')

  def _pop_completed_task(
      self, loop_state: _LoopState, task: asyncio.Task
  ) -> str:
    """Remove a completed task and return its node name."""
    for name, t in loop_state.pending_tasks.items():
      if t is task:
        del loop_state.pending_tasks[name]
        return name
    raise ValueError('Task not found in pending_tasks.')

  async def _cleanup_all_tasks(self, loop_state: _LoopState) -> None:
    """Cancel remaining tasks to prevent leaks."""
    all_tasks = list(loop_state.pending_tasks.values()) + list(
        loop_state.dynamic_pending_tasks.values()
    )
    if all_tasks:
      logger.warning(
          'Workflow %s: cancelling %d leftover tasks.',
          self.name,
          len(all_tasks),
      )
    for task in all_tasks:
      if not task.done():
        task.cancel()
    if all_tasks:
      await asyncio.gather(*all_tasks, return_exceptions=True)
