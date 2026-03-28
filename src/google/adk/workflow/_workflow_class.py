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
from .utils._node_path_utils import direct_child_name
from .utils._node_path_utils import is_descendant
from .utils._node_path_utils import is_direct_child
from .utils._workflow_hitl_utils import unwrap_response as _unwrap_fr_response

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..agents.context import ScheduleDynamicNode

logger = logging.getLogger('google_adk.' + __name__)

# ---------------------------------------------------------------------------
# Loop state (mutable, not persisted)
# ---------------------------------------------------------------------------


@dataclass
class _ChildScanState:
  """Per-child state accumulated during event scanning for resume."""

  execution_id: str = ''
  """Latest execution_id seen for this child."""

  output: Any = None
  """Output value from the latest execution, if any."""

  interrupt_ids: set[str] = field(default_factory=set)
  """Interrupt IDs emitted during the latest execution."""

  resolved_ids: set[str] = field(default_factory=set)
  """Interrupt IDs resolved by FR events in the session."""

  resolved_responses: dict[str, Any] = field(default_factory=dict)
  """FR response data keyed by interrupt ID."""


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

    # --- SETUP: resume from events or start fresh ---
    # TODO: resume from checkpoint event.
    if ctx.resume_inputs:
      loop_state = _LoopState()
      self._restore_static_nodes_from_events(loop_state, ctx)
      if loop_state.nodes:
        self._process_resume(loop_state, ctx)
      else:
        logger.warning(
            'Workflow %s: resume_inputs provided but no resumable state found.',
            self.name,
        )
    else:
      loop_state = _LoopState()
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
    # Terminal node output already has output_for including this
    # workflow's path. Mark output as delegated so the workflow's
    # NodeRunner skips creating a duplicate output event.
    if self._has_terminal_output(loop_state):
      ctx._output_delegated = True
    self._finalize(loop_state, ctx)
    return
    yield  # required to keep _run_impl as async generator

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
        child_ctx: Context = task.result()
        self._handle_completion(loop_state, name, node, child_ctx)

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
    is_terminal = node_name in self._graph._terminal_node_names

    node_state = loop_state.nodes[node_name]
    runner = NodeRunner(
        node=node,
        parent_ctx=ctx,
        triggered_by=trigger.triggered_by,
        in_nodes={  # TODO: move to WorkflowGraph and add tests.
            e.from_node.name
            for e in self._graph.edges
            if e.to_node.name == node_name
        },
        additional_output_for_ancestor=(ctx.node_path if is_terminal else None),
    )
    node_state.execution_id = runner.execution_id
    resume_inputs = (
        dict(node_state.resume_inputs) if node_state.resume_inputs else None
    )
    loop_state.pending_tasks[node_name] = asyncio.create_task(
        runner.run(node_input=trigger.input, resume_inputs=resume_inputs)
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
        use_as_output: bool = False,
    ) -> Context:
      if use_as_output:
        if ctx._output_delegated:
          raise ValueError(
              f'Node {ctx.node_path} already has a use_as_output'
              ' delegate. Only one use_as_output=True call is'
              ' allowed per node execution.'
          )
        ctx._output_delegated = True

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
          additional_output_for_ancestor=(
              ctx.node_path if use_as_output else None
          ),
      )
      task = asyncio.create_task(runner.run(node_input=node_input))

      loop_state.dynamic_pending_tasks[name] = task

      child_ctx = await task
      if child_ctx.interrupt_ids:
        node_state.status = NodeStatus.WAITING
        node_state.interrupts = list(child_ctx.interrupt_ids)
        loop_state.interrupt_ids.update(child_ctx.interrupt_ids)
      else:
        node_state.status = NodeStatus.COMPLETED

      return child_ctx

    return _schedule

  # --- Completion handling ---

  def _handle_completion(
      self,
      loop_state: _LoopState,
      node_name: str,
      node: BaseNode,
      child_ctx: Context,
  ) -> None:
    """Update state and trigger downstream after node completes."""
    node_state = loop_state.nodes[node_name]

    if child_ctx.interrupt_ids:
      node_state.status = NodeStatus.WAITING
      node_state.interrupts = list(child_ctx.interrupt_ids)
      loop_state.interrupt_ids.update(child_ctx.interrupt_ids)
      return

    if node.wait_for_output and child_ctx.output is None:
      node_state.status = NodeStatus.WAITING
      return

    node_state.status = NodeStatus.COMPLETED
    if child_ctx.output is not None:
      loop_state.node_outputs[node_name] = child_ctx.output

    # Buffer downstream triggers.
    self._buffer_downstream_triggers(
        loop_state, node_name, child_ctx.output, child_ctx.route
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

  # --- Resume ---

  def _restore_static_nodes_from_events(
      self, loop_state: _LoopState, ctx: Context
  ) -> None:
    """Reconstruct child node statuses and outputs from session events.

    Single forward pass through session events for this invocation.
    For each direct child, tracks the latest execution_id's state:
    output, interrupt IDs, and resolved IDs (from FR events). Then
    derives NodeState per child.

    Status priority:
      WAITING — has unresolved interrupt IDs
      COMPLETED — has output
      PENDING — all interrupts resolved, no output (re-run)
      WAITING — wait_for_output node triggered but no output yet

    TODO (next CL): Restore node_input via edge walking.
    """
    children = self._scan_child_events(ctx)
    if not children:
      return

    nodes: dict[str, NodeState] = {}
    node_outputs: dict[str, Any] = {}

    for child_name, child in children.items():
      unresolved = child.interrupt_ids - child.resolved_ids

      if unresolved:
        nodes[child_name] = NodeState(
            status=NodeStatus.WAITING,
            interrupts=list(unresolved),
        )
      elif child.output is not None:
        nodes[child_name] = NodeState(status=NodeStatus.COMPLETED)
        node_outputs[child_name] = child.output
      elif child.interrupt_ids:
        # All resolved → PENDING for re-run
        nodes[child_name] = NodeState(
            status=NodeStatus.PENDING,
            resume_inputs=child.resolved_responses,
        )

    # wait_for_output nodes that were triggered but produced no output
    self._add_wait_for_output_nodes(nodes, children)

    loop_state.nodes = nodes
    loop_state.node_outputs = node_outputs
    loop_state.interrupt_ids = {
        interrupt_id
        for state in nodes.values()
        if state.status == NodeStatus.WAITING
        for interrupt_id in state.interrupts
    }

  def _scan_child_events(self, ctx: Context) -> dict[str, _ChildScanState]:
    """Scan session events and collect per-child state.

    Forward pass through events for this invocation. For each direct
    child, tracks the latest execution_id and accumulates output,
    interrupt IDs, and resolved interrupt IDs.

    Returns:
      dict of child_name → _ChildScanState. Empty if no child had
      interrupts (nothing to resume).
    """
    ic = ctx._invocation_context
    workflow_path = ctx.node_path
    invocation_id = ic.invocation_id

    # Keyed by direct child name (first path component after workflow).
    # Events from nested descendants are attributed to their direct child.
    children: dict[str, _ChildScanState] = {}
    # interrupt_id → direct child name, for resolving FRs to children
    interrupt_owner: dict[str, str] = {}

    for event in ic.session.events:
      if event.invocation_id != invocation_id:
        continue

      # FR events resolve interrupts.
      if event.author == 'user' and event.content and event.content.parts:
        for part in event.content.parts:
          fr = part.function_response
          if fr and fr.id and fr.id in interrupt_owner:
            owner = interrupt_owner[fr.id]
            children[owner].resolved_ids.add(fr.id)
            children[owner].resolved_responses[fr.id] = _unwrap_fr_response(
                fr.response
            )
        continue

      if not is_descendant(workflow_path, event.node_info.path):
        continue

      child_name = direct_child_name(workflow_path, event.node_info.path)
      if not child_name:
        continue

      child = children.setdefault(child_name, _ChildScanState())

      # New execution_id → reset child state (previous execution stale).
      exec_id = event.node_info.execution_id
      if exec_id and child.execution_id != exec_id:
        child.execution_id = exec_id
        child.output = None
        child.interrupt_ids.clear()
        child.resolved_ids.clear()

      # Output only from direct children (not nested descendants)
      if (
          is_direct_child(event.node_info.path, workflow_path)
          and event.output is not None
      ):
        child.output = event.output

      # Interrupt from any descendant → attributed to direct child.
      if event.long_running_tool_ids:
        for interrupt_id in event.long_running_tool_ids:
          child.interrupt_ids.add(interrupt_id)
          interrupt_owner[interrupt_id] = child_name

    return children

  def _add_wait_for_output_nodes(
      self,
      nodes: dict[str, NodeState],
      children: dict[str, _ChildScanState],
  ) -> None:
    """Add WAITING for wait_for_output nodes triggered but without output.

    A wait_for_output node is not considered complete until it yields
    an output. We mark it WAITING so the orchestration loop doesn't
    treat it as a fresh node on resume.

    A wait_for_output node "ran" if any predecessor exists in the
    known children or nodes dict (it would have been triggered).
    """
    known_names = set(children) | set(nodes)
    for graph_node in self._graph.nodes:
      if (
          not graph_node.wait_for_output
          or graph_node.name in nodes
          or graph_node.name in children
      ):
        continue
      predecessors = {
          e.from_node.name
          for e in self._graph.edges
          if e.to_node.name == graph_node.name
      }
      if predecessors & known_names:
        nodes[graph_node.name] = NodeState(status=NodeStatus.WAITING)

  def _process_resume(self, loop_state: _LoopState, ctx: Context) -> None:
    """Seed triggers for PENDING nodes and collect interrupt IDs."""
    for node_name, node_state in loop_state.nodes.items():
      if node_state.status == NodeStatus.PENDING:
        loop_state.trigger_buffer.setdefault(node_name, []).append(
            Trigger(
                input=node_state.input,
                triggered_by=node_state.triggered_by or '',
            )
        )

  # --- FINALIZE ---

  def _finalize(self, loop_state: _LoopState, ctx: Context) -> None:
    """Set interrupt_ids or terminal output on ctx.

    If any child interrupted, propagate their interrupt IDs to ctx
    so the parent orchestrator sees them. Otherwise, set the terminal
    node's output on ctx so the parent can read it.
    """
    if loop_state.interrupt_ids:
      ctx._interrupt_ids = set(loop_state.interrupt_ids)
      return

    # Set terminal output on ctx so parent reads ctx.output.
    # Terminal nodes = no outgoing edges.
    # TODO: Replace structural terminal detection with Event.output_for.
    terminal_outputs = [
        loop_state.node_outputs[name]
        for name in self._graph._terminal_node_names
        if name in loop_state.node_outputs
    ]
    if len(terminal_outputs) == 1:
      ctx.output = terminal_outputs[0]
    elif terminal_outputs:
      raise ValueError(
          f'Workflow {self.name}: multiple terminal nodes produced'
          f' output ({len(terminal_outputs)}). A workflow must have'
          ' at most one terminal output.'
      )

  # --- Utilities ---

  def _has_terminal_output(self, loop_state: _LoopState) -> bool:
    """Check if any terminal node produced output."""
    return any(
        name in loop_state.node_outputs
        for name in self._graph._terminal_node_names
    )

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
