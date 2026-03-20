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

from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict

from ._definitions import RouteValue
from ._execution_state import NodeStatus
from ._workflow_graph import DEFAULT_ROUTE
from ._workflow_graph import WorkflowGraph
from .utils._node_output_utils import _get_node_output_and_route
from .utils._node_path_utils import join_paths

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ._run_state import _WorkflowRunState
  from ._workflow import WorkflowAgentState


class Trigger(BaseModel):
  """Represents a trigger for a node."""

  model_config = ConfigDict(ser_json_bytes='base64')

  input: Any = None
  triggered_by: str = ''


_TERMINAL_STATUSES = frozenset({
    NodeStatus.COMPLETED,
    NodeStatus.FAILED,
    NodeStatus.CANCELLED,
})


def _cleanup_child_executions(
    parent_execution_id: str,
    agent_state: WorkflowAgentState,
) -> None:
  """Removes terminal dynamic child nodes spawned by a parent execution.

  When a parent node completes, any dynamic child nodes it spawned
  (identified by parent_execution_id) that have reached a terminal
  state are removed from agent_state to avoid stale state on
  re-trigger.

  Children that are still running or pending are left in place so they
  can finish execution; they will self-clean via the self-cleanup
  block at the end of ``_process_triggers``.

  The terminal-status check is sufficient because node status is only
  set to a terminal value by ``_handle_node_completion``, which
  processes completions sequentially.  A child that appears terminal
  here has already been fully processed.

  Args:
    parent_execution_id: The execution ID of the parent node.
    agent_state: The workflow agent state containing node states.
  """
  nodes_to_remove = [
      name
      for name, state in agent_state.nodes.items()
      if state.parent_execution_id == parent_execution_id
      and state.status in _TERMINAL_STATUSES
  ]
  for name in nodes_to_remove:
    del agent_state.nodes[name]


def _get_next_pending_nodes(
    node_name: str,
    routes_to_match: RouteValue | list[RouteValue] | None,
    graph: WorkflowGraph,
) -> list[str]:
  """Determines the next nodes to transition to PENDING state based on routes."""
  next_pending_nodes: list[str] = []
  matched_specific_route = False
  default_route_node: Optional[str] = None

  for edge in graph.edges:
    if edge.from_node.name == node_name:
      if edge.route is None:
        # Edges with no route tag are always triggered.
        next_pending_nodes.append(edge.to_node.name)
        continue

      if edge.route == DEFAULT_ROUTE:
        default_route_node = edge.to_node.name
        continue

      # Normalize edge routes to a set for matching.
      edge_routes = (
          set(edge.route) if isinstance(edge.route, list) else {edge.route}
      )

      edge_matched = False
      if isinstance(routes_to_match, list):
        if edge_routes & set(routes_to_match):
          edge_matched = True
      elif routes_to_match in edge_routes:
        edge_matched = True

      if edge_matched:
        next_pending_nodes.append(edge.to_node.name)
        matched_specific_route = True

  if not matched_specific_route and default_route_node:
    next_pending_nodes.append(default_route_node)

  return next_pending_nodes


def _process_triggers(
    *,
    run_state: _WorkflowRunState,
    schedule_node: Callable[[str], None],
    node_name: str,
    execution_id: str,
) -> None:
  """Process triggers for a completed node and schedule downstream nodes.

  This function handles the completion of a node by:
  1. Resolving any dynamic futures waiting on the node's output
  2. Cleaning up the node's state (resume_inputs, input, triggered_by, etc.)
  3. Determining downstream nodes based on the node's output routes
  4. Buffering triggers for those downstream nodes

  Nodes in WAITING state (e.g. ``wait_for_output`` nodes that did not
  produce output) are cleaned up but downstream triggering is skipped.

  Args:
    run_state: The workflow runtime state.
    schedule_node: Callback to schedule a node for execution.
    node_name: Name of the node that just completed.
    execution_id: The execution ID of the completed node.
  """
  ctx = run_state.ctx
  graph = run_state.graph
  agent_state = run_state.agent_state
  dynamic_futures = run_state.dynamic_futures
  local_output_events = run_state.local_output_events
  node_state = agent_state.nodes.get(node_name)
  full_node_path = join_paths(run_state.node_path, node_name)

  # Fetch node output once — reused for dynamic future resolution and
  # downstream triggering. If the node has output_schema, validate and
  # coerce the output here at read time.
  node = run_state.nodes_map.get(node_name)

  # If the node is a Workflow, resolve its leaf-level terminal paths
  # so _get_node_output_and_route can find output events from nested
  # terminal nodes without requiring event mutation.
  terminal_paths: set[str] | None = None
  if node is not None:
    from ._workflow import Workflow

    if isinstance(node, Workflow):
      terminal_paths = node._resolve_terminal_paths(full_node_path)

  output_data, routes_to_match = _get_node_output_and_route(
      ctx=ctx,
      node_path=full_node_path,
      execution_id=execution_id,
      local_events=local_output_events,
      output_schema=node.output_schema if node else None,
      terminal_paths=terminal_paths,
  )

  if node_name in dynamic_futures:
    if not dynamic_futures[node_name].done():
      dynamic_futures[node_name].set_result(output_data)
    del dynamic_futures[node_name]
  elif node_state and node_state.parent_execution_id:
    # If the node is dynamic and was resumed (so it's not in
    # dynamic_futures), we need to wake up the parent node if it is
    # currently interrupted.
    for p_name, p_state in agent_state.nodes.items():
      if p_state.execution_id == node_state.parent_execution_id:
        if p_state.status == NodeStatus.WAITING:
          schedule_node(p_name)
        break

  # Clean up node state.
  if node_state:
    node_state.resume_inputs.clear()
    node_state.input = None
    node_state.triggered_by = None
    node_state.execution_id = None

  # Clean up terminal dynamic child nodes spawned by this execution.
  if execution_id:
    _cleanup_child_executions(execution_id, agent_state)

  # Nodes in WAITING state skip downstream triggering.
  if node_state and node_state.status == NodeStatus.WAITING:
    return

  next_nodes_set = set(
      _get_next_pending_nodes(node_name, routes_to_match, graph)
  )

  for next_node in next_nodes_set:
    agent_state.trigger_buffer.setdefault(next_node, []).append(
        Trigger(input=output_data, triggered_by=node_name)
    )

  # Self-cleanup: if this node is a dynamic child and its parent's
  # execution has reached a terminal state (or the execution_id was
  # already cleared), this node's state is no longer needed.  This
  # handles the fire-and-forget case where a child finishes after its
  # parent.
  if node_state and node_state.parent_execution_id:
    parent_execution_active = any(
        s.execution_id == node_state.parent_execution_id
        and s.status not in _TERMINAL_STATUSES
        for s in agent_state.nodes.values()
    )
    if not parent_execution_active:
      del agent_state.nodes[node_name]
