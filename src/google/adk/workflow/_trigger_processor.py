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

from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from ._definitions import RouteValue
from ._node_status import NodeStatus
from ._trigger import Trigger
from ._workflow_graph import DEFAULT_ROUTE
from ._workflow_graph import WorkflowGraph
from .utils._node_output_utils import _get_node_output_and_route
from .utils._node_path_utils import join_paths

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event


_TERMINAL_STATUSES = frozenset({
    NodeStatus.COMPLETED,
    NodeStatus.FAILED,
    NodeStatus.CANCELLED,
})


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
