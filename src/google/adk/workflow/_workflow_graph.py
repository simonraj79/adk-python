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

"""Defines the graph and edges in the Workflow."""

from __future__ import annotations

from collections import Counter
from collections.abc import Set
from typing import Annotated
from typing import Any
from typing import get_args
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializeAsAny

from ._base_node import BaseNode
from ._base_node import START
from ._definitions import NodeLike
from ._definitions import RouteValue
from ._definitions import RoutingMap
from .utils._workflow_graph_utils import build_node
from .utils._workflow_graph_utils import is_node_like

DEFAULT_ROUTE = '__DEFAULT__'


class Edge(BaseModel):
  """An edge in the workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

  from_node: Annotated[BaseNode, SerializeAsAny()]
  """The from node."""

  to_node: Annotated[BaseNode, SerializeAsAny()]
  """The to node."""

  route: Optional[RouteValue | list[RouteValue]] = Field(
      description=(
          'The route(s) that this edge is associated with.'
          ' A single value or a list of values. The edge is followed when the'
          ' emitted route matches any value in the list.'
      ),
      default=None,
  )

  def __init__(
      self, from_node: BaseNode, to_node: BaseNode, **kwargs: Any
  ) -> None:
    super().__init__(from_node=from_node, to_node=to_node, **kwargs)


def _expand_routing_map(
    from_element: ChainElement,
    routing_map: RoutingMap,
) -> list[tuple[ChainElement, NodeLike | tuple[NodeLike, ...], RouteValue]]:
  """Expands a routing map into individual (from, to, route) triples.

  Args:
      from_element: The source node(s). Can be a single NodeLike or a
          tuple of NodeLike for fan-in.
      routing_map: A dict mapping route values to destination nodes.
          Values can be a single NodeLike or a tuple of NodeLike for
          fan-out.

  Returns:
      A list of (from_element, target, route) triples where target can
      be a single NodeLike or a tuple for fan-out.

  Raises:
      ValueError: If the routing map is empty, has non-RouteValue keys,
          or has non-NodeLike values.
  """
  if not routing_map:
    raise ValueError(
        'Routing map must not be empty. Provide at least one'
        ' route -> node mapping.'
    )

  route_value_types = get_args(RouteValue)
  expanded: list[
      tuple[ChainElement, NodeLike | tuple[NodeLike, ...], RouteValue]
  ] = []

  for route_key, target in routing_map.items():
    if not isinstance(route_key, route_value_types):
      raise ValueError(
          f'Invalid routing map key: {route_key!r} (type'
          f' {type(route_key).__name__}). Keys must be RouteValue'
          ' (str, int, or bool).'
      )
    if isinstance(target, tuple):
      for node in target:
        if not is_node_like(node):
          raise ValueError(
              f'Invalid node in fan-out tuple for route {route_key!r}:'
              f' {node!r} (type {type(node).__name__}).'
              ' Values must be NodeLike (BaseNode, BaseAgent, BaseTool,'
              " callable, or 'START')."
          )
    elif not is_node_like(target):
      raise ValueError(
          f'Invalid routing map value for route {route_key!r}:'
          f' {target!r} (type {type(target).__name__}).'
          ' Values must be NodeLike (BaseNode, BaseAgent, BaseTool,'
          " callable, or 'START')."
      )
    expanded.append((from_element, target, route_key))

  return expanded


def _nodes_from_routing_map(
    routing_map: RoutingMap,
) -> list[NodeLike]:
  """Extracts all target nodes from a routing map, flattening fan-out tuples.

  Args:
      routing_map: A dict mapping route values to destination nodes.

  Returns:
      A flat list of all NodeLike targets referenced in the map.
  """
  nodes: list[NodeLike] = []
  for target in routing_map.values():
    if isinstance(target, tuple):
      nodes.extend(target)
    else:
      nodes.append(target)
  return nodes


def _flatten_element(
    element: NodeLike | tuple[NodeLike, ...] | RoutingMap,
) -> list[NodeLike]:
  """Flattens a chain element into a list of individual nodes.

  - A single NodeLike is wrapped in a list.
  - A tuple of NodeLike is converted to a list.
  - A RoutingMap (dict) has its target nodes extracted and flattened.
  """
  if isinstance(element, dict):
    return _nodes_from_routing_map(element)
  if isinstance(element, tuple):
    return list(element)
  return [element]


class WorkflowGraph(BaseModel):
  """A workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

  nodes: list[Annotated[BaseNode, SerializeAsAny()]] = Field(
      default_factory=list
  )
  """The nodes in the workflow graph."""

  edges: list[Edge] = Field(default_factory=list)
  """The edges in the workflow graph."""

  @classmethod
  def from_edge_items(cls, edge_items: list[EdgeItem]) -> WorkflowGraph:
    """Creates a WorkflowGraph from a list of edge items."""
    node_map: dict[int, BaseNode] = {}

    def get_node(node_like: NodeLike) -> BaseNode:
      if node_like == 'START':
        return START
      elif isinstance(node_like, BaseNode):
        node_id = id(node_like)
        if node_id in node_map:
          return node_map[node_id]
        wrapped = build_node(node_like)
        if wrapped is not node_like:
          node_map[node_id] = wrapped
          return wrapped
        return node_like
      node_id = id(node_like)
      if node_id in node_map:
        return node_map[node_id]
      node = build_node(node_like)
      node_map[node_id] = node
      return node

    graph_edges: list[Edge] = []
    for item in edge_items:
      if isinstance(item, Edge):
        graph_edges.append(
            Edge(
                from_node=get_node(item.from_node),
                to_node=get_node(item.to_node),
                route=item.route,
            )
        )
        continue
      if not isinstance(item, tuple):
        raise ValueError(f'Invalid edge type: {type(item)}')

      # Chain with potential fan-in/fan-out and inline routing maps.
      # A routing map (dict) in a chain behaves like a fan-out tuple
      # but with conditioned incoming edges.
      for i in range(len(item) - 1):
        from_el = item[i]
        to_el = item[i + 1]

        if isinstance(to_el, dict):
          # to_el is a routing map: create conditioned edges.
          if isinstance(from_el, dict):
            raise ValueError(
                'Consecutive routing maps are not allowed in a chain.'
                ' Split them into separate edge items.'
            )
          from_chain_el = from_el
          for exp_from, exp_to, route in _expand_routing_map(
              from_chain_el, to_el
          ):
            for from_node in _flatten_element(exp_from):
              for to_node in _flatten_element(exp_to):
                graph_edges.append(
                    Edge(
                        from_node=get_node(from_node),
                        to_node=get_node(to_node),
                        route=route,
                    )
                )
        else:
          # Unconditional edges. _flatten_element handles dicts
          # (fan-in from routing map values) and tuples (fan-in/out).
          for from_node in _flatten_element(from_el):
            for to_node in _flatten_element(to_el):
              graph_edges.append(
                  Edge(
                      from_node=get_node(from_node),
                      to_node=get_node(to_node),
                      route=None,
                  )
              )
    return WorkflowGraph(edges=graph_edges)

  def model_post_init(self, context: Any) -> None:
    """Populates nodes from edges."""
    if 'nodes' in self.model_fields_set and self.nodes:
      raise ValueError(
          'Nodes are inferred from edges, do not set nodes explicitly.'
      )
    if self.edges:
      # Use a dictionary to preserve order and deduplicate nodes by object id.
      nodes = {
          id(node): node
          for edge in self.edges
          for node in [edge.from_node, edge.to_node]
      }
      self.nodes = list(nodes.values())

  def _detect_unconditional_cycles(self, node_names: Set[str]) -> None:
    """Detects unconditional cycles in the graph."""
    unconditional_adj: dict[str, list[str]] = {name: [] for name in node_names}
    for edge in self.edges:
      if edge.route is None:
        unconditional_adj[edge.from_node.name].append(edge.to_node.name)

    in_stack: set[str] = set()
    done: set[str] = set()

    def _dfs(node: str, path: list[str]) -> None:
      in_stack.add(node)
      path.append(node)
      for neighbor in unconditional_adj[node]:
        if neighbor in in_stack:
          cycle_start = path.index(neighbor)
          cycle = path[cycle_start:] + [neighbor]
          raise ValueError(
              'Graph validation failed. Unconditional cycle detected:'
              f' {" -> ".join(cycle)}. Cycles must include at'
              ' least one conditional (routed) edge to avoid'
              ' infinite loops.'
          )
        if neighbor not in done:
          _dfs(neighbor, path)
      path.pop()
      in_stack.remove(node)
      done.add(node)

    for name in node_names:
      if name not in done:
        _dfs(name, [])

  def validate_graph(self) -> None:
    """Validates the workflow graph."""
    names = [node.name for node in self.nodes]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )

    if duplicates:
      raise ValueError(
          'Graph validation failed. Duplicate node names found:'
          f' {duplicates}. This means multiple distinct node objects'
          ' have the same name. If you intended to reuse the same node, ensure'
          ' you pass the exact same object instance. If you intended to have'
          ' distinct nodes, ensure they have unique names.'
      )
    node_names = set(names)

    # 1. Existence of START node.
    if START.name not in node_names:
      raise ValueError(
          'Graph validation failed. START node (name: '
          f"'{START.name}') not found in graph nodes."
      )

    # 2. Connectivity check
    to_nodes = {edge.to_node.name for edge in self.edges}

    unreachable_nodes = node_names - to_nodes - {START.name}
    if unreachable_nodes:
      raise ValueError(
          'Graph validation failed. The following nodes are unreachable (not a'
          f' to_node in any edge): {sorted(list(unreachable_nodes))}'
      )

    if START.name in to_nodes:
      raise ValueError(
          'Graph validation failed. START node must not have incoming edges.'
      )

    # 3. No duplicate edges.
    seen_edges = set()
    for edge in self.edges:
      edge_tuple = (edge.from_node.name, edge.to_node.name)
      if edge_tuple in seen_edges:
        raise ValueError(
            'Graph validation failed. Duplicate edge found: from='
            f'{edge.from_node.name}, to={edge.to_node.name}'
        )
      seen_edges.add(edge_tuple)

    # 4. DEFAULT_ROUTE must not appear inside a list of routes.
    for edge in self.edges:
      if isinstance(edge.route, list) and DEFAULT_ROUTE in edge.route:
        raise ValueError(
            'Graph validation failed. DEFAULT_ROUTE cannot be combined'
            ' with other routes in a list (edge from='
            f'{edge.from_node.name}, to={edge.to_node.name}).'
            ' Use a separate edge for DEFAULT_ROUTE.'
        )

    # 5. At most one DEFAULT_ROUTE edge from any node.
    default_route_edges: dict[str, str] = {}
    for edge in self.edges:
      if edge.route == DEFAULT_ROUTE:
        from_node_name = edge.from_node.name
        if from_node_name in default_route_edges:
          raise ValueError(
              'Graph validation failed. Multiple DEFAULT_ROUTE edges found'
              f' from node {from_node_name} to'
              f' {default_route_edges[from_node_name]} and'
              f' {edge.to_node.name}'
          )
        default_route_edges[from_node_name] = edge.to_node.name

    # 6. Unconditional cycle detection (DFS).
    # Edges with route=None are always followed, so a cycle consisting
    # entirely of such edges would loop forever. Conditional edges
    # (with a route) are allowed to form cycles (loop patterns).
    self._detect_unconditional_cycles(node_names)


ChainElement = NodeLike | tuple[NodeLike, ...] | RoutingMap

EdgeItem = Edge | tuple[ChainElement, ...]
