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

"""Testings for the Workflow routes."""

from typing import Any
from typing import Dict

from google.adk.agents.context import Context
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._workflow_graph import DEFAULT_ROUTE
from google.adk.workflow._workflow_graph import WorkflowGraph
import pytest

from ..workflow_testing_utils import create_parent_invocation_context
from ..workflow_testing_utils import simplify_events_with_node
from ..workflow_testing_utils import TestingNode


@pytest.mark.asyncio
async def test_run_async_with_edge_routes(request: pytest.FixtureRequest):
  route_holder = {'route': 'route_b'}

  def dynamic_router(ctx: Context, node_input: Any):
    return route_holder['route']

  node_a = TestingNode(name='NodeA', output='A', route=dynamic_router)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          Edge(
              node_a,
              node_c,
              route='route_c',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent',
      graph=graph,
  )

  # Test case for route_b
  route_holder['route'] = 'route_b'
  ctx_b = await create_parent_invocation_context(
      request.function.__name__ + '_b', agent
  )
  events_b = [e async for e in agent.run_async(ctx_b)]
  assert simplify_events_with_node(events_b) == [
      ('test_workflow_agent', {'node_name': 'NodeA', 'output': 'A'}),
      ('test_workflow_agent', {'node_name': 'NodeB', 'output': 'B'}),
  ]

  # Test case for route_c
  route_holder['route'] = 'route_c'
  ctx_c = await create_parent_invocation_context(
      request.function.__name__ + '_c', agent
  )
  events_c = [e async for e in agent.run_async(ctx_c)]
  assert simplify_events_with_node(events_c) == [
      ('test_workflow_agent', {'node_name': 'NodeA', 'output': 'A'}),
      ('test_workflow_agent', {'node_name': 'NodeC', 'output': 'C'}),
  ]


@pytest.mark.asyncio
async def test_output_route_int(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route=1)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_workflow_agent_route_int',
      edges=[
          (START, node_a),
          (node_a, {1: node_b, 2: node_c}),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      ('test_workflow_agent_route_int', {'node_name': 'NodeA', 'output': None}),
      ('test_workflow_agent_route_int', {'node_name': 'NodeB', 'output': 'B'}),
  ]


@pytest.mark.asyncio
async def test_output_route_bool(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route=True)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  agent = Workflow(
      name='test_workflow_agent_route_bool',
      edges=[
          (START, node_a),
          (node_a, {True: node_b, False: node_c}),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_route_bool',
          {'node_name': 'NodeA', 'output': None},
      ),
      ('test_workflow_agent_route_bool', {'node_name': 'NodeB', 'output': 'B'}),
  ]


@pytest.mark.asyncio
async def test_output_route_no_data(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          Edge(
              node_a,
              node_c,
              route='route_c',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_route_no_data',
      graph=graph,
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_route_no_data',
          {'node_name': 'NodeA', 'output': None},
      ),
      (
          'test_workflow_agent_route_no_data',
          {'node_name': 'NodeB', 'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_list_of_routes(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route=['route_b', 'route_c'])
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          Edge(
              node_a,
              node_c,
              route='route_c',
          ),
          Edge(
              node_a,
              node_d,
              route='route_d',
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_list_routes',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_list_routes',
      {'node_name': 'NodeA', 'output': 'A'},
  )

  # Check that the other two events are from NodeB and NodeC, in any order.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_list_routes',
          {'node_name': 'NodeB', 'output': 'B'},
      ),
      (
          'test_workflow_agent_list_routes',
          {'node_name': 'NodeC', 'output': 'C'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
async def test_run_async_with_default_route(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route='unmatched_route')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')

  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          # This edge has the DEFAULT_ROUTE tag.
          Edge(
              node_a,
              node_c,
              route=DEFAULT_ROUTE,
          ),
          # This edge has no route tag, so it should always be triggered.
          Edge(
              node_a,
              node_d,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_default_route',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_default_route',
      {'node_name': 'NodeA', 'output': 'A'},
  )

  # Check that NodeC (default route) and NodeD (untagged) are triggered.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_default_route',
          {'node_name': 'NodeC', 'output': 'C'},
      ),
      (
          'test_workflow_agent_default_route',
          {'node_name': 'NodeD', 'output': 'D'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
async def test_run_async_default_route_not_triggered_if_match(
    request: pytest.FixtureRequest,
):
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          # This edge has the DEFAULT_ROUTE tag.
          Edge(
              node_a,
              node_c,
              route=DEFAULT_ROUTE,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_default_route_not_triggered',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  simplified_events = simplify_events_with_node(events)

  assert simplified_events == [
      (
          'test_workflow_agent_default_route_not_triggered',
          {'node_name': 'NodeA', 'output': 'A'},
      ),
      (
          'test_workflow_agent_default_route_not_triggered',
          {'node_name': 'NodeB', 'output': 'B'},
      ),
  ]


@pytest.mark.asyncio
async def test_run_async_with_untagged_edges(request: pytest.FixtureRequest):
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')
  graph = WorkflowGraph(
      edges=[
          Edge(START, node_a),
          Edge(
              node_a,
              node_b,
              route='route_b',
          ),
          Edge(
              node_a,
              node_c,
              route='route_c',
          ),
          # This edge has no route tag, so it should always be triggered.
          Edge(
              node_a,
              node_d,
          ),
      ],
  )
  agent = Workflow(
      name='test_workflow_agent_untagged_edges',
      graph=graph,
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  simplified_events = simplify_events_with_node(events)

  assert len(simplified_events) == 3
  assert simplified_events[0] == (
      'test_workflow_agent_untagged_edges',
      {'node_name': 'NodeA', 'output': 'A'},
  )

  # Check that NodeB and NodeD are triggered.
  other_events = simplified_events[1:]
  expected_other_events = [
      (
          'test_workflow_agent_untagged_edges',
          {'node_name': 'NodeB', 'output': 'B'},
      ),
      (
          'test_workflow_agent_untagged_edges',
          {'node_name': 'NodeD', 'output': 'D'},
      ),
  ]
  assert len(other_events) == len(expected_other_events)
  assert all(item in other_events for item in expected_other_events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'emitted_route, expected_target',
    [
        ('route_a', 'Target'),
        ('route_b', 'Target'),
        ('route_c', 'Other'),
    ],
    ids=['route_a_matches_list', 'route_b_matches_list', 'route_c_single'],
)
async def test_edge_with_multiple_routes(
    request: pytest.FixtureRequest, emitted_route, expected_target
):
  """Tests that an edge with a list of routes matches any of them."""
  node_router = TestingNode(name='Router', output='R', route=emitted_route)
  node_target = TestingNode(name='Target', output='T')
  node_other = TestingNode(name='Other', output='O')

  agent = Workflow(
      name='test_multi_route',
      edges=[
          (START, node_router),
          Edge(node_router, node_target, route=['route_a', 'route_b']),
          (node_router, {'route_c': node_other}),
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      ('test_multi_route', {'node_name': 'Router', 'output': 'R'}),
      (
          'test_multi_route',
          {
              'node_name': expected_target,
              'output': 'T' if expected_target == 'Target' else 'O',
          },
      ),
  ]


# --- Routing map integration tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'emitted_route, expected_target',
    [
        ('route_b', 'NodeB'),
        ('route_c', 'NodeC'),
    ],
    ids=['route_b', 'route_c'],
)
async def test_routing_map_selects_correct_route(
    request: pytest.FixtureRequest, emitted_route: str, expected_target: str
):
  """Tests that a routing map routes to the correct node at runtime."""
  node_a = TestingNode(name='NodeA', output='A', route=emitted_route)
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  agent = Workflow(
      name='test_routing_map',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, 'route_c': node_c}),
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  expected_output = 'B' if expected_target == 'NodeB' else 'C'
  assert simplify_events_with_node(events) == [
      ('test_routing_map', {'node_name': 'NodeA', 'output': 'A'}),
      (
          'test_routing_map',
          {'node_name': expected_target, 'output': expected_output},
      ),
  ]


@pytest.mark.asyncio
async def test_routing_map_with_default_route(
    request: pytest.FixtureRequest,
):
  """Tests that DEFAULT_ROUTE works as a fallback in routing maps."""
  node_a = TestingNode(name='NodeA', output='A', route='unmatched_route')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  agent = Workflow(
      name='test_routing_map_default',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, DEFAULT_ROUTE: node_c}),
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      ('test_routing_map_default', {'node_name': 'NodeA', 'output': 'A'}),
      ('test_routing_map_default', {'node_name': 'NodeC', 'output': 'C'}),
  ]


@pytest.mark.asyncio
async def test_routing_map_mixed_with_other_formats(
    request: pytest.FixtureRequest,
):
  """Tests that routing maps coexist with tuples and Edge objects."""
  node_a = TestingNode(name='NodeA', output='A', route='route_b')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')
  node_d = TestingNode(name='NodeD', output='D')

  agent = Workflow(
      name='test_routing_map_mixed',
      edges=[
          (START, node_a),
          (node_a, {'route_b': node_b, 'route_c': node_c}),
          (node_b, node_d),  # unconditional 2-tuple
          Edge(node_c, node_d),  # Edge object
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  assert simplify_events_with_node(events) == [
      ('test_routing_map_mixed', {'node_name': 'NodeA', 'output': 'A'}),
      ('test_routing_map_mixed', {'node_name': 'NodeB', 'output': 'B'}),
      ('test_routing_map_mixed', {'node_name': 'NodeD', 'output': 'D'}),
  ]


@pytest.mark.asyncio
async def test_routing_map_fan_out_runs_both_targets(
    request: pytest.FixtureRequest,
):
  """Tests that fan-out in routing maps triggers both targets at runtime."""
  node_a = TestingNode(name='NodeA', output='A', route='route_x')
  node_b = TestingNode(name='NodeB', output='B')
  node_c = TestingNode(name='NodeC', output='C')

  agent = Workflow(
      name='test_routing_map_fan_out',
      edges=[
          (START, node_a),
          (node_a, {'route_x': (node_b, node_c)}),
      ],
  )

  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events)

  assert len(simplified) == 3
  assert simplified[0] == (
      'test_routing_map_fan_out',
      {'node_name': 'NodeA', 'output': 'A'},
  )

  # NodeB and NodeC should both be triggered, in any order.
  other = simplified[1:]
  expected = [
      ('test_routing_map_fan_out', {'node_name': 'NodeB', 'output': 'B'}),
      ('test_routing_map_fan_out', {'node_name': 'NodeC', 'output': 'C'}),
  ]
  assert len(other) == len(expected)
  assert all(item in other for item in expected)
