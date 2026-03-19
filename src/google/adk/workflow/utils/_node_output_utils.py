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

"""Utility functions for fetching node output events."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from .._definitions import RouteValue

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...events.event import Event


def _get_node_output_events(
    *,
    ctx: InvocationContext,
    node_path: str,
    execution_id: str,
    local_events: list[Event] | None = None,
    terminal_paths: set[str] | None = None,
) -> list[Event]:
  """Fetches all Events for a node execution.

  history_events are outputs from previous runs, loaded when resuming.
  local_events are new outputs from the current run. We need both to collect
  all outputs from a node's execution, especially if it was interrupted and
  resumed, ensuring downstream nodes get the complete input.

  Args:
    terminal_paths: If the node is a Workflow, the recursively resolved
        leaf-level terminal paths. Events from these paths are included
        as output of this workflow node.
  """
  from ...events.event import Event

  _terminal_paths = terminal_paths or set()

  def _matches(e: Event) -> bool:
    if (
        e.node_info.path == node_path
        and e.node_info.execution_id == execution_id
    ):
      return True
    if _terminal_paths and e.node_info.path in _terminal_paths:
      return True
    return False

  # Fetch from history.
  history_events = [
      e for e in ctx.session.events
      if isinstance(e, Event) and _matches(e)
  ]

  # Fetch from local current run.
  local_node_events = [
      e for e in (local_events or [])
      if _matches(e)
  ]

  # Deduplicate based on event ID.
  all_events = list(history_events)
  seen_ids = {e.id for e in history_events}

  for e in local_node_events:
    if e.id not in seen_ids:
      all_events.append(e)
      seen_ids.add(e.id)

  return all_events


def _get_node_output_and_route(
    *,
    ctx: InvocationContext,
    node_path: str,
    execution_id: str,
    local_events: list[Event] | None = None,
    output_schema: Any | None = None,
    terminal_paths: set[str] | None = None,
) -> tuple[Any, RouteValue | list[RouteValue] | None]:
  """Fetches the Event outputs and route for a node execution.

  Args:
    output_schema: If set, validates and coerces each output value against
        this schema (e.g. filling Pydantic defaults, type coercion).
    terminal_paths: If the node is a Workflow, the recursively resolved
        leaf-level terminal paths.
  """
  from .._base_node import BaseNode

  events = _get_node_output_events(
      ctx=ctx,
      node_path=node_path,
      execution_id=execution_id,
      local_events=local_events,
      terminal_paths=terminal_paths,
  )

  routes_to_match: str | list[str] | None = None
  if events:
    events_with_route = [e for e in events if e.actions.route is not None]
    if len(events_with_route) > 1:
      raise ValueError(
          f'Node {node_path} produced multiple Events with '
          'route tags. Only one Event per execution '
          'can specify routes.'
      )
    if events_with_route:
      routes_to_match = events_with_route[0].actions.route

  events_with_data = [e for e in events if e.output is not None]

  if not events_with_data:
    output_data = None
  elif len(events_with_data) > 1 and not terminal_paths:
    raise ValueError(
        f'Node {node_path} produced multiple Events with output data.'
        ' A node execution should produce at most one output event.'
    )
  else:
    outputs = [e.output for e in events_with_data]
    if output_schema is not None:
      from pydantic import TypeAdapter

      adapter = TypeAdapter(output_schema)
      outputs = [
          BaseNode._to_serializable(adapter.validate_python(o))
          for o in outputs
      ]
    output_data = outputs[0] if len(outputs) == 1 else outputs

  return output_data, routes_to_match
