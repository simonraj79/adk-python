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

"""Utilities for reconstructing WorkflowAgentState from session events."""

from __future__ import annotations

from typing import Any
from typing import Optional

from ...events.event import Event
from .._node_state import NodeState
from .._node_status import NodeStatus
from .._workflow_graph import WorkflowGraph
from ._node_path_utils import is_direct_child


def reconstruct_state_from_events(
    session_events: list[Event],
    current_invocation_id: str,
    workflow_path: str,
    graph: WorkflowGraph,
) -> Optional[dict[str, NodeState]]:
  """Reconstructs workflow node states from session events.

  Scans session events in two passes to rebuild node states for
  non-resumable HITL resume:

  Pass 1: Collect all function response IDs from user events — these tell
  us which interrupts have been resolved.

  Pass 2: Scan non-user events for workflow node activity:
  - Data outputs (event.output) → node produced output
  - Interrupt IDs (event.long_running_tool_ids) → node was interrupted
  - Dynamic metadata (source_node_name, parent_run_id) → node was
    dynamically scheduled

  For each tracked node, cross-reference its interrupt IDs against the
  resolved set from pass 1:
  - Unresolved interrupts → WAITING
  - No unresolved interrupts + has data → COMPLETED

  Returns the reconstructed node states only if at least one node is
  WAITING — otherwise returns None so the workflow starts fresh.

  Args:
    session_events: The list of session events to scan.
    current_invocation_id: The current invocation ID to skip.
    workflow_path: The workflow agent path for filtering direct children.
    graph: The workflow graph containing nodes and edges.

  Returns:
    A dict of node_name → NodeState if interrupted nodes exist, else None.
  """
  # Find the most recent invocation that has workflow events with
  # unresolved interrupts.  We must scope to a single invocation to
  # avoid mixing state from a previous completed run with the
  # current interrupted run.
  target_invocation_id = _find_interrupted_invocation(
      session_events, current_invocation_id, workflow_path
  )
  if not target_invocation_id:
    return None

  # Collect resolved interrupt IDs and their response payloads from
  # user function responses that came AFTER the interrupted invocation.
  # Use counter-based matching: each FR resolves one FC with the same
  # ID. Only IDs where FR count >= FC count are fully resolved.
  resolved_interrupt_ids: set[str] = set()
  resolved_responses: dict[str, Any] = {}
  fc_counts: dict[str, int] = {}
  fr_counts: dict[str, int] = {}
  for event in session_events:
    if event.invocation_id == current_invocation_id:
      continue
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.function_call and part.function_call.id:
        fc_counts[part.function_call.id] = (
            fc_counts.get(part.function_call.id, 0) + 1
        )
      if part.function_response and part.function_response.id:
        fr_id = part.function_response.id
        fr_counts[fr_id] = fr_counts.get(fr_id, 0) + 1
        resolved_responses[fr_id] = part.function_response.response
  # An interrupt ID is resolved only if every FC has a matching FR.
  for iid, fr_count in fr_counts.items():
    if fr_count >= fc_counts.get(iid, 0):
      resolved_interrupt_ids.add(iid)

  # Collect ALL interrupt_ids ever emitted by each node across all
  # non-current invocations.  This is needed for rerun_on_resume nodes
  # that accumulate resume_inputs across multiple invocations.
  all_node_interrupts: dict[str, set[str]] = {}

  # Scan workflow node events from the target invocation only.
  node_data: dict[str, _NodeTracker] = {}

  for event in session_events:
    if event.invocation_id == current_invocation_id:
      continue
    if event.author == 'user':
      continue
    if not event.node_info.path:
      continue

    # Only process events from direct children of this workflow.
    if not is_direct_child(event.node_info.path, workflow_path):
      continue

    # Extract node name from node_path (last component).
    node_name = (
        event.node_info.path.split('/')[-1] if event.node_info.path else ''
    )
    if not node_name:
      continue

    # Track all historical interrupts for resume_inputs reconstruction.
    if event.long_running_tool_ids:
      all_node_interrupts.setdefault(node_name, set()).update(
          event.long_running_tool_ids
      )

    # Only build detailed tracker from the target invocation.
    if event.invocation_id != target_invocation_id:
      continue

    tracker = node_data.setdefault(node_name, _NodeTracker())

    if event.node_info.run_id:
      tracker.run_id = event.node_info.run_id

    if event.output is not None:
      tracker.has_output = True
      tracker.data_values.append(event.output)

    if event.long_running_tool_ids:
      tracker.interrupt_ids.update(event.long_running_tool_ids)

    if event.node_info.source_node_name:
      tracker.source_node_name = event.node_info.source_node_name
    if event.node_info.parent_run_id:
      tracker.parent_run_id = event.node_info.parent_run_id

  if not node_data:
    return None

  # Build node states from tracked data.
  nodes: dict[str, NodeState] = {}
  has_interrupted = False

  for node_name, tracker in node_data.items():
    unresolved = tracker.interrupt_ids - resolved_interrupt_ids

    if unresolved:
      # Reconstruct resume_inputs from previously resolved interrupts
      # that belong to this node (needed for rerun_on_resume nodes).
      # Use all_node_interrupts to capture interrupts from earlier
      # invocations that were already resolved.
      all_interrupts = all_node_interrupts.get(node_name, set())
      resolved_for_node = all_interrupts & resolved_interrupt_ids
      resume_inputs = {
          iid: resolved_responses[iid]
          for iid in resolved_for_node
          if iid in resolved_responses
      }
      nodes[node_name] = NodeState(
          status=NodeStatus.WAITING,
          run_id=tracker.run_id,
          interrupts=list(unresolved),
          resume_inputs=resume_inputs,
          source_node_name=tracker.source_node_name,
          parent_run_id=tracker.parent_run_id,
      )
      has_interrupted = True
    elif tracker.has_output:
      nodes[node_name] = NodeState(
          status=NodeStatus.COMPLETED,
          run_id=tracker.run_id,
          source_node_name=tracker.source_node_name,
          parent_run_id=tracker.parent_run_id,
      )

  if not has_interrupted:
    return None

  # Populate input for WAITING nodes from the data output of
  # their upstream completed node.  This restores the original
  # node_input (e.g. CallLlmResult for execute_tools) so the node
  # can be re-run correctly after all interrupts are resolved.
  for node_name, node_state in nodes.items():
    if node_state.status != NodeStatus.WAITING:
      continue
    for edge in graph.edges:
      if edge.to_node.name != node_name:
        continue
      upstream_name = edge.from_node.name
      upstream_tracker = node_data.get(upstream_name)
      if not upstream_tracker or not upstream_tracker.data_values:
        continue
      # Use the last output value from the upstream node. In a
      # reason-act loop (call_llm ↔ execute_tools), the upstream node
      # may produce multiple outputs across loop iterations. The
      # WAITING downstream node corresponds to the most recent one.
      node_state.input = upstream_tracker.data_values[-1]
      break

  return nodes


def _find_interrupted_invocation(
    session_events: list[Event],
    current_invocation_id: str,
    workflow_path: str,
) -> str | None:
  """Finds the most recent invocation with unresolved interrupts.

  Scans events in reverse to find the latest invocation that produced
  workflow events with long_running_tool_ids.  Then verifies those
  interrupts haven't all been resolved by subsequent user responses.

  Returns the invocation ID to scope reconstruction to, or None.
  """
  # Collect resolved interrupt IDs using counter-based matching.
  # Each FR resolves one FC with the same ID; only fully matched
  # IDs (FR count >= FC count) are considered resolved.
  fc_counts: dict[str, int] = {}
  fr_counts: dict[str, int] = {}
  for event in session_events:
    if event.invocation_id == current_invocation_id:
      continue
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.function_call and part.function_call.id:
        fc_counts[part.function_call.id] = (
            fc_counts.get(part.function_call.id, 0) + 1
        )
      if part.function_response and part.function_response.id:
        fr_counts[part.function_response.id] = (
            fr_counts.get(part.function_response.id, 0) + 1
        )
  resolved_ids: set[str] = {
      iid for iid, count in fr_counts.items() if count >= fc_counts.get(iid, 0)
  }

  # Scan in reverse to find the latest invocation with unresolved
  # interrupts for this workflow.
  for event in reversed(session_events):
    if event.invocation_id == current_invocation_id:
      continue
    if event.author == 'user' or not event.node_info.path:
      continue
    if not is_direct_child(event.node_info.path, workflow_path):
      continue
    if event.long_running_tool_ids:
      unresolved = event.long_running_tool_ids - resolved_ids
      if unresolved:
        return event.invocation_id

  return None


class _NodeTracker:
  """Tracks node activity during event scanning."""

  __slots__ = (
      'run_id',
      'has_output',
      'interrupt_ids',
      'source_node_name',
      'parent_run_id',
      'data_values',
  )

  def __init__(self):
    self.run_id: str | None = None
    self.has_output: bool = False
    self.interrupt_ids: set[str] = set()
    self.source_node_name: str | None = None
    self.parent_run_id: str | None = None
    self.data_values: list = []
