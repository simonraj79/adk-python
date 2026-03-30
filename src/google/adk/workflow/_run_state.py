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

"""Runtime state definitions for workflow execution."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from typing import TYPE_CHECKING
import uuid

from google.genai import types

from ..agents.invocation_context import InvocationContext
from ..agents.llm._transfer_target_info import _TransferTargetInfo
from ..events.event import Event
from ._base_node import BaseNode
from ._workflow_graph import WorkflowGraph

if TYPE_CHECKING:
  from ._workflow import WorkflowAgentState


@dataclasses.dataclass
class _NodeCompletion:
  """Marker to indicate node completion."""

  node_name: str
  """The name of the node that completed."""

  run_id: str | None = None
  """Unique identifier for this specific run of the node."""

  node_interrupted: bool = False
  """Whether the node was interrupted (e.g., by RequestInput or a
  long-running tool)."""

  interrupt_ids: list[str] = dataclasses.field(default_factory=list)
  """List of interrupt IDs associated with the interruption."""

  has_output: bool = False
  """Whether the node produced at least one Event with data set."""

  exception: Exception | None = None
  """Exception that occurred during node run, if any."""

  is_cancelled: bool = False
  """Whether the node was cancelled."""


@dataclasses.dataclass
class _NodeResumption:
  """Represents a node that can be resumed with a function response."""

  node_name: str
  """The name of the node to resume."""

  interrupt_id: str
  """The ID of the interrupt that triggered the pause."""

  response_part: types.Part
  """The function response Part to provide to the node on resume."""

  rerun_on_resume: bool
  """If True, the node should be re-executed from the beginning; otherwise,
  it continues from where it was interrupted."""


@dataclasses.dataclass
class _WorkflowRunState:
  """Encapsulates runtime state for a workflow execution.

  This dataclass holds all the mutable state needed during workflow execution,
  allowing stateless functions to operate on this shared state.
  """

  ctx: InvocationContext
  """The invocation context for the current agent execution."""

  event_queue: asyncio.Queue[Event | _NodeCompletion]
  """Queue for streaming events and completion markers."""

  graph: WorkflowGraph
  """The workflow graph defining node connections and edges."""

  node_path: str
  """The path of the workflow agent authoring events."""

  agent_state: WorkflowAgentState
  """The persisted state of the workflow agent."""

  nodes_map: dict[str, BaseNode]
  """Mapping from node names to node instances."""

  running_tasks: dict[str, asyncio.Task]
  """Dictionary tracking currently running asyncio tasks."""

  dynamic_futures: dict[str, asyncio.Future[Any]]
  """Dictionary tracking futures for dynamic node results."""

  local_output_events: list[Event]
  """List of output events from the current invocation."""

  static_node_names: set[str]
  """Set of node names defined in the static graph."""

  transfer_targets: list[_TransferTargetInfo]
  """Transfer targets to propagate to inner WorkflowContexts."""

  dynamic_output_node: dict[str, str] = dataclasses.field(default_factory=dict)
  """Maps a node's full path to the full path of the dynamic child
  whose output should be used as the node's output (set by
  ctx.run_node(use_as_output=True)). If a node is not in this dict,
  its own output is used."""

  max_concurrency: int | None = None
  """Maximum number of parallel nodes to run."""

  running_node_count: int = 0
  """Current number of nodes executing."""
