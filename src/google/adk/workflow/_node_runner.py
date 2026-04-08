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

"""Node runner utilities for workflow agents.

This module provides stateless functions for scheduling and executing nodes
in a workflow. The _WorkflowRunState dataclass encapsulates all runtime state
needed during workflow execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Callable
import uuid

from ..agents.context import Context
from ..agents.context import ScheduleDynamicNode
from ..agents.invocation_context import InvocationContext
from ..agents.llm._transfer_target_info import _TransferTargetInfo
from ..events.event import Event
from ..events.request_input import RequestInput
from ._base_node import BaseNode
from ._dynamic_node_registry import dynamic_node_registry
from ._errors import NodeInterruptedError
from ._errors import NodeTimeoutError
from ._node_state import NodeState
from ._node_status import NodeStatus
from ._run_state import _NodeCompletion
from ._run_state import _NodeResumption
from ._run_state import _WorkflowRunState
from .utils._event_utils import enrich_event
from .utils._node_output_utils import _get_node_output_and_route
from .utils._node_path_utils import get_node_name_from_path
from .utils._node_path_utils import is_descendant
from .utils._node_path_utils import is_direct_child
from .utils._node_path_utils import join_paths
from .utils._workflow_hitl_utils import create_request_input_event

logger = logging.getLogger('google_adk.' + __name__)


def _schedule_node(
    run_state: _WorkflowRunState,
    node_name: str,
) -> None:
  """Schedules a node to run.

  Sets the node status to RUNNING, assigns a run ID if not present,
  and creates an asyncio task to execute the node.

  Args:
    run_state: The workflow runtime state.
    node_name: The name of the node to schedule.
  """
  if (
      run_state.max_concurrency is not None
      and run_state.running_node_count >= run_state.max_concurrency
  ):
    return

  node_state = run_state.agent_state.nodes[node_name]
  node_state.status = NodeStatus.RUNNING
  if not node_state.run_id:
    node_state.run_id = str(uuid.uuid4())
  node = run_state.nodes_map[node_name]

  resume_inputs = node_state.resume_inputs
  node_input = node_state.input
  triggered_by = node_state.triggered_by or ''

  # Calculate predecessors for Context
  in_nodes = {
      edge.from_node.name
      for edge in run_state.graph.edges
      if edge.to_node.name == node_name
  }

  task = asyncio.create_task(
      _node_runner(
          run_state=run_state,
          node=node,
          node_input=node_input,
          triggered_by=triggered_by,
          in_nodes=in_nodes,
          resume_inputs=resume_inputs,
          run_id=node_state.run_id,
          attempt_count=node_state.attempt_count,
      ),
      name=node_name,
  )
  run_state.running_tasks[node_name] = task
  run_state.running_node_count += 1


def _schedule_retry_task(
    run_state: _WorkflowRunState,
    node_name: str,
    delay: float,
    checkpoint_func: Callable[..., AsyncGenerator[Event, None]],
) -> None:
  """Schedules a task to retry a node after a delay."""

  async def _retry_task() -> None:
    try:
      await asyncio.sleep(delay)
      _schedule_node(run_state, node_name)
      async for event in checkpoint_func(run_state.ctx, run_state.agent_state):
        await run_state.event_queue.put(event)
    finally:
      run_state.running_tasks.pop(f'{node_name}_retry', None)

  task = asyncio.create_task(_retry_task(), name=f'{node_name}_retry')

  # Add task to running tasks, so that workflow waits for it to complete.
  run_state.running_tasks[task.get_name()] = task


def _schedule_dynamic_node(
    run_state: _WorkflowRunState,
    current_ctx: Context,
    node: BaseNode,
    run_id: str,
    node_input: Any,
    *,
    node_name: str | None = None,
    use_as_output: bool = False,
) -> asyncio.Future[Any]:
  """Schedules a dynamic node to run.

  Dynamic nodes are created at runtime (not defined in the static graph).
  This function handles registration, state initialization, and scheduling.
  If the node was previously executed (e.g., during resumption), returns
  the cached result.

  Args:
    run_state: The workflow runtime state.
    current_ctx: The workflow context of the parent node scheduling this.
    node: The node instance to execute.
    run_id: Unique identifier for this run.
    node_input: Input data to pass to the node.
    node_name: Optional name for the node. Defaults to run_id.
    use_as_output: If True, this node's output is used as the parent
        node's output. The parent's own output event is suppressed.

  Returns:
    A Future that will resolve with the node's output when complete.
  """
  node_name = node_name or run_id

  # Record which dynamic child's output should replace the parent's.
  # Full paths as keys avoid name collisions.
  if use_as_output:
    parent_path = current_ctx.node_path
    full_child_path = join_paths(run_state.node_path, node_name)
    run_state.dynamic_output_node[parent_path] = full_child_path

  if node_name in run_state.dynamic_futures:
    return run_state.dynamic_futures[node_name]

  future: asyncio.Future[Any] = asyncio.Future()

  # 1. Check if node already executed (and we are resuming)
  if node_name in run_state.agent_state.nodes:
    node_state = run_state.agent_state.nodes[node_name]
    if node_state.status == NodeStatus.COMPLETED:
      # Already done, return result.
      node = run_state.nodes_map.get(node_name)
      full_path = join_paths(run_state.node_path, node_name)
      terminal_paths: set[str] | None = None
      if node is not None:
        from ._workflow import Workflow

        if isinstance(node, Workflow):
          terminal_paths = node._resolve_terminal_paths(full_path)
      output, _ = _get_node_output_and_route(
          ctx=run_state.ctx,
          node_path=full_path,
          run_id=run_id,
          local_events=run_state.local_output_events,
          output_schema=node.output_schema if node else None,
          terminal_paths=terminal_paths,
      )
      future.set_result(output)
      return future
    elif node_state.status == NodeStatus.FAILED:
      future.set_exception(Exception(f'Node run ({run_id}) failed.'))
      return future
    elif node_state.status == NodeStatus.WAITING:
      run_state.dynamic_futures[node_name] = future
      future.set_exception(NodeInterruptedError())
      return future

  run_state.dynamic_futures[node_name] = future

  # 2. Register node
  wrapped_node = node.model_copy(update={'name': node_name})
  run_state.nodes_map[node_name] = wrapped_node

  # Register the node in the global registry, in case not already added.
  dynamic_node_registry.register(node, run_state.node_path)

  # 3. Initialize/Update State
  if node_name not in run_state.agent_state.nodes:
    # New run
    # This schedule is called from node which will act as parent.
    parent_run_id = current_ctx.run_id if current_ctx else None

    run_state.agent_state.nodes[node_name] = NodeState(
        status=NodeStatus.PENDING,
        input=node_input,
        run_id=run_id,
        parent_run_id=parent_run_id,
        source_node_name=node.name,
    )

  # 4. Schedule
  _schedule_node(run_state, node_name)

  return future


def _check_and_schedule_nodes(run_state: _WorkflowRunState) -> None:
  """Check for nodes that need to be scheduled and schedule them.

  This function checks the state of all nodes and schedules them to run
  based on their status and trigger buffer. It also cleans up dynamic
  nodes that have been removed from state.

  Args:
    run_state: The workflow runtime state.
  """
  # Clean up nodes_map for dynamic nodes that have been removed from state
  # (cleaned up by _cleanup_child_runs in _process_triggers).
  for node_name in list(run_state.nodes_map.keys()):
    if (
        node_name not in run_state.static_node_names
        and node_name not in run_state.agent_state.nodes
    ):
      del run_state.nodes_map[node_name]

  for node_name in run_state.nodes_map:
    node_state = run_state.agent_state.nodes.get(node_name)
    status = node_state.status if node_state else NodeStatus.INACTIVE

    # 1. RUNNING but not in tasks (resuming from persisted state)
    if (
        status == NodeStatus.RUNNING
        and node_name not in run_state.running_tasks
    ):
      _schedule_node(run_state, node_name)
      continue

    # 2. PENDING
    if status == NodeStatus.PENDING:
      if node_name not in run_state.running_tasks:
        _schedule_node(run_state, node_name)
      continue

    # 3. Not Running/Pending but has buffered trigger
    if status not in (NodeStatus.RUNNING, NodeStatus.PENDING):
      buffer = run_state.agent_state.trigger_buffer.get(node_name)
      if buffer:
        # If the node state does not exist, create it now.
        if not node_state:
          node_state = NodeState()
          run_state.agent_state.nodes[node_name] = node_state

        item = buffer.pop(0)
        if not buffer:
          del run_state.agent_state.trigger_buffer[node_name]

        node_state.input = item.input
        node_state.triggered_by = item.triggered_by
        node_state.run_id = None
        node_state.status = NodeStatus.PENDING
        _schedule_node(run_state, node_name)


def _create_error_event(
    ctx: InvocationContext,
    node_path: str,
    node_name: str,
    run_id: str,
    exception: Exception,
) -> Event:
  """Creates an error event for a failed node run."""
  logger.error('Exception caught in node execution', exc_info=exception)
  return enrich_event(
      Event(
          error_code=type(exception).__name__,
          error_message=str(exception),
      ),
      ctx,
      author=get_node_name_from_path(node_path),
      node_path=join_paths(node_path, node_name),
      run_id=run_id,
      branch=True,
  )


async def _node_runner(
    run_state: _WorkflowRunState,
    node: BaseNode,
    node_input: Any,
    triggered_by: str,
    in_nodes: set[str],
    resume_inputs: dict[str, Any] | None,
    run_id: str,
    attempt_count: int,
) -> None:
  """Runs a node in a cancellable task and streams events to the queue.

  This function wraps a single node run. It streams events
  produced by the node into the event_queue. If the node is interrupted
  (e.g., by yielding RequestInput or a long-running tool), it sets
  node_interrupted=True in _NodeCompletion. After the run finishes or
  fails, it puts a final _NodeCompletion object in the queue.

  Args:
    run_state: The workflow runtime state.
    node: The node instance to execute.
    node_input: Input data passed to the node.
    triggered_by: Name of the node that triggered this run.
    in_nodes: Set of predecessor node names in the graph.
    resume_inputs: Inputs provided when resuming from an interrupt.
    run_id: Unique identifier for this run.
    retry_count: Number of times this node has been retried.
  """
  node_interrupted = False
  interrupt_ids: list[str] = []
  node_name = node.name
  has_output = False

  # Capture dynamic node metadata once for tagging events.
  _node_state = run_state.agent_state.nodes.get(node_name)
  _source_node_name = _node_state.source_node_name if _node_state else None
  _parent_run_id = _node_state.parent_run_id if _node_state else None

  # Create a closure for schedule_dynamic_node that captures run_state
  def make_schedule_dynamic_node():
    def schedule_dynamic_node_fn(
        current_ctx: Context,
        node: BaseNode,
        run_id: str,
        node_input: Any,
        *,
        node_name: str | None = None,
        use_as_output: bool = False,
    ) -> asyncio.Future[Any]:
      return _schedule_dynamic_node(
          run_state=run_state,
          current_ctx=current_ctx,
          node=node,
          run_id=run_id,
          node_input=node_input,
          node_name=node_name,
          use_as_output=use_as_output,
      )

    return schedule_dynamic_node_fn

  try:
    timeout = getattr(node, 'timeout', None)
    data_event_count = 0
    full_node_path = join_paths(run_state.node_path, node_name)
    async with asyncio.timeout(timeout):
      async for event in _execute_node(
          node=node,
          ctx=run_state.ctx,
          node_input=node_input,
          triggered_by=triggered_by,
          in_nodes=in_nodes,
          resume_inputs=resume_inputs,
          run_id=run_id,
          attempt_count=attempt_count,
          current_node_path=run_state.node_path,
          schedule_dynamic_node=make_schedule_dynamic_node(),
          transfer_targets=run_state.transfer_targets,
      ):
        # If the node delegates output via use_as_output=True,
        # suppress its own output event so the child's output is used.
        is_direct = isinstance(event, Event) and is_direct_child(
            event.node_info.path, run_state.node_path
        )
        if (
            is_direct
            and event.output is not None
            and full_node_path in run_state.dynamic_output_node
        ):
          # Create a copy without output rather than mutating the
          # original event. The child's output replaces the parent's.
          event = event.model_copy(update={'output': None})
          event.model_fields_set.discard('output')
          has_output = True

        # Enforce: a node can yield either one output or interrupts,
        # not both. rerun_on_resume=False allows at most one interrupt.
        is_output = is_direct and event.output is not None
        is_interrupt = run_state.ctx.should_pause_invocation(event)
        if is_output and (data_event_count > 0 or node_interrupted):
          raise ValueError(
              f'Node {node_name}: multiple outputs or mixed output/interrupt'
              ' not allowed. A node can yield one output or interrupts.'
          )
        if is_output:
          data_event_count += 1
        if is_interrupt and (
            data_event_count > 0
            or (not node.rerun_on_resume and node_interrupted)
        ):
          raise ValueError(
              f'Node {node_name}: mixed output/interrupt or multiple'
              ' interrupts (rerun_on_resume=False) not allowed.'
          )

        if is_interrupt:
          node_interrupted = True
          if event.long_running_tool_ids:
            interrupt_ids.extend(event.long_running_tool_ids)

        # Tag dynamic node metadata on direct child events.
        if (
            isinstance(event, Event)
            and _source_node_name
            and is_direct_child(event.node_info.path, run_state.node_path)
        ):
          if not event.node_info.source_node_name:
            event.node_info.source_node_name = _source_node_name
          if not event.node_info.parent_run_id:
            event.node_info.parent_run_id = _parent_run_id

        if isinstance(event, Event) and event.output is not None:
          has_output = True

        await run_state.event_queue.put(event)

        # Yield control back to event loop.
        # The above await event_queue.put(event) doesn't really create an
        # awaitable if the queue is not full. Since the awaitable is not
        # created, control is not yielded back to the event loop.
        await asyncio.sleep(0)

    await run_state.event_queue.put(
        _NodeCompletion(
            node_name=node_name,
            run_id=run_id,
            node_interrupted=node_interrupted,
            interrupt_ids=interrupt_ids,
            has_output=has_output,
        )
    )
  except TimeoutError:
    timeout_err = NodeTimeoutError(node_name, timeout)
    await run_state.event_queue.put(
        _create_error_event(
            run_state.ctx,
            run_state.node_path,
            node_name,
            run_id,
            timeout_err,
        )
    )
    await run_state.event_queue.put(
        _NodeCompletion(
            node_name=node_name,
            exception=timeout_err,
        )
    )
  except NodeInterruptedError:
    await run_state.event_queue.put(
        _NodeCompletion(
            node_name=node_name,
            run_id=run_id,
            node_interrupted=True,
        )
    )
  except asyncio.CancelledError:
    await run_state.event_queue.put(
        _NodeCompletion(
            node_name=node_name,
            run_id=run_id,
            is_cancelled=True,
        )
    )
    # Re-raise the exception to propagate it to the main asyncio Task,
    # so that it can be marked cancelled.
    raise
  except Exception as e:
    await run_state.event_queue.put(
        _create_error_event(
            run_state.ctx,
            run_state.node_path,
            node_name,
            run_id,
            e,
        )
    )
    await run_state.event_queue.put(
        _NodeCompletion(node_name=node_name, exception=e)
    )


def _enrich_event(
    event: Event,
    ctx: InvocationContext,
    author: str | None,
    node_path: str,
    run_id: str | None,
) -> None:
  """Local enrich for V1 runner that forces run_id assignment."""
  if not event.invocation_id:
    event.invocation_id = ctx.invocation_id
  if not event.author and author:
    event.author = author
  event.node_info.path = node_path
  if run_id:
    event.node_info.run_id = run_id


async def process_next_item(
    ctx: InvocationContext,
    run_id: str,
    parent_node_path: str,
    node: BaseNode,
    item: Any,
) -> AsyncGenerator[Event, None]:
  """Processes the next item yielded by a node's run method.

  This function converts various types of items (e.g., RequestInput, raw data)
  into Event objects and ensures they have the necessary metadata
  (author, node_name, invocation_id, run_id).

  Args:
    ctx: The invocation context.
    run_id: Unique identifier for the current node run.
    parent_node_path: The path of the workflow agent.
    node: The node instance.
    item: The item yielded by the node.

  Yields:
    An Event object.
  """
  if item is None:
    return

  # 1. Normalize non-Event items to Event objects
  # If the item is not an Event at all, wrap it.
  if not isinstance(item, Event):
    if isinstance(item, RequestInput):
      # Create a specialized Event for RequestInput (interpreted as a function
      # call).
      item = create_request_input_event(item)
    else:
      # Wrap raw data in a standard Event
      item = Event(output=item)

  # 2. Handle raw Events that are not Workflow Events
  # These are yielded as-is without further processing or enrichment.
  if not isinstance(item, Event):
    yield item
    return

  # 3. Handle Events from Sub-graphs (Nested Workflows)
  # Check if the event originated from a child node of the current node.
  subgraph_root_path = join_paths(parent_node_path, node.name)
  is_from_subgraph = is_descendant(subgraph_root_path, item.node_info.path)

  if is_from_subgraph:
    # Yield the raw event as-is (for streaming). Subgraph events are not
    # promoted to node output — the subgraph root node (e.g. a Workflow)
    # is responsible for emitting its own output event.
    yield item
    return

  # 4. Enrich Event with Metadata
  _enrich_event(
      item,
      ctx,
      author=get_node_name_from_path(parent_node_path),
      node_path=join_paths(parent_node_path, node.name),
      run_id=run_id,
  )

  # 5. Yield the processed event.
  yield item


async def _execute_node(
    *,
    node: BaseNode,
    ctx: InvocationContext,
    node_input: Any,
    triggered_by: str,
    in_nodes: set[str],
    resume_inputs: dict[str, Any] | None = None,
    run_id: str,
    current_node_path: str,
    schedule_dynamic_node: ScheduleDynamicNode,
    transfer_targets: list[_TransferTargetInfo] | None = None,
    attempt_count: int = 1,
) -> AsyncGenerator[Event, None]:
  """Executes a node in the workflow graph.

  Wraps node.run, providing it with a Context and ensuring
  all yielded items are Events, converting non-Events to Events.

  Args:
    node: The node to execute.
    ctx: The invocation context.
    node_input: Input data for the node.
    triggered_by: Name of the node that triggered this run.
    in_nodes: Set of predecessor node names in the graph.
    resume_inputs: Inputs provided when resuming from an interrupt.
    run_id: Unique identifier for this run.
    current_node_path: The path of the workflow agent authoring
      events.
    schedule_dynamic_node: Function to schedule dynamic nodes.
    transfer_targets: Transfer targets to pass to the WorkflowContext.
    retry_count: Number of times this node has been retried.

  Yields:
    Event objects from the node run.
  """
  child_path = join_paths(current_node_path, node.name)
  local_events: list[Event] = []
  wf_ctx = Context(
      ctx,
      node_path=child_path,
      triggered_by=triggered_by,
      in_nodes=in_nodes,
      resume_inputs=resume_inputs,
      run_id=run_id,
      attempt_count=attempt_count,
      schedule_dynamic_node=schedule_dynamic_node,
      node_rerun_on_resume=node.rerun_on_resume,
      local_events=local_events,
      transfer_targets=transfer_targets,
      state_schema=node.state_schema,
  )

  node_input = node._validate_input_data(node_input)
  async for item in node.run(
      ctx=wf_ctx,
      node_input=node_input,
  ):
    async for event in process_next_item(
        ctx,
        run_id,
        current_node_path,
        node,
        item,
    ):
      local_events.append(event)
      yield event
