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

import asyncio
import contextvars
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_serializer
from pydantic import TypeAdapter
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.base_agent import BaseAgentState
from ..agents.context import Context
from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..events.event_actions import EventActions
from ._base_node import BaseNode
from ._base_node import START
from ._dynamic_node_registry import dynamic_node_registry
from ._errors import NodeInterruptedError
from ._node import Node
from ._node_runner import _check_and_schedule_nodes
from ._node_runner import _schedule_node
from ._node_runner import _schedule_retry_task
from ._node_state import NodeState
from ._node_status import NodeStatus
from ._run_state import _NodeCompletion
from ._run_state import _NodeResumption
from ._run_state import _WorkflowRunState
from ._trigger import Trigger
from ._trigger_processor import _get_next_pending_nodes
from ._trigger_processor import _process_triggers
from ._workflow_graph import EdgeItem
from ._workflow_graph import WorkflowGraph
from .utils._agent_state_utils import reconstruct_state_from_events
from .utils._event_utils import enrich_event
from .utils._node_path_utils import is_descendant
from .utils._node_path_utils import is_direct_child
from .utils._node_path_utils import join_paths
from .utils._retry_utils import _get_retry_delay
from .utils._retry_utils import _should_retry_node
from .utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from .utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from .utils._workflow_hitl_utils import unwrap_response

workflow_node_input: contextvars.ContextVar[Any] = contextvars.ContextVar(
    'workflow_node_input', default=None
)

workflow_transfer_targets: contextvars.ContextVar[list[Any]] = (
    contextvars.ContextVar('workflow_transfer_targets', default=[])
)


class WorkflowAgentState(BaseAgentState):
  """State for Workflow."""

  model_config = ConfigDict(extra='ignore', ser_json_bytes='base64')

  nodes: dict[str, NodeState] = Field(
      default_factory=dict,
      description='The state of each node in the workflow.',
  )

  trigger_buffer: dict[str, list[Trigger]] = Field(
      default_factory=dict,
      description='Buffered triggers for nodes that are currently running.',
  )

  def _dump_exclude_none(self, obj: Any) -> Any:
    if isinstance(obj, BaseModel):
      return obj.model_dump(mode='json', exclude_none=True)
    elif isinstance(obj, list):
      return [self._dump_exclude_none(item) for item in obj]
    elif isinstance(obj, dict):
      return {k: self._dump_exclude_none(v) for k, v in obj.items()}
    else:
      return obj

  @model_serializer(when_used='json')
  def serialize_model(self) -> dict[str, Any]:
    state = {
        'nodes': {k: self._dump_exclude_none(v) for k, v in self.nodes.items()},
    }
    if self.trigger_buffer:
      state['trigger_buffer'] = self._dump_exclude_none(self.trigger_buffer)
    return state


def new_workflow_agent_state() -> WorkflowAgentState:
  """Returns a new empty WorkflowAgentState."""
  return WorkflowAgentState()


class Workflow(BaseAgent, Node):
  """A graph based runtime for workflows.

  This runtime involves agents, tools, and other nodes.
  """

  rerun_on_resume: bool = Field(default=True)

  graph: Optional[WorkflowGraph] = Field(
      description='The workflow graph of the workflow agent.',
      default=None,
  )

  input_schema: Optional[Any] = Field(
      description='The schema to parse the input content in START node.',
      default=None,
  )

  edges: list[EdgeItem] = Field(
      description='Edges to build an implicit workflow graph.',
      default_factory=list,
  )

  max_concurrency: int | None = Field(default=None, ge=1)
  """Maximum number of parallel nodes to run. Default is None (no limit)."""

  @override
  def _load_agent_state(
      self,
      ctx: InvocationContext,
      state_type: type[BaseAgentState],
  ) -> Optional[BaseAgentState]:
    if ctx.node_path is None:
      return super()._load_agent_state(ctx, state_type)

    if not ctx.agent_states or ctx.node_path not in ctx.agent_states:
      return None
    return state_type.model_validate(ctx.agent_states.get(ctx.node_path))

  @override
  def _create_agent_state_event(self, ctx: InvocationContext) -> Event:
    if ctx.node_path is None:
      return super()._create_agent_state_event(ctx)

    event_actions = EventActions()
    if (agent_state := ctx.agent_states.get(ctx.node_path)) is not None:
      event_actions.agent_state = agent_state
    if ctx.end_of_agents.get(ctx.node_path):
      event_actions.end_of_agent = True

    return enrich_event(
        Event(actions=event_actions),
        ctx,
        author=self.name,
    )

  @override
  def _create_invocation_context(
      self, parent_context: InvocationContext
  ) -> InvocationContext:
    """Creates a new invocation context for this agent."""
    path = parent_context.node_path
    if not path:
      path = join_paths(path, self.name)
    return parent_context.model_copy(update={'agent': self, 'node_path': path})

  @override
  async def run_node_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the agent as a node."""
    token_input = workflow_node_input.set(node_input)
    token_targets = workflow_transfer_targets.set(ctx.transfer_targets)
    try:
      # Create the invocation context for this agent execution, deriving from
      # the parent context.
      child_ctx = self._create_invocation_context(ctx._invocation_context)
      if ctx.node_path:
        child_ctx.node_path = ctx.node_path
      async for event in self.run_async(parent_context=child_ctx):
        yield event
    finally:
      # Clear overrides so they don't affect the next run.
      workflow_node_input.reset(token_input)
      workflow_transfer_targets.reset(token_targets)

  @override
  def model_post_init(self, context: Any) -> None:
    if not self.rerun_on_resume:
      raise ValueError(
          f'rerun_on_resume should not be False in Workflow. {self.name}'
      )

    if self.graph and self.edges:
      raise ValueError("Cannot specify both 'graph' and 'edges'.")

    if not self.graph and self.edges:
      self.graph = WorkflowGraph.from_edge_items(self.edges)

    # SequentialAgent, LoopAgent, and ParallelAgent accept sub_agents and
    # convert them into edges/graph in their own model_post_init before
    # calling super(). Skip this check for those subclasses.
    from ..agents.loop_agent import LoopAgent
    from ..agents.parallel_agent import ParallelAgent
    from ..agents.sequential_agent import SequentialAgent

    if self.sub_agents and not isinstance(
        self, (SequentialAgent, LoopAgent, ParallelAgent)
    ):
      raise ValueError('sub_agents is not supported in Workflow.')

    super().model_post_init(context)
    if not self.graph:
      if isinstance(self, (SequentialAgent, LoopAgent, ParallelAgent)):
        # Allow empty sub_agents for backward compatibility.
        return
      if isinstance(self, Workflow):
        raise ValueError(
            "Workflow must have either 'graph' or 'edges' specified."
        )
      return

    self.graph.validate_graph()

  def _resolve_terminal_paths(self, prefix: str) -> set[str]:
    """Recursively resolves leaf-level terminal event paths.

    For each terminal node in the graph, computes its full path. If a
    terminal is itself a Workflow, recurses into it to find its leaf
    terminals. Returns the set of all leaf paths that produce the
    workflow's output events.
    """
    if not self.graph:
      return set()
    paths: set[str] = set()
    for name in self.graph._terminal_node_names:
      full_path = join_paths(prefix, name)
      node = next((n for n in self.graph.nodes if n.name == name), None)
      if isinstance(node, Workflow) and node.graph:
        paths |= node._resolve_terminal_paths(full_path)
      else:
        paths.add(full_path)
    return paths

  def _get_workflow_state(self, ctx: InvocationContext) -> WorkflowAgentState:
    state = self._load_agent_state(ctx, WorkflowAgentState)
    if not state or not state.nodes:
      # Try reconstructing from session events (non-resumable HITL).
      reconstructed_nodes = reconstruct_state_from_events(
          session_events=list(ctx.session.events),
          current_invocation_id=ctx.invocation_id,
          workflow_path=ctx.node_path or '',
          graph=self.graph,
      )
      if reconstructed_nodes:
        state = WorkflowAgentState(nodes=reconstructed_nodes)
      else:
        state = new_workflow_agent_state()
    return state

  async def _checkpoint_agent_state(
      self, ctx: InvocationContext, agent_state: WorkflowAgentState
  ) -> AsyncGenerator[Event, None]:
    ctx.set_agent_state(ctx.node_path, agent_state=agent_state)
    if ctx.is_resumable:
      yield self._create_agent_state_event(ctx)

  def _parse_interrupt_responses(
      self,
      ctx: InvocationContext,
      agent_state: WorkflowAgentState,
      nodes_map: dict[str, BaseNode],
  ) -> list[_NodeResumption]:
    """Parses interrupt responses from user_content and identifies nodes to resume."""
    resumptions: list[_NodeResumption] = []
    if not ctx.user_content:
      return resumptions

    for node_state in agent_state.nodes.values():
      if node_state.interrupts:
        break
    else:
      # No node has any pending interrupts.
      return resumptions

    fr_parts_by_id = {
        part.function_response.id: part
        for part in ctx.user_content.parts
        if part.function_response
    }

    for node_name, node_state in agent_state.nodes.items():
      for interrupt_id in node_state.interrupts:
        if interrupt_id in fr_parts_by_id:
          resolved_part = fr_parts_by_id[interrupt_id]
          node = nodes_map[node_name]
          resumptions.append(
              _NodeResumption(
                  node_name=node_name,
                  interrupt_id=interrupt_id,
                  response_part=resolved_part,
                  rerun_on_resume=node.rerun_on_resume,
              )
          )
    return resumptions

  def _parse_start_input(self, user_content: types.Content) -> Any:
    if self.input_schema:
      text = ''.join(part.text for part in user_content.parts if part.text)
      try:
        return TypeAdapter(self.input_schema).validate_json(text)
      except Exception:
        try:
          return TypeAdapter(self.input_schema).validate_python(text)
        except Exception as e:
          raise ValueError(
              f'Failed to parse input content into schema: {e}'
          ) from e
    else:
      return user_content

  def _seed_start_triggers(
      self, ctx: InvocationContext, agent_state: WorkflowAgentState
  ) -> None:
    """Seeds triggers for START's successors.

    On the first turn (no existing node states), determines the workflow input
    and directly populates the trigger buffer for all nodes downstream of
    START.  START is a sentinel and is never executed.
    """
    # Only seed on the first turn — if nodes already exist, we're resuming.
    if agent_state.nodes:
      return

    # Determine the input.
    node_input = workflow_node_input.get()
    if node_input is None and ctx.user_content and ctx.user_content.parts:
      node_input = self._parse_start_input(ctx.user_content)

    # Find all START successors and seed their trigger buffers.
    from ._trigger import Trigger

    for edge in self.graph.edges:
      if edge.from_node.name == START.name:
        agent_state.trigger_buffer.setdefault(edge.to_node.name, []).append(
            Trigger(input=node_input, triggered_by=START.name)
        )

  def _rehydrate_dynamic_nodes(
      self, agent_state: WorkflowAgentState, nodes_map: dict[str, BaseNode]
  ) -> None:
    """Rehydrates dynamic nodes from the registry."""
    for node_name, node_state in agent_state.nodes.items():
      if node_name not in nodes_map and node_state.source_node_name:
        source_node = dynamic_node_registry.get(
            node_state.source_node_name, self.name
        )
        if not source_node:
          # Dynamic nodes are auto registered when they are scheduled using
          # ctx.run_node(). But in cases of resuming a workflow in another
          # process, the registry might be empty. So devs would need to
          # manually register the node using dynamic_node_registry.register().
          raise ValueError(
              f'Source node [{node_state.source_node_name}] for dynamic node'
              f' [{node_name}] not found in registry. Use'
              ' dynamic_node_registry.register() to manually register the'
              ' node.'
          )
        nodes_map[node_name] = source_node.model_copy(
            update={'name': node_name}
        )

  def _process_resumptions(
      self,
      ctx: InvocationContext,
      agent_state: WorkflowAgentState,
      nodes_map: dict[str, BaseNode],
  ) -> tuple[list[Event], dict[str, str]]:
    """Processes interrupt responses and returns events and completed nodes."""
    resume_response_events: list[Event] = []
    completed_node_execution_ids: dict[str, str] = {}

    # Check for and process any function responses that resolve interrupts.
    # Based on node's rerun_on_resume setting, we either mark the node
    # as PENDING for rerun, or COMPLETE it with function response.
    resumptions = self._parse_interrupt_responses(ctx, agent_state, nodes_map)

    for resumption in resumptions:
      node_name = resumption.node_name
      interrupt_id = resumption.interrupt_id
      response_part = resumption.response_part

      # Remove the interrupt ID that has been resolved by a function response.
      node_state = agent_state.nodes[node_name]
      node_state.interrupts = [
          intr_id
          for intr_id in node_state.interrupts
          if intr_id != interrupt_id
      ]

      response_data = response_part.function_response.response
      if response_part.function_response.name in (
          REQUEST_INPUT_FUNCTION_CALL_NAME,
          REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
      ):
        response_data = unwrap_response(response_data)

      if resumption.rerun_on_resume:
        # Store the function response as resume_inputs.
        node_state.resume_inputs[interrupt_id] = response_data
        # Only rerun when ALL pending interrupts are resolved.
        if not node_state.interrupts:
          node_state.status = NodeStatus.PENDING
      else:
        # If node is not configured to rerun on resume, we treat the function
        # response as node output, and prepare to emit it.
        event = enrich_event(
            Event(output=response_data),
            ctx,
            author=self.name,
            node_path=join_paths(ctx.node_path or '', node_name),
            execution_id=node_state.execution_id or '',
            branch=True,
        )

        resume_response_events.append(event)
        # If all interrupts for this node are resolved, mark as COMPLETED,
        # otherwise it remains WAITING.
        if not node_state.interrupts:
          node_state.status = NodeStatus.COMPLETED
          completed_node_execution_ids[node_name] = (
              event.node_info.execution_id or ''
          )
        else:
          node_state.status = NodeStatus.WAITING

    return resume_response_events, completed_node_execution_ids

  async def _handle_node_completion(
      self,
      completion: _NodeCompletion,
      run_state: _WorkflowRunState,
  ) -> AsyncGenerator[Event, None]:
    """Handles node completion events from the queue."""
    node_name = completion.node_name
    if run_state.running_tasks.pop(node_name, None) is not None:
      run_state.running_node_count -= 1
    agent_state = run_state.agent_state
    node_state = agent_state.nodes[node_name]

    # If a node was cancelled, update its status.
    if completion.is_cancelled:
      node_state.status = NodeStatus.CANCELLED
      async for event in self._checkpoint_agent_state(
          run_state.ctx, agent_state
      ):
        yield event
      return

    # If a node failed, check if we need to retry it based on retry_config
    if completion.exception:
      retry_config = run_state.nodes_map[node_name].retry_config
      if _should_retry_node(completion.exception, retry_config, node_state):
        # Mark node as PENDING and schedule it for retry.
        node_state.status = NodeStatus.PENDING
        node_state.retry_count += 1

        delay = _get_retry_delay(retry_config, node_state)
        if delay:
          _schedule_retry_task(
              run_state, node_name, delay, self._checkpoint_agent_state
          )
        else:
          _schedule_node(run_state, node_name)

        async for event in self._checkpoint_agent_state(
            run_state.ctx, agent_state
        ):
          yield event
        return

      # If there is no retry config or should_retry is false, handle failure
      node_state.status = NodeStatus.FAILED

      async for event in self._checkpoint_agent_state(
          run_state.ctx, agent_state
      ):
        yield event

      logging.error(
          'Node %r failed',
          node_name,
          exc_info=completion.exception,
      )
      # If the node is a dynamic node, resolve its future with the exception.
      if node_name in run_state.dynamic_futures:
        future = run_state.dynamic_futures.pop(node_name)
        if not future.done():
          future.set_exception(completion.exception)

        # If the node is a dynamic node, resolve its future with the exception and just return here.
        # Dynamic node failures should be handled by the parent composite node.
        # It can decide whether to retry, ignore, or fail itself.
        # We return here & do not propagate the exception to the workflow runner,
        # otherwise it will cancel other running nodes.
        return

      raise completion.exception

    # If the node was interrupted, mark it as WAITING and store
    # interrupt IDs in the state. The workflow will pause.
    if completion.node_interrupted:
      node_state.status = NodeStatus.WAITING
      if completion.interrupt_ids:
        node_state.interrupts.extend(completion.interrupt_ids)
      async for event in self._checkpoint_agent_state(
          run_state.ctx, agent_state
      ):
        yield event

      # If a dynamic node is interrupted, we need to propagate the
      # interruption to the parent Future. Otherwise the parent Future
      # will never complete.
      if node_name in run_state.dynamic_futures:
        future = run_state.dynamic_futures[node_name]
        if not future.done():
          future.set_exception(NodeInterruptedError())
      return

    # Nodes with wait_for_output=True only transition to COMPLETED
    # when they produce output. Otherwise they move to WAITING, ready
    # to be re-triggered (e.g. JoinNode waiting for more predecessors).
    node = run_state.nodes_map.get(node_name)
    waiting_for_output = (
        node is not None
        and getattr(node, 'wait_for_output', False)
        and not completion.has_output
    )
    node_state.status = (
        NodeStatus.WAITING if waiting_for_output else NodeStatus.COMPLETED
    )

    _process_triggers(
        run_state=run_state,
        schedule_node=lambda n: _schedule_node(run_state, n),
        node_name=node_name,
        execution_id=completion.execution_id,
    )

    _check_and_schedule_nodes(run_state)

    async for event in self._checkpoint_agent_state(run_state.ctx, agent_state):
      yield event

  async def _process_event_queue_item(
      self,
      item: Event | _NodeCompletion,
      run_state: _WorkflowRunState,
  ) -> AsyncGenerator[Event, None]:
    """Processes a single item from the event queue."""
    if isinstance(item, _NodeCompletion):
      async for event in self._handle_node_completion(item, run_state):
        yield event
    elif isinstance(item, Event):
      async for event in self._handle_adk_event(item, run_state):
        yield event

  async def _handle_adk_event(
      self,
      event: Event,
      run_state: _WorkflowRunState,
  ) -> AsyncGenerator[Event, None]:
    """Handles Events (intermediate output or agent state updates)."""
    is_event_from_direct_child = (
        isinstance(event, Event)
        and event.node_info.path
        and is_direct_child(event.node_info.path, run_state.node_path)
    )

    # Collect events from all descendants into local_output_events so
    # _get_node_output_events can find terminal node events from nested
    # workflows via terminal_paths resolution.
    is_from_descendant = (
        not is_event_from_direct_child
        and isinstance(event, Event)
        and event.node_info.path
        and is_descendant(run_state.node_path or '', event.node_info.path)
    )

    if is_event_from_direct_child:
      run_state.local_output_events.append(event)

      if event.actions and event.actions.state_delta:
        for key, value in event.actions.state_delta.items():
          if value is None:
            run_state.ctx.session.state.pop(key, None)
          else:
            run_state.ctx.session.state[key] = value

    elif is_from_descendant:
      run_state.local_output_events.append(event)

    yield event

  async def _finalize_workflow(
      self,
      ctx: InvocationContext,
      agent_state: WorkflowAgentState,
  ) -> AsyncGenerator[Event, None]:
    """Finalizes the workflow if all nodes completed successfully."""
    # If any node is interrupted, pause execution by returning early.
    if any(
        node_state.status == NodeStatus.WAITING
        for node_state in agent_state.nodes.values()
    ):
      return

    if not self.graph:
      ctx.set_agent_state(ctx.node_path, end_of_agent=True)
      if ctx.is_resumable:
        yield self._create_agent_state_event(ctx)
      return

    # If all nodes completed without interruption, mark agent as ended.
    ctx.set_agent_state(ctx.node_path, end_of_agent=True)
    if ctx.is_resumable:
      yield self._create_agent_state_event(ctx)

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    if self.graph is None:
      return
    agent_state = self._get_workflow_state(ctx)

    nodes_map = {
        node.name: node for node in self.graph.nodes if node.name != START.name
    }
    static_node_names = set(nodes_map.keys())

    # Rehydrate dynamic nodes from registry
    self._rehydrate_dynamic_nodes(agent_state, nodes_map)

    # Holds all dynamically scheduled node futures, keyed by execution_id.
    _dynamic_futures: dict[str, asyncio.Future[Any]] = {}

    # Read/Write buffer for local events.
    local_output_events: list[Event] = []

    # Process interruptions and resumptions
    resume_response_events, completed_node_execution_ids = (
        self._process_resumptions(ctx, agent_state, nodes_map)
    )

    running_tasks: dict[str, asyncio.Task] = {}
    event_queue: asyncio.Queue[Event | _NodeCompletion] = asyncio.Queue()

    run_state = _WorkflowRunState(
        ctx=ctx,
        event_queue=event_queue,
        graph=self.graph,
        node_path=ctx.node_path,
        agent_state=agent_state,
        nodes_map=nodes_map,
        running_tasks=running_tasks,
        dynamic_futures=_dynamic_futures,
        local_output_events=local_output_events,
        static_node_names=static_node_names,
        transfer_targets=workflow_transfer_targets.get(),
        max_concurrency=self.max_concurrency,
    )

    # If any nodes were resumed with rerun_on_resume=False, they
    # yield their Event here. If they are now fully COMPLETED, we also
    # schedule their downstream nodes.
    if resume_response_events:
      local_output_events.extend(resume_response_events)
      for event in resume_response_events:
        yield event

    if completed_node_execution_ids:
      for node_name, execution_id in completed_node_execution_ids.items():
        _process_triggers(
            run_state=run_state,
            schedule_node=lambda n: _schedule_node(run_state, n),
            node_name=node_name,
            execution_id=execution_id,
        )

    # On the first turn, seed triggers for START's successors.
    self._seed_start_triggers(ctx, agent_state)

    _check_and_schedule_nodes(run_state)

    # Snapshot which node has been scheduled to run. This way devs can see in
    # the history which node was scheduled to run.
    async for event in self._checkpoint_agent_state(ctx, agent_state):
      yield event

    # Main event loop: process events from running nodes until all tasks are
    # complete.
    try:
      while running_tasks:
        queue_item = await event_queue.get()
        async for event in self._process_event_queue_item(
            queue_item, run_state
        ):
          yield event
        event_queue.task_done()
    finally:
      # If the workflow is cancelled or node has failed, cancel all running tasks.
      for task in running_tasks.values():
        task.cancel()

      # Drain the event queue to process events from cancelling nodes.
      while running_tasks or not event_queue.empty():
        queue_item = await event_queue.get()
        async for event in self._process_event_queue_item(
            queue_item, run_state
        ):
          yield event
        event_queue.task_done()

    # Finalize the workflow
    async for event in self._finalize_workflow(ctx, agent_state):
      yield event
