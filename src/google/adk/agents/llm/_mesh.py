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

"""_Mesh: pure multi-agent orchestrator node.

A Node that contains a list of internal BaseNode instances and
routes execution between them based on transfer_to_agent event action.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import Field
from typing_extensions import override

from ...events.event import Event
from ...utils.context_utils import Aclosing
from ...workflow._base_node import BaseNode
from ...workflow._node import Node
from ...workflow.utils._node_path_utils import join_paths
from ..context import Context
from ._functions import find_matching_function_call
from ._transfer_target_info import _TransferTargetInfo
from .task._task_models import _as_task_request

if TYPE_CHECKING:
  from ..invocation_context import InvocationContext

logger = logging.getLogger('google_adk.' + __name__)


class _Mesh(Node):
  """Pure multi-agent orchestrator node.

  A Node that contains a list of BaseNode instances and routes
  execution between them based on transfer_to_agent event action.

  It can itself be used as a node inside a parent _Mesh (for nested
  multi-agent hierarchies).

  **Coordinator node**: If one of the internal ``nodes`` has the
  same name as this _Mesh, it is treated as the **coordinator node**.
  Transfer target rules:

    a. Coordinator can transfer to any node in the mesh.
    b. Any node can transfer to coordinator unless explicitly
       disabled (e.g. a ``BaseLlmAgent`` node with
       ``disallow_transfer_to_parent=True``).
    c. Any node can transfer to other non-coordinator nodes
       unless explicitly disabled (e.g. a ``BaseLlmAgent`` node
       with ``disallow_transfer_to_peers=True``).
    d. When coordinator runs, external transfer targets from the
       parent _Mesh's ``Context`` are appended (so
       coordinator can transfer to targets outside this mesh).
  """

  nodes: list[BaseNode] = Field(default_factory=list)
  """The internal nodes to orchestrate."""

  @override
  def model_post_init(self, context: Any) -> None:
    if not self.nodes:
      raise ValueError(f"_Mesh '{self.name}' must have at least one node.")
    super().model_post_init(context)

  # ------------------------------------------------------------------
  # BaseNode interface
  # ------------------------------------------------------------------

  @override
  async def run_node_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Orchestration loop for multi-agent transfer routing.

    Serves as both the ``BaseNode.run()`` entry point (when used as a
    node in a parent workflow) and the orchestration entry point called
    by ``LlmAgent._run_async_impl`` (which builds a ``Context``
    and delegates here).

    Steps:
      1. Find starting node (via ``_find_agent_to_run``).
      2. Build ``Context`` with transfer_targets for current
         node (computed on-the-fly by ``_build_workflow_context``).
      3. Run current node, yielding events.
      4. Check ``transfer_to_agent`` in ``EventActions``.
      5. If internal transfer -> switch ``current_node``, continue.
      6. If external transfer (name not in ``nodes``) ->
         yield event and return (let parent handle it).
      7. If no transfer -> break (agent finished naturally).
      8. Yield the Mesh's own output event (data from the last
         internal node that produced output).

    Args:
      ctx: The outer workflow context. When this _Mesh is a
        node inside a parent _Mesh, ``ctx.transfer_targets``
        carries external targets (parent/peers) that get
        appended for the coordinator node in step 2. When
        this is the top-level entry point, ``ctx`` has no
        external targets.
      node_input: Input from the parent workflow graph or mesh.

    Yields:
      ADK events from the internal nodes.
    """
    # Find starting node from session event history.
    invocation_context = ctx.get_invocation_context()
    current_node, current_task_branch = self._find_agent_to_run(
        invocation_context, mesh_path=ctx.node_path
    )

    # Track the last output from internal nodes. Each node can
    # produce at most one output event (enforced by _node_runner). At the
    # end, Mesh re-yields this output as its own output event so the
    # parent can identify Mesh's output.
    last_output_data = None
    was_interrupted = False

    while True:
      node_ctx = self._build_workflow_context(
          ctx, current_node, task_branch=current_task_branch
      )
      transfer_target_name = None
      request_task_info = None
      finish_task_info = None
      # Reset per iteration so only the final iteration's data is captured.
      last_output_data = None
      was_interrupted = False

      node_path = node_ctx.node_path
      async with Aclosing(
          current_node.run(ctx=node_ctx, node_input=node_input)
      ) as run_gen:
        async for event in run_gen:
          # Track data output from the inner node itself.
          # Only capture: (a) raw events without node_path (from
          # non-Workflow nodes), or (b) the inner Workflow node's own
          # output event (node_path matches the node's context path).
          # This excludes intermediate events from sub-nodes inside
          # a Workflow inner node.
          if isinstance(event, Event) and event.output is not None:
            if not event.node_info.path or event.node_info.path == node_path:
              last_output_data = event.output

          # Detect interruption (long-running tools, HITL).
          if event.long_running_tool_ids:
            was_interrupted = True

          yield event

          # Transfer detection: always active — transfers can cross
          # mesh boundaries (e.g. travel_agent transferring to
          # shopping_agent within root_agent's mesh).
          if (
              not transfer_target_name
              and isinstance(event, Event)
              and event.actions
              and event.actions.transfer_to_agent
          ):
            transfer_target_name = event.actions.transfer_to_agent

          # Task delegation and completion detection: only from the
          # current mesh level. When current_node has its own
          # sub-mesh (sub_agents), its internal request_task and
          # finish_task events belong to that sub-mesh and should
          # not be intercepted here.
          is_task_scope = isinstance(event, Event) and (
              self._is_coordinator_name(current_node.name)
              or not getattr(current_node, 'sub_agents', None)
          )

          if is_task_scope:
            # Task delegation detection.
            if (
                not request_task_info
                and event.actions
                and event.actions.request_task
            ):
              request_task_info = event.actions.request_task

            # Task completion detection: finish_task (task mode) or
            # event.output (single_turn controlled generation).
            if (
                not finish_task_info
                and event.actions
                and event.actions.finish_task
            ):
              finish_task_info = event.actions.finish_task
            # Single_turn agents may yield multiple events with
            # event.output (e.g. routing events from call_llm).
            # Always use the last one — it's the final output.
            if (
                event.output is not None
                and getattr(current_node, 'mode', 'chat') == 'single_turn'
            ):
              finish_task_info = {'output': event.output}

      # node_input is only used for the first node.
      node_input = None

      if transfer_target_name:
        target_node = next(
            (n for n in self.nodes if n.name == transfer_target_name),
            None,
        )
        if target_node:
          # Internal transfer: switch node, continue loop.
          current_node = target_node
          current_task_branch = None
          continue
        else:
          # Cross-mesh transfer: let parent handle routing.
          break

      elif request_task_info:
        # Task delegation: completion gate for multiple delegations.
        # request_task is a dict keyed by fc_id -> TaskRequest.
        # Process unfulfilled delegations sequentially (V1).
        request_task_event = self._find_request_task_event(
            request_task_info, invocation_context.session.events
        )
        if not request_task_event:
          # Event not yet in session; treat all as unfulfilled.
          unfulfilled = set(request_task_info.keys())
        else:
          unfulfilled = self._get_unfulfilled_fc_ids(
              request_task_event, invocation_context.session.events
          )
        if not unfulfilled:
          # All delegations already fulfilled (resume scenario).
          current_task_branch = None
          current_node = self._get_coordinator()
          continue

        fc_id = next(iter(unfulfilled))
        req = _as_task_request(request_task_info[fc_id])
        target_name = req.agent_name
        target_node = next(
            (n for n in self.nodes if n.name == target_name),
            None,
        )
        if target_node:
          node_path = self._get_node_context_path(ctx.node_path, target_name)
          current_task_branch = f'task:{node_path}.{target_name}.{fc_id}'
          current_node = target_node
          continue
        else:
          logger.warning(
              'Task agent %s not found in mesh %s',
              target_name,
              self.name,
          )
          break

      elif finish_task_info:
        # Task completion: check completion gate before returning
        # to coordinator. If the coordinator itself called finish_task,
        # the _Mesh should exit so the parent _Mesh can consume it.
        if self._is_coordinator_name(current_node.name):
          break

        # Find the originating request_task event and check for
        # unfulfilled delegations (completion gate).
        request_task_event = self._find_request_task_event_for_branch(
            current_task_branch, invocation_context.session.events
        )
        if request_task_event:
          unfulfilled = self._get_unfulfilled_fc_ids(
              request_task_event, invocation_context.session.events
          )
          if unfulfilled:
            # More delegations remain -- run next one.
            fc_id = next(iter(unfulfilled))
            req = _as_task_request(
                request_task_event.actions.request_task[fc_id]
            )
            target_name = req.agent_name
            target_node = next(
                (n for n in self.nodes if n.name == target_name),
                None,
            )
            if target_node:
              node_path = self._get_node_context_path(
                  ctx.node_path, target_name
              )
              current_task_branch = f'task:{node_path}.{target_name}.{fc_id}'
              current_node = target_node
              continue

        # All delegations fulfilled (or no originating event found).
        current_task_branch = None
        current_node = self._get_coordinator()
        continue

      else:
        # No transfer, no task action.
        agent_mode = getattr(current_node, 'mode', 'chat')
        if agent_mode == 'task' and not self._is_coordinator_name(
            current_node.name
        ):
          # Task agent produced text (asking user for input).
          # End invocation; _find_agent_to_run resumes next turn.
          break
        break

    # Yield the Mesh's own output event so the parent can identify
    # Mesh's output. The event has no metadata — the parent's
    # process_next_item will enrich it.
    # Only emit when the inner execution completed normally (not
    # interrupted by long-running tools or HITL).
    if last_output_data is not None and not was_interrupted:
      yield Event(output=last_output_data)

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _get_coordinator(self) -> BaseNode:
    """Returns the coordinator node.

    The coordinator is the node whose name matches ``self.name``.
    If no node matches, falls back to the first node in ``nodes``.
    """
    coord = next((n for n in self.nodes if n.name == self.name), None)
    if coord:
      return coord
    return self.nodes[0]

  def _is_coordinator_name(self, name: str) -> bool:
    """Check if the given name is the coordinator's name."""
    return name == self._get_coordinator().name

  def _get_unfulfilled_fc_ids(
      self,
      request_task_event: Event,
      session_events: list[Event],
  ) -> set[str]:
    """Derive unfulfilled delegation fc_ids from session events.

    Compares all fc_ids in the originating ``request_task`` event
    against completion events in the session (``finish_task`` or
    ``event.output``), matching by branch suffix. Returns fc_ids
    that have no matching completion.

    Args:
      request_task_event: The event containing the ``request_task``
        action dict with all delegation fc_ids.
      session_events: The session events to scan for completions.

    Returns:
      Set of fc_ids that have not yet been fulfilled.
    """
    all_fc_ids = set(request_task_event.actions.request_task.keys())
    fulfilled: set[str] = set()
    for event in session_events:
      has_completion = (
          event.actions and event.actions.finish_task
      ) or event.output is not None
      if not has_completion:
        continue
      if event.branch:
        for fc_id in all_fc_ids:
          if event.branch.endswith(f'.{fc_id}'):
            fulfilled.add(fc_id)
    return all_fc_ids - fulfilled

  def _find_request_task_event(
      self,
      request_task_info: dict,
      session_events: list[Event],
  ) -> Event | None:
    """Find the session event that originated a request_task dict.

    Searches for the event whose ``actions.request_task`` contains
    any of the same fc_id keys as ``request_task_info``. Uses
    intersection matching because fc_ids are UUIDs unique across
    invocations, so any overlap means it's the correct event.

    Args:
      request_task_info: The request_task dict (fc_id -> TaskRequest).
      session_events: The session events to search.

    Returns:
      The originating event, or None if not found.
    """
    target_fc_ids = set(request_task_info.keys())
    for event in reversed(session_events):
      if not event.actions or not event.actions.request_task:
        continue
      if target_fc_ids & set(event.actions.request_task.keys()):
        return event
    return None

  def _find_request_task_event_for_branch(
      self,
      task_branch: str | None,
      session_events: list[Event],
  ) -> Event | None:
    """Find the request_task event that created a given task branch.

    Matches the branch's fc_id suffix against ``request_task`` event
    entries. Used after ``finish_task`` to locate the originating
    delegation and check for remaining unfulfilled delegations.

    Args:
      task_branch: The task branch string (e.g.
        ``task:path.agent.fc_id``).
      session_events: The session events to search.

    Returns:
      The originating event, or None if not found.
    """
    if not task_branch:
      return None
    for event in reversed(session_events):
      if not event.actions or not event.actions.request_task:
        continue
      for fc_id in event.actions.request_task:
        if task_branch.endswith(f'.{fc_id}'):
          return event
    return None

  def _get_node_context_path(self, mesh_path: str, node_name: str) -> str:
    """Computes the node_path for a node within this mesh.

    Coordinator shares the mesh's own path. Non-coordinator gets
    ``mesh_path/node_name``, following the same convention as
    ``node_runner._execute_node`` (``join_paths(parent_path,
    node_name)``).

    Used by ``_build_workflow_context`` for context creation.

    Args:
      mesh_path: The mesh's own node_path (from
        ``Context.node_path``).
      node_name: The name of the node.

    Returns:
      The node_path to use for the given node.
    """
    if self._is_coordinator_name(node_name):
      return mesh_path
    return join_paths(mesh_path, node_name)

  def _find_agent_to_run(
      self, ctx: InvocationContext, mesh_path: str
  ) -> tuple[BaseNode, str | None]:
    """Determine which internal node should run next.

    Examines session events to find the correct starting agent.
    Similar to ``Runner._find_agent_to_run`` but scoped to this
    mesh's internal nodes.

    Uses three strategies in priority order:

      **Strategy 1 – Function response matching (HITL resume).**
      Finds an unmatched function_call in the session (a call with
      no corresponding function_response). This indicates a node
      was interrupted (auth, confirmation, or long-running tool)
      and needs to resume. The function_call's author identifies
      which node to re-enter.

      **Strategy 2 – Active task delegation.**
      Scans events in reverse for ``request_task`` /
      ``finish_task`` actions scoped to this mesh. Uses a
      completion gate to find unfulfilled delegations and routes
      to the target task agent with the correct branch.

      **Strategy 3 – Last active node (fallback).**
      Scans events in reverse to find the most recent event
      authored by one of this mesh's nodes. Returns that node so
      it can continue where it left off. Skips bookkeeping events,
      one-shot agents (``disallow_transfer_to_parent``), and
      LoopAgent nodes (which should not be re-entered after
      ``exit_loop`` completion). Falls back to the coordinator
      if no match is found.

    Args:
      ctx: The current invocation context.
      mesh_path: The mesh's own node_path (from
        ``Context.node_path``), used for event scoping.

    Returns:
      A tuple of (node, task_branch). ``task_branch`` is the
      correctly constructed branch string when resuming a task
      delegation, or None for non-task scenarios.
    """
    nodes_by_name: dict[str, BaseNode] = {n.name: n for n in self.nodes}
    coordinator = self._get_coordinator()

    # --- Strategy 1: Function response matching (HITL resume) ---
    # Find an unmatched function_call (no corresponding
    # function_response). Its author tells us which node was
    # interrupted and needs to resume.
    fc_event = find_matching_function_call(ctx.session.events)
    if fc_event and fc_event.author:
      author = fc_event.author
      if author in nodes_by_name:
        # Check event belongs to this mesh (not a same-named node
        # in a different mesh).
        np = fc_event.node_info.path
        if np == mesh_path or np.startswith(mesh_path + '/'):
          node = nodes_by_name[author]
          # Recover task branch for HITL resume of task agents.
          task_branch = None
          if fc_event.branch and fc_event.branch.startswith('task:'):
            task_branch = fc_event.branch
          return node, task_branch

    # --- Strategy 2: Active task delegation ---
    # Scan events in reverse for the most recent request_task
    # action scoped to this mesh, then use the completion gate
    # to find unfulfilled delegations.
    for event in reversed(ctx.session.events):
      if not event.actions:
        continue
      np = getattr(event, 'node_path', '')
      if np and np != mesh_path and not np.startswith(mesh_path + '/'):
        continue

      if event.actions.request_task:
        # Use completion gate: find unfulfilled delegations.
        unfulfilled = self._get_unfulfilled_fc_ids(event, ctx.session.events)
        if not unfulfilled:
          # All delegations fulfilled -- no active delegation.
          break
        # Pick the first unfulfilled delegation.
        fc_id = next(iter(unfulfilled))
        target_name = _as_task_request(
            event.actions.request_task[fc_id]
        ).agent_name
        if target_name and target_name in nodes_by_name:
          node_path = self._get_node_context_path(mesh_path, target_name)
          task_branch = f'task:{node_path}.{target_name}.{fc_id}'
          return nodes_by_name[target_name], task_branch
        break

      if event.actions.finish_task:
        # finish_task found before any request_task -- check if
        # the originating request_task has remaining unfulfilled
        # delegations.
        req_event = self._find_request_task_event_for_branch(
            event.branch, ctx.session.events
        )
        if req_event:
          unfulfilled = self._get_unfulfilled_fc_ids(
              req_event, ctx.session.events
          )
          if unfulfilled:
            fc_id = next(iter(unfulfilled))
            target_name = _as_task_request(
                req_event.actions.request_task[fc_id]
            ).agent_name
            if target_name and target_name in nodes_by_name:
              node_path = self._get_node_context_path(mesh_path, target_name)
              task_branch = f'task:{node_path}.{target_name}.{fc_id}'
              return nodes_by_name[target_name], task_branch
        # All delegations fulfilled -- no active delegation.
        break

    # --- Strategy 3: Last active node (fallback) ---
    # Scan events in reverse to find the most recent event
    # belonging to one of this mesh's nodes. We use node_path
    # (not author) to determine ownership, because wrapper nodes
    # like SequentialAgent only emit bookkeeping events under their
    # own name — their children's events carry different authors
    # but share the same node_path prefix.
    prefix = mesh_path + '/'
    for event in reversed(ctx.session.events):
      if event.author == 'user':
        continue
      if event.actions and (
          event.actions.agent_state is not None or event.actions.end_of_agent
      ):
        continue
      if event.branch:
        continue  # Task/single_turn events — handled by Strategy 2.

      # Scope check: event must belong to this mesh (same as
      # Strategy 2, line 573).
      np = event.node_info.path
      if not np or (np != mesh_path and not np.startswith(prefix)):
        continue

      # Identify the owning mesh node. Try author first (consistent
      # with Strategy 1 & 2), then fall back to node_path extraction
      # for events from descendant nodes (e.g. sub_agent_1_1 inside
      # SequentialAgent sub_agent_1).
      author = event.author
      if author in nodes_by_name:
        node = nodes_by_name[author]
      elif np.startswith(prefix):
        child_name = np[len(prefix) :].split('/')[0]
        if child_name in nodes_by_name:
          node = nodes_by_name[child_name]
        else:
          node = coordinator
      else:
        node = coordinator

      # If the node has disallow_transfer_to_parent=True, it's a
      # "one-shot" agent that auto-returns to coordinator. This
      # keeps the same behavior as the original LlmAgent.
      if getattr(node, 'disallow_transfer_to_parent', False):
        continue

      # LoopAgent should not be re-entered on subsequent
      # invocations — its exit_loop escalation means the loop
      # completed. Skip it so the scan falls through to the
      # coordinator's own events (or to the default coordinator
      # return at the end). This only affects the completion case;
      # interrupt/resume inside a LoopAgent is handled by Strategy 1
      # (function response matching), not Strategy 3.
      from ..loop_agent import LoopAgent

      if isinstance(node, LoopAgent):
        continue

      return node, None

    return coordinator, None

  def _build_workflow_context(
      self,
      ctx: Context,
      node: BaseNode,
      *,
      task_branch: str | None = None,
  ) -> Context:
    """Creates a Context with transfer_targets for the given node.

    Computes transfer targets on-the-fly based on the node's role.
    Ordering matters: ``transfer_targets[0]`` is the default
    fallback agent (used by ``agent_transfer.py``).

      a. Coordinator: external targets first (parent coordinator
         as default fallback for escalation), then all other
         local nodes in the mesh (excluding task/single_turn).
      b. Non-coordinator: coordinator first (default fallback,
         unless ``disallow_transfer_to_parent``), then peers
         (unless ``disallow_transfer_to_peers``, excluding
         task/single_turn).

    Args:
      ctx: The outer workflow context.
      node: The node that will run next.
      task_branch: Optional branch string for task content isolation.

    Returns:
      A ``Context`` with the appropriate transfer_targets.
    """
    node_name = node.name
    is_coordinator = self._is_coordinator_name(node_name)

    if is_coordinator:
      # External targets first (parent coordinator = default
      # fallback for escalation).
      targets = list(ctx.transfer_targets) if ctx.transfer_targets else []
      # Then local non-coordinator nodes (excluding task/single_turn
      # agents, which are reached via request_task tools).
      targets.extend([
          _TransferTargetInfo(
              name=n.name,
              description=getattr(n, 'description', ''),
          )
          for n in self.nodes
          if not self._is_coordinator_name(n.name)
          and getattr(n, 'mode', 'chat') not in ('task', 'single_turn')
      ])
    else:
      # Task/single_turn agents have no transfer targets.
      # They complete via finish_task, not transfer_to_agent.
      agent_mode = getattr(node, 'mode', 'chat')
      if agent_mode in ('task', 'single_turn'):
        targets = []
      else:
        targets = []
        disallow_parent = getattr(node, 'disallow_transfer_to_parent', False)
        disallow_peers = getattr(node, 'disallow_transfer_to_peers', False)

        # Coordinator first (default fallback).
        if not disallow_parent:
          coordinator_node = self._get_coordinator()
          if coordinator_node:
            targets.append(
                _TransferTargetInfo(
                    name=coordinator_node.name,
                    description=getattr(coordinator_node, 'description', ''),
                )
            )

        # Then peers (excluding task/single_turn agents).
        if not disallow_peers:
          for n in self.nodes:
            n_name = n.name
            if n_name == node_name or self._is_coordinator_name(n_name):
              continue
            if getattr(n, 'mode', 'chat') in ('task', 'single_turn'):
              continue
            targets.append(
                _TransferTargetInfo(
                    name=n_name,
                    description=getattr(n, 'description', ''),
                )
            )

    node_path = self._get_node_context_path(ctx.node_path, node_name)

    # Propagate the mesh hierarchy into node_path so that
    # Workflow nodes (e.g. _SingleLlmAgent) compute
    # hierarchical event paths instead of flat ones.
    # Also set branch for task content isolation when applicable.
    ic_update: dict[str, Any] = {'node_path': node_path}
    if task_branch is not None:
      ic_update['branch'] = task_branch

    ic = ctx._invocation_context.model_copy(update=ic_update)

    return Context(
        invocation_context=ic,
        node_path=node_path,
        run_id=ctx.run_id,
        local_events=[],
        transfer_targets=targets,
    )
