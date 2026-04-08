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
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

from typing_extensions import override

from .readonly_context import ReadonlyContext

from opentelemetry import context as context_api

if TYPE_CHECKING:
  from google.genai import types
  from pydantic import BaseModel

  from ..artifacts.base_artifact_service import ArtifactVersion
  from ..auth.auth_credential import AuthCredential
  from ..auth.auth_tool import AuthConfig
  from ..events.event import Event
  from ..events.event_actions import EventActions
  from ..events.ui_widget import UiWidget
  from ..memory.base_memory_service import SearchMemoryResponse
  from ..memory.memory_entry import MemoryEntry
  from ..sessions.session import Session
  from ..sessions.state import State
  from ..tools.tool_confirmation import ToolConfirmation
  from ..workflow._base_node import BaseNode
  from ..workflow._definitions import NodeLike
  from ..workflow._definitions import RouteValue
  from ..workflow._schedule_dynamic_node import ScheduleDynamicNode as ScheduleDynamicNodeInternal
  from .invocation_context import InvocationContext


# This is the signature for the function that schedules a dynamic node in the
# workflow.
# First argument is the Context.
# Second is the node to be scheduled,
# Third is the run_id of the node to be scheduled.
# Fourth is the node_input to be passed to the node.
# Fifth (optional) is the node_name to be used for the dynamic node. If not
# provided, the run_id will be used as the node_name.
# Make sure the node_name is unique across all nodes in the workflow.
# This node_name is also used to skip the node during resume if the node has
# already been executed.
#
# It returns a future that will resolve to the output of the node.
class ScheduleDynamicNode(Protocol):
  """Signature for the function that schedules a dynamic node."""

  def __call__(
      self,
      ctx: 'Context',
      node: Any,
      run_id: str,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
  ) -> asyncio.Future[Any]:
    ...


class _SessionProxy:
  """A proxy for the session that merges local events dynamically.

  Needed for nodes which expect their events to be immediately available in
  the session.

  Reason:
  - Workflow maintains an event queue.
  - Running nodes append events to that queue, workflow pops that queue and emit
    them back to client.
  - In case of a nested workflow, the function response event is emitted by
    client, which is then pushed & popped by the nested workflow.
  - The parent workflow gets the event and emits to its queue.
  - But before it can be yielded back to outer runner, the node inside the
    nested workflow is yielded control and it starts to run.
  - Since runner hasn't got the event yet, it hasn't appended to session.
  """

  def __init__(self, session: Session, local_events: list[Event]):
    # We bypass Pydantic initialization as we are a proxy.
    object.__setattr__(self, '_session', session)
    object.__setattr__(self, '_local_events', local_events)

  @property
  def actual_session(self) -> Session:
    return self._session

  # Need to implement manual attribute getter to play nice with Pydantic.
  def __getattribute__(self, name: str) -> Any:
    if name == 'events':
      session = object.__getattribute__(self, '_session')
      local_events = object.__getattribute__(self, '_local_events')
      session_events = session.events
      session_event_ids = {event.id for event in session_events}
      return session_events + [
          event for event in local_events if event.id not in session_event_ids
      ]

    if name in (
        '_session',
        '_local_events',
        'actual_session',
        '__class__',
        '__dict__',
    ):
      return object.__getattribute__(self, name)
    return getattr(self._session, name)

  def __setattr__(self, name: str, value: Any) -> None:
    if name in ('_session', '_local_events'):
      object.__setattr__(self, name, value)
    elif name in ('events'):
      raise AttributeError(f"Cannot set '{name}' on SessionProxy.")
    else:
      setattr(self._session, name, value)


class Context(ReadonlyContext):
  """The context within an agent run.

  When used in a workflow, additional fields are available:
  ``node_path``, ``run_id``, ``triggered_by``, ``in_nodes``,
  ``resume_inputs``, ``transfer_targets``, ``retry_count``,
  ``run_node()``, and ``get_next_child_run_id()``.
  """

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      event_actions: EventActions | None = None,
      function_call_id: str | None = None,
      tool_confirmation: ToolConfirmation | None = None,
      # Workflow-specific fields (optional)
      node_path: str = '',
      run_id: str = '',
      local_events: list[Event] | None = None,
      triggered_by: str = '',
      in_nodes: set[str] | None = None,
      resume_inputs: dict[str, Any] | None = None,
      schedule_dynamic_node: (
          ScheduleDynamicNode | None
      ) = None,  # TODO: remove after migrating to new Workflow
      schedule_dynamic_node_internal: ScheduleDynamicNodeInternal | None = None,
      node_rerun_on_resume: bool = True,
      transfer_targets: list[Any] | None = None,
      attempt_count: int = 1,
      output_for_ancestors: list[str] | None = None,
      event_author: str = '',
      state_schema: type[BaseModel] | None = None,
      otel_context: context_api.Context | None = None,
  ) -> None:
    """Initializes the Context.

    Args:
      invocation_context: The invocation context.
      event_actions: The event actions for state and artifact deltas.
      function_call_id: The function call id of the current tool call. Required
        for tool-specific methods like request_credential and
        request_confirmation.
      tool_confirmation: The tool confirmation of the current tool call.
      node_path: The path of the current node in the workflow graph.
      run_id: The execution ID of the current node.
      local_events: Local events for session proxy (workflow only).
      triggered_by: The name of the node that triggered the current node.
      in_nodes: Names of predecessor nodes.
      resume_inputs: Inputs for resuming node, keyed by interrupt id.
      schedule_dynamic_node: Function to schedule dynamic nodes.
      node_rerun_on_resume: Whether the node reruns on resume.
      transfer_targets: Valid transfer targets for the current node.
      retry_count: Number of times this node has been retried.
    """
    super().__init__(invocation_context)

    from ..events.event_actions import EventActions
    from ..sessions.state import State

    # TODO: clean up: group the below private fields with categories.

    self._event_actions = event_actions or EventActions()
    self._state = State(
        value=invocation_context.session.state,
        delta=self._event_actions.state_delta,
        schema=state_schema or invocation_context._state_schema,
    )
    self._function_call_id = function_call_id
    self._tool_confirmation = tool_confirmation

    # Workflow-specific fields
    self._node_path = node_path
    self._run_id = run_id
    self._triggered_by = triggered_by
    self._in_nodes = (
        frozenset(in_nodes) if in_nodes is not None else frozenset()
    )
    self._resume_inputs = resume_inputs or {}
    self.schedule_dynamic_node = schedule_dynamic_node
    self._schedule_dynamic_node_internal = schedule_dynamic_node_internal
    self._node_rerun_on_resume = node_rerun_on_resume
    self._child_run_counters: dict[str, int] = {}
    self._child_run_counter = 0
    self._local_events = local_events if local_events is not None else []
    self._transfer_targets = transfer_targets or []
    self._attempt_count = attempt_count
    self._output_delegated = False
    self._output_value: Any = None
    self._output_emitted: bool = False
    self._route_value: RouteValue | list[RouteValue] | None = None
    self._interrupt_ids: set[str] = set()
    self._event_author = event_author
    self._otel_context = otel_context or context_api.get_current()
    self._output_for_ancestors: list[str] = output_for_ancestors or []
    """Ancestor node paths whose output this node's output also represents.

    E.g. ['wf/parent@1', 'wf/grandparent@1'] means this node's output event
    is also considered the output for those ancestor paths.
    """
    self._error: Exception | None = None
    self._error_node_path: str = ''
    # Use a session proxy when local_events are provided (workflow mode).
    if local_events is not None:
      self._session_proxy = _SessionProxy(
          self._invocation_context.session, self._local_events
      )
    else:
      self._session_proxy = None

  @property
  def function_call_id(self) -> str | None:
    """The function call id of the current tool call."""
    return self._function_call_id

  @function_call_id.setter
  def function_call_id(self, value: str | None) -> None:
    """Sets the function call id of the current tool call."""
    self._function_call_id = value

  @property
  def tool_confirmation(self) -> ToolConfirmation | None:
    """The tool confirmation of the current tool call."""
    return self._tool_confirmation

  @tool_confirmation.setter
  def tool_confirmation(self, value: ToolConfirmation | None) -> None:
    """Sets the tool confirmation of the current tool call."""
    self._tool_confirmation = value

  @property
  @override
  def state(self) -> State:
    """The delta-aware state of the current session.

    For any state change, you can mutate this object directly,
    e.g. `ctx.state['foo'] = 'bar'`
    """
    return self._state

  @property
  def actions(self) -> EventActions:
    """The event actions for the current context."""
    return self._event_actions

  @property
  @override
  def session(self) -> Session:
    """Returns the current session for this invocation."""
    if self._session_proxy is not None:
      return self._session_proxy
    return self._invocation_context.session

  # ============================================================================
  # Workflow-specific properties and methods
  # ============================================================================

  @property
  def node_path(self) -> str:
    """Returns the path of the current node in the workflow graph."""
    return self._node_path

  @property
  def run_id(self) -> str:
    """Returns the execution ID of the current node."""
    return self._run_id

  @property
  def triggered_by(self) -> str:
    """Returns the name of the node that triggered the current node."""
    return self._triggered_by


  @property
  def attempt_count(self) -> int:
    """Returns the current attempt number (1-based)."""
    return self._attempt_count

  @property
  def in_nodes(self) -> frozenset[str]:
    """Returns names of nodes that are predecessors of the current node."""
    return self._in_nodes

  @property
  def resume_inputs(self) -> dict[str, Any]:
    """Returns inputs for resuming node, keyed by interrupt id."""
    return self._resume_inputs

  @property
  def error(self) -> Exception | None:
    """The exception raised by the node, if any."""
    return self._error

  @error.setter
  def error(self, value: Exception | None) -> None:
    self._error = value

  @property
  def error_node_path(self) -> str:
    """The path of the node that failed."""
    return self._error_node_path

  @error_node_path.setter
  def error_node_path(self, value: str) -> None:
    self._error_node_path = value

  @property
  def output(self) -> Any:
    """The node's result value. Source of truth for node output.

    Set once per run. Also set by the framework when the node
    yields Event(output=X) or yields a raw value. If the value was
    set via yield, the output Event is already enqueued. If set
    directly, the framework emits the output Event after _run_impl
    returns.

    Raises ValueError if:
    - Set a second time (at most one output per execution).
    - Set when interrupt_ids is non-empty (output and interrupt
      are mutually exclusive).
    """
    return self._output_value

  @output.setter
  def output(self, value: Any) -> None:
    if self._output_value is not None:
      raise ValueError(
          'Output already set. A node can produce at most one output.'
      )
    self._output_value = value

  @property
  def route(self) -> RouteValue | list[RouteValue] | None:
    """Routing value for conditional edges.

    Read by the orchestrator to decide which downstream edge to
    follow. Can be set independently of output.
    """
    return self._route_value

  @route.setter
  def route(self, value: RouteValue | list[RouteValue]) -> None:
    self._route_value = value

  @property
  def interrupt_ids(self) -> set[str]:
    """Interrupt IDs accumulated during this execution. Read-only.

    Set by the framework when the node yields an Event with
    long_running_tool_ids.
    """
    return set(self._interrupt_ids)

  @property
  def event_author(self) -> str:
    """Author name stamped on events emitted by this node.

    Set by the orchestrator to override the default (node name).
    For example, Workflow sets this to its own name so all child
    events appear under the workflow's author.

    Empty string means use the node's own name (default).
    """
    return self._event_author

  @event_author.setter
  def event_author(self, value: str) -> None:
    self._event_author = value

  @property
  def transfer_targets(self) -> list[Any]:
    """Returns the list of valid transfer targets for the current node."""
    return self._transfer_targets

  @property
  def otel_context(self) -> context_api.Context:
    """Returns OpenTelemetry context.

    The OpenTelemetry context holds the current OTel span and baggage.
    """
    return self._otel_context

  def get_invocation_context(self) -> InvocationContext:
    """Returns a copy of the invocation context with the proxy session."""
    ctx = self._invocation_context
    ctx_with_proxy = ctx.model_copy(
        update={
            'session': self.session,
        }
    )
    return ctx_with_proxy

  def get_next_child_run_id(
      self, node_name: str, *, is_static_name: bool = False
  ) -> str:
    """Generates the next deterministic child run ID.

    Args:
      node_name: The name of the child node.
      is_static_name: If True, derive the ID from the name alone (no
        counter). This produces the same ID regardless of call order,
        which is required for parallel ``run_node`` calls that may
        start in non-deterministic order.
    """
    if is_static_name:
      unique_string = f'{self._run_id}-{node_name}'
    else:
      self._child_run_counter += 1
      unique_string = f'{self._run_id}-{self._child_run_counter}-{node_name}'
    # TODO(swapnilag): use a better hash method.
    hashed_id = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:15]
    return f'{node_name}_{hashed_id}'

  async def run_node(
      self,
      node: NodeLike,
      node_input: Any = None,
      *,
      use_as_output: bool = False,
      run_id: str | None = None,
      sub_branch: str | None = None,
  ) -> Any:
    """Executes a node dynamically.

    This method allows a node within a workflow to trigger the run of
    another node (or a callable that can be built into a node) and
    asynchronously wait for its result. The dynamically executed node becomes
    a child run of the current node in the workflow.

    IMPORTANT: Always ``await`` this method directly. Wrapping it in
    ``asyncio.create_task()`` means the task runs unsupervised — errors
    are silently swallowed and the task is not cancelled if the parent
    node is interrupted (e.g. via HITL).

    Args:
      node: The node to be executed. This can be a BaseNode instance or a
        callable that can be built into a node.
      node_input: The input data to be passed to the dynamically executed node.
        Defaults to None.
      use_as_output: If True, the dynamic node's output is used as the
        calling node's output. The calling node's own output event is
        suppressed to avoid duplication.
      run_id: An optional custom run ID for the dynamic node execution.
        If not provided, a default run ID is generated. Useful for
        correlating events across runs.

    Returns:
      The output of the dynamically executed node, once it finishes executing.
    """

    if not self._node_rerun_on_resume:
      raise ValueError(
          'A node must have rerun_on_resume=True. Reason is that dynamically'
          ' scheduled nodes might be interrupted, and the workflow'
          ' wakes-up/re-runs the parent node, so it can get the child node'
          ' response.'
      )

    from ..workflow.utils._workflow_graph_utils import build_node  # pylint: disable=g-import-not-at-top

    built_node = build_node(node)

    # Prefer the internal scheduler (new Workflow architecture) which
    # returns child Context. Fall back to the legacy scheduler.
    if self._schedule_dynamic_node_internal:
      from ..workflow._errors import NodeInterruptedError

      # Output delegation: once set, the calling node's own output
      # events are suppressed — the child's output (annotated with
      # output_for) becomes the calling node's output.
      if use_as_output:
        if self._output_delegated:
          raise ValueError(
              f'Node {self.node_path} already has a use_as_output delegate.'
          )
        self._output_delegated = True

      if run_id:
        if run_id.isdigit():
          raise ValueError(
              f'Explicit run_id "{run_id}" for node "{built_node.name}" must'
              ' contain non-numeric characters to prevent collision with'
              ' auto-generated IDs.'
          )
      else:
        self._child_run_counters[built_node.name] = (
            self._child_run_counters.get(built_node.name, 0) + 1
        )
        run_id = str(self._child_run_counters[built_node.name])

      child_ctx = await self._schedule_dynamic_node_internal(
          self,
          built_node,
          node_input,
          node_name=built_node.name,
          use_as_output=use_as_output,
          run_id=run_id,
          sub_branch=sub_branch,
      )
      if child_ctx.error:
        from ..workflow._errors import DynamicNodeFailError

        raise DynamicNodeFailError(
            f'Dynamic node {built_node.name} failed',
            error=child_ctx.error,
            error_node_path=child_ctx.error_node_path,
        )
      if child_ctx.interrupt_ids:
        # Propagate child's interrupt_ids to this node's ctx
        # so NodeRunner sees them after catching the error.
        self._interrupt_ids.update(child_ctx.interrupt_ids)
        raise NodeInterruptedError()
      return child_ctx.output

    # Fall back to the early-2.0 scheduler (used by _workflow.py,
    # LoopAgent, ParallelAgent, SequentialAgent).
    if self.schedule_dynamic_node:
      run_id = self.get_next_child_run_id(built_node.name, is_static_name=False)
      return await self.schedule_dynamic_node(
          self,
          built_node,
          run_id,
          node_input,
          use_as_output=use_as_output,
      )

    # No orchestrator scheduler — run the node directly.
    result = await self._run_node_via_runner(
        built_node,
        node_input,
        use_as_output=use_as_output,
    )
    return result.output

  async def _run_node_internal(
      self,
      node: NodeLike,
      node_input: Any = None,
      *,
      run_id: str | None = None,
  ) -> Context:
    """Internal: run a node and return its child Context.

    Unlike run_node() which returns just the output, this returns the
    child Context for orchestrators that need output, route, and
    interrupt info.
    """
    from ..workflow.utils._workflow_graph_utils import build_node

    built_node = build_node(node)
    if self._schedule_dynamic_node_internal:
      return await self._schedule_dynamic_node_internal(
          self,
          node,
          node_input,
          node_name=node.name,
          run_id=run_id or '1',
      )

    return await self._run_node_via_runner(
        built_node, node_input, run_id=run_id
    )

  async def _run_node_via_runner(
      self,
      node: BaseNode,
      node_input: Any,
      *,
      use_as_output: bool = False,
      run_id: str | None = None,
  ) -> Context:
    """Run a node directly via NodeRunner without an orchestrator."""
    from ..workflow._node_runner_class import NodeRunner

    runner = NodeRunner(
        node=node,
        parent_ctx=self,
        run_id=run_id,
        additional_output_for_ancestor=(
            self.node_path if use_as_output else None
        ),
    )
    return await runner.run(node_input=node_input)

  # ============================================================================
  # Artifact methods
  # ============================================================================

  async def load_artifact(
      self, filename: str, version: int | None = None
  ) -> types.Part | None:
    """Loads an artifact attached to the current session.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.load_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def save_artifact(
      self,
      filename: str,
      artifact: types.Part,
      custom_metadata: dict[str, Any] | None = None,
  ) -> int:
    """Saves an artifact and records it as delta for the current session.

    Args:
      filename: The filename of the artifact.
      artifact: The artifact to save.
      custom_metadata: Custom metadata to associate with the artifact.

    Returns:
     The version of the artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    version = await self._invocation_context.artifact_service.save_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        artifact=artifact,
        custom_metadata=custom_metadata,
    )
    self._event_actions.artifact_delta[filename] = version
    return version

  async def get_artifact_version(
      self, filename: str, version: int | None = None
  ) -> ArtifactVersion | None:
    """Gets artifact version info.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact version info.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.get_artifact_version(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def list_artifacts(self) -> list[str]:
    """Lists the filenames of the artifacts attached to the current session."""
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.list_artifact_keys(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
    )

  # ============================================================================
  # Credential methods
  # ============================================================================

  async def save_credential(self, auth_config: AuthConfig) -> None:
    """Saves a credential to the credential service.

    Args:
      auth_config: The authentication configuration containing the credential.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError('Credential service is not initialized.')
    await self._invocation_context.credential_service.save_credential(
        auth_config, self
    )

  async def load_credential(
      self, auth_config: AuthConfig
  ) -> AuthCredential | None:
    """Loads a credential from the credential service.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The loaded credential, or None if not found.
    """
    if self._invocation_context.credential_service is None:
      raise ValueError('Credential service is not initialized.')
    return await self._invocation_context.credential_service.load_credential(
        auth_config, self
    )

  def get_auth_response(self, auth_config: AuthConfig) -> AuthCredential | None:
    """Gets the auth response credential from session state.

    This method retrieves an authentication credential that was previously
    stored in session state after a user completed an OAuth flow or other
    authentication process.

    Args:
      auth_config: The authentication configuration for the credential.

    Returns:
      The auth credential from the auth response, or None if not found.
    """
    from ..auth.auth_handler import AuthHandler

    return AuthHandler(auth_config).get_auth_response(self.state)

  def request_credential(self, auth_config: AuthConfig) -> None:
    """Requests a credential for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set. For callback contexts, use save_credential/load_credential instead.

    Args:
      auth_config: The authentication configuration for the credential.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..auth.auth_handler import AuthHandler

    if not self.function_call_id:
      raise ValueError(
          'request_credential requires function_call_id. '
          'This method can only be used in a tool context, not a callback '
          'context. Consider using save_credential/load_credential instead.'
      )
    self._event_actions.requested_auth_configs[self.function_call_id] = (
        AuthHandler(auth_config).generate_auth_request()
    )

  # ============================================================================
  # Tool methods
  # ============================================================================

  def request_confirmation(
      self,
      *,
      hint: str | None = None,
      payload: Any | None = None,
  ) -> None:
    """Requests confirmation for the current tool call.

    This method can only be called in a tool context where function_call_id
    is set.

    Args:
      hint: A hint to the user on how to confirm the tool call.
      payload: The payload used to confirm the tool call.

    Raises:
      ValueError: If function_call_id is not set.
    """
    from ..tools.tool_confirmation import ToolConfirmation

    if not self.function_call_id:
      raise ValueError(
          'request_confirmation requires function_call_id. '
          'This method can only be used in a tool context.'
      )
    self._event_actions.requested_tool_confirmations[self.function_call_id] = (
        ToolConfirmation(
            hint=hint,
            payload=payload,
        )
    )

  # ============================================================================
  # Memory methods
  # ============================================================================

  async def add_session_to_memory(self) -> None:
    """Triggers memory generation for the current session.

    This method saves the current session's events to the memory service,
    enabling the agent to recall information from past interactions.

    Raises:
      ValueError: If memory service is not available.

    Example:
      ```python
      async def my_after_agent_callback(ctx: Context):
          # Save conversation to memory at the end of each interaction
          await ctx.add_session_to_memory()
      ```
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          'Cannot add session to memory: memory service is not available.'
      )
    await self._invocation_context.memory_service.add_session_to_memory(
        self._invocation_context.session
    )

  async def add_events_to_memory(
      self,
      *,
      events: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds an explicit list of events to the memory service.

    Uses this callback's current session identifiers as memory scope.

    Args:
      events: Explicit events to add to memory.
      custom_metadata: Optional metadata forwarded to the configured memory
        service. Supported keys are implementation-specific.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError(
          'Cannot add events to memory: memory service is not available.'
      )
    await self._invocation_context.memory_service.add_events_to_memory(
        app_name=self._invocation_context.session.app_name,
        user_id=self._invocation_context.session.user_id,
        session_id=self._invocation_context.session.id,
        events=events,
        custom_metadata=custom_metadata,
    )

  async def add_memory(
      self,
      *,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    """Adds explicit memory items directly to the memory service.

    Uses this callback's current session identifiers as memory scope.

    Args:
      memories: Explicit memory items to add.
      custom_metadata: Optional metadata forwarded to the configured memory
        service. Supported keys are implementation-specific.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError('Cannot add memory: memory service is not available.')
    await self._invocation_context.memory_service.add_memory(
        app_name=self._invocation_context.session.app_name,
        user_id=self._invocation_context.session.user_id,
        memories=memories,
        custom_metadata=custom_metadata,
    )

  async def search_memory(self, query: str) -> SearchMemoryResponse:
    """Searches the memory of the current user.

    Args:
      query: The search query.

    Returns:
      The search results from the memory service.

    Raises:
      ValueError: If memory service is not available.
    """
    if self._invocation_context.memory_service is None:
      raise ValueError('Memory service is not available.')
    return await self._invocation_context.memory_service.search_memory(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        query=query,
    )

  # ============================================================================
  # UI Widget methods
  # ============================================================================

  def render_ui_widget(self, ui_widget: UiWidget) -> None:
    """Adds a UI widget to the current event's actions for the UI to render.

    UI widgets provide rendering payload/metadata that the UI Host uses to
    display rich interactive components (e.g., MCP App iframes) alongside agent
    responses.

    Args:
      ui_widget: A ``UiWidget`` instance.
    """
    if self._event_actions.render_ui_widgets is None:
      self._event_actions.render_ui_widgets = []

    for existing_widget in self._event_actions.render_ui_widgets:
      if existing_widget.id == ui_widget.id:
        raise ValueError(
            f"UI widget with ID '{ui_widget.id}' already exists in the current"
            ' event actions.'
        )

    self._event_actions.render_ui_widgets.append(ui_widget)
