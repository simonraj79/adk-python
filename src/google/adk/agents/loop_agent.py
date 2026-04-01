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

"""Loop agent implementation using workflow graph."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import logging
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional

from pydantic import Field
from typing_extensions import override

from ..events.event import Event
from ..features import experimental
from ..features import FeatureName
from ..workflow import BaseNode
from ..workflow import START
from ..workflow import Workflow
from ..workflow.utils._workflow_graph_utils import build_node
from .base_agent_config import BaseAgentConfig
from .context import Context
from .loop_agent_config import LoopAgentConfig

logger = logging.getLogger('google_adk.' + __name__)
LOOP_COUNT_KEY = 'loop_count'
_DEFAULT_ROUTE = 'next'


class _DefaultRouteNode(BaseNode):
  """A node wrapper that provides a default route if none is set.

  If no escalation occurred, this wrapper yields the default_route.
  """

  inner_node: BaseNode = Field(...)
  name: str = Field(default='')

  def model_post_init(self, context: Any) -> None:
    self.name = self.inner_node.name

  @override
  async def run(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Event, None]:
    escalated = False
    async for event in self.inner_node.run(ctx=ctx, node_input=node_input):
      yield event

      # Only consider events strictly from the direct child agent
      # to avoid grandchild escalation breaking the loop.
      if event.node_info.path == ctx.node_path and event.actions.escalate:
        escalated = True

    if escalated:
      return
    # If no escalation occurred, continue the loop.
    yield Event(route=_DEFAULT_ROUTE)


class LoopAgent(Workflow):
  """A shell agent that runs its sub-agents in a loop using workflow graph."""

  config_type: ClassVar[type[BaseAgentConfig]] = LoopAgentConfig
  """The config type for this agent."""

  max_iterations: Optional[int] = None
  """The maximum number of iterations to run the loop agent."""

  @property
  def _loop_count_key(self) -> str:
    return f'{self.name}_{LOOP_COUNT_KEY}'

  async def _increment_loop_count(
      self,
      ctx: Context,
  ) -> AsyncGenerator[Event, None]:
    """Increments the loop count.

    Args:
      ctx: The workflow context.

    Yields:
      An `Event` updating the loop count in the context state.
      An `Event` with a 'continue_loop' route if the loop should continue,
      or an `Event` clearing the loop count from the state otherwise.
    """
    loop_count = ctx.state.get(self._loop_count_key, 0) + 1
    yield Event(state={self._loop_count_key: loop_count})

    should_continue = not (
        self.max_iterations and loop_count >= self.max_iterations
    )
    if should_continue:
      yield Event(route='continue_loop')
    else:
      yield Event(state={self._loop_count_key: None})

  @override
  def model_post_init(self, context: Any) -> None:
    if self.sub_agents:
      if self.graph is not None or self.edges:
        raise ValueError(
            'LoopAgent constructs its graph internally and does not'
            " accept 'graph' or 'edges' arguments."
        )

      # Wrap sub-agents to handle escalate action
      wrappers = []
      for agent in self.sub_agents:
        inner_node = build_node(agent)
        wrappers.append(_DefaultRouteNode(inner_node=inner_node))

      #  START -> agent1 -> agent2 -> ... -> agentN -> increment -> agent1
      self.edges.append((START, wrappers[0]))
      for i in range(len(wrappers) - 1):
        self.edges.append((wrappers[i], {_DEFAULT_ROUTE: wrappers[i + 1]}))

      # Store the bound method in a local variable so both edges share the
      # same object identity.
      increment_fn = self._increment_loop_count
      self.edges.extend([
          (wrappers[-1], {_DEFAULT_ROUTE: increment_fn}),
          (increment_fn, {'continue_loop': wrappers[0]}),
      ])

    super().model_post_init(context)

  @override
  async def _run_live_impl(self, ctx: Context) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('This is not supported yet for LoopAgent.')
    yield  # AsyncGenerator requires having at least one yield statement

  @override
  @classmethod
  @experimental(FeatureName.AGENT_CONFIG)
  def _parse_config(
      cls: type[LoopAgent],
      config: LoopAgentConfig,
      config_abs_path: str,
      kwargs: Dict[str, Any],
  ) -> Dict[str, Any]:
    if config.max_iterations:
      kwargs['max_iterations'] = config.max_iterations
    return kwargs
