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

from typing import Any
from typing import AsyncGenerator

from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.context import Context
from ..events.event import Event
from ._base_node import BaseNode


class AgentNode(BaseNode):
  """A node that wraps a BaseAgent."""

  agent: BaseAgent = Field(...)

  @model_validator(mode='before')
  @classmethod
  def _set_name(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if data.get('name') is None and 'agent' in data:
        data['name'] = getattr(data['agent'], 'name', '')
    return data

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> Any:
    """Overrides model_copy to propagate name updates to the agent."""
    copied = super().model_copy(update=update, deep=deep)
    if update and 'name' in update:
      # If the node name is updated (e.g., by ParallelWorker), also update
      # the agent's name so its events carry the correct researcher name.
      copied.agent = copied.agent.clone(update={'name': update['name']})
    return copied

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Runs the agent as a node."""
    # For now BaseAgent event's content is not recorded as node output.
    # Developers should override this method to define their own node output
    # in their own agent classes.
    async for event in self.agent.run_async(
        parent_context=ctx.get_invocation_context()
    ):
      # Convert AdkEvent to WorkflowEvent to support node_path
      if not isinstance(event, Event):
        event = Event(**event.model_dump())
      if not event.node_info.path and event.author == self.agent.name:
        event.node_info.path = ctx.node_path
      yield event
