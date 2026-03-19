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

import logging
from typing import Any
from typing import AsyncGenerator

from typing_extensions import override

from ..agents.context import Context
from ..events.event import Event
from ._base_node import BaseNode

logger = logging.getLogger('google_adk.' + __name__)


class JoinNode(BaseNode):
  """A node that waits for all specified predecessors to trigger it before outputting."""

  wait_for_output: bool = True

  def _get_state_key(self, node_path: str) -> str:
    return f'{node_path}_join_state'

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if not ctx.in_nodes:
      raise ValueError(
          f'JoinNode {self.name} has no predecessors defined in graph.'
      )

    state_key = self._get_state_key(ctx.node_path)
    join_state = (ctx.state.get(state_key) or {}).copy()

    triggering_node = ctx.triggered_by
    if not triggering_node:
      logger.warning(
          'JoinNode %s received trigger from node with no name. Ignoring.',
          self.name,
      )
      return

    if triggering_node not in ctx.in_nodes:
      logger.warning(
          'JoinNode %s received trigger from unexpected node %s. Ignoring.',
          self.name,
          triggering_node,
      )
      return

    # Recording the output from previous node.
    join_state[triggering_node] = node_input

    if set(join_state.keys()) == ctx.in_nodes:
      yield Event(
          output=dict(join_state),
          # Clear state for future runs
          state={state_key: None},
      )
    else:
      # Update state with recorded outputs from previous nodes and wait for
      # more triggers.
      yield Event(
          state={state_key: join_state},
      )
