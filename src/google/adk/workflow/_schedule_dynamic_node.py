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

"""Internal protocol for scheduling dynamic nodes with full result."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..agents.context import Context


class ScheduleDynamicNode(Protocol):
  """Schedules a dynamic node and returns its child Context."""

  def __call__(
      self,
      ctx: Context,
      node: Any,
      run_id: str,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
  ) -> Awaitable[Context]:
    ...
