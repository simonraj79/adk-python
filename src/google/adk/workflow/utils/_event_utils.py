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

"""Utilities for enriching workflow events with context metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext


def enrich_event(
    event: Event,
    ctx: InvocationContext,
    *,
    author: str = '',
    node_path: str = '',
    run_id: str = '',
    branch: bool = False,
) -> Event:
  """Fills in standard metadata fields on an event if not already set.

  Fields that are already set on the event are not overwritten.

  Only ``invocation_id`` and ``node_path`` are filled from ``ctx``
  automatically.  Other fields (``author``, ``run_id``, ``branch``)
  are only set when explicitly requested — they have no safe default
  derivable from ``ctx`` alone (e.g. nested workflows need the parent's
  name, not ``ctx.agent.name``).

  Args:
    event: The event to enrich (mutated in place).
    ctx: The invocation context providing default values.
    author: If provided, sets the event author when missing.
    node_path: Override for the node path. Falls back to ``ctx.node_path``.
    run_id: If provided, sets the run ID when missing.
    branch: If True, sets the event branch from ``ctx.branch`` when missing.

  Returns:
    The same event instance.
  """
  if not event.invocation_id:
    event.invocation_id = ctx.invocation_id
  if not event.author and author:
    event.author = author
  if not event.node_info.path:
    event.node_info.path = node_path or ctx.node_path or ''
  if not event.node_info.run_id and run_id:
    event.node_info.run_id = run_id
  if branch and not event.branch and ctx.branch:
    event.branch = ctx.branch
  return event
