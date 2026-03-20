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
from typing import final
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import TypeAdapter

from ..utils._schema_utils import SchemaType
from ._retry_config import RetryConfig

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..events.event import Event


class BaseNode(BaseModel):
  """A base class for all nodes in the workflow graph."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(...)

  description: str = ''
  """A human-readable description of what this node does."""

  rerun_on_resume: bool = False
  """If True, the node will be rerun after being interrupted and resumed.
  If False, the node will be marked as completed and the resuming input will be
  treated as the node's output."""

  wait_for_output: bool = False
  """If True, the node only transitions to COMPLETED when it yields output.

  When a node with ``wait_for_output=True`` finishes without yielding any
  ``Event`` with data set, it moves to WAITING state instead of COMPLETED,
  and downstream nodes are not triggered. The node can then be re-triggered
  by upstream predecessors. This is useful for nodes like ``JoinNode`` that
  may run multiple times (once per predecessor) before producing a final
  output.
  """

  retry_config: RetryConfig | None = None
  """Configuration for retrying the node."""

  timeout: float | None = None
  """Maximum time in seconds for this node to complete.

  If the node does not finish within this duration, it is cancelled and
  treated as a failure (raising ``NodeTimeoutError``).  This integrates
  with ``retry_config`` — a timed-out node can be retried if retries
  are configured.

  ``None`` means no timeout (the node runs until completion).
  """

  input_schema: SchemaType | None = None
  """Schema to validate and coerce node input data.

  Supports all ``SchemaType`` variants. Validation uses ``TypeAdapter``
  and runs centrally in the node runner before ``node.run()`` is called.

  ``None`` means no input validation (the default).
  """

  output_schema: SchemaType | None = None
  """Schema to validate and coerce node output data.

  Supports all ``SchemaType`` variants (Pydantic ``BaseModel`` subclass,
  generic aliases like ``list[str]``, raw ``dict`` schemas, etc.).

  When set to a ``BaseModel`` subclass, the node's output data is validated:
    - dict → ``output_schema.model_validate(data).model_dump()``
    - BaseModel instance → ``data.model_dump()`` (already converted)

  ``None`` means no output validation (the default).
  """

  def _validate_schema(self, data: Any, schema: Any) -> Any:
    """Validates data against a schema using ``TypeAdapter``.

    Handles BaseModel, list[BaseModel], primitive types, unions, and
    generic aliases.  Descriptive schemas (``types.Schema``, raw
    ``dict``) are skipped.  Any BaseModel instances in the validated
    result are converted to dicts via ``model_dump()`` to keep
    ``Event.output`` JSON-serializable.
    """
    if data is None or schema is None:
      return data

    if isinstance(schema, (dict, types.Schema)):
      return data

    validated = TypeAdapter(schema).validate_python(data)
    return self._to_serializable(validated)

  def _validate_input_data(self, data: Any) -> Any:
    """Validates data against input_schema if set."""
    return self._validate_schema(data, self.input_schema)

  def _validate_output_data(self, data: Any) -> Any:
    """Validates data against output_schema if set."""
    return self._validate_schema(data, self.output_schema)

  @staticmethod
  def _to_serializable(data: Any) -> Any:
    """Converts BaseModel instances to dicts recursively."""
    if isinstance(data, BaseModel):
      return data.model_dump()
    if isinstance(data, list):
      return [BaseNode._to_serializable(item) for item in data]
    if isinstance(data, dict):
      return {k: BaseNode._to_serializable(v) for k, v in data.items()}
    return data

  @final
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Event, None]:
    """Public entry point. Calls _run_impl, normalizes yields to Event.

    Normalization rules:
    - None -> skipped
    - Event -> pass through
    - RequestInput -> convert to interrupt Event
    - Any other value -> Event(output=value)
    """
    from ..events.event import Event
    from ..events.request_input import RequestInput

    async for item in self._run_impl(ctx=ctx, node_input=node_input):
      if item is None:
        continue
      if isinstance(item, Event):
        yield item
      elif isinstance(item, RequestInput):
        from .utils._workflow_hitl_utils import create_request_input_event

        yield create_request_input_event(item)
      else:
        yield Event(output=item)

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Override point for node execution logic.

    Yields any of: Event, RequestInput, raw data, or None.
    BaseNode.run() normalizes all yields to Event before the caller
    sees them.
    """
    raise NotImplementedError(
        f'_run_impl for {type(self).__name__} is not implemented.'
    )
    yield  # AsyncGenerator requires at least one yield statement


START = BaseNode(name='__START__')
"""Sentinel node marking the entry point of a workflow graph.

START is never executed — ``Workflow._seed_start_triggers`` bypasses it
and seeds triggers for its successors directly.
"""
