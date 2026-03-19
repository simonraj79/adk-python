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

import collections.abc
import inspect
import logging
import typing
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic import TypeAdapter
from typing_extensions import override

from ..events.event import Event
from ..events.request_input import RequestInput
from ._base_node import BaseNode
from ._retry_config import RetryConfig

logger = logging.getLogger('google_adk.' + __name__)


async def _sync_to_async_gen(sync_gen):
  """Wraps a synchronous generator as an async generator."""
  for item in sync_gen:
    yield item


if TYPE_CHECKING:
  from ..agents.context import Context

# Output types that are framework control-flow items, not data schemas.
_PASSTHROUGH_OUTPUT_TYPES = (types.Content, Event, RequestInput)

# Generator origins used for unwrapping yield types.
_GENERATOR_ORIGINS = (
    collections.abc.Generator,
    collections.abc.AsyncGenerator,
)


def _content_to_str(
    content: types.Content, func_name: str, param_name: str
) -> str:
  """Extracts text from a Content object, warning on non-text parts."""
  texts = []
  for part in content.parts or []:
    if part.text is not None:
      texts.append(part.text)
    elif part.inline_data or part.file_data or part.executable_code:
      logger.warning(
          'Parameter "%s" of function "%s" expects str but received'
          ' Content with non-text parts (e.g. inline_data, file_data).'
          ' Non-text parts are dropped during auto-conversion.',
          param_name,
          func_name,
      )
  return ''.join(texts)


def _expects_str(annotated_type: Any) -> bool:
  """Returns True if the annotation is or contains ``str``."""
  if annotated_type is str:
    return True
  if typing.get_origin(annotated_type) is typing.Union:
    return any(_expects_str(a) for a in typing.get_args(annotated_type))
  return False


class FunctionNode(BaseNode):
  """A node that wraps a Python sync/async function or generator.

  Type coercions applied to function parameters (via ``TypeAdapter``):
    - ``dict`` → ``BaseModel`` when the annotation is a Pydantic model.
    - ``list[dict]`` → ``list[BaseModel]``, ``dict[K, dict]`` →
      ``dict[K, BaseModel]``, etc.
    - ``types.Content`` → ``str`` when the annotation expects ``str``
      (including ``Optional[str]`` / ``Union[str, ...]``).
    - All other values are validated/coerced by Pydantic's ``TypeAdapter``.
  """

  # Private attributes (won't be serialized)
  _func: Callable[..., Any] = PrivateAttr()
  _sig: inspect.Signature = PrivateAttr()
  _type_hints: dict[str, Any] = PrivateAttr()

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      name: str | None = None,
      rerun_on_resume: bool = False,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    """Initializes FunctionNode.

    Args:
      func: A sync/async function or sync/async generator function that forms
        the node's logic. It can accept 'ctx: Context' and 'node_input: Any' as
        arguments, depending on its signature. If the function is not a
        generator, its return value will be wrapped in an Event, unless the
        return value is None.
      name: The name of the node. If None, it defaults to func.__name__.
      rerun_on_resume: If True, the node will be rerun after being interrupted
        and resumed. If False, the node will be marked as completed and the
        resuming input will be treated as the node's output.
      retry_config: If provided, the node will be retried on failure based on
        this configuration.
      timeout: Maximum time in seconds for this node to complete.
    """

    if not callable(func):
      raise TypeError('Function must be callable.')

    inferred_name = name or getattr(func, '__name__', None)
    if not inferred_name:
      raise ValueError(
          'FunctionNode must have a name. If the wrapped callable does not'
          " have a '__name__' attribute, please provide a name explicitly."
      )

    super().__init__(
        name=inferred_name,
        description=inspect.getdoc(func) or '',
        rerun_on_resume=rerun_on_resume,
        retry_config=retry_config,
        timeout=timeout,
    )

    sig = inspect.signature(func)
    try:
      type_hints = typing.get_type_hints(func)
    except Exception:
      type_hints = {}

    # Infer output_schema from the return type hint.
    # For generators (Generator[T, ...] / AsyncGenerator[T, ...]),
    # extract the yield type T as the schema.
    return_hint = type_hints.get('return')
    schema_hint = return_hint

    # Unwrap Generator[T, ...] / AsyncGenerator[T, ...] to T.
    if return_hint is not None:
      origin = typing.get_origin(return_hint)
      if origin in _GENERATOR_ORIGINS:
        args = typing.get_args(return_hint)
        schema_hint = args[0] if args else None

    if (
        schema_hint is not None
        and inspect.isclass(schema_hint)
        and issubclass(schema_hint, BaseModel)
        and not issubclass(schema_hint, _PASSTHROUGH_OUTPUT_TYPES)
    ):
      self.output_schema = schema_hint

    # Infer input_schema from node_input type hint.
    input_hint = type_hints.get('node_input')
    if (
        input_hint is not None
        and inspect.isclass(input_hint)
        and issubclass(input_hint, BaseModel)
    ):
      self.input_schema = input_hint

    # Set private attributes
    self._func = func
    self._sig = sig
    self._type_hints = type_hints

  def _to_event(self, ctx: Context, data: Any) -> Event | None:
    """Converts a function return value to an Event.

    Pass-through types (returned as-is): Event, RequestInput.
    None is returned as None (caller skips it) unless there are pending
    state changes.
    All other values are wrapped in an Event(output=...).

    State changes made via ``ctx.state`` during function execution are
    captured in ``ctx.actions.state_delta`` and attached to the emitted
    event so that they are persisted by the session service.
    """
    state_delta = (
        dict(ctx.actions.state_delta) if ctx.actions.state_delta else None
    )

    if data is None:
      if state_delta:
        return Event(state=state_delta)
      return None

    if isinstance(data, Event):
      if data.output is not None:
        data.output = self._validate_output_data(data.output)
      if state_delta:
        data.actions.state_delta.update(state_delta)
      return data
    if isinstance(data, RequestInput):
      return data
    if isinstance(data, types.Content):
      return Event(
          content=data,
          state=state_delta,
      )

    if isinstance(data, BaseModel):
      data = data.model_dump()

    data = self._validate_output_data(data)

    return Event(
        output=data,
        state=state_delta,
    )

  def _coerce_param(
      self,
      param_name: str,
      value: Any,
      annotated_type: Any,
  ) -> Any:
    """Coerces a parameter value to match its type annotation.

    Uses Pydantic's ``TypeAdapter`` for validation and coercion (handles
    ``dict`` → ``BaseModel``, ``list[dict]`` → ``list[BaseModel]``, unions,
    primitives, etc.).  A special case converts ``types.Content`` → ``str``
    when the annotation expects ``str``.

    Args:
      param_name: The name of the parameter (for error messages).
      value: The value to coerce.
      annotated_type: The type annotation of the parameter.

    Returns:
      The coerced value.
    """
    # Content → str auto-conversion (e.g. user content from START node).
    if isinstance(value, types.Content) and _expects_str(annotated_type):
      return _content_to_str(value, self.name, param_name)
    return TypeAdapter(annotated_type).validate_python(value)

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ):
    copied = super().model_copy(update=update, deep=deep)
    if not update or 'name' not in update:
      return copied

    # If the wrapped function is a bound method of a Node, we need to clone
    # the Node and re-bind the function to the new instance.
    # This is needed if the function is referring to params like 'name' from the "self" reference.
    # Like Workflow or LLM use that name for event node_paths or retreving session events.
    func = self._func
    if inspect.ismethod(func) and isinstance(
        getattr(func, '__self__', None), BaseNode
    ):
      method_self = getattr(func, '__self__')
      method_name = getattr(func, '__name__')

      # Pass the name update to the cloned agent instance if it's being passed
      # to the FunctionNode (case for parallel workers).
      agent_update = {
          'name': update['name'],
      }

      new_obj = method_self.model_copy(update=agent_update)
      copied._func = getattr(new_obj, method_name)
    else:
      copied._func = func

    copied._sig = self._sig
    copied._type_hints = self._type_hints
    return copied

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    kwargs: dict[str, Any] = {}
    for param_name, param in self._sig.parameters.items():
      if param_name == 'ctx':
        kwargs['ctx'] = ctx
        continue
      elif param_name == 'node_input':
        if param_name in self._type_hints:
          node_input = self._coerce_param(
              param_name,
              node_input,
              self._type_hints[param_name],
          )
        kwargs[param_name] = node_input
        continue

      # Parameters other than ctx and node_input are sourced from state.
      if param_name in ctx.state:
        value = ctx.state[param_name]
        if param_name in self._type_hints:
          value = self._coerce_param(
              param_name,
              value,
              self._type_hints[param_name],
          )
        kwargs[param_name] = value
      elif param.default is not inspect.Parameter.empty:
        kwargs[param_name] = param.default
      else:
        raise ValueError(
            f'Missing value for parameter "{param_name}" of function'
            f' "{self.name}". It was not found in state and has no default'
            ' value.'
        )

    if inspect.isasyncgenfunction(self._func):
      items = self._func(**kwargs)
    elif inspect.isgeneratorfunction(self._func):
      items = _sync_to_async_gen(self._func(**kwargs))
    else:
      items = None

    if items is not None:
      async for item in items:
        event = self._to_event(ctx, item)
        if event is not None:
          yield event
    else:
      if inspect.iscoroutinefunction(self._func):
        result = await self._func(**kwargs)
      else:  # Sync function
        result = self._func(**kwargs)

      event = self._to_event(ctx, result)
      if event is not None:
        yield event
