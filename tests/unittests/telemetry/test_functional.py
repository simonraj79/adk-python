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

import gc
import sys

from google.adk.agents import base_agent
from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.telemetry import tracing
from google.adk.tools import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.genai.types import Part
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner


@pytest.fixture
def test_model() -> BaseLlm:
  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name='some_tool', args={}),
          Part.from_text(text='text response'),
      ]
  )
  return mock_model


@pytest.fixture
def test_agent(test_model: BaseLlm) -> Agent:
  def some_tool():
    pass

  root_agent = Agent(
      name='some_root_agent',
      model=test_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )
  return root_agent


@pytest.fixture
async def test_runner(test_agent: Agent) -> TestInMemoryRunner:
  runner = TestInMemoryRunner(test_agent)
  return runner


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  tracer_provider = TracerProvider()
  span_exporter = InMemorySpanExporter()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  def do_replace(tracer):
    monkeypatch.setattr(
        tracer, 'start_as_current_span', real_tracer.start_as_current_span
    )

  do_replace(tracing.tracer)
  do_replace(base_agent.tracer)

  return span_exporter


@pytest.mark.asyncio
async def test_tracer_start_as_current_span(
    test_runner: TestInMemoryRunner,
    span_exporter: InMemorySpanExporter,
):
  """Test creation of multiple spans in an E2E runner invocation.

  Additionally tests if each async generator invoked is wrapped in Aclosing.
  This is necessary because instrumentation utilizes contextvars, which ran into "ContextVar was created in a different Context" errors,
  when a given coroutine gets indeterminately suspended.
  """
  firstiter, finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro):
    nonlocal firstiter
    # Skip check for:
    # - tracing context managers (use_inference_span, etc.)
    # - workflow-internal generators: the workflow engine uses many
    #   small async generators (_checkpoint_agent_state,
    #   _handle_adk_event, _execute_node, etc.) that run within a
    #   single task and don't hold cross-context state. These are
    #   identified by their source file path containing 'workflow'.
    # - run_async: called internally by Workflow.run() for the
    #   coordinator _SingleLlmAgent; already wrapped at the
    #   top-level runner invocation.
    # - run_node_impl: LlmAgent.run_node_impl lives in agents/
    #   (outside workflow/) but is workflow-internal; it wraps
    #   its inner delegation with Aclosing.
    # - call_llm, execute_tools: live in agents/llm/ (outside
    #   workflow/) but are workflow-internal node functions.
    # - _run_node_async, _consume_event_queue: live in runners.py
    #   but are part of the new workflow node runtime path.
    # - start_as_current_node_span: @asynccontextmanager in
    #   telemetry/node_tracing.py; cleanup is handled by
    #   __aexit__, not Aclosing.
    if coro.__name__ in (
        'use_inference_span',
        '_use_native_generate_content_span',
        'run_async',
        '_run_node_async',
        '_consume_event_queue',
        'run_node_impl',
        'call_llm',
        'execute_tools',
        'start_as_current_node_span',
    ):
      firstiter(coro)
      return
    assert any(
        isinstance(referrer, Aclosing)
        or isinstance(indirect_referrer, Aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in Python 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), f'Coro `{coro.__name__}` is not wrapped with Aclosing'
    firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, finalizer)

  # Act
  async with Aclosing(test_runner.run_async_with_new_session_agen('')) as agen:
    async for _ in agen:
      pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert list(sorted(span.name for span in spans)) == [
      'call_llm',
      'call_llm',
      'execute_tool some_tool',
      'generate_content mock',
      'generate_content mock',
      'invoke_agent some_root_agent',
      'invoke_agent some_root_agent',
  ]


@pytest.mark.asyncio
async def test_exception_preserves_attributes(
    test_model: BaseLlm, span_exporter: InMemorySpanExporter
):
  """Test when an exception occurs during tool execution, span attributes are still present on spans where they are expected."""

  # Arrange
  async def some_tool():
    raise ValueError('This tool always fails')

  test_agent = Agent(
      name='some_root_agent',
      model=test_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )

  test_runner = TestInMemoryRunner(test_agent)

  # Act
  with pytest.raises(ValueError, match='This tool always fails'):
    async with Aclosing(
        test_runner.run_async_with_new_session_agen('')
    ) as agen:
      async for _ in agen:
        pass

  # Assert
  spans = span_exporter.get_finished_spans()

  assert len(spans) > 1
  assert all(
      span.attributes is not None and len(span.attributes) > 0
      for span in spans
      if span.name != 'invocation'  # not expected to have attributes
  )


@pytest.mark.asyncio
async def test_no_generate_content_for_gemini_model_when_already_instrumented(
    test_runner: TestInMemoryRunner,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
  """Tests"""
  # Arrange
  monkeypatch.setattr(
      tracing,
      '_instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: True,
  )
  monkeypatch.setattr(
      tracing,
      '_is_gemini_agent',
      lambda _: True,
  )

  # Act
  async with Aclosing(test_runner.run_async_with_new_session_agen('')) as agen:
    async for _ in agen:
      pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert not any(span.name.startswith('generate_content') for span in spans)


def test_instrumented_with_opentelemetry_instrumentation_google_genai():
  instrumentor = GoogleGenAiSdkInstrumentor()

  assert (
      not tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
  )
  try:
    instrumentor.instrument()
    assert (
        tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
    )
  finally:
    instrumentor.uninstrument()
  assert (
      not tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
  )
