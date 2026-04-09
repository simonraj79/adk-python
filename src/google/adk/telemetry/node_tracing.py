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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_DESCRIPTION
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.util.types import Attributes

from ..agents.context import Context
from ..workflow._agent_node import AgentNode
from ..workflow._base_node import BaseNode
from .tracing import tracer

if TYPE_CHECKING:
  from ..workflow._v1_llm_agent_wrapper import _V1LlmAgentWrapper
  from ..workflow._workflow_class import Workflow


@dataclass
class TelemetryContext:
  """Telemetry specific context tied to the lifetime of the span.

  Created to accommodate possible future of capturing of Agent
  inputs and outputs, similar to how tracing for inference is
  implemented in `google.adk.telemetry.tracing`.
  """

  otel_context: context_api.Context


@dataclass
class _SpanMetadata:
  name: str
  attributes: Attributes


@asynccontextmanager
async def start_as_current_node_span(
    context: Context, node: BaseNode
) -> AsyncIterator[TelemetryContext]:
  """Creates a scope-based OpenTelemetry span, representing a node invocation.

  Implements emitting of the following spans:
  - `invoke_agent {agent.name}`
  - `invoke_workflow {workflow.name}`
  - `invoke_node {node.name}`

  invoke_agent spans align with OpenTelemetry Semantic Conventions (semconv) version 1.36 spans for backwards compatibility.
  https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/README.md

  invoke_workflow spans align with semconv version 1.41, because these were not included in any prior releases.
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/README.md

  invoke_node spans are not present in any semconv release.
  We will create a proposal to standardize them.

  Args:
    context: Context in which the span is created.
    node: The node to be invoked inside the created span.

  Yields:
    Context with the started span.
  """

  span_metadata = _span_metadata(context, node)
  if span_metadata is None:
    yield TelemetryContext(otel_context=context.otel_context)
    return

  with tracer.start_as_current_span(
      span_metadata.name,
      attributes=span_metadata.attributes,
      context=context.otel_context,
  ):
    yield TelemetryContext(otel_context=context_api.get_current())


def _span_metadata(context: Context, node: BaseNode) -> _SpanMetadata | None:
  from ..workflow._v1_llm_agent_wrapper import _V1LlmAgentWrapper
  from ..workflow._workflow_class import Workflow

  if isinstance(node, (AgentNode, _V1LlmAgentWrapper)):
    return _agent_span_metadata(context, node)
  elif isinstance(node, Workflow):
    return _workflow_span_metadata(context, node)
  else:
    return _default_node_span_metadata(context, node)


def _agent_span_metadata(
    context: Context, agent_node: AgentNode | _V1LlmAgentWrapper
) -> _SpanMetadata:
  agent = agent_node.agent
  return _SpanMetadata(
      name=f'invoke_agent {agent.name}',
      attributes={
          GEN_AI_OPERATION_NAME: 'invoke_agent',
          GEN_AI_AGENT_DESCRIPTION: agent.description,
          GEN_AI_AGENT_NAME: agent.name,
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
  )


def _workflow_span_metadata(
    context: Context, workflow: Workflow
) -> _SpanMetadata:
  return _SpanMetadata(
      name=f'invoke_workflow {workflow.name}',
      attributes={
          GEN_AI_OPERATION_NAME: 'invoke_workflow',
          'gen_ai.workflow.name': workflow.name,
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
  )


def _default_node_span_metadata(
    context: Context, node: BaseNode
) -> _SpanMetadata:
  return _SpanMetadata(
      name=f'invoke_node {node.name}',
      attributes={
          GEN_AI_OPERATION_NAME: 'invoke_node',
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
  )
