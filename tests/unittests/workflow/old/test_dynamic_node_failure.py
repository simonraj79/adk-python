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

from google.adk.agents.context import Context
from google.adk.workflow import Edge
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._node import node
from google.adk.workflow._node_status import NodeStatus
import pytest

from ..workflow_testing_utils import create_parent_invocation_context


@node
async def failing_node(node_input: str):
  if node_input == "fail":
    raise ValueError("Intentional Failure")
  return f"Processed {node_input}"


@node(rerun_on_resume=True)
async def parent_node(ctx: Context):
  results = []
  try:
    await ctx.run_node(failing_node, node_input="fail")
  except ValueError as e:
    results.append(f"Caught: {str(e)}")

  # This should still work
  res = await ctx.run_node(failing_node, node_input="work")
  results.append(f"Success: {res}")
  yield results


@pytest.mark.asyncio
async def test_dynamic_node_failure_handling(request: pytest.FixtureRequest):
  agent = Workflow(
      name="test_dynamic_failure_workflow",
      edges=[
          (START, parent_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  results = []
  async for event in agent.run_async(ctx):
    if event.output:
      if isinstance(event.output, list):
        results.extend(event.output)
      else:
        results.append(event.output)

  assert "Caught: Intentional Failure" in results
  assert "Success: Processed work" in results
