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

"""Tests for node registry duplicate detection."""

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._agent_node import AgentNode
from google.adk.workflow._dynamic_node_registry import dynamic_node_registry
from google.adk.workflow._tool_node import _ToolNode
from google.adk.workflow._v1_llm_agent_wrapper import _V1LlmAgentWrapper
import pytest

from .. import testing_utils


@pytest.fixture(autouse=True)
def clear_dynamic_node_registry():
  """Clears the dynamic_node_registry before each test."""
  dynamic_node_registry.clear()
  yield


def test_duplicate_registration_explicit():
  """Tests that explicitly registering a duplicate node name raises ValueError."""
  unique_name = "unique_node_explicit"

  workflow_name = "test_workflow"
  node1 = FunctionNode(lambda: 1, name=unique_name)
  dynamic_node_registry.register(node1, workflow_name)

  node2 = FunctionNode(lambda: 2, name=unique_name)

  with pytest.raises(
      ValueError,
      match=(
          f"Dynamic node with name '{unique_name}' already exists in registry"
          f" for workflow '{workflow_name}'"
      ),
  ):
    dynamic_node_registry.register(node2, workflow_name)


def test_register_same_node_again_is_allowed():
  """Tests that registering the same node object again is allowed."""
  node_name = "same_node_allowed"
  workflow_name = "workflow_same_node"
  node1 = FunctionNode(lambda: 1, name=node_name)
  dynamic_node_registry.register(node1, workflow_name)
  # Should not raise if registering the same node object again
  dynamic_node_registry.register(node1, workflow_name)
  assert dynamic_node_registry.get(node_name, workflow_name) is node1


def test_duplicate_node_name_different_workflows():
  """Tests that registering nodes with same name under different workflows is allowed."""
  node_name = "shared_node_name"
  workflow1_name = "workflow1"
  workflow2_name = "workflow2"

  node1 = FunctionNode(lambda: 1, name=node_name)
  node2 = FunctionNode(lambda: 2, name=node_name)

  dynamic_node_registry.register(node1, workflow1_name)
  dynamic_node_registry.register(node2, workflow2_name)

  # Assert that nodes are registered and retrievable
  assert dynamic_node_registry.get(node_name, workflow1_name) is node1
  assert dynamic_node_registry.get(node_name, workflow2_name) is node2


@pytest.mark.asyncio
async def test_duplicate_dynamic_node_execution(request):
  """Tests that running a dynamic node with a duplicate name raises ValueError."""

  unique_name = "unique_node_dynamic"

  workflow_name = "test_agent_duplicate_dynamic"
  # Register the first node
  node1 = FunctionNode(lambda: 1, name=unique_name)
  dynamic_node_registry.register(node1, workflow_name)

  # Define the second node (different object) with the same name
  node2 = FunctionNode(lambda: 2, name=unique_name)

  async def workflow_logic(ctx: Context) -> str:
    # This should fail when scheduling node2
    await ctx.run_node(node2)
    return "done"

  workflow_logic = FunctionNode(func=workflow_logic, rerun_on_resume=True)

  agent = Workflow(
      name="test_agent_duplicate_dynamic",
      edges=[(START, workflow_logic)],
  )

  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  user_event = testing_utils.get_user_content("start")

  # The run should fail with an exception
  with pytest.raises(
      ValueError,
      match=(
          f"Dynamic node with name '{unique_name}' already exists in registry"
          f" for workflow '{workflow_name}'"
      ),
  ):
    await runner.run_async(user_event)


class MockTool(BaseTool):

  def run(self, **kwargs):
    return "done"


def test_llm_agent_wrapper_deep_check_success():
  """Tests that registering effectively same LlmAgentWrapper is allowed."""
  agent = LlmAgent(
      name="test_agent", model="gemini-2.5-flash", instruction="test"
  )
  node1 = _V1LlmAgentWrapper(agent=agent)
  node2 = _V1LlmAgentWrapper(agent=agent)

  assert node1 is not node2
  assert node1 == node2

  workflow_name = "test_workflow"
  dynamic_node_registry.register(node1, workflow_name)
  # Should not raise ValueError
  dynamic_node_registry.register(node2, workflow_name)


def test_agent_node_deep_check_success():
  """Tests that registering effectively same AgentNode is allowed."""
  agent = LlmAgent(
      name="test_agent", model="gemini-2.5-flash", instruction="test"
  )
  node1 = AgentNode(agent=agent)
  node2 = AgentNode(agent=agent)

  assert node1 is not node2
  assert node1 == node2

  workflow_name = "test_workflow"
  dynamic_node_registry.register(node1, workflow_name)
  # Should not raise ValueError
  dynamic_node_registry.register(node2, workflow_name)


def test_tool_node_deep_check_success():
  """Tests that registering effectively same ToolNode is allowed."""
  tool = MockTool(name="test_tool", description="test")
  node1 = _ToolNode(tool=tool)
  node2 = _ToolNode(tool=tool)

  assert node1 is not node2
  assert node1 == node2

  workflow_name = "test_workflow"
  dynamic_node_registry.register(node1, workflow_name)
  # Should not raise ValueError
  dynamic_node_registry.register(node2, workflow_name)


def test_different_nodes_same_name_still_fails():
  """Tests that registering node with same name but different config fails."""
  agent1 = LlmAgent(
      name="test_agent", model="gemini-2.5-flash", instruction="t1"
  )
  agent2 = LlmAgent(
      name="test_agent", model="gemini-2.5-flash", instruction="t2"
  )
  node1 = _V1LlmAgentWrapper(agent=agent1)
  node2 = _V1LlmAgentWrapper(agent=agent2)

  assert node1 != node2

  workflow_name = "test_workflow"
  dynamic_node_registry.register(node1, workflow_name)

  with pytest.raises(ValueError, match="already exists"):
    dynamic_node_registry.register(node2, workflow_name)
