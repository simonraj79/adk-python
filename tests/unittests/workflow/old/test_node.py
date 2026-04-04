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

"""Tests for @node decorator."""

from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._workflow_class import Workflow as WorkflowV2
from google.adk.workflow._agent_node import AgentNode
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper
from google.adk.workflow._node import node
from google.adk.workflow._node import Node
from google.adk.workflow._parallel_worker import _ParallelWorker as ParallelWorker
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.workflow._workflow import workflow_node_input
import pytest

from ..workflow_testing_utils import create_parent_invocation_context
from ..workflow_testing_utils import simplify_events_with_node
from .. import testing_utils

ANY = mock.ANY


@pytest.mark.asyncio
async def test_node_decorator(request: pytest.FixtureRequest):
  """Tests that @node decorator can wrap a function and override its name."""

  @node(name="decorated_node")
  def my_func():
    return "Hello from decorated_func"

  assert my_func.name == "decorated_node"

  agent = Workflow(
      name="test_agent",
      edges=[
          (START, my_func),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]

  assert simplify_events_with_node(events) == [
      (
          "test_agent",
          {
              "node_name": "decorated_node",
              "output": "Hello from decorated_func",
          },
      ),
  ]


def test_node_parallel_worker_instance():
  """Tests that node() can wrap a node in ParallelWorker."""

  @node(parallel_worker=True)
  def my_func(node_input):
    return node_input

  assert isinstance(my_func, ParallelWorker)
  assert my_func.name == "my_func"

  def other_func(x):
    return x

  parallel_node = node(other_func, parallel_worker=True)
  assert isinstance(parallel_node, ParallelWorker)
  assert parallel_node.name == "other_func"


@pytest.mark.asyncio
async def test_node_parallel_worker_execution(request: pytest.FixtureRequest):
  """Tests that a node with parallel_worker=True correctly processes inputs."""

  @node(parallel_worker=True)
  async def my_func(node_input):
    return node_input * 2

  async def producer_func() -> list[int]:
    return [1, 2, 3]

  agent = WorkflowV2(
      name="test_agent",
      edges=[
          (START, producer_func),
          (producer_func, my_func),
      ],
  )
  from google.adk.apps.app import App
  test_app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=test_app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  simplified_events = simplify_events_with_node(
      events, use_node_path=True, include_run_id=True
  )

  assert simplified_events == [
      (
          'test_agent@1/producer_func@1',
          {
              'node_name': 'producer_func',
              'output': [1, 2, 3],
              'run_id': None,
          },
      ),
      (
          'test_agent@1/my_func@1/my_func@1',
          {
              'node_name': 'my_func',
              'output': 2,
              'run_id': None,
          },
      ),
      (
          'test_agent@1/my_func@1/my_func@2',
          {
              'node_name': 'my_func',
              'output': 4,
              'run_id': None,
          },
      ),
      (
          'test_agent@1/my_func@1/my_func@3',
          {
              'node_name': 'my_func',
              'output': 6,
              'run_id': None,
          },
      ),
      (
          'test_agent@1/my_func@1',
          {
              'node_name': 'my_func',
              'output': [2, 4, 6],
              'run_id': None,
          },
      ),
  ]


def test_node_decorator_rerun_on_resume():
  """Tests that @node decorator can override rerun_on_resume."""

  @node(name="decorated_node", rerun_on_resume=True)
  def my_func():
    return "Hello from decorated_func"

  assert isinstance(my_func, FunctionNode)
  assert my_func.rerun_on_resume

  @node()
  def my_func2():
    return "Hello from decorated_func2"

  assert isinstance(my_func2, FunctionNode)
  assert not my_func2.rerun_on_resume


def test_node_function_with_base_node():
  """Tests that node() function returns a copied node when given a BaseNode."""

  @node(name="original")
  def original():
    pass

  wrapped = node(original, name="overridden", rerun_on_resume=True)

  assert isinstance(wrapped, FunctionNode)
  assert wrapped is not original
  assert wrapped.name == "overridden"
  assert wrapped.rerun_on_resume


# BaseTool
class MyTool(BaseTool):
  name = "tool"
  description = "desc"

  async def _run_async_impl(self):
    return "done"


def test_node_no_unnecessary_wrap():
  """Tests that node() does not wrap LlmAgent, Agent, Tool, or func in OverridingNode."""

  # LlmAgent: auto-converted to single_turn and wrapped in LlmAgentWrapper
  from google.adk.workflow._llm_agent_wrapper import _LlmAgentWrapper

  llm_agent = LlmAgent(name="llm")
  llm_node = node(llm_agent, name="overridden_llm")
  assert isinstance(llm_node, _LlmAgentWrapper)
  assert llm_node.name == "overridden_llm"
  assert llm_agent.mode == "single_turn"

  # BaseAgent
  agent = BaseAgent(name="agent")
  agent_node_inst = node(agent, name="overridden_agent", rerun_on_resume=True)
  assert isinstance(agent_node_inst, AgentNode)
  assert agent_node_inst.name == "overridden_agent"
  assert agent_node_inst.rerun_on_resume

  tool_inst = MyTool(name="tool", description="desc")
  t_node = node(tool_inst, name="overridden_tool")
  assert isinstance(t_node, ToolNode)
  assert t_node.name == "overridden_tool"

  # Callable
  def my_func():
    pass

  f_node = node(my_func, name="overridden_func", rerun_on_resume=True)
  assert isinstance(f_node, FunctionNode)
  assert f_node.name == "overridden_func"
  assert f_node.rerun_on_resume


class StatefulTool(BaseTool):
  """A tool that modifies state via tool_context."""

  async def run_async(self, *, args, tool_context):
    tool_context.state["tool_key"] = "tool_value"
    tool_context.state["tool_count"] = 10
    return {"status": "ok"}


class StatefulToolNoReturn(BaseTool):
  """A tool that modifies state but returns None."""

  async def run_async(self, *, args, tool_context):
    tool_context.state["silent_key"] = "silent_value"
    return None


@pytest.mark.asyncio
async def test_tool_node_state_delta(request: pytest.FixtureRequest):
  """Tests that state set via tool_context.state in ToolNode is persisted."""

  tool_node = ToolNode(
      tool=StatefulTool(name="stateful_tool", description="Sets state values"),
  )

  def read_state(tool_key: str, tool_count: int) -> str:
    return f"tool_key={tool_key}, tool_count={tool_count}"

  agent = Workflow(
      name="test_tool_node_state_delta",
      edges=[
          (START, tool_node),
          (tool_node, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          "test_tool_node_state_delta",
          {
              "node_name": "stateful_tool",
              "output": {"status": "ok"},
              "state_delta": {"tool_key": "tool_value", "tool_count": 10},
          },
      ),
      (
          "test_tool_node_state_delta",
          {
              "node_name": "read_state",
              "output": "tool_key=tool_value, tool_count=10",
          },
      ),
  ]


@pytest.mark.asyncio
async def test_tool_node_state_delta_no_return(
    request: pytest.FixtureRequest,
):
  """Tests that state is persisted even when tool returns None."""

  tool_node = ToolNode(
      tool=StatefulToolNoReturn(
          name="stateful_tool_no_return",
          description="Sets state, returns None",
      ),
  )

  def read_state(silent_key: str) -> str:
    return f"silent_key={silent_key}"

  agent = Workflow(
      name="test_tool_node_state_delta_no_return",
      edges=[
          (START, tool_node),
          (tool_node, read_state),
      ],
  )
  ctx = await create_parent_invocation_context(request.function.__name__, agent)
  events = [e async for e in agent.run_async(ctx)]
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          "test_tool_node_state_delta_no_return",
          {
              "node_name": "stateful_tool_no_return",
              "output": None,
              "state_delta": {"silent_key": "silent_value"},
          },
      ),
      (
          "test_tool_node_state_delta_no_return",
          {
              "node_name": "read_state",
              "output": "silent_key=silent_value",
          },
      ),
  ]


def test_node_class_with_run_node_impl():
  """Tests that Node class uses run_node_impl when subclassed."""

  class MyClassNode(Node):

    async def run_node_impl(self, ctx, node_input):
      yield node_input * 3

  n = MyClassNode(name="my_class_node", rerun_on_resume=True)
  assert n.name == "my_class_node"
  assert n.rerun_on_resume


def test_node_model_copy():
  """Tests that model_copy updates the node appropriately."""

  class MyClassNode(Node):

    async def run_node_impl(self, ctx, node_input):
      yield node_input * 3

  n1 = MyClassNode(name="original_name", rerun_on_resume=False)
  n2 = n1.model_copy(update={"name": "new_name", "rerun_on_resume": True})

  assert n2.name == "new_name"
  assert n2.rerun_on_resume is True


def test_node_parallel_worker_model_copy():
  """Tests that model_copy updates the inner node correctly for parallel worker."""

  class MyClassNode(Node):

    async def run_node_impl(self, ctx, node_input):
      yield node_input * 3

  n1 = MyClassNode(name="original_name", parallel_worker=True)
  assert isinstance(n1._inner_node, ParallelWorker)

  n2 = n1.model_copy(update={"name": "new_name"})

  assert n2.name == "new_name"
  assert isinstance(n2._inner_node, ParallelWorker)
  assert n2._inner_node.name == "new_name"
  assert n2._inner_node._node.name == "new_name"


def test_node_parallel_worker_frozen():
  """Tests that parallel_worker field is frozen after initialization."""
  from pydantic import ValidationError

  n = Node(name="test_node", parallel_worker=False)
  assert not n.parallel_worker

  with pytest.raises(ValidationError, match="Field is frozen"):
    n.parallel_worker = True


def test_node_decorator_supports_retry_config_and_timeout():
  """Tests that @node decorator supports retry_config and timeout."""
  retry_config = RetryConfig(max_attempts=3)
  timeout = 10.5

  @node(retry_config=retry_config, timeout=timeout)
  def my_func():
    pass

  assert isinstance(my_func, FunctionNode)
  assert my_func.retry_config == retry_config
  assert my_func.timeout == timeout


def test_node_function_supports_retry_config_and_timeout_for_callables():
  """Tests that node() function supports retry_config and timeout for callables."""
  retry_config = RetryConfig(max_attempts=5)
  timeout = 20.0

  def my_func():
    pass

  n = node(my_func, retry_config=retry_config, timeout=timeout)
  assert isinstance(n, FunctionNode)
  assert n.retry_config == retry_config
  assert n.timeout == timeout


def test_node_function_supports_retry_config_and_timeout_for_agents():
  """Tests that node() function supports retry_config and timeout for agents."""
  retry_config = RetryConfig(max_attempts=2)
  timeout = 5.0
  agent = LlmAgent(name="test_agent")

  n = node(agent, retry_config=retry_config, timeout=timeout)
  assert isinstance(n, _LlmAgentWrapper)
  assert n.retry_config == retry_config
  assert n.timeout == timeout


def test_node_function_supports_retry_config_and_timeout_for_tools():
  """Tests that node() function supports retry_config and timeout for tools."""
  retry_config = RetryConfig(max_attempts=4)
  timeout = 15.0

  class MyLocalTool(BaseTool):
    name = "my_tool"
    description = "desc"

    async def _run_async_impl(self):
      pass

  tool = MyLocalTool(name="my_tool", description="desc")
  n = node(tool, retry_config=retry_config, timeout=timeout)
  assert isinstance(n, ToolNode)
  assert n.retry_config == retry_config
  assert n.timeout == timeout


def test_node_subclass_propagates_retry_config_and_timeout_with_parallel_worker():
  """Tests that retry_config and timeout are propagated in Node subclass with parallel_worker."""
  retry_config = RetryConfig(max_attempts=3)
  timeout = 30.0

  class MyLocalNode(Node):

    async def run_node_impl(self, ctx, node_input):
      yield "done"

  n = MyLocalNode(
      name="my_node",
      parallel_worker=True,
      retry_config=retry_config,
      timeout=timeout,
  )

  assert n.retry_config == retry_config
  assert n.timeout == timeout
  assert n._inner_node.retry_config == None
  assert n._inner_node.timeout == None
  # The inner node of ParallelWorker also needs it
  assert n._inner_node._node.retry_config == retry_config
  assert n._inner_node._node.timeout == timeout


def test_node_model_copy_propagates_retry_config_and_timeout_with_parallel_worker():
  """Tests that retry_config and timeout are propagated during model_copy with parallel_worker."""
  retry_config1 = RetryConfig(max_attempts=1)
  timeout1 = 10.0
  retry_config2 = RetryConfig(max_attempts=2)
  timeout2 = 20.0

  class MyLocalNode(Node):

    async def run_node_impl(self, ctx, node_input):
      yield "done"

  n1 = MyLocalNode(
      name="my_node",
      parallel_worker=True,
      retry_config=retry_config1,
      timeout=timeout1,
  )

  n2 = n1.model_copy(
      update={"retry_config": retry_config2, "timeout": timeout2}
  )

  assert n2.retry_config == retry_config2
  assert n2.timeout == timeout2
  assert n2._inner_node.retry_config == None
  assert n2._inner_node.timeout == None
  assert n2._inner_node._node.retry_config == retry_config2
  assert n2._inner_node._node.timeout == timeout2
