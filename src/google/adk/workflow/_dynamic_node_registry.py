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

"""A registry for dynamic nodes in a workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ._base_node import BaseNode


class DynamicNodeRegistry:
  """A registry for workflow nodes."""

  def __init__(self) -> None:
    # This dict stores reference to the source dynamic nodes.
    # The keys are workflow names and the values are dicts of node name to
    # node implementation.
    # The workflow name is used to scope the node registration to the workflow
    # that registered them. This prevents conflicts between nodes with the same
    # name in different workflows.
    #
    # Example:
    # {
    #     "my_workflow": {
    #         "ask_input": AskInputNode,
    #     },
    # }
    #
    self._nodes: dict[str, dict[str, BaseNode]] = {}

  def register(self, node: "BaseNode", workflow_name: str) -> None:
    """Registers a node in the registry."""
    workflow_nodes = self._nodes.setdefault(workflow_name, {})

    if node.name in workflow_nodes:
      existing_node = workflow_nodes[node.name]
      if existing_node is not node:
        # Check if they are effectively the same.
        if type(existing_node) is type(node) and existing_node == node:
          return
        raise ValueError(
            f"Dynamic node with name '{node.name}' already exists in"
            f" registry for workflow '{workflow_name}'."
        )
      return
    workflow_nodes[node.name] = node

  def get(self, name: str, workflow_name: str) -> "BaseNode | None":
    """Retrieves a node by name."""
    if workflow_name not in self._nodes:
      return None
    return self._nodes[workflow_name].get(name)

  def clear(self) -> None:
    """Clears the registry."""
    self._nodes = {}


dynamic_node_registry = DynamicNodeRegistry()
