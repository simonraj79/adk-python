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

"""Type definitions for building workflow graphs."""

from collections.abc import Callable
from typing import Annotated
from typing import Any
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SerializeAsAny

from ..tools.base_tool import BaseTool
from ._base_node import BaseNode


RouteValue: TypeAlias = bool | int | str
"""Type alias for valid routing values used in conditional graph edges."""

NodeLike: TypeAlias = BaseNode | BaseTool | Callable[..., Any] | Literal["START"]
"""Type alias for objects that can be converted to a workflow node."""

RoutingMap: TypeAlias = dict[RouteValue, NodeLike | tuple[NodeLike, ...]]
"""A mapping from route values to destination nodes.

Syntactic sugar for declaring multiple routed edges from a single source.
Values can be a single node or a tuple of nodes (fan-out).

Examples::

    {"route_a": node_a, "route_b": node_b}
    {"route_x": (node_a, node_b)}  # fan-out: both triggered
"""


class Edge(BaseModel):
    """An edge in the workflow graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    from_node: Annotated[BaseNode, SerializeAsAny()]
    """The from node."""

    to_node: Annotated[BaseNode, SerializeAsAny()]
    """The to node."""

    route: RouteValue | list[RouteValue] | None = Field(
        description=(
            "The route(s) that this edge is associated with."
            " A single value or a list of values. The edge is followed when the"
            " emitted route matches any value in the list."
        ),
        default=None,
    )


ChainElement: TypeAlias = NodeLike | tuple[NodeLike, ...] | RoutingMap
"""Type alias for an element in a workflow chain.

Can be a single NodeLike, a tuple of NodeLike (fan-out), or a RoutingMap.
"""

EdgeItem: TypeAlias = Edge | tuple[ChainElement, ...]
"""Type alias for an item that can be parsed into workflow edges.

Can be an explicit Edge object, or a tuple representing a chain of elements.
"""
