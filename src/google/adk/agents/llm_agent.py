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

"""Dynamic gateway for LlmAgent.

Toggles between 1.X and 2.0 implementations based on the V1_LLM_AGENT feature flag.
"""

from __future__ import annotations

from .llm_agent_1x import _convert_tool_union_to_tools
from .llm_agent_1x import AfterModelCallback
from .llm_agent_1x import AfterToolCallback
from .llm_agent_1x import Agent
from .llm_agent_1x import BeforeModelCallback
from .llm_agent_1x import BeforeToolCallback
from .llm_agent_1x import InstructionProvider
from .llm_agent_1x import LlmAgent
from .llm_agent_1x import LlmAgentConfig
from .llm_agent_1x import OnModelErrorCallback
from .llm_agent_1x import OnToolErrorCallback
from .llm_agent_1x import ToolUnion

__all__ = [
    "Agent",
    "LlmAgent",
    "LlmAgentConfig",
    "BeforeModelCallback",
    "AfterModelCallback",
    "OnModelErrorCallback",
    "BeforeToolCallback",
    "AfterToolCallback",
    "OnToolErrorCallback",
    "InstructionProvider",
    "ToolUnion",
]
