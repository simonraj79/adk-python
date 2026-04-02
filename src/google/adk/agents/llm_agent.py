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

"""Re-export shim for the 2.0 Mesh-based LlmAgent.

The actual implementation lives in llm_agent_workflow/llm_agent.py.
This module preserves the original import path so that existing code
using ``from google.adk.agents.llm_agent import ...`` continues to work.
"""

from .llm_agent_workflow.llm_agent import Agent
from .llm_agent_workflow.llm_agent import AfterModelCallback
from .llm_agent_workflow.llm_agent import AfterToolCallback
from .llm_agent_workflow.llm_agent import BeforeModelCallback
from .llm_agent_workflow.llm_agent import BeforeToolCallback
from .llm_agent_workflow.llm_agent import InstructionProvider
from .llm_agent_workflow.llm_agent import LlmAgent
from .llm_agent_workflow.llm_agent import LlmAgentConfig
from .llm_agent_workflow.llm_agent import OnModelErrorCallback
from .llm_agent_workflow.llm_agent import OnToolErrorCallback
from .llm_agent_workflow.llm_agent import ToolUnion
from .llm_agent_workflow.llm_agent import _convert_tool_union_to_tools
