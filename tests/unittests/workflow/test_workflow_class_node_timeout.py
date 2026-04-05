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

'''Tests for node timeout behavior.'''

from __future__ import annotations

import asyncio
from google.adk.workflow import START
from google.adk.workflow._workflow_class import Workflow
from google.adk.workflow._node import node
from google.adk.workflow._errors import NodeTimeoutError
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_workflow(wf, message='start'):
  '''Run a Workflow through Runner, return collected events.'''
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


def _output_by_node(events):
  '''Extract (node_name_from_path, output) for child node events.'''
  results = []
  for e in events:
    if e.output is not None and e.node_info.path and '/' in e.node_info.path:
      node_name = e.node_info.path.rsplit('/', 1)[-1]
      if '@' in node_name:
        node_name = node_name.rsplit('@', 1)[0]
      results.append((node_name, e.output))
  return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_completes_within_timeout():
  '''A node that finishes before the timeout should succeed normally.'''
  
  @node(timeout=5.0)
  async def my_slow_node():
    await asyncio.sleep(0.01)
    return "done"

  wf = Workflow(
      name="test_workflow",
      edges=[
          (START, my_slow_node),
      ],
  )
  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)
  
  assert ('my_slow_node', 'done') in by_node


@pytest.mark.asyncio
async def test_node_exceeds_timeout():
  '''A node that exceeds its timeout should fail.'''
  
  from google.adk.workflow import FunctionNode
  
  async def raw_slow_func():
    await asyncio.sleep(1.0)
    return "done"
    
  my_too_slow_node = FunctionNode(name='my_too_slow_node', func=raw_slow_func, timeout=0.05)

  wf = Workflow(
      name="test_workflow",
      edges=[
          (START, my_too_slow_node),
      ],
  )
  events, _, _ = await _run_workflow(wf)


  print(f"\n@@@ RAW EVENTS: {events}")
  
  # Verify that an error event was yielded

  error_events = [e for e in events if e.error_code is not None]
  assert len(error_events) >= 1
  assert any('my_too_slow_node' in e.node_info.path for e in error_events)


@pytest.mark.asyncio
async def test_node_no_timeout():
  '''A node with timeout=None should run without any time limit.'''
  
  @node(timeout=None)
  async def my_no_timeout_node():
    await asyncio.sleep(0.01)
    return "done"

  wf = Workflow(
      name="test_workflow",
      edges=[
          (START, my_no_timeout_node),
      ],
  )
  events, _, _ = await _run_workflow(wf)
  by_node = _output_by_node(events)
  
  assert ('my_no_timeout_node', 'done') in by_node
