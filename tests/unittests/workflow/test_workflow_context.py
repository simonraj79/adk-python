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

"""Unit tests for workflow_context."""

from unittest import mock

from google.adk.agents import context as workflow_context
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import new_invocation_context_id
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


def test_events_merges_local_events_without_duplicates():
  """Tests that events merges local events without duplicates."""
  session = Session(
      id='session_id',
      app_name='test_app',
      user_id='test_user',
      events=[Event(id='1', author='test')],
  )
  local_events = [
      Event(id='1', author='test'),
      Event(id='2', author='test'),
  ]
  invocation_context = InvocationContext(
      invocation_id=new_invocation_context_id(),
      agent=mock.create_autospec(BaseAgent),
      session=session,
      session_service=mock.create_autospec(BaseSessionService),
  )
  ctx = workflow_context.Context(
      invocation_context,
      node_path='test_node_path',
      run_id='test_run_id',
      local_events=local_events,
  )
  proxy = ctx.session
  assert len(proxy.events) == 2
  assert proxy.events[0].id == '1'
  assert proxy.events[1].id == '2'


def test_other_attributes_are_delegated():
  """Tests that other attributes are delegated to the underlying session."""
  session = Session(
      id='session_id',
      app_name='test_app',
      user_id='test_user',
      state={'prop1': 'value1'},
  )
  invocation_context = InvocationContext(
      invocation_id=new_invocation_context_id(),
      agent=mock.create_autospec(BaseAgent),
      session=session,
      session_service=mock.create_autospec(BaseSessionService),
  )
  ctx = workflow_context.Context(
      invocation_context,
      node_path='test_node_path',
      run_id='test_run_id',
      local_events=[],
  )
  proxy = ctx.session
  assert proxy.id == 'session_id'
  assert proxy.state == {'prop1': 'value1'}


def test_set_events_raises_attribute_error():
  """Tests that setting events raises an AttributeError."""
  session = Session(id='session_id', app_name='test_app', user_id='test_user')
  invocation_context = InvocationContext(
      invocation_id=new_invocation_context_id(),
      agent=mock.create_autospec(BaseAgent),
      session=session,
      session_service=mock.create_autospec(BaseSessionService),
  )
  ctx = workflow_context.Context(
      invocation_context,
      node_path='test_node_path',
      run_id='test_run_id',
      local_events=[],
  )
  proxy = ctx.session
  with pytest.raises(
      AttributeError, match="Cannot set 'events' on SessionProxy."
  ):
    proxy.events = []


def test_actual_session_returns_underlying_session():
  """Tests that actual_session returns the underlying session."""
  session = Session(id='session_id', app_name='test_app', user_id='test_user')
  invocation_context = InvocationContext(
      invocation_id=new_invocation_context_id(),
      agent=mock.create_autospec(BaseAgent),
      session=session,
      session_service=mock.create_autospec(BaseSessionService),
  )
  ctx = workflow_context.Context(
      invocation_context,
      node_path='test_node_path',
      run_id='test_run_id',
      local_events=[],
  )
  proxy = ctx.session
  assert proxy.actual_session is session
