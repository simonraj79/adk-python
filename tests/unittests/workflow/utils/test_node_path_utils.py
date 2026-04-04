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

"""Tests for node_path_utils."""

from google.adk.workflow.utils import _node_path_utils as node_path_utils


def test_get_node_name_from_path():
  assert (
      node_path_utils.get_node_name_from_path('workflow/agent/node') == 'node'
  )
  assert node_path_utils.get_node_name_from_path('node') == 'node'
  assert node_path_utils.get_node_name_from_path('') == ''


def test_get_parent_path():
  assert (
      node_path_utils.get_parent_path('workflow/agent/node') == 'workflow/agent'
  )
  assert node_path_utils.get_parent_path('node') == ''
  assert node_path_utils.get_parent_path('') == ''


def test_join_paths():
  assert (
      node_path_utils.join_paths('workflow/agent', 'node')
      == 'workflow/agent/node'
  )
  assert node_path_utils.join_paths(None, 'node') == 'node'
  assert node_path_utils.join_paths('', 'node') == 'node'


def test_is_direct_child():
  assert node_path_utils.is_direct_child(
      'workflow/agent/node', 'workflow/agent'
  )
  assert not node_path_utils.is_direct_child(
      'workflow/agent/node/temp', 'workflow/agent'
  )
  assert not node_path_utils.is_direct_child('node', 'workflow/agent')
  assert node_path_utils.is_direct_child('node', '')
  assert not node_path_utils.is_direct_child(None, 'workflow/agent')


def test_direct_child_name():
  assert node_path_utils.direct_child_name('wf', 'wf/nodeA') == 'nodeA'
  assert node_path_utils.direct_child_name('wf', 'wf/inner/nodeA') == 'inner'
  assert (
      node_path_utils.direct_child_name('wf/inner', 'wf/inner/deep/leaf')
      == 'deep'
  )
  assert node_path_utils.direct_child_name('', 'nodeA') == 'nodeA'
  assert node_path_utils.direct_child_name('', 'nodeA/subnode') == 'nodeA'


def test_is_descendant():
  assert node_path_utils.is_descendant('workflow/agent', 'workflow/agent/node')
  assert node_path_utils.is_descendant(
      'workflow/agent', 'workflow/agent/node/temp'
  )
  assert not node_path_utils.is_descendant(
      'workflow/agent', 'workflow/other_agent'
  )
  assert not node_path_utils.is_descendant('workflow/agent', 'node')
  assert not node_path_utils.is_descendant('workflow/agent', None)
  assert node_path_utils.is_descendant('', 'node')
  assert node_path_utils.is_descendant('', 'workflow/agent/node')
  assert not node_path_utils.is_descendant('', '')
