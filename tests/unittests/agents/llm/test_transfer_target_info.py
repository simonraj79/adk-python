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

"""Unit tests for _TransferTargetInfo."""

from __future__ import annotations

from google.adk.agents.llm._transfer_target_info import _TransferTargetInfo
import pytest


class Test_TransferTargetInfo:
  """Tests for _TransferTargetInfo dataclass."""

  def test_create_with_name_only(self):
    """Should create _TransferTargetInfo with name only."""
    info = _TransferTargetInfo(name='expense_agent')

    assert info.name == 'expense_agent'
    assert info.description == ''

  def test_create_with_name_and_description(self):
    """Should create _TransferTargetInfo with name and description."""
    info = _TransferTargetInfo(
        name='expense_agent',
        description='Handles expense reports and reimbursements.',
    )

    assert info.name == 'expense_agent'
    assert info.description == 'Handles expense reports and reimbursements.'

  def test_name_is_required(self):
    """Should raise error when name is missing."""
    with pytest.raises(Exception):
      _TransferTargetInfo()

  def test_model_dump(self):
    """Should serialize to dict correctly."""
    info = _TransferTargetInfo(
        name='timeoff_agent',
        description='Manages time off requests.',
    )

    data = info.model_dump()

    assert data == {
        'name': 'timeoff_agent',
        'description': 'Manages time off requests.',
    }

  def test_model_validate(self):
    """Should deserialize from dict correctly."""
    data = {
        'name': 'hr_agent',
        'description': 'Human resources assistant.',
    }

    info = _TransferTargetInfo.model_validate(data)

    assert info.name == 'hr_agent'
    assert info.description == 'Human resources assistant.'

  def test_equality(self):
    """Should compare equal when fields match."""
    info1 = _TransferTargetInfo(name='agent_a', description='Description A')
    info2 = _TransferTargetInfo(name='agent_a', description='Description A')
    info3 = _TransferTargetInfo(name='agent_b', description='Description A')

    assert info1 == info2
    assert info1 != info3

  def test_immutability_via_model_copy(self):
    """Should create modified copy via model_copy."""
    original = _TransferTargetInfo(name='original', description='Original desc')

    modified = original.model_copy(update={'description': 'Modified desc'})

    assert original.description == 'Original desc'
    assert modified.description == 'Modified desc'
    assert modified.name == 'original'
