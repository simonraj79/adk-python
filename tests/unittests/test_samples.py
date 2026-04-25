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

import json
from pathlib import Path

from google.adk.apps.app import App
from google.adk.cli.agent_test_runner import test_agent_replay
from google.genai import types
import pytest

CONTRIBUTING_DIR = Path(__file__).parent.parent.parent / "contributing"


def get_test_files():
  """Yields (sample_dir, test_file_path)."""
  if not CONTRIBUTING_DIR.exists():
    return
  for category_dir in CONTRIBUTING_DIR.iterdir():
    if category_dir.is_dir():
      for sample_dir in category_dir.iterdir():
        if sample_dir.is_dir():
          tests_dir = sample_dir / "tests"
          if tests_dir.exists() and tests_dir.is_dir():
            for test_file in tests_dir.glob("*.json"):
              if test_file.stem.endswith("_xfail"):
                yield pytest.param(
                    sample_dir, test_file, marks=pytest.mark.xfail
                )
              else:
                yield sample_dir, test_file


@pytest.mark.parametrize(
    "sample_dir, test_file",
    list(get_test_files()),
    ids=lambda val: val.name if isinstance(val, Path) else val,
)
def test_sample(sample_dir: Path, test_file: Path, monkeypatch):
  """Tests a sample by replaying exported session events."""
  test_agent_replay(sample_dir, test_file, monkeypatch)
