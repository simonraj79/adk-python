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

"""Guard tests for the release-cut dependency contract.

These tests pin the public dependency surface so that release-blocking
regressions documented in the bare-install audit cannot silently re-emerge:

* ``packaging`` MUST be declared in main deps (used at import-time by
  ``utils/model_name_utils.py`` and ``cli/cli_deploy.py``; reachable from
  ``from google.adk import Runner`` and from ``adk --help``).
* ``ValidationError`` in ``environment_simulation_config`` MUST come from
  ``pydantic`` (which always installs alongside the package), NOT from the
  undeclared ``pydantic_core``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_PATH = _REPO_ROOT / 'pyproject.toml'


@pytest.fixture(scope='module')
def pyproject() -> dict:
  """Parses the project's pyproject.toml exactly once for the module."""
  with _PYPROJECT_PATH.open('rb') as fh:
    return tomllib.load(fh)


def _requirement_names(requirements: list[str]) -> set[str]:
  """Returns the lowercased PEP 508 distribution names from ``requirements``.

  Strips extras specifiers, version specifiers, and environment markers so the
  caller can do exact-name membership checks.
  """
  names: set[str] = set()
  for req in requirements:
    # Drop everything after a marker, version specifier, or extras block.
    head = req.split(';', 1)[0].strip()
    for sep in ('[', '>', '<', '=', '!', '~', ' '):
      head = head.split(sep, 1)[0]
    names.add(head.strip().lower())
  return names


def test_main_deps_include_packaging(pyproject: dict) -> None:
  """``packaging`` is imported unguarded by core ADK; it must be a main dep."""
  main_deps = _requirement_names(pyproject['project']['dependencies'])
  assert 'packaging' in main_deps, (
      'packaging must be declared in [project] dependencies because '
      'src/google/adk/utils/model_name_utils.py and '
      'src/google/adk/cli/cli_deploy.py import it unguarded at module top '
      'level. Without this declaration, `pip install google-adk` is one '
      'transitive resolver change away from breaking on `import google.adk`.'
  )


def test_environment_simulation_config_imports_validation_error_from_pydantic() -> (
    None
):
  """The ValidationError used by the config module must come from pydantic.

  pydantic-core is undeclared; importing from it directly is fragile. pydantic
  re-exports ValidationError, so use that.
  """
  source_path = (
      _REPO_ROOT
      / 'src'
      / 'google'
      / 'adk'
      / 'tools'
      / 'environment_simulation'
      / 'environment_simulation_config.py'
  )
  source = source_path.read_text(encoding='utf-8')
  assert 'from pydantic import ValidationError' in source, (
      'environment_simulation_config.py must import ValidationError from '
      'pydantic, not pydantic_core. pydantic_core is undeclared as a main '
      'dep and pydantic re-exports the same class.'
  )
  assert 'from pydantic_core import ValidationError' not in source, (
      'environment_simulation_config.py must not import ValidationError '
      'from pydantic_core (undeclared dep).'
  )


def test_injection_config_validation_raises_pydantic_validation_error() -> None:
  """Behavioral check: invalid config raises the pydantic ValidationError."""
  # Local import keeps this test focused on the post-fix code path and
  # surfaces ImportError clearly if the module's import block regresses.
  from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
  from pydantic import ValidationError

  with pytest.raises(ValidationError):
    # Both required fields missing — pydantic must reject the construction.
    InjectedError()  # type: ignore[call-arg]
