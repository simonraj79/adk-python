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

import pytest
from google.adk.features import is_feature_enabled, FeatureName

def pytest_collection_modifyitems(items):
  """Skip tests in workflow/old if NEW_WORKFLOW feature is enabled."""
  try:
    if is_feature_enabled(FeatureName.NEW_WORKFLOW):
      skip_marker = pytest.mark.skip(reason="Disabled in NEW_WORKFLOW mode")
      for item in items:
        if "workflow/old" in str(item.fspath):
          item.add_marker(skip_marker)
  except Exception:
    # If import fails or something else goes wrong during collection,
    # don't break the test session.
    pass
