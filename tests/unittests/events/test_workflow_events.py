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

from google.adk.events.event import Event


def test_event_constructor_with_state():
  """Tests that the event constructor handles the state argument."""
  my_event = Event(state={"key": "value"})
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {"key": "value"}


def test_event_constructor_without_state():
  """Tests that the event constructor works without the state argument."""
  my_event = Event()
  assert my_event.actions is not None
  assert my_event.actions.state_delta == {}
