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

"""Task-aware content processor for the workflow runtime.

Universal replacement for ``contents.request_processor`` in the workflow
processor pipeline. When no task delegations are present, behaves
identically to the standard content processor. When task delegations
exist, provides strict branch filtering for task/single_turn agents
(exclude parent history) and for parent agents (exclude task internals,
construct FC->FR view).
"""

from __future__ import annotations

import copy
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.genai import types
from typing_extensions import override

from ....events.event import Event
from ....models.llm_request import LlmRequest
from ...invocation_context import InvocationContext
from .._base_llm_processor import BaseLlmRequestProcessor
from .._contents import _add_instructions_to_user_content
from .._contents import _contains_empty_content
from .._contents import _get_contents
from .._contents import _get_current_turn_contents
from .._contents import _is_adk_framework_event
from .._contents import _is_auth_event
from .._contents import _is_request_confirmation_event
from .._contents import _is_request_input_event
from .._functions import remove_client_function_call_id
from ._request_task_tool import render_task_input
from ._task_models import _as_task_request
from ._task_models import TaskResult

logger = logging.getLogger('google_adk.' + __name__)


def _is_event_visible_to_task_agent(
    task_branch: str,
    event: Event,
) -> bool:
  """Filter for task/single_turn agents (branch-matched events only).

  Only includes events from THIS specific delegation. Excludes parent
  events (branch=None) and events from other delegations.

  For user events (branch=None) that appear during a multi-turn task
  delegation, use ``_filter_events_for_task_agent`` which handles
  positional inclusion.

  Args:
    task_branch: The branch assigned to this delegation.
    event: The event to check.

  Returns:
    True if the event should be visible to the task agent.
  """
  if not event.branch:
    return False
  return event.branch == task_branch or event.branch.startswith(
      f'{task_branch}.'
  )


def _filter_events_for_task_agent(
    task_branch: str,
    task_branches: set[str],
    events: list[Event],
) -> list[Event]:
  """Build the filtered event list for a task/single_turn agent.

  Includes events matching the task branch, plus user events
  (author='user', branch=None) that appear positionally AFTER the
  most recent ``request_task`` event targeting this agent. This avoids
  mutating event.branch in the _Mesh while still making user replies
  visible to the task agent during multi-turn delegations.

  Args:
    task_branch: The branch for this task delegation.
    task_branches: All known task branches.
    events: The full session event list.

  Returns:
    Filtered event list for the task agent's LLM context.
  """
  # Find the index of the request_task event that created this branch.
  # Events after this index with author='user' and branch=None are
  # user replies to the task agent.
  delegation_start = -1
  for i, event in enumerate(events):
    if not event.actions:
      continue
    if event.actions.request_task:
      for fc_id in event.actions.request_task:
        # The branch format is task:{node_path}.{agent_name}.{fc_id}.
        # Check if the fc_id matches the branch suffix.
        if task_branch.endswith(f'.{fc_id}'):
          delegation_start = i
          break

  result = []
  for i, event in enumerate(events):
    if _should_include_event_task_aware(task_branch, task_branches, event):
      result.append(event)
    elif (
        i > delegation_start >= 0
        and event.author == 'user'
        and not event.branch
        and not _contains_empty_content(event)
    ):
      result.append(event)

  return result


def _is_event_visible_to_parent(
    task_branches: set[str],
    event: Event,
) -> bool:
  """Filter for parent agents viewing task delegations.

  Excludes internal events from task agent branches. The parent should
  only see the collapsed FC->FR pair (constructed separately), not the
  task agent's internal tool calls and LLM responses.

  Args:
    task_branches: Set of all task delegation branch strings.
    event: The event to check.

  Returns:
    True if the event should be visible to the parent agent.
  """
  if not event.branch:
    return True
  # Exclude events whose branch matches any task delegation branch.
  for tb in task_branches:
    if event.branch == tb or event.branch.startswith(f'{tb}.'):
      return False
  return True


def _collect_task_branches(
    events: list[Event],
) -> set[str]:
  """Collect all task delegation branch strings from session events.

  Task branches use a ``task:`` prefix, so a single pass over events
  is sufficient to identify them.

  Args:
    events: The session events to scan.

  Returns:
    Set of branch strings used by task delegations.
  """
  branches: set[str] = set()
  for event in events:
    if event.branch and event.branch.startswith('task:'):
      branches.add(event.branch)
  return branches


def _should_include_event_task_aware(
    invocation_branch: Optional[str],
    task_branches: set[str],
    event: Event,
) -> bool:
  """Task-aware event filter replacing the standard branch check.

  Args:
    invocation_branch: The current agent's branch (None for parent).
    task_branches: Set of task delegation branches.
    event: The event to check.

  Returns:
    True if the event should be included in the LLM context.
  """
  # Apply standard non-branch filters first.
  if _contains_empty_content(event):
    return False
  if _is_adk_framework_event(event):
    return False
  if _is_auth_event(event):
    return False
  if _is_request_confirmation_event(event):
    return False
  if _is_request_input_event(event):
    return False

  # Apply task-aware branch filtering.
  if invocation_branch:
    # Task/single_turn agent: strict isolation.
    return _is_event_visible_to_task_agent(invocation_branch, event)
  else:
    # Parent agent: exclude task internals.
    if task_branches:
      return _is_event_visible_to_parent(task_branches, event)
    return True


def _find_task_input_text(
    branch: str,
    events: list[Event],
    *,
    is_single_turn: bool,
) -> Optional[str]:
  """Find and render the task input for a delegated agent.

  Extracts the function-call ID from the branch suffix, looks up the
  matching ``request_task`` action in session events, and renders the
  task input as a human-readable string.

  Args:
    branch: The task branch string (e.g. ``task:path.agent.fc_id``).
    events: The session events to scan.
    is_single_turn: Whether the target agent is single-turn.

  Returns:
    Rendered task input string, or None if no matching request
    was found.
  """
  fc_id = branch.rsplit('.', 1)[-1]
  for event in reversed(events):
    if not event.actions.request_task:
      continue
    if fc_id in event.actions.request_task:
      req = _as_task_request(event.actions.request_task[fc_id])
      return render_task_input(req.input, is_single_turn)
  return None


class _TaskContentLlmRequestProcessor(BaseLlmRequestProcessor):
  """Task-aware content processor.

  Delegates to the standard ``_get_contents`` for the base content
  assembly, but uses task-aware branch filtering. For parent agents
  with task delegations, constructs FC->FR Content pairs from the
  factual request_task/finish_task events.
  """

  @override
  async def run_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    from ....models.google_llm import Gemini

    agent = invocation_context.agent
    preserve_function_call_ids = False
    if hasattr(agent, 'canonical_model'):
      canonical_model = agent.canonical_model
      preserve_function_call_ids = (
          isinstance(canonical_model, Gemini)
          and canonical_model.use_interactions_api
      )

    instruction_related_contents = llm_request.contents

    branch = invocation_context.branch
    events = invocation_context.session.events

    # Honor include_contents='none' (current turn only, no history).
    if agent.include_contents != 'default':
      llm_request.contents = _get_current_turn_contents(
          branch,
          events,
          agent.name,
          preserve_function_call_ids=preserve_function_call_ids,
      )
      await _add_instructions_to_user_content(
          invocation_context, llm_request, instruction_related_contents
      )
      return

    task_branches = _collect_task_branches(events)

    if branch:
      # Task/single_turn agent: filter events to this branch, plus
      # user events that appear after the delegation started.
      filtered_events = _filter_events_for_task_agent(
          branch, task_branches, events
      )
      llm_request.contents = _get_contents(
          branch,
          filtered_events,
          agent.name,
          preserve_function_call_ids=preserve_function_call_ids,
      )
      # Prepend task input so delegated agents see it as the first
      # user message. Looks up the originating request_task event;
      # falls back to user_content from the invocation context.
      if branch.startswith('task:'):
        is_single_turn = getattr(agent, 'mode', 'task') == 'single_turn'
        trigger_text = _find_task_input_text(
            branch, events, is_single_turn=is_single_turn
        )
        if trigger_text:
          trigger = types.Content(
              role='user',
              parts=[types.Part(text=trigger_text)],
          )
        elif invocation_context.user_content:
          trigger = invocation_context.user_content
        else:
          trigger = types.Content(
              role='user',
              parts=[types.Part(text='[Delegated task]')],
          )
        llm_request.contents = [trigger] + list(llm_request.contents or [])
    elif task_branches:
      # Parent agent with task delegations: exclude task internals
      # and construct FC->FR pairs for completed delegations.
      filtered_events = [
          e
          for e in events
          if _should_include_event_task_aware(None, task_branches, e)
      ]
      # Force preserve IDs so FR replacement can match by fc_id.
      llm_request.contents = _get_contents(
          None,
          filtered_events,
          agent.name,
          preserve_function_call_ids=True,
      )

      # Replace placeholder FRs for completed delegations with
      # actual task results so the coordinator sees real outputs.
      _replace_task_result_responses(events, llm_request.contents)

      # Strip client-generated IDs if the model doesn't need them.
      if not preserve_function_call_ids:
        for content in llm_request.contents:
          remove_client_function_call_id(content)
    else:
      # No task delegation: use standard content assembly.
      llm_request.contents = _get_contents(
          branch,
          events,
          agent.name,
          preserve_function_call_ids=preserve_function_call_ids,
      )

    await _add_instructions_to_user_content(
        invocation_context, llm_request, instruction_related_contents
    )

    if False:
      yield


def _replace_task_result_responses(
    events: list[Event],
    contents: list[types.Content],
) -> None:
  """Replace placeholder FRs for completed delegations with task results.

  Scans session events for request_task/finish_task pairs. For each
  completed delegation, finds the matching FR part in ``contents``
  (by ``fc_id``) and replaces the placeholder response with the
  actual ``TaskResult.output``. Requires ``preserve_function_call_ids=True``
  when building contents so IDs are available for matching.

  Args:
    events: The session events to scan.
    contents: The content list to modify in place.
  """
  # Build a map of fc_id -> task result output for completed delegations.
  pending: dict[str, str] = {}  # fc_id -> agent_name
  result_by_fc_id: dict[str, Any] = {}

  for event in events:
    if event.actions.request_task:
      for fc_id, req_value in event.actions.request_task.items():
        req = _as_task_request(req_value)
        pending[fc_id] = req.agent_name

    # Detect task completion: finish_task (task mode) or event.output
    # (single_turn controlled generation). Single_turn agents may
    # yield multiple events with event.output (e.g. routing events
    # from call_llm). Keep overwriting so the last output wins.
    task_output = None
    if event.actions.finish_task and pending:
      result = TaskResult.model_validate(event.actions.finish_task)
      task_output = result.output
    elif event.output is not None and pending:
      task_output = event.output

    if task_output is not None:
      # Match completion to pending delegation by branch suffix.
      # Each delegation has a unique branch ending with .{fc_id}.
      matched_fc_id = None
      if event.branch:
        for fc_id in pending:
          if event.branch.endswith(f'.{fc_id}'):
            matched_fc_id = fc_id
            break
      # Fallback: match by author name (single delegation case).
      if not matched_fc_id:
        for cid, aname in pending.items():
          if event.author == aname:
            matched_fc_id = cid
      if not matched_fc_id and pending:
        matched_fc_id = list(pending.keys())[-1]

      if matched_fc_id:
        # Overwrite (not delete) so later outputs on the same
        # branch replace earlier ones (e.g. routing events).
        result_by_fc_id[matched_fc_id] = copy.deepcopy(task_output)

  if not result_by_fc_id:
    return

  # Walk contents and replace matching FR parts in place.
  for content in contents:
    if not content.parts:
      continue
    for i, part in enumerate(content.parts):
      if not part.function_response:
        continue
      fr_id = part.function_response.id
      if fr_id in result_by_fc_id:
        result_value = result_by_fc_id.pop(fr_id)
        # FunctionResponse.response must be a dict.
        if not isinstance(result_value, dict):
          result_value = {'result': result_value}
        content.parts[i] = types.Part(
            function_response=types.FunctionResponse(
                name=part.function_response.name,
                response=result_value,
                id=fr_id,
            )
        )
        if not result_by_fc_id:
          return


request_processor = _TaskContentLlmRequestProcessor()
