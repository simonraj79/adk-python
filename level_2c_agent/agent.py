"""Level 2c — Dynamic Workflow + HITL Pause/Resume (ADK 2.0).

A second sibling to `level_2_agent`. Same capability tier (Level 2 —
Strategic Problem-Solver / context engineering via a graph), but the
new axis is **framework-enforced human-in-the-loop**: a `@node`-
decorated function pauses the workflow with `RequestInput` until a
human responds, and the framework checkpoints state so the workflow
can resume durably — even after `adk web` is killed and restarted.

Use case
--------
Refund processing with a manager-approval gate. The user submits a
refund request (customer ID, amount, reason). Under $100 the
workflow auto-approves and processes. **At or above $100** it pauses,
asks a manager 'yes' or 'no', then either issues the refund or
records the rejection.

Inspired by the `refund_approval` demo in the *ADK 2.0 launch* video.
The video's pedagogical message: in v1, "human-in-the-loop" was a
prompt-level convention you hoped the LLM honoured; in v2, the
framework guarantees it — the workflow physically cannot continue
past `RequestInput` until a real response comes back, and the state
is durably checkpointed so a server restart doesn't lose it. That
guarantee is what enables enterprise approval / compliance / fraud-
review patterns.

Why this is `level_2c_agent` and not `level_5_agent`
-----------------------------------------------------
The published Google taxonomy stops at L4. New ADK 2.0 *primitives*
are absorbed as `Na/Nb/Nc` variants of the closest capability tier;
they don't inflate the L0–L4 ladder. `RequestInput` is a new
*primitive*, but the underlying capability — workflow orchestration
with a deterministic gate — sits inside L2's strategic-planning tier.
Same rationale as L4a (MCP variant of L4) and L1a (voice variant
of L1). See `AGENT_LEVELS.md` for the full ladder explanation.

Difference vs L4's HITL
-----------------------
L4's `agent_creator` also has HITL — but that's *LLM-discretionary*:
the model in `mode='task'` decides to ask a clarifying question
(or not). L2c's HITL is *framework-enforced*: the gate function
returns a `RequestInput` value, and the runner physically halts the
workflow. The LLM has no role in choosing whether to pause; the rule
is in code (`if amount >= THRESHOLD`). Different guarantees.

ADK 2.0 features in use (genuine — not feature theatre)
-------------------------------------------------------
- `RequestInput(message=, response_schema=, payload=)` from
  `google.adk.events.request_input` — the framework's HITL primitive.
- `@node(rerun_on_resume=True)` on the gate — when the user responds,
  the gate **re-executes** with `ctx.resume_inputs` populated,
  letting the same function handle both first-run (ask) and
  post-resume (decide) paths. Without `rerun_on_resume=True`, the
  user's response would simply *become* the node's output, requiring
  a follow-up node to interpret it.
- `App(root_agent=..., resumability_config=ResumabilityConfig(
  is_resumable=True))` — durable checkpointing. Per
  `references/human-in-the-loop.md`, "the agent loader checks for
  `app` before `root_agent`, so export both from `agent.py`."
- `output_schema=Pydantic` on the intake LLM agent → typed parse of
  the user's natural-language refund request into a `RefundRequest`.
- `output_key` to persist the parsed request in state so downstream
  function nodes can read via parameter-name resolution
  (`def gate(refund_request: dict)` ← reads from `ctx.state`).
- `Event(state={...}, output=...)` to both update state AND pass the
  decision down the graph in one go — the canonical v2 dual-purpose
  event from the workflow skill.

Graph
-----

    START
      ↓
    process_input    (function — stash raw text)
      ↓
    intake           (LlmAgent — parse → RefundRequest)
      ↓
    gate             (@node async, rerun_on_resume=True)
      │ if amount < $100 → emit RefundDecision(approved=True), continue
      │ if amount ≥ $100 first run → yield RequestInput, PAUSE
      │ if amount ≥ $100 resumed   → emit RefundDecision based on response, continue
      ↓
    process_refund   (@node async — issue refund OR record rejection)
      ↓
      END

Sample queries
--------------
- "Process a $50 refund for customer C-001 — wrong size shipped."
    → auto-approve path, no pause
- "Refund $350 to customer C-002 — defective laptop returned."
    → pause for manager approval; reply 'yes' → process, 'no' → reject
- "I'm customer C-077 and I want $99.99 back, the item never arrived."
    → auto-approve (under threshold)

Run
---
    adk web .
    # → pick `level_2c_agent` in the picker
"""

from __future__ import annotations

import secrets

from google.adk import Agent
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.request_input import RequestInput
from google.adk.workflow import node
from google.genai import types
from pydantic import BaseModel
from pydantic import Field


# Manager-approval threshold. Anything strictly under this auto-approves;
# anything at or above pauses for human review. Match the video demo.
_APPROVAL_THRESHOLD_USD = 100.0


# ---------------------------------------------------------------------------
# Schemas — typed contracts at every boundary.
# ---------------------------------------------------------------------------


class RefundRequest(BaseModel):
  """Parsed refund request — produced by the `intake` LLM agent."""

  customer_id: str = Field(
      description=(
          "Customer identifier as the user wrote it (e.g., 'C-001',"
          " 'customer_42'). If the user didn't provide one, set this"
          " to 'UNKNOWN' — do not invent."
      )
  )
  amount_usd: float = Field(
      description=(
          "Refund amount in US dollars as a float. Strip currency"
          " symbols and commas. If the user gave a range or no amount,"
          " set to 0.0 — the gate will catch the malformed case."
      )
  )
  reason: str = Field(
      description=(
          "Short plain-text reason for the refund (e.g., 'defective"
          " laptop returned', 'wrong size shipped', 'never arrived')."
          " 1 sentence."
      )
  )


# ---------------------------------------------------------------------------
# Function nodes — the workflow's deterministic glue.
# ---------------------------------------------------------------------------


def process_input(node_input: types.Content):
  """Stash the user's raw refund request in state for the intake LLM.

  Same idiom as L2 / L2b — function node at START captures
  `node_input` (which is `types.Content` for the very first node) and
  stores the text in state.
  """
  text = node_input.parts[0].text if node_input.parts else ""
  return Event(state={"request": text})


@node(rerun_on_resume=True)
async def gate(ctx: Context, refund_request: dict):
  """Auto-approve under threshold, otherwise pause for manager approval.

  Decorated with `@node(rerun_on_resume=True)` so that on the user's
  reply the framework **re-executes this function** with
  `ctx.resume_inputs` populated, instead of treating the user's reply
  as the node's output. That lets the same function handle both
  first-run (ask) and post-resume (decide) cases without a separate
  follow-up node.

  Parameter resolution (per `function-nodes.md`):
    - `ctx`             → the Workflow Context (special name)
    - `refund_request`  → auto-injected from `ctx.state['refund_request']`
                          (the intake agent stored it there via output_key)

  Output: emits an Event whose `output` is a `RefundDecision`-shaped
  dict, plus a `state` update so `process_refund` can read either the
  emitted output OR the state copy by parameter name.
  """
  amount = float(refund_request.get("amount_usd", 0.0))
  customer = refund_request.get("customer_id", "UNKNOWN")
  reason = refund_request.get("reason", "")

  # ---- Path 1: auto-approve (strictly under threshold) -------------
  if amount < _APPROVAL_THRESHOLD_USD:
    decision = {
        "approved": True,
        "approval_path": "auto",
        "notes": (
            f"Auto-approved: ${amount:.2f} is under the"
            f" ${_APPROVAL_THRESHOLD_USD:.0f} manager-approval threshold."
        ),
    }
    yield Event(state={"decision": decision}, output=decision)
    return

  # ---- Path 2: at or above threshold — needs a manager. -----------
  # On the SECOND execution (after resume) ctx.resume_inputs is
  # populated. On the FIRST execution it's empty → emit RequestInput
  # and the framework pauses the workflow.
  if ctx.resume_inputs:
    # The user's reply arrived. With response_schema={"type":"string"}
    # the response is the raw text the user typed.
    response = list(ctx.resume_inputs.values())[0]
    response_text = str(response).strip().lower()
    approved = response_text in ("yes", "y", "approve", "approved", "ok")
    decision = {
        "approved": approved,
        "approval_path": "manager",
        "notes": (
            f"Manager {'approved' if approved else 'rejected'} the refund"
            f" of ${amount:.2f} for customer {customer}."
            f" (Response received: {response!r}.)"
        ),
    }
    yield Event(state={"decision": decision}, output=decision)
    return

  # First execution at/above threshold — no response yet, ASK.
  yield RequestInput(
      interrupt_id="manager_approval",
      message=(
          f"⚠️ Manager approval required.\n\n"
          f"Refund of **${amount:.2f}** for customer **{customer}**\n"
          f"Reason: {reason}\n\n"
          f"This is at or above the ${_APPROVAL_THRESHOLD_USD:.0f}"
          f" threshold. Reply 'yes' to approve or 'no' to reject."
      ),
      payload={
          "customer_id": customer,
          "amount_usd": amount,
          "reason": reason,
      },
      response_schema={"type": "string"},
  )


@node
async def process_refund(decision: dict, refund_request: dict):
  """Final step: issue the refund or record the rejection.

  Both `decision` and `refund_request` resolve from `ctx.state` via
  parameter-name injection — the gate stored `decision` in state, and
  the intake agent stored `refund_request` there earlier.

  In a real system this would call a payment processor; here we mint
  a fake confirmation reference so the demo's output is concrete.
  """
  customer = refund_request["customer_id"]
  amount = float(refund_request["amount_usd"])
  approved = bool(decision["approved"])
  notes = decision["notes"]
  approval_path = decision["approval_path"]

  if not approved:
    summary = (
        f"### Refund rejected\n\n"
        f"- **Customer**: {customer}\n"
        f"- **Amount**: ${amount:.2f}\n"
        f"- **Reason for refund request**: {refund_request.get('reason', '')}\n"
        f"- **Decision**: rejected by manager\n"
        f"- **Notes**: {notes}\n\n"
        f"No funds were transferred."
    )
    yield Event(message=summary, state={"final_summary": summary})
    return

  confirmation = f"RFND-{secrets.token_hex(4).upper()}"
  summary = (
      f"### Refund processed\n\n"
      f"- **Customer**: {customer}\n"
      f"- **Amount**: ${amount:.2f}\n"
      f"- **Confirmation**: `{confirmation}`\n"
      f"- **Approval path**: {approval_path} ({notes})\n"
      f"- **Reason for refund**: {refund_request.get('reason', '')}\n\n"
      f"Funds will appear on the customer's statement within 3-5 business days."
  )
  yield Event(
      message=summary,
      state={"final_summary": summary, "confirmation": confirmation},
  )


# ---------------------------------------------------------------------------
# Intake LLM agent — natural-language → RefundRequest.
# ---------------------------------------------------------------------------


intake_agent = Agent(
    name="intake",
    model="gemini-2.5-flash",
    description=(
        "Parses a free-text refund request into structured fields"
        " (customer_id, amount_usd, reason)."
    ),
    instruction=(
        'The user said: "{request}"\n\n'
        "Parse this refund request into the structured RefundRequest"
        " output. Rules:\n"
        "- amount_usd: strip $ / commas, return a float. If the user"
        " mentions multiple amounts, use the one being refunded (not"
        " the original purchase price). If no amount is given, return"
        " 0.0 — do not guess.\n"
        "- customer_id: take any explicit ID ('C-001', 'customer 42',"
        " 'account #99'). Normalise minor differences (e.g. drop"
        " 'customer ' prefix). If nothing is given, return 'UNKNOWN'.\n"
        "- reason: condense to 1 sentence. If the user gave no reason,"
        " return 'no reason given'.\n\n"
        "Do not respond conversationally; emit only the structured"
        " output. Greetings and unrelated input should still be parsed"
        " as best-effort with amount_usd=0.0 — the workflow's gate"
        " handles the malformed case."
    ),
    output_schema=RefundRequest,
    output_key="refund_request",
)


# ---------------------------------------------------------------------------
# Workflow — explicit edges, four nodes.
# ---------------------------------------------------------------------------


root_agent = Workflow(
    name="level_2c_agent",
    description=(
        "Refund-processing workflow with a framework-enforced manager-"
        "approval gate. Under $100 auto-approves; at or above $100"
        " pauses via RequestInput until a manager replies. State is"
        " durably checkpointed (App + ResumabilityConfig) so the"
        " workflow survives a restart while waiting for a human."
    ),
    edges=[
        ("START", process_input, intake_agent, gate, process_refund),
    ],
)


# Resumable App wrapper — required for proper checkpointing across a
# server restart. The agent loader picks `app` over `root_agent` when
# both are exported, so this is the actual runtime entry point in
# `adk web`. (Reference: `human-in-the-loop.md` §"Resumability
# Configuration" — "Required for multi-step HITL... and complex
# workflows".)
app = App(
    name="level_2c_agent",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
