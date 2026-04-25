"""Level 1a — Voice variant of Level 1 (ADK 2.0 rewrite, native real-time audio).

Same shape as `level_1_agent`: one `LlmAgent`, one tool (`google_search`),
no planning, no multi-agent topology. The delta is that this agent is
configured to run on Gemini's **native-audio Live** model, so users can
speak to it via the mic icon in `adk web` and it speaks back with low-
latency, natural-prosody synthesised speech.

What changed v1 → v2 for voice
------------------------------
v1 pattern (see `V1_level_1a_agent/agent.py`):

    from google.adk.models.google_llm import Gemini
    from google.genai import types

    root_agent = Agent(
        model=Gemini(
            model="gemini-3.1-flash-live-preview",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Zephyr"
                    )
                )
            ),
        ),
        ...
    )

v2 pattern (this file): just a model id string. **No `Gemini(...)`
wrapper, no `speech_config` on the agent.** Voice and modality config
moved off the agent and onto `RunConfig`:

    # src/google/adk/agents/run_config.py:195
    speech_config: Optional[types.SpeechConfig] = None
    response_modalities: Optional[list[str]] = None
    output_audio_transcription: Optional[types.AudioTranscriptionConfig]
    input_audio_transcription: Optional[types.AudioTranscriptionConfig]
    realtime_input_config: Optional[types.RealtimeInputConfig] = None
    enable_affective_dialog: Optional[bool] = None
    proactivity: Optional[types.ProactivityConfig] = None
    session_resumption: Optional[types.SessionResumptionConfig] = None

Why the move? It separates "who the agent is" from "how it's heard right
now." The same agent can be invoked with Zephyr in one session and Kore
in another, with audio output on, audio output off, transcription on or
off — all without changing this file. In `adk web`, those switches live
in the **Live Flags** panel that opens when you click the mic. For
programmatic callers, construct a `RunConfig` and pass it to
`runner.run_live(run_config=...)`.

Why `gemini-3.1-flash-live-preview` (half-cascade), not native-audio
-------------------------------------------------------------------
v2's reference Live sample (`contributing/samples/live_bidi_streaming_single_agent/agent.py:67-71`)
documents two model choices:

    model='gemini-live-2.5-flash-native-audio',          # Vertex (default)
    # model='gemini-2.5-flash-native-audio-preview-12-2025',  # Gemini API

Note the Gemini-API native-audio line is **commented out** in the
upstream sample. Empirically, native-audio + `adk web`'s bidi audio
path on the **Gemini API** backend currently fails to negotiate audio
input — the Live connection establishes, the client loads
`audio-processor.js`, and then the model silently waits for audio it
never accepts (observed: ~19s of silence followed by client disconnect,
no server-side errors, no transcription). The same `google_search` +
native-audio combination works at the raw API layer using text input,
so the gap is in the audio-frame negotiation between adk-web's worklet
and the native-audio model on Gemini API specifically.

Half-cascade Live (`gemini-3.1-flash-live-preview`) is the v1-proven
working path on Gemini API + adk-web. It runs text generation through
the model, then a post-step TTS, so the audio negotiation is simpler
and adk-web handles it correctly today. Trade-off: prosody is less
natural than native-audio, and capabilities like
`RunConfig.enable_affective_dialog` / `RunConfig.proactivity` are
*native-audio-only* — they have no effect on this model.

To upgrade to native-audio later (when adk-web's Gemini-API audio path
fixes the issue, OR when running on Vertex AI):

    _LIVE_MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"  # Gemini API
    # or, on Vertex:
    _LIVE_MODEL_ID = "gemini-live-2.5-flash-native-audio"

Then flip on `enable_affective_dialog` and `proactivity` in adk-web's
Live Flags panel — no agent-file change needed (that's the v2 lesson:
voice/runtime config is RunConfig, not agent state).

What is genuinely "v2" in this file
-----------------------------------
1. **Model as a plain string** — no `Gemini(...)` wrapper. Voice config
   is RunConfig's job (see above).
2. **`output_key="last_answer"`** — v2 formalized state-delta flushing
   onto yielded events ("Supported flushing state/artifact deltas onto
   yielded events", `CHANGELOG-v2.md`). Even in audio mode, the
   agent's final text response is captured as a state delta on the
   event stream — visible in the web UI's State panel and in
   `Event.actions.state_delta` for downstream consumers (e.g. an
   eval harness or a transcript logger).
3. **Leaf-agent fast path** — same v2 optimisation as `level_1_agent`:
   for a single-tool, single-turn `LlmAgent`, the v2 runtime bypasses
   the Mesh and runs the leaf path directly.

How to run
----------
    adk web .                 # repo root (D:\\vscode\\adk-python)
    # → pick `level_1a_agent` in the picker
    # → click the mic icon (NOT the chat input — typing fails because
    #   bidi-only Live models do not support generateContent)
    # → speak; the agent answers with native synthesised speech
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.google_search_tool import google_search


# Gemini API half-cascade Live model. Currently the working path with
# `adk web` on the Gemini API backend (see file docstring for the
# native-audio caveat and upgrade path).
_LIVE_MODEL_ID = "gemini-3.1-flash-live-preview"


def _suppress_telemetry_only_responses(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
  """Suppress empty/telemetry-only events from the chat panel.

  Why this exists
  ---------------
  Gemini Live emits an LLM message PER audio chunk, each one carrying
  `usage_metadata` (token counting telemetry) — and, when session
  resumption is enabled in the Live Flags panel, also a steady stream
  of `live_session_resumption_update` messages (resume-token rotation).
  ADK's runtime emits an Event for each (`base_llm_flow.py:1023-1033`),
  even when there is no user-visible content. The `adk web` chat panel
  in v2.0.0b1 renders one bubble per Event — so a single voice turn
  can produce 200+ empty bubbles, drowning the actual transcription
  events.

  How this works
  --------------
  v2's after-model callback can return a *replacement* LlmResponse.
  If the replacement has none of the fields the runtime checks for
  event emission (`content`, `error_code`, `interrupted`,
  `turn_complete`, `input_transcription`, `output_transcription`,
  `usage_metadata`, `live_session_resumption_update`), the early-
  return at `base_llm_flow.py:1023-1033` skips creating the event
  entirely.

  We detect "telemetry-only" by checking the original response: if it
  has `usage_metadata` or `live_session_resumption_update` but NONE
  of the user-visible fields, we replace it with a bare `LlmResponse()`
  to suppress the event. Any response with content, transcription,
  errors, or turn_complete passes through unchanged.

  This is the most native v2 fix possible at the demo level — no
  framework patch, no plugin, just an after-model callback that uses
  the framework's own event-skip path.
  """
  has_visible = (
      llm_response.content
      or llm_response.error_code
      or llm_response.interrupted
      or llm_response.turn_complete
      or llm_response.input_transcription
      or llm_response.output_transcription
      or llm_response.grounding_metadata
  )
  if has_visible:
    # Pass through unchanged — None lets the original response be
    # used (per the after_model_callback contract).
    return None

  # Telemetry-only response: replace with an empty response that
  # triggers the event-skip early-return in base_llm_flow.
  return LlmResponse()


root_agent = Agent(
    name="level_1a_agent",
    model=_LIVE_MODEL_ID,
    description=(
        "A voice-native connected problem-solver that uses Google Search"
        " to answer questions requiring real-time information. Speak via"
        " the mic icon in adk web; the agent answers in synthesised"
        " speech. Voice and Live runtime knobs are set in the web UI's"
        " Live Flags panel (RunConfig), not on the agent."
    ),
    instruction="""You answer questions using Google Search over audio. One search per question, speak the answer directly.

# Spoken output discipline — your output is pronounced by TTS
- NO markdown, ever: asterisks, hashes, hyphen bullets, square brackets, pipes, backticks are all pronounced literally.
- NO section headers spoken aloud ("Key findings:", "Sources:", "In summary,").
- NO numbered citations ("Source 1", "(1, 2, 5)", "[1]"). Grounding metadata is for the UI, not the listener — weave attribution INLINE: "according to Reuters, ...".
- NEVER read a URL. Say the source domain as a human would: "according to Google for Education" / "according to Harvard Business Review".
- NO list enumeration ("firstly... secondly... thirdly..."). Use natural prose connectors.

# Answer shape
Give a 2 to 4 sentence spoken summary that fits in about 15 seconds of speech. Lead with the single most important finding, then 1 to 2 supporting points with inline domain attribution.

# Greetings and meta questions ("hi", "hello", "what can you do?")
Respond directly, without a search. One sentence: "I'm a voice search assistant — ask me anything that needs a Google Search, like the latest advances in solid-state batteries or the current exchange rate for the Swiss franc."

# Ambiguity
If the question is ambiguous, ask one short clarifying question before searching.
""",
    tools=[google_search],
    # v2: the agent's final text response is auto-written to
    # state['last_answer'] AND flushed as a state_delta on the yielded
    # Event. Useful in audio mode for transcript logging without having
    # to subscribe to output_audio_transcription separately.
    output_key="last_answer",
    # v2 telemetry-event suppression (see callback docstring above).
    # Without this, a single voice turn produces 200+ empty bubbles
    # in adk-web's chat panel because Gemini Live emits one
    # usage_metadata-only message per audio chunk, and ADK emits an
    # Event for each. The callback rewrites those telemetry-only
    # responses so the framework's event-skip early-return fires.
    after_model_callback=_suppress_telemetry_only_responses,
)
