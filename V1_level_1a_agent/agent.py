"""Level 1a — Voice variant of Level 1.

Identical structure to `level_1_agent`: a single LlmAgent with the
`google_search` built-in tool. The only delta is that the model is wired
as a Gemini Live model (`Gemini(model=..., speech_config=...)`) so the
agent can be spoken to via the mic button in adk web and responds with
synthesised speech.

Why L1a, not L3a:
  L3a (the multi-agent voice variant) was removed. Its `sub_agents` →
  `transfer_to_agent` path required the model to emit a structured
  function call in bidi audio mode, which `gemini-3.1-flash-live-preview`
  does not reliably do (observed symptom: the coordinator would greet
  successfully but silently freeze on delegation turns). Single-agent
  Live — L1a — sidesteps that entirely: `google_search` is a model
  built-in tool (not a user-defined function tool), so the model handles
  it internally rather than emitting a transfer call. No sub_agents
  means no `transfer_to_agent` auto-injection means no function-call
  capability required in audio mode.

Model: `gemini-3.1-flash-live-preview` by default. This is a bidi-only
(voice-only) Live model on the Gemini API. Typing text fails with HTTP
404 "is not supported for generateContent" — click the mic in adk web.

Voice: single agent, single voice. We use Zephyr by default — change
`_VOICE_NAME` below to pick from Kore, Puck, Aoede, Charon, Fenrir,
Leda, Orus, or any other prebuilt Gemini Live voice.

Live Flags: `RunConfig.enable_affective_dialog`, `RunConfig.proactivity`,
`RunConfig.session_resumption`, and `RunConfig.save_live_blob` are
set by adk web's "Live Flags" panel, not by this file. Programmatic
callers should construct their own `RunConfig` — see level_3a-era
docstring history in git for a copy-pasteable template (the pattern
is identical for a single-agent Live runner).
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types


_LIVE_MODEL_ID = "gemini-3.1-flash-live-preview"
# Alternatives verified available on Gemini API (run
# `client.models.list()` filtered by `bidiGenerateContent` in
# supported_actions to regenerate this list):
# _LIVE_MODEL_ID = "gemini-2.5-flash-native-audio-latest"            # native-audio, more natural prosody, occasionally 1011s
# _LIVE_MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"   # pinned native-audio

_VOICE_NAME = "Zephyr"  # Kore, Puck, Aoede, Charon, Fenrir, Leda, Orus, ...


root_agent = Agent(
    name="level_1a_agent",
    model=Gemini(
        model=_LIVE_MODEL_ID,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=_VOICE_NAME
                )
            )
        ),
    ),
    description=(
        "A voice-native connected problem-solver that uses Google Search"
        " to answer questions requiring real-time information. Speak to"
        " it via the mic button; it speaks back with the prebuilt"
        f" {_VOICE_NAME} voice."
    ),
    instruction="""You answer questions using Google Search over audio. One search per question, speak the answer directly.

# Spoken output discipline — your output is pronounced by TTS
- NO markdown, ever: asterisks, hashes, hyphen bullets, square brackets, pipes, backticks are all pronounced literally.
- NO section headers spoken aloud ("Key findings:", "Sources:", "In summary,").
- NO numbered citations ("Source 1", "(1, 2, 5)", "[1]"). Grounding metadata is for the UI, not the listener — weave attribution INLINE: "according to Reuters, ...".
- NEVER read a URL. Say the source domain as a human would: "according to Google for Education" / "according to Harvard Business Review".
- NO list enumeration ("firstly... secondly... thirdly..."). Use natural prose connectors.

# Answer shape
Give a 2–4 sentence spoken summary that fits in about 15 seconds of speech. Lead with the single most important finding, then 1–2 supporting points with inline domain attribution.

# Greetings and meta questions ("hi", "hello", "what can you do?")
Respond directly, without a search. One sentence: "I'm a voice search assistant — ask me anything that needs a Google Search, like the latest advances in solid-state batteries or the current exchange rate for the Swiss franc."

# Ambiguity
If the question is ambiguous, ask one short clarifying question before searching.
""",
    tools=[google_search],
)
