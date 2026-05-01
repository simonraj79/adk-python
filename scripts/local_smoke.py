"""Local smoke probe — InMemoryRunner against any A2A agent's root_agent.

Usage:
    python scripts/local_smoke.py level_1_agent  "Search for X."
    python scripts/local_smoke.py a2a_orchestrator "Without consulting any tool, just say hello."

Pass criteria:
- Elapsed < 90s (>90s on a code-executor probe = gotcha #21 hang).
- Non-empty text output.
- No "404 NOT_FOUND" / "Publisher Model" strings (would indicate the
  Pro 2.5 + us-central1 pairing isn't working for the deployed Vertex
  call path).

Required environment:
    GOOGLE_GENAI_USE_VERTEXAI=TRUE
    GOOGLE_CLOUD_PROJECT=gcp-cits-ccat-poc-d4d2
    GOOGLE_CLOUD_LOCATION=us-central1
    SMITHERY_API_KEY=<...>   # only level_4_agent uses it
"""
from __future__ import annotations

# Inject truststore BEFORE any Google import so all Python TLS goes through
# the Windows trust store (NTU's network injects a corporate cert that
# certifi's bundle doesn't trust).
import truststore
truststore.inject_into_ssl()

import asyncio
import importlib
import os
import sys
import time

# Add the parent dir (repo root) to sys.path so we can import sibling
# packages like level_1_agent. Script is in scripts/, packages are at root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.adk.runners import InMemoryRunner
from google.genai import types


HANG_THRESHOLD_S = 90.0
HARD_TIMEOUT_S = 180.0


async def run_probe(module_name: str, prompt: str) -> None:
    mod = importlib.import_module(f"{module_name}.agent")
    runner = InMemoryRunner(agent=mod.root_agent, app_name=mod.root_agent.name)
    session = await runner.session_service.create_session(
        app_name=mod.root_agent.name,
        user_id="smoke",
    )

    print(f"[{module_name}] probe: {prompt!r}")
    start = time.time()
    chunks: list[str] = []
    fn_calls: list[str] = []

    async for ev in runner.run_async(
        user_id="smoke",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        if time.time() - start > HARD_TIMEOUT_S:
            raise TimeoutError(
                f"{module_name} > {HARD_TIMEOUT_S}s — likely gotcha #21 hang"
            )
        if ev.content:
            for p in ev.content.parts or []:
                if getattr(p, "text", None):
                    chunks.append(p.text)
                if getattr(p, "function_call", None):
                    fn_calls.append(p.function_call.name)

    elapsed = time.time() - start
    text = "".join(chunks).strip()

    if "404 NOT_FOUND" in text or "Publisher Model" in text:
        raise RuntimeError(f"{module_name} model-region 404 in response:\n{text[:400]}")

    flag = "⚠ SLOW" if elapsed > HANG_THRESHOLD_S else "OK"
    print(f"[{module_name}] {flag} elapsed={elapsed:.1f}s len={len(text)} fn_calls={fn_calls}")
    print(f"--- output (first 400 chars) ---\n{text[:400]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    asyncio.run(run_probe(sys.argv[1], sys.argv[2]))
