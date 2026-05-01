"""Deploy any ADK root_agent to Vertex Agent Engine with A2A enabled.

Usage (run from D:\\vscode\\adk-python with the upstream venv active):
    python deploy_a2a.py level_1_agent  --display "Level 1 (A2A)"
    python deploy_a2a.py level_2_agent  --display "Level 2 Day Planner (A2A)"
    python deploy_a2a.py level_2b_agent --display "Level 2b Graph Router (A2A)"
    python deploy_a2a.py level_3_agent  --display "Level 3 Research Coordinator (A2A)"
    python deploy_a2a.py level_4_agent  --display "Level 4 Self-Evolving BI (A2A)"

Auto-forwarded env vars: if `SMITHERY_API_KEY` is set in the deploy shell,
it is automatically baked into the deployed engine's container env so any
gahmen-mcp wiring inside the agent (e.g. level_4_agent's data_fetcher_agent)
becomes active. Set it with:
    $env:SMITHERY_API_KEY = "<your key>"

Requires:
    Python >= 3.10
    google-adk[a2a] >= 2.0.0b1
    google-cloud-aiplatform[agent_engines] >= 1.130.0

After each deploy, capture the printed resource ID and add it to the
A2A engine registry at:
    D:\\vscode\\nbs-market-intelligence-swarm (1)\\adk\\tests\\integration\\a2a_engines.py

See ../nbs-market-intelligence-swarm (1)/new features/14-vertex-a2a-migration-plan.md §5
for the source-verified rationale behind each line.
"""
from __future__ import annotations

# truststore must be injected BEFORE any HTTPS-using import (vertexai,
# google-auth, requests, etc.) so all SSL goes through the Windows trust
# store. Required on NTU's network where TLS inspection injects a
# corporate root CA that's not in certifi's bundle.
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except ImportError:
    # truststore is optional — required only when running against a
    # network that does TLS inspection (e.g., NTU). On other networks
    # certifi handles validation fine.
    pass

import argparse
import importlib
import os

import vertexai
from a2a.types import AgentSkill
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import (
    include_artifacts_in_a2a_event_interceptor,
)
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import A2aAgent
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

PROJECT = "gcp-cits-ccat-poc-d4d2"
APPSPOT_SA = f"{PROJECT}@appspot.gserviceaccount.com"

# Env vars auto-forwarded into the deployed engine's runtime container if
# present in the deploy shell. Lifted from main()'s body to a module
# constant so tests can import + assert against it (W9.2 §6.3.5).
AUTO_FORWARD_ENV_VARS = (
    # gahmen-mcp toolset (level_4_agent's data_fetcher_agent reads at import).
    "SMITHERY_API_KEY",
    "SMITHERY_GAHMEN_URL",
    # Per-Level A2A peer routing (orchestrator + level_4 consume these).
    # All Levels live in us-central1 post-W9.2 (was asia-southeast1).
    "LEVEL_1_A2A_ENGINE_ID", "LEVEL_1_A2A_REGION",
    "LEVEL_2_A2A_ENGINE_ID", "LEVEL_2_A2A_REGION",
    "LEVEL_2B_A2A_ENGINE_ID", "LEVEL_2B_A2A_REGION",
    "LEVEL_3_A2A_ENGINE_ID", "LEVEL_3_A2A_REGION",
    "LEVEL_4_A2A_ENGINE_ID", "LEVEL_4_A2A_REGION",
    # Generic project / region overrides (orchestrator's remote_tools reads
    # LEVEL_REGION as the cross-Level default).
    "LEVEL_PROJECT_NUMBER",
    "LEVEL_REGION",
)


def _executor_builder(root_agent):
    """Return a no-arg callable that builds a fresh A2aAgentExecutor.

    Vertex's A2aAgent.set_up() invokes the builder as
    `agent_executor_builder(**agent_executor_kwargs)`. We pass no kwargs,
    so the builder is called as `_build()` — must take no positional or
    required keyword args. `root_agent` is captured via closure.

    Source: vertexai/preview/reasoning_engines/templates/a2a.py:255-257.
    """
    def _build() -> A2aAgentExecutor:
        runner = Runner(
            app_name=root_agent.name,
            agent=root_agent,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService(),
        )
        # W9.3 (2026-05-01): register the include_artifacts interceptor so
        # session artifacts produced by BuiltInCodeExecutor (matplotlib charts
        # from chart_agent / analyst_agent) get packaged into the A2A
        # response's TaskArtifactUpdateEvents — visible to A2A callers as
        # FileParts in Task.artifacts. Without this, charts live and die in
        # the engine's session and never reach the orchestrator/bot.
        # Default config has execute_interceptors=None; we explicitly set
        # the artifact passthrough one. Source: ADK 2.0
        # google/adk/a2a/executor/interceptors/include_artifacts_in_a2a_event.py.
        config = A2aAgentExecutorConfig(
            execute_interceptors=[include_artifacts_in_a2a_event_interceptor],
        )
        return A2aAgentExecutor(runner=runner, config=config)
    return _build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Agent package, e.g. level_1_agent")
    parser.add_argument("--display", required=True, help='Display name, e.g. "Level 1 (A2A)"')
    # W9.2 default: us-central1 (Pro 2.5 lives here; was asia-southeast1).
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--service_account",
        default=APPSPOT_SA,
        help=(
            "Runtime SA for the deployed engine. Default: appspot SA "
            "(the only enabled SA on this project with roles/editor). "
            "Without this, Vertex assigns the default Reasoning Engine "
            "Service Agent — which lacks aiplatform.user, so any "
            "engine→engine agent_engines.get() call 403s. See W9.2 plan §3."
        ),
    )
    parser.add_argument(
        "--description",
        default="ADK agent exposed via A2A on Vertex Agent Engine.",
    )
    parser.add_argument(
        "--staging_bucket",
        default=f"gs://{PROJECT}-vertex-staging",
        help=(
            "GCS bucket for extra_packages upload. The Vertex agent_engines.create() "
            "API (which we use because it accepts an A2aAgent instance directly) "
            "requires this. Default: gs://<project>-vertex-staging. "
            "If the bucket does not exist, create once with: "
            "gsutil mb -p <project> -l <region> gs://<project>-vertex-staging"
        ),
    )
    args = parser.parse_args()

    # Auto-load <module>/.env BEFORE importing the agent — same convention
    # as `adk web` / `adk run`. This means SMITHERY_API_KEY (and any other
    # per-agent secrets) defined in the agent's .env will be present in
    # os.environ when (a) the agent module loads (so McpToolset gets wired)
    # and (b) the env-var auto-forward block below runs (so the values get
    # baked into the deployed engine's container).
    _env_path = os.path.join(args.module, ".env")
    if os.path.exists(_env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(_env_path)
            print(f"Loaded .env from {_env_path}")
        except ImportError:
            # python-dotenv comes in via google-adk's deps; if missing,
            # do a minimal hand-parse (KEY=VALUE per line, no quoting).
            with open(_env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            print(f"Loaded .env from {_env_path} (manual parse)")

    vertexai.init(
        project=PROJECT,
        location=args.region,
        staging_bucket=args.staging_bucket,
    )

    mod = importlib.import_module(f"{args.module}.agent")
    root_agent = mod.root_agent

    skill = AgentSkill(
        id=root_agent.name,
        name=root_agent.name,
        description=getattr(root_agent, "description", args.description),
        tags=["adk", args.module],
        examples=["What can you do?"],
    )

    # create_agent_card auto-sets the three Vertex-mandatory fields:
    #   preferred_transport=http_json (a2a.py:107) — Vertex rejects anything else
    #   streaming=False                (a2a.py:105) — Vertex hard-disables streaming
    #   supports_authenticated_extended_card=True (a2a.py:108) — for handle_authenticated_agent_card
    card = create_agent_card(
        agent_name=args.display,
        description=args.description,
        skills=[skill],
    )

    a2a_app = A2aAgent(
        agent_card=card,
        agent_executor_builder=_executor_builder(root_agent),
        # No agent_executor_kwargs — builder is no-arg.
        # No task_store_builder    — defaults to InMemoryTaskStore.
        # No request_handler_builder — defaults to DefaultRequestHandler.
    )

    # Do NOT call a2a_app.set_up() here. set_up() rewrites agent_card.url
    # using os.environ["GOOGLE_CLOUD_AGENT_ENGINE_ID"], which is only
    # populated on the deployed container — calling it locally pollutes
    # the card. agent_engines.create() runs set_up() server-side with
    # the real engine ID.
    # Source: vertexai/preview/reasoning_engines/templates/a2a.py:233-301.

    requirements = [
        "google-cloud-aiplatform[agent_engines]>=1.130.0",
        "google-adk[a2a]>=2.0.0b1,<3.0.0",
    ]

    # Auto-forward env vars that the agent might need at runtime. Source of
    # truth is AUTO_FORWARD_ENV_VARS at module top — single place to add new
    # ones. Anything not set in the deploy shell falls back to the in-code
    # defaults (or the agent runs without that capability).
    env_vars: dict[str, str] = {}
    for name in AUTO_FORWARD_ENV_VARS:
        value = os.environ.get(name)
        if value:
            env_vars[name] = value
    if env_vars:
        print(f"Forwarding {len(env_vars)} env var(s) to engine: {sorted(env_vars)}")

    print(f"Deploying {args.module} to {args.region} as {args.display!r} ...")
    print(f"Runtime SA: {args.service_account}")
    remote = agent_engines.create(
        agent_engine=a2a_app,
        requirements=requirements,
        extra_packages=[args.module],   # uploads e.g. ./level_1_agent
        display_name=args.display,
        env_vars=env_vars or None,
        service_account=args.service_account,
    )
    print(f"\n✅ Deployed: {remote.resource_name}")
    print(
        "Verify with `python check_a2a.py` from the swarm repo's adk/ — "
        "expect api_modes to include 'a2a_extension'."
    )


if __name__ == "__main__":
    main()
