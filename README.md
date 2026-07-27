# telegram-persona-bot

A Telegram DM companion bot with a persistent, per-user persona. Facts and
memory live in Postgres (+ pgvector); the bot process itself is stateless.

## Overview

Every incoming DM runs through a three-node LangGraph pipeline:

```
load_memory -> agent -> compose_persona
```

- **`load_memory`** fetches the user's profile and top-K semantically
  relevant memories from Postgres/pgvector.
- **`agent`** is a hand-rolled tool-calling loop over the `bot.llm` facade
  (DeepSeek via OpenRouter). It can call `web_search` (Tavily), `recall`
  (pgvector memory search), `image_search` (Brave), and `vision_analyze`
  (vision model, for photos the user sent). It returns raw facts
  (`raw_result`) and, if any, a `found_image_url` — never sent to the user
  directly.
- **`compose_persona`** is the *only* call whose output reaches Telegram. It
  takes the raw facts and rewrites them in-character, strips markdown,
  extracts an optional `[VOICE]` tag, and returns `{text, voice, image_url}`.

**The two-call silence guarantee:** no tool name, tool-call JSON, or raw
search/vision result ever reaches the user — only `compose_persona`'s
in-character text. Two independent layers enforce this: `compose_persona`
never sees the `tools` list (its LLM call carries no tool schema, so it
cannot itself request one), and `bot/ingress.py`'s `send_guard()` rejects any
outgoing text that still looks like raw tool/JSON output as a last-resort
backstop.

### Fallback branch: no langstage-hermes

The design originally planned to run the "work" call through
`langstage-hermes` as an agent node. A Task-1 spike (see
[`docs/spike-notes.md`](docs/spike-notes.md)) found the package's
`MemoryProvider`, plugin-discovery, and toolset-filtering seams are not
wired into the runtime in the pinned version — using it would have required
global-lock/monkeypatch workarounds the plan explicitly forbids. This repo
therefore runs the **fallback branch**: `bot/nodes/agent.py` is a plain loop
over `bot.llm.chat_with_tools`, bounded by `agent_max_iterations`, with tools
defined directly in `bot/tools.py`. `langstage-hermes` is not a runtime
dependency, and there is no per-user `HERMES_HOME` — all durable state
(profile, memories, conversation turns) lives in Postgres, and `/forget`
deletes those rows.

## Prerequisites

- Python 3.12+
- Postgres with the `pgvector` extension (the `pgvector/pgvector:pg16` image
  provides this out of the box)
- Docker + Docker Compose, if running via `docker compose up`

### Environment variables

Defined in `bot/config.py` (`Settings`, loaded from `.env` via
`pydantic-settings`; snake_case field names map to upper-case env vars). See
`.env.example` for a starter file.

| Variable | Default | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | `""` | Telegram bot API token |
| `OPENROUTER_API_KEY` | `""` | Chat/vision/embedding calls via OpenRouter |
| `OPENAI_API_KEY` | `""` | OpenAI TTS (`tts.synth`) and optional OpenAI embeddings backend |
| `TAVILY_API_KEY` | `""` | `web_search` tool |
| `BRAVE_API_KEY` | `""` | `image_search` tool |
| `DATABASE_URL` | `""` | Postgres connection string, e.g. `postgresql+psycopg://bot:bot@db:5432/bot` |
| `MODEL_CHAT` | `deepseek/deepseek-v4-flash` | Chat + tool-loop model |
| `MODEL_VISION` | `google/gemini-2.5-flash` | `vision_analyze` model |
| `MODEL_EMBED` | `openai/text-embedding-3-small` | Embedding model (OpenRouter backend) |
| `TTS_MODEL` | `gpt-4o-mini-tts` | OpenAI text-to-speech model for voice replies |
| `EMBED_DIM` | `1536` | Vector dimension (must match the `memories.embedding` column) |
| `EMBED_BACKEND` | `openrouter` | `openrouter` or `openai` — which provider serves embeddings |
| `PACING_ENABLED` | `true` | Adds a human-like delay between multi-chunk reply sends |
| `PACING_DELAY_MIN_S` | `0.5` | Minimum pacing delay (seconds) |
| `PACING_DELAY_MAX_S` | `2.0` | Maximum pacing delay (seconds) |
| `AGENT_MAX_ITERATIONS` | `90` | Upper bound on the tool-calling loop in `bot/nodes/agent.py` |

## Local run

### Option A: Docker Compose (recommended)

```bash
docker compose up
```

This builds the bot image, starts a `pgvector/pgvector:pg16` Postgres
container, waits for its healthcheck, then starts the bot. The container
entrypoint runs migrations before the bot starts:

```
python -m bot.migrate && python -m bot.ingress
```

Put real secrets in a `.env` file (see `.env.example`); `docker-compose.yml`
loads it and overrides `DATABASE_URL` to point at the `db` service.

### Option B: Python venv

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
# start Postgres separately, e.g.:
docker run -d -e POSTGRES_USER=bot -e POSTGRES_PASSWORD=bot -e POSTGRES_DB=bot \
  -p 5432:5432 pgvector/pgvector:pg16
cp .env.example .env   # fill in real values; set DATABASE_URL to
                       # postgresql+psycopg://bot:bot@localhost:5432/bot for the container above
.venv/bin/python -m bot.migrate
.venv/bin/python -m bot.ingress
```

`bot/ingress.py` runs the bot via long-polling (`Application.updater.start_polling()`).
It only handles direct messages — non-private chats are ignored — and
registers `/start` and `/forget`, plus a text/photo message handler.

## Tests

```bash
.venv/bin/pytest -m "not slow" -q
```

runs the unit suite plus the integration tests that spin up a real
`pgvector/pgvector:pg16` container via `testcontainers`. On this machine,
Ryuk (testcontainers' reaper sidecar) needs to be disabled for the container
port-mapping lookup to succeed:

```bash
TESTCONTAINERS_RYUK_DISABLED=true .venv/bin/pytest -m "not slow" -q
```

The `-m "not slow"` flag deselects `tests/integration/test_compose_smoke.py`,
which shells out to `docker compose config` — run it separately (needs
Docker, not a container) with:

```bash
TESTCONTAINERS_RYUK_DISABLED=true .venv/bin/pytest -m slow -q
```

or simply:

```bash
docker compose config
```

Running the full suite (unit + all integration, including the compose
smoke test) needs both Docker and the env var set:

```bash
TESTCONTAINERS_RYUK_DISABLED=true .venv/bin/pytest -q
```

## Deployment

- **Dockerfile**: installs the package (`pip install .`), copies `bot/` and
  `migrations/`, and entrypoints `python -m bot.migrate && python -m bot.ingress`
  — migrations always run before the bot starts polling.
- **docker-compose.yml**: two services — `db` (`pgvector/pgvector:pg16`,
  with a `pg_isready` healthcheck) and `bot` (built from the Dockerfile,
  waits on `db`'s healthcheck, reads secrets from `.env`).
- **Kubernetes** (`deploy/`):
  - `deploy/bot-deployment.yaml` — a stateless `Deployment` (1 replica) for
    the bot, pulling all env vars from a `telegram-persona-bot-secrets`
    Secret via `envFrom`.
  - `deploy/postgres-statefulset.yaml` — a `StatefulSet` + headless
    `Service` running `pgvector/pgvector:pg16` with a 5Gi
    `PersistentVolumeClaim`, so memory data survives pod restarts.
  - `deploy/secrets.example.yaml` — template for the `Secret` (bot API
    keys, `DATABASE_URL`, and the Postgres credentials); copy, fill in real
    values, and `kubectl apply -f` it before deploying the other manifests.

Because all durable state lives in Postgres (profile, memories, turns) and
LangGraph checkpoints (`AsyncPostgresSaver`), the bot Deployment itself is
fully stateless and safe to scale to zero/restart without losing user data.

## Deferred to v2

Not implemented in this version:

- An external-actions tool (e.g. calendar, reminders, third-party APIs
  beyond web/image search)
- Voice-in / speech-to-text (only voice-*out* via OpenAI TTS is supported)
- Image generation (only image *search*, via Brave, is supported)
- Group chats (the bot is DM-only by design — `on_message` ignores any
  non-`ChatType.PRIVATE` chat)
