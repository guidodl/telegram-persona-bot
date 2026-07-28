# telegram-persona-bot

A Telegram companion bot with a persistent, per-user persona. Works in direct
messages and in group chats (where it replies only when addressed). Facts and
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

When a reply carries an `image_url`, `bot/ingress.py` downloads the image bytes
(`media.fetch_image`) and **uploads** them to Telegram as a file
(`media.to_photo`), rather than handing Telegram the URL to hotlink — so the
photo arrives as if a person picked it and sent it. If the download or upload
fails, it is logged and the text reply still goes through.

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

### Personas

By default the bot speaks with a generic warm-companion preamble. Set
`PERSONA_FILE` to a markdown/text file with a persona definition (identity,
style, triggers — see `bot/personas/erminio.md` for a full example) to give
the bot a specific character. The persona text becomes the system preamble of
`compose_persona` (the hard `SILENCE_RULES` are always appended), and a
short agent-facing briefing derived from it is prepended to the `agent` node's
message list so tool choices (e.g. when to `image_search`) follow the
persona's triggers. A missing/unreadable file falls back to the generic
preamble with a warning.

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
| `BOT_NAME` | `""` | Plain name the bot answers to in groups (no `@` needed) |
| `BOT_ALIASES` | `""` | Comma-separated extra names the bot answers to in groups |
| `GROUP_ALLOWED_CHATS` | `""` | Comma-separated group chat IDs the bot may answer in; empty = any group |
| `ALLOWED_USERS` | `""` | Comma-separated Telegram user IDs allowed to DM the bot; empty = everyone |
| `PERSONA_FILE` | `""` | Path to a persona definition file (e.g. `bot/personas/erminio.md`); empty = generic companion persona |
| `OPENROUTER_API_KEY` | `""` | Chat/vision/embedding calls via OpenRouter |
| `OPENAI_API_KEY` | `""` | Optional OpenAI embeddings backend (`EMBED_BACKEND=openai`); not used for TTS |
| `TAVILY_API_KEY` | `""` | `web_search` tool |
| `BRAVE_API_KEY` | `""` | `image_search` tool |
| `DATABASE_URL` | `""` | Postgres connection string, e.g. `postgresql+psycopg://bot:bot@db:5432/bot` |
| `MODEL_CHAT` | `deepseek/deepseek-v4-flash` | Persona reply model (`compose_persona`) |
| `MODEL_AGENT` | `""` | Tool-loop model deciding whether to call tools (e.g. `image_search`); empty falls back to `MODEL_CHAT` |
| `MODEL_VISION` | `google/gemini-2.5-flash` | `vision_analyze` model |
| `MODEL_EMBED` | `openai/text-embedding-3-small` | Embedding model (OpenRouter backend) |
| `TTS_MODEL` | `x-ai/grok-voice-tts-1.0` | OpenRouter speech model for voice replies (list via `GET /models?output_modalities=speech`) |
| `TTS_VOICE` | `leo` | Voice ID; must be in the model's `supported_voices`. Voices are provider-namespaced |
| `EMBED_DIM` | `1536` | Vector dimension (must match the `memories.embedding` column) |
| `EMBED_BACKEND` | `openrouter` | `openrouter` or `openai` — which provider serves embeddings |
| `PACING_ENABLED` | `true` | Adds a human-like delay between multi-chunk reply sends |
| `PACING_DELAY_MIN_S` | `0.5` | Minimum pacing delay (seconds) |
| `PACING_DELAY_MAX_S` | `2.0` | Maximum pacing delay (seconds) |
| `AGENT_MAX_ITERATIONS` | `6` | Upper bound on the tool-calling loop in `bot/nodes/agent.py`; each iteration is a serial LLM round-trip, so this caps worst-case reply latency |
| `HERMES_URL` | `http://hermes:8642` | Base URL of the Hermes agent sidecar (`bot/hermes_client.py`'s `call_hermes`) |
| `MCP_TOOLS_URL` | `http://mcp-tools:8000` | Base URL of the mcp-tools sidecar Hermes calls out to for tool execution |

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
It handles both direct messages and group chats, and registers `/start` and
`/forget`, plus a text/photo message handler.

### Group chats

In a group the bot stays silent unless it is **addressed**:

- someone `@mention`s its username, or
- mentions it by `BOT_NAME` (plain name, no `@`) or by a `BOT_ALIASES` entry
  (comma-separated extra names), or
- replies to one of the bot's own messages.

When `GROUP_ALLOWED_CHATS` is set (comma-separated chat IDs), the bot only
answers in those groups and stays silent everywhere else; empty means any
group.

Memory and persona are keyed by the **speaker's** Telegram user id, not the
chat — so the bot knows each person consistently across their DMs and any
group they share. In a DM the chat id equals the user id, so DM behavior is
unchanged. Conversation history is checkpointed per thread: one thread per
user in DMs, one per `(group, user)` in groups, so members never share a
conversation. `/forget` wipes the calling user's own memory.

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
- Voice-in / speech-to-text (only voice-*out* via OpenRouter TTS is supported, sent as an mp3 audio file)
- Image generation (only image *search*, via Brave, is supported)
