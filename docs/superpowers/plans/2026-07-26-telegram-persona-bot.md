# Telegram Persona Bot Implementation Plan

> **Status: ✅ ALL 16 TASKS COMPLETE.** Implemented on `feat/persona-bot-impl` and merged to `master` (merge commit `58219d7`, "Merge feat/persona-bot-impl: Telegram persona bot (16 tasks, fallback branch)"). Full test suite: 72 passed (unit + Docker-backed integration via testcontainers). See the per-task commit references inline below and `README.md` for run/test/deploy instructions. Remaining manual-verification items (AC-5, AC-7, AC-12) are noted in their Acceptance Criteria rows — code and mocked-test coverage are done, live end-to-end runs against real Telegram/Postgres/API keys have not been performed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch Telegram DM bot that roleplays a per-user adaptive persona, reasons and uses tools silently, and never reveals tool use or reasoning to the user.

**Architecture:** An outer langgraph graph (`load_memory → agent → compose_persona → END`) runs per inbound Telegram message. The `agent` node is a **hand-rolled DeepSeek tool loop** over the `bot.llm` facade (the "work" call — the Task-1 spike ruled out embedding `langstage-hermes`, see `docs/spike-notes.md`); `compose_persona` is a tool-free "speak" call and is the *only* component wired to Telegram's outbound path — the structural silence guarantee. Long-term memory is Postgres + pgvector behind a `memory` facade, which the tools call directly. `persist_memory` runs post-send as a fire-and-forget task.

**Tech Stack:** Python 3.12 (async), python-telegram-bot, langgraph, langgraph-checkpoint-postgres, httpx → OpenRouter (DeepSeek `deepseek/deepseek-v4-flash`), Postgres + pgvector (SQLAlchemy async + psycopg), OpenAI TTS, Tavily (web), Brave Images, pytest + pytest-asyncio + testcontainers, Docker/k8s. (`langstage-hermes`/`langchain-openai` were spike-only and dropped from runtime deps.)

## Global Constraints

- **Python 3.12, async throughout.** No sync I/O on the reply path.
- **Facades are the only seams to externals:** every model call goes through `bot/llm.py`; every storage access goes through `bot/memory.py`. Swapping OpenRouter or the DB touches one file each.
- **`compose_persona` is the ONLY component that produces user-facing text.** Telegram's outbound path is wired to it alone.
- **Config via env vars only** (pydantic-settings); secrets never hardcoded.
- **Default model `deepseek/deepseek-v4-flash` via OpenRouter** for all four call sites (Hermes work loop, Hermes aux_model, `compose_persona`, `persist_memory`); each overridable independently via config. (O11)
- **Vision model `google/gemini-2.5-flash`; embed model `openai/text-embedding-3-small`, `EMBED_DIM=1536`.** Chat model is text-only, so images ALWAYS route through the custom `vision_analyze` tool. (O7)
- **`EMBED_BACKEND=openrouter|openai`** — a config seam; switching to OpenAI-direct embeddings is a one-line env change, no code change. (O6)
- **Toolsets (O4):** enable `web`, `session_search`, `skills`+reflection, memory (custom provider), plus custom plugin tools `recall`/`image_search`/`vision_analyze`. Disable `terminal`, `process`, `execute_code`, `file`, `cronjob`, `image_generate`, `delegate_task`, `messaging`, `kanban`, and the bundled `vision_analyze` toolset.
- **Hermes default budgets; no hard latency target** — `max_iterations=90`, all reflection/compression defaults stand. (O5)
- **Per-user isolation (fallback form):** O2's per-user `HERMES_HOME` is dropped (the spike proved it unsafe in-process). Isolation is instead: per-user rows in Postgres keyed by `chat_id`, and a per-turn `contextvars` turn context so concurrent DMs never share `user_id`/image bytes. Never store the current user on a shared instance (concurrency rule still holds).
- **`actions` tool deferred to v2.** (O9) Voice-in/STT and image generation deferred to v2.
- **DM-only.** Ignore group updates; state keyed by `chat_id` (== user in a DM).
- **Fully containerized (O13):** bot image + `pgvector/pgvector:pg16`; `docker-compose.yml` for local, `deploy/` for k8s. Migrations run at container start before polling.
- **FALLBACK BRANCH ACTIVE (O12).** The Task-1 gating spike (see `docs/spike-notes.md`, commit `bec1c9a`) found `langstage-hermes==0.4.25` structurally unusable: its `MemoryProvider` is never invoked by the runtime, plugin discovery never runs on the factory path, there is no built-in `web`/Tavily toolset, per-user `HERMES_HOME` is unsafe in-process (per-call env re-resolution bleeds across concurrent users), and the system prompt is not host-replaceable. Per O12 (no improvised workarounds), the `agent` node is a **hand-rolled DeepSeek tool loop** in `bot/nodes/agent.py` over the `bot.llm` facade, with plain tools in `bot/tools.py`. `langstage-hermes` and `langchain-openai` are **dropped from runtime deps** (spike-only). No `bot/hermes_node.py`, no `bot/hermes_plugin/`, no per-user `HERMES_HOME`. AC-4 is satisfied by "the recorded fallback loop with the same tool surface."
- **TDD:** every task writes a failing test first, then minimal code, then green. Frequent commits.

---

## File Structure

```
telegram-persona-bot/
  bot/
    __init__.py
    config.py            # pydantic-settings; all env config
    llm.py               # OpenRouter facade: chat / chat_vision / embed
    memory.py            # pg+pgvector facade (all storage goes through here)
    persona.py           # build_system_prompt(profile, memories)
    turn_context.py      # ContextVar bridging image bytes in / image url out
    tools.py             # web_search / recall / image_search / vision_analyze (plain async tools)
    tts.py               # OpenAI TTS voice-out facade
    media.py             # telegram photo→bytes, voice/photo send helpers
    ingress.py           # telegram handlers, send-guard, /forget, pacing, __main__
    state.py             # GraphState TypedDict
    graph.py             # outer graph wiring + checkpointer
    migrate.py           # apply migrations/*.sql at startup
    nodes/
      __init__.py
      load_memory.py     # load_memory(state)
      agent.py           # hand-rolled DeepSeek tool loop (FALLBACK BRANCH — O12)
      compose_persona.py # compose_persona(state) — sole user-facing text
      persist_memory.py  # post-send fact extraction (plain async fn, not a node)
  migrations/
    0001_init.sql        # pgvector schema
  tests/
    unit/
    integration/
  docs/spike-notes.md    # step-1 gating-spike findings
  pyproject.toml
  Dockerfile
  docker-compose.yml
  deploy/                # k8s manifests
  README.md
```

**Fallback branch (O12) — ACTIVE.** The Task-1 spike (recorded in `docs/spike-notes.md`, committed `bec1c9a`) determined `langstage-hermes==0.4.25` is structurally unusable, so the `agent` node is a hand-rolled DeepSeek tool loop in `bot/nodes/agent.py` over the plain tools in `bot/tools.py`. Tasks 6 and 7 below are **replaced** by the fallback Tasks 6 and 7 (plain tool module + local loop); every other task is unchanged. `bot/hermes_node.py` and `bot/hermes_plugin/` are NOT created.

---

### Task 1: Gating spike — verify the installed `langstage-hermes` API surface ✅ DONE (commit `bec1c9a`)

> **Outcome: FALLBACK BRANCH.** `langstage-hermes==0.4.25` is structurally unusable (see `docs/spike-notes.md`). All steps below are complete; `langstage-hermes`/`langchain-openai` were dropped from runtime deps. Tasks 6–7 use the fallback (hand-rolled loop).

**Files:**
- Create: `pyproject.toml` (minimal, just enough to install the package)
- Create: `scripts/spike_hermes_api.py` (throwaway check script)
- Create: `docs/spike-notes.md` (findings — the durable output)

**Interfaces:**
- Produces: a recorded decision in `docs/spike-notes.md` — either "bridge path confirmed, field names = {…}" or "fallback branch: hand-rolled loop". **Gates Tasks 6–7.**

- [x] **Step 1: Add the pinned dependency**

In `pyproject.toml`, under `[project]` dependencies, add `langstage-hermes[openai]==<exact-latest>` (resolve the exact version at install time; pin it — no floating). Create a venv and install.

- [x] **Step 2: Write the API-surface probe script**

`scripts/spike_hermes_api.py` — import and introspect, printing each name this plan relies on:

```python
import inspect, langstage_hermes as lh

def show(label, obj):
    print(f"\n== {label} ==")
    print(obj)

show("create_hermes_agent signature", inspect.signature(lh.create_hermes_agent))
cfg = lh.HermesConfig
show("HermesConfig fields", getattr(cfg, "model_fields", getattr(cfg, "__annotations__", cfg)))
show("MemoryProvider ABC", [m for m in dir(lh.MemoryProvider) if not m.startswith("_")])
show("PluginContext.register_tool", inspect.signature(lh.PluginContext.register_tool))
# plugin discovery: entry-point group vs HERMES_HOME/plugins dir scan
show("plugin discovery hints", [n for n in dir(lh) if "plugin" in n.lower()])
```

Expected reference values (from the package SPEC — adapt names cosmetically if the installed package differs, and note the diff):
- `create_hermes_agent(config) -> CompiledStateGraph`
- `HermesConfig` fields: home dir, `disabled_toolsets`, `memory.provider`, `model.default`, `model.aux_model`
- `MemoryProvider`: `setup_session(session_id, user_id)`, `recall(query, mode) -> list[str]`, `record_turn(role, content)`, `teardown()`
- `PluginContext.register_tool(fn, toolset=...)`, `register_memory_provider(provider)`
- built-in `web` toolset honors `TAVILY_API_KEY`

- [x] **Step 3: Run the probe and the package's own live check**

Run: `python scripts/spike_hermes_api.py` and record output.
Run: `langstage-hermes verify` with OpenRouter env set (see Task 4 env). Expected: live round-trip passes.

- [x] **Step 4: Record the decision in `docs/spike-notes.md`**

Write findings: confirmed field names (or cosmetic diffs), plugin discovery mechanism (entry-point group name vs dir scan), whether the bridge path is viable. **If structurally unviable → record "FALLBACK BRANCH" explicitly**, with the reason. Do NOT improvise a workaround.

- [x] **Step 5: Commit**

```bash
git init && git add pyproject.toml scripts/spike_hermes_api.py docs/spike-notes.md
git commit -m "chore: gating spike — verify langstage-hermes API surface"
```

**Depends on:** —

---

### Task 2: Scaffold the project ✅ DONE (commit `b3f1c65`)

**Files:**
- Create: `pyproject.toml` (full dependency manifest), `bot/__init__.py`, `bot/nodes/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: an installable package `bot` with all deps and a green (empty) pytest run.

- [x] **Step 1: Write the failing test**

`tests/unit/test_scaffold.py`:

```python
def test_bot_package_imports():
    import bot  # noqa: F401
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bot'`

- [x] **Step 3: Fill `pyproject.toml` and create package dirs**

`pyproject.toml` dependencies (**fallback branch — no `langstage-hermes`/`langchain-openai`**; this is already the committed state after Task 1):
```toml
[project]
name = "telegram-persona-bot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "python-telegram-bot[ext]>=21",
  "langgraph>=0.2",
  "langgraph-checkpoint-postgres>=1.0",
  "psycopg[binary]>=3.1",
  "sqlalchemy[asyncio]>=2.0",
  "pgvector>=0.2",
  "httpx>=0.27",
  "pydantic-settings>=2.0",
]
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "testcontainers[postgres]>=4", "respx>=0.21"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Create the empty `__init__.py` files and `tests/conftest.py` (empty for now). Drop `bot/hermes_plugin/__init__.py` from the scaffold (not used on the fallback branch). `pip install -e ".[dev]"`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_scaffold.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add pyproject.toml bot tests
git commit -m "chore: scaffold package layout and dependencies"
```

**Depends on:** 1

---

### Task 3: Postgres + pgvector schema + migration runner ✅ DONE (commit `5bee409`)

**Files:**
- Create: `migrations/0001_init.sql`, `bot/migrate.py`
- Test: `tests/integration/test_migrate.py`

**Interfaces:**
- Produces: `async def apply_migrations(database_url: str) -> None` — applies `migrations/*.sql` in filename order, tracking applied files in `schema_migrations`. Idempotent.
- Schema: `users(telegram_user_id BIGINT PK, name, preferences JSONB, tone, summary, created_at, updated_at)`, `memories(id BIGSERIAL PK, telegram_user_id FK, text, embedding vector(1536), source, created_at)` + HNSW cosine index, `turns(id BIGSERIAL PK, telegram_user_id, role, content, created_at)`.

- [x] **Step 1: Write the failing test**

`tests/integration/test_migrate.py` (testcontainers Postgres with the `pgvector/pgvector:pg16` image):

```python
import pytest
from testcontainers.postgres import PostgresContainer
from bot.migrate import apply_migrations
import psycopg

@pytest.fixture(scope="module")
def pg_url():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "")

async def test_migrations_create_tables_and_are_idempotent(pg_url):
    await apply_migrations(pg_url)
    await apply_migrations(pg_url)  # second run must not error
    with psycopg.connect(pg_url.replace("+psycopg", "")) as c:
        rows = c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"users", "memories", "turns", "schema_migrations"} <= names
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_migrate.py -v`
Expected: FAIL — `ModuleNotFoundError: bot.migrate`

- [x] **Step 3: Write the migration SQL and runner**

`migrations/0001_init.sql`:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS users (
  telegram_user_id BIGINT PRIMARY KEY,
  name TEXT, preferences JSONB NOT NULL DEFAULT '{}',
  tone TEXT, summary TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS memories (
  id BIGSERIAL PRIMARY KEY,
  telegram_user_id BIGINT NOT NULL REFERENCES users(telegram_user_id) ON DELETE CASCADE,
  text TEXT NOT NULL, embedding vector(1536),
  source TEXT NOT NULL DEFAULT 'extracted',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memories_embedding_idx
  ON memories USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS turns (
  id BIGSERIAL PRIMARY KEY,
  telegram_user_id BIGINT NOT NULL,
  role TEXT NOT NULL, content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`bot/migrate.py`:
```python
import pathlib, psycopg

MIGRATIONS = pathlib.Path(__file__).parent.parent / "migrations"

async def apply_migrations(database_url: str) -> None:
    url = database_url.replace("+psycopg", "").replace("+asyncpg", "")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                     "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())")
        applied = {r[0] for r in conn.execute("SELECT filename FROM schema_migrations")}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in applied:
                continue
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations(filename) VALUES (%s)", (path.name,))
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_migrate.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add migrations/0001_init.sql bot/migrate.py tests/integration/test_migrate.py
git commit -m "feat: pgvector schema and idempotent migration runner"
```

**Depends on:** 2

---

### Task 4: Config + `llm` facade ✅ DONE (commit `5f30e09`; retry/backoff + openai-embed coverage added in `c179136`)

**Files:**
- Create: `bot/config.py`, `bot/llm.py`
- Test: `tests/unit/test_config.py`, `tests/unit/test_llm.py`

**Interfaces:**
- Produces: `bot.config.settings` (a `Settings` instance with all env fields below).
- Produces: `bot.llm.chat(messages: list[dict], model: str | None = None) -> str`, `bot.llm.chat_vision(messages: list[dict], model: str | None = None) -> str`, `bot.llm.embed(texts: list[str]) -> list[list[float]]`. All async, `httpx`, 3 retries with backoff.
- Produces (fallback branch): `bot.llm.chat_with_tools(messages: list[dict], tools: list[dict], model: str | None = None) -> dict` — returns the raw assistant message object (may contain `tool_calls`), for the hand-rolled agent loop in Task 7.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_config.py`:
```python
def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    from bot.config import Settings
    s = Settings()
    assert s.model_chat == "deepseek/deepseek-v4-flash"
    assert s.embed_dim == 1536
    assert s.embed_backend == "openrouter"
```

`tests/unit/test_llm.py` (mock the HTTP layer):
```python
import respx, httpx, pytest
from bot import llm

@respx.mock
async def test_chat_posts_to_openrouter_and_returns_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    out = await llm.chat([{"role": "user", "content": "yo"}])
    assert out == "hi"
    assert route.called

@respx.mock
async def test_embed_returns_vectors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}))
    vecs = await llm.embed(["hello"])
    assert vecs == [[0.1, 0.2]]
```

(Add `respx` to dev deps.)

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config.py tests/unit/test_llm.py -v`
Expected: FAIL — modules not defined.

- [x] **Step 3: Write config and the facade**

`bot/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    telegram_bot_token: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    brave_api_key: str = ""
    database_url: str = ""
    model_chat: str = "deepseek/deepseek-v4-flash"
    model_vision: str = "google/gemini-2.5-flash"
    model_embed: str = "openai/text-embedding-3-small"
    embed_dim: int = 1536
    embed_backend: str = "openrouter"  # or "openai"
    pacing_enabled: bool = True
    pacing_delay_min_s: float = 0.5
    pacing_delay_max_s: float = 2.0
    agent_max_iterations: int = 90  # O5 default; bounds the hand-rolled tool loop (Task 7)

settings = Settings()
```

`bot/llm.py`:
```python
import asyncio, httpx
from bot.config import settings

OPENROUTER = "https://openrouter.ai/api/v1"
OPENAI = "https://api.openai.com/v1"

async def _post(url: str, headers: dict, payload: dict) -> dict:
    last = None
    async with httpx.AsyncClient(timeout=60) as c:
        for attempt in range(3):
            try:
                r = await c.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last = e
                await asyncio.sleep(0.5 * (2 ** attempt))
    raise last

def _or_headers() -> dict:
    return {"Authorization": f"Bearer {settings.openrouter_api_key}"}

async def chat(messages: list[dict], model: str | None = None) -> str:
    data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(),
                       {"model": model or settings.model_chat, "messages": messages})
    return data["choices"][0]["message"]["content"]

async def chat_with_tools(messages: list[dict], tools: list[dict],
                          model: str | None = None) -> dict:
    data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(),
                       {"model": model or settings.model_chat, "messages": messages,
                        "tools": tools})
    return data["choices"][0]["message"]  # raw assistant msg; may carry tool_calls

async def chat_vision(messages: list[dict], model: str | None = None) -> str:
    data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(),
                       {"model": model or settings.model_vision, "messages": messages})
    return data["choices"][0]["message"]["content"]

async def embed(texts: list[str]) -> list[list[float]]:
    if settings.embed_backend == "openai":
        url, headers, model = f"{OPENAI}/embeddings", \
            {"Authorization": f"Bearer {settings.openai_api_key}"}, "text-embedding-3-small"
    else:
        url, headers, model = f"{OPENROUTER}/embeddings", _or_headers(), settings.model_embed
    data = await _post(url, headers, {"model": model, "input": texts})
    return [row["embedding"] for row in data["data"]]
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py tests/unit/test_llm.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/config.py bot/llm.py tests/unit/test_config.py tests/unit/test_llm.py pyproject.toml
git commit -m "feat: env config and OpenRouter llm facade (chat/vision/embed)"
```

**Depends on:** 2

---

### Task 5: `memory` facade (Postgres + pgvector) ✅ DONE (commit `03f50b3`; race-safety fix in `734bcd9`)

**Files:**
- Create: `bot/memory.py`
- Test: `tests/integration/test_memory.py`

**Interfaces:**
- Consumes: `bot.llm.embed`, `bot.config.settings.database_url`.
- Produces (all async): `get_profile(user_id) -> dict` (lazy-creates row), `search_memories(user_id, query, k=5) -> list[str]`, `upsert_memories(user_id, facts: list[str]) -> None`, `update_profile_summary(user_id, summary) -> None`, `log_turn(user_id, role, content) -> None`, `recent_turns(user_id, n=10) -> list[dict]`, `forget_user(user_id) -> None`.

- [x] **Step 1: Write the failing test**

`tests/integration/test_memory.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from testcontainers.postgres import PostgresContainer
from bot.migrate import apply_migrations

@pytest.fixture(scope="module")
def pg_url():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "+psycopg")

@pytest.fixture(autouse=True)
def _wire(pg_url, monkeypatch):
    monkeypatch.setattr("bot.config.settings.database_url", pg_url)

async def test_upsert_then_semantic_recall_and_forget(pg_url):
    await apply_migrations(pg_url)
    from bot import memory
    # deterministic fake embeddings: "guitar" near query, "weather" far
    def fake_embed(texts):
        table = {"learning guitar": [1.0, 0.0], "hates cold weather": [0.0, 1.0],
                 "guitar": [1.0, 0.0]}
        return AsyncMock(return_value=[table.get(t, [0.5, 0.5]) for t in texts])()
    with patch("bot.memory.llm.embed", side_effect=fake_embed):
        await memory.get_profile(42)                       # lazy create
        await memory.upsert_memories(42, ["learning guitar", "hates cold weather"])
        top = await memory.search_memories(42, "guitar", k=1)
        assert top == ["learning guitar"]
        await memory.forget_user(42)
        assert await memory.search_memories(42, "guitar", k=5) == []
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_memory.py -v`
Expected: FAIL — `bot.memory` not defined.

- [x] **Step 3: Write the facade**

`bot/memory.py` — SQLAlchemy async engine built lazily from `settings.database_url`; embeddings via `bot.llm.embed`; pgvector cosine ordering (`embedding <=> :q`). Functions exactly per the Interfaces block. `forget_user` deletes from `memories` + `turns` and resets the `users` row (name/summary/preferences/tone cleared). `search_memories` embeds the query, orders by cosine distance ascending, returns the `text` column of the top-K.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_memory.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/memory.py tests/integration/test_memory.py
git commit -m "feat: pg+pgvector memory facade with semantic recall and /forget wipe"
```

**Depends on:** 3, 4

---

### Task 6: Turn context + plain async tools (FALLBACK BRANCH) ✅ DONE (commit `57ec44c`; test hardening in `906547b`)

> Replaces the original Task 6 (Hermes plugin). No plugin, no `MemoryProvider` — the tools are plain async functions the Task-7 loop dispatches directly. Tool *bodies* are as originally designed; per-user isolation comes from the turn context + `chat_id`, not a session-keyed provider.

**Files:**
- Create: `bot/turn_context.py`, `bot/tools.py`
- Test: `tests/unit/test_turn_context.py`, `tests/unit/test_tools.py`

**Interfaces:**
- Consumes: `bot.memory.*`, `bot.llm.chat_vision`, `bot.config.settings`.
- Produces: `bot.turn_context.turn_context: ContextVar[dict]` with keys `{"image_bytes": bytes|None, "found_image_url": str|None, "user_id": int}`.
- Produces: async tools `web_search(query: str) -> str` (Tavily, O8), `recall(query: str) -> str` (top-K from `memory.search_memories` for the turn's `user_id`), `image_search(query: str) -> str` (Brave `safesearch=strict`; sets `found_image_url` in turn context, returns a confirmation string), `vision_analyze() -> str` (reads `image_bytes` from turn context, returns a description). All resolve `user_id`/`image_bytes` from `turn_context`, so no user identity is ever passed through the model.
- Produces: `TOOL_SPECS: list[dict]` — the OpenAI-format tool schemas for the four tools, and `TOOL_FUNCS: dict[str, callable]` mapping tool name → coroutine, both consumed by Task 7.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_turn_context.py`:
```python
import asyncio
from bot.turn_context import turn_context

async def test_context_is_task_local():
    async def worker(uid):
        turn_context.set({"image_bytes": None, "found_image_url": None, "user_id": uid})
        await asyncio.sleep(0.01)
        return turn_context.get()["user_id"]
    results = await asyncio.gather(worker(1), worker(2), worker(3))
    assert sorted(results) == [1, 2, 3]  # no cross-task bleed
```

`tests/unit/test_tools.py`:
```python
from unittest.mock import AsyncMock, patch
import respx, httpx
from bot.turn_context import turn_context
from bot import tools

async def test_recall_uses_turn_context_user():
    turn_context.set({"image_bytes": None, "found_image_url": None, "user_id": 100})
    with patch("bot.tools.memory.search_memories",
               new=AsyncMock(return_value=["learning guitar"])) as m:
        out = await tools.recall("music")
    m.assert_awaited_once_with(100, "music")
    assert "guitar" in out

async def test_vision_analyze_reads_turn_context_bytes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    turn_context.set({"image_bytes": b"\x89PNG...", "found_image_url": None, "user_id": 1})
    with patch("bot.tools.llm.chat_vision", new=AsyncMock(return_value="a red bicycle")):
        assert await tools.vision_analyze() == "a red bicycle"

@respx.mock
async def test_image_search_sets_found_url(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    turn_context.set({"image_bytes": None, "found_image_url": None, "user_id": 1})
    respx.get("https://api.search.brave.com/res/v1/images/search").mock(
        return_value=httpx.Response(200, json={"results": [{"properties": {"url": "http://img/1.jpg"}}]}))
    msg = await tools.image_search("red bicycle")
    assert turn_context.get()["found_image_url"] == "http://img/1.jpg"
    assert "bicycle" in msg.lower() or "found" in msg.lower()

@respx.mock
async def test_web_search_returns_summary(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(200, json={"results": [{"title": "T", "content": "answer text"}]}))
    out = await tools.web_search("who won")
    assert "answer text" in out
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_turn_context.py tests/unit/test_tools.py -v`
Expected: FAIL — modules not defined.

- [x] **Step 3: Write turn_context and tools**

`bot/turn_context.py`:
```python
from contextvars import ContextVar
turn_context: ContextVar[dict] = ContextVar("turn_context")
```

`bot/tools.py`:
```python
import base64, httpx
from bot import llm, memory
from bot.config import settings
from bot.turn_context import turn_context

async def web_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.tavily.com/search",
                         json={"api_key": settings.tavily_api_key, "query": query,
                               "max_results": 3})
        r.raise_for_status()
        results = r.json().get("results", [])
    return "\n".join(f"- {x.get('content','')}" for x in results) or "(no results)"

async def recall(query: str) -> str:
    uid = turn_context.get()["user_id"]
    mems = await memory.search_memories(uid, query)
    return "\n".join(f"- {m}" for m in mems) if mems else "(nothing relevant remembered)"

async def image_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.search.brave.com/res/v1/images/search",
                        headers={"X-Subscription-Token": settings.brave_api_key},
                        params={"q": query, "safesearch": "strict", "count": 1})
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "no suitable image found"
    turn_context.get()["found_image_url"] = results[0]["properties"]["url"]
    return f"found an image for '{query}'"

async def vision_analyze() -> str:
    img = turn_context.get()["image_bytes"]
    if not img:
        return "no image was attached"
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
    return await llm.chat_vision([{"role": "user", "content": [
        {"type": "text", "text": "Describe this image factually."},
        {"type": "image_url", "image_url": {"url": data_url}}]}])

def _spec(name, desc, props=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": list((props or {}).keys())}}}

TOOL_SPECS = [
    _spec("web_search", "Search the web for fresh facts.", {"query": {"type": "string"}}),
    _spec("recall", "Recall durable facts remembered about this user.", {"query": {"type": "string"}}),
    _spec("image_search", "Find an existing image to send.", {"query": {"type": "string"}}),
    _spec("vision_analyze", "Describe the photo the user attached this turn."),
]
TOOL_FUNCS = {"web_search": web_search, "recall": recall,
              "image_search": image_search, "vision_analyze": vision_analyze}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_turn_context.py tests/unit/test_tools.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/turn_context.py bot/tools.py tests/unit/test_turn_context.py tests/unit/test_tools.py
git commit -m "feat: turn context + plain async tools (web/recall/image/vision) — fallback branch"
```

**Depends on:** 5

---

### Task 7: `bot/nodes/agent.py` — hand-rolled DeepSeek tool loop (FALLBACK BRANCH) ✅ DONE (commit `a730a6e`; extra loop-edge-case tests in `4d24a73`)

> Replaces the original Task 7 (per-user Hermes provisioning). No `HERMES_HOME`, no LRU graph cache. The `agent` node is a bounded loop over `llm.chat_with_tools`, dispatching `TOOL_FUNCS` until the model stops requesting tools. Conversation continuity comes from `memory.recent_turns(chat_id)`, not a checkpointed inner graph.

**Files:**
- Create: `bot/nodes/agent.py`
- Test: `tests/unit/test_agent_loop.py`

**Interfaces:**
- Consumes: `bot.llm.chat_with_tools`, `bot.tools.TOOL_SPECS`/`TOOL_FUNCS`, `bot.memory.recent_turns`, `bot.turn_context`, `bot.config.settings.agent_max_iterations`.
- Produces: `async def agent_node(state: GraphState) -> dict` returning `{"raw_result": str|None, "found_image_url": str|None}` on success, or `{"raw_result": None, "agent_error": str}` on exception (never raises into the reply path).
- Note: no `evict_graph` on this branch — `/forget` (Task 13) is a Postgres wipe only; drop the `hermes_node.evict_graph` reference there.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_agent_loop.py`:
```python
from unittest.mock import AsyncMock, patch
from bot.nodes import agent

def _msg(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}

async def test_loop_dispatches_tool_then_returns_final():
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "recall", "arguments": '{"query":"music"}'}}]),
        _msg(content="You mentioned you're learning guitar."),
    ]
    with patch("bot.nodes.agent.llm.chat_with_tools",
               new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS",
                    {"recall": AsyncMock(return_value="- learning guitar")}, clear=False):
        out = await agent.agent_node({"chat_id": 7, "user_text": "what do I play?",
                                      "image_bytes": None})
    assert out["raw_result"] == "You mentioned you're learning guitar."

async def test_loop_sets_hint_and_found_image_url():
    async def img_search(query):  # simulate tool writing to turn context
        from bot.turn_context import turn_context
        turn_context.get()["found_image_url"] = "http://img/1.jpg"
        return "found an image"
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "image_search", "arguments": '{"query":"cat"}'}}]),
        _msg(content="here's a cat"),
    ]
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS", {"image_search": img_search}, clear=False):
        out = await agent.agent_node({"chat_id": 7, "user_text": "send a cat", "image_bytes": None})
    assert out["found_image_url"] == "http://img/1.jpg"

async def test_photo_adds_vision_hint_to_first_user_message():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        await agent.agent_node({"chat_id": 7, "user_text": "what is this", "image_bytes": b"x"})
    assert any("vision_analyze" in str(m.get("content", "")) for m in captured["msgs"])

async def test_agent_node_never_raises():
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=RuntimeError("boom"))), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        out = await agent.agent_node({"chat_id": 7, "user_text": "hi", "image_bytes": None})
    assert out["raw_result"] is None and "boom" in out["agent_error"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -v`
Expected: FAIL — `bot.nodes.agent` not defined.

- [x] **Step 3: Write the loop**

`bot/nodes/agent.py`:
```python
import json
from bot import llm, memory
from bot.config import settings
from bot.tools import TOOL_SPECS, TOOL_FUNCS
from bot.turn_context import turn_context

async def agent_node(state) -> dict:
    token = turn_context.set({"image_bytes": state.get("image_bytes"),
                              "found_image_url": None, "user_id": state["chat_id"]})
    try:
        recent = await memory.recent_turns(state["chat_id"], 10)
        messages = [{"role": r["role"], "content": r["content"]} for r in recent]
        prompt = state["user_text"]
        if state.get("image_bytes"):
            prompt += "\n\n(The user sent a photo with this message; call vision_analyze to see it.)"
        messages.append({"role": "user", "content": prompt})

        for _ in range(settings.agent_max_iterations):
            msg = await llm.chat_with_tools(messages, TOOL_SPECS)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return {"raw_result": msg.get("content"),
                        "found_image_url": turn_context.get()["found_image_url"]}
            messages.append(msg)
            for tc in tool_calls:
                fn = TOOL_FUNCS[tc["function"]["name"]]
                args = json.loads(tc["function"]["arguments"] or "{}")
                result = await fn(**args)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        # iteration budget exhausted: return the last content if any
        return {"raw_result": messages[-1].get("content"),
                "found_image_url": turn_context.get()["found_image_url"]}
    except Exception as exc:
        return {"raw_result": None, "agent_error": str(exc)}
    finally:
        turn_context.reset(token)
```

Add `agent_max_iterations: int = 90` to `bot/config.py` (Task 4) — O5 default. (If Task 4 is already committed, add it as a one-line config edit here and note it in the commit.)

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_agent_loop.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/nodes/agent.py tests/unit/test_agent_loop.py bot/config.py
git commit -m "feat: hand-rolled DeepSeek tool loop agent node (fallback branch)"
```

**Depends on:** 6

---

### Task 8: `load_memory` node + persona assembly ✅ DONE (commit `bc498d7`)

**Files:**
- Create: `bot/nodes/load_memory.py`, `bot/persona.py`
- Test: `tests/unit/test_load_memory.py`, `tests/unit/test_persona.py`

**Interfaces:**
- Consumes: `bot.memory.get_profile`, `bot.memory.search_memories`.
- Produces: `async def load_memory(state) -> dict` returning `{"profile": dict, "memories": list[str]}`.
- Produces: `def build_system_prompt(profile: dict, memories: list[str]) -> str`.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_persona.py`:
```python
from bot.persona import build_system_prompt

def test_prompt_includes_facts_and_silence_rules():
    p = build_system_prompt({"name": "Sam", "summary": "likes short replies", "tone": "warm"},
                            ["learning guitar"])
    assert "Sam" in p and "guitar" in p
    for rule in ["never mention", "tool", "As an AI"]:
        assert rule.lower() in p.lower()
```

`tests/unit/test_load_memory.py`:
```python
from unittest.mock import AsyncMock, patch
from bot.nodes.load_memory import load_memory

async def test_load_memory_merges_profile_and_memories():
    with patch("bot.nodes.load_memory.memory.get_profile",
               new=AsyncMock(return_value={"name": "Sam"})), \
         patch("bot.nodes.load_memory.memory.search_memories",
               new=AsyncMock(return_value=["m1", "m2"])):
        out = await load_memory({"chat_id": 1, "user_text": "hey"})
    assert out == {"profile": {"name": "Sam"}, "memories": ["m1", "m2"]}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_load_memory.py tests/unit/test_persona.py -v`
Expected: FAIL — modules not defined.

- [x] **Step 3: Write persona builder and load_memory**

`bot/persona.py` — assemble: persona voice preamble, the user's known name/preferences/tone/running summary, the top-K memories, and explicit instructions: never mention tools/lookups/reasoning; no "As an AI" hedging; no markdown-heavy formatting; restyle facts faithfully (never drop or invent). `bot/nodes/load_memory.py` — call `get_profile(chat_id)` and `search_memories(chat_id, user_text)`, return the merged dict.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_load_memory.py tests/unit/test_persona.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/nodes/load_memory.py bot/persona.py tests/unit/test_load_memory.py tests/unit/test_persona.py
git commit -m "feat: load_memory node and persona system-prompt builder"
```

**Depends on:** 4, 5

---

### Task 9: `compose_persona` node (the speak call) ✅ DONE (commit `2ab676c`; review fixes in `0bcc162`, `7ad4033`)

**Files:**
- Create: `bot/nodes/compose_persona.py`
- Test: `tests/unit/test_compose_persona.py`

**Interfaces:**
- Consumes: `bot.llm.chat` (NO tools), `bot.persona.build_system_prompt`.
- Produces: `async def compose_persona(state) -> dict` returning `{"reply": {"text": str, "voice": bool, "image_url": str | None}}`.
- **Rule:** `image_url` is copied from `state["found_image_url"]` (never parsed from model text). If `agent_error` set or `raw_result` empty → in-character "couldn't pull that up right now" line.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_compose_persona.py`:
```python
from unittest.mock import AsyncMock, patch
from bot.nodes.compose_persona import compose_persona

BASE = {"chat_id": 1, "profile": {"name": "Sam"}, "memories": [],
        "user_text": "tell me about the weather", "found_image_url": None}

async def test_reply_uses_only_llm_chat_no_tools():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="It's sunny out!")) as m:
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert out["reply"]["text"] == "It's sunny out!"
    # llm.chat called without a tools kwarg
    assert "tools" not in m.call_args.kwargs

async def test_image_url_comes_from_state_not_model():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="here you go")):
        out = await compose_persona({**BASE, "raw_result": "found one",
                                     "found_image_url": "http://img/1.jpg"})
    assert out["reply"]["image_url"] == "http://img/1.jpg"

async def test_graceful_fallback_on_agent_error():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="hmm, I couldn't pull that up right now")):
        out = await compose_persona({**BASE, "raw_result": None, "agent_error": "boom"})
    assert out["reply"]["text"]
    assert out["reply"]["image_url"] is None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_compose_persona.py -v`
Expected: FAIL — module not defined.

- [x] **Step 3: Write the node**

One `llm.chat` call (no `tools`): messages = `[{"role":"system", build_system_prompt(...)}, {"role":"user", user_text}, {"role":"assistant"/context, raw_result-as-facts}]`. When `agent_error` or empty `raw_result`, feed a "you couldn't retrieve it, apologize in character" instruction instead. Post-process: strip markdown artifacts/tool jargon. `voice` decided here (the persona chooses). `image_url = state.get("found_image_url")`.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_compose_persona.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/nodes/compose_persona.py tests/unit/test_compose_persona.py
git commit -m "feat: compose_persona speak call — sole user-facing text producer"
```

**Depends on:** 8

---

### Task 10: `persist_memory` (post-send background) ✅ DONE (commit `8cd768d`; fix in `22d69f6`)

**Files:**
- Create: `bot/nodes/persist_memory.py`
- Test: `tests/unit/test_persist_memory.py`

**Interfaces:**
- Consumes: `bot.llm.chat`, `bot.memory.upsert_memories`, `bot.memory.update_profile_summary`.
- Produces: `async def persist_memory(user_id: int, user_text: str, reply_text: str, recent: list[dict]) -> None` — plain async function (NOT a graph node, per O10). Errors logged, never raised.

- [x] **Step 1: Write the failing test**

`tests/unit/test_persist_memory.py`:
```python
from unittest.mock import AsyncMock, patch
from bot.nodes.persist_memory import persist_memory

async def test_extracts_facts_and_upserts():
    with patch("bot.nodes.persist_memory.llm.chat",
               new=AsyncMock(return_value='["user is learning guitar"]')), \
         patch("bot.nodes.persist_memory.memory.upsert_memories",
               new=AsyncMock()) as up, \
         patch("bot.nodes.persist_memory.memory.update_profile_summary", new=AsyncMock()):
        await persist_memory(1, "I started guitar lessons", "nice!", [])
    up.assert_awaited_once()
    assert "guitar" in up.call_args.args[1][0]

async def test_errors_are_swallowed():
    with patch("bot.nodes.persist_memory.llm.chat",
               new=AsyncMock(side_effect=RuntimeError("x"))):
        await persist_memory(1, "hi", "hey", [])  # must not raise
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_persist_memory.py -v`
Expected: FAIL — module not defined.

- [x] **Step 3: Write the function**

One cheap `llm.chat` extraction call → parse a JSON list of durable facts → `memory.upsert_memories(user_id, facts)` when non-empty; refresh the running summary via `memory.update_profile_summary` when warranted. Wrap the whole body in try/except that logs and returns.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_persist_memory.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/nodes/persist_memory.py tests/unit/test_persist_memory.py
git commit -m "feat: persist_memory post-send fact extraction (off the reply path)"
```

**Depends on:** 4, 5

---

### Task 11: Outer graph wiring ✅ DONE (commit `406401c`; checkpointer setup fix in `116faab`)

**Files:**
- Create: `bot/state.py`, `bot/graph.py`
- Test: `tests/unit/test_graph.py`

**Interfaces:**
- Consumes: `load_memory`, `agent_node`, `compose_persona`, `langgraph`, `langgraph-checkpoint-postgres`.
- Produces: `GraphState` TypedDict; `def build_graph(checkpointer=None)` returning a compiled graph with edges `load_memory → agent → compose_persona → END`.
- `GraphState` keys: `chat_id: int`, `user_text: str`, `image_bytes: bytes | None`, `profile: dict`, `memories: list[str]`, `raw_result: str | None`, `found_image_url: str | None`, `agent_error: str | None`, `reply: dict`.

- [x] **Step 1: Write the failing test**

`tests/unit/test_graph.py` (stub all three nodes; assert only `compose_persona`'s reply survives and no tool string leaks):
```python
from unittest.mock import AsyncMock, patch
from bot.graph import build_graph

async def test_only_compose_persona_output_reaches_state():
    with patch("bot.graph.load_memory",
               new=AsyncMock(return_value={"profile": {}, "memories": []})), \
         patch("bot.graph.agent_node",
               new=AsyncMock(return_value={"raw_result": "TOOL:web_lookup rawJSON{...}",
                                           "found_image_url": None})), \
         patch("bot.graph.compose_persona",
               new=AsyncMock(return_value={"reply": {"text": "Nice and sunny!",
                                                     "voice": False, "image_url": None}})):
        graph = build_graph()
        out = await graph.ainvoke({"chat_id": 1, "user_text": "weather?", "image_bytes": None})
    assert out["reply"]["text"] == "Nice and sunny!"
    assert "TOOL:" not in out["reply"]["text"]
    assert "rawJSON" not in out["reply"]["text"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_graph.py -v`
Expected: FAIL — module not defined.

- [x] **Step 3: Write state and graph wiring**

`bot/state.py` — the `GraphState` TypedDict above. `bot/graph.py` — `StateGraph(GraphState)`, add nodes `load_memory`/`agent`/`compose_persona`, edges `START → load_memory → agent → compose_persona → END`, compile with `langgraph-checkpoint-postgres` on `settings.database_url` when a checkpointer isn't injected. Reference wiring style: `agentBerry/orchestrator/graph.py` node-registration + checkpointer pattern (do NOT copy its delegate/escalate logic). Import the node functions as module-level names in `bot/graph.py` (`from bot.nodes.load_memory import load_memory`, `from bot.nodes.agent import agent_node`, `from bot.nodes.compose_persona import compose_persona`) so tests can patch them as `bot.graph.<name>`.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_graph.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/state.py bot/graph.py tests/unit/test_graph.py
git commit -m "feat: outer graph wiring (load_memory→agent→compose_persona) with checkpointer"
```

**Depends on:** 7, 8, 9

---

### Task 12: TTS + media helpers ✅ DONE (commit `932125e`)

**Files:**
- Create: `bot/tts.py`, `bot/media.py`
- Test: `tests/unit/test_tts.py`, `tests/unit/test_media.py`

**Interfaces:**
- Produces: `bot.tts.synth(text: str) -> bytes` (OpenAI TTS, OGG/Opus for `sendVoice`).
- Produces: `bot.media.photo_to_bytes(photo) -> bytes` (download largest size), `bot.media.to_voice(audio_bytes)`, and photo-send helpers used by ingress.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_tts.py`:
```python
import respx, httpx
from unittest.mock import patch
from bot import tts

@respx.mock
async def test_synth_calls_openai_and_returns_bytes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    respx.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"OGGaudio"))
    assert await tts.synth("hello there") == b"OGGaudio"
```

`tests/unit/test_media.py`:
```python
from unittest.mock import AsyncMock, MagicMock
from bot import media

async def test_photo_to_bytes_downloads_largest():
    photo_sizes = [MagicMock(file_size=100), MagicMock(file_size=900)]
    largest = photo_sizes[1]
    file_obj = MagicMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"imgbytes"))
    largest.get_file = AsyncMock(return_value=file_obj)
    out = await media.photo_to_bytes(photo_sizes)
    assert bytes(out) == b"imgbytes"
    largest.get_file.assert_awaited_once()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tts.py tests/unit/test_media.py -v`
Expected: FAIL — modules not defined.

- [x] **Step 3: Write tts and media**

`bot/tts.py` — `POST https://api.openai.com/v1/audio/speech` with model from config, `response_format="opus"`, returns `r.content`. `bot/media.py` — `photo_to_bytes` picks the max-`file_size` PhotoSize, `get_file()`, `download_as_bytearray()`; `to_voice` wraps bytes into a `telegram.InputFile` for `sendVoice`; photo-send helper for `sendPhoto`.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_tts.py tests/unit/test_media.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/tts.py bot/media.py tests/unit/test_tts.py tests/unit/test_media.py
git commit -m "feat: OpenAI TTS voice-out and telegram media helpers"
```

**Depends on:** 4

---

### Task 13: Ingress — handlers, send-guard, `/forget`, pacing ✅ DONE (commit `d5caebf`; fix in `8892804`)

**Files:**
- Create: `bot/ingress.py` (with `__main__`)
- Test: `tests/unit/test_send_guard.py`, `tests/unit/test_ingress.py`

**Interfaces:**
- Consumes: `bot.graph.build_graph`, `bot.nodes.persist_memory.persist_memory`, `bot.media.*`, `bot.tts.synth`, `bot.memory.forget_user`.
- Produces: `bot.ingress.send_guard(text: str) -> bool` (True = safe to send), `on_message`, `forget_command`, `start_command`, `main()`.

- [x] **Step 1: Write the failing tests**

`tests/unit/test_send_guard.py`:
```python
import pytest
from bot.ingress import send_guard

@pytest.mark.parametrize("text,ok", [
    ("Hey, how's it going?", True),
    ("I think x = {a, b} works — try it!", True),   # braces in normal prose: NOT rejected
    ("", False),
    ('{"tool":"web_lookup","result":"..."}', False), # raw JSON payload: rejected
    ("TOOL_CALL: web_lookup(query=...)", False),
])
def test_send_guard(text, ok):
    assert send_guard(text) is ok
```

`tests/unit/test_ingress.py` (send-guard fallback + persist_memory fire-and-forget):
```python
from unittest.mock import AsyncMock, patch
from bot import ingress

async def test_persist_memory_launched_after_send():
    with patch("bot.ingress.persist_memory", new=AsyncMock()) as pm, \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)  # test helper
        await ingress.on_message(update, ctx)
        # allow the created task to schedule
    pm.assert_awaited()  # persist ran, off the reply path
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_send_guard.py tests/unit/test_ingress.py -v`
Expected: FAIL — module/attrs not defined.

- [x] **Step 3: Write ingress**

`python-telegram-bot` async app (polling; DM-only — ignore group updates). `send_guard`: reject empty; reject text that parses as a JSON object/array or matches a raw-tool pattern (e.g. `^\s*[\[{]`, `TOOL_CALL:`, `TOOL:`), but do NOT reject prose that merely contains braces. `on_message`: fire `typing` chat action (re-fire every ~4s while the graph runs); photo → `media.photo_to_bytes` → `state["image_bytes"]`; build state; `_graph.ainvoke` (Task 11); render `reply` — text through `send_guard` first (on fail, log + replace with an in-character fallback line), optional pacing delay (`PACING_*`) + chunking into 1–2 messages, `voice` → `tts.synth` → `sendVoice`, `image_url` → `sendPhoto`; then `asyncio.create_task(persist_memory(...))`. `/forget`: `memory.forget_user(chat_id)` + in-character confirmation (fallback branch: no per-user Hermes home or cached graph to evict — Postgres wipe is the whole story). `/start`: brief in-character intro. Never send error traces. Provide the `_fake_text_update` test helper.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_send_guard.py tests/unit/test_ingress.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add bot/ingress.py tests/unit/test_send_guard.py tests/unit/test_ingress.py
git commit -m "feat: ingress — handlers, send-guard backstop, /forget, pacing"
```

**Depends on:** 10, 11, 12

---

### Task 14: Dockerfile, compose + k8s manifests ✅ DONE (commit `24a3cc1`)

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `deploy/bot-deployment.yaml`, `deploy/postgres-statefulset.yaml`, `deploy/secrets.example.yaml`
- Test: `tests/integration/test_compose_smoke.py` (marked slow; optional in CI)

**Interfaces:**
- Produces: a container that runs `python -m bot.migrate && python -m bot.ingress`; a compose stack (`bot` + `db`) and k8s manifests (bot Deployment single-replica — stateless, no PVC on the fallback branch; Postgres StatefulSet + PVC + ClusterIP).

- [x] **Step 1: Write the failing smoke test**

`tests/integration/test_compose_smoke.py`:
```python
import subprocess, pytest

@pytest.mark.slow
def test_compose_config_is_valid():
    r = subprocess.run(["docker", "compose", "config"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "pgvector/pgvector:pg16" in r.stdout
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_compose_smoke.py -v`
Expected: FAIL — no `docker-compose.yml`.

- [x] **Step 3: Write the container + orchestration files**

`Dockerfile`: `python:3.12-slim`, install the package, `ENTRYPOINT ["sh","-c","python -m bot.migrate && python -m bot.ingress"]`. `docker-compose.yml`: `bot` (build `.`, env from `.env`) + `db` (`pgvector/pgvector:pg16`, named volume, healthcheck; `bot depends_on db healthy`); `DATABASE_URL` points at `db`. `bot/migrate.py` gains a `python -m bot.migrate` entrypoint (`if __name__ == "__main__": asyncio.run(apply_migrations(settings.database_url))`). `deploy/`: bot Deployment (single replica, stateless — env from Secret, no PVC) + Postgres StatefulSet (`pgvector/pgvector:pg16`, own PVC, ClusterIP Service). The bot holds no per-user disk state on the fallback branch; all durable state is in Postgres.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_compose_smoke.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml deploy tests/integration/test_compose_smoke.py bot/migrate.py
git commit -m "feat: containerization — Dockerfile, compose, k8s manifests, migrate entrypoint"
```

**Depends on:** 13

---

### Task 15: Full-graph integration + faithfulness + multimodal tests ✅ DONE (commits `6086bc1`, `30c8a34`, `2d4a54b`; live-API stub fix in `3c598f9`)

**Files:**
- Create: `tests/integration/test_full_graph.py`, `tests/integration/test_faithfulness.py`, `tests/integration/test_multimodal.py`

**Interfaces:**
- Consumes: everything above; testcontainers Postgres; stubbed `llm` (`chat`/`chat_with_tools`).

- [x] **Step 1: Write the failing tests**

`tests/integration/test_full_graph.py` — real Postgres, stubbed `llm.chat_with_tools` (agent loop) + stubbed `llm.chat` (compose); assert only `compose_persona` output reaches the send sink and no tool string leaks; and a cross-session memory test (turn 1 states a fact, `persist_memory` writes it, turn 2 recalls it).
`tests/integration/test_faithfulness.py` — `compose_persona` preserves facts in `raw_result` (feed `raw_result="the capital is Lima"`, stub `llm.chat` to echo-with-restyle, assert "Lima" survives; assert it doesn't invent a different capital).
`tests/integration/test_multimodal.py` — photo → `image_bytes` in state → turn context → `vision_analyze` returns a description into the agent loop; voice-flagged reply → `tts.synth` + `sendVoice`; `image_search` sets `found_image_url` → `sendPhoto`.

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration -v`
Expected: FAIL initially (assertions not yet satisfied / helpers missing).

- [x] **Step 3: Make them pass**

Wire the test doubles and any small production gaps they expose (e.g. a missing `voice`/`image_url` passthrough). Do NOT weaken the assertions — fix the code.

- [x] **Step 4: Run the whole suite**

Run: `pytest -v`
Expected: PASS (unit + integration).

- [x] **Step 5: Commit**

```bash
git add tests/integration
git commit -m "test: full-graph, faithfulness, and multimodal integration coverage"
```

**Depends on:** 13, 14

---

### Task 16: README ✅ DONE (commit `d370e8d`; checkpointer/chunking fix in `d69ff6c`)

**Files:**
- Create: `README.md`

**Interfaces:**
- Produces: setup/run/test/deploy docs. (Per user's global rule: document features in the README.)

- [x] **Step 1: Write the README**

Sections: overview + the two-call silence guarantee; prereqs (env vars from Task 4, Postgres+pgvector); local run (`docker compose up`); `langstage-hermes verify`; test commands (`pytest`, note `-m slow` for the compose smoke test); deployment summary (Task 14); the v2 deferred list (actions tool, voice-in/STT, image generation, group chats); and a note on the O12 fallback branch recorded in `docs/spike-notes.md`.

- [x] **Step 2: Verify the documented commands run**

Run each command block in the README (env-setup, `pytest`, `docker compose config`) and confirm it matches actual behavior.

- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README — setup, run, test, deploy, v2 deferred list"
```

**Depends on:** 15

---

## Acceptance Criteria

Traceable to the design Goals/Testing and the plan's overrides.

- **AC-1 — Conversation:** a DM text produces an in-character reply; native "typing…" shows while working. → Tasks 9, 13
- **AC-2 — Hidden reasoning:** no tool names/traces/raw results/JSON reach the user; outbound wired only to `compose_persona`; send-guard unit-tested. → Tasks 9, 11, 13, 15
- **AC-3 — Per-user adaptation:** profile + episodic memories compound; a second session uses facts from the first (integration test). → Tasks 5, 6, 8, 10, 15
- **AC-4 — Hermes as a node (fallback form):** the spike (Task 1) proved the embedded `langstage-hermes` path unusable, so the work call runs through the recorded fallback loop with the same tool surface (`web_search`/`recall`/`image_search`/`vision_analyze`), bounded by `agent_max_iterations` (O5 default 90). Decision recorded in `docs/spike-notes.md`. → Tasks 1, 6, 7
- **AC-5 — Silent web lookup:** persona answers a fresh-facts question via Tavily with no tool tell. → Tasks 6, 7 (manual: 15)
- **AC-6 — Photo understanding:** a user photo is described via custom `vision_analyze` and answered in character. → Tasks 4, 6, 12, 13, 15
- **AC-7 — Voice-out:** a voice-flagged reply is synthesized via OpenAI TTS and delivered as a voice note. → Tasks 9, 12, 13, 15 (manual for audio)
- **AC-8 — Image send:** `image_search` stores the URL in turn context → surfaces as `found_image_url` → `sendPhoto`. → Tasks 6, 7, 13, 15
- **AC-9 — `/forget`:** wipes the user's Postgres rows (profile + memories + turns); next turn has no memory. (Fallback branch: no per-user Hermes home to wipe.) → Tasks 5, 13, 15
- **AC-10 — Faithfulness:** `compose_persona` preserves the facts in `raw_result` (no drop/invent). → Tasks 9, 15
- **AC-11 — Memory round-trip:** pgvector upsert → semantic top-K recall against real Postgres. → Tasks 3, 5, 15
- **AC-12 — Deployable, containerized:** `docker compose up` brings up bot + Postgres and answers a DM; k8s manifests deploy the same two containers; migrations run at start; memory survives restart. → Tasks 3, 14 (manual verification)

## Self-Review Notes

- **Spec coverage:** every design section maps to a task — ingress/silence (13, 11), two-call split (7, 9), memory 3-layer (3, 5, 8, 10), multimodal (6, 7, 12, 13), providers (4, 6, 12), containerization (14), testing (15). Overrides after the Task-1 spike: O1 (Hermes-as-node) → superseded by the fallback loop (7); O2 (per-user HERMES_HOME) → dropped, isolation via Postgres+turn-context (5, 6); O3 (MemoryProvider bridge) → dropped, tools call `memory` directly (6); O4/O5 → the loop's tool set + `agent_max_iterations` (6, 7); O6→4, O7→6/7, O8→4/6/12, O9→deferred (noted 16), O10→10/13, O11→4, **O12→ FALLBACK BRANCH taken (1, 6, 7)**, O13→14.
- **Type consistency:** `GraphState` keys (Task 11) match producers/consumers — `raw_result`/`found_image_url`/`agent_error` (Task 7 `agent_node`) → `reply` (Task 9); `agent_node` (from `bot.nodes.agent`), `load_memory`, `compose_persona` are imported into `bot.graph` and patched there in tests (Task 11). `reply` dict shape `{text, voice, image_url}` is identical in Tasks 9, 11, 13. `TOOL_SPECS`/`TOOL_FUNCS` (Task 6) are the exact names Task 7 imports. `chat_with_tools` (Task 4) returns the raw assistant message the Task-7 loop inspects for `tool_calls`.
- **No placeholders:** every code step carries real code or an exact spec of the function body plus its test.
