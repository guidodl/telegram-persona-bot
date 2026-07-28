# Hermes-Agent Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled tool loop in the `agent` node with the Nous Hermes-Agent framework running as an out-of-process sidecar, with the four existing tools re-exposed via an MCP server and per-turn context held in-memory in that server.

**Architecture:** Three containers via docker-compose — `bot` (LangGraph app, only `agent_node` changes), `hermes` (`nousresearch/hermes-agent`, runs the tool loop server-side over loopback `:8642`), and `mcp-tools` (a small Python MCP + HTTP server wrapping the four tools and holding a per-turn context dict). The bot registers per-turn inputs on `mcp-tools`, calls Hermes, reads back the image side-channel, and returns the same `{raw_result, found_image_url, agent_error}` shape so `compose_persona` and ingress are untouched.

**Tech Stack:** Python 3.12, httpx (async HTTP + SSE), FastAPI/uvicorn (MCP-tools server), `mcp` Python SDK (tool exposure), Docker Compose, pgvector (unchanged), pytest + respx.

## Global Constraints

- Python `>=3.12`.
- `compose_persona`, memory schema, and Telegram ingress/egress MUST NOT change behavior. `agent_node`'s return shape stays `{raw_result: str|None, found_image_url: str|None, agent_error: str|None}`.
- The four tools MUST keep working: `web_search` (Tavily), `recall` (pgvector), `image_search` (Brave, `safesearch=strict`, writes side-channel), `vision_analyze` (vision model over user photo).
- No new persistent infra: per-turn context is an in-memory dict in `mcp-tools`, dropped after each turn. `mcp-tools` runs as a single instance (no horizontal scaling).
- Hermes runs no local model weights — it calls OpenRouter/etc. over the network. Pin the **arm64** image tag for the Pi.
- `agent_node` MUST NEVER raise into the graph — errors return `agent_error`, mirroring today.
- No comments unless the why is non-obvious. Update README when adding the new services/tool surface.

---

### Task 1: Per-turn context store + tool bodies in `mcp-tools` server

**Files:**
- Create: `mcp_tools/__init__.py`
- Create: `mcp_tools/store.py`
- Create: `mcp_tools/tools.py`
- Test: `tests/unit/test_mcp_tools_store.py`, `tests/unit/test_mcp_tools_bodies.py`

**Interfaces:**
- Consumes: existing tool logic from `bot/tools.py` (Tavily/Brave/vision HTTP calls), `bot.config.settings`, `bot.memory.search_memories`, `bot.llm.chat_vision`.
- Produces:
  - `store.register(turn_id: str, user_id: int, image_bytes: bytes | None) -> None`
  - `store.get(turn_id: str) -> dict` (`{user_id, image_bytes, found_image_url}`; raises `KeyError` if absent)
  - `store.set_found_image(turn_id: str, url: str) -> None`
  - `store.pop_found_image(turn_id: str) -> str | None` (reads `found_image_url`, deletes the entry)
  - `tools.web_search(query: str) -> str`
  - `tools.recall(query: str, turn_id: str) -> str`
  - `tools.image_search(query: str, turn_id: str) -> str`
  - `tools.vision_analyze(turn_id: str) -> str`

- [ ] **Step 1: Write the failing test for the store**

```python
# tests/unit/test_mcp_tools_store.py
import pytest
from mcp_tools import store

def test_register_get_roundtrip():
    store.register("t1", user_id=7, image_bytes=b"x")
    entry = store.get("t1")
    assert entry["user_id"] == 7
    assert entry["image_bytes"] == b"x"
    assert entry["found_image_url"] is None

def test_set_and_pop_found_image_drops_entry():
    store.register("t2", user_id=7, image_bytes=None)
    store.set_found_image("t2", "http://img/1.jpg")
    assert store.pop_found_image("t2") == "http://img/1.jpg"
    with pytest.raises(KeyError):
        store.get("t2")

def test_pop_missing_turn_returns_none():
    assert store.pop_found_image("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mcp_tools_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_tools'`

- [ ] **Step 3: Implement the store**

```python
# mcp_tools/__init__.py
# (empty package marker)
```

```python
# mcp_tools/store.py
_turns: dict[str, dict] = {}

def register(turn_id: str, user_id: int, image_bytes: bytes | None) -> None:
    _turns[turn_id] = {"user_id": user_id, "image_bytes": image_bytes,
                       "found_image_url": None}

def get(turn_id: str) -> dict:
    return _turns[turn_id]

def set_found_image(turn_id: str, url: str) -> None:
    _turns[turn_id]["found_image_url"] = url

def pop_found_image(turn_id: str) -> str | None:
    entry = _turns.pop(turn_id, None)
    return entry["found_image_url"] if entry else None
```

- [ ] **Step 4: Run store test to verify it passes**

Run: `pytest tests/unit/test_mcp_tools_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing test for tool bodies**

```python
# tests/unit/test_mcp_tools_bodies.py
from unittest.mock import AsyncMock, patch
from mcp_tools import store, tools

async def test_recall_reads_user_id_from_store():
    store.register("t1", user_id=7, image_bytes=None)
    with patch("mcp_tools.tools.memory.search_memories",
               new=AsyncMock(return_value=["learning guitar"])):
        out = await tools.recall("music", "t1")
    assert "learning guitar" in out

async def test_image_search_writes_side_channel(respx_mock):
    store.register("t2", user_id=7, image_bytes=None)
    respx_mock.get("https://api.search.brave.com/res/v1/images/search").respond(
        json={"results": [{"properties": {"url": "http://img/1.jpg"}}]})
    out = await tools.image_search("cat on a sofa", "t2")
    assert store.get("t2")["found_image_url"] == "http://img/1.jpg"
    assert "cat on a sofa" in out

async def test_vision_analyze_reads_image_bytes_from_store():
    store.register("t3", user_id=7, image_bytes=b"jpegbytes")
    with patch("mcp_tools.tools.llm.chat_vision",
               new=AsyncMock(return_value="a ginger cat")):
        out = await tools.vision_analyze("t3")
    assert out == "a ginger cat"

async def test_vision_analyze_no_image_returns_message():
    store.register("t4", user_id=7, image_bytes=None)
    out = await tools.vision_analyze("t4")
    assert out == "no image was attached"
```

- [ ] **Step 6: Run tool-body test to verify it fails**

Run: `pytest tests/unit/test_mcp_tools_bodies.py -v`
Expected: FAIL with `AttributeError: module 'mcp_tools.tools' has no attribute 'recall'`

- [ ] **Step 7: Implement tool bodies (lifted from bot/tools.py, turn_context → store)**

```python
# mcp_tools/tools.py
import base64, httpx
from bot import llm, memory
from bot.config import settings
from mcp_tools import store

async def web_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.tavily.com/search",
                         json={"api_key": settings.tavily_api_key, "query": query,
                               "max_results": 3})
        r.raise_for_status()
        results = r.json().get("results", [])
    return "\n".join(f"- {x.get('content','')}" for x in results) or "(no results)"

async def recall(query: str, turn_id: str) -> str:
    uid = store.get(turn_id)["user_id"]
    mems = await memory.search_memories(uid, query)
    return "\n".join(f"- {m}" for m in mems) if mems else "(nothing relevant remembered)"

async def image_search(query: str, turn_id: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.search.brave.com/res/v1/images/search",
                        headers={"X-Subscription-Token": settings.brave_api_key},
                        params={"q": query, "safesearch": "strict", "count": 1})
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "no suitable image found"
    store.set_found_image(turn_id, results[0]["properties"]["url"])
    return f"found an image for '{query}'"

async def vision_analyze(turn_id: str) -> str:
    img = store.get(turn_id)["image_bytes"]
    if not img:
        return "no image was attached"
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
    return await llm.chat_vision([{"role": "user", "content": [
        {"type": "text", "text": "Describe this image factually."},
        {"type": "image_url", "image_url": {"url": data_url}}]}])
```

- [ ] **Step 8: Run tool-body test to verify it passes**

Run: `pytest tests/unit/test_mcp_tools_bodies.py -v`
Expected: PASS (4 passed)

- [ ] **Step 9: Commit**

```bash
git add mcp_tools/ tests/unit/test_mcp_tools_store.py tests/unit/test_mcp_tools_bodies.py
git commit -m "feat: mcp-tools per-turn store and tool bodies"
```

---

### Task 2: `mcp-tools` HTTP + MCP server surface

**Files:**
- Create: `mcp_tools/server.py`
- Test: `tests/unit/test_mcp_tools_server.py`

**Interfaces:**
- Consumes: `mcp_tools.store`, `mcp_tools.tools` from Task 1.
- Produces: an ASGI `app` (FastAPI) with:
  - `POST /turns` body `{turn_id: str, user_id: int, image_bytes_b64: str | null}` → registers a turn, `204`.
  - `GET /turns/{turn_id}/found_image` → `{found_image_url: str | null}`, pops the entry.
  - MCP endpoint mounted at `/mcp` exposing the four tools (`web_search`, `recall`, `image_search`, `vision_analyze`) with `turn_id` threaded on the three that need it.
- The MCP tool schemas MUST carry the same descriptions as `bot/tools.py` `TOOL_SPECS` (esp. the `image_search` neutral-English-query guidance) so Hermes calls them well.

- [ ] **Step 1: Write the failing test for the HTTP surface**

```python
# tests/unit/test_mcp_tools_server.py
import base64
from fastapi.testclient import TestClient
from mcp_tools.server import app
from mcp_tools import store

def test_post_turn_registers_and_get_found_image_pops():
    client = TestClient(app)
    b64 = base64.b64encode(b"jpeg").decode()
    r = client.post("/turns", json={"turn_id": "t1", "user_id": 7, "image_bytes_b64": b64})
    assert r.status_code == 204
    assert store.get("t1")["image_bytes"] == b"jpeg"
    store.set_found_image("t1", "http://img/1.jpg")
    r = client.get("/turns/t1/found_image")
    assert r.json() == {"found_image_url": "http://img/1.jpg"}
    # entry dropped after pop
    r2 = client.get("/turns/t1/found_image")
    assert r2.json() == {"found_image_url": None}

def test_post_turn_null_image():
    client = TestClient(app)
    r = client.post("/turns", json={"turn_id": "t2", "user_id": 7, "image_bytes_b64": None})
    assert r.status_code == 204
    assert store.get("t2")["image_bytes"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mcp_tools_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_tools.server'`

- [ ] **Step 3: Implement the server**

```python
# mcp_tools/server.py
import base64
from fastapi import FastAPI, Response
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp_tools import store, tools

app = FastAPI()

class TurnIn(BaseModel):
    turn_id: str
    user_id: int
    image_bytes_b64: str | None = None

@app.post("/turns", status_code=204)
def register_turn(t: TurnIn) -> Response:
    img = base64.b64decode(t.image_bytes_b64) if t.image_bytes_b64 else None
    store.register(t.turn_id, t.user_id, img)
    return Response(status_code=204)

@app.get("/turns/{turn_id}/found_image")
def found_image(turn_id: str) -> dict:
    return {"found_image_url": store.pop_found_image(turn_id)}

mcp = FastMCP("persona-tools")

@mcp.tool(description="Search the web for fresh facts.")
async def web_search(query: str) -> str:
    return await tools.web_search(query)

@mcp.tool(description="Recall durable facts remembered about this user.")
async def recall(query: str, turn_id: str) -> str:
    return await tools.recall(query, turn_id)

@mcp.tool(description=("Find an existing image to send. Phrase the query as neutral "
    "descriptive English (e.g. 'portrait of a young woman smiling', 'ginger cat on a "
    "sofa'): slang or compliment-heavy phrasing in any language returns zero results. "
    "If a search finds nothing, retry once with simpler neutral wording."))
async def image_search(query: str, turn_id: str) -> str:
    return await tools.image_search(query, turn_id)

@mcp.tool(description="Describe the photo the user attached this turn.")
async def vision_analyze(turn_id: str) -> str:
    return await tools.vision_analyze(turn_id)

app.mount("/mcp", mcp.sse_app())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mcp_tools_server.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add runtime deps and commit**

Add to `pyproject.toml` `dependencies`: `"fastapi>=0.110"`, `"uvicorn>=0.27"`, `"mcp>=1.0"`. Add to dev deps: `"httpx>=0.27"` already present for TestClient.

```bash
git add mcp_tools/server.py tests/unit/test_mcp_tools_server.py pyproject.toml
git commit -m "feat: mcp-tools HTTP + MCP server surface"
```

---

### Task 3: Hermes SSE client in the bot

**Files:**
- Create: `bot/hermes_client.py`
- Test: `tests/unit/test_hermes_client.py`

**Interfaces:**
- Consumes: `bot.config.settings` (new fields added here).
- Produces:
  - `async call_hermes(messages: list[dict], turn_id: str, model: str | None) -> str` — POSTs to `settings.hermes_url` `/v1/chat/completions` with `stream=true`, threads `turn_id` (as a leading system message `f"turn_id={turn_id}"`), concatenates `delta.content` chunks, ignores `hermes.tool.progress` events, returns the final text.
- New `settings` fields: `hermes_url: str = "http://hermes:8642"`, `mcp_tools_url: str = "http://mcp-tools:8000"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hermes_client.py
import respx, httpx
from bot import hermes_client

SSE = (
    b'event: hermes.tool.progress\ndata: {"tool":"web_search","status":"running"}\n\n'
    b'data: {"choices":[{"delta":{"content":"Sofia"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":", 24."}}]}\n\n'
    b'data: [DONE]\n\n'
)

@respx.mock
async def test_call_hermes_concatenates_content_ignores_progress():
    respx.post("http://hermes:8642/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=SSE,
                                    headers={"content-type": "text/event-stream"}))
    out = await hermes_client.call_hermes(
        [{"role": "user", "content": "chi è la tua fiamma?"}], turn_id="t1", model=None)
    assert out == "Sofia, 24."

@respx.mock
async def test_call_hermes_threads_turn_id_as_system_message():
    captured = {}
    def _capture(request):
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b'data: [DONE]\n\n',
                              headers={"content-type": "text/event-stream"})
    respx.post("http://hermes:8642/v1/chat/completions").mock(side_effect=_capture)
    await hermes_client.call_hermes([{"role": "user", "content": "hi"}], turn_id="t9", model=None)
    msgs = captured["body"]["messages"]
    assert any(m["role"] == "system" and "turn_id=t9" in m["content"] for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hermes_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.hermes_client'`

- [ ] **Step 3: Implement the client**

```python
# bot/hermes_client.py
import json
import httpx
from bot.config import settings

async def call_hermes(messages: list[dict], turn_id: str, model: str | None) -> str:
    body = {
        "model": model or settings.model_agent or settings.model_chat,
        "stream": True,
        "messages": [{"role": "system", "content": f"turn_id={turn_id}"}, *messages],
    }
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{settings.hermes_url}/v1/chat/completions",
                            json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0].get("delta", {})
                if delta.get("content"):
                    parts.append(delta["content"])
    return "".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hermes_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add settings fields and commit**

Add to `bot/config.py` `Settings`: `hermes_url: str = "http://hermes:8642"` and `mcp_tools_url: str = "http://mcp-tools:8000"`.

```bash
git add bot/hermes_client.py tests/unit/test_hermes_client.py bot/config.py
git commit -m "feat: Hermes SSE client with turn_id threading"
```

---

### Task 4: Rewrite `agent_node` to drive the sidecar

**Files:**
- Modify: `bot/nodes/agent.py` (full rewrite of `agent_node`)
- Modify: `tests/unit/test_agent_loop.py` (replace loop-internals tests with sidecar-driven tests)

**Interfaces:**
- Consumes: `bot.hermes_client.call_hermes`, `bot.memory.recent_turns`, `bot.persona.build_agent_briefing`, `settings.mcp_tools_url`, `httpx`.
- Produces: `async agent_node(state) -> dict` returning `{raw_result, found_image_url, agent_error}` — unchanged shape. Uses a per-turn `turn_id` (from `state["user_id"]` + a monotonic counter is NOT allowed — use a stable per-invocation id; see step 3).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_agent_loop.py  (replaces old contents)
from unittest.mock import AsyncMock, patch
from bot.nodes import agent

async def test_agent_node_returns_hermes_content_as_raw_result():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(return_value="You mentioned guitar.")):
        out = await agent.agent_node({"user_id": 7, "user_text": "what do I play?",
                                      "image_bytes": None})
    assert out["raw_result"] == "You mentioned guitar."
    assert out["agent_error"] is None

async def test_agent_node_reads_found_image_url_after_hermes():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image",
               new=AsyncMock(return_value="http://img/1.jpg")), \
         patch("bot.nodes.agent.call_hermes", new=AsyncMock(return_value="here's a cat")):
        out = await agent.agent_node({"user_id": 7, "user_text": "send a cat",
                                      "image_bytes": None})
    assert out["found_image_url"] == "http://img/1.jpg"

async def test_agent_node_registers_turn_with_image_bytes():
    reg = AsyncMock()
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=reg), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes", new=AsyncMock(return_value="ok")):
        await agent.agent_node({"user_id": 7, "user_text": "what is this", "image_bytes": b"x"})
    # register called with (turn_id, user_id=7, image_bytes=b"x")
    _, kwargs = reg.call_args
    assert kwargs["user_id"] == 7 and kwargs["image_bytes"] == b"x"

async def test_agent_node_never_raises():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        out = await agent.agent_node({"user_id": 7, "user_text": "hi", "image_bytes": None})
    assert out["raw_result"] is None and "boom" in out["agent_error"]
    assert out["found_image_url"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_agent_loop.py -v`
Expected: FAIL (`AttributeError: module 'bot.nodes.agent' has no attribute '_register_turn'`)

- [ ] **Step 3: Rewrite `agent_node`**

```python
# bot/nodes/agent.py
import base64
import logging
import uuid

import httpx

from bot.config import settings
from bot.hermes_client import call_hermes
from bot.memory import recent_turns
from bot.persona import build_agent_briefing

logger = logging.getLogger(__name__)


async def _register_turn(turn_id: str, *, user_id: int, image_bytes: bytes | None) -> None:
    b64 = base64.b64encode(image_bytes).decode() if image_bytes else None
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{settings.mcp_tools_url}/turns",
                         json={"turn_id": turn_id, "user_id": user_id, "image_bytes_b64": b64})
        r.raise_for_status()


async def _pop_found_image(turn_id: str) -> str | None:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{settings.mcp_tools_url}/turns/{turn_id}/found_image")
        r.raise_for_status()
        return r.json().get("found_image_url")


async def agent_node(state) -> dict:
    turn_id = uuid.uuid4().hex
    try:
        recent = await recent_turns(state["user_id"], 10)
        messages = [{"role": r["role"], "content": r["content"]} for r in reversed(recent)]
        prompt = state["user_text"]
        if state.get("image_bytes"):
            prompt += "\n\n(The user sent a photo with this message; call vision_analyze to see it.)"
        messages.append({"role": "user", "content": prompt})
        briefing = build_agent_briefing()
        if briefing:
            messages.insert(0, {"role": "system", "content": briefing})

        await _register_turn(turn_id, user_id=state["user_id"],
                             image_bytes=state.get("image_bytes"))
        raw = await call_hermes(messages, turn_id=turn_id,
                                model=settings.model_agent or None)
        found = await _pop_found_image(turn_id)
        return {"raw_result": raw or None, "found_image_url": found, "agent_error": None}
    except Exception as exc:
        logger.exception("agent node failed for user_id=%s", state["user_id"])
        try:
            await _pop_found_image(turn_id)  # best-effort cleanup
        except Exception:
            pass
        return {"raw_result": None, "found_image_url": None, "agent_error": str(exc)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_agent_loop.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Delete the now-dead in-process tool wiring**

`bot/tools.py`, `bot/turn_context.py`, and `tests/unit/test_turn_context.py` are no longer imported by the runtime (tools live in `mcp_tools/`). Confirm no remaining imports, then remove them:

Run: `grep -rn "bot.turn_context\|from bot.tools\|bot\.tools\|TOOL_SPECS\|TOOL_FUNCS" bot/ tests/`
Expected: only hits are in the files being removed (`bot/tools.py`, `tests/unit/test_tools.py`, `tests/unit/test_turn_context.py`). If `bot/nodes/agent.py` still references them, the rewrite in Step 3 was incomplete — fix before deleting.

```bash
git rm bot/tools.py bot/turn_context.py tests/unit/test_tools.py tests/unit/test_turn_context.py
```

(The tool LOGIC survives in `mcp_tools/tools.py` from Task 1 — this only removes the old in-process copies.)

- [ ] **Step 6: Run the full unit suite**

Run: `pytest tests/unit -v`
Expected: PASS. `test_compose_persona.py`, `test_ingress.py`, `test_graph.py` must be green unchanged — proof the boundary held.

- [ ] **Step 7: Commit**

```bash
git add bot/nodes/agent.py tests/unit/test_agent_loop.py
git commit -m "feat: drive Hermes sidecar from agent_node, drop in-process tool loop"
```

---

### Task 5: Container packaging + Hermes config + docker-compose

**Files:**
- Create: `Dockerfile.mcp-tools`
- Create: `deploy/hermes/config.yaml`
- Modify: `docker-compose.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `mcp_tools.server:app` (Task 2), `settings.hermes_url` / `settings.mcp_tools_url` (Task 3).
- Produces: a running three-service stack. `mcp-tools` served by uvicorn on `:8000`; `hermes` on `:8642` configured to reach `mcp-tools` MCP endpoint and allowlist the four tools.

- [ ] **Step 1: Write the mcp-tools Dockerfile**

```dockerfile
# Dockerfile.mcp-tools
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir . uvicorn
COPY bot ./bot
COPY mcp_tools ./mcp_tools
CMD ["uvicorn", "mcp_tools.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write the Hermes config**

`deploy/hermes/config.yaml` — declares the mcp-tools MCP server, restricts Hermes to the four tools, disables mutating built-ins, and instructs terse factual output (silence-guarantee mitigation from the spec):

```yaml
mcp_servers:
  - id: persona_tools
    uri: http://mcp-tools:8000/mcp
agent:
  tools:
    include: [web_search, recall, image_search, vision_analyze]
    disabled_toolsets: [file, patch, execute_code, browser, cronjob, delegate_task, todo, skills]
  system_prefix: >
    Return only terse factual findings for a downstream persona layer to voice.
    Do not narrate your process, mention tools, searches, or that you are an AI.
```

> NOTE (open item from spec): the exact key names (`mcp_servers`, `agent.tools.include`, `system_prefix`) and the `turn_id` correlation mechanism must be pinned against the deployed Hermes image's actual config schema during Step 4. If Hermes cannot forward a per-call `turn_id` argument to MCP tools automatically, fall back to passing it via the system prefix and having each tool default `turn_id` from the most-recent registered turn. Verify before proceeding.

- [ ] **Step 3: Extend docker-compose**

Add to `docker-compose.yml` `services:` (keep existing `db` and `bot`; add `hermes_url`/`mcp_tools_url` env to `bot`, and `depends_on` hermes):

```yaml
  mcp-tools:
    build:
      context: .
      dockerfile: Dockerfile.mcp-tools
    restart: unless-stopped
    env_file:
      - path: .env
        required: false
    environment:
      DATABASE_URL: postgresql+psycopg://bot:bot@db:5432/bot
    depends_on:
      db:
        condition: service_healthy

  hermes:
    image: nousresearch/hermes-agent:0.17.0  # PIN the arm64-supporting tag verified for the Pi
    restart: unless-stopped
    env_file:
      - path: .env
        required: false
    volumes:
      - ./deploy/hermes/config.yaml:/etc/hermes/config.yaml:ro
    depends_on:
      - mcp-tools
```

And add to the `bot` service `environment:`:

```yaml
      HERMES_URL: http://hermes:8642
      MCP_TOOLS_URL: http://mcp-tools:8000
```

- [ ] **Step 4: Local end-to-end validation (docker-compose)**

Run: `docker compose up --build`
Then exercise all four turn types against the bot and confirm:
1. Plain text turn → reply in persona voice.
2. Web-research turn (ask something needing fresh facts) → `web_search` fires (check `hermes` logs for `hermes.tool.progress`), facts appear voiced.
3. Photo-in turn (send a photo, ask about it) → `vision_analyze` fires, description voiced.
4. Photo-out turn (ask Erminio for a picture) → `image_search` fires, a real Telegram photo arrives.

Expected: all four succeed; no tool/search mention leaks into replies (silence guarantee). If Hermes output leaks meta-narration, tighten `system_prefix` and re-test.

- [ ] **Step 5: Update README**

Document the new architecture: three services (`bot`, `hermes`, `mcp-tools`), the tool surface now lives in `mcp_tools/`, the `HERMES_URL`/`MCP_TOOLS_URL` env vars, and the local `docker compose up --build` flow. Remove any README references to the old in-process `bot/tools.py` loop.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.mcp-tools deploy/hermes/config.yaml docker-compose.yml README.md
git commit -m "feat: package mcp-tools + hermes services in docker-compose"
```

---

### Task 6: Pi rollout

**Files:** none (deployment only)

- [ ] **Step 1: Confirm the arm64 image tag**

Run: `docker manifest inspect nousresearch/hermes-agent:0.17.0 | grep -A2 architecture`
Expected: an entry with `"architecture": "arm64"`. If the pinned tag lacks arm64, find the newest tag that has it and update `docker-compose.yml`.

- [ ] **Step 2: Deploy via the existing flow**

rsync the repo to `hermespi.local` and `docker compose up --build -d` (the project's established deploy path). Confirm all three containers are healthy: `docker compose ps`.

- [ ] **Step 3: Live smoke test on the Pi**

Repeat the four turn types from Task 5 Step 4 against the live bot. Watch latency — compare against the old `AGENT_MAX_ITERATIONS=6` behavior; if Hermes' server-side loop is too slow on the Pi, cap its max iterations in `config.yaml` and re-test.

- [ ] **Step 4: Commit any config adjustments**

```bash
git add docker-compose.yml deploy/hermes/config.yaml
git commit -m "chore: pin hermes arm64 tag and tune loop for Pi"
```

---

## Self-Review

**Spec coverage:**
- Three-container architecture → Task 5. ✓
- `agent_node` rewrite to HTTP/SSE → Tasks 3–4. ✓
- Four tools re-exposed via MCP → Tasks 1–2. ✓
- Per-turn in-memory context store + side-channel readback → Tasks 1, 2, 4. ✓
- `compose_persona`/ingress unchanged (verified green) → Task 4 Step 6. ✓
- Silence-guarantee mitigation (terse Hermes output) → Task 5 Step 2 + Step 4. ✓
- Runtime model switching → Task 3 (`model` param, `settings.model_agent`). ✓
- Pi arm64 + no local weights → Task 6. ✓
- Open items (turn_id correlation, config schema, latency) → flagged in Task 5 Step 2 NOTE + Task 6 Step 3. ✓
- README update → Task 5 Step 5. ✓

**Placeholder scan:** The one deferred decision (Hermes config schema / turn_id mechanism) is an genuine external unknown from the spec's Open Items — it is flagged with a concrete fallback in Task 5 Step 2, not left blank. All code steps carry real code.

**Type consistency:** `store.register/get/set_found_image/pop_found_image`, `call_hermes(messages, turn_id, model)`, `_register_turn(turn_id, *, user_id, image_bytes)`, `_pop_found_image(turn_id)`, and the `{raw_result, found_image_url, agent_error}` return shape are used identically across Tasks 1–5. ✓
