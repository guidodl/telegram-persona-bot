import respx, httpx, pytest
from bot import llm
from bot.config import settings


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


@respx.mock
async def test_chat_with_tools_posts_tools_and_returns_raw_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"}}],
            }}]
        }))
    out = await llm.chat_with_tools([{"role": "user", "content": "yo"}], tools)
    assert route.called
    assert route.calls.last.request.content
    import json
    sent = json.loads(route.calls.last.request.content)
    assert sent["tools"] == tools
    assert out["tool_calls"][0]["function"]["name"] == "lookup"


@respx.mock
async def test_post_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    sleeps = []

    async def _fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}),
        ])
    out = await llm.chat([{"role": "user", "content": "yo"}])
    assert out == "hi"
    assert route.call_count == 3
    assert len(sleeps) == 2


@respx.mock
async def test_post_raises_last_error_after_all_retries_fail(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    async def _fake_sleep(s):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPError):
        await llm.chat([{"role": "user", "content": "yo"}])
    assert route.call_count == 3


@respx.mock
async def test_chat_retries_on_null_content_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    async def _fake_sleep(s):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]}),
        ])
    out = await llm.chat([{"role": "user", "content": "yo"}])
    assert out == "hi there"
    assert route.call_count == 2


@respx.mock
async def test_chat_strips_control_token_and_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    async def _fake_sleep(s):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {
                "content": "<｜end▁of▁sentence｜>"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "real reply"}}]}),
        ])
    out = await llm.chat([{"role": "user", "content": "yo"}])
    assert out == "real reply"
    assert route.call_count == 2


@respx.mock
async def test_chat_returns_empty_string_never_none_when_all_degenerate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    async def _fake_sleep(s):
        return None

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    out = await llm.chat([{"role": "user", "content": "yo"}])
    assert out == ""


@respx.mock
async def test_embed_uses_openai_backend_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "embed_backend", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "ok")
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.3, 0.4]}]}))
    vecs = await llm.embed(["hello"])
    assert vecs == [[0.3, 0.4]]
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer ok"
    import json
    sent = json.loads(request.content)
    assert sent["model"] == "text-embedding-3-small"
