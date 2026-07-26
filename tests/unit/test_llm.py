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
