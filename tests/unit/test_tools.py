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
