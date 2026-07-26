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
