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


async def test_voice_tag_sets_voice_true_and_is_stripped_from_text():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="[VOICE] hey there")):
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert out["reply"]["voice"] is True
    assert "[VOICE]" not in out["reply"]["text"]
    assert "[voice]" not in out["reply"]["text"].lower()


async def test_adorned_voice_tag_still_detected_and_stripped():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="**[VOICE]** hey")):
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert out["reply"]["voice"] is True
    assert "[voice]" not in out["reply"]["text"].lower()


async def test_no_voice_tag_leaves_voice_false_and_text_unchanged():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="just some regular reply")):
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert out["reply"]["voice"] is False
    assert out["reply"]["text"] == "just some regular reply"


async def test_leading_tool_jargon_is_stripped():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="I looked that up, it's sunny out!")):
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert "looked that up" not in out["reply"]["text"].lower()
    assert out["reply"]["text"] == "it's sunny out!"


async def test_ordinary_prose_is_left_intact():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="it's sunny and warm today")):
        out = await compose_persona({**BASE, "raw_result": "temp 24C clear"})
    assert out["reply"]["text"] == "it's sunny and warm today"
