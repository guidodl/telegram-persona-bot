import base64
from unittest.mock import AsyncMock, patch

import httpx
import respx

from bot import ingress, tools
from bot.nodes import agent
from bot.turn_context import turn_context


def _msg(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


async def test_photo_bytes_flow_through_turn_context_into_vision_analyze():
    """photo -> state["image_bytes"] -> turn_context -> real vision_analyze tool
    -> its description feeds back into the agent loop's final answer."""
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "vision_analyze", "arguments": "{}"}}]),
        _msg(content="That's a red bicycle leaning against a brick wall."),
    ]
    image_bytes = b"\x89PNGrawbytes"
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.tools.llm.chat_vision", new=AsyncMock(return_value="a red bicycle")) as vision_mock:
        out = await agent.agent_node({"user_id": 1, "user_text": "what's in this photo?",
                                      "image_bytes": image_bytes})

    assert out["raw_result"] == "That's a red bicycle leaning against a brick wall."
    vision_mock.assert_awaited_once()
    sent_messages = vision_mock.await_args.args[0]
    content = sent_messages[0]["content"]
    image_part = next(p for p in content if p["type"] == "image_url")
    expected_b64 = base64.b64encode(image_bytes).decode()
    assert image_part["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"


@respx.mock
async def test_voice_flagged_reply_synthesizes_real_audio_and_sends_voice(monkeypatch):
    """voice-flagged reply -> real tts.synth (HTTP mocked) + real media.to_voice
    -> ingress hands the actual synthesized bytes to sendVoice."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    respx.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"OGGVOICEBYTES"))

    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hey there",
                                                       "voice": True, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)

    ctx.bot.send_voice.assert_awaited_once()
    kwargs = ctx.bot.send_voice.await_args.kwargs
    assert kwargs["chat_id"] == 5
    assert kwargs["voice"].input_file_content == b"OGGVOICEBYTES"
    assert kwargs["voice"].filename == "voice.ogg"


@respx.mock
async def test_image_search_found_url_flows_to_send_photo(monkeypatch):
    """real tools.image_search (HTTP mocked) sets found_image_url via turn_context;
    that url reaching state["reply"]["image_url"] drives a real sendPhoto call."""
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    turn_context.set({"image_bytes": None, "found_image_url": None, "user_id": 1})
    respx.get("https://api.search.brave.com/res/v1/images/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"properties": {"url": "http://img/bike.jpg"}}]}))

    await tools.image_search("red bicycle")
    found_url = turn_context.get()["found_image_url"]
    assert found_url == "http://img/bike.jpg"

    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "here you go", "voice": False,
                                                       "image_url": found_url}})
        update, ctx = ingress._fake_text_update("find a bike pic", chat_id=5)
        await ingress.on_message(update, ctx)

    ctx.bot.send_photo.assert_awaited_once_with(chat_id=5, photo=found_url)
