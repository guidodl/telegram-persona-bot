from unittest.mock import AsyncMock, patch

import httpx
import respx

from bot import ingress


@respx.mock
async def test_voice_flagged_reply_synthesizes_real_audio_and_sends_audio():
    """voice-flagged reply -> real tts.synth (HTTP mocked) + real media.to_audio
    -> ingress hands the actual synthesized bytes to sendAudio."""
    respx.post("https://openrouter.ai/api/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"MP3VOICEBYTES"))

    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hey there",
                                                       "voice": True, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)

    ctx.bot.send_audio.assert_awaited_once()
    kwargs = ctx.bot.send_audio.await_args.kwargs
    assert kwargs["chat_id"] == 5
    assert kwargs["audio"].input_file_content == b"MP3VOICEBYTES"
    assert kwargs["audio"].filename == "voice.mp3"


@respx.mock
async def test_image_search_found_url_flows_to_send_photo():
    """a reply carrying an image_url (as set by the Hermes sidecar's image_search
    tool) drives a real sendPhoto call with the fetched bytes."""
    respx.get("http://img/bike.jpg").mock(return_value=httpx.Response(200, content=b"bikebytes"))

    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "here you go", "voice": False,
                                                       "image_url": "http://img/bike.jpg"}})
        update, ctx = ingress._fake_text_update("find a bike pic", chat_id=5)
        await ingress.on_message(update, ctx)

    ctx.bot.send_photo.assert_awaited_once()
    assert ctx.bot.send_photo.call_args.kwargs["chat_id"] == 5
