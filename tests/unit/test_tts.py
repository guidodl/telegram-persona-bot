import json
import respx, httpx
from unittest.mock import patch
from bot import tts
from bot.config import settings


@respx.mock
async def test_synth_calls_openrouter_and_returns_bytes():
    route = respx.post("https://openrouter.ai/api/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"MP3audio"))
    assert await tts.synth("hello there") == b"MP3audio"
    sent = route.calls.last.request
    assert b'"response_format":"mp3"' in sent.content.replace(b" ", b"")


@respx.mock
async def test_synth_omits_voice_when_unset():
    route = respx.post("https://openrouter.ai/api/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"MP3audio"))
    with patch.object(settings, "tts_voice", ""):
        await tts.synth("hello there")
    payload = json.loads(route.calls.last.request.content)
    assert "voice" not in payload
