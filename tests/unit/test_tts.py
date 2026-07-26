import respx, httpx
from unittest.mock import patch
from bot import tts


@respx.mock
async def test_synth_calls_openai_and_returns_bytes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    respx.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"OGGaudio"))
    assert await tts.synth("hello there") == b"OGGaudio"
