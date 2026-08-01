import httpx
from bot.config import settings

OPENROUTER = "https://openrouter.ai/api/v1"


async def synth(text: str) -> bytes:
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    payload = {"model": settings.tts_model, "input": text, "response_format": "mp3"}
    if settings.tts_voice:
        payload["voice"] = settings.tts_voice
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{OPENROUTER}/audio/speech", headers=headers, json=payload)
        r.raise_for_status()
        return r.content
