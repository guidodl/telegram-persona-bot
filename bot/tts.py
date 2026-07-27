import httpx
from bot.config import settings

OPENROUTER = "https://openrouter.ai/api/v1"


async def synth(text: str) -> bytes:
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    payload = {"model": settings.tts_model, "input": text,
               "voice": settings.tts_voice, "response_format": "mp3"}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{OPENROUTER}/audio/speech", headers=headers, json=payload)
        r.raise_for_status()
        return r.content
