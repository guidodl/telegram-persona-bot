import httpx
from bot.config import settings

OPENAI = "https://api.openai.com/v1"


async def synth(text: str) -> bytes:
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    payload = {"model": settings.tts_model, "input": text, "voice": "alloy",
               "response_format": "opus"}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{OPENAI}/audio/speech", headers=headers, json=payload)
        r.raise_for_status()
        return r.content
