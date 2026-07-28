import json
import httpx
from bot.config import settings

async def call_hermes(messages: list[dict], turn_id: str, model: str | None) -> str:
    body = {
        "model": model or settings.model_agent or settings.model_chat,
        "stream": True,
        "messages": [{"role": "system", "content": f"turn_id={turn_id}"}, *messages],
    }
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{settings.hermes_url}/v1/chat/completions",
                            json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                choices = json.loads(payload).get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    parts.append(delta["content"])
    return "".join(parts)
