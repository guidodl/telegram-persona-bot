import json
import httpx
from bot.config import settings

async def call_hermes(messages: list[dict], turn_id: str, model: str | None) -> str:
    body = {
        "model": model or settings.model_agent or settings.model_chat,
        "stream": True,
        # System line carries turn_id so the model passes it as the tool-call
        # argument our mcp-tools require; the header below scopes Hermes' own
        # session. Both are needed — the header alone does not reach tool args.
        "messages": [{"role": "system", "content": f"turn_id={turn_id}"}, *messages],
    }
    headers = {"X-Hermes-Session-Id": turn_id}
    if settings.hermes_api_key:
        headers["Authorization"] = f"Bearer {settings.hermes_api_key}"
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{settings.hermes_url}/v1/chat/completions",
                            json=body, headers=headers) as r:
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
