import asyncio, re, httpx
from bot.config import settings

OPENROUTER = "https://openrouter.ai/api/v1"
OPENAI = "https://api.openai.com/v1"


async def _post(url: str, headers: dict, payload: dict) -> dict:
    last = None
    async with httpx.AsyncClient(timeout=60) as c:
        for attempt in range(3):
            try:
                r = await c.post(url, headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last = e
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
    raise last


def _or_headers() -> dict:
    return {"Authorization": f"Bearer {settings.openrouter_api_key}"}


# Some models (e.g. deepseek reasoning variants) intermittently return a
# null/empty completion or a raw end-of-sequence control token instead of
# prose. Retrying the same request almost always yields real text.
_CONTROL_TOKEN_RE = re.compile(r"<[｜|].*?[｜|]>")


def _usable_content(msg: dict) -> str:
    content = msg.get("content") or ""
    content = _CONTROL_TOKEN_RE.sub("", content).strip()
    return content


async def chat(messages: list[dict], model: str | None = None) -> str:
    payload = {"model": model or settings.model_chat, "messages": messages}
    text = ""
    for _ in range(3):
        data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(), payload)
        text = _usable_content(data["choices"][0]["message"])
        if text:
            return text
    return text


async def chat_with_tools(messages: list[dict], tools: list[dict],
                          model: str | None = None) -> dict:
    data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(),
                       {"model": model or settings.model_chat, "messages": messages,
                        "tools": tools})
    return data["choices"][0]["message"]  # raw assistant msg; may carry tool_calls


async def chat_vision(messages: list[dict], model: str | None = None) -> str:
    data = await _post(f"{OPENROUTER}/chat/completions", _or_headers(),
                       {"model": model or settings.model_vision, "messages": messages})
    return data["choices"][0]["message"]["content"]


async def embed(texts: list[str]) -> list[list[float]]:
    if settings.embed_backend == "openai":
        url, headers, model = f"{OPENAI}/embeddings", \
            {"Authorization": f"Bearer {settings.openai_api_key}"}, "text-embedding-3-small"
    else:
        url, headers, model = f"{OPENROUTER}/embeddings", _or_headers(), settings.model_embed
    data = await _post(url, headers, {"model": model, "input": texts})
    return [row["embedding"] for row in data["data"]]
