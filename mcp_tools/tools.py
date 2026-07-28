import base64, httpx
from bot import llm, memory
from bot.config import settings
from mcp_tools import store

async def web_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.tavily.com/search",
                         json={"api_key": settings.tavily_api_key, "query": query,
                               "max_results": 3})
        r.raise_for_status()
        results = r.json().get("results", [])
    return "\n".join(f"- {x.get('content','')}" for x in results) or "(no results)"

async def recall(query: str, turn_id: str) -> str:
    try:
        uid = store.get(turn_id)["user_id"]
    except KeyError:
        # Hermes decides turn_id from the model; a wrong/missing one must not surface as an error
        return "(nothing relevant remembered)"
    mems = await memory.search_memories(uid, query)
    return "\n".join(f"- {m}" for m in mems) if mems else "(nothing relevant remembered)"

async def image_search(query: str, turn_id: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.search.brave.com/res/v1/images/search",
                        headers={"X-Subscription-Token": settings.brave_api_key},
                        params={"q": query, "safesearch": "strict", "count": 1})
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "no suitable image found"
    try:
        store.set_found_image(turn_id, results[0]["properties"]["url"])
    except KeyError:
        return "no suitable image found"
    return f"found an image for '{query}'"

async def vision_analyze(turn_id: str) -> str:
    try:
        img = store.get(turn_id)["image_bytes"]
    except KeyError:
        return "no image was attached"
    if not img:
        return "no image was attached"
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
    return await llm.chat_vision([{"role": "user", "content": [
        {"type": "text", "text": "Describe this image factually."},
        {"type": "image_url", "image_url": {"url": data_url}}]}])
