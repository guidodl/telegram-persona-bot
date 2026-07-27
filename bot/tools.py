import base64, httpx
from bot import llm, memory
from bot.config import settings
from bot.turn_context import turn_context

async def web_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post("https://api.tavily.com/search",
                         json={"api_key": settings.tavily_api_key, "query": query,
                               "max_results": 3})
        r.raise_for_status()
        results = r.json().get("results", [])
    return "\n".join(f"- {x.get('content','')}" for x in results) or "(no results)"

async def recall(query: str) -> str:
    uid = turn_context.get()["user_id"]
    mems = await memory.search_memories(uid, query)
    return "\n".join(f"- {m}" for m in mems) if mems else "(nothing relevant remembered)"

async def image_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.search.brave.com/res/v1/images/search",
                        headers={"X-Subscription-Token": settings.brave_api_key},
                        params={"q": query, "safesearch": "strict", "count": 1})
        r.raise_for_status()
        results = r.json().get("results", [])
    if not results:
        return "no suitable image found"
    turn_context.get()["found_image_url"] = results[0]["properties"]["url"]
    return f"found an image for '{query}'"

async def vision_analyze() -> str:
    img = turn_context.get()["image_bytes"]
    if not img:
        return "no image was attached"
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
    return await llm.chat_vision([{"role": "user", "content": [
        {"type": "text", "text": "Describe this image factually."},
        {"type": "image_url", "image_url": {"url": data_url}}]}])

def _spec(name, desc, props=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": list((props or {}).keys())}}}

TOOL_SPECS = [
    _spec("web_search", "Search the web for fresh facts.", {"query": {"type": "string"}}),
    _spec("recall", "Recall durable facts remembered about this user.", {"query": {"type": "string"}}),
    _spec("image_search", "Find an existing image to send.", {"query": {"type": "string"}}),
    _spec("vision_analyze", "Describe the photo the user attached this turn."),
]
TOOL_FUNCS = {"web_search": web_search, "recall": recall,
              "image_search": image_search, "vision_analyze": vision_analyze}
