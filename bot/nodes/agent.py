import base64
import logging
import uuid

import httpx

from bot.config import settings
from bot.hermes_client import call_hermes
from bot.memory import recent_turns
from bot.persona import build_agent_briefing

logger = logging.getLogger(__name__)


async def _register_turn(turn_id: str, *, user_id: int, image_bytes: bytes | None) -> None:
    b64 = base64.b64encode(image_bytes).decode() if image_bytes else None
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{settings.mcp_tools_url}/turns",
                         json={"turn_id": turn_id, "user_id": user_id, "image_bytes_b64": b64})
        r.raise_for_status()


async def _pop_found_image(turn_id: str) -> str | None:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{settings.mcp_tools_url}/turns/{turn_id}/found_image")
        r.raise_for_status()
        return r.json().get("found_image_url")


async def agent_node(state) -> dict:
    turn_id = uuid.uuid4().hex
    try:
        recent = await recent_turns(state["user_id"], 10)
        messages = [{"role": r["role"], "content": r["content"]} for r in reversed(recent)]
        prompt = state["user_text"]
        if state.get("image_bytes"):
            prompt += "\n\n(The user sent a photo with this message; call vision_analyze to see it.)"
        messages.append({"role": "user", "content": prompt})
        briefing = build_agent_briefing()
        if briefing:
            messages.insert(0, {"role": "system", "content": briefing})

        await _register_turn(turn_id, user_id=state["user_id"],
                             image_bytes=state.get("image_bytes"))
        raw = await call_hermes(messages, turn_id=turn_id,
                                model=settings.model_agent or None)
        found = await _pop_found_image(turn_id)
        return {"raw_result": raw or None, "found_image_url": found, "agent_error": None}
    except Exception as exc:
        logger.exception("agent node failed for user_id=%s", state["user_id"])
        try:
            await _pop_found_image(turn_id)  # best-effort cleanup
        except Exception:
            pass
        return {"raw_result": None, "found_image_url": None, "agent_error": str(exc)}
