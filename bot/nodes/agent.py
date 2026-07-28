import json
import logging
from bot import llm, memory
from bot.config import settings
from bot.persona import build_agent_briefing
from bot.tools import TOOL_SPECS, TOOL_FUNCS
from bot.turn_context import turn_context

logger = logging.getLogger(__name__)

async def agent_node(state) -> dict:
    token = turn_context.set({"image_bytes": state.get("image_bytes"),
                              "found_image_url": None, "user_id": state["user_id"]})
    try:
        recent = await memory.recent_turns(state["user_id"], 10)
        messages = [{"role": r["role"], "content": r["content"]} for r in reversed(recent)]
        prompt = state["user_text"]
        if state.get("image_bytes"):
            prompt += "\n\n(The user sent a photo with this message; call vision_analyze to see it.)"
        messages.append({"role": "user", "content": prompt})
        briefing = build_agent_briefing()
        if briefing:
            messages.insert(0, {"role": "system", "content": briefing})

        for _ in range(settings.agent_max_iterations):
            msg = await llm.chat_with_tools(messages, TOOL_SPECS,
                                            model=settings.model_agent or None)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return {"raw_result": msg.get("content"), "agent_error": None,
                        "found_image_url": turn_context.get()["found_image_url"]}
            messages.append(msg)
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    fn = TOOL_FUNCS[name]
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    result = await fn(**args)
                except Exception as exc:
                    # A failing tool (bad key, provider down, malformed args)
                    # must not kill the whole turn: tell the model it failed
                    # and let it finish the reply without that data.
                    logger.warning("tool %s failed: %s", name, exc)
                    result = f"(tool {name} failed; answer without it)"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        # iteration budget exhausted: return the last content if any
        return {"raw_result": messages[-1].get("content"), "agent_error": None,
                "found_image_url": turn_context.get()["found_image_url"]}
    except Exception as exc:
        logger.exception("agent node failed for user_id=%s", state["user_id"])
        return {"raw_result": None, "found_image_url": None, "agent_error": str(exc)}
    finally:
        turn_context.reset(token)
