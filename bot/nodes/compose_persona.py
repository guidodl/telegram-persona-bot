import re
from bot import llm
from bot.persona import build_system_prompt

FALLBACK_INSTRUCTION = (
    "You tried to look something up for them just now but it didn't come through — "
    "no facts were retrieved. Apologize briefly and warmly, in character, that you "
    "couldn't pull that up right now, without explaining why or mentioning any tool, "
    "error, or technical detail."
)

VOICE_INSTRUCTION = (
    "If a short spoken voice note would land better than text for this reply, say so "
    "by starting your reply with the exact tag [VOICE] followed by the message; "
    "otherwise just reply normally with no tag."
)

_MARKDOWN_RE = re.compile(r"[*_`#]+")


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_RE.sub("", text).strip()


async def compose_persona(state) -> dict:
    system_prompt = build_system_prompt(state["profile"], state["memories"])
    raw_result = state.get("raw_result")
    agent_error = state.get("agent_error")

    if agent_error or not raw_result:
        facts_message = FALLBACK_INSTRUCTION
    else:
        facts_message = f"Facts to speak from:\n{raw_result}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_text"]},
        {"role": "system", "content": f"{facts_message}\n\n{VOICE_INSTRUCTION}"},
    ]

    text = await llm.chat(messages)

    voice = False
    if text.strip().startswith("[VOICE]"):
        voice = True
        text = text.strip()[len("[VOICE]"):]

    text = _strip_markdown(text)

    image_url = None if (agent_error or not raw_result) else state.get("found_image_url")

    return {"reply": {"text": text, "voice": voice, "image_url": image_url}}
