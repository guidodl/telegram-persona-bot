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

# Matches the [VOICE] tag regardless of leading punctuation/markdown wrapper
# left over from the model's formatting (e.g. "**[VOICE]**", "- [VOICE]").
# Runs AFTER markdown stripping so wrapper characters are already gone; the
# leftover \W* guards against stray punctuation/whitespace.
_VOICE_TAG_RE = re.compile(r"^\W*\[voice\]\s*", re.IGNORECASE)

# Defense-in-depth only: the real silence guarantees are the tool-free speak
# call (no `tools` kwarg ever reaches the model here) and the Task-13
# send-guard. This is a conservative, leading-position-only backstop in case
# the model still leaks an obvious tool-tell despite the persona prompt's
# SILENCE_RULES. Deliberately narrow to avoid false positives on legitimate
# prose.
# \b after each alternative's final word prevents matching into a longer
# word (e.g. "as an ai" must not match the start of "air"/"aid"; "search"
# must not match the start of "searches"; "tool" must not match the start
# of "toolkit") — the phrase must be a complete word/clause, not a prefix.
_JARGON_RE = re.compile(
    r"^(?:as an ai\b|i (?:just )?looked that up\b|according to my (?:search|tool)\b|let me search\b)"
    r"[,:]?\s*",
    re.IGNORECASE,
)


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_RE.sub("", text).strip()


def _extract_voice_tag(text: str) -> tuple[str, bool]:
    new_text, matched = _VOICE_TAG_RE.subn("", text, count=1)
    return new_text, matched > 0


def _strip_jargon(text: str) -> str:
    return _JARGON_RE.sub("", text, count=1)


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

    text = await llm.chat(messages) or ""

    text = _strip_markdown(text)
    text, voice = _extract_voice_tag(text)
    text = _strip_jargon(text).strip()

    # A [VOICE] tag with no message after it strips down to empty text, which
    # the send-guard rejects — the user gets the tongue-tied fallback. Recompose
    # once as plain text (drop the voice offer) so a real reply still lands.
    if voice and not text:
        messages[-1]["content"] = facts_message
        text = _strip_jargon(_strip_markdown(await llm.chat(messages) or "")).strip()
        voice = False

    image_url = None if (agent_error or not raw_result) else state.get("found_image_url")

    return {"reply": {"text": text, "voice": voice, "image_url": image_url}}
