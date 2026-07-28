import json
import logging
import re
from bot import llm, memory

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

EXTRACTION_PROMPT = (
    "Extract any new, durable facts about the user from this exchange — "
    "stable preferences, ongoing projects, relationships, goals, or traits "
    "that would still be true weeks from now. Skip small talk, one-off "
    "questions, and anything already obvious from context. "
    "Reply with ONLY a JSON list of short fact strings, e.g. "
    '["user is learning guitar"]. If there is nothing durable to record, '
    "reply with []. If (and only if) these facts meaningfully change the "
    "user's overall profile, you may instead reply with a JSON object "
    '{"facts": [...], "summary": "<updated one-sentence profile summary>"}.'
)


def _parse_extraction(raw: str) -> tuple[list[str], str | None]:
    raw = _FENCE_RE.sub("", raw.strip()).strip()
    if not raw:
        return [], None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("persist_memory got non-JSON extraction output: %r", raw[:200])
        return [], None
    if isinstance(data, list):
        items = data
        summary = None
    elif isinstance(data, dict):
        items = data.get("facts", [])
        summary = data.get("summary")
        summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
    else:
        return [], None
    facts = [f.strip() for f in items if isinstance(f, str) and f.strip()] if isinstance(items, list) else []
    return facts, summary


async def persist_memory(user_id: int, user_text: str, reply_text: str, recent: list[dict]) -> None:
    try:
        history = [{"role": r["role"], "content": r["content"]} for r in reversed(recent)]
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            *history,
            {"role": "user", "content": f"User: {user_text}\nAssistant: {reply_text}"},
        ]
        raw = await llm.chat(messages)
        facts, summary = _parse_extraction(raw)
        if facts:
            await memory.upsert_memories(user_id, facts)
        if summary:
            await memory.update_profile_summary(user_id, summary)
    except Exception:
        logger.exception("persist_memory failed for user_id=%s", user_id)
