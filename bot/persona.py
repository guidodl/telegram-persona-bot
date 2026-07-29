import functools
import logging
from pathlib import Path

from bot.config import settings

logger = logging.getLogger(__name__)

PERSONA_PREAMBLE = (
    "You are a warm, present companion having a natural conversation with someone you "
    "know well. Speak as yourself, in your own voice — never as an assistant."
)

SILENCE_RULES = (
    "Rules you must never break:\n"
    "- Never mention tools, lookups, databases, retrieval, or your own reasoning process. "
    "The person you're talking to must never know you used any tool.\n"
    "- Never hedge with phrases like \"As an AI\" or \"As a language model\" — you are not "
    "describing what you are, you are just talking.\n"
    "- Do not use markdown-heavy formatting (no headers, bullet lists, or bold text) — write "
    "like a person texting, in plain flowing sentences.\n"
    "- Restyle the facts below faithfully in your own voice: never drop a known fact and "
    "never invent one that isn't given to you."
)


AGENT_BRIEFING = (
    "You are the fact-gathering engine behind a Telegram persona bot. Another model "
    "rewrites your output into the final in-character reply — you never write "
    "user-facing text yourself. Use your tools to fetch whatever the persona's reply "
    "will need: fresh facts via web_search, remembered details via recall, photos via "
    "image_search, and the contents of a shared link via web_extract. When the message "
    "contains a URL — including one carried in a quoted \"[Replying to a message from "
    "...]\" block — always call web_extract on it before answering, even when the "
    "question is as vague as \"what do you think?\": the link is the subject. "
    "Whenever the user's message asks for a photo or picture — directly "
    "or via a persona trigger that answers with a picture — call image_search "
    "immediately: never ask clarifying questions instead of searching, and if a search "
    "finds nothing retry once with simpler wording. Use a query specific enough to "
    "guarantee the right subject, then return minimal raw facts. Never mention tools "
    "in your output.\n\nPersona definition:\n"
)


@functools.lru_cache(maxsize=1)
def _read_persona_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_persona_preamble() -> str:
    """Deployment-specific persona from PERSONA_FILE, else the generic preamble."""
    if settings.persona_file:
        try:
            return _read_persona_file(settings.persona_file)
        except OSError:
            logger.warning("persona_file %r unreadable, using generic preamble",
                           settings.persona_file)
    return PERSONA_PREAMBLE


def build_agent_briefing() -> str | None:
    """System message for the agent (fact-gathering) node when a persona file is
    configured; None for the generic persona, which leaves the agent persona-free."""
    if not settings.persona_file:
        return None
    return AGENT_BRIEFING + load_persona_preamble()


def build_system_prompt(profile: dict, memories: list[str]) -> str:
    sections = [load_persona_preamble()]

    known = []
    name = profile.get("name")
    if name:
        known.append(f"Name: {name}")
    preferences = profile.get("preferences")
    if preferences:
        known.append(f"Preferences: {preferences}")
    tone = profile.get("tone")
    if tone:
        known.append(f"Tone to use: {tone}")
    summary = profile.get("summary")
    if summary:
        known.append(f"Running summary of who they are: {summary}")
    if known:
        sections.append("What you know about this person:\n" + "\n".join(known))

    if memories:
        sections.append("Relevant memories to draw on:\n" + "\n".join(f"- {m}" for m in memories))

    sections.append(SILENCE_RULES)

    return "\n\n".join(sections)
