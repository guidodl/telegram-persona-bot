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


def build_system_prompt(profile: dict, memories: list[str]) -> str:
    sections = [PERSONA_PREAMBLE]

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
