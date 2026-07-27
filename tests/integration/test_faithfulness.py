from unittest.mock import AsyncMock, patch

from bot.nodes.compose_persona import compose_persona

BASE = {"user_id": 1, "profile": {"name": "Sam"}, "memories": [],
        "user_text": "what's the capital of Peru?", "found_image_url": None}

OTHER_CAPITALS = ["Quito", "Bogotá", "Santiago", "Buenos Aires", "Brasília"]


def _echo_with_restyle(messages) -> str:
    """Simulates a faithful model: pulls the facts block out of the prompt
    and restyles it in a different sentence shape, but never drops or
    swaps the underlying fact text."""
    facts_block = next(m["content"] for m in messages if m["role"] == "system"
                        and m["content"].startswith("Facts to speak from:"))
    facts = facts_block.split("Facts to speak from:\n", 1)[1].split("\n\n", 1)[0]
    return f"So, to answer that — {facts}, just so you know!"


async def test_fact_survives_llm_restyle():
    raw_result = "the capital of Peru is Lima"
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(side_effect=lambda messages: _echo_with_restyle(messages))) as m:
        out = await compose_persona({**BASE, "raw_result": raw_result})

    assert "Lima" in out["reply"]["text"]
    sent_messages = m.await_args.args[0]
    facts_message = next(msg["content"] for msg in sent_messages if msg["role"] == "system"
                          and msg["content"].startswith("Facts to speak from:"))
    assert raw_result in facts_message


async def test_no_contradictory_fact_invented():
    raw_result = "the capital of Peru is Lima"
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(side_effect=lambda messages: _echo_with_restyle(messages))):
        out = await compose_persona({**BASE, "raw_result": raw_result})

    text = out["reply"]["text"]
    assert "Lima" in text
    for other in OTHER_CAPITALS:
        assert other not in text


async def test_multiple_facts_all_survive_restyle():
    raw_result = "the capital of Peru is Lima; the user's dog is named Toby"
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(side_effect=lambda messages: _echo_with_restyle(messages))):
        out = await compose_persona({**BASE, "raw_result": raw_result})

    text = out["reply"]["text"]
    assert "Lima" in text
    assert "Toby" in text


async def test_fallback_path_never_fabricates_a_fact_when_none_given():
    with patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="I couldn't pull that up right now, sorry!")):
        out = await compose_persona({**BASE, "raw_result": None, "agent_error": "boom"})

    text = out["reply"]["text"]
    for other in ["Lima", "Quito"] + OTHER_CAPITALS:
        assert other not in text
