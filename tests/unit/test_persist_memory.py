from unittest.mock import AsyncMock, patch
from bot.nodes.persist_memory import _parse_extraction, persist_memory


def test_parse_extraction_handles_empty_output():
    assert _parse_extraction("") == ([], None)


def test_parse_extraction_handles_malformed_json():
    assert _parse_extraction("not json") == ([], None)


def test_parse_extraction_strips_markdown_fence():
    facts, summary = _parse_extraction('```json\n["user is learning guitar"]\n```')
    assert facts == ["user is learning guitar"]
    assert summary is None


async def test_extracts_facts_and_upserts():
    with patch("bot.nodes.persist_memory.llm.chat",
               new=AsyncMock(return_value='["user is learning guitar"]')), \
         patch("bot.nodes.persist_memory.memory.upsert_memories",
               new=AsyncMock()) as up, \
         patch("bot.nodes.persist_memory.memory.update_profile_summary", new=AsyncMock()):
        await persist_memory(1, "I started guitar lessons", "nice!", [])
    up.assert_awaited_once()
    assert "guitar" in up.call_args.args[1][0]


async def test_errors_are_swallowed():
    with patch("bot.nodes.persist_memory.llm.chat",
               new=AsyncMock(side_effect=RuntimeError("x"))):
        await persist_memory(1, "hi", "hey", [])  # must not raise


async def test_malformed_extraction_skips_upsert_without_raising():
    with patch("bot.nodes.persist_memory.llm.chat", new=AsyncMock(return_value="")), \
         patch("bot.nodes.persist_memory.memory.upsert_memories", new=AsyncMock()) as up, \
         patch("bot.nodes.persist_memory.memory.update_profile_summary", new=AsyncMock()) as us:
        await persist_memory(1, "hi", "hey", [])  # must not raise
    up.assert_not_awaited()
    us.assert_not_awaited()
