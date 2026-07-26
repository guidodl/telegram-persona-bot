from unittest.mock import AsyncMock, patch
from bot.nodes.persist_memory import persist_memory


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
