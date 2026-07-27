from unittest.mock import AsyncMock, patch
from bot.nodes.load_memory import load_memory

async def test_load_memory_merges_profile_and_memories():
    with patch("bot.nodes.load_memory.memory.get_profile",
               new=AsyncMock(return_value={"name": "Sam"})), \
         patch("bot.nodes.load_memory.memory.search_memories",
               new=AsyncMock(return_value=["m1", "m2"])):
        out = await load_memory({"chat_id": 1, "user_text": "hey"})
    assert out == {"profile": {"name": "Sam"}, "memories": ["m1", "m2"]}
