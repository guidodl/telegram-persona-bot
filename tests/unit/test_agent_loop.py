from unittest.mock import AsyncMock, patch
from bot.nodes import agent

async def test_agent_node_returns_hermes_content_as_raw_result():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(return_value="You mentioned guitar.")):
        out = await agent.agent_node({"user_id": 7, "user_text": "what do I play?",
                                      "image_bytes": None})
    assert out["raw_result"] == "You mentioned guitar."
    assert out["agent_error"] is None

async def test_agent_node_reads_found_image_url_after_hermes():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image",
               new=AsyncMock(return_value="http://img/1.jpg")), \
         patch("bot.nodes.agent.call_hermes", new=AsyncMock(return_value="here's a cat")):
        out = await agent.agent_node({"user_id": 7, "user_text": "send a cat",
                                      "image_bytes": None})
    assert out["found_image_url"] == "http://img/1.jpg"

async def test_agent_node_registers_turn_with_image_bytes():
    reg = AsyncMock()
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=reg), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes", new=AsyncMock(return_value="ok")):
        await agent.agent_node({"user_id": 7, "user_text": "what is this", "image_bytes": b"x"})
    # register called with (turn_id, user_id=7, image_bytes=b"x")
    _, kwargs = reg.call_args
    assert kwargs["user_id"] == 7 and kwargs["image_bytes"] == b"x"

async def test_agent_node_never_raises():
    with patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        out = await agent.agent_node({"user_id": 7, "user_text": "hi", "image_bytes": None})
    assert out["raw_result"] is None and "boom" in out["agent_error"]
    assert out["found_image_url"] is None
