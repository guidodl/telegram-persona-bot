from unittest.mock import AsyncMock, patch
from bot.nodes import agent

def _msg(content=None, tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}

async def test_loop_dispatches_tool_then_returns_final():
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "recall", "arguments": '{"query":"music"}'}}]),
        _msg(content="You mentioned you're learning guitar."),
    ]
    with patch("bot.nodes.agent.llm.chat_with_tools",
               new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS",
                    {"recall": AsyncMock(return_value="- learning guitar")}, clear=False):
        out = await agent.agent_node({"chat_id": 7, "user_text": "what do I play?",
                                      "image_bytes": None})
    assert out["raw_result"] == "You mentioned you're learning guitar."

async def test_loop_sets_hint_and_found_image_url():
    async def img_search(query):  # simulate tool writing to turn context
        from bot.turn_context import turn_context
        turn_context.get()["found_image_url"] = "http://img/1.jpg"
        return "found an image"
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "image_search", "arguments": '{"query":"cat"}'}}]),
        _msg(content="here's a cat"),
    ]
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS", {"image_search": img_search}, clear=False):
        out = await agent.agent_node({"chat_id": 7, "user_text": "send a cat", "image_bytes": None})
    assert out["found_image_url"] == "http://img/1.jpg"

async def test_photo_adds_vision_hint_to_first_user_message():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        await agent.agent_node({"chat_id": 7, "user_text": "what is this", "image_bytes": b"x"})
    assert any("vision_analyze" in str(m.get("content", "")) for m in captured["msgs"])

async def test_agent_node_never_raises():
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=RuntimeError("boom"))), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        out = await agent.agent_node({"chat_id": 7, "user_text": "hi", "image_bytes": None})
    assert out["raw_result"] is None and "boom" in out["agent_error"]
