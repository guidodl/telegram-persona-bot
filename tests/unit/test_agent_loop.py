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
        out = await agent.agent_node({"user_id": 7, "user_text": "what do I play?",
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
        out = await agent.agent_node({"user_id": 7, "user_text": "send a cat", "image_bytes": None})
    assert out["found_image_url"] == "http://img/1.jpg"

async def test_photo_adds_vision_hint_to_first_user_message():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        await agent.agent_node({"user_id": 7, "user_text": "what is this", "image_bytes": b"x"})
    assert any("vision_analyze" in str(m.get("content", "")) for m in captured["msgs"])
    assert not any(m.get("content") == b"x" for m in captured["msgs"])
    assert all(not isinstance(m.get("content"), bytes) for m in captured["msgs"])

async def test_agent_node_never_raises():
    with patch("bot.nodes.agent.llm.chat_with_tools", new=AsyncMock(side_effect=RuntimeError("boom"))), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        out = await agent.agent_node({"user_id": 7, "user_text": "hi", "image_bytes": None})
    assert out["raw_result"] is None and "boom" in out["agent_error"]

async def test_failing_tool_does_not_kill_turn():
    calls = [
        _msg(tool_calls=[{"id": "1", "type": "function",
                          "function": {"name": "image_search", "arguments": '{"query":"cat"}'}}]),
        _msg(content="no pic today, sorry"),
    ]
    with patch("bot.nodes.agent.llm.chat_with_tools",
               new=AsyncMock(side_effect=calls)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS",
                    {"image_search": AsyncMock(side_effect=RuntimeError("brave 422"))},
                    clear=False):
        out = await agent.agent_node({"user_id": 7, "user_text": "send a cat",
                                      "image_bytes": None})
    assert out["raw_result"] == "no pic today, sorry"
    assert out["agent_error"] is None
    assert out["found_image_url"] is None


async def test_successful_turn_clears_stale_agent_error():
    """A prior turn's agent_error is checkpointed in the thread state; a later
    successful turn must clear it so compose_persona doesn't drop the image."""
    with patch("bot.nodes.agent.llm.chat_with_tools",
               new=AsyncMock(return_value=_msg(content="all good"))), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])):
        out = await agent.agent_node({"user_id": 7, "user_text": "hi", "image_bytes": None,
                                      "agent_error": "stale 422 from a past turn"})
    assert out["agent_error"] is None

async def test_recent_turns_reversed_to_chronological_with_current_message_last():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    newest_first = [{"role": "assistant", "content": "B"}, {"role": "user", "content": "A"}]
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=newest_first)):
        await agent.agent_node({"user_id": 7, "user_text": "current", "image_bytes": None})
    contents = [m["content"] for m in captured["msgs"]]
    assert contents == ["A", "B", "current"]

async def test_iteration_budget_exhausted_returns_without_raising():
    always_tool_call = _msg(tool_calls=[{"id": "1", "type": "function",
                            "function": {"name": "recall", "arguments": '{"query":"x"}'}}])
    with patch("bot.nodes.agent.llm.chat_with_tools",
               new=AsyncMock(return_value=always_tool_call)), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent.settings.agent_max_iterations", 2), \
         patch.dict("bot.nodes.agent.TOOL_FUNCS",
                    {"recall": AsyncMock(return_value="- some fact")}, clear=False):
        out = await agent.agent_node({"user_id": 7, "user_text": "loop forever?",
                                      "image_bytes": None})
    assert "raw_result" in out and out["agent_error"] is None

async def test_briefing_prepended_when_persona_configured():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent.build_agent_briefing", return_value="BRIEFING: sei Erminio"):
        await agent.agent_node({"user_id": 7, "user_text": "ciao", "image_bytes": None})
    assert captured["msgs"][0] == {"role": "system", "content": "BRIEFING: sei Erminio"}

async def test_no_briefing_for_generic_persona():
    captured = {}
    async def cap(messages, tools, model=None):
        captured["msgs"] = messages
        return _msg(content="ok")
    with patch("bot.nodes.agent.llm.chat_with_tools", new=cap), \
         patch("bot.nodes.agent.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.agent.build_agent_briefing", return_value=None):
        await agent.agent_node({"user_id": 7, "user_text": "ciao", "image_bytes": None})
    assert all(m["role"] != "system" for m in captured["msgs"])
