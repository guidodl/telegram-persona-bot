from unittest.mock import AsyncMock, patch

from bot.graph import build_graph


async def test_only_compose_persona_output_reaches_state():
    with patch("bot.graph.load_memory",
               new=AsyncMock(return_value={"profile": {}, "memories": []})), \
         patch("bot.graph.agent_node",
               new=AsyncMock(return_value={"raw_result": "TOOL:web_lookup rawJSON{...}",
                                           "found_image_url": None})), \
         patch("bot.graph.compose_persona",
               new=AsyncMock(return_value={"reply": {"text": "Nice and sunny!",
                                                     "voice": False, "image_url": None}})):
        graph = build_graph()
        out = await graph.ainvoke({"user_id": 1, "user_text": "weather?", "image_bytes": None})
    assert out["reply"]["text"] == "Nice and sunny!"
    assert "TOOL:" not in out["reply"]["text"]
    assert "rawJSON" not in out["reply"]["text"]
