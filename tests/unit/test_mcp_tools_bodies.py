from unittest.mock import AsyncMock, patch
from mcp_tools import store, tools

async def test_recall_reads_user_id_from_store():
    store.register("t1", user_id=7, image_bytes=None)
    with patch("mcp_tools.tools.memory.search_memories",
               new=AsyncMock(return_value=["learning guitar"])):
        out = await tools.recall("music", "t1")
    assert "learning guitar" in out

async def test_image_search_writes_side_channel(respx_mock):
    store.register("t2", user_id=7, image_bytes=None)
    respx_mock.get("https://api.search.brave.com/res/v1/images/search").respond(
        json={"results": [{"properties": {"url": "http://img/1.jpg"}}]})
    out = await tools.image_search("cat on a sofa", "t2")
    assert store.get("t2")["found_image_url"] == "http://img/1.jpg"
    assert "cat on a sofa" in out

async def test_vision_analyze_reads_image_bytes_from_store():
    store.register("t3", user_id=7, image_bytes=b"jpegbytes")
    with patch("mcp_tools.tools.llm.chat_vision",
               new=AsyncMock(return_value="a ginger cat")):
        out = await tools.vision_analyze("t3")
    assert out == "a ginger cat"

async def test_vision_analyze_no_image_returns_message():
    store.register("t4", user_id=7, image_bytes=None)
    out = await tools.vision_analyze("t4")
    assert out == "no image was attached"
