import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from telegram import InputFile
from telegram.constants import ChatType
from langgraph.checkpoint.memory import InMemorySaver
from bot import ingress
from bot.graph import build_graph

async def test_persist_memory_launched_after_send():
    with patch("bot.ingress.persist_memory", new=AsyncMock()) as pm, \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)  # test helper
        await ingress.on_message(update, ctx)
        # allow the created task to schedule
    pm.assert_awaited()  # persist ran, off the reply path


async def test_recent_turns_fetched_and_passed_to_persist_memory():
    with patch("bot.ingress.persist_memory", new=AsyncMock()) as pm, \
         patch("bot.ingress.memory.recent_turns",
               new=AsyncMock(return_value=[{"role": "user", "content": "earlier"}])) as rt, \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    rt.assert_awaited_once_with(5, 10)
    pm.assert_awaited_once_with(5, "hello", "hi", [{"role": "user", "content": "earlier"}])


async def test_log_turn_called_for_user_and_assistant_after_reply():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()) as lt, \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi there", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    lt.assert_any_await(5, "user", "hello")
    lt.assert_any_await(5, "assistant", "hi there")
    assert lt.await_count == 2


async def test_post_send_memory_work_happens_after_reply_and_never_raises():
    call_order = []

    async def recording_reply_text(*args, **kwargs):
        call_order.append("reply_text")

    async def recording_recent_turns(*args, **kwargs):
        call_order.append("recent_turns")
        raise RuntimeError("db down")

    async def recording_log_turn(*args, **kwargs):
        call_order.append("log_turn")
        raise RuntimeError("db down")

    with patch("bot.ingress.persist_memory", new=AsyncMock(side_effect=RuntimeError("db down"))), \
         patch("bot.ingress.memory.recent_turns", new=recording_recent_turns), \
         patch("bot.ingress.memory.log_turn", new=recording_log_turn), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        update.effective_message.reply_text = AsyncMock(side_effect=recording_reply_text)
        await ingress.on_message(update, ctx)  # must not raise despite the memory-side failures
    assert call_order[0] == "reply_text"
    assert "recent_turns" in call_order
    assert call_order.index("reply_text") < call_order.index("recent_turns")


async def test_group_chat_unaddressed_ignored():
    with patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_group_chat_at_mention_answered():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hey @personabot how are you", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_user.id = 555
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()
    assert g.ainvoke.call_args.args[0]["user_id"] == 555
    assert g.ainvoke.call_args.kwargs["config"]["configurable"]["thread_id"] == "-100:555"


async def test_group_chat_name_mention_without_at_answered():
    with patch("bot.ingress.settings.bot_name", "Aria"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("Aria, what do you think?", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_group_chat_reply_to_bot_answered():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("and what about tomorrow?", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.reply_to_message = MagicMock()
        update.effective_message.reply_to_message.from_user.id = ctx.bot.id
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_group_chat_reply_to_other_user_ignored():
    with patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("no worries", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.reply_to_message = MagicMock()
        update.effective_message.reply_to_message.from_user.id = 12345  # some other human
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_dm_from_disallowed_user_ignored():
    with patch("bot.ingress.settings.allowed_users", "163829883"), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hello", chat_id=999)  # user 999 != allowed
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_dm_from_allowed_user_answered():
    with patch("bot.ingress.settings.allowed_users", "163829883, 5"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_group_chat_alias_mention_answered():
    with patch("bot.ingress.settings.bot_name", "Erminio"), \
         patch("bot.ingress.settings.bot_aliases", "ermi, erm"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("ermi ci sei?", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_group_not_in_allowlist_ignored_even_when_addressed():
    with patch("bot.ingress.settings.group_allowed_chats", "-1003765317870"), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_group_in_allowlist_answered():
    with patch("bot.ingress.settings.group_allowed_chats", "-1003765317870, -100"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_group_wrong_topic_ignored_even_when_addressed():
    with patch("bot.ingress.settings.group_allowed_topics", "-100:42"), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.message_thread_id = 7
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_group_general_topic_ignored_when_topics_restricted():
    with patch("bot.ingress.settings.group_allowed_topics", "-100:42"), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.message_thread_id = None
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


async def test_group_allowed_topic_answered():
    with patch("bot.ingress.settings.group_allowed_topics", "-100:42"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.message_thread_id = 42
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_other_chat_unaffected_by_another_chats_topic_rule():
    with patch("bot.ingress.settings.group_allowed_topics", "-999:42"), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.message_thread_id = 7
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_awaited_once()


async def test_photo_sent_into_the_originating_topic():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress.media.fetch_image", new=AsyncMock(return_value=b"img")), \
         patch("bot.ingress.media.to_photo", return_value=b"img"), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False,
                                                      "image_url": "http://x/i.png"}})
        update, ctx = ingress._fake_text_update("hey @personabot", chat_id=-100)
        update.effective_chat.type = ChatType.GROUP
        update.effective_message.message_thread_id = 42
        await ingress.on_message(update, ctx)
    assert ctx.bot.send_photo.await_args.kwargs["message_thread_id"] == 42


async def test_raw_tool_leak_replaced_with_fallback():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "TOOL_CALL: web_lookup(x)",
                                                       "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    sent = [c.args[0] for c in update.effective_message.reply_text.call_args_list]
    assert all("TOOL_CALL" not in s for s in sent)
    assert ingress.SEND_GUARD_FALLBACK in sent


async def test_graph_error_sends_fallback_not_traceback():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    update.effective_message.reply_text.assert_awaited_once_with(ingress.SEND_GUARD_FALLBACK)


async def test_voice_reply_sends_audio_only_no_text():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g, \
         patch("bot.ingress.tts.synth", new=AsyncMock(return_value=b"audio")), \
         patch("bot.ingress.media.to_audio", return_value="AUDIO_FILE"):
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi there",
                                                       "voice": True, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    ctx.bot.send_audio.assert_awaited_once_with(chat_id=5, audio="AUDIO_FILE")
    update.effective_message.reply_text.assert_not_awaited()


async def test_voice_reply_falls_back_to_text_when_synth_fails():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g, \
         patch("bot.ingress.tts.synth", new=AsyncMock(side_effect=RuntimeError("tts down"))):
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi there",
                                                       "voice": True, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    ctx.bot.send_audio.assert_not_awaited()
    update.effective_message.reply_text.assert_awaited_once_with("hi there")


async def test_image_url_reply_downloads_and_uploads_photo():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.media.fetch_image", new=AsyncMock(return_value=b"imgbytes")) as fi, \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "here you go", "voice": False,
                                                       "image_url": "http://img/1.jpg"}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    fi.assert_awaited_once_with("http://img/1.jpg")
    ctx.bot.send_photo.assert_awaited_once()
    kwargs = ctx.bot.send_photo.call_args.kwargs
    assert kwargs["chat_id"] == 5
    assert isinstance(kwargs["photo"], InputFile)


async def test_photo_message_extracts_image_bytes():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g, \
         patch("bot.ingress.media.photo_to_bytes", new=AsyncMock(return_value=b"imgbytes")):
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "nice pic!", "voice": False,
                                                       "image_url": None}})
        update, ctx = ingress._fake_text_update("look", chat_id=5)
        update.effective_message.photo = ["some_photo_size"]
        await ingress.on_message(update, ctx)
    assert g.ainvoke.call_args.args[0]["image_bytes"] == b"imgbytes"


async def test_forget_command_wipes_and_confirms():
    with patch("bot.ingress.forget_user", new=AsyncMock()) as fu:
        update, ctx = ingress._fake_text_update("/forget", chat_id=7)
        await ingress.forget_command(update, ctx)
    fu.assert_awaited_once_with(7)
    update.effective_message.reply_text.assert_awaited_once_with(ingress.FORGET_MESSAGE)


async def test_forget_command_error_swallowed():
    with patch("bot.ingress.forget_user", new=AsyncMock(side_effect=RuntimeError("db down"))):
        update, ctx = ingress._fake_text_update("/forget", chat_id=7)
        await ingress.forget_command(update, ctx)  # must not raise
    update.effective_message.reply_text.assert_awaited_once_with(ingress.SEND_GUARD_FALLBACK)


async def test_start_command_sends_intro():
    update, ctx = ingress._fake_text_update("/start", chat_id=9)
    await ingress.start_command(update, ctx)
    update.effective_message.reply_text.assert_awaited_once_with(ingress.START_MESSAGE)


# --- Finding 1: real-checkpointer regression (missing thread_id) ----------
#
# Bug: on_message called _graph.ainvoke(state) with no config. Any graph
# built with a checkpointer (as main() does against postgres) raises
# ValueError from LangGraph because no configurable.thread_id was supplied.
# on_message's broad except swallowed that and always sent the fallback —
# every real DM degraded to SEND_GUARD_FALLBACK. These tests build a graph
# with a real (in-memory) checkpointer, matching production shape, so the
# missing-config bug is actually exercised rather than masked by a mocked
# _graph.

async def test_graph_with_checkpointer_requires_thread_id():
    graph = build_graph(checkpointer=InMemorySaver())
    with pytest.raises(ValueError):
        await graph.ainvoke({"user_id": 1, "user_text": "hi", "image_bytes": None})


async def test_graph_with_checkpointer_and_thread_id_succeeds():
    with patch("bot.graph.load_memory",
               new=AsyncMock(return_value={"profile": {}, "memories": []})), \
         patch("bot.graph.agent_node",
               new=AsyncMock(return_value={"raw_result": "hi", "found_image_url": None})), \
         patch("bot.graph.compose_persona",
               new=AsyncMock(return_value={"reply": {"text": "hello", "voice": False,
                                                      "image_url": None}})):
        graph = build_graph(checkpointer=InMemorySaver())
        out = await graph.ainvoke({"user_id": 1, "user_text": "hi", "image_bytes": None},
                                   config={"configurable": {"thread_id": "1"}})
    assert out["reply"]["text"] == "hello"


async def test_on_message_with_real_checkpointer_sends_reply_not_fallback():
    """Drives ingress.on_message end-to-end with _graph set to a
    real-checkpointer graph (nodes stubbed, as other tests do via
    bot.graph patches). Against the pre-fix ainvoke(state) with no config,
    this raises inside on_message, gets caught, and sends
    SEND_GUARD_FALLBACK instead of the real reply — proving the bug was
    invisible to the mocked-_graph tests above. With the fix
    (ainvoke(state, config={"configurable": {"thread_id": ...}})) the real
    reply goes out."""
    old_graph = ingress._graph
    try:
        with patch("bot.graph.load_memory",
                   new=AsyncMock(return_value={"profile": {}, "memories": []})), \
             patch("bot.graph.agent_node",
                   new=AsyncMock(return_value={"raw_result": "hi", "found_image_url": None})), \
             patch("bot.graph.compose_persona",
                   new=AsyncMock(return_value={"reply": {"text": "hello there", "voice": False,
                                                          "image_url": None}})), \
             patch("bot.ingress.persist_memory", new=AsyncMock()), \
             patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
             patch("bot.ingress.memory.log_turn", new=AsyncMock()):
            ingress._graph = build_graph(checkpointer=InMemorySaver())
            update, ctx = ingress._fake_text_update("hello", chat_id=42)
            await ingress.on_message(update, ctx)
    finally:
        ingress._graph = old_graph
    sent = [c.args[0] for c in update.effective_message.reply_text.call_args_list]
    assert sent == ["hello there"]
    assert ingress.SEND_GUARD_FALLBACK not in sent


# --- Finding 2: concurrent_updates / per-turn isolation --------------------

async def test_concurrent_messages_from_different_users_do_not_cross_talk():
    """Fires two on_message calls concurrently (as concurrent_updates(True)
    now allows) for different chat_ids/photos and asserts each reply
    reflects only its own turn's user_id and image_bytes — no bleed through
    the mcp-tools per-turn registration or shared graph."""
    registered = {}

    async def fake_register_turn(turn_id, *, user_id, image_bytes):
        registered[turn_id] = (user_id, image_bytes)

    async def fake_call_hermes(messages, turn_id, model=None):
        await asyncio.sleep(0.01)  # force interleaving between the two turns
        uid, img = registered[turn_id]
        return f"uid={uid}|img={img}"

    async def fake_chat(messages):
        for m in messages:
            content = m.get("content")
            if isinstance(content, str) and content.startswith("Facts to speak from:"):
                return content.split("Facts to speak from:\n", 1)[1].split("\n\n")[0]
        return "no facts"

    async def fake_photo_to_bytes(photo_sizes):
        return photo_sizes[0].encode()

    with patch("bot.graph.load_memory",
               new=AsyncMock(return_value={"profile": {}, "memories": []})), \
         patch("bot.nodes.agent._register_turn", new=fake_register_turn), \
         patch("bot.nodes.agent.call_hermes", new=fake_call_hermes), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.nodes.compose_persona.llm.chat", new=fake_chat), \
         patch("bot.ingress.media.photo_to_bytes", new=fake_photo_to_bytes), \
         patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph", build_graph()):
        update_a, ctx_a = ingress._fake_text_update("hi from A", chat_id=101)
        update_a.effective_message.photo = ["imgA"]
        update_b, ctx_b = ingress._fake_text_update("hi from B", chat_id=202)
        update_b.effective_message.photo = ["imgB"]

        await asyncio.gather(
            ingress.on_message(update_a, ctx_a),
            ingress.on_message(update_b, ctx_b),
        )

    sent_a = update_a.effective_message.reply_text.call_args_list[0].args[0]
    sent_b = update_b.effective_message.reply_text.call_args_list[0].args[0]
    assert "uid=101" in sent_a and "img=b'imgA'" in sent_a
    assert "uid=202" in sent_b and "img=b'imgB'" in sent_b
    assert "202" not in sent_a and "imgB" not in sent_a
    assert "101" not in sent_b and "imgA" not in sent_b


# --- Finding 3: _chunk_reply split-point edge case -------------------------

def test_chunk_reply_spaceless_long_string_splits_sanely():
    """No space before the midpoint used to make rfind() return -1, and
    `-1 or midpoint` evaluated truthy -1 (since -1 is falsy... actually
    -1 is truthy in Python), producing text[:-1] / text[-1:] — an N-1 char
    chunk plus a single stray trailing character. The fix treats a
    not-found rfind (-1) the same as one at position 0: fall back to the
    midpoint."""
    text = "a" * 4000  # exceeds the 3500-char limit, no spaces anywhere
    chunks = ingress._chunk_reply(text)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 2000
    assert chunks[1] == "a" * 2000
    assert chunks[0] + chunks[1] == text
    assert len(chunks[1]) > 1  # not a stray single trailing character


# --- Quoted-message context ------------------------------------------------
#
# Bug: replying to (or quoting) another message and asking "che ne pensi?"
# sent only the bare question to the graph. The quoted text — often the whole
# subject, e.g. a shared link — never reached the agent, so the bot answered
# "you didn't send me anything".

async def test_reply_to_another_message_includes_quoted_text_in_user_text():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("che ne pensi?", chat_id=5)
        quoted = MagicMock()
        quoted.from_user.id = 4242
        quoted.from_user.first_name = "Guido"
        quoted.text = "https://docs.litellm.ai/docs/proxy/auto_routing"
        quoted.caption = None
        update.effective_message.reply_to_message = quoted
        await ingress.on_message(update, ctx)
    sent_text = g.ainvoke.call_args.args[0]["user_text"]
    assert "https://docs.litellm.ai/docs/proxy/auto_routing" in sent_text
    assert "che ne pensi?" in sent_text


async def test_reply_to_bot_own_message_does_not_duplicate_quoted_text():
    """The bot's own prior turn is already in the conversation history the
    agent node loads, so re-injecting it would duplicate context."""
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("and tomorrow?", chat_id=5)
        quoted = MagicMock()
        quoted.from_user.id = ctx.bot.id
        quoted.text = "today is sunny"
        quoted.caption = None
        update.effective_message.reply_to_message = quoted
        await ingress.on_message(update, ctx)
    assert g.ainvoke.call_args.args[0]["user_text"] == "and tomorrow?"


async def test_message_without_reply_keeps_user_text_unchanged():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("plain question", chat_id=5)
        await ingress.on_message(update, ctx)
    assert g.ainvoke.call_args.args[0]["user_text"] == "plain question"


async def test_reply_to_empty_message_adds_no_quote_block():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress.memory.recent_turns", new=AsyncMock(return_value=[])), \
         patch("bot.ingress.memory.log_turn", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("che ne pensi?", chat_id=5)
        quoted = MagicMock()
        quoted.from_user.id = 4242
        quoted.text = None
        quoted.caption = None
        update.effective_message.reply_to_message = quoted
        await ingress.on_message(update, ctx)
    assert g.ainvoke.call_args.args[0]["user_text"] == "che ne pensi?"
