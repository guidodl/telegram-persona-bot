from unittest.mock import AsyncMock, patch
from telegram.constants import ChatType
from bot import ingress

async def test_persist_memory_launched_after_send():
    with patch("bot.ingress.persist_memory", new=AsyncMock()) as pm, \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi", "voice": False, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)  # test helper
        await ingress.on_message(update, ctx)
        # allow the created task to schedule
    pm.assert_awaited()  # persist ran, off the reply path


async def test_group_chat_ignored():
    with patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock()
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        update.effective_chat.type = ChatType.GROUP
        await ingress.on_message(update, ctx)
    g.ainvoke.assert_not_awaited()


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


async def test_voice_reply_sends_voice():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g, \
         patch("bot.ingress.tts.synth", new=AsyncMock(return_value=b"audio")), \
         patch("bot.ingress.media.to_voice", return_value="VOICE_FILE"):
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "hi there",
                                                       "voice": True, "image_url": None}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    ctx.bot.send_voice.assert_awaited_once_with(chat_id=5, voice="VOICE_FILE")


async def test_image_url_reply_sends_photo():
    with patch("bot.ingress.persist_memory", new=AsyncMock()), \
         patch("bot.ingress._graph") as g:
        g.ainvoke = AsyncMock(return_value={"reply": {"text": "here you go", "voice": False,
                                                       "image_url": "http://img/1.jpg"}})
        update, ctx = ingress._fake_text_update("hello", chat_id=5)
        await ingress.on_message(update, ctx)
    ctx.bot.send_photo.assert_awaited_once_with(chat_id=5, photo="http://img/1.jpg")


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
