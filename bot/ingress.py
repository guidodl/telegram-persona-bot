import asyncio
import json
import logging
import random
from unittest.mock import AsyncMock, MagicMock

from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bot import media, memory, tts
from bot.config import settings
from bot.graph import build_graph, postgres_checkpointer
from bot.memory import forget_user
from bot.nodes.persist_memory import persist_memory

logger = logging.getLogger(__name__)

# Built by main() against the postgres checkpointer at startup; tests patch
# this module global directly (bot.ingress._graph) rather than invoking a
# real graph, so importing this module never touches a database.
_graph = None

START_MESSAGE = "Hey, it's really good to hear from you. What's going on?"
FORGET_MESSAGE = "Done — I've let all of that go. We're starting fresh."
SEND_GUARD_FALLBACK = "Sorry, I got a little tongue-tied there — could you say that again?"

_TYPING_REFIRE_S = 4


def _looks_like_raw_tool_output(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped[0] in "[{":
        try:
            json.loads(stripped)
            return True
        except (ValueError, TypeError):
            pass
    if "TOOL_CALL:" in text or "TOOL:" in text:
        return True
    return False


def send_guard(text: str) -> bool:
    """True if text is safe to send to the user as-is."""
    if not text or not text.strip():
        return False
    if _looks_like_raw_tool_output(text):
        return False
    return True


def _chunk_reply(text: str) -> list[str]:
    limit = 3500
    if len(text) <= limit:
        return [text]
    midpoint = len(text) // 2
    split_at = text.rfind(" ", 0, midpoint) or midpoint
    return [text[:split_at].strip(), text[split_at:].strip()]


async def _pacing_delay() -> None:
    if not settings.pacing_enabled:
        return
    delay = random.uniform(settings.pacing_delay_min_s, settings.pacing_delay_max_s)
    await asyncio.sleep(delay)


async def _keep_typing(chat) -> None:
    try:
        while True:
            await chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(_TYPING_REFIRE_S)
    except asyncio.CancelledError:
        pass


async def _post_send_memory_work(chat_id: int, user_text: str, reply_text: str) -> None:
    """Fire-and-forget, off the reply critical path — mirrors persist_memory's
    own swallow-everything contract. Fetches history and launches extraction
    BEFORE logging the current turn, so this turn never pollutes the context
    used to extract facts from itself."""
    recent = []
    try:
        recent = await memory.recent_turns(chat_id, 10)
    except Exception:
        logger.exception("recent_turns fetch failed for chat_id=%s", chat_id)

    try:
        await persist_memory(chat_id, user_text, reply_text, recent)
    except Exception:
        logger.exception("persist_memory failed for chat_id=%s", chat_id)

    try:
        await memory.log_turn(chat_id, "user", user_text)
        await memory.log_turn(chat_id, "assistant", reply_text)
    except Exception:
        logger.exception("log_turn failed for chat_id=%s", chat_id)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_MESSAGE)


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    try:
        await forget_user(chat_id)
    except Exception:
        logger.exception("forget_user failed for chat_id=%s", chat_id)
        await update.message.reply_text(SEND_GUARD_FALLBACK)
        return
    await update.message.reply_text(FORGET_MESSAGE)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.type != ChatType.PRIVATE:
        return

    message = update.effective_message
    chat = update.effective_chat
    user_text = message.text or message.caption or ""

    image_bytes = None
    if message.photo:
        image_bytes = await media.photo_to_bytes(message.photo)

    state = {
        "chat_id": chat.id,
        "user_text": user_text,
        "image_bytes": image_bytes,
    }

    typing_task = asyncio.create_task(_keep_typing(chat))
    try:
        result = await _graph.ainvoke(state)
    except Exception:
        logger.exception("graph invocation failed for chat_id=%s", chat.id)
        await message.reply_text(SEND_GUARD_FALLBACK)
        return
    finally:
        typing_task.cancel()

    reply = result.get("reply") or {}
    text = reply.get("text") or ""

    if not send_guard(text):
        logger.warning("send_guard rejected reply text for chat_id=%s", chat.id)
        text = SEND_GUARD_FALLBACK

    chunks = _chunk_reply(text)
    for i, chunk in enumerate(chunks):
        if i > 0:
            await _pacing_delay()
        await message.reply_text(chunk)

    if reply.get("voice"):
        try:
            voice_bytes = await tts.synth(text)
            await context.bot.send_voice(chat_id=chat.id, voice=media.to_voice(voice_bytes))
        except Exception:
            logger.exception("voice send failed for chat_id=%s", chat.id)

    image_url = reply.get("image_url")
    if image_url:
        try:
            await context.bot.send_photo(chat_id=chat.id, photo=image_url)
        except Exception:
            logger.exception("photo send failed for chat_id=%s", chat.id)

    asyncio.create_task(_post_send_memory_work(chat.id, user_text, text))
    await asyncio.sleep(0)  # yield once so the background task gets scheduled


def _fake_text_update(text: str, chat_id: int = 1):
    """Test helper: builds a minimal (update, context) pair for a DM text message."""
    update = MagicMock()
    update.effective_chat.type = ChatType.PRIVATE
    update.effective_chat.id = chat_id
    update.effective_message.text = text
    update.effective_message.caption = None
    update.effective_message.photo = []
    update.effective_message.reply_text = AsyncMock()
    update.message = update.effective_message

    context = MagicMock()
    context.bot.send_voice = AsyncMock()
    context.bot.send_photo = AsyncMock()

    return update, context


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    async def _run() -> None:
        global _graph
        async with postgres_checkpointer() as checkpointer:
            _graph = build_graph(checkpointer=checkpointer)
            application = ApplicationBuilder().token(settings.telegram_bot_token).build()
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("forget", forget_command))
            application.add_handler(
                MessageHandler(
                    (filters.TEXT | filters.PHOTO) & filters.ChatType.PRIVATE & ~filters.COMMAND,
                    on_message,
                )
            )
            async with application:
                await application.start()
                await application.updater.start_polling()
                try:
                    await asyncio.Event().wait()
                finally:
                    await application.updater.stop()
                    await application.stop()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
