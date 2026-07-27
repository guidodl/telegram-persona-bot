import asyncio
import json
import logging
import random
import re
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


def _mention_names() -> list[str]:
    names = [settings.bot_name]
    names.extend(a.strip() for a in settings.bot_aliases.split(","))
    return [n.lower() for n in names if n]


def _group_allowed(chat_id: int) -> bool:
    allowed = [c.strip() for c in settings.group_allowed_chats.split(",") if c.strip()]
    return not allowed or str(chat_id) in allowed


def _user_allowed(user_id: int) -> bool:
    allowed = [u.strip() for u in settings.allowed_users.split(",") if u.strip()]
    return not allowed or str(user_id) in allowed


def _is_addressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """In a group, the bot only responds when it's addressed: an @mention of
    its username, a plain-name mention (settings.bot_name or a bot_aliases
    entry, no @), or a reply to one of the bot's own messages."""
    message = update.effective_message
    reply_to = message.reply_to_message
    if reply_to and reply_to.from_user and reply_to.from_user.id == context.bot.id:
        return True

    text = message.text or message.caption or ""
    lowered = text.lower()

    username = getattr(context.bot, "username", None)
    if username and f"@{username.lower()}" in lowered:
        return True

    for name in _mention_names():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
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
    split_at = text.rfind(" ", 0, midpoint)
    split_at = split_at if split_at > 0 else midpoint
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
    if not _user_allowed(update.effective_user.id):
        return
    await update.message.reply_text(START_MESSAGE)


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not _user_allowed(user_id):
        return
    try:
        await forget_user(user_id)
    except Exception:
        logger.exception("forget_user failed for user_id=%s", user_id)
        await update.message.reply_text(SEND_GUARD_FALLBACK)
        return
    await update.message.reply_text(FORGET_MESSAGE)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    is_private = chat.type == ChatType.PRIVATE
    if is_private:
        if not _user_allowed(update.effective_user.id):
            return
    else:
        if not _group_allowed(chat.id):
            return
        if not _is_addressed(update, context):
            return

    message = update.effective_message
    user_id = update.effective_user.id
    # Memory is keyed by the speaker (user_id); in a DM chat.id == user.id, so
    # DM behavior is unchanged. The conversation checkpoint is per-thread: one
    # thread per user in DMs, one per (group, user) so members don't share a
    # thread in a group.
    thread_id = str(user_id) if is_private else f"{chat.id}:{user_id}"
    user_text = message.text or message.caption or ""

    image_bytes = None
    if message.photo:
        image_bytes = await media.photo_to_bytes(message.photo)

    state = {
        "user_id": user_id,
        "user_text": user_text,
        "image_bytes": image_bytes,
    }

    typing_task = asyncio.create_task(_keep_typing(chat))
    try:
        result = await _graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
    except Exception:
        logger.exception("graph invocation failed for user_id=%s", user_id)
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

    asyncio.create_task(_post_send_memory_work(user_id, user_text, text))
    await asyncio.sleep(0)  # yield once so the background task gets scheduled


def _fake_text_update(text: str, chat_id: int = 1):
    """Test helper: builds a minimal (update, context) pair for a DM text message."""
    update = MagicMock()
    update.effective_chat.type = ChatType.PRIVATE
    update.effective_chat.id = chat_id
    update.effective_user.id = chat_id  # in a DM chat.id == user.id
    update.effective_message.text = text
    update.effective_message.caption = None
    update.effective_message.photo = []
    update.effective_message.reply_to_message = None
    update.effective_message.reply_text = AsyncMock()
    update.message = update.effective_message

    context = MagicMock()
    context.bot.id = 999
    context.bot.username = "personabot"
    context.bot.send_voice = AsyncMock()
    context.bot.send_photo = AsyncMock()

    return update, context


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    async def _run() -> None:
        global _graph
        async with postgres_checkpointer() as checkpointer:
            _graph = build_graph(checkpointer=checkpointer)
            application = (
                ApplicationBuilder()
                .token(settings.telegram_bot_token)
                .concurrent_updates(True)
                .build()
            )
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("forget", forget_command))
            application.add_handler(
                MessageHandler(
                    (filters.TEXT | filters.PHOTO)
                    & (filters.ChatType.PRIVATE | filters.ChatType.GROUPS)
                    & ~filters.COMMAND,
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
