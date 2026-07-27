import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from bot import llm
from bot.config import settings

# Fixed for process lifetime by design in production; settings.database_url
# never changes after startup. _reset_engine_for_tests() exists only so
# integration tests can rebind to a different container/URL.
_engine: AsyncEngine | None = None
_engine_lock = asyncio.Lock()


def _async_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://") or database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


async def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                _engine = create_async_engine(_async_url(settings.database_url))
    return _engine


async def _reset_engine_for_tests() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def _embed(texts: list[str]) -> list[list[float]]:
    return await llm.embed(texts)


async def get_profile(user_id: int) -> dict:
    engine = await _get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO users (telegram_user_id) VALUES (:uid) "
                "ON CONFLICT (telegram_user_id) DO NOTHING"
            ),
            {"uid": user_id},
        )
        row = (await conn.execute(
            text("SELECT * FROM users WHERE telegram_user_id = :uid"), {"uid": user_id}
        )).mappings().first()
        return dict(row)


async def search_memories(user_id: int, query: str, k: int = 5) -> list[str]:
    [vec] = await _embed([query])
    engine = await _get_engine()
    async with engine.begin() as conn:
        rows = (await conn.execute(
            text(
                "SELECT text FROM memories WHERE telegram_user_id = :uid "
                "ORDER BY embedding <=> CAST(:q AS vector) ASC LIMIT :k"
            ),
            {"uid": user_id, "q": str(vec), "k": k},
        )).all()
    return [r[0] for r in rows]


async def upsert_memories(user_id: int, facts: list[str]) -> None:
    if not facts:
        return
    vectors = await _embed(facts)
    engine = await _get_engine()
    async with engine.begin() as conn:
        for fact, vec in zip(facts, vectors):
            await conn.execute(
                text(
                    "INSERT INTO memories (telegram_user_id, text, embedding) "
                    "VALUES (:uid, :text, CAST(:embedding AS vector))"
                ),
                {"uid": user_id, "text": fact, "embedding": str(vec)},
            )


async def update_profile_summary(user_id: int, summary: str) -> None:
    engine = await _get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE users SET summary = :summary, updated_at = now() "
                "WHERE telegram_user_id = :uid"
            ),
            {"uid": user_id, "summary": summary},
        )


async def log_turn(user_id: int, role: str, content: str) -> None:
    engine = await _get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO turns (telegram_user_id, role, content) "
                "VALUES (:uid, :role, :content)"
            ),
            {"uid": user_id, "role": role, "content": content},
        )


async def recent_turns(user_id: int, n: int = 10) -> list[dict]:
    """Returns the n most recent turns, newest-first."""
    engine = await _get_engine()
    async with engine.begin() as conn:
        rows = (await conn.execute(
            text(
                "SELECT role, content FROM turns WHERE telegram_user_id = :uid "
                "ORDER BY created_at DESC LIMIT :n"
            ),
            {"uid": user_id, "n": n},
        )).mappings().all()
    return [dict(r) for r in rows]


async def forget_user(user_id: int) -> None:
    engine = await _get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM memories WHERE telegram_user_id = :uid"), {"uid": user_id})
        await conn.execute(text("DELETE FROM turns WHERE telegram_user_id = :uid"), {"uid": user_id})
        await conn.execute(
            text(
                "UPDATE users SET name = NULL, summary = '', preferences = '{}', "
                "tone = NULL, updated_at = now() WHERE telegram_user_id = :uid"
            ),
            {"uid": user_id},
        )
