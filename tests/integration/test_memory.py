import pytest
from unittest.mock import AsyncMock, patch
from testcontainers.postgres import PostgresContainer
from bot.migrate import apply_migrations

@pytest.fixture(scope="module")
def pg_url():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "+psycopg")

@pytest.fixture(autouse=True)
def _wire(pg_url, monkeypatch):
    monkeypatch.setattr("bot.config.settings.database_url", pg_url)

async def test_upsert_then_semantic_recall_and_forget(pg_url):
    await apply_migrations(pg_url)
    from bot import memory
    # deterministic fake embeddings: "guitar" near query, "weather" far.
    # padded to 1536 dims (zeros don't change cosine direction) to satisfy the
    # fixed-dimension vector(1536) column from migrations/0001_init.sql.
    def _pad(v):
        return v + [0.0] * (1536 - len(v))

    def fake_embed(texts):
        table = {"learning guitar": [1.0, 0.0], "hates cold weather": [0.0, 1.0],
                 "guitar": [1.0, 0.0]}
        return AsyncMock(return_value=[_pad(table.get(t, [0.5, 0.5])) for t in texts])()
    with patch("bot.memory.llm.embed", side_effect=fake_embed):
        await memory.get_profile(42)                       # lazy create
        await memory.upsert_memories(42, ["learning guitar", "hates cold weather"])
        top = await memory.search_memories(42, "guitar", k=1)
        assert top == ["learning guitar"]
        await memory.forget_user(42)
        assert await memory.search_memories(42, "guitar", k=5) == []
