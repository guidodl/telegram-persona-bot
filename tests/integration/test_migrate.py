import pytest
from testcontainers.postgres import PostgresContainer
from bot.migrate import apply_migrations
import psycopg

@pytest.fixture(scope="module")
def pg_url():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "")

async def test_migrations_create_tables_and_are_idempotent(pg_url):
    await apply_migrations(pg_url)
    await apply_migrations(pg_url)  # second run must not error
    with psycopg.connect(pg_url.replace("+psycopg", "")) as c:
        rows = c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert {"users", "memories", "turns", "schema_migrations"} <= names
