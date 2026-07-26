import pathlib, psycopg

MIGRATIONS = pathlib.Path(__file__).parent.parent / "migrations"

async def apply_migrations(database_url: str) -> None:
    url = database_url.replace("+psycopg", "").replace("+asyncpg", "")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                     "(filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())")
        applied = {r[0] for r in conn.execute("SELECT filename FROM schema_migrations")}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in applied:
                continue
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations(filename) VALUES (%s)", (path.name,))
