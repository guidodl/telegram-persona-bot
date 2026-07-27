def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    from bot.config import Settings
    s = Settings()
    assert s.model_chat == "deepseek/deepseek-v4-flash"
    assert s.embed_dim == 1536
    assert s.embed_backend == "openrouter"
