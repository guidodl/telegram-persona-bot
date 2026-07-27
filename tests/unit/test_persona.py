import pytest

from bot.config import settings
from bot.persona import (
    PERSONA_PREAMBLE,
    _read_persona_file,
    build_agent_briefing,
    build_system_prompt,
)


@pytest.fixture(autouse=True)
def _clear_persona_cache():
    _read_persona_file.cache_clear()
    yield
    _read_persona_file.cache_clear()


def test_prompt_includes_facts_and_silence_rules():
    p = build_system_prompt({"name": "Sam", "summary": "likes short replies", "tone": "warm"},
                            ["learning guitar"])
    assert "Sam" in p and "guitar" in p
    for rule in ["never mention", "tool", "As an AI"]:
        assert rule.lower() in p.lower()


def test_persona_file_replaces_preamble(tmp_path, monkeypatch):
    f = tmp_path / "erminio.md"
    f.write_text("Ti chiami Erminio. Sei un attore.", encoding="utf-8")
    monkeypatch.setattr(settings, "persona_file", str(f))
    p = build_system_prompt({"name": "Guido"}, [])
    assert "Erminio" in p
    assert "warm, present companion" not in p
    # hard rules survive any persona file
    assert "never mention" in p.lower()
    # profile facts still flow in
    assert "Guido" in p


def test_missing_persona_file_falls_back_to_generic(monkeypatch):
    monkeypatch.setattr(settings, "persona_file", "/nonexistent/persona.md")
    p = build_system_prompt({}, [])
    assert PERSONA_PREAMBLE in p


def test_agent_briefing_none_without_persona_file(monkeypatch):
    monkeypatch.setattr(settings, "persona_file", "")
    assert build_agent_briefing() is None


def test_agent_briefing_wraps_persona_file(tmp_path, monkeypatch):
    f = tmp_path / "erminio.md"
    f.write_text("Ti chiami Erminio.", encoding="utf-8")
    monkeypatch.setattr(settings, "persona_file", str(f))
    briefing = build_agent_briefing()
    assert "Erminio" in briefing
    assert "image_search" in briefing  # agent gets tool guidance, not just identity
