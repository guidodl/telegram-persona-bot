from bot.persona import build_system_prompt

def test_prompt_includes_facts_and_silence_rules():
    p = build_system_prompt({"name": "Sam", "summary": "likes short replies", "tone": "warm"},
                            ["learning guitar"])
    assert "Sam" in p and "guitar" in p
    for rule in ["never mention", "tool", "As an AI"]:
        assert rule.lower() in p.lower()
