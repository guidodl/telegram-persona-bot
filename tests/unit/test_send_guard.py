import pytest
from bot.ingress import send_guard

@pytest.mark.parametrize("text,ok", [
    ("Hey, how's it going?", True),
    ("I think x = {a, b} works — try it!", True),   # braces in normal prose: NOT rejected
    ("", False),
    ('{"tool":"web_lookup","result":"..."}', False), # raw JSON payload: rejected
    ("TOOL_CALL: web_lookup(query=...)", False),
])
def test_send_guard(text, ok):
    assert send_guard(text) is ok
