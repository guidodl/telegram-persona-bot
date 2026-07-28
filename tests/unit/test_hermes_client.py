import respx, httpx
from bot import hermes_client

SSE = (
    b'event: hermes.tool.progress\ndata: {"tool":"web_search","status":"running"}\n\n'
    b'data: {"choices":[{"delta":{"content":"Sofia"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":", 24."}}]}\n\n'
    b'data: [DONE]\n\n'
)

@respx.mock
async def test_call_hermes_concatenates_content_ignores_progress():
    respx.post("http://hermes:8642/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=SSE,
                                    headers={"content-type": "text/event-stream"}))
    out = await hermes_client.call_hermes(
        [{"role": "user", "content": "chi è la tua fiamma?"}], turn_id="t1", model=None)
    assert out == "Sofia, 24."

@respx.mock
async def test_call_hermes_threads_turn_id_as_system_message():
    captured = {}
    def _capture(request):
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b'data: [DONE]\n\n',
                              headers={"content-type": "text/event-stream"})
    respx.post("http://hermes:8642/v1/chat/completions").mock(side_effect=_capture)
    await hermes_client.call_hermes([{"role": "user", "content": "hi"}], turn_id="t9", model=None)
    msgs = captured["body"]["messages"]
    assert any(m["role"] == "system" and "turn_id=t9" in m["content"] for m in msgs)
