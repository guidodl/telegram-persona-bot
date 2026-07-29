from unittest.mock import AsyncMock, patch

import pytest
from testcontainers.postgres import PostgresContainer

from bot.graph import build_graph
from bot.migrate import apply_migrations
from bot.nodes.persist_memory import persist_memory


@pytest.fixture(scope="module")
def pg_url():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "+psycopg")


@pytest.fixture(autouse=True)
def _wire(pg_url, monkeypatch):
    monkeypatch.setattr("bot.config.settings.database_url", pg_url)


def _pad(v):
    return v + [0.0] * (1536 - len(v))


async def _fake_embed(texts):
    return [_pad([0.1, 0.2]) for _ in texts]


async def test_full_graph_only_compose_persona_output_reaches_sink(pg_url):
    """Real Postgres-backed load_memory/agent/compose_persona wiring: Hermes
    returns a tool-flavored string, but only compose_persona's persona-styled
    text is allowed out of the graph — no tool string or raw_result leaks into
    the reply."""
    await apply_migrations(pg_url)
    from bot import memory
    await memory._reset_engine_for_tests()

    leaky_raw_result = "RAW_TOOL_JSON:{'temp_c': 24, 'condition': 'clear'}"
    persona_reply = "It's sunny and warm out there today!"

    with patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(return_value=leaky_raw_result)), \
         patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value=persona_reply)), \
         patch("bot.memory.llm.embed", new=AsyncMock(side_effect=_fake_embed)):
        graph = build_graph()
        out = await graph.ainvoke({"user_id": 501, "user_text": "what's the weather like?",
                                   "image_bytes": None})

    assert out["reply"]["text"] == persona_reply
    assert "RAW_TOOL_JSON" not in out["reply"]["text"]
    assert "temp_c" not in out["reply"]["text"]
    # the leaky content did reach agent state internally...
    assert out["raw_result"] == leaky_raw_result
    # ...but never the outward-facing reply.
    assert out["raw_result"] not in out["reply"]["text"]


async def test_cross_session_memory_fact_persists_and_is_recalled(pg_url):
    """Turn 1 states a durable fact; persist_memory + log_turn (mirroring
    ingress._post_send_memory_work) write it to the real DB off the reply
    path. Turn 2's load_memory/search_memories must then surface that fact
    from a fresh graph invocation — proving the Task 13 memory-continuity
    chain actually works end-to-end against Postgres, not just in unit tests
    with mocked memory calls."""
    await apply_migrations(pg_url)
    from bot import memory
    await memory._reset_engine_for_tests()

    chat_id = 777
    fact = "user's favorite color is teal"

    # --- turn 1: user states the fact, bot replies, memory work runs post-send ---
    with patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(return_value="Got it, I'll remember that!")), \
         patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="Got it, I'll remember that!")), \
         patch("bot.memory.llm.embed", new=AsyncMock(side_effect=_fake_embed)):
        graph = build_graph()
        user_text_1 = "My favorite color is teal."
        out1 = await graph.ainvoke({"user_id": chat_id, "user_text": user_text_1,
                                    "image_bytes": None})
    reply_text_1 = out1["reply"]["text"]

    recent = await memory.recent_turns(chat_id, 10)
    assert recent == []  # nothing logged yet — this turn hasn't been written

    with patch("bot.nodes.persist_memory.llm.chat",
               new=AsyncMock(return_value=f'["{fact}"]')), \
         patch("bot.memory.llm.embed", new=AsyncMock(side_effect=_fake_embed)):
        await persist_memory(chat_id, user_text_1, reply_text_1, recent)
    await memory.log_turn(chat_id, "user", user_text_1)
    await memory.log_turn(chat_id, "assistant", reply_text_1)

    # --- turn 2: a fresh graph invocation must recall the fact via load_memory ---
    with patch("bot.nodes.agent._register_turn", new=AsyncMock()), \
         patch("bot.nodes.agent._pop_found_image", new=AsyncMock(return_value=None)), \
         patch("bot.nodes.agent.call_hermes",
               new=AsyncMock(return_value="Teal is a great color!")), \
         patch("bot.nodes.compose_persona.llm.chat",
               new=AsyncMock(return_value="Teal is a great color!")), \
         patch("bot.memory.llm.embed", new=AsyncMock(side_effect=_fake_embed)):
        graph2 = build_graph()
        out2 = await graph2.ainvoke({"user_id": chat_id, "user_text": "what's my favorite color?",
                                     "image_bytes": None})

    assert fact in out2["memories"]

    # and log_turn from turn 1 is visible to turn 2's agent history
    turns_after = await memory.recent_turns(chat_id, 10)
    assert {"role": "user", "content": user_text_1} in turns_after
    assert {"role": "assistant", "content": reply_text_1} in turns_after
