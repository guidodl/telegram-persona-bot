from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from bot.config import settings
from bot.state import GraphState
from bot.nodes.load_memory import load_memory
from bot.nodes.agent import agent_node
from bot.nodes.compose_persona import compose_persona


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("agent", agent_node)
    graph.add_node("compose_persona", compose_persona)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "agent")
    graph.add_edge("agent", "compose_persona")
    graph.add_edge("compose_persona", END)
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def postgres_checkpointer():
    """Production checkpointer, opened lazily against settings.database_url.

    Not used by build_graph() directly so that build_graph() stays DB-free
    and unit-testable. Runs .setup() on entry — required by langgraph to
    create the checkpoints/checkpoint_migrations tables on first use, and
    idempotent on subsequent calls. Callers that need real persistence
    (e.g. the bot's startup code) should do:

        async with postgres_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            ...
    """
    conn_string = settings.database_url.replace("+psycopg", "").replace("+asyncpg", "")
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
