from contextvars import ContextVar
turn_context: ContextVar[dict] = ContextVar("turn_context")
