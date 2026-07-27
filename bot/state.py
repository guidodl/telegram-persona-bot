from typing import TypedDict


class GraphState(TypedDict):
    user_id: int
    user_text: str
    image_bytes: bytes | None
    profile: dict
    memories: list[str]
    raw_result: str | None
    found_image_url: str | None
    agent_error: str | None
    reply: dict
