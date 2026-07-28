_turns: dict[str, dict] = {}

def register(turn_id: str, user_id: int, image_bytes: bytes | None) -> None:
    _turns[turn_id] = {"user_id": user_id, "image_bytes": image_bytes,
                       "found_image_url": None}

def get(turn_id: str) -> dict:
    return _turns[turn_id]

def set_found_image(turn_id: str, url: str) -> None:
    _turns[turn_id]["found_image_url"] = url

def pop_found_image(turn_id: str) -> str | None:
    entry = _turns.pop(turn_id, None)
    return entry["found_image_url"] if entry else None
