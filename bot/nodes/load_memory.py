from bot import memory


async def load_memory(state) -> dict:
    user_id = state["user_id"]
    profile = await memory.get_profile(user_id)
    memories = await memory.search_memories(user_id, state["user_text"])
    return {"profile": profile, "memories": memories}
