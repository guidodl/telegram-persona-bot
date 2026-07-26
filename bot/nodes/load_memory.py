from bot import memory


async def load_memory(state) -> dict:
    chat_id = state["chat_id"]
    profile = await memory.get_profile(chat_id)
    memories = await memory.search_memories(chat_id, state["user_text"])
    return {"profile": profile, "memories": memories}
