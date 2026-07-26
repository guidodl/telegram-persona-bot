import asyncio
from bot.turn_context import turn_context

async def test_context_is_task_local():
    async def worker(uid):
        turn_context.set({"image_bytes": None, "found_image_url": None, "user_id": uid})
        await asyncio.sleep(0.01)
        return turn_context.get()["user_id"]
    results = await asyncio.gather(worker(1), worker(2), worker(3))
    assert sorted(results) == [1, 2, 3]  # no cross-task bleed
