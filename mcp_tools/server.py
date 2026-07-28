import base64
from fastapi import FastAPI, Response
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp_tools import store, tools

app = FastAPI()

class TurnIn(BaseModel):
    turn_id: str
    user_id: int
    image_bytes_b64: str | None = None

@app.post("/turns", status_code=204)
def register_turn(t: TurnIn) -> Response:
    img = base64.b64decode(t.image_bytes_b64) if t.image_bytes_b64 else None
    store.register(t.turn_id, t.user_id, img)
    return Response(status_code=204)

@app.get("/turns/{turn_id}/found_image")
def found_image(turn_id: str) -> dict:
    return {"found_image_url": store.pop_found_image(turn_id)}

mcp = FastMCP("persona-tools")

@mcp.tool(description="Search the web for fresh facts.")
async def web_search(query: str) -> str:
    return await tools.web_search(query)

@mcp.tool(description="Recall durable facts remembered about this user.")
async def recall(query: str, turn_id: str) -> str:
    return await tools.recall(query, turn_id)

@mcp.tool(description=("Find an existing image to send. Phrase the query as neutral "
    "descriptive English (e.g. 'portrait of a young woman smiling', 'ginger cat on a "
    "sofa'): slang or compliment-heavy phrasing in any language returns zero results. "
    "If a search finds nothing, retry once with simpler neutral wording."))
async def image_search(query: str, turn_id: str) -> str:
    return await tools.image_search(query, turn_id)

@mcp.tool(description="Describe the photo the user attached this turn.")
async def vision_analyze(turn_id: str) -> str:
    return await tools.vision_analyze(turn_id)

app.mount("/mcp", mcp.sse_app())
