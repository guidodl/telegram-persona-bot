import base64
import contextlib
import logging
from fastapi import FastAPI, Response
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp_tools import store, tools


class _DropPingRequest(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "PingRequest" not in record.getMessage()


# Hermes's MCP client keepalive-pings this server continually; the SDK logs
# every request at INFO, and a ping carries no information beyond "still up".
logging.getLogger("mcp.server.lowlevel.server").addFilter(_DropPingRequest())

# streamable_http_path="/" so mounting the app at "/mcp" yields the final
# route "/mcp" (Hermes' `type: http` transport POSTs there). sse_app() would
# instead serve /mcp/sse and 404 the streamable-HTTP client.
# DNS-rebinding protection defaults to localhost-only allowed_hosts, which
# 421s the docker-DNS host "mcp-tools:8000" Hermes connects with. Disabled:
# this server binds only to the internal compose network, never a host port.
mcp = FastMCP("persona-tools", streamable_http_path="/",
              transport_security=TransportSecuritySettings(
                  enable_dns_rebinding_protection=False))

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

mcp_app = mcp.streamable_http_app()


# The streamable-HTTP app runs a session manager in its lifespan; a mounted
# sub-app's lifespan is not run by Starlette unless the parent delegates to it.
@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


app = FastAPI(lifespan=lifespan)

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

app.mount("/mcp", mcp_app)
