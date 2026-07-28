# Design: Hermes-Agent sidecar as the agent rung

**Date:** 2026-07-28
**Status:** approved (brainstorm), pending spec review
**Supersedes decision:** revisits [2026-07-26 — custom langgraph graph over Hermes Agent framework], which rejected the Nous framework for this bot. This spec adopts it deliberately, accepting the two known tradeoffs (see Accepted Risks).

## Goal

Replace the hand-rolled tool loop in the `agent` node with the **Nous Research Hermes-Agent** framework (`nousresearch/hermes-agent`, the MCP-native tool-calling agent) running as an out-of-process **sidecar**. Hermes owns the reason-act-observe loop server-side; it hands final facts back to `compose_persona`, which is unchanged and remains the sole source of user-visible text.

### Why the framework (user's stated drivers)
- **Server-side agentic loop** — Hermes owns multi-step tool chains / self-correction instead of the bounded 6-iteration hand-rolled loop.
- **Runtime model switching** — swap provider/model per request via Hermes' provider abstraction, no app-code change.
- **Consistency with agentBerry** — reuse the same Hermes-sidecar + MCP-tools pattern agentBerry already runs, so both projects share one integration shape.

### Hard requirement
All four existing tools must keep working: `web_search` (Tavily), `recall` (pgvector memory), `image_search` (Brave), `vision_analyze`. They are **re-exposed as an MCP server**; Hermes discovers them from an allowlist and calls them during its loop.

## Non-goals
- Changing `compose_persona`, the persona/silence guarantee, the memory schema, or the Telegram ingress/egress.
- Running model weights locally. Hermes is provider-agnostic and calls out to OpenRouter/Bedrock/etc. for inference. The Pi carries only the orchestration runtime + a small MCP server.
- Multi-tenant hardening beyond today's posture (see Accepted Risks).

## Architecture

Three containers via `docker-compose` (adds two to today's `bot` + `db`):

```
Telegram
   |
   v
[bot]  --HTTP/SSE-->  [hermes]  --MCP-->  [mcp-tools]
   |  POST turn ctx ------------------------^  (reads/writes per-turn ctx)
   |
   +-- load_memory -> agent(HTTP call) -> compose_persona -> END
   |
[db: pgvector]  <--- recall tool queries memory
```

- **`bot`** — unchanged LangGraph app except the `agent` node body. Owns Telegram I/O, `load_memory`, `compose_persona`, memory persistence.
- **`hermes`** — `nousresearch/hermes-agent` (arm64 variant confirmed on Docker Hub). Loopback gateway on `:8642`, OpenAI-shaped `POST /v1/chat/completions`, `stream=true`. Runs the tool loop server-side; emits `hermes.tool.progress` SSE status events and final `delta.content`. Config (mounted file) declares the MCP server + a `tools.include` allowlist and disables Hermes' mutating built-ins.
- **`mcp-tools`** — small Python MCP server wrapping the four existing tool functions from `bot/tools.py`. Also holds the per-turn context store (below).

## The agent node, rewritten

`agent_node` (`bot/nodes/agent.py`) changes from a hand-rolled `llm.chat_with_tools` loop to:

1. Generate a `turn_id`.
2. `POST` `{turn_id: {user_id, image_bytes}}` to `mcp-tools` (registers per-turn inputs).
3. `POST` the messages to Hermes at `:8642` with `stream=true`, model selected per request (runtime switching). Prefix the `turn_id` so tool calls can correlate (system message or session id — resolved at implementation against Hermes' config surface).
4. Read the SSE stream; take the final `delta.content` as `raw_result`. Ignore `hermes.tool.progress` events (status only).
5. `GET`/read back `found_image_url` from `mcp-tools` by `turn_id`, then delete the entry.
6. Return the **same shape as today**: `{raw_result, found_image_url, agent_error}`.

`compose_persona` and `ingress` are untouched — they still consume `raw_result` and `reply.image_url` exactly as now.

## Per-turn context store (the turn_context replacement)

Today `turn_context` is an in-process `ContextVar` carrying `user_id`, `image_bytes` (inputs) and `found_image_url` (output side-channel). That in-process channel does not cross the sidecar boundary. Replacement — **keep it simple, no DB, nothing persisted**:

- The **MCP server holds an in-memory dict** keyed by `turn_id`: `{turn_id: {user_id, image_bytes, found_image_url}}`.
- `agent_node` POSTs `{user_id, image_bytes}` under a fresh `turn_id` before calling Hermes.
- Each MCP tool receives `turn_id` as an argument and reads inputs (`user_id` for `recall`, `image_bytes` for `vision_analyze`) / writes outputs (`found_image_url` for `image_search`) against that dict.
- After Hermes returns, `agent_node` reads `found_image_url` back and the entry is dropped.

Rationale for in-memory over Postgres: the bot serves essentially serial personal/DM traffic; a process-local dict is sufficient, adds no infra, and fits the Pi. Entries are short-lived (one turn) and dropped after read. **Consequence:** `mcp-tools` becomes stateful per-turn — it must be a single instance (no horizontal scaling), which is fine for this deployment.

Why the side-channel needs this at all: Hermes folds tool *results* into its reasoning and returns only prose. A tool's *side effect* (`found_image_url`) won't appear in the SSE stream, so the bot retrieves it out-of-band from the store after the loop finishes.

## Tool re-exposure

`mcp-tools` wraps the four functions. Bodies are largely lifted from `bot/tools.py`, with the `turn_context.get()[...]` accesses replaced by lookups into the per-turn store via the `turn_id` argument:
- `web_search(query)` — unchanged logic (Tavily).
- `recall(query, turn_id)` — reads `user_id` from store, queries pgvector.
- `image_search(query, turn_id)` — Brave search; writes `found_image_url` into store.
- `vision_analyze(turn_id)` — reads `image_bytes` from store; calls the vision model.

Hermes config allowlists exactly these four and disables its own mutating built-ins (file/patch/execute_code/etc.), mirroring agentBerry's read-only-scoping approach.

## Deployment (Pi)

- Add `hermes` and `mcp-tools` services to `docker-compose.yml`; pin the arm64 image tag for `hermes`.
- Hermes reaches inference providers over the network (OpenRouter key, etc.) — no local weights.
- Staging: bring the three-container stack up **locally via docker-compose first**, validate end-to-end (text turn, web-research turn, photo-in `vision_analyze` turn, photo-out `image_search` turn), then roll to `hermespi.local` via the existing rsync + `docker-compose` rebuild flow.

## Silence guarantee

Preserved at the interface: `compose_persona` remains the choke point and the only text sent to Telegram. **New failure mode to guard:** Hermes' final `delta.content` is prose that may contain meta-narration ("I searched…", hedges). `compose_persona` already strips jargon and re-voices, but its current input is minimal facts, not agent prose. Mitigation: instruct Hermes (via its system/config prompt) to return terse factual findings, not narrated answers — keeping its output shaped like the current `raw_result`. Verified during local staging.

## Testing

- **Unit:** `agent_node` mocked against a fake Hermes SSE stream (final content + progress events) — asserts `raw_result` extraction, `found_image_url` round-trip via a mock store, and `agent_error` on stream failure. MCP tool bodies unit-tested directly (as `bot/tools.py` is today).
- **Integration (local compose):** the four end-to-end turn types above.
- Existing `compose_persona`/ingress tests remain green unchanged (proof the boundary held).

## Accepted risks (carried from the 2026-07-26 rejection, knowingly accepted)

1. **Opacity vs. auditability** — Hermes' loop is server-side; the bot loses the inspectable per-tool-call trace it has today. Downgrade in defense-in-depth, accepted for the framework's benefits.
2. **Security model mismatch** — Hermes-Agent's `SECURITY.md`: the OS is its only security boundary; built for single-tenant personal use. This bot serves group chats with arbitrary members and holds per-user memory. Untrusted input (incl. web_search results) flows into that loop. Mitigated only by the read-only tool allowlist + disabled built-ins; residual risk accepted for a personal bot.

## Open items to resolve at implementation
- Exact mechanism to pass/correlate `turn_id` into Hermes tool calls (system-message prefix vs. session id) — pin against the deployed Hermes version's config surface.
- Hermes config file format + the arm64 image tag to pin.
- Latency budget: Hermes' server-side loop vs. today's `AGENT_MAX_ITERATIONS=6` cap — confirm acceptable on the Pi during staging.
