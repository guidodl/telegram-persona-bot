# Spike notes — langstage-hermes API surface (Task 1, gating)

**Date:** 2026-07-26
**Package:** `langstage-hermes[openai]==0.4.25` (pinned; Python 3.12.13)
**Decision:** ⛔ **FALLBACK BRANCH** — hand-rolled DeepSeek tool loop in
`bot/nodes/agent.py`, plain tool module `bot/tools.py`. Recorded **before** any
bridge code was written, per plan Task 1 / O12.

## What was verified

- `create_hermes_agent(config, *, workspace, session_id, extra_middleware,
  backend, model, aux_model)` exists; live `langstage-hermes verify` **passes**
  against OpenRouter with `openai:deepseek/deepseek-v4-flash`
  (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`). Model slug and round-trip work.
- `HermesConfig` (flat field names, cosmetic diff from plan):
  `model_default`, `model_aux`, `agent_disabled_toolsets`,
  `agent_max_iterations=90` (O5 default ✓), `memory_provider`, `memory_enabled`,
  `plugins_enabled`/`plugins_disabled`. No nested `model.*`/`memory.*` objects.
- `MemoryProvider` ABC (`langstage_hermes.memory.provider`):
  `setup_session(session_id, user_id)`, `recall(query, mode="hybrid") -> list[str]`,
  `record_turn(role, content)`, `teardown()` — all **synchronous** (plan assumed async).
- Plugin discovery: 4 sources, incl. pip entry-point group
  `langstage_hermes.plugins` and dir-scan (`plugin.yaml` + `register(ctx)`).
  `PluginContext.register_tool(tool, *, toolset)`,
  `register_memory_provider(name, cls)`.

## Why the bridge is structurally unviable in 0.4.25

1. **`MemoryProvider` is never invoked by the runtime.** `create_hermes_agent`
   instantiates the registered class and attaches it to the compiled graph
   (`compiled.langstage_hermes_provider`), but no middleware calls
   `setup_session`/`recall`/`record_turn` (grep: zero call sites outside
   `provider.py`). The O3 bridge — "Hermes' recall middleware reads from
   pgvector" — is inert in this version.
2. **Plugin discovery never runs on the factory path.** `create_hermes_agent`
   calls only `ensure_builtin_providers()`; `HermesPluginLoader.discover()` is
   CLI-only. Tools registered via `register(ctx)` never reach the agent. The
   only seam for custom tools is `extra_middleware` (documented for
   "tracing or auth", not tools).
3. **No built-in `web` toolset / no Tavily.** Zero Tavily references in the
   package; the factory wires `skill_tools + session_search_tool` only. O4's
   "enable `web`, it honors `TAVILY_API_KEY`" does not exist — web lookup would
   be a custom tool either way.
4. **Per-user `HERMES_HOME` is unsafe in-process.** Home resolves from
   process-global env (`DEEPAGENT_HERMES_HOME → HERMES_HOME → ~/.langstage-hermes`)
   and `MemoryToolMiddleware` re-resolves it **per call**. With concurrent
   users, per-invocation env swaps bleed markdown memory across users; a global
   lock would serialize all turns. O2's in-process per-user isolation is not
   safely achievable. (`state.db` *is* captured per-build — but markdown
   memory, skills, and session stores are not all build-time.)
5. **System prompt is not host-replaceable.** `PromptAssemblyMiddleware` owns
   the system prompt (full CLI coding-agent prompt); no config seam to
   substitute our own. Fights the persona use case and inflates every work call.
6. **Toolset config filters prompt text only.** Built-in tools (memory, file,
   delegation, todo) are wired unconditionally via middleware;
   `agent_disabled_toolsets` does not unwired them.

Per plan Task 1 step 4: the provider/plugin/toolset-config mechanisms the plan
relies on are **not usable as designed**; proceeding would require the exact
"improvised workarounds" (global locks, monkeypatches, undocumented seams)
O12 forbids. → **FALLBACK BRANCH**.

## Fallback consequences (plan Tasks 6–7 shrink; all other tasks unchanged)

- `bot/tools.py` — plain async tools: `web_search` (Tavily, O8), `recall`,
  `image_search` (Brave, `safesearch=strict`), `vision_analyze` (custom,
  `llm.chat_vision`). Tool bodies identical to the plan's Task 6.
- `bot/nodes/agent.py` — hand-rolled tool loop over `llm.chat` with tools
  bound, bounded iterations (≤ `settings.agent_max_iterations`, default 90 per
  O5), same turn-context set/reset and return shape
  (`raw_result` / `found_image_url` / `agent_error`). Never raises into the
  reply path. Conversation continuity via `memory.recent_turns(chat_id)`.
- `bot/hermes_node.py`, `bot/hermes_plugin/` — **not created**. No per-user
  `HERMES_HOME` (nothing to isolate); `/forget` = Postgres wipe only.
- `langstage-hermes` is **dropped from runtime dependencies** (spike-only);
  `langchain-openai` also dropped — the loop uses the `bot.llm` facade directly.
  `bot.llm` gains a tool-calling variant (returns the raw assistant message
  with `tool_calls`) alongside the str-returning `chat`.
- AC-4 resolves via the sanctioned alternative: "the recorded fallback loop
  with the same tool surface".

## Env notes

- `langstage-hermes verify` was run with `OPENAI_BASE_URL`/`OPENAI_API_KEY`
  pointed at OpenRouter: PASS (3.4s round-trip). Env var prefix is
  `LANGSTAGE_HERMES_*` (`DEEPAGENT_HERMES_*` deprecated).
