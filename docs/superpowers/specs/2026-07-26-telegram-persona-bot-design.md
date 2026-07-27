# Telegram Persona Bot — Design

**Date:** 2026-07-26
**Status:** Approved (design). Hermes-as-node vs. Hermes-as-model resolved by the
implementation plan (O1: `agent` node is an embedded `langstage-hermes` graph, not
a bare model call). See `2026-07-26-telegram-persona-bot-design.PLAN.md`.

## Summary

A new, from-scratch Telegram bot that roleplays a **per-user adaptive persona**. It
holds a natural conversation, remembers each user and evolves its relationship with
them over time, and can use tools (web/knowledge lookup, long-term memory recall,
external actions) — but **never reveals that it is using tools or reasoning**. The
user only ever sees in-character replies. The brain is the **Hermes** model served
via **OpenRouter** (OpenAI-compatible, native function-calling), orchestrated with
**langgraph/langchain**.

agentBerry (in this workspace) is a *reference only* for how to wire a langgraph
graph with a Hermes tool loop. This bot shares none of its code, deployment, or the
Slack-oriented "show what I'm doing" behavior — in fact it does the opposite.

## Goals

- Conversational Telegram bot with a believable, human-like voice.
- Persona **adapts per user**: remembers facts, preferences, tone, and the running
  relationship; the character's knowledge of each user compounds over time.
- Retains capability: **web/knowledge lookup**, **long-term memory recall**,
  **external actions** — executed silently.
- **Multimodal (v1)**: understands images the user sends (vision), can reply with a
  **voice note (TTS)**, and can **send images** it fetches.
- **Hidden reasoning**: tool names, step traces, and raw results never reach the
  user. Only in-character text/voice/images are sent.
- Small-production scale: handful-to-hundreds of users, deployed on existing
  container/k8s infra, must stay up and scale modestly.

## Non-Goals (v1 — YAGNI)

- No multi-persona configuration UI or admin dashboard.
- No analytics/telemetry pipeline.
- No large-scale queuing/horizontal-autoscale engineering (revisit if usage grows).
- No self-hosting the model (OpenRouter only).
- **No voice-in / transcription** (STT). Users' voice notes are not processed in v1;
  deferred to a later version. Voice is **output-only** (TTS).
- **No image generation** — sent images are *fetched* existing images, not generated.
- **DM-only.** v1 handles 1:1 chats. Group chats are out of scope: thread state and
  memory are keyed by `chat_id`, which equals the user only in a DM. Group support
  needs a separate user dimension in the memory model — deferred.

## Architecture

A single langgraph graph inside one Telegram bot process. Per inbound message:

```
Telegram update (text | photo | [voice-in: NOT in v1])
  → ingress (python-telegram-bot handler; shows native "typing…")
       photo? → attach image as OpenAI vision content (base64 data URL) to the turn
  → graph.ainvoke(state, thread_id = telegram_chat_id)
       load_memory      → user profile + top-K semantic memories from Postgres
       agent            → Hermes tool loop (the "WORK" call); tools bound.
                          If the turn carries an image and the model is text-only,
                          a vision_analyze tool routes it to a vision model.
                          ↺ web_lookup / recall / actions / image_search / vision_analyze
       compose_persona  → rewrites raw result into in-character voice
                          (the "SPEAK" call; no tools). Decides reply modality:
                          text, and optionally voice-out and/or an image to send.
  → ingress renders the reply (send-guard enforced):
       text  → sendMessage
       voice → TTS synth → sendVoice
       image → fetched image → sendPhoto
  → persist_memory   → extracts + stores new durable facts, updates summary.
                       Runs AFTER the reply is sent (fire-and-forget / background
                       task keyed by thread_id) so it never blocks the reply path.
```

### Latency budget

A plain turn is at least three model round-trips — embedding (`load_memory`) +
work call (`agent`) + speak call (`compose_persona`) — plus any tool loop or vision
call, plus the optional pacing delay. `persist_memory`'s extraction call is a
*fourth* round-trip but is deliberately moved off the critical path (post-send). The
human-feel goal constrains this stack-up: set an explicit end-to-end latency target
at implementation and keep the pre-send path to those three calls.

### The two-call split (core design decision)

Every turn makes **two distinct model calls**:

1. **Work call** (`agent`) — Hermes with tools bound. Reasons, calls tools in a
   loop, produces a raw factual result. **This output never goes to Telegram.**
2. **Speak call** (`compose_persona`) — Hermes with **no** tools, given the persona
   system prompt + per-user relationship context + the raw result. Produces the
   **only** text the user sees.

This split is the *structural* guarantee for hidden reasoning: the Telegram
outbound path is wired only to `compose_persona`. Tool names, step traces, and raw
lookups have no code path to the user.

### Thread state

The langgraph checkpointer keys short-term conversation state by Telegram chat ID,
so working context survives across messages within a session. This is ephemeral
working memory, distinct from long-term persisted memory.

## Components & Boundaries

Every unit has one responsibility, a clear interface, and known dependencies.
Everything reaches the model through the `llm` facade and storage through the
`memory` facade — swapping OpenRouter or the DB later touches one file each.
`compose_persona` is the **only** component that produces user-facing text.

| Component | Responsibility | Interface | Depends on |
|---|---|---|---|
| `ingress` | Receive Telegram updates, send replies, show "typing…", enforce send-guard | `on_message(update)` → graph | python-telegram-bot |
| `graph` | Wire nodes/edges, checkpointer | `ainvoke(state)` | langgraph |
| `state` | GraphState schema | TypedDict | — |
| `nodes/load_memory` | Fetch profile + top-K memories | `(state) → {profile, memories}` | memory |
| `nodes/agent` | Hermes tool loop ("work" call) | `(state) → {raw_result}` | llm, tools |
| `nodes/compose_persona` | Rewrite raw result in-character ("speak" call) | `(state) → {reply_text}` | llm, persona |
| `nodes/persist_memory` | Extract + store new facts, update summary | `(state) → {}` | memory, llm |
| `tools/web_lookup` | Web/knowledge search | `(query) → text` | search provider |
| `tools/recall` | Semantic memory query mid-loop | `(query) → memories` | memory |
| `tools/actions` | External actions | `(action, args) → result` | external APIs |
| `tools/image_search` | Find an existing image to send | `(query) → image_url/bytes` | image search provider |
| `tools/vision_analyze` | Describe an image via a vision model (fallback when the main model is text-only) | `(image) → text` | llm (vision model) |
| `llm` | Model facade (chat + vision via OpenRouter; embeddings possibly a second backend) | `chat(msgs, tools?)`, `embed(text)` | OpenRouter (+ embedding provider if needed) |
| `tts` | Text→speech synthesis for voice-out | `synth(text) → audio bytes` | TTS provider |
| `media` | Ingress helpers: download Telegram photo → base64 vision content; build voice/photo sends | `to_vision_content(file)`, `to_voice(audio)` | python-telegram-bot |
| `memory` | Persistence facade | `get_profile / search / upsert / forget` | Postgres+pgvector |
| `persona` | Persona assembly per user | `build_system_prompt(profile)` | memory |
| `config` | Env/settings | constants | — |

## Memory & Persona (per-user adaptive)

Store: **Postgres + pgvector** — structured facts and semantic recall in one DB,
production-friendly at the target scale.

**Three layers:**

1. **User profile** (`users` table) — stable per-user facts: name, stated
   preferences, relationship tone, and a running summary of who the user is. One row
   per Telegram user. Loaded every turn and injected into the persona system prompt.
2. **Episodic memories** (`memories` table, pgvector) — timestamped snippets
   ("learning guitar", "prefers short replies"), embedded for semantic search.
   `load_memory` pulls top-K by relevance to the current message; the `recall` tool
   can query more mid-loop.
3. **Thread state** (langgraph checkpointer) — short-term conversation buffer keyed
   by chat ID. Ephemeral working context, not long-term truth.

**How adaptation works:** `persist_memory` runs after each reply is *sent* (off the
critical path — see Latency budget). A cheap Hermes
call extracts any new durable facts from the exchange and (a) upserts episodic
memories, (b) updates the running profile summary. Next turn,
`build_system_prompt(profile)` folds that into the persona, so the character's
knowledge of the user compounds. Tone/relationship fields let the persona shift per
person.

**Embeddings:** an OpenAI-compatible embedding model via the `llm` facade.
**Caveat:** OpenRouter is a chat-completions router and may not expose an
`/embeddings` endpoint; if so, embeddings need a second provider (OpenAI direct,
Voyage, Cohere, …) and the `llm` facade fans out to more than one backend. Verify
before assuming "no extra infra." See Open Questions.

**Privacy:** a `/forget` command wipes a user's rows (profile + memories). Included
from day one given the bot stores personal facts.

## Silent Reasoning & Human-Like Behavior

Enforced structurally, not by prompt alone.

**Silence guarantees:**
- Telegram's outbound path is wired to `compose_persona` **only**. The `agent`
  node's raw output, tool names, and step traces are never a message source.
- Tool results stay in graph state; `compose_persona` receives them as *facts to
  speak from*, with an explicit instruction never to mention how it knows or that a
  lookup/tool ran.
- The only in-flight signal is Telegram's native **"typing…"** action — generic and
  human. No "Searching GitHub…"-style status leaks.

**Human-like behavior:**
- **Persona system prompt** carries voice + per-user relationship context, so
  replies are personal.
- **Natural pacing** (toggleable): optional short randomized delay and chunking long
  replies into 1–2 messages, the way a person texts.
- **No bot tells**: compose prompt strips markdown-heavy formatting, tool jargon, and
  "As an AI…" hedging.
- **Graceful failure**: if a tool or the work call fails, `compose_persona` still
  speaks in-character ("hmm, I couldn't pull that up right now") rather than emitting
  an error trace.

**Send-guard (code-level backstop):** `ingress` refuses to send if the reply is
empty or looks like a raw tool/JSON payload — so a bug can never leak mechanics.
This is a best-effort heuristic, not a guarantee: "looks like JSON" can
false-positive on a legitimate reply containing braces or a code snippet. Tune the
check to minimize both leaks and false rejects; the structural two-call split, not
the guard, is the real silence guarantee.

## Multimodal (v1)

All modalities stay inside the two-call split — the model reasons over inputs in the
`agent` call; `compose_persona` decides what modality to reply in and produces only
in-character output. Nothing about tools/vision internals reaches the user.

**Image understanding (vision input):** When a user sends a photo, `ingress`
downloads it and attaches it as **OpenAI vision content** (a base64 `data:` URL in
`{"type":"image_url",...}` form) to the turn. Hermes on OpenRouter passes images in
exactly this format. If the configured model slug is **vision-capable**, the image
goes inline to the `agent` call. If it is **text-only**, the `vision_analyze` tool
routes the image to a vision-capable model via the same `llm` facade and returns a
textual description — the same fallback the Hermes agent product uses. The
implementation picks the path based on the model slug's capabilities.

**Voice output (TTS):** `compose_persona` may flag a reply for voice. `ingress` then
synthesizes the text via the `tts` facade and sends it as a Telegram **voice note**
(`sendVoice`). This is output-only — **voice-in / transcription is not in v1.**

**Send images (fetch):** The `image_search` tool retrieves an existing image
(web-image search or a curated source) matching what the persona wants to share.
`compose_persona` may attach it; `ingress` sends via `sendPhoto`. **No image
generation** in v1. **Content safety:** fetching arbitrary web images to send raises
content-safety and copyright exposure. Prefer a curated/licensed source or a
provider with safe-search filtering; the chosen provider must address this (see Open
Questions).

**Reply modality:** a turn can produce text, a voice note, an image, or a
combination. `compose_persona` returns a structured reply (`text`, optional
`voice: bool`, optional `image_query`); `ingress` renders each present field. The
send-guard applies to the text/caption fields as before.

## Tech Stack

- **Python 3.12**
- **python-telegram-bot** (async) — ingress
- **langgraph** — graph + checkpointer
- **langchain-openai** client pointed at OpenRouter (OpenAI-compatible) — Hermes chat
  + embeddings + vision (inline image_url content)
- **Postgres + pgvector** — memory; **SQLAlchemy** or `psycopg` behind the `memory`
  facade
- **TTS provider** — voice-out synthesis (chosen at implementation). Note OpenRouter
  does not do audio synthesis, so TTS is a separate provider/facade regardless;
  an OpenAI-compatible TTS is preferred to minimize new client surface
- **Image search provider** — for fetched images (chosen at implementation)
- **Deployment:** container on existing k8s/small-production infra; config via env vars

## Project Layout

```
telegram-persona-bot/
  bot/
    ingress.py          # telegram handlers, send guard, media render (text/voice/photo)
    graph.py            # node/edge wiring, checkpointer
    state.py            # GraphState TypedDict
    config.py
    llm.py              # OpenRouter facade (chat + embeddings + vision)
    memory.py           # pg+pgvector facade
    persona.py          # build_system_prompt(profile)
    tts.py              # text→speech facade (voice-out)
    media.py            # telegram photo→vision content, voice/photo send helpers
    nodes/
      load_memory.py  agent.py  compose_persona.py  persist_memory.py
    tools/
      web_lookup.py  recall.py  actions.py  image_search.py  vision_analyze.py
  tests/
    unit/  integration/
  migrations/           # pgvector schema
  pyproject.toml
  README.md
  Dockerfile
```

## Testing

- **Unit:** each node with mocked `llm`/`memory`; send-guard rejecting raw-payload
  text; persona prompt assembly.
- **Integration:** real Postgres (or testcontainers) for the memory round-trip
  (upsert → semantic recall); a full graph run with stubbed OpenRouter verifying
  **only** `compose_persona` output reaches the sink and no tool string leaks.
- **Faithfulness:** assert `compose_persona` preserves the facts in `raw_result` —
  the persona rewrite must not drop or invent facts, only restyle them. (The
  no-tools speak call has no grounding of its own, so drift is the failure mode to
  guard, distinct from the no-leak check above.)
- **Multimodal:** unit-test that an inbound photo becomes OpenAI vision content;
  that a text-only model routes to `vision_analyze`; that a voice-flagged reply calls
  `tts` and `sendVoice`; that `image_search` output reaches `sendPhoto`.
- **Manual:** a live Telegram test chat for voice-out, image understanding, sending a
  fetched image, pacing, and the `/forget` command.

## Open Questions / Deferred

- **Does OpenRouter expose an OpenAI-compatible `/embeddings` endpoint?** If not,
  pick a dedicated embedding provider and accept a multi-backend `llm` facade. This
  is a load-bearing assumption behind "no extra infra" — verify first.
- **End-to-end latency target.** A plain turn is 3 pre-send model round-trips
  (+ tool loop / vision / pacing). Set a concrete target and confirm the pacing
  delay and call stack-up stay within it, given the human-feel goal.
- Which web-search provider backs `web_lookup` (decide at implementation).
- Which image-search provider backs `image_search` (decide at implementation) —
  must support safe-search / licensed or curated sources to bound content-safety and
  copyright risk.
- Which TTS provider backs `tts` (OpenAI-compatible preferred; decide at
  implementation).
- Concrete scope of `external actions` for v1 (which actions, which APIs) — start
  with a single well-defined action and a clean tool interface.
- Exact Hermes model slug on OpenRouter, its **vision capability** (determines inline
  vision vs. `vision_analyze` fallback), and the embedding model choice.
- **Deferred to a later version:** voice-in / transcription (STT); image generation.
