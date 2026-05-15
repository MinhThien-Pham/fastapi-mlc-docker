# Chat Architecture: Why and How

## Original repository scope

This repository started as a FastAPI service wrapper around
[`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/) — a Go project
that automates the MLC-LLM build pipeline (TVM compilation, weight conversion,
model compilation).

The original API surface maps directly onto `mlc-cli` sub-commands:

| Endpoint | Underlying tool |
|---|---|
| `POST /build` | `go run . build` |
| `POST /quantize` | `go run . quantize` |
| `POST /compile` | `go run . compile` |
| `POST /run` | `go run . run` |

These endpoints all follow the same pattern: spawn a subprocess, stream its
stdout/stderr as Server-Sent Events, return when the process exits.
They are batch, fire-and-forget operations and that pattern serves them well.

---

## Why `/run` is not a good fit for interactive chat

The `go run . run` command shells out to `python -m mlc_llm.cli run`, which is
an **interactive terminal REPL**. It reads user prompts from `stdin` and writes
responses to `stdout`.

When driven by this API:

- No stdin is provided → the REPL receives EOF immediately after model init
- The model loads successfully, prints a ready message, then exits
- This is useful as a **load-test / runtime verification** step, not as a chat API

The `POST /run` endpoint's own docstring captures this explicitly:

> **LIMITATION**: The upstream `mlc-cli run` command is interactive by default
> and does NOT support a non-interactive single-shot `--prompt` flag. When called
> via this API endpoint, no standard input is provided. The subprocess will
> initialize the model, print its ready state, and immediately exit upon
> encountering EOF. This effectively serves as a "load test" to verify model
> and compiled library compatibility.

Beyond this stdin limitation, making the REPL work programmatically would
require parsing unstructured terminal output — ANSI codes, prompt markers,
no machine-readable delimiters — which is fragile and not maintainable.

---

## Design directions considered

Three backend approaches were evaluated before picking the one this repo uses.

| Approach | What it means | Pros | Cons | Decision |
|---|---|---|---|---|
| **A — Pipe the Go CLI REPL** | Keep calling `go run . run`, but pipe user messages to its stdin and read its stdout for replies. | No architectural change — stays consistent with existing subprocess pattern. | The REPL output contains ANSI codes and prompt markers with no stable format. Parsing it reliably is effectively guesswork and will break when the upstream tool changes. Too fragile to maintain. | ❌ Rejected |
| **B — Call `MLCEngine` directly** _(chosen)_ | Import `mlc_llm.MLCEngine` in the FastAPI process and call its `chat.completions.create()` method. | Clean API boundary: structured OpenAI-compatible responses, no output parsing, easy to mock in tests, full GPU lifecycle control under FastAPI's supervision. | `mlc_llm` must be importable in the FastAPI process. Confirmed acceptable — the Docker image already installs it. | ✅ Selected |
| **C — Proxy to `mlc_llm serve`** | Launch `mlc_llm serve` as a subprocess (it starts its own Uvicorn server) and forward requests to it. | `mlc_llm serve` is OpenAI-compatible out of the box — minimal request/response logic to write. | Runs a Uvicorn server inside a Uvicorn server. Doubles the failure surface: port management, subprocess health, startup timing, and opaque error propagation. Unnecessarily complex. | ❌ Rejected |

> **Note on WebSockets:** A WebSocket transport (`WS /chat/ws`) was also considered, but it is not a competing backend approach — it would still use the same direct `MLCEngine` integration as Approach B. WebSocket is a transport-layer choice, not a backend-engine choice. It could be added later as an optional enhancement on top of the existing HTTP endpoints, but the harder problem first was getting `MLCEngine` integrated cleanly. Solving that (Approach B) also gives compatibility with `curl`, the OpenAI Python SDK, and any standard HTTP client with no special setup.

---

## Why the direct `MLCEngine` path was chosen

The `mlc_llm.MLCEngine` Python class is the intended programmatic API from the
MLC-LLM project. It exposes a `chat.completions.create()` method that:

- accepts a full `messages` array (identical shape to the OpenAI API)
- supports both streaming (`stream=True`) and non-streaming (`stream=False`)
- returns typed response objects with `choices[0].message.content` or `choices[0].delta.content`
- handles the GPU lifecycle cleanly via `engine.terminate()`

This means:
- no fragile output parsing
- no subprocess management for chat
- the engine is fully under the FastAPI process's control
- it is easy to mock in unit tests
- conversation state is client-managed (client sends full message history each
  request), which is stateless and scalable

The trade-off: `mlc_llm` must be importable in the FastAPI process. The project
already installs it as part of the build pipeline, so this is acceptable.

---

## What was verified before committing to this path

Before writing any endpoint code, two things were confirmed:

1. `mlc_llm.MLCEngine` could be imported, initialized with a real compiled
   model, used to generate output, and terminated cleanly — all without
   involving the Go CLI.
2. A real model loaded correctly, produced output, and freed GPU memory as
   expected inside the actual container environment.

Once both of those checks passed, there was enough confidence to build out the
full `/chat/*` endpoint group.

---

## Current chat lifecycle and API

The chat path is intentionally kept separate from the build pipeline path.
There are now two distinct interaction modes in this repo:

```
Client
  │
  ├── Build pipeline (Bryan/CLI mode)
  │     POST /build  /quantize  /compile  /run
  │     ↓
  │     go run . <subcommand>   [subprocess, SSE stream]
  │     ↓
  │     Artifacts on disk
  │
  └── Direct chat (MLCEngine mode)
        POST /chat/load          → MLCEngine(...) constructed, held in memory
        GET  /chat/status        → reports {loaded, model, device}
        POST /chat/completions   → engine.chat.completions.create(messages, ...)
        POST /chat/unload        → engine.terminate(), memory freed
```

### Lifecycle rules

- **Explicit load required.** `POST /chat/completions` returns `503` if no engine
  is loaded. There is no auto-load.
- **One model at a time.** Loading a different model while one is already loaded
  returns `409`. Call `/chat/unload` first.
- **Idempotent reload.** Loading the same model/lib/device configuration while
  already loaded is a no-op (returns success without re-initializing).
- **Auto-unload on shutdown.** The FastAPI lifespan hook calls `unload_engine()`
  on server shutdown, so the GPU is always freed cleanly.
- **Conversation state is client-managed.** The server holds no message history.
  Clients send the full `messages` array on every request, exactly as with the
  OpenAI API.

### Request shape (`POST /chat/completions`)

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Hello!"}
  ],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "stream": false
}
```

`stream` defaults to `false`. When `true`, the response is an SSE stream:

```
data: {"delta": "Hello"}

data: {"delta": ", how can I help?"}

data: [DONE]
```

On error during streaming, an `{"error": "<message>"}` event is emitted before
`[DONE]` so the client always sees a clean stream end.

### Non-streaming response shape

```json
{
  "object": "chat.completion",
  "model": "/path/to/loaded/model",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello, how can I help?"},
      "finish_reason": "stop"
    }
  ]
}
```

---

## Known limitations

**Single-model, single-user only.**
Only one `MLCEngine` instance is held at a time. Concurrent chat requests are
not serialized — if two requests hit `generate_completion()` simultaneously they
share the same engine without a guard. This is acceptable for local/dev use but
would need an `asyncio.Lock` around generation before production multi-user use.

**Sync `MLCEngine`, not `AsyncMLCEngine`.**
The implementation uses the synchronous `MLCEngine`. The MLC-LLM project also
provides `AsyncMLCEngine` for true async generation. Switching to it would allow
the event loop to remain unblocked during long generations. Not done yet because
it is not needed for single-user local use.

**No idle-timeout auto-unload.**
A long-running loaded engine holds GPU memory indefinitely. An inactivity timer
could auto-unload after N minutes of no requests. Not implemented.

**Streaming SSE format is simplified.**
The streaming response uses `{"delta": "..."}` per event rather than the full
OpenAI streaming shape (`choices[0].delta.content`, etc.). This is intentional
for the current scope. The key names are chosen to be mechanically easy to
migrate to the full shape later.

**`mlc_llm` must be importable.**
If `mlc_llm` is not installed in the FastAPI process's Python environment,
`POST /chat/load` returns `503`. The build pipeline (`POST /build`) produces
the wheels, but they must be installed before the chat path works.

---

## What was deliberately kept unchanged

- The Go CLI / Bryan pipeline (`/build`, `/quantize`, `/compile`, `/run`) is
  entirely intact. This repo's original purpose has not changed.
- `POST /run` remains as a runtime verification / load-test endpoint. Its
  limitation (interactive REPL → immediate EOF exit) is intentional and documented.
- No `/v1/chat/completions` alias has been added. The endpoint lives at
  `/chat/completions` to keep it clearly scoped.
- No usage/token counting, no full OpenAI response schema, no session management.
  The chat path exists to make local chat practical, not to be a general-purpose
  OpenAI-compatible server.
