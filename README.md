<h1 align="center">
  🚀 MLC-CLI Build Service
</h1>

<p align="center">
  <strong>A FastAPI service around a pinned <a href="https://github.com/ballinyouup/mlc-cli/">mlc-cli</a> runtime<br>
  for repeatable MLC model builds, quantization, compilation, and local chat.</strong>
</p>

<p align="center">
  <a href="docs/API_ENDPOINTS.md"><img src="https://img.shields.io/badge/API_Reference-Endpoints-blue?style=for-the-badge" alt="API Reference"></a>
  <a href="docs/DEVELOPMENT.md"><img src="https://img.shields.io/badge/Dev_Guide-Workflows-green?style=for-the-badge" alt="Dev Guide"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-blue?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go&logoColor=white" alt="Go">
</p>

---

## ✨ What this project does

This repository wraps the upstream Go project [`mlc-cli`](https://github.com/ballinyouup/mlc-cli/)
in a FastAPI service.

`mlc-cli` handles the MLC-LLM build pipeline — compiling TVM, converting model weights,
and producing compiled model libraries. This service adds:

- **REST endpoints** for build, quantize, compile, run/load-test, artifact discovery, and chat
- **Server-Sent Events (SSE)** so long-running steps stream progress live to your terminal
- **A pinned mlc-cli source** packaged into the Docker image at a verified commit, so builds are reproducible
- **A writable runtime workspace** that preserves generated artifacts across container restarts
- **Direct local chat** via the MLC-LLM Python engine, once a model has been built and compiled

---

## 🧠 Why it exists

Using the upstream tool directly has a few practical problems:

| Problem | What this project adds |
|---|---|
| Upstream changes can silently break your workflow | Pins a verified mlc-cli commit into the Docker image |
| Long-running steps are hard to observe | Streams progress live via SSE |
| Generated artifacts should survive restarts | Persists models, wheels, compiled libraries in a writable workspace |
| A wrapper API needs guardrails | Adds status checks, typed schemas, tests, and clear API workflows |

---

## 🏗️ Architecture

```
docker/mlc-cli.lock        ← pinned mlc-cli repo and commit SHA
       ↓
docker build               ← mlc-cli source packaged into the image
       ↓
/opt/mlc-cli               ← read-only pinned source inside the container
       ↓
docker/entrypoint.sh       ← syncs /opt/mlc-cli into /workspace/mlc-cli on startup
       ↓
/workspace/mlc-cli         ← writable runtime workspace (artifacts preserved here)
       ↓
FastAPI service            ← build / quantize / compile / run / chat
```

**Key paths inside the container:**

| Path | Purpose |
|---|---|
| `/opt/mlc-cli` | Pinned mlc-cli source included in the Docker image |
| `/workspace/mlc-cli` | Writable runtime workspace used by the API |
| `/workspace/mlc-cli/models/` | Raw downloaded model weights |
| `/workspace/mlc-cli/dist/` | Quantized models and compiled libraries |
| `/workspace/mlc-cli/wheels/` | Built Python wheels |

Updating the mlc-cli version means editing `docker/mlc-cli.lock` and rebuilding the image.
The container does not fetch or pull mlc-cli at runtime.

---

## ⚡ Quick setup

> This section covers the **one-time initial setup**.
> Day-to-day usage is in the [Regular usage](#-regular-usage) section below.

**Prerequisites:** Docker + Docker Compose v2.x, NVIDIA GPU + drivers, NVIDIA Container Toolkit.

**1. Build and start the service:**

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`.

**2. Verify the environment:**

```bash
curl http://localhost:8000/health
curl -s http://localhost:8000/repo-status   | python3 -m json.tool
curl -s http://localhost:8000/setup-check   | python3 -m json.tool
```

**3. Install Python wheels (if needed):**

Check the `setup-check` response. If `mlc_llm_importable` or `tvm_importable` is `false`,
install the built wheels into the runtime environment:

```bash
curl -N -X POST http://localhost:8000/build \
  -H 'Content-Type: application/json' \
  -d '{"action": "install-wheels"}'
```

You only need to do this once, or after a rebuild.

---

## 🎥 Demo / Showcase

A short demo can show:

- Starting the Docker service
- Checking runtime status
- Installing wheels if needed
- Preparing a model (quantize + compile)
- Loading a compiled model
- Sending a chat request

Demo video: **coming soon**

---

## 📦 Get your first model

> This step is done once per model — not every time you start the service.

The easiest way to get started is by passing a Hugging Face model ID directly to the API.

**1. Find a model ID:**

Go to [Hugging Face Models (Text Generation)](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) and find a model you want to try.

Copy the model ID from the page title or URL path.
- **URL:** `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Model ID:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**2. Quantize and compile it:**

Pass the model ID directly to `/quantize` (see Regular usage below). The API will handle downloading the weights automatically.

> **Note on private models:** For gated or private models on Hugging Face, ensure your container has the necessary Hugging Face access configured before running `/quantize`.

*(For advanced local folder setups, see [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md)).*

---

## 🔁 Regular usage

> After setup and model preparation, your day-to-day workflow looks like this.

**Start the service:**

```bash
docker compose up
```

**Check status:**

```bash
curl http://localhost:8000/health
curl -s http://localhost:8000/setup-check | python3 -m json.tool
```

**See what artifacts are available:**

```bash
curl -s http://localhost:8000/artifacts | python3 -m json.tool
```

**If you have model weights and need to prepare them (once per model):**

```bash
# Quantize from Hugging Face → dist/<ModelName>-<quant>-MLC/
# conv_template defaults to "auto" — the API infers it from the model name.
curl -N -X POST http://localhost:8000/quantize \
  -H 'Content-Type: application/json' \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

# Compile the model library → dist/libs/<ModelName>-<quant>-cuda.so
curl -N -X POST http://localhost:8000/compile \
  -H 'Content-Type: application/json' \
  -d '{"model": "dist/<ModelName>-q4f16_1-MLC", "quant": "q4f16_1"}'
```

**Load a compiled model, chat, then unload:**

```bash
# Load
curl -X POST http://localhost:8000/chat/load \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TinyLlama-1.1B-Chat-v1.0"
  }'

# Chat
curl -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Unload when done
curl -X POST http://localhost:8000/chat/unload
```

See [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md) for all request fields, full schemas, and error codes.

---

## ⚡ Quick API demo

**With TinyLlama (replace paths with your actual artifact names from `/artifacts`):**

```bash
# Load
curl -X POST http://localhost:8000/chat/load \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TinyLlama-1.1B-Chat-v1.0"
  }'

# Chat
curl -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is 2 + 2?"}]}'

# Stream chat
curl -N -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Tell me a short joke."}], "stream": true}'

# Unload
curl -X POST http://localhost:8000/chat/unload
```

For full build/quantize/compile/run examples, see [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md).

---

## 🧪 Testing

```bash
# Fast local tests — no Docker or GPU required
pytest tests/unit/ tests/integration/ -q

# Smoke test against a running container
API_URL=http://localhost:8000 python tests/integration/test_smoke.py

# Full pipeline test against a running container
API_URL=http://localhost:8000 python tests/integration/test_full_pipeline.py
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full test strategy, CI workflows,
and the candidate mlc-cli verification flow.

---

## ⚠️ Limitations

- **GPU-backed flows require the full CUDA + container setup.**
  Some checks run without a GPU; build and inference require it.

- **Updating mlc-cli requires an image rebuild.**
  Edit `docker/mlc-cli.lock`, rebuild, and validate the new runtime.

- **A fresh container may need wheel installation.**
  If `mlc_llm` or `tvm` is not importable, run `/build` with `action=install-wheels`.

- **One model at a time.**
  The chat path loads one model into GPU memory.
  Call `/chat/unload` before loading a different model.

- **No server-side conversation history.**
  Clients send the full message array with each request.

- **Model output quality depends on the model and prompt format.**
  A successful API response proves the runtime works, not that the model behaves well.

---

## 📚 More documentation

| Document | Contents |
|---|---|
| [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md) | Full endpoint reference with request schemas, curl examples, and response examples |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local setup, test strategy, candidate verification, CI workflows, contribution checklist |
| [docs/architecture/baked-mlc-cli-runtime-architecture.md](docs/architecture/baked-mlc-cli-runtime-architecture.md) | Architecture notes on the pinned runtime design |
| [docs/architecture/chat-direction.md](docs/architecture/chat-direction.md) | Why the direct MLCEngine path was chosen for chat |

---

## License

MIT License — see [LICENSE](LICENSE) for details.