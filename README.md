<h1 align="center">
  🚀 MLC-CLI Build Service
</h1>

<p align="center">
  <strong>A FastAPI wrapper around a baked, pinned <a href="https://github.com/ballinyouup/mlc-cli/">mlc-cli</a><br>
  runtime for repeatable MLC build, quantize, compile, run/load-test, artifact discovery, and direct local chat.</strong>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-5_min-blue?style=for-the-badge" alt="Quick Start"></a>
  <a href="#-demo--test-results"><img src="https://img.shields.io/badge/Demos-3_demos-green?style=for-the-badge" alt="Demos"></a>
  <a href="#-test-strategy"><img src="https://img.shields.io/badge/Tests-Layered-brightgreen?style=for-the-badge" alt="Tests"></a>
  <a href="#-api-overview"><img src="https://img.shields.io/badge/API-SSE_Streaming-blueviolet?style=for-the-badge" alt="API"></a>
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

## 🎯 What This Repository Does

This repository provides a **FastAPI service** that wraps the upstream Go project [`mlc-cli`](https://github.com/ballinyouup/mlc-cli/).

The upstream `mlc-cli` project handles the actual MLC build and model workflow. This repository adds a service layer around it so the workflow is easier to run, inspect, and reuse from an API.

It provides:

- **REST endpoints** for setup checks, build, quantize, compile, run/load-test, artifact discovery, and chat
- **Server-Sent Events (SSE)** so long-running operations stream progress live
- **Baked source pinning** so the Docker image uses an approved `mlc-cli` source revision instead of pulling random runtime changes
- **A writable runtime workspace** for models, converted artifacts, compiled libraries, wheels, TVM, and MLC-LLM outputs
- **Direct local chat** via the MLC-LLM Python engine, once a model has been built and compiled

In short:

> [`mlc-cli`](https://github.com/ballinyouup/mlc-cli/) is the upstream build tool.  
> This repository is the API + runtime layer around that tool, plus a practical local chat path on top of compiled MLC artifacts.

---

## 🤔 Why This Project Exists

Using an upstream build tool directly is convenient, but it creates a few practical problems:

| Problem                                                 | Why it matters                                                                                                            | What this repository adds                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Upstream changes can break your workflow**            | A newer commit in [`mlc-cli`](https://github.com/ballinyouup/mlc-cli/) may change flags, scripts, or behavior             | Bake an approved `mlc-cli` revision into the Docker image                 |
| **Runtime source drift is risky**                       | Runtime clone/pull/fetch can silently change the tool being used                                                          | Use a pinned source at image build time instead of repairing at runtime   |
| **Generated artifacts should survive restarts**         | Models, wheels, compiled libraries, TVM, and MLC-LLM outputs are expensive to recreate                                    | Keep generated artifacts in a writable Docker workspace                   |
| **Long-running steps are hard to observe**              | Builds and model steps can take a while                                                                                   | Stream progress live through SSE                                          |
| **A wrapper API needs stronger operational guardrails** | A plain wrapper is easy to break silently                                                                                 | Add status checks, runtime checks, tests, and clear API workflows         |

This makes the service more predictable for repeated local use, demos, and future maintenance.

---

## 🏗️ Architecture Overview

<p align="center">
  <em>📊 Placeholder: static architecture diagram showing this repo, the upstream <code>mlc-cli</code> repo, baked source pinning, runtime workspace sync, and build outputs.</em><br>
  <img src="assets/architecture-placeholder.svg" alt="Architecture Diagram" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

**High-level flow:**

```text
This repo (FastAPI service)
        ↓
Docker image with baked, pinned mlc-cli source
        ↓
Writable runtime workspace
        ↓
Build / quantize / compile / run / chat
        ↓
Artifacts + streamed logs
```

**Runtime layout:**

```text
/opt/mlc-cli
  baked mlc-cli source inside the Docker image

/workspace/mlc-cli
  writable runtime workspace used by the API and CLI
```

At container startup, the baked source is copied into the writable workspace. Generated artifacts such as models, compiled libraries, wheels, TVM, and MLC-LLM outputs are preserved.

---

## 🔐 Baked Runtime Safety Model

The upstream project is [`mlc-cli`](https://github.com/ballinyouup/mlc-cli/).  
This repository does **not** blindly follow the latest upstream `HEAD` at runtime.

Instead, the Docker image is built from a pinned `mlc-cli` source revision.

### 1️⃣ Pinned source revision

The active `mlc-cli` source/ref pin is stored in:

```text
docker/mlc-cli.lock
```

During Docker build, that pinned source is cloned into:

```text
/opt/mlc-cli
```

That baked source becomes the approved baseline for the image.

### 2️⃣ Writable runtime workspace

At container startup, the entrypoint syncs the baked source into:

```text
/workspace/mlc-cli
```

This workspace is where the API runs the build, quantize, compile, run, and chat workflows.

The sync preserves generated artifact directories:

```text
models/
dist/
wheels/
mlc-llm/
tvm/
```

That means source files can be refreshed from the baked image, while expensive runtime outputs are kept.

### 3️⃣ Updating the upstream pin

To evaluate a newer upstream `mlc-cli` revision:

1. Update `docker/mlc-cli.lock`
2. Rebuild the Docker image
3. Run `/repo-status`
4. Run `/setup-check`
5. Run the relevant build/chat workflow manually

A passing lightweight test is useful, but it is not a replacement for real Docker/GPU validation when changing the upstream build tool.

### Typical flows

If you are new to the project, these are the main flows to remember:

1. **Normal use** — start the service, run setup checks, then use the API endpoints.
2. **Build workflow** — use `/build`, `/quantize`, `/compile`, and `/run` to prepare and test artifacts.
3. **Chat workflow** — use `/chat/load`, `/chat/completions`, and `/chat/unload` after a model has been compiled.
4. **Upstream update** — update `docker/mlc-cli.lock`, rebuild the image, and validate the new runtime.

---

## 🔧 Quick Start

### Prerequisites

| Requirement                  | Notes                                                                   |
| ---------------------------- | ----------------------------------------------------------------------- |
| **Docker + Docker Compose**  | v2.x or later                                                           |
| **NVIDIA GPU + drivers**     | optional for some checks, required for GPU-backed build/inference flows |
| **NVIDIA Container Toolkit** | required for GPU passthrough inside Docker                              |

### Launch the service

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

### Try the basic flow

```bash
curl http://localhost:8000/health
curl http://localhost:8000/repo-status
curl http://localhost:8000/setup-check
```

If `setup-check` says `mlc_llm` or `tvm` is not importable yet, install the built wheels into the runtime environment:

```bash
curl -N -X POST http://localhost:8000/build \
  -H 'Content-Type: application/json' \
  -d '{"action":"install-wheels"}'
```

---

## ⚙️ Environment Variables

| Variable       | Default   | Description                                |
| -------------- | --------- | ------------------------------------------ |
| `BUILD_ACTION` | `full`    | `full` \| `build-only` \| `install-wheels` |
| `CUDA_ARCH`    | `86`      | CUDA compute capability                    |
| `TVM_SOURCE`   | `bundled` | `bundled` \| `relax` \| `custom`           |
| `BUILD_WHEELS` | `y`       | Build Python wheels (`y`/`n`)              |
| `MLC_DEVICE`   | `cuda`    | Target device for MLC inference            |

---

## 📡 API Overview

This service exposes REST endpoints for:

- environment checks
- baked runtime status
- artifact discovery
- build pipeline steps
- direct local chat

The long-running pipeline endpoints stream output with **Server-Sent Events (SSE)**.

### Utility endpoints

| Method | Endpoint       | Purpose                                                                                         |
| ------ | -------------- | ----------------------------------------------------------------------------------------------- |
| `GET`  | `/health`      | Service health check                                                                            |
| `GET`  | `/setup-check` | Verify environment readiness                                                                    |
| `GET`  | `/repo-status` | Show baked source, runtime workspace, pinned ref, workspace match status, and artifact dirs     |
| `GET`  | `/artifacts`   | Discover built wheels, converted models, and compiled libraries                                 |

### Build pipeline endpoints

| Method | Endpoint    | Purpose                             | Output     |
| ------ | ----------- | ----------------------------------- | ---------- |
| `POST` | `/build`    | Compile or install TVM + MLC artifacts | SSE stream |
| `POST` | `/quantize` | Convert model weights to MLC format | SSE stream |
| `POST` | `/compile`  | Compile the model library           | SSE stream |
| `POST` | `/run`      | Load-test model initialization      | SSE stream |

> **Note on `/run`:** This endpoint spawns the upstream `mlc-cli run` flow. It is useful for verifying that a compiled model loads cleanly on the target hardware. It is **not** the chat API.
>
> **Optional `quant` field for compiled-library auto-resolution.** If you omit
> `model_lib` but provide `quant` (e.g. `"q4f16_1"`), the wrapper looks for a
> matching compiled library under `dist/libs/<model_name>-<quant>-<device>.so`
> inside the runtime `mlc-cli` workspace. If exactly one match is found it is passed
> as `--model-lib` automatically. If no match is found the request proceeds without
> `--model-lib` (JIT fallback). If multiple matches are found the request fails
> with an error — pass `model_lib` explicitly to disambiguate.

### Chat endpoints

Once a model has been built and compiled, you can chat with it directly without
going through the Go CLI. The chat path uses the MLC-LLM Python engine directly.

| Method | Endpoint              | Purpose                                                    |
| ------ | --------------------- | ---------------------------------------------------------- |
| `POST` | `/chat/load`          | Load a compiled model into GPU memory                      |
| `GET`  | `/chat/status`        | Check whether an engine is loaded and ready                |
| `POST` | `/chat/completions`   | Send a message array, receive a reply (streaming or not)   |
| `POST` | `/chat/unload`        | Free the engine and release GPU memory                     |

The engine must be explicitly loaded before sending completions. The server
holds no conversation history — clients send the full `messages` array each
request (same pattern as the OpenAI API).

`/chat/load` accepts both absolute container paths and workspace-relative paths such as:

```text
dist/ModelName-q4f16_1-MLC
dist/libs/ModelName-q4f16_1-MLC-q4f16_1-cuda.so
```

The chat path runs in the same conda runtime environment used for the MLC/TVM Python stack.

See [`docs/architecture/chat-direction.md`](docs/architecture/chat-direction.md)
for the full rationale and known limitations.

For full request/response schemas, use the OpenAPI docs when the service is running:

```text
http://localhost:8000/docs
```

---

## 🔄 Runtime Verification Workflow

Use this workflow to check whether the local service, baked source, runtime workspace, and Python runtime are ready.

```bash
curl -s http://localhost:8000/repo-status | python3 -m json.tool
curl -s http://localhost:8000/setup-check | python3 -m json.tool
```

Important fields to look for:

```text
source_management: baked-image
baked_mlc_cli_path: /opt/mlc-cli
mlc_cli_path: /workspace/mlc-cli
workspace_matches_baked: true
mlc_llm_importable: true
tvm_importable: true
```

If `mlc_llm_importable` or `tvm_importable` is `false`, run:

```bash
curl -N -X POST http://localhost:8000/build \
  -H 'Content-Type: application/json' \
  -d '{"action":"install-wheels"}'
```

GPU-backed Docker and end-to-end model validation are expected to be run manually/local because they require CUDA, NVIDIA Container Toolkit, and model artifacts.

---

## 🔧 Updating the Pinned mlc-cli Runtime

This is the workflow for evaluating and adopting a newer `mlc-cli` source revision.

1. Edit `docker/mlc-cli.lock`
2. Rebuild the Docker image
3. Run `/repo-status`
4. Run `/setup-check`
5. Run the relevant build, quantize, compile, run, and/or chat flow

The service does not repair or re-align the `mlc-cli` source from the network at runtime. Runtime source is expected to come from the baked Docker image.

This keeps the running container predictable: changing the tool version is an explicit image-build decision, not a hidden runtime side effect.

---

## 🧪 Test Strategy

The project uses different test layers for different goals.

| Layer                             | Main purpose                                                    |
| --------------------------------- | --------------------------------------------------------------- |
| **Unit tests**                    | Validate service logic quickly and locally                      |
| **Integration tests**             | Validate API lifecycle behavior with mocks                      |
| **Architecture contract tests**   | Guard baked runtime assumptions such as lock file, entrypoint sync, and artifact preservation |
| **Manual Docker/GPU tests**       | Validate real CUDA, image, runtime, and model behavior          |

### Running tests locally

```bash
pip install -r requirements.txt
pytest tests/unit/ -v
pytest tests/unit/ tests/integration/ -v
pytest tests/unit/ tests/integration/ -v --cov=app --cov-report=term-missing
```

### Repository test summary

```text
=========================== Repository Test Summary ===========================
unit tests                local service logic                 ✅
integration tests         API lifecycle behavior              ✅
architecture contracts    baked runtime assumptions           ✅
manual Docker/GPU         real CUDA/runtime validation        manual/local
-------------------------------------------------------------------------------
```

### CI

This repository runs GitHub Actions on push and pull request.

The fast CI path:

- runs the Python test suite
- reports coverage
- provides fast feedback for service-level code changes

GPU-backed and Docker-backed checks still require manual/local execution.

---

## 📸 Demo & Test Results

### End-to-end workflow demo

<p align="center">
  <em>🧭 Placeholder: GIF showing startup → repo-status → setup-check → build/install-wheels → chat/load with streamed output.</em><br>
  <img src="assets/e2e-workflow-placeholder.gif" alt="End-to-End Workflow Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Baked runtime update demo

<p align="center">
  <em>📹 Placeholder: GIF showing <code>docker/mlc-cli.lock</code> update, image rebuild, <code>/repo-status</code> validation, and successful runtime check.</em><br>
  <img src="assets/verify-workflow-placeholder.gif" alt="Baked Runtime Update Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Runtime artifact preservation demo

<p align="center">
  <em>🛠️ Placeholder: GIF showing generated artifacts preserved across container restarts while the baked source remains pinned.</em><br>
  <img src="assets/drift-handling-placeholder.gif" alt="Artifact Preservation Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Test results

<p align="center">
  <em>✅ Placeholder: static screenshot or chart showing unit, integration, architecture-contract, and manual Docker/GPU validation results.</em><br>
  <img src="assets/test-results-placeholder.png" alt="Test Results Summary" width="700" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
</p>

---

## 📂 Project Structure

```text
.
├── 📄 README.md                    # This file
│
├── 🔌 app/
│   ├── main.py                     # FastAPI routes & streaming endpoints
│   ├── chat_engine_manager.py      # Direct MLCEngine lifecycle (load/chat/unload)
│   └── helpers.py                  # Command builders, artifact discovery, and tool helpers
│
├── 🧪 tests/
│   ├── unit/                       # Fast mocked tests (no Docker/GPU)
│   └── integration/                # API lifecycle tests with mocks
│
├── 📚 docs/
│   └── architecture/
│       ├── chat-direction.md       # Why and how the direct-engine chat path works
│       └── baked-mlc-cli-runtime-architecture.md # Architecture Note: Build-time Baked `mlc-cli` Runtime
│
├── 🐳 Dockerfile                   # CUDA 12.6 + Go 1.24 + Miniconda + baked mlc-cli source
├── 🐳 docker/
│   ├── entrypoint.sh               # Sync baked source into runtime workspace
│   └── mlc-cli.lock                # Public mlc-cli repo/ref pin
├── 📋 docker-compose.yml           # GPU-enabled service definition
├── 📋 pyproject.toml               # Python project metadata
├── 📋 requirements.txt             # Dependencies (FastAPI, pytest, etc.)
│
├── 🖼️ assets/
│   ├── architecture-placeholder.svg
│   ├── e2e-workflow-placeholder.gif
│   ├── verify-workflow-placeholder.gif
│   ├── drift-handling-placeholder.gif
│   └── test-results-placeholder.png
└── ⚙️ .github/
    └── workflows/
        └── ci.yml                  # GitHub Actions: Python tests on push/PR
```

---

## ⚠️ Limitations

- **GPU-backed flows still need the right local environment.** Some checks can run without a GPU, but full build / inference flows depend on the proper CUDA + container setup.
- **Updating `mlc-cli` requires an image rebuild.** Change `docker/mlc-cli.lock`, rebuild the Docker image, and validate the new runtime.
- **A fresh runtime may need wheel installation.** If `mlc_llm` or `tvm` is not importable, run `/build` with `action=install-wheels`.
- **Full Docker/GPU validation is intentionally heavier.** It is slower and more resource-intensive than unit tests or mocked integration tests.
- **This repository depends on the upstream `mlc-cli` project.** If upstream behavior changes in deeper ways, you may need to investigate, update the pin, rebuild, and validate before continuing.
- **The chat path is local/dev oriented.** One model is loaded at a time, and the server does not store conversation history.
- **Model output quality depends on the model and prompt format.** A successful API response only proves that the runtime path works; it does not guarantee good model behavior.

---

## 🤝 Contributing

Contributions are welcome. Before opening a pull request:

1. Add or update tests for changed behavior.
2. Run the relevant unit and integration checks locally.
3. Update documentation when public behavior or workflows change.
4. For Docker/GPU changes, include the manual validation performed.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.