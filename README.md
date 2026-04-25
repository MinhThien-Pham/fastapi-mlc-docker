<h1 align="center">
  🚀 MLC-CLI Build Service
</h1>

<p align="center">
  <strong>A FastAPI wrapper around <a href="https://github.com/ballinyouup/mlc-cli/">ballinyouup/mlc-cli</a><br>
  for repeatable builds, pinned-upstream safety, and real-time streaming output.</strong>
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
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go&logoColor=white" alt="Go">
</p>

---

## 🎯 What This Repository Does

This repository provides a **FastAPI service** that wraps the upstream Go project [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/).

The upstream `mlc-cli` project handles the actual MLC build and model workflow. This repository adds the service layer around it:

- **REST endpoints** for setup, build, quantize, compile, run, and artifact discovery
- **Server-Sent Events (SSE)** so long-running operations stream progress live
- **Pinned upstream management** so the service does not silently drift to an unverified upstream commit
- **Verification and promotion tooling** before accepting a newer upstream version
- **Repair / re-alignment utilities** when the local upstream checkout drifts from the approved state

In short:

> `ballinyouup/mlc-cli` is the upstream tool.  
> This repository is the API + safety layer around that tool.

---

## 🤔 Why This Project Exists

Using an upstream build tool directly is convenient, but it creates a few practical problems:

| Problem                                                 | Why it matters                                                                                                            | What this repository adds                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Upstream changes can break your workflow**            | A newer commit in [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/) may change flags, scripts, or behavior | Pin an approved upstream SHA and only move forward after verification     |
| **The local upstream checkout can drift**               | Tracked source files may change locally by accident                                                                       | Detect and restore tracked drift, then re-align when needed               |
| **Long-running steps are hard to observe**              | Builds and model steps can take a while                                                                                   | Stream progress live through SSE                                          |
| **A wrapper API needs stronger operational guardrails** | A plain wrapper is easy to break silently                                                                                 | Add status checks, contract checks, manual verification, and repair flows |

This makes the service more predictable for repeated local use, demos, and future maintenance.

---

## 🏗️ Architecture Overview

<p align="center">
  <em>📊 Placeholder: static architecture diagram showing this repo, the upstream <code>mlc-cli</code> repo, pinned SHA metadata, verification flow, and build outputs.</em><br>
  <img src="assets/architecture-placeholder.svg" alt="Architecture Diagram" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

**High-level flow:**

```text
This repo (FastAPI service)
        ↓
Managed checkout of upstream ballinyouup/mlc-cli
        ↓
Build / quantize / compile / run
        ↓
Artifacts + streamed logs
```

**Safety flow:**

```text
Pinned SHA  →  verify candidate  →  promote if verified  →  repair back to pin if needed
```

---

## 🔐 Upstream Safety Model

The upstream project is [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/).  
This repository does **not** blindly follow its latest `HEAD`. Instead, it uses a **pinned + verify + promote** model.

### 1️⃣ Pinned SHA (approved baseline)

A known-good upstream commit is stored in `.upstream-sha.json`:

```json
{
  "repo": "https://github.com/ballinyouup/mlc-cli.git",
  "pinned_sha": "abc1234567...",
  "pinned_date": "2026-04-24T09:00:00-04:00"
}
```

That pinned SHA is the baseline used for:

- repair / re-alignment
- startup status checks
- deciding whether a newer upstream commit still needs verification

### 2️⃣ Lightweight contract check

This repository includes a lightweight upstream contract check to catch obvious interface drift early.

It runs in the upstream drift workflow:

- **automatically every Monday at 14:00 UTC**  
  (**10:00 AM EDT / 9:00 AM EST**)
- **manually on demand** through GitHub Actions workflow dispatch

Its job is to answer questions like:

- do the expected CLI flags still exist?
- do required scripts still exist?
- does the upstream surface still look compatible enough to keep evaluating?

If the weekly drift workflow detects that the pinned SHA is behind the latest upstream `HEAD`, it runs this contract check against the newer candidate.

Possible outcomes:

- **contract still looks compatible** → the workflow records a summary, but does **not** auto-promote anything
- **contract check fails** → the workflow opens an issue for investigation
- **result is inconclusive** → manual review is still required

This check is useful, but it is **not enough by itself**. A passing contract check does **not** prove that the full build pipeline still works.

### 3️⃣ Manual verification and promotion

When a newer commit from [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/) is being evaluated, run:

```bash
python verify_upstream.py
```

That flow performs:

- a **smoke integration check**
- a **full integration check**
- promotion of the new SHA **only if the candidate passes**

If the candidate fails, the pin remains unchanged.

### Why multiple verification layers exist

Different checks answer different questions. That is why manual verification still matters.

| Verification layer     | Rough confidence | What it tells us                                                                      |
| ---------------------- | ---------------: | ------------------------------------------------------------------------------------- |
| **CLI contract check** |             ~50% | The upstream CLI still looks compatible at the surface/interface level                |
| **Smoke integration**  |             ~70% | The basic service → upstream → result path still works                                |
| **Full integration**   |             ~95% | The full build → quantize → compile → run path still works on the evaluated candidate |

**Notes**

- These percentages are rough, subjective estimates for comparison only.
- They are not formal measurements.
- The point of the table is to show why a lightweight check is helpful, but still not enough to replace manual verification.

### Typical flows

If you are new to the project, these are the three flows to remember:

1. **Normal use** — start the service, run setup checks, then use the API endpoints.
2. **Upstream update** — run `python verify_upstream.py`, review the result, and only promote when it passes.
3. **Repair / re-alignment** — call `/ensure-repo-exists` when the managed upstream checkout drifts away from the approved state.

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
curl http://localhost:8000/setup-check
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
- managed upstream repair / alignment
- artifact discovery
- build pipeline steps

The long-running pipeline endpoints stream output with **Server-Sent Events (SSE)**.

### Utility endpoints

| Method | Endpoint              | Purpose                                                                                                   |
| ------ | --------------------- | --------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`             | Service health check                                                                                      |
| `GET`  | `/setup-check`        | Verify environment readiness                                                                              |
| `POST` | `/ensure-repo-exists` | Create or repair the managed checkout of [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/) |
| `GET`  | `/repo-status`        | Show alignment / dirty-state status for the managed upstream checkout                                     |
| `GET`  | `/artifacts`          | Discover built wheels, converted models, and compiled libraries                                           |

### Build pipeline endpoints

| Method | Endpoint    | Purpose                             | Output     |
| ------ | ----------- | ----------------------------------- | ---------- |
| `POST` | `/build`    | Compile TVM + MLC from source       | SSE stream |
| `POST` | `/quantize` | Convert model weights to MLC format | SSE stream |
| `POST` | `/compile`  | Compile the model library           | SSE stream |
| `POST` | `/run`      | Load-test model initialization      | SSE stream |

For full request/response schemas, use the OpenAPI docs when the service is running:

```text
http://localhost:8000/docs
```

---

## 🔄 Verification Workflow

This is the operator workflow for evaluating a newer upstream commit from [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/).

```bash
python verify_upstream.py
git log -1
python verify_upstream.py --push
```

Under the hood:

1. **Preflight** — confirm the managed upstream checkout is safe to touch
2. **Smoke integration** — catch quick failures early
3. **Full integration** — run the complete pipeline
4. **Promotion** — update the pin only after success

> See the demo section below for the verification GIF placeholder.

---

## 🔧 Repair & Alignment

This is the fallback path when the managed checkout of [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/) is no longer in the expected state.

### Tracked file protection

If tracked upstream source files were modified locally:

```bash
POST /ensure-repo-exists
```

The service restores tracked source changes while preserving untracked artifacts such as caches and build outputs.

### Alignment to the pinned SHA

If the local checkout exists but is not on the approved pinned commit:

```bash
POST /ensure-repo-exists
```

The service re-aligns that checkout back to the pinned SHA.

---

## 🧪 Test Strategy

The project uses different test layers for different goals.

| Layer                  | Main purpose                                  |
| ---------------------- | --------------------------------------------- |
| **Unit tests**         | Validate service logic quickly and locally    |
| **CLI contract check** | Catch obvious upstream interface drift        |
| **Smoke integration**  | Verify the basic service → upstream path      |
| **Full integration**   | Verify the full end-to-end candidate workflow |

### Running tests locally

```bash
pip install -r requirements.txt
pytest tests/unit/ -v
pytest tests/unit/ -v --cov=app --cov-report=term-missing

docker compose up -d
python tests/integration/test_smoke.py
python tests/integration/test_full_pipeline.py
```

### Repository test summary

```text
=========================== Repository Test Summary ===========================
unit tests                local service logic         ✅
smoke integration         basic API sanity            ✅
full integration          end-to-end pipeline         ✅
contract check            upstream interface gate     ✅
-------------------------------------------------------------------------------
promotion gate            smoke + full required       ✅
```

### CI

This repository currently uses two GitHub Actions workflows:

1. **Push / PR CI**
   - runs the unit test suite
   - reports coverage
   - provides fast feedback for service-level code changes

2. **Weekly upstream drift workflow**
   - runs every Monday at **14:00 UTC**  
     (**10:00 AM EDT / 9:00 AM EST**)
   - compares the pinned upstream SHA against the latest `HEAD` of [`ballinyouup/mlc-cli`](https://github.com/ballinyouup/mlc-cli/)
   - runs the lightweight contract check when drift is detected
   - records a summary on contract-check success
   - opens an issue on contract-check failure

GPU-backed and Docker-backed checks still require manual/local execution.

---

## 📸 Demo & Test Results

### End-to-end workflow demo

<p align="center">
  <em>🧭 Placeholder: GIF showing startup → setup-check → ensure-repo-exists → successful build flow with streamed output.</em><br>
  <img src="assets/e2e-workflow-placeholder.gif" alt="End-to-End Workflow Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Upstream verification demo

<p align="center">
  <em>📹 Placeholder: GIF showing <code>verify_upstream.py</code> from preflight through smoke/full verification and successful promotion.</em><br>
  <img src="assets/verify-workflow-placeholder.gif" alt="Verification Workflow Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Drift / incident handling demo

<p align="center">
  <em>🛠️ Placeholder: GIF showing intentional upstream breakage or drift, followed by failure handling / repair behavior.</em><br>
  <img src="assets/drift-handling-placeholder.gif" alt="Drift Handling Demo" width="700" style="max-width: 100%; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
</p>

### Test results

<p align="center">
  <em>✅ Placeholder: static screenshot or chart showing unit, smoke, full, and contract-check results.</em><br>
  <img src="assets/test-results-placeholder.png" alt="Test Results Summary" width="700" style="max-width: 100%; border: 1px solid #ccc; padding: 10px;">
</p>

---

## 📂 Project Structure

```
.
├── 📄 README.md                    # This file
├── 📋 .upstream-sha.json           # Pinned upstream commit metadata
├── 📋 verify_upstream.py           # Manual verification + promotion script
│
├── 🔌 app/
│   ├── main.py                     # FastAPI routes & streaming endpoints
│   └── helpers.py                  # Repo alignment and dirty-state helpers
│
├── 🧪 tests/
│   ├── unit/                       # Fast mocked tests (no Docker/GPU)
│   ├── integration/                # Smoke and full pipeline tests
│   └── upstream/                   # CLI contract check helper
│
├── 🐳 Dockerfile                   # CUDA 12.6 + Go 1.24 + Miniconda
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
        └── ci.yml                  # GitHub Actions: unit tests on push/PR
        └── upstream-drift.yml      # Weekly upstream drift / contract-check workflow
```

---

## ⚠️ Limitations

- **GPU-backed flows still need the right local environment.** Some checks can run without a GPU, but full build / inference flows depend on the proper CUDA + container setup.
- **A passing contract check is not enough by itself.** Manual verification is still needed before promoting a new upstream SHA.
- **Full integration is intentionally heavier.** It is slower and more resource-intensive than unit tests or lightweight checks.
- **This repository depends on the upstream `mlc-cli` project.** If upstream behavior changes in deeper ways, you may need to investigate, verify, and re-pin before continuing.

---

## 🤝 Contributing

Contributions are welcome. Before opening a pull request:

1. Add or update tests for changed behavior.
2. Run the relevant unit and integration checks locally.
3. Update documentation when public behavior or workflows change.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
