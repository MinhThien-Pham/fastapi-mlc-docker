# fastapi-mlc-docker

A Dockerised FastAPI service that **clones, builds, and drives [`mlc-cli`](https://github.com/ballinyouup/mlc-cli)** — a Go CLI tool for compiling MLC-LLM — and streams real-time build output back to the caller via **Server-Sent Events (SSE)**.

## Prerequisites

| Requirement                                                                                                      | Notes                                    |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Docker + Docker Compose                                                                                          | v2.x or later                            |
| NVIDIA GPU + drivers                                                                                             | CUDA 12.6 compatible                     |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Enables GPU passthrough to the container |

## 🚀 Quick Start

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## ⚙️ Environment Variables

These can be overridden at runtime (e.g. `docker compose run -e CUDA_ARCH=89 web`):

| Variable       | Default   | Description                                                                   |
| -------------- | --------- | ----------------------------------------------------------------------------- |
| `BUILD_ACTION` | `full`    | Build action passed to `mlc-cli` (`full` \| `build-only` \| `install-wheels`) |
| `CUDA_ARCH`    | `86`      | CUDA compute capability (e.g. `86` for RTX 30xx, `89` for RTX 40xx)           |
| `TVM_SOURCE`   | `bundled` | TVM source (`bundled` \| `relax` \| `custom`)                                 |
| `BUILD_WHEELS` | `y`       | Whether to build Python wheels (`y`/`n`)                                      |
| `MLC_DEVICE`   | `cuda`    | Target device for MLC inference                                               |

## 📌 API Endpoints

### Utility

| Method | Path                  | Description                                                 |
| ------ | --------------------- | ----------------------------------------------------------- |
| `GET`  | `/`                   | Welcome message                                             |
| `GET`  | `/health`             | Service health check                                        |
| `GET`  | `/setup-check`        | Inspect repo, Go, Conda, nvidia-smi, and nvcc               |
| `GET`  | `/repo-status`        | Check if `mlc-cli` repo is clean or has uncommitted changes |
| `POST` | `/ensure-repo-exists` | Clone `mlc-cli` repo into `/workspace/mlc-cli` if absent    |
| `GET`  | `/artifacts`          | List locally built wheels, converted models, and compiled libs|

### Pipeline

| Method | Path       | Description                                                                   |
| ------ | ---------- | ----------------------------------------------------------------------------- |
| `POST` | `/build`   | Build TVM + MLC from source; stream output as SSE                             |
| `POST` | `/convert` | Convert/quantize raw model weights to MLC format; stream output as SSE        |
| `POST` | `/compile` | Compile model library; stream output as SSE                                   |
| `POST` | `/run`     | Run a single prompt against a model; stream output as SSE                     |

### `GET /setup-check`

Checks five things and returns a structured result:

| Check        | What it verifies                            |
| ------------ | ------------------------------------------- |
| `repo`       | `mlc-cli` repo present at `/workspace/mlc-cli` |
| `go`         | `go version` exits 0                        |
| `conda`      | `conda --version` exits 0                   |
| `nvidia_smi` | `nvidia-smi` can query the GPU              |
| `nvcc`       | `nvcc --version` exits 0                    |

**Response shape:**

```json
{
  "repo_exists": true,
  "status": "ok",
  "checks": {
    "repo":       { "available": true,  "path": "/workspace/mlc-cli", "origin": "https://..." },
    "go":         { "available": true,  "output": "go version go1.24.0 linux/amd64", "returncode": 0 },
    "conda":      { "available": true,  "output": "conda 24.1.2", "returncode": 0 },
    "nvidia_smi": { "available": true,  "output": "NVIDIA GeForce RTX 3090", "returncode": 0 },
    "nvcc":       { "available": true,  "output": "nvcc: NVIDIA (R) Cuda compiler driver, V12.6.0", "returncode": 0 }
  },
  "warnings": []
}
```

`status` is one of `"ok"` | `"warning"` | `"error"`:
- `"ok"` — repo exists, Go and Conda are available.
- `"warning"` — tools are present but the repo hasn't been cloned yet.
- `"error"` — Go or Conda is missing; the build cannot proceed.

GPU tools (`nvidia-smi`, `nvcc`) that are unavailable are listed in `warnings` but do not change `status` to `"error"` on their own.

### `GET /artifacts`

A convenience endpoint to view the outputs of the build, convert, and compile steps without needing to manually inspect the container filesystem.

This endpoint scans the local `mlc-cli` workspace and returns a structured JSON list of:
- **Built wheels** (`.whl`)
- **Converted models** (directories containing `mlc-chat-config.json`)
- **Compiled model libraries** (`.so`, `.dylib`, `.dll`)

**Example Response:**

```json
{
  "status": "ok",
  "root_paths_searched": [
    "/workspace/mlc-cli"
  ],
  "counts": {
    "build": 1,
    "convert": 1,
    "compile": 1,
    "total": 3
  },
  "artifacts": [
    {
      "type": "wheel",
      "name": "mlc_llm-0.1.0-cp310-cp310-linux_x86_64.whl",
      "path": "build/mlc_llm-0.1.0-cp310-cp310-linux_x86_64.whl",
      "source_step": "build",
      "size_bytes": 1420583,
      "modified_time": 1714589212.0
    },
    {
      "type": "model_dir",
      "name": "Llama-3-8B-q4f16_1-MLC",
      "path": "dist/Llama-3-8B-q4f16_1-MLC",
      "source_step": "convert",
      "size_bytes": 4512938491,
      "modified_time": 1714589500.0
    },
    {
      "type": "compiled_lib",
      "name": "Llama-3-8B-q4f16_1-cuda.so",
      "path": "dist/Llama-3-8B-q4f16_1-MLC/Llama-3-8B-q4f16_1-cuda.so",
      "source_step": "compile",
      "size_bytes": 84592,
      "modified_time": 1714589600.0
    }
  ]
}
```

### Build

#### `POST /build`

Triggers `mlc-cli build` non-interactively and **streams stdout/stderr as SSE**.

If a **cutlass / flash-attn failure** is detected in the output, a `[HINT]` line is automatically emitted after the offending line with a ready-to-paste retry command — no searching the logs required.

**Request body (all fields optional):**

```json
{
  "action": "full",
  "tvm_source": "bundled",
  "cuda": "y",
  "cuda_arch": "86",
  "cutlass": "n",
  "cublas": "n",
  "flash_infer": "n",
  "rocm": "n",
  "vulkan": "n",
  "opencl": "n",
  "build_wheels": "y",
  "force_clone": "n"
}
```

**Example — stream a wheel-only install:**

```bash
curl -N -X POST http://localhost:8000/build \
     -H 'Content-Type: application/json' \
     -d '{"action": "install-wheels"}'
```

Each SSE line is prefixed with `data: `. The stream ends with `data: [DONE]` on success or `data: [ERROR] ...` on failure.

**Cutlass / flash-attn failures** — if the stream contains a known error signature, you will automatically see:

```
data: [HINT] This looks like a cutlass / flash-attn build failure.
data: [HINT] Retry with cutlass and flash_infer disabled:
data: [HINT]
data: [HINT]   curl -N -X POST http://localhost:8000/build \
data: [HINT]        -H 'Content-Type: application/json' \
data: [HINT]        -d '{"action":"full","cutlass":"n","flash_infer":"n"}'
```

> **Note:** `cutlass` and `flash_infer` already default to `"n"`. This hint is mainly useful if you explicitly enabled them and hit a build error.

### Convert

#### `POST /convert`

Quantizes (converts) raw model weights to MLC format and **streams stdout/stderr as SSE**.

Internally this calls the mlc-cli `quantize` sub-command, which runs two steps in sequence:

1. `mlc_llm convert_weight` — convert Hugging Face weights to MLC format with the selected quantization.
2. `mlc_llm gen_config`     — write the runtime config alongside the converted weights.

**Pipeline position:** Run `/convert` *after* `/build` (which installs the `mlc-llm` Python package) and *before* `/run`.

**Request body (`model` is required; all other fields are optional):**

```json
{
  "model": "models/Llama-3-8B",
  "quant": "q4f16_1",
  "device": "cuda",
  "conv_template": "llama-3",
  "output": ""
}
```

| Field           | Type   | Default    | Notes                                                                                        |
| --------------- | ------ | ---------- | -------------------------------------------------------------------------------------------- |
| `model`         | string | *(required)* | Path to a local model dir (e.g. `models/Llama-3-8B`) or a Hugging Face hub identifier     |
| `quant`         | string | `q4f16_1`  | Quantization: `q4f16_1`, `q4f16_ft`, `q4f32_1`, `q3f16_1`, `q8f16_1`, `q0f16`, `q0f32`  |
| `device`        | string | `cuda`     | Target device: `cuda`, `metal`, `vulkan`, `opencl`, `rocm`                                  |
| `conv_template` | string | `llama-3`  | Conversation template: `llama-3`, `chatml`, `mistral_default`, `phi-2`, `gemma`, `qwen2`   |
| `output`        | string | `""`       | Output directory. If empty, mlc-cli uses `dist/<model_basename>-<quant>-MLC`               |

**Example — convert a Llama-3 8B model:**

```bash
curl -N -X POST http://localhost:8000/convert \
     -H 'Content-Type: application/json' \
     -d '{"model": "models/Llama-3-8B", "quant": "q4f16_1", "device": "cuda"}'
```

Each SSE line is prefixed with `data: `. The stream ends with `data: [DONE]` on success or `data: [ERROR] ...` on failure.

### Compile

#### `POST /compile`

Compiles a model library and **streams stdout/stderr as SSE**.

Internally this calls the mlc-cli `compile` sub-command.

**Pipeline position:** Run `/compile` *after* `/convert` and *before* `/run`.

**Request body (`model` is required; all other fields are optional):**

```json
{
  "model": "models/Llama-3-8B",
  "quant": "q4f16_1",
  "device": "cuda",
  "output": ""
}
```

| Field    | Type   | Default   | Notes                                                              |
| -------- | ------ | --------- | ------------------------------------------------------------------ |
| `model`  | string | *(req)*   | Path to a local model dir (e.g. `models/Llama-3-8B`)               |
| `quant`  | string | `q4f16_1` | Quantization: `q4f16_1`, `q0f32`, etc.                             |
| `device` | string | `cuda`    | Target device: `cuda`, `metal`, `vulkan`, `opencl`, `rocm`         |
| `output` | string | `""`      | Output directory or file path. If empty, uses default              |

**Example — compile a Llama-3 8B model:**

```bash
curl -N -X POST http://localhost:8000/compile \
     -H 'Content-Type: application/json' \
     -d '{"model": "models/Llama-3-8B", "quant": "q4f16_1", "device": "cuda"}'
```

Each SSE line is prefixed with `data: `. The stream ends with `data: [DONE]` on success or `data: [ERROR] ...` on failure.

### Run

#### `POST /run`

Load-tests a model by initializing the interactive REPL and **streams stdout/stderr as SSE**.

Internally this calls the mlc-cli `run` sub-command.

**LIMITATION**: The upstream `mlc-cli run` command is interactive by default and does NOT support a non-interactive single-shot `--prompt` flag. When called via this API endpoint, no standard input is provided. The subprocess will initialize the model, print its ready state, and immediately exit upon encountering EOF. This effectively serves as a "load test" to verify model and compiled library compatibility.

**Pipeline position:** Run `/run` *after* `/compile` (and `/convert`). 

**Request body (`model_name` is required; all other fields are optional):**

```json
{
  "model_name": "Llama-3-8B",
  "model_url": "",
  "device": "cuda",
  "profile": "default",
  "model_lib": ""
}
```

| Field        | Type   | Default   | Notes                                                               |
| ------------ | ------ | --------- | ------------------------------------------------------------------- |
| `model_name` | string | *(req)*   | Model directory name in models/ (e.g. `Llama-3-8B`)                 |
| `model_url`  | string | `""`      | Optional Git URL to clone the model from                            |
| `device`     | string | `cuda`    | Target device: `cuda`, `metal`, `vulkan`, `opencl`, `rocm`          |
| `profile`    | string | `default` | Compute profile: `really-low`, `low`, `default`, `high`             |
| `model_lib`  | string | `""`      | Optional path to compiled `.so` library. If empty, uses default     |

**Example — load test a model:**

```bash
curl -N -X POST http://localhost:8000/run \
     -H 'Content-Type: application/json' \
     -d '{"model_name": "Llama-3-8B", "device": "cuda", "profile": "default"}'
```

Each SSE line is prefixed with `data: `. The stream ends with `data: [DONE]` on success or `data: [ERROR] ...` on failure.

## 🗂️ Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI application (routes)
│   └── helpers.py       # Pure helper functions (testable without FastAPI)
├── tests/
│   ├── conftest.py      # Shared pytest fixtures
│   ├── test_health.py   # /health and / endpoint tests
│   ├── test_setup_check.py  # /setup-check tests (all mocked)
│   ├── test_helpers.py  # Unit tests for app/helpers.py
│   ├── test_convert.py  # build_convert_command helper + POST /convert route tests
│   ├── test_compile.py  # build_compile_command helper + POST /compile route tests
│   ├── test_artifacts.py # discover_artifacts helper + GET /artifacts tests
│   └── test_run.py      # build_run_command helper + POST /run route tests
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI (push + PR)
├── Dockerfile           # CUDA 12.6 + Go 1.24 + Miniconda image
├── docker-compose.yml   # Service definition with GPU passthrough
├── requirements.txt     # FastAPI, Uvicorn, HTTPX, pytest, pytest-cov
└── test_pipeline.py     # End-to-end integration script (requires live container)
```

## 🧪 Running Tests Locally

No Docker, GPU, Go, or Conda needed — tests use mocks.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full test suite
pytest tests/ -v

# 3. Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing
```

## 🤖 CI (GitHub Actions)

The workflow in `.github/workflows/ci.yml` runs automatically on every push and pull request:

1. Checks out the code
2. Sets up Python 3.12
3. Installs `requirements.txt`
4. Runs `pytest tests/ -v --cov=app --cov-report=term-missing`

No GPU or Docker is required in CI — all tests use mocks.

## 🐳 Docker Details

- **Base image**: `nvidia/cuda:12.6.3-devel-ubuntu24.04`
- **Go**: 1.24.0 (installed from upstream tarball)
- **Conda**: Miniconda latest (used by `mlc-cli` build scripts)
- **Python venv**: isolated at `/opt/venv` for the FastAPI app
- **Workspace volume**: `mlc_workspace` is mounted at `/workspace` — the `mlc-cli` repo and build artefacts persist across container restarts

## 🧪 `test_pipeline.py` — End-to-End API Test

`test_pipeline.py` is an automated test script that interacts with the containerised API to verify the full `mlc-cli` build lifecycle:

```bash
# Requires httpx
pip install httpx
python test_pipeline.py
```

It performs the following steps sequentially against the API (`http://localhost:8000`):

1. **Setup Check**: Verifies the repository exists and forces a clone if missing.
2. **Repo Status**: Calls `/repo-status` to check if the repository is clean or has uncommitted changes.
3. **Full Build**: Starts a complete build with CUDA 86, cuBLAS, and bundled TVM, streaming the output live via Server-Sent Events (SSE).
