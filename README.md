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
| `GET`  | `/setup-check`        | Inspect repo, Go, and Conda status                          |
| `GET`  | `/repo-status`        | Check if `mlc-cli` repo is clean or has uncommitted changes |
| `POST` | `/ensure-repo-exists` | Clone `mlc-cli` repo into `/workspace/mlc-cli` if absent    |

### Build

#### `GET /repo-status`

Check if the `mlc-cli` repository is clean (no uncommitted changes) or dirty.

**Response:**

```json
{
  "status": "ok",
  "repo_exists": true,
  "is_clean": true,
  "message": "Repository is clean",
  "changes": []
}
```

**Example — check if repo is clean:**

```bash
curl http://localhost:8000/repo-status | jq
```

When dirty, the `changes` array will contain git status lines:

```json
{
  "status": "ok",
  "repo_exists": true,
  "is_clean": false,
  "message": "Repository has uncommitted changes",
  "changes": [" M file1.go", "?? newfile.txt"]
}
```

#### `POST /build`

Triggers `mlc-cli build` non-interactively and **streams stdout/stderr as SSE**.

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

## 🗂️ Project Structure

```
.
├── app/
│   └── main.py          # FastAPI application
├── Dockerfile           # CUDA 12.6 + Go 1.24 + Miniconda image
├── docker-compose.yml   # Service definition with GPU passthrough
├── requirements.txt     # FastAPI, Uvicorn, HTTPX
└── test_pipeline.py     # End-to-end test script for the API
```

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
3. **Full Build**: Starts a complete build with CUDA 86, Cutlass, cuBLAS, and bundled TVM, streaming the output live via Server-Sent Events (SSE).
