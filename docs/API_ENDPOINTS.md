# API Endpoint Reference

This document covers every public endpoint in the FastAPI MLC-CLI service.
For setup and dev workflows, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Streaming endpoints

Several endpoints stream output as **Server-Sent Events (SSE)**.
Each event is a line of the form:

```
data: <text>
```

The stream ends with `data: [DONE]` on success, or `data: [ERROR] ...` on failure.
Use `curl -N` to receive the stream without buffering.

---

## Utility endpoints

### `GET /health`

Basic liveness check. Returns immediately.

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy"}
```

---

### `GET /repo-status`

Shows the pinned mlc-cli source included in the Docker image versus the current runtime workspace state.

```bash
curl -s http://localhost:8000/repo-status | python3 -m json.tool
```

**Example response:**

```json
{
  "source_management": "baked-image",
  "mlc_cli_path": "/workspace/mlc-cli",
  "baked_mlc_cli_path": "/opt/mlc-cli",
  "baked_ref_file": "a846fcea6894e9bff76e0dd663990f06cd1f93eb",
  "baked_repo_file": "https://github.com/ballinyouup/mlc-cli.git",
  "baked_actual_head": "a846fcea6894e9bff76e0dd663990f06cd1f93eb",
  "workspace_head": "a846fcea6894e9bff76e0dd663990f06cd1f93eb",
  "workspace_matches_baked": true,
  "artifact_dirs": {
    "models": false,
    "dist": false,
    "wheels": false,
    "mlc-llm": false,
    "tvm": false
  },
  "dev_mode": false
}
```

**Key fields:**

| Field | Meaning |
|---|---|
| `source_management` | Always `"baked-image"` — mlc-cli is included in the Docker image |
| `workspace_matches_baked` | `true` means the workspace was synced correctly from the included image source |
| `artifact_dirs` | Which output directories exist in the runtime workspace |

---

### `GET /setup-check`

Verifies environment readiness: mlc-cli workspace, Go, Conda, GPU drivers, Python imports.

```bash
curl -s http://localhost:8000/setup-check | python3 -m json.tool
```

**Example response (abbreviated):**

```json
{
  "repo_exists": true,
  "status": "ok",
  "checks": {
    "repo": {"available": true, "path": "/workspace/mlc-cli"},
    "go": {"available": true, "output": "go version go1.24 linux/amd64"},
    "conda": {"available": true, "output": "conda 23.11.0"},
    "nvidia_smi": {"available": true, "output": "NVIDIA A100-SXM4-80GB"},
    "nvcc": {"available": true, "output": "Cuda compilation tools, release 12.6"}
  },
  "warnings": [],
  "wrapper_info": {
    "mlc_cli_path": "/workspace/mlc-cli",
    "mlc_llm_importable": true,
    "tvm_importable": true,
    "artifact_dirs_present": {
      "models": false,
      "dist": true,
      "wheels": true,
      "mlc-llm": true,
      "tvm": true
    }
  }
}
```

**Status values:**

| `status` | Meaning |
|---|---|
| `"ok"` | Workspace and critical tools (Go, Conda) are ready |
| `"warning"` | Tools ready but workspace missing or GPU unavailable |
| `"error"` | Critical tools (Go or Conda) not found |

**Notes:**
- If `mlc_llm_importable` or `tvm_importable` is `false`, run `/build` with `action=install-wheels` first.
- GPU warnings are informational — non-GPU checks still pass.

---

### `GET /artifacts`

Discovers wheels, converted model directories, and compiled libraries in the workspace.
Safe to call at any time; returns an empty list if the workspace is empty.

```bash
curl -s http://localhost:8000/artifacts | python3 -m json.tool
```

**Example response:**

```json
{
  "status": "ok",
  "root_paths_searched": ["/workspace/mlc-cli"],
  "counts": {
    "build": 2,
    "convert": 1,
    "quantize": 1,
    "compile": 1,
    "total": 4
  },
  "artifacts": [
    {
      "type": "wheel",
      "name": "mlc_llm-0.1-cp313-cp313-linux_x86_64.whl",
      "path": "wheels/mlc_llm-0.1-cp313-cp313-linux_x86_64.whl",
      "source_step": "build",
      "size_bytes": 12345678,
      "modified_time": 1700000000.0
    },
    {
      "type": "model_dir",
      "name": "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
      "path": "dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
      "source_step": "quantize",
      "size_bytes": 654321000,
      "modified_time": 1700000100.0
    },
    {
      "type": "compiled_lib",
      "name": "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so",
      "path": "dist/libs/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so",
      "source_step": "compile",
      "size_bytes": 98765432,
      "modified_time": 1700000200.0
    }
  ]
}
```

**Artifact types:**

| `type` | Produced by | Path pattern |
|---|---|---|
| `wheel` | `/build` | `wheels/*.whl` |
| `model_dir` | `/quantize` | `dist/<ModelName>-<quant>-MLC/` |
| `compiled_lib` | `/compile` | `dist/libs/<ModelName>-<quant>-<device>.so` |

---

## Build pipeline endpoints

### `POST /build`

Runs `go run . build` inside the mlc-cli workspace. Streams build output as SSE.

**When to use:** Install Python wheels (`mlc_llm`, `tvm`) into the runtime environment. Most users run this once after starting the container.

**Request fields:**

| Field | Type | Default | Options | Notes |
|---|---|---|---|---|
| `action` | string | `"full"` | `"full"`, `"build-only"`, `"install-wheels"` | `"install-wheels"` is the fastest option if wheels are already built |
| `cuda` | string | `"y"` | `"y"`, `"n"` | Enable CUDA support |
| `cuda_arch` | string | `"86"` | e.g. `"80"`, `"86"`, `"89"`, `"90"` | CUDA compute capability of your GPU |
| `cutlass` | string | `"n"` | `"y"`, `"n"` | Enable Cutlass kernel support |
| `cublas` | string | `"n"` | `"y"`, `"n"` | Enable cuBLAS support |
| `flash_infer` | string | `"n"` | `"y"`, `"n"` | Enable FlashInfer; disable if build fails |
| `build_wheels` | string | `"y"` | `"y"`, `"n"` | Build Python wheels |
| `tvm_source` | string | `"bundled"` | `"bundled"`, `"relax"`, `"custom"` | TVM source variant |

**Install wheels only (fastest, most common):**

```bash
curl -N -X POST http://localhost:8000/build \
  -H 'Content-Type: application/json' \
  -d '{"action": "install-wheels"}'
```

**Full build with CUDA:**

```bash
curl -N -X POST http://localhost:8000/build \
  -H 'Content-Type: application/json' \
  -d '{"action": "full", "cuda": "y", "cuda_arch": "86", "cutlass": "n", "flash_infer": "n"}'
```

**Example stream output:**

```
data: [mlc-cli] Installing mlc_llm wheel...
data: Successfully installed mlc_llm-0.1.dev0 tvm-0.15.dev0
data: [DONE]
```

**Notes:**
- If the build fails due to a cutlass/flash-attn error, a `[HINT]` line is automatically appended with the retry command.
- If `mlc_llm` and `tvm` are already importable (see `/setup-check`), you may not need to run this.

---

### `GET /conv-templates`

Returns the full list of supported `conv_template` values for `POST /quantize`.

The list is loaded from the pinned mlc-llm source inside the Docker image and
falls back to a hardcoded list if the source is unavailable.
It reflects the actual pinned runtime, not just what public MLC docs list.

```bash
curl -s http://localhost:8000/conv-templates | python3 -m json.tool
```

**Example response:**

```json
{
  "templates": ["LM", "aya-23", "chatml", "deepseek", "deepseek_v3", "llama-3", "tinyllama_v1_0", "..."],
  "default": "auto",
  "note": "Use \"auto\" (the default) to let the API infer the template from the model name. Pass an explicit name only to override inference."
}
```

---

### `POST /quantize`

Converts raw Hugging Face model weights to MLC format and streams output as SSE.

Internally runs two steps:
1. `mlc_llm convert_weight` — quantize weights
2. `mlc_llm gen_config` — generate the runtime config

**When to use:** After raw model weights are available in `models/`, before compiling.

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | **required** | Hugging Face model ID (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) or local path (e.g. `models/Llama-3-8B`) |
| `quant` | string | `"q4f16_1"` | Quantization format |
| `device` | string | `"cuda"` | Target device |
| `conv_template` | string | `"auto"` | Conversation template — tells MLC how to format chat prompts for the model. See below. |
| `output` | string | `""` | Output path. If omitted, defaults to `dist/<model_basename>-<quant>-MLC` |

**`conv_template` behavior:**

| Value | Behavior |
|---|---|
| `"auto"` (default) | API infers the template from the model name. Emits `[INFO]` to the stream with the chosen template. If no template can be inferred, returns HTTP 400 before streaming. |
| Explicit name (e.g. `"chatml"`) | Used as-is. If it looks like a mismatch for the model, the API emits a `[WARNING]` to the stream, then continues. |
| Unknown name (not in supported list) | API emits a `[WARNING]` then continues — mlc-cli may warn further or produce a broken chat config. |

Template names come from the pinned mlc-llm source bundled in the Docker image, not only public docs.
The full list is available at runtime via `GET /conv-templates`.

> **Note:** For models requiring custom templates (e.g. gpt-oss, Harmony), `auto` mode will return HTTP 400.
> Pass `conv_template` explicitly if you intentionally want to experiment with these.

**Supported quantization formats:** `q4f16_1`, `q4f16_ft`, `q4f32_1`, `q3f16_1`, `q8f16_1`, `q0f16`, `q0f32`

**Common model / template mappings (auto-inferred):**

| Model family | Inferred `conv_template` |
|---|---|
| TinyLlama | `tinyllama_v1_0` |
| Llama 3.1, Llama 3.2 | `llama-3_1` |
| Llama 3 | `llama-3` |
| Llama 2 | `llama-2` |
| DeepSeek-V3 | `deepseek_v3` |
| DeepSeek-V2 | `deepseek_v2` |
| DeepSeek-R1 (Qwen distill) | `deepseek_r1_qwen` |
| DeepSeek-R1 (Llama distill) | `deepseek_r1_llama` |
| Mistral / Mixtral | `mistral_default` |
| Ministral 3B | `ministral3` |
| Phi-4 | `phi-4` |
| Phi-3 | `phi-3` |
| Phi-2 | `phi-2` |
| Gemma 3 | `gemma3_instruction` |
| Gemma | `gemma_instruction` |
| Qwen 3.5 | `qwen3_5` |
| Qwen 3 | `qwen3` |
| Qwen 2 / 2.5 | `qwen2` |
| Nemotron | `nemotron` |
| ChatML-style (e.g. Hermes 2) | `chatml` |

**Example (auto-inferred — recommended for beginners):**

```bash
curl -N -X POST http://localhost:8000/quantize \
  -H 'Content-Type: application/json' \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'
```

**Example (explicit template):**

```bash
curl -N -X POST http://localhost:8000/quantize \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "quant": "q4f16_1",
    "device": "cuda",
    "conv_template": "tinyllama_v1_0"
  }'
```


**Example stream output:**

```
data: [mlc-cli] Running convert_weight...
data: Quantizing: 100%|██████████| 201/201 [02:15<00:00]
data: [mlc-cli] Running gen_config...
data: Config written to dist/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC/mlc-chat-config.json
data: [DONE]
```

**Notes:**
- The `model` field accepts Hugging Face model IDs (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`). The API will download the weights automatically.
- For private or gated Hugging Face models, ensure your container has the necessary access token configured.
- The `model` field also accepts relative workspace paths (`models/MyModel`) and absolute paths (`/workspace/mlc-cli/models/MyModel`).
- The output directory will be discoverable via `/artifacts` after quantization.

**Advanced: Using local model weights**

If you prefer to provide model weights manually instead of letting the API download them:

Raw model weights need to be placed under `/workspace/mlc-cli/models/<ModelName>/`.

**Option A — Download inside the container:**

```bash
docker compose exec web bash
cd /workspace/mlc-cli
mkdir -p models

# Example: download or copy your Hugging Face model into models/<ModelName>
# If git-lfs is available in your environment:
# git clone https://huggingface.co/<org>/<ModelName> models/<ModelName>
```

**Option B — Copy model files from your host:**

```bash
docker compose cp /path/to/your/model web:/workspace/mlc-cli/models/
```

**Option C — Mount a host directory:**

Add a volume mount in `docker-compose.yml` to point a host model directory
at `/workspace/mlc-cli/models`.

---

### `POST /compile`

Compiles the model library (`.so` file) from quantized model weights. Streams output as SSE.

**When to use:** After `/quantize` succeeds, before `/chat/load` or `/run`.

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | **required** | Path to quantized model directory (e.g. `dist/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC`) |
| `quant` | string | `"q4f16_1"` | Must match the quant used during `/quantize` |
| `device` | string | `"cuda"` | Target device |
| `output` | string | `""` | Output path for the compiled `.so`. Defaults to `dist/libs/<model>-<quant>-<device>.so` |

**Example:**

```bash
curl -N -X POST http://localhost:8000/compile \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dist/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    "quant": "q4f16_1",
    "device": "cuda"
  }'
```

**Example stream output:**

```
data: [mlc-cli] Compiling model library...
data: Compiling: 100%|██████████| 201/201 [05:30<00:00]
data: Library written to dist/libs/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC-q4f16_1-cuda.so
data: [DONE]
```

---

### `POST /run`

Load-tests a model by initializing the mlc-cli REPL and immediately closing it. Streams output as SSE.

**When to use:** Verify that a compiled model and library load correctly on the target hardware, before using `/chat/load`.

> **Note:** This is a load test, not a chat interface. The upstream `mlc-cli run` command is interactive and does not support a `--prompt` flag. When called via this API, the subprocess initializes the model, prints a ready message, and exits on EOF. Use `/chat/completions` for actual chat.

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_name` | string | **required** | Name of the model (e.g. `TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC`) |
| `model_url` | string | `""` | HuggingFace URL to download from if model is not local |
| `device` | string | `"cuda"` | Target device |
| `profile` | string | `"default"` | Memory profile: `"really-low"`, `"low"`, `"default"`, `"high"` |
| `model_lib` | string | `""` | Path to compiled `.so` library. Accepts workspace-relative paths |
| `quant` | string | `""` | If provided with no `model_lib`, auto-resolves the library from `dist/libs/` |

**Example (with explicit model lib):**

```bash
curl -N -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    "device": "cuda",
    "profile": "low",
    "model_lib": "dist/libs/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC-q4f16_1-cuda.so"
  }'
```

**Example (auto-resolve library with `quant`):**

```bash
curl -N -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
    "device": "cuda",
    "profile": "low",
    "quant": "q4f16_1"
  }'
```

**Notes:**
- If `quant` is given and exactly one matching library is found in `dist/libs/`, it is resolved automatically.
- If multiple libraries match the `quant`/`device` pattern, the request fails — pass `model_lib` explicitly.
- If no library is found, the run proceeds without `--model-lib` (JIT fallback).

---

## Chat endpoints

The chat path uses the MLC-LLM Python engine directly — no subprocess.
One model can be loaded at a time. The server does not store conversation history.

### Complete chat flow

```
1. Build/install wheels (if mlc_llm is not yet importable)
2. POST /chat/load    — load compiled model into GPU memory
3. GET  /chat/status  — verify the engine is ready
4. POST /chat/completions — send messages, receive reply
5. POST /chat/unload  — free GPU memory when done
```

---

### `POST /chat/load`

Loads a compiled model into GPU memory. Must be called before `/chat/completions`.

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | **required** | Path to quantized model directory. Accepts workspace-relative paths |
| `model_lib` | string | **required** | Path to compiled `.so` library. Accepts workspace-relative paths |
| `device` | string | `"cuda:0"` | Target device |

**Example (workspace-relative paths):**

```bash
curl -X POST http://localhost:8000/chat/load \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
    "model_lib": "dist/libs/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so",
    "device": "cuda:0"
  }'
```

**Example response:**

```json
{
  "status": "success",
  "message": "Engine loaded for model /workspace/mlc-cli/dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC"
}
```

**Error codes:**

| Code | Reason |
|---|---|
| `400` | Model directory or library file not found |
| `409` | Another model is already loaded — call `/chat/unload` first |
| `503` | `mlc_llm` is not importable in the runtime environment |
| `500` | Engine initialization failed |

**Notes:**
- Both `model` and `model_lib` accept workspace-relative paths (relative to `/workspace/mlc-cli`).
  Replace the above example paths with the actual paths from your `/artifacts` output.
- Only one model can be loaded at a time. To switch models, call `/chat/unload` first.

---

### `GET /chat/status`

Returns the current engine state.

```bash
curl http://localhost:8000/chat/status
```

**When no engine is loaded:**

```json
{"loaded": false}
```

**When an engine is loaded:**

```json
{
  "loaded": true,
  "model": "/workspace/mlc-cli/dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
  "device": "cuda:0"
}
```

---

### `POST /chat/completions`

Sends a message array and receives a reply from the loaded engine.

The engine must be loaded first via `/chat/load`.
The client sends the full conversation history with each request (no server-side history).

**Request fields:**

| Field | Type | Default | Notes |
|---|---|---|---|
| `messages` | array | **required** | List of `{"role": ..., "content": ...}` objects |
| `max_tokens` | int | `512` | Maximum tokens to generate |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_p` | float | `1.0` | Top-p sampling |
| `stream` | bool | `false` | Stream the reply as SSE events |

**Non-streaming example:**

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

**Non-streaming response:**

```json
{
  "object": "chat.completion",
  "model": "/workspace/mlc-cli/dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "2 + 2 equals 4."},
      "finish_reason": "stop"
    }
  ]
}
```

**Streaming example:**

```bash
curl -N -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a short joke."}],
    "stream": true
  }'
```

**Streaming output:**

```
data: {"delta": "Why"}
data: {"delta": " did"}
data: {"delta": " the"}
data: {"delta": " scarecrow"}
data: {"delta": " win"}
data: {"delta": " an award?"}
data: {"delta": " Because he was outstanding in his field."}
data: [DONE]
```

**Multi-turn conversation example:**

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Nice to meet you, Alice!"},
      {"role": "user", "content": "What is my name?"}
    ]
  }'
```

**Error codes:**

| Code | Reason |
|---|---|
| `422` | `messages` is empty, or a message has a blank `role` or `content` |
| `503` | No engine is loaded — call `/chat/load` first |
| `500` | Engine generation failed |

---

### `POST /chat/unload`

Unloads the active engine and frees GPU memory. Safe to call even if no engine is loaded.

```bash
curl -X POST http://localhost:8000/chat/unload
```

**Response:**

```json
{"status": "success", "message": "Engine unloaded"}
```

---

## Interactive API docs

When the service is running, full request/response schemas are available at:

```
http://localhost:8000/docs
```
