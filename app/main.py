from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import chat_engine_manager

from app.helpers import (
    build_compile_command,
    build_mlc_cli_command,
    build_quantize_command,
    build_run_command,
    detect_known_failure,
    discover_artifacts,
    get_supported_conv_templates,
    is_hf_model_id,
    prepare_conv_template_for_quantize,
    resolve_quantized_model_dir,
    run_tool_check,
    resolve_chat_artifacts,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: Log-only Local Alignment Check ───────────────────────────────
    try:
        print(f"[BOOT] FastAPI wrapper starting...")
        print(f"[BOOT] MLC_CLI_PATH: {MLC_CLI_PATH}")
        print(f"[BOOT] BAKED_MLC_CLI_PATH: {BAKED_MLC_CLI_PATH}")
    except Exception as e:
        print(f"[BOOT] Failed to log startup paths: {e}")

    try:
        yield
    finally:
        try:
            chat_engine_manager.unload_engine()
            print("[BOOT] Chat engine unloaded cleanly during shutdown.")
        except Exception as e:
            print(f"[BOOT] Error unloading chat engine during shutdown: {e}")


app = FastAPI(title="FastAPI MLC-CLI", lifespan=lifespan)

# ── Paths ─────────────────────────────────────────────────────────────────────
MLC_CLI_PATH = Path(os.getenv("MLC_CLI_PATH", "/workspace/mlc-cli"))
BAKED_MLC_CLI_PATH = Path(os.getenv("BAKED_MLC_CLI_PATH", "/opt/mlc-cli"))


# ── Internal subprocess helpers ───────────────────────────────────────────────

def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)

def read_text_file(path: Path) -> str | None:
    try:
        return path.read_text().strip() if path.exists() else None
    except Exception:
        return None

def git_head(repo_dir: Path) -> str | None:
    if not repo_dir.exists():
        return None
    res = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else None


def run_git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


async def stream_subprocess(command: list[str], cwd: Path | None = None) -> AsyncIterator[str]:
    """Yield stdout/stderr lines from a subprocess as SSE-formatted strings.

    Known build failures (e.g. cutlass / flash-attn) are detected line-by-line.
    When a match is found a single ``[HINT]`` line is emitted immediately after
    the offending line so the caller knows exactly how to retry.
    """
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    assert proc.stdout is not None

    hint_emitted = False  # emit at most one hint per build run
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip()
        if not line:
            # Skip blank / whitespace-only lines — they produce noisy "data: "
            # events in terminal demos without conveying any useful information.
            continue
        yield f"data: {line}\n\n"

        if not hint_emitted:
            hint = detect_known_failure(line)
            if hint:
                yield f"data: [HINT] {hint}\n\n"
                hint_emitted = True

    await proc.wait()
    if proc.returncode != 0:
        yield f"data: [ERROR] Process exited with code {proc.returncode}\n\n"
    else:
        yield "data: [DONE]\n\n"


# ── Request / Response models ─────────────────────────────────────────────────

class BuildRequest(BaseModel):
    action: Literal["full", "build-only", "install-wheels"] = "full"
    tvm_source: Literal["bundled", "relax", "custom"] = "bundled"
    cuda: Literal["y", "n"] = "y"
    cuda_arch: str = "86"
    cutlass: Literal["y", "n"] = "n"
    cublas: Literal["y", "n"] = "n"
    flash_infer: Literal["y", "n"] = "n"
    rocm: Literal["y", "n"] = "n"
    vulkan: Literal["y", "n"] = "n"
    opencl: Literal["y", "n"] = "n"
    build_wheels: Literal["y", "n"] = "y"
    force_clone: Literal["y", "n"] = "n"

QUANT_OPTIONS = Literal[
    "q4f16_1",
    "q4f16_ft",
    "q4f32_1",
    "q3f16_1",
    "q8f16_1",
    "q0f16",
    "q0f32",
]

CONV_TEMPLATE_OPTIONS = Literal[
    "auto",
    "llama-4",
    "llama-3_1",
    "llama-3",
    "llama-2",
    "chatml",
    "chatml_nosystem",
    "mistral_default",
    "ministral3",
    "ministral3_reasoning",
    "phi-4",
    "phi-3",
    "phi-3-vision",
    "phi-2",
    "gemma3_instruction",
    "gemma_instruction",
    "qwen3_5",
    "qwen3_5_nothink",
    "qwen3",
    "qwen2",
    "deepseek_v3",
    "deepseek_v2",
    "deepseek_r1_qwen",
    "deepseek_r1_llama",
    "deepseek",
    "hermes3_llama-3_1",
    "hermes2_pro_llama3",
    "open_hermes_mistral",
    "neural_hermes_mistral",
    "tinyllama_v1_0",
    "nemotron",
    "gorilla",
    "gorilla-openfunctions-v2",
    "gpt2",
    "gpt_bigcode",
    "dolly",
    "oasst",
    "glm",
    "olmo",
    "olmo2",
    "orion",
    "llava",
    "llm-jp",
    "redpajama_chat",
    "rwkv_world",
    "stablelm",
    "stablelm-3b",
    "stablelm-2",
    "wizardlm_7b",
    "wizard_coder_or_math",
    "LM",
    "aya-23",
    "codellama_completion",
    "codellama_instruct",
    "llama_default",
]


class QuantizeRequest(BaseModel):
    """Request body for POST /quantize.

    ``model`` is required — it must be a path to a Hugging Face model directory
    (e.g. ``models/Llama-3-8B``) or a Hugging Face hub identifier.

    The mlc-cli ``quantize`` sub-command drives the conversion: it first calls
    ``mlc_llm convert_weight`` and then ``mlc_llm gen_config``.

    ``conv_template`` defaults to ``"auto"``, which infers the correct MLC
    conversation template from the model name.  Pass an explicit template name
    to override inference.
    """
    model: str
    quant: QUANT_OPTIONS = "q4f16_1"  # type: ignore[valid-type]
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    conv_template: str = "auto"
    # Optional: if empty, mlc-cli derives a default from model name + quant
    output: str = ""


class CompileRequest(BaseModel):
    """Request body for POST /compile."""
    model: str
    quant: QUANT_OPTIONS = "q4f16_1"  # type: ignore[valid-type]
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    output: str = ""


class ChatLoadRequest(BaseModel):
    """Request body for POST /chat/load to initialize the direct MLCEngine."""
    model: str = ""
    model_name: str = ""
    model_lib: str = ""
    device: str = "cuda:0"
    quant: str = "q4f16_1"


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str   # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Request body for POST /chat/completions.

    Only the fields needed right now are included.  This model is deliberately
    kept small so it is easy to extend later (more sampling params, stop
    sequences, etc.) without breaking callers.
    """
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class RunRequest(BaseModel):
    """Request body for POST /run."""
    model_name: str
    model_url: str = ""
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    profile: Literal["really-low", "low", "default", "high"] = "default"
    model_lib: str = ""
    # Optional: when provided alongside model_name, enables auto-resolution of
    # the compiled library from dist/libs/<model_name>-<quant>-<device>.so
    quant: str = ""


# ── Chat Engine Endpoints ─────────────────────────────────────────────────────

@app.post("/chat/load")
def chat_load(req: ChatLoadRequest):
    """
    Load the MLCEngine with the specified model and library.
    This is an explicit initialization step before any completions can be requested.
    """
    if req.model_lib:
        target_model = req.model_name if req.model_name else req.model
        if not target_model:
            raise HTTPException(status_code=400, detail="When model_lib is provided, model or model_name is required.")

        model_path = Path(target_model)
        if not model_path.is_absolute():
            if not ("/" in target_model or "\\" in target_model) and target_model.endswith("-MLC"):
                model_path = MLC_CLI_PATH / "dist" / target_model
            else:
                model_path = MLC_CLI_PATH / target_model
        if not model_path.exists() or not model_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Model directory not found: {model_path}")

        model_lib_path = Path(req.model_lib)
        if not model_lib_path.is_absolute():
            model_lib_path = MLC_CLI_PATH / req.model_lib
        if not model_lib_path.exists() or not model_lib_path.is_file():
            raise HTTPException(status_code=400, detail=f"Model library file not found: {model_lib_path}")
        
        final_model = str(model_path)
        final_model_lib = str(model_lib_path)
    else:
        try:
            final_model, final_model_lib = resolve_chat_artifacts(
                mlc_cli_path=MLC_CLI_PATH,
                model=req.model,
                model_name=req.model_name,
                quant=req.quant,
                device=req.device,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    engine_device = req.device
    if engine_device == "cuda":
        engine_device = "cuda:0"

    try:
        chat_engine_manager.load_engine(
            model=final_model,
            model_lib=final_model_lib,
            device=engine_device
        )
        return {"status": "success", "message": f"Engine loaded for model {final_model}"}
    except chat_engine_manager.InvalidArtifactPathError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except chat_engine_manager.EngineConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except chat_engine_manager.EngineImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except chat_engine_manager.EngineInitializationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load engine: {str(e)}")


@app.get("/chat/status")
def chat_status():
    """
    Return the current status of the loaded chat engine.
    """
    return chat_engine_manager.get_status()


@app.post("/chat/unload")
def chat_unload():
    """
    Unload the active MLCEngine, freeing its resources.
    Safe to call even if no engine is currently loaded.
    """
    try:
        chat_engine_manager.unload_engine()
        return {"status": "success", "message": "Engine unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload engine: {str(e)}")


@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Run a chat completion against the currently loaded engine.

    The engine must already be loaded via ``POST /chat/load`` before calling
    this endpoint.  Pass a list of messages (role + content) and optional
    generation parameters.

    - ``stream=false`` (default): waits for the full reply and returns JSON.
    - ``stream=true``: returns a Server-Sent Events stream.  Each event is
      ``data: {"delta": "<text>"}``.  The stream ends with ``data: [DONE]``.
      On error, a ``data: {"error": "<message>"}`` event is emitted before
      ``data: [DONE]`` so the client always sees a clean stream termination.

    Examples::

        # non-streaming
        curl -s -X POST http://localhost:8000/chat/completions \\
             -H 'Content-Type: application/json' \\
             -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

        # streaming
        curl -N -X POST http://localhost:8000/chat/completions \\
             -H 'Content-Type: application/json' \\
             -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
    """
    import json as _json

    # ── Basic payload validation ──────────────────────────────────────────────
    if not req.messages:
        raise HTTPException(
            status_code=422,
            detail="messages must be a non-empty list.",
        )

    for i, msg in enumerate(req.messages):
        if not msg.role.strip():
            raise HTTPException(
                status_code=422,
                detail=f"messages[{i}].role must not be blank.",
            )
        if not msg.content.strip():
            raise HTTPException(
                status_code=422,
                detail=f"messages[{i}].content must not be blank.",
            )

    # ── Serialise to plain dicts for the engine ───────────────────────────────
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    # ── Streaming path ────────────────────────────────────────────────────────
    if req.stream:
        async def _sse_generator():
            try:
                async for delta in chat_engine_manager.stream_completion(
                    messages=messages_dicts,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                ):
                    yield f"data: {_json.dumps({'delta': delta})}\n\n"
            except (
                chat_engine_manager.EngineNotLoadedError,
                chat_engine_manager.EngineStreamError,
            ) as exc:
                yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            except Exception as exc:
                yield f"data: {_json.dumps({'error': f'Unexpected error: {exc}'})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming path (unchanged) ────────────────────────────────────────
    try:
        reply = chat_engine_manager.generate_completion(
            messages=messages_dicts,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except chat_engine_manager.EngineNotLoadedError as e:
        # 503 Service Unavailable: the service exists but the engine isn't ready
        raise HTTPException(status_code=503, detail=str(e))
    except chat_engine_manager.EngineGenerationError as e:
        # 500 Internal Server Error: the engine is loaded but generation failed
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during generation: {str(e)}")

    # ── Non-streaming response ────────────────────────────────────────────────
    # Lean envelope — forward-compatible: add 'id', 'model', 'usage', etc. later.
    return {
        "object": "chat.completion",
        "model": chat_engine_manager.get_status().get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI + MLC CLI"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/setup-check")
def setup_check():
    """Inspect the environment: mlc-cli repo, Go, Conda, nvidia-smi, and nvcc.

    Returns a structured ``checks`` dict (one entry per tool) plus a top-level
    ``status`` ("ok" | "warning" | "error") and a ``warnings`` list.

    The ``repo_exists`` field is kept for backward compatibility with
    ``test_pipeline.py``.
    """
    repo_exists = MLC_CLI_PATH.exists()
    baked_head = git_head(BAKED_MLC_CLI_PATH)
    workspace_head = git_head(MLC_CLI_PATH)

    # ── Python checks ────────────────────────────────────────────────────────
    py_version_res = run_command(["python", "--version"])
    py_version = py_version_res.stdout.strip() if py_version_res.returncode == 0 else ""
    
    expected_py = None
    versions_sh = MLC_CLI_PATH / "scripts" / "config" / "versions.sh"
    if versions_sh.exists():
        for line in versions_sh.read_text().splitlines():
            if line.startswith("PYTHON_VERSION="):
                expected_py = line.split("=", 1)[1].strip('"\'')
                break

    py_match = None
    if expected_py and py_version:
        py_match = expected_py in py_version

    mlc_import = run_command(["python", "-c", "import mlc_llm"]).returncode == 0
    tvm_import = run_command(["python", "-c", "import tvm"]).returncode == 0

    # ── Per-tool checks ───────────────────────────────────────────────────────
    checks: dict = {
        "repo": {
            "available": repo_exists,
            "path": str(MLC_CLI_PATH),
            "output": "" if repo_exists else "mlc-cli workspace not found",
        },
        "go":         run_tool_check(["go", "version"]),
        "conda":      run_tool_check(["conda", "--version"]),
        "nvidia_smi": run_tool_check(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "nvcc":       run_tool_check(["nvcc", "--version"]),
    }

    if repo_exists:
        git_check = run_tool_check(["git", "-C", str(MLC_CLI_PATH), "remote", "get-url", "origin"])
        checks["repo"]["origin"] = git_check.get("output", "")

    # ── Derive overall status ─────────────────────────────────────────────────
    critical_ok = checks["go"]["available"] and checks["conda"]["available"]
    gpu_ok = checks["nvidia_smi"]["available"] and checks["nvcc"]["available"]

    if repo_exists and critical_ok:
        overall = "ok"
    elif critical_ok:
        overall = "warning"
    else:
        overall = "error"

    warnings: list[str] = []
    if not gpu_ok:
        warnings.append(
            "nvidia-smi or nvcc is unavailable. "
            "GPU-dependent build steps will fail. "
            "Make sure CUDA drivers are installed and the GPU is visible to the container."
        )

    return {
        "repo_exists": repo_exists,
        "status":   overall,
        "checks":   checks,
        "warnings": warnings,
        "wrapper_info": {
            "mlc_cli_path": str(MLC_CLI_PATH),
            "baked_mlc_cli_path": str(BAKED_MLC_CLI_PATH),
            "baked_ref": read_text_file(Path("/opt/mlc-cli-ref.txt")),
            "baked_actual_head": baked_head,
            "workspace_head": workspace_head,
            "workspace_matches_baked": baked_head == workspace_head if baked_head and workspace_head else None,
            "python_runtime_version": py_version,
            "expected_python_version": expected_py,
            "python_match": py_match,
            "mlc_llm_importable": mlc_import,
            "tvm_importable": tvm_import,
            "artifact_dirs_present": {
                "models": (MLC_CLI_PATH / "models").exists(),
                "dist": (MLC_CLI_PATH / "dist").exists(),
                "wheels": (MLC_CLI_PATH / "wheels").exists(),
                "mlc-llm": (MLC_CLI_PATH / "mlc-llm").exists(),
                "tvm": (MLC_CLI_PATH / "tvm").exists(),
            }
        }
    }


@app.get("/repo-status")
def repo_status():
    """Read-only endpoint to check baked vs workspace status."""
    baked_head = git_head(BAKED_MLC_CLI_PATH)
    workspace_head = git_head(MLC_CLI_PATH)
    
    matches = None
    if baked_head and workspace_head:
        matches = baked_head == workspace_head

    return {
        "source_management": "baked-image",
        "mlc_cli_path": str(MLC_CLI_PATH),
        "baked_mlc_cli_path": str(BAKED_MLC_CLI_PATH),
        "baked_ref_file": read_text_file(Path("/opt/mlc-cli-ref.txt")),
        "baked_repo_file": read_text_file(Path("/opt/mlc-cli-repo.txt")),
        "baked_actual_head": baked_head,
        "workspace_head": workspace_head,
        "workspace_matches_baked": matches,
        "artifact_dirs": {
            "models": (MLC_CLI_PATH / "models").exists(),
            "dist": (MLC_CLI_PATH / "dist").exists(),
            "wheels": (MLC_CLI_PATH / "wheels").exists(),
            "mlc-llm": (MLC_CLI_PATH / "mlc-llm").exists(),
            "tvm": (MLC_CLI_PATH / "tvm").exists(),
        },
        "dev_mode": False
    }


# ── Build endpoint ────────────────────────────────────────────────────────────

@app.post("/build")
async def build(req: BuildRequest):
    """Trigger ``mlc-cli build`` non-interactively and stream stdout/stderr as SSE.

    Known failures (cutlass / flash-attn) are automatically detected and
    followed by a ``[HINT]`` line that tells you exactly how to retry.

    Example — stream a wheel-only install::

        curl -N -X POST http://localhost:8000/build \\
             -H 'Content-Type: application/json' \\
             -d '{"action":"install-wheels"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli workspace not found. The Docker image should bake mlc-cli and the entrypoint should sync it into /workspace/mlc-cli. Check /repo-status and rebuild the image if needed.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    cmd = build_mlc_cli_command(req)

    return StreamingResponse(
        stream_subprocess(cmd, cwd=MLC_CLI_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent nginx from buffering SSE
        },
    )


# ── Quantize endpoint ─────────────────────────────────────────────────

@app.post("/quantize")
@app.post("/convert", include_in_schema=False)
async def quantize_model(req: QuantizeRequest):
    """Quantize (convert) raw model weights to MLC format and stream output as SSE.

    Internally this calls the mlc-cli ``quantize`` sub-command which runs:

    1. ``mlc_llm convert_weight`` — convert weights to MLC format.
    2. ``mlc_llm gen_config``     — generate the runtime config file.

    The ``model`` field is required.  All other fields have sensible defaults.
    If ``output`` is omitted, mlc-cli derives a default path of the form
    ``dist/<model_basename>-<quant>-MLC``.

    ``conv_template`` defaults to ``"auto"``, which automatically selects
    the right MLC conversation template based on the model name.  Pass an
    explicit template name if you need to override the inferred value.

    Example — quantize TinyLlama (conv_template auto-inferred)::

        curl -N -X POST http://localhost:8000/quantize \\\
             -H 'Content-Type: application/json' \\\
             -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli workspace not found. The Docker image should bake mlc-cli and the entrypoint should sync it into /workspace/mlc-cli. Check /repo-status and rebuild the image if needed.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # ── Resolve conv_template before any I/O ────────────────────────────────
    try:
        resolved_template, template_messages = prepare_conv_template_for_quantize(
            model=req.model,
            requested_template=req.conv_template,
            mlc_cli_path=MLC_CLI_PATH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Normalize model: HF IDs pass through; local paths are resolved ─────────
    if is_hf_model_id(req.model):
        # Hugging Face hub identifier — mlc-cli will download the model.
        # No local path resolution needed; pass the ID through unchanged.
        resolved_req = req.model_copy(update={"conv_template": resolved_template})
    else:
        model_path = Path(req.model)
        if not model_path.is_absolute():
            candidate = (MLC_CLI_PATH / model_path).resolve()
            if candidate.exists():
                model_path = candidate
            else:
                model_path = (Path.cwd() / model_path).resolve()
        resolved_model = model_path.resolve()

        if not resolved_model.exists():
            original = req.model
            async def model_error_stream():
                yield (
                    f"data: [ERROR] model path not found.\n\n"
                    f"data:   original:  {original}\n\n"
                    f"data:   resolved:  {resolved_model}\n\n"
                    f"data:   Tip: for models on Hugging Face Hub, pass the model ID "
                    f"(e.g. \"Owner/ModelName\") to have mlc-cli download weights automatically.\n\n"
                )
            return StreamingResponse(model_error_stream(), media_type="text/event-stream")

        resolved_req = req.model_copy(update={
            "model": str(resolved_model),
            "conv_template": resolved_template,
        })

    cmd = build_quantize_command(resolved_req)

    async def _quantize_stream():
        # Emit conv_template info/warnings before mlc-cli output.
        for msg in template_messages:
            yield msg
        async for chunk in stream_subprocess(cmd, cwd=MLC_CLI_PATH):
            yield chunk

    return StreamingResponse(
        _quantize_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conv-templates endpoint ───────────────────────────────────────────────────

@app.get("/conv-templates")
def list_conv_templates():
    """Return the set of supported conv_template values for POST /quantize.

    The list is loaded from the pinned mlc-llm source bundled in the Docker
    image (``mlc-llm/python/mlc_llm/interface/gen_config.py``) and falls back
    to a hardcoded list if the source is not available.

    Example::

        curl http://localhost:8000/conv-templates
    """
    supported = get_supported_conv_templates(MLC_CLI_PATH)
    return {
        "templates": sorted(supported),
        "default": "auto",
        "note": (
            "Use \"auto\" (the default) to let the API infer the template from "
            "the model name. Pass an explicit name only to override inference."
        ),
    }




@app.post("/compile")
async def compile_model(req: CompileRequest):
    """Compile model library and stream output as SSE.

    Internally this calls the mlc-cli ``compile`` sub-command.

    The ``model`` field accepts any of the following forms:

    * **Hugging Face model ID** (recommended) — e.g.
      ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``.  The basename after ``/`` is
      used to search ``dist/`` for the matching quantized artifact directory.
    * **Short model name** — e.g. ``TinyLlama-1.1B-Chat-v1.0``.
    * **Artifact folder name** — the exact name of a directory under ``dist/``
      (e.g. ``TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC``).
    * **Exact artifact path** (advanced) — an absolute path or a path
      relative to the mlc-cli workspace.

    The ``quant`` field (default ``q4f16_1``) disambiguates when multiple
    quantized artifacts exist for the same model name.  Use ``GET /artifacts``
    to list all available artifacts if resolution is ambiguous.

    Example — compile using an HF model ID (beginner flow)::

        curl -N -X POST http://localhost:8000/compile \\
             -H 'Content-Type: application/json' \\
             -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

    Example — compile using an exact artifact path (advanced)::

        curl -N -X POST http://localhost:8000/compile \\
             -H 'Content-Type: application/json' \\
             -d '{"model": "dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli workspace not found. The Docker image should bake mlc-cli and the entrypoint should sync it into /workspace/mlc-cli. Check /repo-status and rebuild the image if needed.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # ── Resolve model: accept HF ID, short name, artifact folder, or exact path ─
    resolution = resolve_quantized_model_dir(MLC_CLI_PATH, req.model, req.quant)

    if resolution == "none":
        original = req.model
        quant = req.quant
        async def no_artifact_stream():
            yield (
                f"data: [ERROR] No quantized artifact found for model '{original}' "
                f"with quant '{quant}'.\n\n"
                f"data:   Run POST /quantize first to generate the artifact, "
                f"then retry POST /compile.\n\n"
                f"data:   Use GET /artifacts to list all available artifacts.\n\n"
            )
        return StreamingResponse(no_artifact_stream(), media_type="text/event-stream")

    if resolution == "multiple":
        original = req.model
        quant = req.quant
        async def multiple_artifacts_stream():
            yield (
                f"data: [ERROR] Multiple quantized artifacts match model '{original}' "
                f"with quant '{quant}'.\n\n"
                f"data:   Use GET /artifacts to list candidates, then pass an exact "
                f"artifact path in the 'model' field.\n\n"
            )
        return StreamingResponse(multiple_artifacts_stream(), media_type="text/event-stream")

    # Single match or exact path — proceed with compilation
    req = req.model_copy(update={"model": str(resolution)})
    cmd = build_compile_command(req)

    return StreamingResponse(
        stream_subprocess(cmd, cwd=MLC_CLI_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Run endpoint ──────────────────────────────────────────────────────────────────

@app.post("/run")
async def run_model(req: RunRequest):
    """Load-test a model by initializing the interactive REPL.

    Internally this calls the mlc-cli ``run`` sub-command.

    **LIMITATION**: The upstream ``mlc-cli run`` command is interactive by default
    and does NOT support a non-interactive single-shot ``--prompt`` flag. When
    called via this API endpoint, no standard input is provided. The subprocess
    will initialize the model, print its ready state, and immediately exit upon
    encountering EOF. This effectively serves as a "load test" to verify model
    and compiled library compatibility.

    The ``model_name`` field is required.

    Example — load test a model::

        curl -N -X POST http://localhost:8000/run \\
             -H 'Content-Type: application/json' \\
             -d '{"model_name": "Llama-3-8B", "device": "cuda", "profile": "default"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli workspace not found. The Docker image should bake mlc-cli and the entrypoint should sync it into /workspace/mlc-cli. Check /repo-status and rebuild the image if needed.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # ── Normalize model_lib: resolve relative paths against the workspace ─────
    if req.model_lib:
        lib_path = Path(req.model_lib)
        if not lib_path.is_absolute():
            lib_path = MLC_CLI_PATH / lib_path
        resolved_lib = lib_path.resolve()

        if not resolved_lib.is_file():
            original = req.model_lib
            async def lib_error_stream():
                yield (
                    f"data: [ERROR] model_lib not found.\n\n"
                    f"data:   original:  {original}\n\n"
                    f"data:   resolved:  {resolved_lib}\n\n"
                )
            return StreamingResponse(lib_error_stream(), media_type="text/event-stream")

        # Replace with the resolved absolute path so the upstream script works
        # regardless of its own cwd changes.
        req = req.model_copy(update={"model_lib": str(resolved_lib)})

    elif req.quant:
        # ── Auto-resolve model_lib from dist/libs/ when quant is provided ────
        # Pattern: dist/libs/<model_name>-<quant>-<device>.so
        from app.helpers import resolve_model_lib
        resolved_lib = resolve_model_lib(
            mlc_cli_path=MLC_CLI_PATH,
            model_name=req.model_name,
            quant=req.quant,
            device=req.device,
        )
        if resolved_lib == "multiple":
            async def multi_lib_error_stream():
                yield (
                    f"data: [ERROR] Multiple compiled libraries found for "
                    f"{req.model_name}/{req.quant}/{req.device}.\n\n"
                    f"data:   Pass model_lib explicitly to disambiguate.\n\n"
                )
            return StreamingResponse(multi_lib_error_stream(), media_type="text/event-stream")
        if resolved_lib:
            req = req.model_copy(update={"model_lib": resolved_lib})
        # If resolved_lib is None (not found), proceed without --model-lib (JIT fallback)

    cmd = build_run_command(req)

    return StreamingResponse(
        stream_subprocess(cmd, cwd=MLC_CLI_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Artifacts endpoint ────────────────────────────────────────────────────────

@app.get("/artifacts")
def get_artifacts():
    """Discover outputs from build, convert, and compile steps.

    Returns a structured JSON response of discovered local artifacts.
    Safe to call at any time. If the mlc-cli repository is missing or
    empty, it will return an empty list of artifacts.
    """
    artifacts = discover_artifacts(MLC_CLI_PATH)
    
    counts = {
        "build": sum(1 for a in artifacts if a["source_step"] == "build"),
        "convert": sum(1 for a in artifacts if a["source_step"] in ("convert", "quantize")),
        "quantize": sum(1 for a in artifacts if a["source_step"] in ("convert", "quantize")),
        "compile": sum(1 for a in artifacts if a["source_step"] == "compile"),
        "total": len(artifacts),
    }

    return {
        "status": "ok",
        "root_paths_searched": [str(MLC_CLI_PATH)],
        "counts": counts,
        "artifacts": artifacts
    }
