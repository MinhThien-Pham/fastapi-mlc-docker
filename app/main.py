from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.helpers import build_compile_command, build_mlc_cli_command, build_quantize_command, build_run_command, detect_known_failure, discover_artifacts, run_tool_check

app = FastAPI(title="FastAPI MLC-CLI")

# ── Paths ─────────────────────────────────────────────────────────────────────
MLC_CLI_PATH = Path("/workspace/mlc-cli")
MLC_CLI_URL = "https://github.com/ballinyouup/mlc-cli.git"


# ── Internal subprocess helpers ───────────────────────────────────────────────

def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


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
    "llama-3.1",
    "llama-3",
    "llama-2",
    "chatml",
    "mistral_default",
    "ministral",
    "phi-3",
    "phi-2",
    "gemma",
    "qwen2",
]


class QuantizeRequest(BaseModel):
    """Request body for POST /quantize.

    ``model`` is required — it must be a path to a Hugging Face model directory
    (e.g. ``models/Llama-3-8B``) or a Hugging Face hub identifier.

    The mlc-cli ``quantize`` sub-command drives the conversion: it first calls
    ``mlc_llm convert_weight`` and then ``mlc_llm gen_config``.
    """
    model: str
    quant: QUANT_OPTIONS = "q4f16_1"  # type: ignore[valid-type]
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    conv_template: CONV_TEMPLATE_OPTIONS = "llama-3"  # type: ignore[valid-type]
    # Optional: if empty, mlc-cli derives a default from model name + quant
    output: str = ""


class CompileRequest(BaseModel):
    """Request body for POST /compile."""
    model: str
    quant: QUANT_OPTIONS = "q4f16_1"  # type: ignore[valid-type]
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    output: str = ""


class RunRequest(BaseModel):
    """Request body for POST /run."""
    model_name: str
    model_url: str = ""
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    profile: Literal["really-low", "low", "default", "high"] = "default"
    model_lib: str = ""


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

    # ── Per-tool checks ───────────────────────────────────────────────────────
    checks: dict = {
        "repo": {
            "available": repo_exists,
            "path": str(MLC_CLI_PATH),
            "output": (
                ""
                if repo_exists
                else "mlc-cli repo not found — call POST /ensure-repo-exists first"
            ),
        },
        "go":         run_tool_check(["go", "version"]),
        "conda":      run_tool_check(["conda", "--version"]),
        "nvidia_smi": run_tool_check(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "nvcc":       run_tool_check(["nvcc", "--version"]),
    }

    # Enrich repo entry with remote URL when the repo is present
    if repo_exists:
        git_check = run_tool_check(["git", "remote", "get-url", "origin"])
        checks["repo"]["origin"] = git_check.get("output", "")

    # ── Derive overall status ─────────────────────────────────────────────────
    # "critical" tools — without these the build cannot start at all
    critical_ok = checks["go"]["available"] and checks["conda"]["available"]
    gpu_ok = checks["nvidia_smi"]["available"] and checks["nvcc"]["available"]

    if repo_exists and critical_ok:
        overall = "ok"
    elif critical_ok:
        # repo missing is a warning, not a hard error (it can be cloned on demand)
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
        # kept for backward compat with test_pipeline.py
        "repo_exists": repo_exists,
        # new structured output
        "status":   overall,
        "checks":   checks,
        "warnings": warnings,
    }


@app.post("/ensure-repo-exists")
def ensure_repo_exists():
    if MLC_CLI_PATH.exists():
        return {
            "status": "ok",
            "message": "mlc-cli already exists",
            "path": str(MLC_CLI_PATH),
        }

    print(f"[INFO] Cloning {MLC_CLI_URL} into {MLC_CLI_PATH}...")
    try:
        run_git(["clone", MLC_CLI_URL, str(MLC_CLI_PATH)])
        return {
            "status": "ok",
            "message": "mlc-cli cloned successfully",
            "path": str(MLC_CLI_PATH),
        }
    except subprocess.CalledProcessError as exc:
        return {
            "status": "error",
            "message": "failed to clone mlc-cli",
            "path": str(MLC_CLI_PATH),
            "stderr": exc.stderr.strip(),
        }


@app.get("/repo-status")
def repo_status():
    """Check if the mlc-cli repository is clean or dirty (has uncommitted changes)."""
    if not MLC_CLI_PATH.exists():
        return {
            "status": "error",
            "message": "mlc-cli repo not found",
            "repo_exists": False,
        }

    result = run_command(["git", "status", "--porcelain"], cwd=MLC_CLI_PATH)
    if result.returncode != 0:
        return {
            "status": "error",
            "message": "failed to check git status",
            "stderr": result.stderr.strip(),
        }

    status_output = result.stdout.strip()
    is_clean = len(status_output) == 0

    return {
        "status": "ok",
        "repo_exists": True,
        "is_clean": is_clean,
        "message": "Repository is clean" if is_clean else "Repository has uncommitted changes",
        "changes": status_output.split("\n") if status_output else [],
    }


# ── Build endpoint ─────────────────────────────────────────────────────────────

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
            yield "data: [ERROR] mlc-cli repo not found. Call /ensure-repo-exists first.\n\n"
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


# ── Quantize endpoint ────────────────────────────────────────────────────────────────

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

    Example — quantize a locally-cloned Llama-3 8B model::

        curl -N -X POST http://localhost:8000/quantize \\
             -H 'Content-Type: application/json' \\
             -d '{"model": "models/Llama-3-8B", "quant": "q4f16_1", "device": "cuda"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli repo not found. Call /ensure-repo-exists first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # ── Normalize model path: resolve relative paths against the workspace ────
    model_path = Path(req.model)
    if not model_path.is_absolute():
        model_path = MLC_CLI_PATH / model_path
    resolved_model = model_path.resolve()

    if not resolved_model.exists():
        original = req.model
        async def model_error_stream():
            yield (
                f"data: [ERROR] model path not found.\n\n"
                f"data:   original:  {original}\n\n"
                f"data:   resolved:  {resolved_model}\n\n"
            )
        return StreamingResponse(model_error_stream(), media_type="text/event-stream")

    req = req.model_copy(update={"model": str(resolved_model)})

    cmd = build_quantize_command(req)

    return StreamingResponse(
        stream_subprocess(cmd, cwd=MLC_CLI_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Compile endpoint ────────────────────────────────────────────────────────────────

@app.post("/compile")
async def compile_model(req: CompileRequest):
    """Compile model library and stream output as SSE.

    Internally this calls the mlc-cli ``compile`` sub-command.
    The ``model`` field is required.  All other fields have sensible defaults.

    Example — compile a locally-cloned Llama-3 8B model::

        curl -N -X POST http://localhost:8000/compile \\
             -H 'Content-Type: application/json' \\
             -d '{"model": "models/Llama-3-8B", "quant": "q4f16_1", "device": "cuda"}'

    Each SSE line is prefixed with ``data: ``.
    The stream ends with ``data: [DONE]`` on success or ``data: [ERROR] ...``
    on failure.
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield "data: [ERROR] mlc-cli repo not found. Call /ensure-repo-exists first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # ── Normalize model path: resolve relative paths against the workspace ────
    model_path = Path(req.model)
    if not model_path.is_absolute():
        model_path = MLC_CLI_PATH / model_path
    resolved_model = model_path.resolve()

    if not resolved_model.exists():
        original = req.model
        async def model_error_stream():
            yield (
                f"data: [ERROR] model path not found.\n\n"
                f"data:   original:  {original}\n\n"
                f"data:   resolved:  {resolved_model}\n\n"
            )
        return StreamingResponse(model_error_stream(), media_type="text/event-stream")

    req = req.model_copy(update={"model": str(resolved_model)})

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
            yield "data: [ERROR] mlc-cli repo not found. Call /ensure-repo-exists first.\n\n"
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
