from __future__ import annotations

import asyncio
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
    get_git_dirty_state,
    get_repo_alignment,
    get_startup_alignment_message,
    restore_tracked_changes,
    run_tool_check,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: Log-only Local Alignment Check ───────────────────────────────
    # Perform a lightweight check of the Bryan mlc-cli repo state.
    # This is local-only and does not fetch or repair.
    try:
        upstream_meta = Path("/app/.upstream-sha.json")
        align = get_repo_alignment(MLC_CLI_PATH, upstream_meta)
        msg = get_startup_alignment_message(align)
        print(f"[BOOT] {msg}")
    except Exception as e:
        print(f"[BOOT] Failed to perform startup repo alignment check: {e}")

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


class ChatLoadRequest(BaseModel):
    """Request body for POST /chat/load to initialize the direct MLCEngine."""
    model: str
    model_lib: str
    device: str = "cuda:0"


class RunRequest(BaseModel):
    """Request body for POST /run."""
    model_name: str
    model_url: str = ""
    device: Literal["cuda", "metal", "vulkan", "opencl", "rocm"] = "cuda"
    profile: Literal["really-low", "low", "default", "high"] = "default"
    model_lib: str = ""


# ── Chat Engine Endpoints ─────────────────────────────────────────────────────

@app.post("/chat/load")
def chat_load(req: ChatLoadRequest):
    """
    Load the MLCEngine with the specified model and library.
    This is an explicit initialization step before any completions can be requested.
    """
    try:
        chat_engine_manager.load_engine(
            model=req.model,
            model_lib=req.model_lib,
            device=req.device
        )
        return {"status": "success", "message": f"Engine loaded for model {req.model}"}
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
    # ── 1. Restore tracked source changes in managed repo, keep untracked files ──
    dirty = get_git_dirty_state(MLC_CLI_PATH)
    if dirty["exists"] and dirty["tracked_dirty"]:
        cleanup = restore_tracked_changes(MLC_CLI_PATH)
        if not cleanup["ok"]:
            detail = cleanup["error"] or "failed to restore tracked changes"
            return {
                "status": "error",
                "message": (
                    f"Managed Bryan repo at {MLC_CLI_PATH} has tracked source modifications and cleanup failed. "
                    "Verification and repair/re-alignment are unsafe until tracked code is clean. "
                    f"Details: {detail}"
                ),
                "path": str(MLC_CLI_PATH),
                "action": "tracked-cleanup-failed",
                "tracked_changes": dirty["tracked_changes"],
                "dirty_error": dirty["error"],
            }
        print(
            "[INFO] Managed repo was dirty and was automatically restored to the current checked-out commit "
            "before alignment. Artifacts, downloaded files, and cache were not affected."
        )

    # ── 2. Determine local alignment ─────────────────────────────────────────
    # Path inside container for metadata
    upstream_meta = Path("/app/.upstream-sha.json")
    # In this repair-oriented flow, we attempt self-recovery of metadata if missing
    align = get_repo_alignment(MLC_CLI_PATH, upstream_meta, auto_restore=True)

    pinned_sha = align["pinned_sha"]
    current_sha = align["current_sha"]

    # ── 3. Handle Repo Exists: Ensure Alignment ─────────────────────────────
    if align["exists"]:
        # If we have a pin and it doesn't match, re-align
        # We re-align if relation is behind, ahead, or diverged.
        # Basically anything other than "match" (or "unknown" if pinning is disabled)
        if align["relation"] != "match" and pinned_sha:
            try:
                print(f"[INFO] mlc-cli exists ({align['relation']}), but pinned is {pinned_sha[:12]}. Re-aligning...")
                run_git(["fetch", "origin"], cwd=MLC_CLI_PATH)
                run_git(["checkout", pinned_sha], cwd=MLC_CLI_PATH)
                return {
                    "status": "ok",
                    "message": f"mlc-cli re-aligned (was {align['relation']}) to pinned SHA {pinned_sha[:12]}",
                    "path": str(MLC_CLI_PATH),
                    "pinned_sha": pinned_sha,
                    "previous_sha": current_sha,
                    "action": "re-aligned",
                }
            except subprocess.CalledProcessError as exc:
                return {
                    "status": "error",
                    "message": f"mlc-cli exists ({align['relation']}) but failed to re-align",
                    "path": str(MLC_CLI_PATH),
                    "stderr": exc.stderr.strip(),
                }

        return {
            "status": "ok",
            "message": "mlc-cli already exists and is aligned" if pinned_sha else "mlc-cli exists (no pinning active)",
            "path": str(MLC_CLI_PATH),
            "pinned_sha": pinned_sha,
            "current_sha": current_sha,
            "action": "already-aligned" if pinned_sha else "none",
        }

    # ── 3. Handle Repo Missing: Clone + Pin ──────────────────────────────
    print(f"[INFO] Cloning {MLC_CLI_URL} into {MLC_CLI_PATH}...")
    try:
        run_git(["clone", MLC_CLI_URL, str(MLC_CLI_PATH)])

        # Pin to the approved SHA so fresh clones use the tested version
        if pinned_sha:
            print(f"[INFO] Checking out pinned SHA {pinned_sha[:12]}...")
            run_git(["-C", str(MLC_CLI_PATH), "checkout", pinned_sha])

        return {
            "status": "ok",
            "message": f"mlc-cli cloned and pinned to {pinned_sha[:12]}" if pinned_sha else "mlc-cli cloned (HEAD)",
            "path": str(MLC_CLI_PATH),
            "pinned_sha": pinned_sha,
            "action": "cloned",
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
    """Check both the git dirty state and the alignment with pinned upstream SHA."""
    # ── 1. Alignment check (local-only foundation) ──────────────────────────
    upstream_meta = Path("/app/.upstream-sha.json")
    align = get_repo_alignment(MLC_CLI_PATH, upstream_meta)

    # ── 2. Git status check (dirtyness) ──────────────────────────────────────
    is_clean = True
    git_status_msg = "mlc-cli repo not found"
    changes = []

    if align["exists"]:
        result = run_command(["git", "status", "--porcelain"], cwd=MLC_CLI_PATH)
        if result.returncode == 0:
            status_output = result.stdout.strip()
            is_clean = len(status_output) == 0
            git_status_msg = "Repository is clean" if is_clean else "Repository has uncommitted changes"
            changes = status_output.split("\n") if status_output else []
        else:
            git_status_msg = f"failed to check git status: {result.stderr.strip()}"
            is_clean = False

    # ── 3. Alignment human messaging ─────────────────────────────────────────
    rel = align["relation"]
    pinned = align["pinned_sha"]
    current = align["current_sha"]

    if rel == "match":
        align_msg = f"Aligned with pinned SHA {pinned[:12]}"
    elif rel == "ahead":
        align_msg = f"Ahead of pinned SHA (pinned: {pinned[:12]}, local: {current[:12]})"
    elif rel == "behind":
        align_msg = f"Behind pinned SHA (pinned: {pinned[:12]}, local: {current[:12]})"
    elif rel == "diverged":
        align_msg = f"Diverged from pinned SHA {pinned[:12]}"
    elif rel == "missing":
        align_msg = "Repository missing"
    elif rel == "unpinned":
        align_msg = "No pinning active"
    else:
        align_msg = "Alignment unknown"

    # ── 4. Derive overall health status ──────────────────────────────────────
    if not align["exists"]:
        status = "missing"
    elif rel == "unknown":
        status = "unknown"
    elif is_clean and rel in ("match", "unpinned"):
        status = "healthy"
    else:
        status = "degraded"

    # ── 5. Tighten repair_possible ───────────────────────────────────────────
    # A repair/re-alignment is meaningful if we have a target pinned SHA
    # and we are not already matched.
    repair_possible = pinned is not None and rel in ("ahead", "behind", "diverged", "missing", "unpinned")

    return {
        "status": status,
        "is_clean": is_clean if align["exists"] else None,
        "message": f"{git_status_msg}. {align_msg}.",
        "alignment": {
            "pinned_sha": pinned,
            "current_sha": current,
            "relation": rel,
            "repair_possible": repair_possible,
            "message": align_msg,
        },
        "changes": changes,
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

    # ── Normalize model path: resolve relative paths against known roots ─────
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

    # ── Normalize model path: resolve relative paths against known roots ─────
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
