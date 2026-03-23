from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="FastAPI MLC-CLI")

# ── Paths ─────────────────────────────────────────────────────────────────────
MLC_CLI_PATH = Path("/workspace/mlc-cli")
MLC_CLI_URL = "https://github.com/ballinyouup/mlc-cli.git"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def run_git(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


async def stream_subprocess(command: list[str], cwd: Path | None = None) -> AsyncIterator[str]:
    """Yield stdout/stderr lines from a subprocess as SSE-formatted strings."""
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip()
        yield f"data: {line}\n\n"

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


# ── Existing endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI + MLC CLI"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/setup-check")
def setup_check():
    if not MLC_CLI_PATH.exists():
        return {
            "status": "error",
            "message": "mlc-cli repo not found",
            "repo_exists": False,
            "path": str(MLC_CLI_PATH),
        }

    git_result = run_command(["git", "remote", "get-url", "origin"], MLC_CLI_PATH)
    go_result = run_command(["go", "version"], MLC_CLI_PATH)
    conda_result = run_command(["conda", "--version"])

    return {
        "status": "ok",
        "repo_exists": True,
        "path": str(MLC_CLI_PATH),
        "origin": git_result.stdout.strip(),
        "go_version": go_result.stdout.strip(),
        "conda_version": conda_result.stdout.strip(),
        "git_returncode": git_result.returncode,
        "go_returncode": go_result.returncode,
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
    """
    Trigger `mlc-cli build` non-interactively and stream stdout/stderr back
    as Server-Sent Events (SSE).

    The client should listen with:
        curl -N -X POST http://localhost:8000/build -H 'Content-Type: application/json' \\
             -d '{"action":"install-wheels"}'
    """
    if not MLC_CLI_PATH.exists():
        async def error_stream():
            yield f"data: [ERROR] mlc-cli repo not found. Call /ensure-repo-exists first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # Build CLI arg list from request body using 'go run . build ...'
    # mlc-cli build --os linux --action <action> --cuda y --cuda-arch 86 ...
    cmd = [
        "go", "run", ".", "build",
        "--os", "linux",
        "--action", req.action,
        "--tvm-source", req.tvm_source,
        "--cuda", req.cuda,
        "--cuda-arch", req.cuda_arch,
        "--cutlass", req.cutlass,
        "--cublas", req.cublas,
        "--flash-infer", req.flash_infer,
        "--rocm", req.rocm,
        "--vulkan", req.vulkan,
        "--opencl", req.opencl,
        "--build-wheels", req.build_wheels,
        "--force-clone", req.force_clone,
    ]

    return StreamingResponse(
        stream_subprocess(cmd, cwd=MLC_CLI_PATH),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind a proxy
        },
    )
