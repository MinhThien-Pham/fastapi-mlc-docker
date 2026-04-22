"""
app/helpers.py
~~~~~~~~~~~~~~
Pure helper functions extracted from main.py for testability.

All functions here are side-effect-free or have their side effects
(subprocess calls) well-contained so they can be mocked easily in tests.

Functions
---------
detect_known_failure   – detect known build-log failure signatures
run_tool_check         – thin wrapper around subprocess for tool availability
get_repo_alignment     – local-only check of current vs pinned SHA
build_mlc_cli_command  – construct ``go run . build`` argv list
build_convert_command  – construct ``go run . quantize`` argv list
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


# ── Known build-failure signatures ────────────────────────────────────────────

# Strings that, when found (case-insensitively) in a build log line,
# indicate a cutlass / flash-attn related failure.
KNOWN_FAILURE_SIGNATURES: list[str] = [
    "flash_attn",
    "libflash_attn",
    "FlashAttention",
    "cutlass",
]

CUTLASS_RETRY_HINT: str = (
    "This looks like a cutlass / flash-attn build failure.\n"
    "Retry with cutlass and flash_infer disabled:\n"
    "\n"
    '  curl -N -X POST http://localhost:8000/build \\\n'
    '       -H \'Content-Type: application/json\' \\\n'
    '       -d \'{"action":"full","cutlass":"n","flash_infer":"n"}\''
)


def detect_known_failure(line: str) -> str | None:
    """Return a hint string if *line* matches a known build-failure signature.

    Returns ``None`` when no known signature is found.
    The check is case-insensitive so it catches log lines written in any casing.

    Examples
    --------
    >>> detect_known_failure("error: flash_attn module not found")
    '...'
    >>> detect_known_failure("Build succeeded.") is None
    True
    """
    lower = line.lower()
    if any(sig.lower() in lower for sig in KNOWN_FAILURE_SIGNATURES):
        return CUTLASS_RETRY_HINT
    return None


# ── Tool / command availability checks ───────────────────────────────────────

def run_tool_check(command: list[str]) -> dict[str, Any]:
    """Run *command* and return a structured availability dict.

    Never raises — ``FileNotFoundError`` (tool not on PATH) and
    ``subprocess.TimeoutExpired`` are both caught and surfaced as structured
    data so callers can handle them uniformly.

    Returns
    -------
    dict with keys:
        available (bool)  – True iff returncode == 0
        output    (str)   – stdout if non-empty, otherwise stderr
        returncode (int)  – process exit code, or -1 on error
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return {
            "available": result.returncode == 0,
            "output": output,
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {"available": False, "output": "command not found", "returncode": -1}
    except subprocess.TimeoutExpired:
        return {"available": False, "output": "timed out", "returncode": -1}


# ── Repo Alignment logic ──────────────────────────────────────────────────────

def get_repo_alignment(repo_path: Path, metadata_path: Path) -> dict[str, Any]:
    """Determine the relationship between the local repo and the pinned metadata.

    This is a local-only inspection (no network fetch).

    Returns
    -------
    dict with keys:
        exists (bool)      – True if repo_path exists
        pinned_sha (str)   – SHA from metadata_path (or None)
        current_sha (str)  – HEAD SHA from repo (or None)
        relation (str)     – "match" | "ahead" | "behind" | "diverged" | "missing" | "unknown"
    """
    res: dict[str, Any] = {
        "exists": repo_path.exists(),
        "pinned_sha": None,
        "current_sha": None,
        "relation": "unknown",
    }

    # 1. Read pinned SHA from metadata
    if metadata_path.is_file():
        try:
            res["pinned_sha"] = json.loads(metadata_path.read_text()).get("pinned_sha")
        except Exception:
            pass

    if not res["exists"]:
        res["relation"] = "missing"
        return res

    # 2. Get current HEAD SHA
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    if r.returncode != 0:
        # Not a git repo or other git error: this is a true unknown
        res["relation"] = "unknown"
        return res
    res["current_sha"] = r.stdout.strip()

    pinned = res["pinned_sha"]
    current = res["current_sha"]

    # If we have a repo but no pinned SHA to compare against
    if not pinned:
        res["relation"] = "unpinned"
        return res

    if not current:
        res["relation"] = "unknown"
        return res

    if pinned == current:
        res["relation"] = "match"
        return res

    # 3. Determine ancestry (local-only)
    def is_ancestor(a: str, b: str) -> bool:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", a, b],
            cwd=repo_path
        ).returncode == 0

    try:
        # Is pinned an ancestor of current? (current is ahead of pinned)
        if is_ancestor(pinned, current):
            res["relation"] = "ahead"
            return res

        # Is current an ancestor of pinned? (current is behind pinned)
        if is_ancestor(current, pinned):
            res["relation"] = "behind"
            return res

        # Neither is ancestor -> diverged
        res["relation"] = "diverged"
    except Exception:
        # e.g. pinned SHA not present in local history
        pass

    return res


# ── Build command construction ────────────────────────────────────────────────

def build_mlc_cli_command(req: Any) -> list[str]:
    """Translate a *BuildRequest* into the ``go run . build`` argument list.

    Keeping this logic here (rather than inline in the route handler) makes
    it trivial to unit-test without spinning up a full FastAPI app.
    """
    return [
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



# ── Quantize command construction ─────────────────────────────────────────────

def build_quantize_command(req: Any) -> list[str]:
    """Translate a *QuantizeRequest* into the ``go run . quantize`` argument list.

    This wraps the mlc-cli ``quantize`` sub-command which runs two steps:

    1. ``mlc_llm convert_weight`` — convert raw Hugging Face weights to MLC
       format and apply quantization.
    2. ``mlc_llm gen_config``     — write the runtime config alongside the
       converted weights.

    Keeping this logic here (rather than inline in the route handler) makes
    it trivial to unit-test without spinning up a full FastAPI app.

    Assumption: ``output`` defaults to ``dist/<model_basename>-<quant>-MLC``
    when not supplied — this mirrors what ``mlc-cli quantize`` does
    internally when ``--output`` is omitted.
    """
    cmd = [
        "go", "run", ".", "quantize",
        "--os",     "linux",
        "--model",  req.model,
        "--quant",  req.quant,
        "--device", req.device,
        "--template", req.conv_template,
    ]
    if req.output:
        cmd.extend(["--output", req.output])
    return cmd


# ── Compile command construction ──────────────────────────────────────────────

def build_compile_command(req: Any) -> list[str]:
    """Translate a *CompileRequest* into the ``go run . compile`` argument list.

    This wraps the mlc-cli ``compile`` sub-command.
    """
    cmd = [
        "go", "run", ".", "compile",
        "--os",     "linux",
        "--model",  req.model,
        "--quant",  req.quant,
        "--device", req.device,
    ]
    if req.output:
        cmd.extend(["--output", req.output])
    return cmd


# ── Run command construction ──────────────────────────────────────────────────

def build_run_command(req: Any) -> list[str]:
    """Translate a *RunRequest* into the ``go run . run`` argument list.

    This wraps the mlc-cli ``run`` sub-command. Note that upstream is interactive
    and does not support a ``--prompt`` flag. When run without stdin, it acts as
    a load-test.
    """
    cmd = [
        "go", "run", ".", "run",
        "--os", "linux",
        "--model-name", req.model_name,
        "--device", req.device,
        "--profile", req.profile,
    ]
    if req.model_url:
        cmd.extend(["--model-url", req.model_url])
    if req.model_lib:
        cmd.extend(["--model-lib", req.model_lib])
    return cmd


# ── Artifact discovery ────────────────────────────────────────────────────────

def discover_artifacts(base_path: Path) -> list[dict]:
    """Scan base_path for wheels, converted models, and compiled libraries.

    Returns a list of dicts with:
    - type: "wheel" | "model_dir" | "compiled_lib"
    - name: file or folder name
    - path: relative path string
    - source_step: "build" | "convert" | "compile"
    - size_bytes: int
    - modified_time: float
    """
    artifacts = []
    
    if not base_path.exists() or not base_path.is_dir():
        return artifacts

    # 1. Look for wheels (build step)
    for whl in base_path.rglob("*.whl"):
        if "node_modules" in whl.parts or ".git" in whl.parts:
            continue
        try:
            stat = whl.stat()
            artifacts.append({
                "type": "wheel",
                "name": whl.name,
                "path": str(whl.relative_to(base_path)),
                "source_step": "build",
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
            })
        except OSError:
            pass
        
    # 2. Look for converted models (quantize step)
    for config in base_path.rglob("mlc-chat-config.json"):
        if "node_modules" in config.parts or ".git" in config.parts:
            continue
        p = config.parent
        try:
            stat = p.stat()
            # Calculate total size of the directory
            total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            artifacts.append({
                "type": "model_dir",
                "name": p.name,
                "path": str(p.relative_to(base_path)),
                "source_step": "quantize",
                "size_bytes": total_size,
                "modified_time": stat.st_mtime,
            })
        except OSError:
            pass

    # 3. Look for compiled libraries (compile step)
    for ext in ("*.so", "*.dylib", "*.dll"):
        for lib in base_path.rglob(ext):
            if "node_modules" in lib.parts or ".git" in lib.parts:
                continue
            try:
                stat = lib.stat()
                artifacts.append({
                    "type": "compiled_lib",
                    "name": lib.name,
                    "path": str(lib.relative_to(base_path)),
                    "source_step": "compile",
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                })
            except OSError:
                pass

    return artifacts
