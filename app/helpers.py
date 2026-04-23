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
from typing import Any, Literal


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

def try_restore_metadata(metadata_path: Path) -> bool:
    """Attempt to restore the metadata file from the local git repo if missing or malformed.

    Returns True if the file exists and is valid JSON after the attempt.
    """
    needs_restore = False

    if not metadata_path.is_file():
        needs_restore = True
    else:
        try:
            json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, IOError, Exception):
            needs_restore = True

    if needs_restore:
        try:
            # Try to restore from git.
            subprocess.run(
                ["git", "checkout", "--", metadata_path.name],
                cwd=metadata_path.parent,
                capture_output=True,
                check=True
            )
        except Exception:
            pass

    # Final validation
    if not metadata_path.is_file():
        return False
    try:
        json.loads(metadata_path.read_text())
        return True
    except Exception:
        return False


def get_repo_alignment(repo_path: Path, metadata_path: Path, auto_restore: bool = False) -> dict[str, Any]:
    """Determine the relationship between the local repo and the pinned metadata.

    This is a local-only inspection (no network fetch).

    Returns
    -------
    dict with keys:
        exists (bool)      – True if repo_path exists
        pinned_sha (str)   – SHA from metadata_path (or None)
        current_sha (str)  – HEAD SHA from repo (or None)
        relation (str)     – "match" | "ahead" | "behind" | "diverged" | "missing" | "unpinned" | "unknown"
    """
    res: dict[str, Any] = {
        "exists": repo_path.exists(),
        "pinned_sha": None,
        "current_sha": None,
        "relation": "unknown",
    }

    # 1. Read pinned SHA from metadata (with optional self-recovery)
    if auto_restore:
        try_restore_metadata(metadata_path)

    if metadata_path.is_file():
        try:
            res["pinned_sha"] = json.loads(metadata_path.read_text()).get("pinned_sha")
        except Exception:
            # Exists but unreadable: if we already tried restore and failed, this is unknown
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

    # 4. Check ancestry
    # Is pinned an ancestor of current? (ahead)
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", pinned, current],
        cwd=repo_path,
        capture_output=True
    )
    if r.returncode == 0:
        res["relation"] = "ahead"
        return res

    # Is current an ancestor of pinned? (behind)
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", current, pinned],
        cwd=repo_path,
        capture_output=True
    )
    if r.returncode == 0:
        res["relation"] = "behind"
        return res

    # Neither is an ancestor of the other
    res["relation"] = "diverged"
    return res


def get_startup_alignment_message(align: dict[str, Any]) -> str:
    """Translate repo alignment into a clear human-readable startup log message."""
    rel = align["relation"]
    pinned = align["pinned_sha"]
    current = align["current_sha"]

    if rel == "match":
        return f"Bryan mlc-cli repo is aligned with pinned SHA {pinned[:12]}"
    elif rel == "ahead":
        return (f"Bryan mlc-cli repo is AHEAD of pinned SHA (pinned: {pinned[:12]}, local: {current[:12]}). "
                "Recommend running 'verify_upstream.py' if this was intentional.")
    elif rel == "behind":
        return (f"Bryan mlc-cli repo is BEHIND pinned SHA {pinned[:12]} (local: {current[:12]}). "
                "Use POST /ensure-repo-exists to re-align.")
    elif rel == "diverged":
        return (f"Bryan mlc-cli repo has DIVERGED from pinned SHA {pinned[:12]}. "
                "Recommend manual inspection or repair via POST /ensure-repo-exists.")
    elif rel == "missing":
        return "Bryan mlc-cli repo is MISSING. Use POST /ensure-repo-exists to clone and align it."
    elif rel == "unpinned":
        return "Bryan mlc-cli repo exists but no pinning metadata is active."
    else:
        return "Bryan mlc-cli repo status is UNKNOWN (failed to inspect Git state)."


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
