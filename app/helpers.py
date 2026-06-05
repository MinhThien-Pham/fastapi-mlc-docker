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


def resolve_model_lib(
    mlc_cli_path: Path,
    model_name: str,
    quant: str,
    device: str,
) -> str | None:
    """Attempt to resolve a pre-compiled model library from dist/libs/.

    Looks for files matching the pattern::

        dist/libs/<model_name>-<quant>-<device>.so
        dist/libs/<model_name>-<quant>-<device>.dylib   (macOS)

    Returns
    -------
    str
        Absolute path to the library if exactly one match is found.
    ``"multiple"``
        Sentinel string when more than one candidate exists; caller should
        ask the user to pass ``model_lib`` explicitly.
    ``None``
        When no matching library is found; caller should fall back to JIT.
    """
    libs_dir = mlc_cli_path / "dist" / "libs"
    if not libs_dir.is_dir():
        return None

    stem = f"{model_name}-{quant}-{device}"
    matches: list[Path] = []
    for ext in (".so", ".dylib"):
        candidate = libs_dir / f"{stem}{ext}"
        if candidate.is_file():
            matches.append(candidate)

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        return "multiple"
    return str(matches[0])




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


def resolve_chat_artifacts(
    mlc_cli_path: Path,
    model: str,
    model_name: str,
    quant: str,
    device: str,
) -> tuple[str, str]:
    """Resolve model shorthand to explicit model and model_lib paths.
    
    Returns:
        (resolved_model_path, resolved_model_lib_path)
    
    Raises:
        ValueError with clear error message if resolution fails.
    """
    target = model_name if model_name else model
    if not target:
        raise ValueError("Must provide 'model' or 'model_name'.")

    dist_dir = mlc_cli_path / "dist"
    base_device = device.split(":")[0] if ":" in device else device

    candidate_dirs: list[Path] = []
    
    # 1. Check if target is a direct path to an existing artifact directory
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = mlc_cli_path / target
        
    if target_path.is_dir() and target_path.name.endswith("-MLC"):
        candidate_dirs.append(target_path)
    else:
        # 2. Fallback to shorthand / HF ID search
        search_term = target
        if "/" in search_term and not search_term.startswith("dist/"):
            search_term = search_term.split("/")[-1]

        if search_term.endswith("-MLC"):
            exact_dir = dist_dir / search_term
            if exact_dir.is_dir():
                candidate_dirs.append(exact_dir)
                
        if not candidate_dirs and dist_dir.is_dir():
            for d in dist_dir.iterdir():
                if d.is_dir() and d.name.startswith(search_term) and d.name.endswith("-MLC"):
                    candidate_dirs.append(d)
                    
    candidate_dirs.sort()
                
    if not candidate_dirs:
        raise ValueError(f"No compiled MLC artifact found for '{target}'. Run /quantize and /compile first, then try /chat/load again.")

    if len(candidate_dirs) > 1:
        quant_matches = [d for d in candidate_dirs if f"-{quant}-" in d.name or d.name.endswith(f"-{quant}-MLC")]
        if quant_matches:
            candidate_dirs = quant_matches
            
    if len(candidate_dirs) > 1:
        candidates_str = ", ".join(d.name for d in candidate_dirs)
        raise ValueError(f"Multiple artifact candidates found for '{target}': {candidates_str}. Please specify 'model_name' or 'quant' to disambiguate.")

    resolved_model_dir = candidate_dirs[0]
    
    libs_dir = mlc_cli_path / "dist" / "libs"
    lib_matches: list[Path] = []
    if libs_dir.is_dir():
        for ext in (".so", ".dylib"):
            for candidate in libs_dir.glob(f"{resolved_model_dir.name}-*-{base_device}{ext}"):
                lib_matches.append(candidate)
    
    lib_matches.sort()
                
    if not lib_matches:
        raise ValueError(f"Found model artifact directory, but no compiled library was found. Run /compile first. (Expected lib under dist/libs/{resolved_model_dir.name}-*-{base_device}.so or .dylib)")
        
    if len(lib_matches) > 1:
        quant_matches = [p for p in lib_matches if f"-{quant}-" in p.name]
        if quant_matches:
            lib_matches = quant_matches
            
    if len(lib_matches) > 1:
        raise ValueError(f"Multiple compiled libraries found for '{resolved_model_dir.name}' and device '{base_device}'. Please specify 'quant' to disambiguate.")

    return str(resolved_model_dir), str(lib_matches[0])
