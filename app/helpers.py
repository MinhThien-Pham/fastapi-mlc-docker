"""
app/helpers.py
~~~~~~~~~~~~~~
Pure helper functions extracted from main.py for testability.

All functions here are side-effect-free or have their side effects
(subprocess calls) well-contained so they can be mocked easily in tests.

Functions
---------
detect_known_failure              – detect known build-log failure signatures
run_tool_check                    – thin wrapper around subprocess for tool availability
build_mlc_cli_command             – construct ``go run . build`` argv list
build_quantize_command            – construct ``go run . quantize`` argv list
build_compile_command             – construct ``go run . compile`` argv list
build_run_command                 – construct ``go run . run`` argv list
resolve_model_lib                 – locate a compiled .so/.dylib for ``/run``
is_hf_model_id                    – detect Hugging Face hub identifiers for ``/quantize``
resolve_quantized_model_dir       – find a quantized artifact dir in dist/ for ``/compile``
resolve_chat_artifacts            – resolve model + model_lib shorthand for ``/chat/load``
get_supported_conv_templates      – load the CONV_TEMPLATES set from pinned mlc-llm source
infer_conv_template_for_model     – heuristic model-name → template name mapping
prepare_conv_template_for_quantize – resolve + warn before /quantize calls mlc-cli
discover_artifacts                – scan workspace for wheels, model dirs, compiled libs
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal


# ── Hugging Face model ID detection ──────────────────────────────────────────

# Prefixes that unambiguously indicate a local filesystem path, not an HF ID.
# A string starting with any of these should never be passed to HF Hub.
_LOCAL_PATH_PREFIXES: tuple[str, ...] = (
    "models/",
    "dist/",
    "./",
    "../",
    "~/",
    "~\\",
)


def is_hf_model_id(model: str) -> bool:
    """Return True if *model* looks like a Hugging Face hub identifier.

    A Hugging Face ID has the form ``Owner/ModelName``.  This function
    returns False for anything that looks like a local filesystem path
    so that relative paths such as ``models/Llama-3-8B`` are never
    accidentally forwarded to the HF Hub.

    Rejected (returns False):
    - Empty string
    - Absolute paths  (starts with ``/`` or a Windows drive letter like ``C:``)
    - Well-known local-path prefixes: ``models/``, ``dist/``, ``./``, ``../``,
      ``~/``, ``~\\``
    - Strings that already exist as a local filesystem path
    - Strings with no ``/`` at all (bare model name without owner)

    Accepted (returns True):
    - ``Owner/ModelName`` patterns that do NOT match any of the above

    Examples
    --------
    >>> is_hf_model_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    True
    >>> is_hf_model_id("meta-llama/Meta-Llama-3-8B")
    True
    >>> is_hf_model_id("/workspace/mlc-cli/models/Llama-3-8B")
    False
    >>> is_hf_model_id("models/Llama-3-8B")
    False
    >>> is_hf_model_id("dist/TinyLlama-1.1B-q4f16_1-MLC")
    False
    >>> is_hf_model_id("./local/path")
    False
    >>> is_hf_model_id("C:/models/Llama")
    False
    """
    if not model:
        return False
    # Absolute paths: Unix root or Windows drive (e.g. "C:")
    if model.startswith("/") or (len(model) > 1 and model[1] == ":"):
        return False
    # Well-known local-path prefixes
    if model.startswith(_LOCAL_PATH_PREFIXES):
        return False
    # Must contain at least one slash (Owner/ModelName)
    if "/" not in model:
        return False
    # Already exists as a local path
    if Path(model).exists():
        return False
    # Looks like an HF hub identifier
    return True


# ── Quantized artifact resolution (for /compile) ──────────────────────────────

def resolve_quantized_model_dir(
    mlc_cli_path: Path,
    model: str,
    quant: str,
) -> "Path | Literal['none'] | Literal['multiple']":
    """Resolve *model* to a quantized artifact directory inside ``dist/``.

    Accepts four forms of *model*:

    1. **Exact existing path** — an absolute path or a relative path that
       exists under *mlc_cli_path*.  Returned unchanged (as a ``Path``).
    2. **Artifact folder name** — the literal name of a directory under
       ``dist/`` (e.g. ``TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC``).
    3. **Short model name** — the base part of the model name, used to
       search ``dist/`` for ``*<stem>*<quant>*MLC`` directories.
    4. **Hugging Face hub ID** — ``Owner/ModelName``.  The basename after
       the last ``/`` is used as the search stem.

    Returns
    -------
    Path
        Resolved absolute path to the single matching artifact directory.
    ``"none"``
        Sentinel: no matching artifact found — caller should tell the user
        to run ``/quantize`` first.
    ``"multiple"``
        Sentinel: more than one candidate matches — caller should tell the
        user to pass an exact artifact path from ``GET /artifacts``.
    """
    # ── Case 1: exact path that already exists ───────────────────────────────
    p = Path(model)
    if p.is_absolute() and p.is_dir():
        return p
    if not p.is_absolute():
        candidate = (mlc_cli_path / p).resolve()
        if candidate.is_dir():
            return candidate

    # ── Derive stem for fuzzy search ──────────────────────────────────────
    # For HF IDs like "Owner/ModelName" use the basename; otherwise use model as-is.
    stem = model.split("/")[-1] if "/" in model else model

    # ── Case 2/3/4: scan dist/ for matching quantized dirs ──────────────────
    dist_dir = mlc_cli_path / "dist"
    if not dist_dir.is_dir():
        return "none"

    stem_lower = stem.lower()
    quant_lower = quant.lower()
    matches: list[Path] = []
    for entry in dist_dir.iterdir():
        if not entry.is_dir():
            continue
        name_lower = entry.name.lower()
        # Must contain the stem, the quant string, and end with "-mlc"
        if stem_lower in name_lower and quant_lower in name_lower and name_lower.endswith("-mlc"):
            # Confirm it is a real quantized artifact (has mlc-chat-config.json)
            if (entry / "mlc-chat-config.json").exists():
                matches.append(entry)

    if len(matches) == 0:
        return "none"
    if len(matches) > 1:
        return "multiple"
    return matches[0]


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


# ── Conv-template helpers ─────────────────────────────────────────────────────

# Fallback list derived from mlc-llm gen_config.py CONV_TEMPLATES set at the
# pinned MLC_LLM_REF=2008fe8343e1f40ef89ee57b9287aebcf1b86c98.
# Update when the pinned ref is bumped.
_FALLBACK_CONV_TEMPLATES: frozenset[str] = frozenset({
    "LM",
    "aya-23",
    "chatml",
    "chatml_nosystem",
    "codellama_completion",
    "codellama_instruct",
    "deepseek",
    "deepseek_r1_llama",
    "deepseek_r1_qwen",
    "deepseek_v2",
    "deepseek_v3",
    "dolly",
    "gemma_instruction",
    "gemma3_instruction",
    "glm",
    "gorilla",
    "gorilla-openfunctions-v2",
    "gpt2",
    "gpt_bigcode",
    "hermes2_pro_llama3",
    "hermes3_llama-3_1",
    "llama-2",
    "llama-3",
    "llama-3_1",
    "llama-4",
    "llama_default",
    "llava",
    "llm-jp",
    "ministral3",
    "ministral3_reasoning",
    "mistral_default",
    "nemotron",
    "neural_hermes_mistral",
    "oasst",
    "olmo",
    "olmo2",
    "open_hermes_mistral",
    "orion",
    "phi-2",
    "phi-3",
    "phi-3-vision",
    "phi-4",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "qwen3_5_nothink",
    "redpajama_chat",
    "rwkv_world",
    "stablelm",
    "stablelm-2",
    "stablelm-3b",
    "tinyllama_v1_0",
    "wizard_coder_or_math",
    "wizardlm_7b",
})


def get_supported_conv_templates(mlc_cli_path: Path) -> frozenset[str]:
    """Return the set of supported conv_template names from the pinned runtime.

    Preferred source: parse the CONV_TEMPLATES set from
    ``<mlc_cli_path>/mlc-llm/python/mlc_llm/interface/gen_config.py``
    using the ``ast`` module (no import of mlc_llm required).

    Falls back to the hardcoded ``_FALLBACK_CONV_TEMPLATES`` set if the file
    cannot be found or parsed.
    """
    gen_config_py = (
        mlc_cli_path / "mlc-llm" / "python" / "mlc_llm" / "interface" / "gen_config.py"
    )
    if not gen_config_py.is_file():
        return _FALLBACK_CONV_TEMPLATES

    try:
        source = gen_config_py.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(gen_config_py))
    except (OSError, SyntaxError):
        return _FALLBACK_CONV_TEMPLATES

    for node in ast.walk(tree):
        # Look for:  CONV_TEMPLATES = { ... }
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets
        if len(targets) != 1:
            continue
        if not (isinstance(targets[0], ast.Name) and targets[0].id == "CONV_TEMPLATES"):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (set, frozenset)):
            return frozenset(str(v) for v in value)

    return _FALLBACK_CONV_TEMPLATES


# Models whose names match these patterns CANNOT be auto-inferred safely.
# They appear to require custom (gpt-oss / harmony) templates that are not
# registered in the current runtime.
_UNSUPPORTED_AUTO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bgpt.?oss\b", re.IGNORECASE),
    re.compile(r"\bharmon[yi]\b", re.IGNORECASE),
]


def infer_conv_template_for_model(
    model: str,
    supported_templates: frozenset[str],
) -> str | None:
    """Heuristically infer a supported conv_template from a model identifier.

    Matches against the Hugging Face model ID or local path basename.
    Checks that the inferred template actually exists in *supported_templates*.

    Returns the template name if found, or ``None`` if no confident match.
    """
    # Normalise: use the basename of a HF ID or local path, lower-cased.
    name = model.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    # Ordered: more-specific patterns first so they match before shorter ones.
    _RULES: list[tuple[re.Pattern[str], str]] = [
        # TinyLlama
        (re.compile(r"tinyllama"),                         "tinyllama_v1_0"),
        # Hermes variants (must come before plain llama)
        (re.compile(r"hermes.*3.*llama.*3.*1|hermes.*3_1.*llama"), "hermes3_llama-3_1"),
        (re.compile(r"hermes.*2.*llama.*3|hermes.*2.*pro.*llama"), "hermes2_pro_llama3"),
        (re.compile(r"openhermes.*mistral|open_hermes.*mistral"), "open_hermes_mistral"),
        (re.compile(r"neuralhermes.*mistral|neural_hermes.*mistral"), "neural_hermes_mistral"),
        # Nemotron (must come before llama-3_1 since some Nemotron names include Llama-3.1)
        (re.compile(r"nemotron"),                          "nemotron"),
        # DeepSeek — most-specific first
        (re.compile(r"deepseek.*r1.*llama"),               "deepseek_r1_llama"),
        (re.compile(r"deepseek.*r1.*qwen"),                "deepseek_r1_qwen"),
        (re.compile(r"deepseek.*v3"),                      "deepseek_v3"),
        (re.compile(r"deepseek.*v2"),                      "deepseek_v2"),
        (re.compile(r"deepseek"),                          "deepseek"),
        # Llama family
        (re.compile(r"llama.*4"),                          "llama-4"),
        (re.compile(r"llama.*3.*1|llama.*3_1"),            "llama-3_1"),
        (re.compile(r"llama.*3"),                          "llama-3"),
        (re.compile(r"llama.*2"),                          "llama-2"),
        # Mistral / Ministral
        (re.compile(r"ministral.*3"),                      "ministral3"),
        (re.compile(r"mistral|mixtral"),                   "mistral_default"),
        # Phi — exact version digits. After normalisation, separators become "_".
        # We match the version digit and ensure the NEXT character (if any) is not
        # also a digit, so phi_4 does not accidentally match phi_40 or phi_3 match phi_4.
        (re.compile(r"phi[_]*4(?:\D|$)"),                  "phi-4"),
        (re.compile(r"phi[_]*3(?:\D|$)"),                  "phi-3"),
        (re.compile(r"phi[_]*2(?:\D|$)"),                  "phi-2"),
        # Gemma
        (re.compile(r"gemma.*3|gemma3"),                   "gemma3_instruction"),
        (re.compile(r"gemma"),                             "gemma_instruction"),
        # Qwen — most-specific first
        (re.compile(r"qwen.*3.*5|qwen3_5"),               "qwen3_5"),
        (re.compile(r"qwen.*3"),                           "qwen3"),
        (re.compile(r"qwen.*2.*5|qwen.*2"),               "qwen2"),
    ]

    for pattern, template in _RULES:
        if pattern.search(name):
            # Only return the template if the current runtime actually supports it.
            if template in supported_templates:
                return template
    return None


def prepare_conv_template_for_quantize(
    model: str,
    requested_template: str,
    mlc_cli_path: Path,
) -> tuple[str, list[str]]:
    """Resolve the conv_template for a /quantize request.

    Parameters
    ----------
    model:
        The model identifier (HF ID or local path) from the request.
    requested_template:
        The ``conv_template`` field value from the request.
        ``"auto"`` triggers inference; anything else is an explicit choice.
    mlc_cli_path:
        Path to the mlc-cli workspace (used to load the live template list).

    Returns
    -------
    (resolved_template, messages)
        ``resolved_template`` is the template string to pass to mlc-cli.
        ``messages`` is a list of ``data: [INFO/WARNING] ...\\n\\n`` SSE lines
        to emit *before* mlc-cli output.

    Raises
    ------
    ValueError
        Only in ``"auto"`` mode when no template can be safely inferred.
    """
    supported = get_supported_conv_templates(mlc_cli_path)
    messages: list[str] = []

    if requested_template == "auto":
        # Block known-unsupported model families immediately.
        model_lower = model.lower()
        for pat in _UNSUPPORTED_AUTO_PATTERNS:
            if pat.search(model_lower):
                raise ValueError(
                    f"Could not infer a supported conv_template for '{model}'. "
                    "This model appears to require a custom template such as Harmony "
                    "or gpt-oss, but the current runtime does not list harmony/gpt-oss "
                    "as supported. Pass conv_template explicitly only if you intentionally "
                    "want to experiment."
                )

        inferred = infer_conv_template_for_model(model, supported)
        if inferred is None:
            raise ValueError(
                f"Could not infer conv_template for '{model}'. "
                "Pass conv_template explicitly. "
                "See GET /conv-templates for the supported list."
            )

        messages.append(
            f"data: [INFO] Auto-selected conv_template='{inferred}' "
            f"for model '{model}'.\n\n"
        )
        return inferred, messages

    # Explicit template path.
    # Warn if the template is not in the known supported list.
    if requested_template not in supported:
        messages.append(
            f"data: [WARNING] conv_template='{requested_template}' was not found in the "
            "current runtime template list. mlc-cli may warn or produce a broken chat "
            "config.\n\n"
        )
        return requested_template, messages

    # Template is known — optionally warn on likely mismatch.
    inferred = infer_conv_template_for_model(model, supported)
    if inferred is not None and inferred != requested_template:
        messages.append(
            f"data: [WARNING] Model '{model}' usually maps to "
            f"conv_template='{inferred}', but request used "
            f"'{requested_template}'. Behavior may be incorrect.\n\n"
        )

    return requested_template, messages
