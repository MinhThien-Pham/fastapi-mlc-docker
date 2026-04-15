#!/usr/bin/env python3
"""
Lightweight upstream CLI contract check for mlc-cli.

Validates that the interface boundary between our FastAPI helpers
(app/helpers.py) and Bryan's upstream mlc-cli Go CLI has not changed.

Checks:
  1. Go code compiles (go build)
  2. Expected subcommand routing exists
  3. Expected CLI flags still exist in each subcommand function
  4. Expected shell scripts still exist

Does NOT check runtime behavior (needs GPU/Docker/models).

Usage:
    python check_cli_contract.py <path-to-mlc-cli-repo>

Exit codes:
    0 — all checks pass
    1 — one or more checks failed (upstream interface changed)
    2 — inconclusive (source missing, parse error, etc.)
"""

import subprocess
import sys
from pathlib import Path

# ── Contract: flags our app/helpers.py emits for each subcommand ─────────────

CONTRACT = {
    "build": {
        "function": "runBuildCmd",
        "flags": [
            "os", "action", "tvm-source", "cuda", "cuda-arch",
            "cutlass", "cublas", "flash-infer", "rocm", "vulkan",
            "opencl", "build-wheels", "force-clone",
        ],
    },
    "quantize": {
        "function": "runQuantizeCmd",
        "flags": ["os", "model", "quant", "device", "template", "output"],
    },
    "compile": {
        "function": "runCompileCmd",
        "flags": ["os", "model", "quant", "device", "output"],
    },
    "run": {
        "function": "runRunCmd",
        "flags": ["os", "model-name", "model-url", "device", "profile", "model-lib"],
    },
}

REQUIRED_SCRIPTS = [
    "scripts/linux_run_model.sh",
    "scripts/linux_compile_model.sh",
]

REQUIRED_SUBCOMMANDS = ["build", "quantize", "compile", "run"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_function_body(source: str, func_name: str) -> str | None:
    """Extract text from a top-level Go function to the next top-level func."""
    marker = f"func {func_name}("
    start = source.find(marker)
    if start == -1:
        return None
    next_func = source.find("\nfunc ", start + len(marker))
    if next_func == -1:
        return source[start:]
    return source[start:next_func]


def check_go_build(repo_path: Path) -> tuple[bool, str]:
    """Verify that the Go code compiles."""
    try:
        result = subprocess.run(
            ["go", "build", "-o", "/dev/null", "."],
            cwd=repo_path, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return True, "go build succeeded"
        return False, f"go build failed:\n{result.stderr.strip()}"
    except FileNotFoundError:
        return False, "go not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "go build timed out (120s)"


def check_subcommands(source: str) -> list[str]:
    """Check that expected subcommand case labels exist in routing switch."""
    return [sub for sub in REQUIRED_SUBCOMMANDS if f'case "{sub}"' not in source]


def check_flags(source: str) -> dict[str, list[str]]:
    """Check each subcommand function contains its expected flag definitions."""
    missing_by_cmd: dict[str, list[str]] = {}
    for cmd, spec in CONTRACT.items():
        func_body = extract_function_body(source, spec["function"])
        if func_body is None:
            missing_by_cmd[cmd] = [f"(function {spec['function']} not found)"]
            continue
        missing = [f for f in spec["flags"] if f'"{f}"' not in func_body]
        if missing:
            missing_by_cmd[cmd] = missing
    return missing_by_cmd


def check_scripts(repo_path: Path) -> list[str]:
    """Check that required shell scripts exist."""
    return [s for s in REQUIRED_SCRIPTS if not (repo_path / s).is_file()]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-mlc-cli-repo>")
        sys.exit(2)

    repo_path = Path(sys.argv[1])
    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory")
        sys.exit(2)

    main_go = repo_path / "main.go"
    if not main_go.is_file():
        print(f"Error: {main_go} not found — repo structure may have changed")
        sys.exit(2)

    source = main_go.read_text()
    failures: list[str] = []

    # 1. Go compilation
    print("1. Checking Go compilation...")
    ok, msg = check_go_build(repo_path)
    print(f"   {'✓' if ok else '✗'} {msg}")
    if not ok:
        failures.append(f"Go build: {msg}")

    # 2. Subcommand routing
    print("2. Checking subcommand routing...")
    missing_subs = check_subcommands(source)
    if not missing_subs:
        print(f"   ✓ All subcommands present: {', '.join(REQUIRED_SUBCOMMANDS)}")
    else:
        msg = f"Missing subcommands: {', '.join(missing_subs)}"
        print(f"   ✗ {msg}")
        failures.append(msg)

    # 3. CLI flag definitions
    print("3. Checking CLI flag definitions...")
    missing_flags = check_flags(source)
    if not missing_flags:
        total = sum(len(spec["flags"]) for spec in CONTRACT.values())
        print(f"   ✓ All {total} expected flags found across {len(CONTRACT)} subcommands")
    else:
        for cmd, flags in missing_flags.items():
            msg = f"  {cmd}: missing {', '.join(flags)}"
            print(f"   ✗{msg}")
            failures.append(f"Flags in '{cmd}': missing {', '.join(flags)}")

    # 4. Shell scripts
    print("4. Checking required shell scripts...")
    missing_scripts = check_scripts(repo_path)
    if not missing_scripts:
        print(f"   ✓ All required scripts present")
    else:
        msg = f"Missing scripts: {', '.join(missing_scripts)}"
        print(f"   ✗ {msg}")
        failures.append(msg)

    # Summary
    print()
    if not failures:
        print("=== Contract Check PASSED ===")
        sys.exit(0)
    else:
        print("=== Contract Check FAILED ===")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
