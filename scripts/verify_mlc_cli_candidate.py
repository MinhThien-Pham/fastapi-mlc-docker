#!/usr/bin/env python3
"""
scripts/verify_mlc_cli_candidate.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Local-dev tool for safely testing a newer mlc-cli ref before updating
docker/mlc-cli.lock.

Usage:
    python scripts/verify_mlc_cli_candidate.py           # full candidate flow
    python scripts/verify_mlc_cli_candidate.py finalize  # re-open post-pass prompts
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "docker" / "mlc-cli.lock"
STATE_DIR = REPO_ROOT / ".mlc-cli-verify"
STATE_FILE = STATE_DIR / "last-pass.json"

SMOKE_SCRIPT = REPO_ROOT / "tests" / "integration" / "test_smoke.py"
FULL_SCRIPT = REPO_ROOT / "tests" / "integration" / "test_full_pipeline.py"

MAIN_IMAGE_TAG = "fastapi-mlc-docker-web:latest"
MAIN_VOLUME = os.environ.get("MAIN_VOLUME", "fastapi-mlc-docker_mlc_workspace")

HEALTH_TIMEOUT_SECONDS = 120
HEALTH_POLL_INTERVAL = 2


# ── Helpers ────────────────────────────────────────────────────────────────────

def die(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], *, check: bool = True, capture: bool = False,
        env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command, streaming output unless capture=True."""
    merged_env = {**os.environ, **(env or {})}
    if capture:
        return subprocess.run(
            cmd, check=check, text=True, capture_output=True,
            env=merged_env, cwd=cwd,
        )
    return subprocess.run(cmd, check=check, env=merged_env, cwd=cwd)


def ask(prompt: str, help_text: str, default: bool) -> bool:
    """Ask a y/n question. '?' prints help. Returns bool."""
    default_hint = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(f"\n{prompt} [{default_hint}/?] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if raw in ("", "\n"):
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw == "?":
            print(f"\n{help_text}\n")
        else:
            print("  Please enter y, n, or ? for help.")


def ask_choice(title: str, choices: list[tuple[str, str]], default: int) -> int:
    """
    Ask the user to choose an option by number.
    '?' prints the detailed explanations.
    """
    while True:
        print(f"\n{title}")
        for i, (short_text, _) in enumerate(choices, start=1):
            print(f"  {i}. {short_text}")
        
        try:
            raw = input(f"Choice [{default}/?]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
            
        if raw in ("", "\n"):
            return default
            
        if raw == "?":
            print("\n  --- Explanations ---")
            for i, (_, help_text) in enumerate(choices, start=1):
                print(f"  Option {i}:")
                for line in help_text.strip().splitlines():
                    print(f"    {line}")
            print("  --------------------")
            continue
            
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(choices):
                return choice
                
        print(f"  Please enter a number between 1 and {len(choices)}, or ? for help.")


def read_lock_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE pairs from docker/mlc-cli.lock."""
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def write_lock_file(path: Path, repo: str, ref: str) -> None:
    """Write MLC_CLI_REPO and MLC_CLI_REF to docker/mlc-cli.lock."""
    path.write_text(f"MLC_CLI_REPO={repo}\nMLC_CLI_REF={ref}\n")


def fetch_remote_head(repo_url: str) -> str:
    """Return the current HEAD SHA of a remote git repo (no clone required)."""
    print(f"  Fetching HEAD of {repo_url} ...")
    result = run(
        ["git", "ls-remote", repo_url, "HEAD"],
        capture=True, check=True,
    )
    line = result.stdout.strip().split("\n")[0]
    sha = line.split()[0]
    if len(sha) != 40:
        die(f"Unexpected SHA format from git ls-remote: {sha!r}")
    return sha


def find_free_port(preferred: int = 8001) -> int:
    """Return preferred port if free, otherwise find another free port."""
    for port in [preferred, *range(8002, 8100)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    die("Could not find a free local port in range 8001-8100.")


def wait_for_health(api_url: str, timeout: int = HEALTH_TIMEOUT_SECONDS) -> bool:
    """Poll /health until it returns 200 or timeout expires. Returns True on success."""
    import urllib.request

    url = api_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    print(f"  Waiting for {url} to respond (up to {timeout}s) ...", end="", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(" ready.")
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(HEALTH_POLL_INTERVAL)
    print(" timed out.")
    return False


def docker_image_exists(tag: str) -> bool:
    r = run(["docker", "image", "inspect", tag], capture=True, check=False)
    return r.returncode == 0


def docker_container_exists(name: str) -> bool:
    r = run(["docker", "inspect", name], capture=True, check=False)
    return r.returncode == 0


def docker_container_running(name: str) -> bool:
    r = run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture=True, check=False,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def docker_volume_exists(name: str) -> bool:
    r = run(["docker", "volume", "inspect", name], capture=True, check=False)
    return r.returncode == 0


def sha12(sha: str) -> str:
    return sha[:12]


def candidate_names(sha: str) -> dict[str, str]:
    s = sha12(sha)
    return {
        "image": f"fastapi-mlc-docker-web:verify-{s}",
        "container": f"fastapi-mlc-verify-{s}",
        "volume": f"fastapi_mlc_verify_{s}",
    }


# ── Build candidate image ──────────────────────────────────────────────────────

def build_candidate_image(repo: str, candidate_sha: str, image_tag: str) -> None:
    """Build candidate Docker image using a temp build context with patched lock file."""
    print(f"\n[BUILD] Creating candidate Docker image: {image_tag}")
    print(f"  mlc-cli ref: {candidate_sha}")

    with tempfile.TemporaryDirectory(prefix="mlc-verify-ctx-") as tmpdir:
        ctx = Path(tmpdir)

        # Copy repo tree (excluding .git, __pycache__, .venv*, .raw_model_cache, .mlc-cli-verify)
        excludes = {".git", "__pycache__", ".venv", ".venv_test", ".raw_model_cache", ".mlc-cli-verify", "*.pyc"}
        print("  Copying build context (excluding large/irrelevant directories) ...")

        def ignore(src, names):
            ignored = set()
            for n in names:
                if n in excludes or n.endswith(".pyc"):
                    ignored.add(n)
            return ignored

        shutil.copytree(str(REPO_ROOT), str(ctx / "build"), ignore=ignore, symlinks=False)
        build_ctx = ctx / "build"

        # Patch docker/mlc-cli.lock with candidate SHA only
        candidate_lock = build_ctx / "docker" / "mlc-cli.lock"
        write_lock_file(candidate_lock, repo, candidate_sha)
        print(f"  Patched lock file in build context: MLC_CLI_REF={candidate_sha}")

        print(f"\n  Running: docker build -t {image_tag} .\n")
        run(["docker", "build", "-t", image_tag, "."], cwd=build_ctx)

    print(f"\n[BUILD] Image ready: {image_tag}")


# ── Start candidate container ──────────────────────────────────────────────────

def start_candidate_container(names: dict[str, str], port: int) -> None:
    """Create and start the candidate container."""
    image = names["image"]
    container = names["container"]
    volume = names["volume"]

    print(f"\n[START] Starting candidate container: {container}")
    print(f"  Image:  {image}")
    print(f"  Volume: {volume}")
    print(f"  Port:   {port}")

    if docker_container_exists(container):
        die(f"Candidate container {container} already exists. Re-run the script and choose how to handle it.")

    run([
        "docker", "run",
        "--name", container,
        "--detach",
        "--gpus", "all",
        "-p", f"{port}:8000",
        "-v", f"{volume}:/workspace",
        "-e", "MLC_CLI_PATH=/workspace/mlc-cli",
        "-e", "BAKED_MLC_CLI_PATH=/opt/mlc-cli",
        "-e", "TVM_HOME=/workspace/mlc-cli/tvm",
        "-e", "PYTHONPATH=/workspace/mlc-cli/tvm/python",
        "-e", "LD_LIBRARY_PATH=/workspace/mlc-cli/tvm/build/lib:/workspace/mlc-cli/mlc-llm/build/lib",
        image,
    ])
    print(f"  Candidate container started (detached).")


# ── Run integration tests ──────────────────────────────────────────────────────

def run_integration_test(script: Path, api_url: str, label: str) -> bool:
    """Run an integration test script. Returns True on success."""
    print(f"\n[TEST] Running {label} ...")
    print(f"  Script:  {script}")
    print(f"  API_URL: {api_url}")
    print()
    env = {"API_URL": api_url}
    result = run([sys.executable, str(script)], check=False, env=env)
    if result.returncode == 0:
        print(f"\n[TEST] {label} PASSED ✓")
        return True
    else:
        print(f"\n[TEST] {label} FAILED ✗ (exit code {result.returncode})")
        return False


def ensure_local_test_deps() -> None:
    """Ensure required local Python packages are available before starting heavy builds."""
    for mod in ["httpx"]:
        r = run([sys.executable, "-c", f"import {mod}"], capture=True, check=False)
        if r.returncode != 0:
            die(
                f"Missing local Python test dependency: {mod}\n"
                "  Activate your test environment or install dependencies before running candidate verification.\n"
                "  Example:\n"
                f"    python -m pip install {mod}"
            )


# ── Promotion helpers ──────────────────────────────────────────────────────────

def prompt_promote_lock(repo: str, old_ref: str, candidate_sha: str) -> bool:
    """Prompt 1: Update docker/mlc-cli.lock."""
    print(f"\n  Repo:                    {repo}")
    print(f"  Current pinned ref:      {old_ref}")
    print(f"  Tested candidate ref:    {candidate_sha}")
    help_text = (
        "YES  — Writes the tested mlc-cli ref into docker/mlc-cli.lock so that the\n"
        "       next `docker build` will use this exact ref. The repo URL is not\n"
        "       changed. This is a local file change only; you can review it before\n"
        "       committing.\n\n"
        "NO   — Leaves docker/mlc-cli.lock unchanged. The tested SHA is printed\n"
        "       so you can update the file manually later if you want.\n"
    )
    if ask("Promote this tested mlc-cli ref?", help_text, default=False):
        write_lock_file(LOCK_FILE, repo, candidate_sha)
        print(f"  [OK] docker/mlc-cli.lock updated to {sha12(candidate_sha)}")
        return True
    else:
        print(f"\n  Lock file not updated.")
        print(f"  To promote manually:\n    MLC_CLI_REF={candidate_sha}")
        return False


def prompt_snapshot_main_image(names: dict[str, str]) -> bool:
    """Prompt 2: docker commit candidate container → fastapi-mlc-docker-web:latest."""
    container = names["container"]
    s12 = sha12(names["image"].split("verify-")[-1] if "verify-" in names["image"] else names["image"])
    help_text = (
        "YES  — Takes a snapshot of the tested candidate container and saves it as\n"
        f"       the main local image ({MAIN_IMAGE_TAG}).\n"
        "       This avoids having to rebuild the Docker image from scratch for local use.\n"
        "       The existing main local image (if any) is backed up first.\n"
        "       After a successful snapshot, the candidate container is removed because\n"
        "       its state has been saved into the image.\n"
        "       Note: This is a local convenience only. The Dockerfile and\n"
        "       docker/mlc-cli.lock remain the source of truth for reproducible builds.\n\n"
        "NO   — Leaves the main local image unchanged. The candidate image stays\n"
        f"       available as {names['image']}.\n"
    )
    if not docker_container_exists(container):
        print(f"\n  Candidate container {container} no longer exists — skipping snapshot.")
        return False

    if ask("Use the tested candidate container as the main local Docker image?", help_text, default=True):
        # Backup existing main image
        if docker_image_exists(MAIN_IMAGE_TAG):
            backup_tag = f"fastapi-mlc-docker-web:backup-before-{s12}"
            print(f"  Backing up current {MAIN_IMAGE_TAG} → {backup_tag}")
            run(["docker", "tag", MAIN_IMAGE_TAG, backup_tag])

        # Commit candidate container → main image
        print(f"  Snapshotting {container} → {MAIN_IMAGE_TAG} ...")
        run(["docker", "commit", container, MAIN_IMAGE_TAG])
        print(f"  [OK] {MAIN_IMAGE_TAG} updated.")

        # Remove candidate container
        print(f"  Removing candidate container {container} (state is now in image) ...")
        run(["docker", "rm", "-f", container])
        return True

    return False


def check_main_volume_in_use() -> bool:
    """Return True if no running container is using MAIN_VOLUME, or user chose to continue."""
    r = run(["docker", "ps", "--filter", f"volume={MAIN_VOLUME}", "--format", "{{.Names}}"], capture=True, check=False)
    names = [n.strip() for n in r.stdout.strip().splitlines() if n.strip()]
    if not names:
        return True

    print("\n[WARN] Main workspace volume is currently used by running container(s):")
    for name in names:
        print(f"  - {name}")
    help_text = (
        "YES:\n"
        "- continues replacing artifacts while containers may be using the volume\n"
        "- can cause confusing runtime behavior if the main app is reading files at the same time\n\n"
        "NO:\n"
        "- cancels artifact import\n"
        "- recommended: stop the main container first, then run finalize again\n"
    )
    return ask("Main workspace volume is currently used by running container(s). Continue artifact replacement anyway?", help_text, default=False)


def prompt_import_artifacts(names: dict[str, str], candidate_sha: str) -> bool:
    """Prompt 3: Import dist/, wheels/, mlc-llm/, tvm/ from candidate volume → main volume."""
    volume = names["volume"]
    help_text = (
        "YES  — Copies the tested build artifacts from the candidate workspace into\n"
        "       the main workspace. This avoids having to rerun the heavy /build,\n"
        "       /quantize, or /compile steps in the main container.\n\n"
        "       The following directories are imported:\n"
        "         dist/      — quantized and compiled model outputs\n"
        "         wheels/    — built Python wheels\n"
        "         mlc-llm/   — built MLC-LLM source tree\n"
        "         tvm/       — built TVM source tree\n\n"
        "       models/ is NOT imported by default (model downloads are large and\n"
        "       usually independent of the mlc-cli ref).\n\n"
        "NO   — Leaves the main workspace unchanged.\n"
    )
    if not docker_volume_exists(volume):
        print(f"\n  Candidate volume {volume} does not exist — skipping artifact import.")
        return False

    if not docker_volume_exists(MAIN_VOLUME):
        print(f"\n  Main workspace volume {MAIN_VOLUME} not found.")
        print(f"  Skipping artifact import. You may set MAIN_VOLUME env var to override.")
        return False

    if not ask("Import tested build artifacts into the main workspace?", help_text, default=True):
        return False

    if not check_main_volume_in_use():
        print("  Artifact import cancelled.")
        return False

    dirs_to_import = ["dist", "wheels", "mlc-llm", "tvm"]

    # Estimate backup size
    size_script = (
        "total=0; "
        "for d in dist wheels mlc-llm tvm; do "
        "  if [ -d /main/mlc-cli/$d ]; then "
        "    s=$(du -sm /main/mlc-cli/$d 2>/dev/null | awk '{print $1}'); "
        "    if [ -n \"$s\" ]; then total=$((total + s)); fi; "
        "  fi; "
        "done; "
        "echo $total"
    )
    r = run([
        "docker", "run", "--rm", "-v", f"{MAIN_VOLUME}:/main", "busybox", "sh", "-c", size_script
    ], capture=True, check=False)
    
    size_mb = r.stdout.strip()
    if size_mb and size_mb.isdigit():
        smb = int(size_mb)
        size_str = f"{smb}MB" if smb < 1024 else f"{smb/1024:.1f}GB"
        print(f"\n  Current main artifact size: {size_str}")
        print("  Backup would use about this much additional disk space.")
    else:
        print("\n  Current main artifact size: unknown")

    backup_help = (
        "YES:\n"
        "- safer rollback if artifact import fails\n"
        "- copies current main dist/, wheels/, mlc-llm/, and tvm/ before replacing them\n"
        "- uses extra disk space\n\n"
        "NO:\n"
        "- faster and uses less disk\n"
        "- current main artifacts will be deleted before importing candidate artifacts\n"
        "- no automatic rollback is possible if import fails\n"
    )
    do_backup = ask("Back up current main artifacts before replacing them?", backup_help, default=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = f"/main/mlc-cli/.artifact-backups/verify-{sha12(candidate_sha)}-{timestamp}"

    if do_backup:
        print(f"\n  Backing up main artifact dirs to {backup_path} ...")
        backup_scripts = []
        for d in dirs_to_import:
            backup_scripts.append(
                f"if [ -d /main/mlc-cli/{d} ]; then cp -a /main/mlc-cli/{d} {backup_path}/{d}; fi"
            )
        backup_cmd = " && ".join(backup_scripts)
        r_backup = run([
            "docker", "run", "--rm", "-v", f"{MAIN_VOLUME}:/main", "busybox", "sh", "-c",
            f"mkdir -p {backup_path} && {backup_cmd}"
        ], check=False)
        if r_backup.returncode != 0:
            print("  [ERROR] Backup failed. Artifact import cancelled.")
            return False

    print(f"\n  Replacing main artifacts with candidate artifacts ...")
    replace_scripts = []
    for d in dirs_to_import:
        replace_scripts.append(
            f"if [ -d /cand/mlc-cli/{d} ]; then "
            f"  rm -rf /main/mlc-cli/{d}; "
            f"  mkdir -p /main/mlc-cli/{d}; "
            f"  cp -a /cand/mlc-cli/{d}/. /main/mlc-cli/{d}/; "
            f"else "
            f"  echo '  [WARN] /cand/mlc-cli/{d} not found in candidate volume.'; "
            f"  rm -rf /main/mlc-cli/{d}; "
            f"fi"
        )
    replace_cmd = " && ".join(replace_scripts)

    r_import = run([
        "docker", "run", "--rm",
        "-v", f"{volume}:/cand",
        "-v", f"{MAIN_VOLUME}:/main",
        "busybox", "sh", "-c",
        replace_cmd
    ], check=False)

    if r_import.returncode != 0:
        print("\n  [ERROR] Artifact import failed.")
        if do_backup:
            print(f"  Attempting automatic rollback from {backup_path} ...")
            rollback_scripts = []
            for d in dirs_to_import:
                rollback_scripts.append(
                    f"rm -rf /main/mlc-cli/{d}; "
                    f"if [ -d {backup_path}/{d} ]; then cp -a {backup_path}/{d} /main/mlc-cli/{d}; fi"
                )
            r_rollback = run([
                "docker", "run", "--rm", "-v", f"{MAIN_VOLUME}:/main", "busybox", "sh", "-c",
                " && ".join(rollback_scripts)
            ], check=False)
            if r_rollback.returncode == 0:
                print("  [OK] Rollback succeeded.")
            else:
                print("  [ERROR] Rollback failed. Artifacts may be in an inconsistent state.")
        else:
            print("  Artifact import failed. No backup was created, so automatic rollback is not available.")
        return False

    # Check candidate models/ for non-TinyLlama content
    r = run([
        "docker", "run", "--rm",
        "-v", f"{volume}:/cand",
        "busybox", "sh", "-c",
        "ls /cand/mlc-cli/models/ 2>/dev/null || echo",
    ], capture=True, check=False)
    model_entries = [e.strip() for e in r.stdout.strip().splitlines() if e.strip()]
    non_tiny = [e for e in model_entries if "TinyLlama" not in e and "tinyllama" not in e.lower()]
    if non_tiny:
        model_help = (
            f"YES  — Also copies models/ from the candidate workspace:\n"
            f"       {', '.join(non_tiny)}\n\n"
            f"NO   — Leaves models/ in the main workspace unchanged.\n"
        )
        if ask(f"Also import models/ ({', '.join(non_tiny)})?", model_help, default=False):
            run([
                "docker", "run", "--rm",
                "-v", f"{volume}:/cand",
                "-v", f"{MAIN_VOLUME}:/main",
                "busybox", "sh", "-c",
                "mkdir -p /main/mlc-cli/models && cp -a /cand/mlc-cli/models/. /main/mlc-cli/models/",
            ])
            print("  [OK] models/ imported.")

    print("  [OK] Artifact import complete.")
    return True


def prompt_commit_push(candidate_sha: str) -> None:
    """Prompt 4: Commit and push docker/mlc-cli.lock."""
    help_text = (
        "YES  — Stages only docker/mlc-cli.lock and commits it with the message\n"
        "       'chore: update pinned mlc-cli ref'. Then pushes to the current\n"
        "       branch's upstream remote if available. Other dirty files are NOT\n"
        "       staged or committed.\n\n"
        "NO   — Nothing is committed or pushed. You can review the diff and\n"
        "       commit later with: git add docker/mlc-cli.lock && git commit\n"
    )
    if not ask("Commit and push the docker/mlc-cli.lock update now?", help_text, default=False):
        print("\n  Lock file change not committed. Review with: git diff docker/mlc-cli.lock")
        return

    # Show other dirty files as info only
    r = run(["git", "status", "--short"], capture=True, check=False, cwd=REPO_ROOT)
    dirty_others = [
        line for line in r.stdout.splitlines()
        if "docker/mlc-cli.lock" not in line and line.strip()
    ]
    if dirty_others:
        print("\n  Note: these other files are also modified but will NOT be committed:")
        for f in dirty_others:
            print(f"    {f}")

    run(["git", "add", "docker/mlc-cli.lock"], cwd=REPO_ROOT)

    staged = run(
        ["git", "diff", "--cached", "--quiet", "--", "docker/mlc-cli.lock"],
        capture=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if staged.returncode == 0:
        print("\n  No docker/mlc-cli.lock changes to commit.")
        return

    run(["git", "commit", "-m", "chore: update pinned mlc-cli ref"], cwd=REPO_ROOT)

    # Try to push to upstream
    r = run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        capture=True, check=False, cwd=REPO_ROOT,
    )
    if r.returncode == 0 and r.stdout.strip():
        run(["git", "push"], cwd=REPO_ROOT)
    else:
        r2 = run(["git", "branch", "--show-current"], capture=True, check=False, cwd=REPO_ROOT)
        branch = r2.stdout.strip() or "your-branch"
        print(f"\n  No upstream set. To push manually:\n    git push origin {branch}")


# ── Cleanup ────────────────────────────────────────────────────────────────────

def cleanup_prompts(names: dict[str, str], *, container_removed: bool, artifacts_imported: bool) -> None:
    """Ask about removing candidate volume and image after a successful pass."""
    volume = names["volume"]
    image = names["image"]
    container = names["container"]

    # Container may already be removed by snapshot step
    if not container_removed and docker_container_exists(container):
        if ask(f"Remove candidate container {container}?",
               "YES — stops and removes the candidate container.\nNO  — leaves it running for inspection.",
               default=True):
            run(["docker", "rm", "-f", container])

    # Volume
    if docker_volume_exists(volume):
        default_remove = artifacts_imported
        help_v = (
            "YES  — Removes the candidate workspace volume (frees disk space).\n"
            + ("       Artifacts were already imported into the main workspace.\n" if artifacts_imported else
               "       Note: artifacts were NOT imported, so removing this volume\n"
               "       will also discard any candidate build outputs.\n")
            + "NO   — Keeps the volume for manual inspection."
        )
        r = run([
            "docker", "run", "--rm",
            "-v", f"{volume}:/v", "busybox", "du", "-sh", "/v",
        ], capture=True, check=False)
        size = r.stdout.strip().split("\t")[0] if r.returncode == 0 else "unknown size"
        if ask(f"Remove candidate volume {volume} ({size})?", help_v, default=default_remove):
            run(["docker", "volume", "rm", volume])

    # Image
    if docker_image_exists(image):
        help_i = (
            f"YES  — Removes the candidate image {image} (frees disk space).\n"
            f"NO   — Keeps it locally. You can reuse it with: docker run {image}"
        )
        if ask(f"Remove candidate image {image}?", help_i, default=False):
            run(["docker", "rmi", image])


def print_fail_info(names: dict[str, str], port: int, stage: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Candidate verification FAILED at: {stage}")
    print(f"  Container:  {names['container']}")
    print(f"  Image:      {names['image']}")
    print(f"  Volume:     {names['volume']}")
    print(f"  Port:       {port}")
    print(f"\n  The candidate container is left running for debugging.")
    print(f"  Connect with: docker exec -it {names['container']} bash")
    print(f"  API:          http://localhost:{port}")
    print(f"\n  To clean up when done:")
    print(f"    docker rm -f {names['container']}")
    print(f"    docker volume rm {names['volume']}")
    print(f"    docker rmi {names['image']}")
    print(f"{'='*60}")

    if ask(f"Remove candidate container, volume, and image now?",
           "YES — immediately deletes the candidate container, volume, and image.\n"
           "NO  — leaves them running/available for debugging.",
           default=False):
        run(["docker", "rm", "-f", names["container"]], check=False)
        run(["docker", "volume", "rm", names["volume"]], check=False)
        run(["docker", "rmi", names["image"]], check=False)


# ── Post-pass prompts (shared by main flow and finalize) ──────────────────────

def run_post_pass_prompts(state: dict) -> None:
    """Run the four promotion prompts using saved state."""
    repo = state["repo"]
    old_ref = state["old_ref"]
    candidate_sha = state["candidate_sha"]
    names = state["names"]

    print(f"\n{'='*60}")
    print("  ✅  Both smoke and full integration tests passed.")
    print(f"  Candidate ref: {candidate_sha}")
    print(f"{'='*60}")

    lock_updated = prompt_promote_lock(repo, old_ref, candidate_sha)
    container_removed = prompt_snapshot_main_image(names)
    artifacts_imported = prompt_import_artifacts(names, candidate_sha)

    if lock_updated:
        prompt_commit_push(candidate_sha)

    cleanup_prompts(names, container_removed=container_removed, artifacts_imported=artifacts_imported)

    print("\n✅  Candidate verification complete.\n")


# ── Save / load state ──────────────────────────────────────────────────────────

def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def load_state() -> dict:
    if not STATE_FILE.exists():
        die(
            f"No saved pass state found at {STATE_FILE}.\n"
            "  Run without 'finalize' first to verify a candidate."
        )
    return json.loads(STATE_FILE.read_text())


# ── Finalize subcommand ────────────────────────────────────────────────────────

def cmd_finalize() -> None:
    print("=== Re-opening post-pass options (finalize mode) ===")
    state = load_state()
    names = state["names"]

    # Validate that needed Docker objects still exist
    missing = []
    if not docker_image_exists(names["image"]) and not docker_container_exists(names["container"]):
        missing.append(f"image {names['image']} and container {names['container']}")
    if missing:
        print(f"\n[WARN] Some candidate objects are missing: {', '.join(missing)}")
        print("  You may still update the lock file or skip other steps.")

    run_post_pass_prompts(state)


# ── Main verification flow ─────────────────────────────────────────────────────

def cmd_verify() -> None:
    print("=== mlc-cli Candidate Verification ===\n")

    # Step 1: Read lock file
    if not LOCK_FILE.exists():
        die(f"Lock file not found: {LOCK_FILE}")
    lock = read_lock_file(LOCK_FILE)
    repo = lock.get("MLC_CLI_REPO", "")
    old_ref = lock.get("MLC_CLI_REF", "")
    if not repo or not old_ref:
        die(f"docker/mlc-cli.lock must contain MLC_CLI_REPO and MLC_CLI_REF.\nGot: {lock!r}")

    print(f"  Configured repo:  {repo}")
    print(f"  Pinned ref:       {old_ref}")

    # Step 2: Fetch candidate HEAD
    candidate_sha = fetch_remote_head(repo)
    print(f"  Remote HEAD:      {candidate_sha}")

    # Step 3: Already pinned?
    if candidate_sha == old_ref:
        print(
            f"\n✅  The configured mlc-cli repo is already at the pinned ref ({sha12(old_ref)}).\n"
            "  Nothing to verify. The current image is up to date.\n"
        )
        return

    print(
        f"\n  New candidate available: {sha12(candidate_sha)}\n"
        f"  (differs from pinned ref {sha12(old_ref)})\n"
    )

    ensure_local_test_deps()

    # Build names
    names = candidate_names(candidate_sha)

    # Pre-flight: Check existing container
    if docker_container_exists(names["container"]):
        running = docker_container_running(names["container"])
        status = "running" if running else "stopped"
        print(f"\n[WARN] Candidate container {names['container']} already exists.")
        print(f"  Status:       {status}")
        print(f"  Image:        {names['image']}")
        print(f"  Volume:       {names['volume']}")
        print(f"  Candidate ref: {candidate_sha}")
        print("\n  This candidate container already exists. It may be from a previous")
        print("  verification run, and this script cannot know how far that run got.")
        
        choices = [
            (
                "Delete it and rebuild/retest from scratch",
                "- Removes the existing candidate container.\n"
                "- Removes the candidate volume if it exists.\n"
                "- Rebuilds the candidate image and reruns smoke + full from a clean workspace.\n"
                "- This is the safest option for a clean verification."
            ),
            (
                "Start/reuse it for manual testing, then exit",
                "- If stopped, run docker start <container>.\n"
                "- If already running, leave it running.\n"
                "- Do not rebuild.\n"
                "- Do not rerun smoke/full.\n"
                "- Exit with code 0."
            ),
            (
                "Cancel without changing anything",
                "- Do nothing and exit with code 0.\n"
                "- This is the safest default."
            )
        ]
        
        choice = ask_choice("How do you want to handle it?", choices, default=3)
        if choice == 3:
            print("\n  Cancelled.")
            return
        elif choice == 2:
            print(f"\n  Reusing existing container: {names['container']}")
            if not running:
                run(["docker", "start", names["container"]])
                print("  [OK] Container started.")
            else:
                print("  [OK] Container is already running.")
                
            print(f"\n  To inspect it:")
            print(f"    docker logs -f {names['container']}")
            print(f"    docker exec -it {names['container']} bash")
            
            r = run(["docker", "port", names["container"], "8000"], capture=True, check=False)
            if r.returncode == 0 and r.stdout.strip():
                print(f"  Published port: {r.stdout.strip()}")
            return
        elif choice == 1:
            print(f"\n  Removing existing container {names['container']} ...")
            run(["docker", "rm", "-f", names["container"]])
            if docker_volume_exists(names["volume"]):
                print(f"  Removing existing volume {names['volume']} ...")
                run(["docker", "volume", "rm", names["volume"]])
    elif docker_volume_exists(names["volume"]):
        help_v = (
            "YES:\n"
            "- removes old candidate workspace volume before testing\n"
            "- prevents old dist/, wheels/, mlc-llm/, tvm/, or model artifacts from making the candidate look like it passed\n\n"
            "NO:\n"
            "- reuses the existing candidate volume\n"
            "- faster, but can hide problems because old artifacts may already be present"
        )
        if ask(f"Candidate workspace volume {names['volume']} already exists. Remove it for a clean verification?", help_v, default=True):
            run(["docker", "volume", "rm", names["volume"]])

    port = find_free_port(8001)

    # Step 4: Build candidate image
    try:
        build_candidate_image(repo, candidate_sha, names["image"])
    except subprocess.CalledProcessError as exc:
        print(f"\n{'=' * 60}")
        print("  Candidate verification FAILED at: Docker image build")
        print(f"  Candidate ref: {candidate_sha}")
        print(f"  Image tag:     {names['image']}")
        print()
        print("  Docker could not build the candidate image.")
        print("  The candidate ref was checked out successfully, but the Dockerfile")
        print("  expected something that was not present or no longer compatible.")
        print()
        print("  Check the Docker error block above for the exact failing Dockerfile line.")
        print("  Example causes:")
        print("  - required script/file path changed in mlc-cli")
        print("  - Dockerfile still expects an older mlc-cli layout")
        print("  - candidate ref is from a different branch/history than the pinned ref")
        print()
        print("  Result:")
        print("  - smoke/full tests were not run")
        print("  - docker/mlc-cli.lock was not updated")
        print("  - current pinned ref remains safe to use")
        print(f"{'=' * 60}\n")
        sys.exit(exc.returncode)

    # Step 5: Start container
    start_candidate_container(names, port)

    api_url = f"http://localhost:{port}"

    # Step 6: Wait for /health
    if not wait_for_health(api_url):
        print_fail_info(names, port, "health check (container did not start)")
        sys.exit(1)

    # Step 7: Smoke test
    smoke_ok = run_integration_test(SMOKE_SCRIPT, api_url, "Smoke Integration Test")
    if not smoke_ok:
        print_fail_info(names, port, "smoke test")
        sys.exit(1)

    # Step 8: Full integration test
    full_ok = run_integration_test(FULL_SCRIPT, api_url, "Full Integration Test")
    if not full_ok:
        print_fail_info(names, port, "full integration test")
        sys.exit(1)

    # Step 9: Save state for finalize
    state = {
        "repo": repo,
        "old_ref": old_ref,
        "candidate_sha": candidate_sha,
        "names": names,
        "port": port,
        "passed_at": datetime.datetime.now().isoformat(),
    }
    save_state(state)

    # Step 10: Post-pass prompts
    run_post_pass_prompts(state)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    if not args:
        cmd_verify()
    elif args[0] == "finalize":
        cmd_finalize()
    else:
        print(f"Usage: {sys.argv[0]} [finalize]", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
