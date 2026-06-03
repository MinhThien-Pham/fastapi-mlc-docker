"""
tests/unit/test_architecture.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight tests to lock in the baked mlc-cli runtime architecture.
"""
from __future__ import annotations

from pathlib import Path


def test_dockerfile_baked_architecture():
    dockerfile = Path("Dockerfile").read_text()
    
    assert "docker/mlc-cli.lock" in dockerfile, "Dockerfile must use the lock file for MLC_CLI_REF"
    assert "/opt/mlc-cli" in dockerfile, "Dockerfile must bake mlc-cli to /opt/mlc-cli"
    assert "/opt/mlc-cli-ref.txt" in dockerfile, "Dockerfile must record the baked ref"
    assert "/opt/mlc-cli-repo.txt" in dockerfile, "Dockerfile must record the baked repo"


def test_entrypoint_preserves_artifacts():
    entrypoint = Path("docker/entrypoint.sh").read_text()
    
    assert "rsync -a --delete" in entrypoint, "Entrypoint must sync the workspace"
    
    for artifact in ["models", "dist", "wheels", "mlc-llm", "tvm"]:
        assert f"--exclude '/{artifact}/'" in entrypoint, f"Entrypoint must preserve {artifact}"


def test_entrypoint_no_runtime_git_fetch():
    entrypoint = Path("docker/entrypoint.sh").read_text()
    
    assert "git clone" not in entrypoint, "Entrypoint must not clone repo at runtime"
    assert "git fetch" not in entrypoint, "Entrypoint must not fetch from network at runtime"
    assert "git pull" not in entrypoint, "Entrypoint must not pull from network at runtime"
