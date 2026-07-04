"""
tests/unit/test_repo_status.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the GET /repo-status endpoint in the baked mlc-cli architecture.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_repo_status_reports_baked_and_workspace_state(client, monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    baked = tmp_path / "baked"
    workspace.mkdir()
    baked.mkdir()

    for name in ["models", "dist", "wheels", "mlc-llm", "tvm"]:
        (workspace / name).mkdir()

    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)
    monkeypatch.setattr("app.main.BAKED_MLC_CLI_PATH", baked)

    def fake_git_head(path: Path):
        if path == baked:
            return "sha-baked"
        if path == workspace:
            return "sha-baked"
        return None

    def fake_read_text_file(path: Path):
        if path == Path("/opt/mlc-cli-ref.txt"):
            return "sha-baked"
        if path == Path("/opt/mlc-cli-repo.txt"):
            return "https://github.com/MinhThien-Pham/mlc-cli.git"
        return None

    with patch("app.main.git_head", side_effect=fake_git_head), \
         patch("app.main.read_text_file", side_effect=fake_read_text_file):
        resp = client.get("/repo-status")

    assert resp.status_code == 200
    data = resp.json()

    assert data["source_management"] == "baked-image"
    assert data["mlc_cli_path"] == str(workspace)
    assert data["baked_mlc_cli_path"] == str(baked)
    assert data["baked_ref_file"] == "sha-baked"
    assert data["baked_repo_file"] == "https://github.com/MinhThien-Pham/mlc-cli.git"
    assert data["baked_actual_head"] == "sha-baked"
    assert data["workspace_head"] == "sha-baked"
    assert data["workspace_matches_baked"] is True
    assert data["dev_mode"] is False

    for name in ["models", "dist", "wheels", "mlc-llm", "tvm"]:
        assert data["artifact_dirs"][name] is True


def test_repo_status_handles_missing_git_heads(client, monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    baked = tmp_path / "baked"

    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)
    monkeypatch.setattr("app.main.BAKED_MLC_CLI_PATH", baked)

    with patch("app.main.git_head", return_value=None), \
         patch("app.main.read_text_file", return_value=None):
        resp = client.get("/repo-status")

    assert resp.status_code == 200
    data = resp.json()

    assert data["source_management"] == "baked-image"
    assert data["baked_actual_head"] is None
    assert data["workspace_head"] is None
    assert data["workspace_matches_baked"] is None

    for name in ["models", "dist", "wheels", "mlc-llm", "tvm"]:
        assert data["artifact_dirs"][name] is False
