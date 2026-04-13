"""
tests/test_setup_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the GET /setup-check endpoint.

We monkeypatch:
  - app.main.MLC_CLI_PATH  →  a real tmp_path so Path.exists() works naturally
  - subprocess.run          →  a lightweight mock so no real Go/conda/GPU needed

All tests run on any developer machine — no CUDA, Go, or Conda required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Test helpers ──────────────────────────────────────────────────────────────

def _proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like mock."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


def _all_tools_ok(cmd: list[str], **_kwargs) -> MagicMock:
    """subprocess.run side-effect: every tool reports success."""
    if cmd[0] == "go":
        return _proc("go version go1.24.0 linux/amd64")
    if cmd[0] == "conda":
        return _proc("conda 24.1.0")
    if cmd[0] == "nvidia-smi":
        return _proc("NVIDIA GeForce RTX 3090")
    if cmd[0] == "nvcc":
        return _proc("nvcc: NVIDIA (R) Cuda compiler driver, V12.6.0")
    # git remote get-url origin
    return _proc("https://github.com/ballinyouup/mlc-cli.git")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSetupCheckRepoMissing:
    """Repo does not exist on disk."""

    def test_returns_200(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        with patch("subprocess.run", side_effect=_all_tools_ok):
            resp = client.get("/setup-check")
        assert resp.status_code == 200

    def test_repo_exists_is_false(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert data["repo_exists"] is False
        assert data["checks"]["repo"]["available"] is False

    def test_status_is_warning_when_tools_are_present(self, client, monkeypatch, tmp_path):
        """Repo missing but tools ok → warning (it can be cloned on demand)."""
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert data["status"] == "warning"

    def test_repo_output_contains_helpful_hint(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        output = data["checks"]["repo"]["output"].lower()
        assert "ensure-repo-exists" in output


class TestSetupCheckAllOk:
    """Repo exists and every tool is available."""

    def test_status_is_ok(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert data["status"] == "ok"

    def test_repo_exists_is_true(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert data["repo_exists"] is True
        assert data["checks"]["repo"]["available"] is True

    def test_all_tool_checks_are_available(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        checks = data["checks"]
        assert checks["go"]["available"] is True
        assert checks["conda"]["available"] is True
        assert checks["nvidia_smi"]["available"] is True
        assert checks["nvcc"]["available"] is True

    def test_no_warnings_when_everything_ok(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert data["warnings"] == []

    def test_origin_is_populated(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=_all_tools_ok):
            data = client.get("/setup-check").json()
        assert "origin" in data["checks"]["repo"]
        assert "github" in data["checks"]["repo"]["origin"].lower()


class TestSetupCheckMissingTools:
    """Individual tools are unavailable."""

    def _make_mock(self, fail_cmd: str):
        """Return a subprocess mock where *fail_cmd* raises FileNotFoundError."""
        def side_effect(cmd: list[str], **kwargs) -> MagicMock:
            if cmd[0] == fail_cmd:
                raise FileNotFoundError(f"{fail_cmd} not found")
            return _all_tools_ok(cmd, **kwargs)
        return side_effect

    def test_missing_go_makes_status_error(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=self._make_mock("go")):
            data = client.get("/setup-check").json()
        assert data["checks"]["go"]["available"] is False
        assert data["status"] == "error"

    def test_missing_conda_makes_status_error(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=self._make_mock("conda")):
            data = client.get("/setup-check").json()
        assert data["checks"]["conda"]["available"] is False
        assert data["status"] == "error"

    def test_missing_nvidia_smi_adds_warning(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=self._make_mock("nvidia-smi")):
            data = client.get("/setup-check").json()
        assert data["checks"]["nvidia_smi"]["available"] is False
        # GPU missing is a warning, not an error (build might still work for CPU-only)
        assert len(data["warnings"]) > 0
        assert any("nvidia" in w.lower() or "gpu" in w.lower() for w in data["warnings"])

    def test_missing_nvcc_adds_warning(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        with patch("subprocess.run", side_effect=self._make_mock("nvcc")):
            data = client.get("/setup-check").json()
        assert data["checks"]["nvcc"]["available"] is False
        assert len(data["warnings"]) > 0
