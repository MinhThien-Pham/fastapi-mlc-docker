"""
tests/unit/test_setup_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the GET /setup-check endpoint.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


def _run_command_ok(cmd: list[str], **_kwargs) -> MagicMock:
    if cmd == ["python", "--version"]:
        return _proc("Python 3.13.0")
    if cmd == ["python", "-c", "import mlc_llm"]:
        return _proc("")
    if cmd == ["python", "-c", "import tvm"]:
        return _proc("")
    return _proc("")


def _tool_check_ok(cmd: list[str]) -> dict:
    return {"available": True, "output": "ok", "returncode": 0}


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    versions = workspace / "scripts" / "config"
    versions.mkdir(parents=True)
    (versions / "versions.sh").write_text('PYTHON_VERSION="3.13"\n')
    for name in ["models", "dist", "wheels"]:
        (workspace / name).mkdir()
    return workspace


def test_setup_check_reports_baked_wrapper_info(client, monkeypatch, tmp_path):
    workspace = _make_workspace(tmp_path)
    baked = tmp_path / "baked"

    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)
    monkeypatch.setattr("app.main.BAKED_MLC_CLI_PATH", baked)

    with patch("app.main.run_command", side_effect=_run_command_ok), \
         patch("app.main.run_tool_check", side_effect=_tool_check_ok), \
         patch("app.main.git_head", side_effect=["sha-baked", "sha-baked"]), \
         patch("app.main.read_text_file", return_value="sha-baked"):
        resp = client.get("/setup-check")

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["repo_exists"] is True
    assert data["checks"]["repo"]["available"] is True

    info = data["wrapper_info"]
    assert info["mlc_cli_path"] == str(workspace)
    assert info["baked_mlc_cli_path"] == str(baked)
    assert info["baked_ref"] == "sha-baked"
    assert info["baked_actual_head"] == "sha-baked"
    assert info["workspace_head"] == "sha-baked"
    assert info["workspace_matches_baked"] is True
    assert info["python_runtime_version"] == "Python 3.13.0"
    assert info["expected_python_version"] == "3.13"
    assert info["python_match"] is True
    assert info["mlc_llm_importable"] is True
    assert info["tvm_importable"] is True
    assert info["artifact_dirs_present"]["models"] is True
    assert info["artifact_dirs_present"]["dist"] is True
    assert info["artifact_dirs_present"]["wheels"] is True


def test_setup_check_missing_workspace_is_warning_when_critical_tools_exist(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "missing-workspace")
    monkeypatch.setattr("app.main.BAKED_MLC_CLI_PATH", tmp_path / "baked")

    with patch("app.main.run_command", side_effect=_run_command_ok), \
         patch("app.main.run_tool_check", side_effect=_tool_check_ok), \
         patch("app.main.git_head", return_value=None), \
         patch("app.main.read_text_file", return_value=None):
        resp = client.get("/setup-check")

    assert resp.status_code == 200
    data = resp.json()

    assert data["repo_exists"] is False
    assert data["status"] == "warning"
    assert data["checks"]["repo"]["available"] is False
    assert "workspace not found" in data["checks"]["repo"]["output"].lower()


def test_setup_check_missing_go_makes_status_error(client, monkeypatch, tmp_path):
    workspace = _make_workspace(tmp_path)
    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)

    def tool_check(cmd: list[str]) -> dict:
        if cmd[0] == "go":
            return {"available": False, "output": "go not found", "returncode": None}
        return _tool_check_ok(cmd)

    with patch("app.main.run_command", side_effect=_run_command_ok), \
         patch("app.main.run_tool_check", side_effect=tool_check), \
         patch("app.main.git_head", return_value="sha"), \
         patch("app.main.read_text_file", return_value="sha"):
        data = client.get("/setup-check").json()

    assert data["checks"]["go"]["available"] is False
    assert data["status"] == "error"


def test_setup_check_missing_conda_makes_status_error(client, monkeypatch, tmp_path):
    workspace = _make_workspace(tmp_path)
    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)

    def tool_check(cmd: list[str]) -> dict:
        if cmd[0] == "conda":
            return {"available": False, "output": "conda not found", "returncode": None}
        return _tool_check_ok(cmd)

    with patch("app.main.run_command", side_effect=_run_command_ok), \
         patch("app.main.run_tool_check", side_effect=tool_check), \
         patch("app.main.git_head", return_value="sha"), \
         patch("app.main.read_text_file", return_value="sha"):
        data = client.get("/setup-check").json()

    assert data["checks"]["conda"]["available"] is False
    assert data["status"] == "error"


def test_setup_check_missing_gpu_tools_adds_warning(client, monkeypatch, tmp_path):
    workspace = _make_workspace(tmp_path)
    monkeypatch.setattr("app.main.MLC_CLI_PATH", workspace)

    def tool_check(cmd: list[str]) -> dict:
        if cmd[0] in {"nvidia-smi", "nvcc"}:
            return {"available": False, "output": f"{cmd[0]} not found", "returncode": None}
        return _tool_check_ok(cmd)

    with patch("app.main.run_command", side_effect=_run_command_ok), \
         patch("app.main.run_tool_check", side_effect=tool_check), \
         patch("app.main.git_head", return_value="sha"), \
         patch("app.main.read_text_file", return_value="sha"):
        data = client.get("/setup-check").json()

    assert data["status"] == "ok"
    assert data["checks"]["nvidia_smi"]["available"] is False
    assert data["checks"]["nvcc"]["available"] is False
    assert data["warnings"]
    assert any("nvidia" in w.lower() or "gpu" in w.lower() for w in data["warnings"])
