"""
tests/unit/test_ensure_repo_exists.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the POST /ensure-repo-exists endpoint.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

def _proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like mock."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m

class TestEnsureRepoExists:
    
    @patch("app.main.run_git")
    def test_repo_missing_no_pin_clones_head(self, mock_git, client, monkeypatch, tmp_path):
        """Repo missing, no pinned SHA in metadata -> clones HEAD."""
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        
        # Ensure Path("/app/.upstream-sha.json").is_file() returns False
        with patch("pathlib.Path.is_file", return_value=False):
            mock_git.return_value = _proc()
            resp = client.post("/ensure-repo-exists")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "cloned"
        assert "cloned (HEAD)" in data["message"]
        
        # Check git calls: should just be clone
        mock_git.assert_called_once()
        assert "clone" in mock_git.call_args[0][0]

    @patch("app.main.run_git")
    def test_repo_missing_with_pin_clones_and_checkouts(self, mock_git, client, monkeypatch, tmp_path):
        """Repo missing, pinned SHA present -> clones then checkouts pin."""
        pinned_sha = "pinned123456789"
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        
        # Mock metadata reading
        mock_meta = MagicMock(spec=Path)
        mock_meta.is_file.return_value = True
        mock_meta.read_text.return_value = json.dumps({"pinned_sha": pinned_sha})
        
        # Patch Path so when it's called with /app/... it returns our mock
        original_path = Path
        def mocked_path(p):
            if str(p) == "/app/.upstream-sha.json":
                return mock_meta
            return original_path(p)
            
        with patch("app.main.Path", side_effect=mocked_path):
            mock_git.return_value = _proc()
            resp = client.post("/ensure-repo-exists")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "cloned"
        assert pinned_sha[:12] in data["message"]
        
        # Check git calls: clone then checkout
        assert mock_git.call_count == 2
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert "clone" in calls[0]
        assert "checkout" in calls[1]
        assert pinned_sha in calls[1]

    @patch("app.main.run_git")
    def test_repo_exists_already_aligned(self, mock_git, client, monkeypatch, tmp_path):
        """Repo exists and current SHA matches pinned SHA -> no-op."""
        pinned_sha = "pinned123"
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        
        # Mock metadata reading
        mock_meta = MagicMock(spec=Path)
        mock_meta.is_file.return_value = True
        mock_meta.read_text.return_value = json.dumps({"pinned_sha": pinned_sha})
        
        original_path = Path
        def mocked_path(p):
            if str(p) == "/app/.upstream-sha.json":
                return mock_meta
            return original_path(p)
            
        with patch("app.main.Path", side_effect=mocked_path):
            mock_git.return_value = _proc(stdout=pinned_sha)
            resp = client.post("/ensure-repo-exists")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "already-aligned"
        assert "already exists and is aligned" in data["message"]
        
        # Only one git call (rev-parse)
        mock_git.assert_called_once()
        assert "rev-parse" in mock_git.call_args[0][0]

    @patch("app.main.run_git")
    def test_repo_exists_misaligned_triggers_realignment(self, mock_git, client, monkeypatch, tmp_path):
        """Repo exists but SHA differs -> fetches and checkouts pin."""
        pinned_sha = "pinned123"
        current_sha = "old456"
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        
        # Mock metadata reading
        mock_meta = MagicMock(spec=Path)
        mock_meta.is_file.return_value = True
        mock_meta.read_text.return_value = json.dumps({"pinned_sha": pinned_sha})
        
        original_path = Path
        def mocked_path(p):
            if str(p) == "/app/.upstream-sha.json":
                return mock_meta
            return original_path(p)
            
        with patch("app.main.Path", side_effect=mocked_path):
            # Mock sequence of git calls
            mock_git.side_effect = [
                _proc(stdout=current_sha), # rev-parse
                _proc(),                   # fetch
                _proc()                    # checkout
            ]
            resp = client.post("/ensure-repo-exists")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "re-aligned"
        assert pinned_sha[:12] in data["message"]
        assert data["previous_sha"] == current_sha
        
        assert mock_git.call_count == 3
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert "rev-parse" in calls[0]
        assert "fetch" in calls[1]
        assert "checkout" in calls[2]
        assert pinned_sha in calls[2]

    @patch("app.main.run_git")
    def test_repo_exists_no_pin_fallback(self, mock_git, client, monkeypatch, tmp_path):
        """Repo exists, no pinning active -> returns success with warning."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        
        # Mock metadata file missing
        with patch("pathlib.Path.is_file", return_value=False):
            mock_git.return_value = _proc(stdout="some-sha")
            resp = client.post("/ensure-repo-exists")
            
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "none"
        assert "no pinning active" in data["message"]
        mock_git.assert_called_once()
