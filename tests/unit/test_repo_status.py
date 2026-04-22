"""
tests/unit/test_repo_status.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the GET /repo-status endpoint.
"""
from __future__ import annotations

import json
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

class TestRepoStatus:
    
    @patch("app.main.run_command")
    @patch("app.main.get_repo_alignment")
    def test_repo_missing(self, mock_align, mock_run, client):
        """When repo is missing, status is 'missing'."""
        mock_align.return_value = {
            "exists": False,
            "pinned_sha": "pinned123",
            "current_sha": None,
            "relation": "missing"
        }
        resp = client.get("/repo-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "missing"
        assert data["is_clean"] is None
        assert "Repository missing" in data["message"]
        assert data["alignment"]["relation"] == "missing"
        assert data["alignment"]["repair_possible"] is True

    @patch("app.main.run_command")
    @patch("app.main.get_repo_alignment")
    def test_repo_aligned_and_clean(self, mock_align, mock_run, client):
        """Repo exists, matches pinned SHA, and has no local changes -> healthy."""
        mock_align.return_value = {
            "exists": True,
            "pinned_sha": "sha123",
            "current_sha": "sha123",
            "relation": "match"
        }
        mock_run.return_value = _proc(stdout="") # git status clean
        
        resp = client.get("/repo-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["is_clean"] is True
        assert data["alignment"]["relation"] == "match"
        assert data["alignment"]["repair_possible"] is False  # Already matched

    @patch("app.main.run_command")
    @patch("app.main.get_repo_alignment")
    def test_repo_behind_and_dirty(self, mock_align, mock_run, client):
        """Repo exists, is behind pinned SHA, and has uncommitted changes -> degraded."""
        mock_align.return_value = {
            "exists": True,
            "pinned_sha": "new_sha",
            "current_sha": "old_sha",
            "relation": "behind"
        }
        mock_run.return_value = _proc(stdout=" M app/main.py") 
        
        resp = client.get("/repo-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["is_clean"] is False
        assert data["alignment"]["relation"] == "behind"
        assert data["alignment"]["repair_possible"] is True
        assert data["changes"] == ["M app/main.py"]

    @patch("app.main.run_command")
    @patch("app.main.get_repo_alignment")
    def test_repo_diverged(self, mock_align, mock_run, client):
        """Repo has diverged from the pinned SHA -> degraded."""
        mock_align.return_value = {
            "exists": True,
            "pinned_sha": "sha_a",
            "current_sha": "sha_b",
            "relation": "diverged"
        }
        mock_run.return_value = _proc(stdout="") 
        
        resp = client.get("/repo-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["alignment"]["relation"] == "diverged"
        assert data["alignment"]["repair_possible"] is True

    @patch("app.main.run_command")
    @patch("app.main.get_repo_alignment")
    def test_no_pin_active(self, mock_align, mock_run, client):
        """No pinning metadata exists -> healthy (but unpinned)."""
        mock_align.return_value = {
            "exists": True,
            "pinned_sha": None,
            "current_sha": "some_sha",
            "relation": "unpinned"
        }
        mock_run.return_value = _proc(stdout="")
        
        resp = client.get("/repo-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["alignment"]["pinned_sha"] is None
        assert data["alignment"]["relation"] == "unpinned"
        assert data["alignment"]["repair_possible"] is False
        assert "No pinning active" in data["message"]

