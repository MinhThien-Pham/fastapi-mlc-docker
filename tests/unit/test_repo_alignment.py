"""
tests/unit/test_repo_alignment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the get_repo_alignment helper in app/helpers.py.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from app.helpers import get_repo_alignment

def _proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like mock."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m

class TestGetRepoAlignment:
    
    def test_repo_missing(self, tmp_path):
        """Relationship is 'missing' if repo path does not exist."""
        repo = tmp_path / "nonexistent"
        meta = tmp_path / "meta.json"
        res = get_repo_alignment(repo, meta)
        assert res["exists"] is False
        assert res["relation"] == "missing"

    def test_meta_missing_but_repo_exists(self, tmp_path):
        """Relationship is 'unknown' if metadata is missing but repo exists."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "missing.json"
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _proc(stdout="sha123")
            res = get_repo_alignment(repo, meta)
            
        assert res["exists"] is True
        assert res["pinned_sha"] is None
        assert res["current_sha"] == "sha123"
        assert res["relation"] == "unpinned"

    def test_exact_match(self, tmp_path):
        """Relationship is 'match' if current SHA == pinned SHA."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        sha = "sha123abc"
        meta.write_text(json.dumps({"pinned_sha": sha}))
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _proc(stdout=sha)
            res = get_repo_alignment(repo, meta)
            
        assert res["relation"] == "match"
        assert res["current_sha"] == sha
        assert res["pinned_sha"] == sha

    def test_ahead(self, tmp_path):
        """Relationship is 'ahead' if pinned is an ancestor of current."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        pinned = "sha_old"
        current = "sha_new"
        meta.write_text(json.dumps({"pinned_sha": pinned}))
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=current), # rev-parse
                _proc(returncode=0),    # is-ancestor old new (ahead)
            ]
            res = get_repo_alignment(repo, meta)
            
        assert res["relation"] == "ahead"
        assert res["current_sha"] == current
        assert res["pinned_sha"] == pinned

    def test_behind(self, tmp_path):
        """Relationship is 'behind' if current is an ancestor of pinned."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        pinned = "sha_new"
        current = "sha_old"
        meta.write_text(json.dumps({"pinned_sha": pinned}))
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=current), # rev-parse
                _proc(returncode=1),    # is-ancestor pinned current (not ahead)
                _proc(returncode=0),    # is-ancestor current pinned (behind)
            ]
            res = get_repo_alignment(repo, meta)
            
        assert res["relation"] == "behind"

    def test_diverged(self, tmp_path):
        """Relationship is 'diverged' if neither is an ancestor of the other."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        pinned = "sha_left"
        current = "sha_right"
        meta.write_text(json.dumps({"pinned_sha": pinned}))
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                _proc(stdout=current), # rev-parse
                _proc(returncode=1),    # not ahead
                _proc(returncode=1),    # not behind
            ]
            res = get_repo_alignment(repo, meta)
            
        assert res["relation"] == "diverged"

    def test_invalid_json_meta_behaves_like_missing(self, tmp_path):
        """Relationship is 'unknown' if metadata is malformed JSON."""
        repo = tmp_path / "repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        meta.write_text("invalid json")
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _proc(stdout="sha123")
            res = get_repo_alignment(repo, meta)
            
        assert res["relation"] == "unpinned"
        assert res["pinned_sha"] is None

    def test_not_a_git_repo_returns_unknown(self, tmp_path):
        """Relationship is 'unknown' if repo path is not a git repository."""
        repo = tmp_path / "not-a-repo"
        repo.mkdir()
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({"pinned_sha": "sha123"}))
        
        # git rev-parse HEAD fails
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _proc(returncode=128)
            res = get_repo_alignment(repo, meta)
            
        assert res["exists"] is True
        assert res["relation"] == "unknown"
