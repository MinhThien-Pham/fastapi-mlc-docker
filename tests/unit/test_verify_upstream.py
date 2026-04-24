"""
tests/unit/test_verify_upstream.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Focused tests for the manual upstream verification script.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

import verify_upstream


def _proc(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like mock."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


class TestVerifyUpstreamDirtyGuard:
    """Tracked Bryan changes are restored before verification continues."""

    @patch("verify_upstream.try_restore_metadata")
    @patch("verify_upstream.restore_tracked_changes")
    @patch("verify_upstream.get_git_dirty_state")
    @patch("verify_upstream.sh")
    def test_preflight_restores_tracked_dirty_before_recovery(self, mock_sh, mock_dirty, mock_cleanup, mock_restore):
        mock_dirty.return_value = {
            "exists": True,
            "tracked_dirty": True,
            "tracked_changes": [" M app/main.py"],
            "untracked_files": ["scratch.txt"],
            "error": None,
        }
        mock_cleanup.return_value = {
            "ok": True,
            "restored": True,
            "before": mock_dirty.return_value,
            "after": {
                "exists": True,
                "tracked_dirty": False,
                "tracked_changes": [],
                "untracked_files": ["scratch.txt"],
                "error": None,
            },
            "error": None,
        }
        mock_restore.return_value = True

        def fake_sh(cmd):
            if cmd[:4] == ["docker", "compose", "ps", "--status"]:
                return _proc(stdout="web\n")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return _proc(stdout="")
            if cmd[:3] == ["git", "branch", "--show-current"]:
                return _proc(stdout="main\n")
            return _proc(stdout="")

        mock_sh.side_effect = fake_sh

        with patch.object(httpx, "get") as mock_httpx_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_httpx_get.return_value = mock_response

            verify_upstream.preflight()

        mock_cleanup.assert_called_once()
        mock_restore.assert_called_once()

    @patch("verify_upstream.try_restore_metadata")
    @patch("verify_upstream.restore_tracked_changes")
    @patch("verify_upstream.get_git_dirty_state")
    @patch("verify_upstream.sh")
    def test_preflight_untracked_only_skips_cleanup(self, mock_sh, mock_dirty, mock_cleanup, mock_restore):
        mock_dirty.return_value = {
            "exists": True,
            "tracked_dirty": False,
            "tracked_changes": [],
            "untracked_files": ["cache/model.bin"],
            "error": None,
        }
        mock_restore.return_value = True

        def fake_sh(cmd):
            if cmd[:4] == ["docker", "compose", "ps", "--status"]:
                return _proc(stdout="web\n")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return _proc(stdout="")
            if cmd[:3] == ["git", "branch", "--show-current"]:
                return _proc(stdout="main\n")
            return _proc(stdout="")

        mock_sh.side_effect = fake_sh

        with patch.object(httpx, "get") as mock_httpx_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_httpx_get.return_value = mock_response

            verify_upstream.preflight()

        mock_cleanup.assert_not_called()

    @patch("verify_upstream.try_restore_metadata")
    @patch("verify_upstream.restore_tracked_changes")
    @patch("verify_upstream.get_git_dirty_state")
    @patch("verify_upstream.sh")
    def test_preflight_allows_clean_repo(self, mock_sh, mock_dirty, mock_cleanup, mock_restore):
        mock_dirty.return_value = {
            "exists": True,
            "tracked_dirty": False,
            "tracked_changes": [],
            "untracked_files": [],
            "error": None,
        }
        mock_restore.return_value = True

        def fake_sh(cmd):
            if cmd[:4] == ["docker", "compose", "ps", "--status"]:
                return _proc(stdout="web\n")
            if cmd[:3] == ["git", "status", "--porcelain"]:
                return _proc(stdout="")
            if cmd[:3] == ["git", "branch", "--show-current"]:
                return _proc(stdout="main\n")
            return _proc(stdout="")

        mock_sh.side_effect = fake_sh

        with patch.object(httpx, "get") as mock_httpx_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_httpx_get.return_value = mock_response

            verify_upstream.preflight()

        mock_restore.assert_called_once()
        mock_cleanup.assert_not_called()
        assert mock_sh.call_count >= 3
