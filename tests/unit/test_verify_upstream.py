"""
tests/unit/test_verify_upstream.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Focused tests for the manual upstream verification script.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import httpx
import pytest

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


class TestVerifyUpstreamFailureRollback:
    """Failures stop promotion and still roll back the in-progress candidate state."""

    @patch("verify_upstream.handle_issues")
    @patch("verify_upstream.git_push")
    @patch("verify_upstream.commit_metadata")
    @patch("verify_upstream.run_test")
    @patch("verify_upstream.checkout_in_container")
    @patch("verify_upstream.read_container_sha")
    @patch("verify_upstream.fetch_upstream_head")
    @patch("verify_upstream.read_pinned_sha")
    @patch("verify_upstream.preflight")
    @patch("verify_upstream.RECOVERY_MARKER")
    @patch.object(argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(push=False))
    def test_success_path_clears_recovery_marker_before_finish(
        self,
        mock_parse_args,
        mock_marker,
        mock_preflight,
        mock_read_pinned,
        mock_fetch_head,
        mock_read_container,
        mock_checkout,
        mock_run_test,
        mock_commit,
        mock_push,
        mock_handle_issues,
    ):
        mock_marker.exists.side_effect = [False, True]
        mock_read_pinned.return_value = "pinned-sha"
        mock_fetch_head.return_value = "candidate-sha"
        mock_read_container.side_effect = ["original-sha", "candidate-sha"]
        mock_run_test.side_effect = [True, True]

        verify_upstream.main()

        mock_preflight.assert_called_once_with(want_push=False)
        mock_checkout.assert_any_call("candidate-sha")
        mock_commit.assert_called_once_with("candidate-sha")
        mock_push.assert_not_called()
        mock_handle_issues.assert_called_once_with("candidate-sha", True, True, True, False)
        mock_marker.write_text.assert_called_once()
        mock_marker.unlink.assert_called_once()

    @patch("verify_upstream.handle_issues")
    @patch("verify_upstream.git_push")
    @patch("verify_upstream.commit_metadata")
    @patch("verify_upstream.run_test")
    @patch("verify_upstream.checkout_in_container")
    @patch("verify_upstream.read_container_sha")
    @patch("verify_upstream.fetch_upstream_head")
    @patch("verify_upstream.read_pinned_sha")
    @patch("verify_upstream.preflight")
    @patch("verify_upstream.RECOVERY_MARKER")
    @patch.object(argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(push=False))
    def test_full_failure_rolls_back_and_blocks_promotion(
        self,
        mock_parse_args,
        mock_marker,
        mock_preflight,
        mock_read_pinned,
        mock_fetch_head,
        mock_read_container,
        mock_checkout,
        mock_run_test,
        mock_commit,
        mock_push,
        mock_handle_issues,
    ):
        mock_marker.exists.side_effect = [False, True]
        mock_read_pinned.return_value = "pinned-sha"
        mock_fetch_head.return_value = "candidate-sha"
        mock_read_container.side_effect = ["original-sha", "candidate-sha"]
        mock_run_test.side_effect = [True, False]

        with pytest.raises(SystemExit):
            verify_upstream.main()

        mock_preflight.assert_called_once_with(want_push=False)
        mock_checkout.assert_any_call("candidate-sha")
        mock_checkout.assert_any_call("original-sha")
        assert mock_checkout.call_count == 2
        mock_run_test.assert_any_call("Smoke Integration Test", verify_upstream.SMOKE)
        mock_run_test.assert_any_call("Full Integration Test", verify_upstream.FULL)
        mock_commit.assert_not_called()
        mock_push.assert_not_called()
        mock_handle_issues.assert_called_once_with("candidate-sha", True, False, False, False)
        mock_marker.write_text.assert_called_once()
        mock_marker.unlink.assert_called_once()

    @patch("verify_upstream.handle_issues")
    @patch("verify_upstream.git_push")
    @patch("verify_upstream.commit_metadata")
    @patch("verify_upstream.run_test")
    @patch("verify_upstream.checkout_in_container")
    @patch("verify_upstream.read_container_sha")
    @patch("verify_upstream.fetch_upstream_head")
    @patch("verify_upstream.read_pinned_sha")
    @patch("verify_upstream.preflight")
    @patch("verify_upstream.RECOVERY_MARKER")
    @patch.object(argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(push=False))
    def test_smoke_failure_rolls_back_and_blocks_promotion(
        self,
        mock_parse_args,
        mock_marker,
        mock_preflight,
        mock_read_pinned,
        mock_fetch_head,
        mock_read_container,
        mock_checkout,
        mock_run_test,
        mock_commit,
        mock_push,
        mock_handle_issues,
    ):
        mock_marker.exists.side_effect = [False, True]
        mock_read_pinned.return_value = "pinned-sha"
        mock_fetch_head.return_value = "candidate-sha"
        mock_read_container.side_effect = ["original-sha", "candidate-sha"]
        mock_run_test.return_value = False

        with pytest.raises(SystemExit):
            verify_upstream.main()

        mock_preflight.assert_called_once_with(want_push=False)
        mock_checkout.assert_any_call("candidate-sha")
        mock_checkout.assert_any_call("original-sha")
        assert mock_checkout.call_count == 2
        mock_run_test.assert_called_once_with("Smoke Integration Test", verify_upstream.SMOKE)
        mock_commit.assert_not_called()
        mock_push.assert_not_called()
        mock_handle_issues.assert_called_once_with("candidate-sha", False, None, False, False)
        mock_marker.write_text.assert_called_once()
        mock_marker.unlink.assert_called_once()

    @patch("verify_upstream.checkout_in_container")
    @patch("verify_upstream.preflight")
    @patch("verify_upstream.RECOVERY_MARKER")
    @patch.object(argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(push=False))
    def test_stale_recovery_marker_short_circuits_and_restores_original_sha(
        self,
        mock_parse_args,
        mock_marker,
        mock_preflight,
        mock_checkout,
    ):
        mock_marker.exists.side_effect = [True]
        mock_marker.read_text.return_value = '{"original_container_sha": "orig-sha", "candidate_sha": "cand-sha"}'

        with pytest.raises(SystemExit):
            verify_upstream.main()

        mock_checkout.assert_called_once_with("orig-sha")
        mock_preflight.assert_called_once_with(want_push=False)
        mock_marker.unlink.assert_called_once()

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
