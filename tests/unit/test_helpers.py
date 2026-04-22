"""
tests/test_helpers.py
~~~~~~~~~~~~~~~~~~~~~
Unit tests for the pure helper functions in app/helpers.py.

These do not spin up the FastAPI app — they test logic directly,
which keeps them fast and pinpoints failures precisely.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.helpers import (
    CUTLASS_RETRY_HINT,
    KNOWN_FAILURE_SIGNATURES,
    build_mlc_cli_command,
    detect_known_failure,
    run_tool_check,
    try_restore_metadata,
)
from app.main import BuildRequest


# ── detect_known_failure ──────────────────────────────────────────────────────

class TestDetectKnownFailure:
    """detect_known_failure(line) → hint string or None."""

    def test_normal_line_returns_none(self):
        assert detect_known_failure("Build succeeded.") is None

    def test_empty_line_returns_none(self):
        assert detect_known_failure("") is None

    def test_unrelated_error_returns_none(self):
        assert detect_known_failure("error: undefined reference to main") is None

    def test_flash_attn_detected(self):
        result = detect_known_failure("ImportError: No module named flash_attn")
        assert result is not None

    def test_libflash_attn_detected(self):
        result = detect_known_failure("ld: cannot find -llibflash_attn: No such file")
        assert result is not None

    def test_flash_attention_camelcase_detected(self):
        result = detect_known_failure("Cannot import FlashAttention kernel")
        assert result is not None

    def test_cutlass_detected(self):
        result = detect_known_failure("error: cannot find include path for cutlass")
        assert result is not None

    def test_case_insensitive_cutlass(self):
        # All-caps variant should still match
        assert detect_known_failure("CUTLASS build step failed") is not None

    def test_case_insensitive_flash_attn(self):
        # Mixed case — lower() makes it match "flash_attn"
        assert detect_known_failure("Flash_Attn not installed") is not None

    def test_hint_is_the_module_constant(self):
        """Returned hint should be exactly CUTLASS_RETRY_HINT."""
        result = detect_known_failure("flash_attn error")
        assert result == CUTLASS_RETRY_HINT

    def test_hint_contains_curl_command(self):
        result = detect_known_failure("cutlass failure")
        assert result is not None
        assert "curl" in result

    def test_hint_contains_retry_json_payload(self):
        """Hint must include a ready-to-paste payload with cutlass disabled."""
        result = detect_known_failure("cutlass failure")
        assert result is not None
        # The embedded JSON payload should explicitly disable cutlass
        assert '"cutlass":"n"' in result

    def test_only_one_signature_needed(self):
        """A single matching signature is enough to trigger the hint."""
        for sig in KNOWN_FAILURE_SIGNATURES:
            assert detect_known_failure(f"build error: {sig}") is not None


# ── run_tool_check ────────────────────────────────────────────────────────────

class TestRunToolCheck:
    """run_tool_check(command) → structured availability dict."""

    def test_successful_command_is_available(self):
        mock_result = MagicMock(returncode=0, stdout="go version go1.24.0", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = run_tool_check(["go", "version"])
        assert result["available"] is True
        assert result["returncode"] == 0
        assert "go version" in result["output"]

    def test_failing_command_is_not_available(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="conda: command not found")
        with patch("subprocess.run", return_value=mock_result):
            result = run_tool_check(["conda", "--version"])
        assert result["available"] is False
        assert result["returncode"] == 1

    def test_stderr_used_when_stdout_empty(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="something went wrong")
        with patch("subprocess.run", return_value=mock_result):
            result = run_tool_check(["bad-tool"])
        assert "something went wrong" in result["output"]

    def test_file_not_found_returns_structured_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_tool_check(["nonexistent-binary", "--version"])
        assert result["available"] is False
        assert result["returncode"] == -1
        assert "not found" in result["output"]

    def test_timeout_returns_structured_error(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=10)):
            result = run_tool_check(["slow-tool"])
        assert result["available"] is False
        assert result["returncode"] == -1
        assert "timed out" in result["output"]

    def test_never_raises(self):
        """run_tool_check must never propagate exceptions."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            try:
                run_tool_check(["anything"])
            except Exception as exc:  # pragma: no cover
                pytest.fail(f"run_tool_check raised unexpectedly: {exc}")


# ── build_mlc_cli_command ─────────────────────────────────────────────────────

class TestBuildMlcCliCommand:
    """build_mlc_cli_command(req) → list[str] for go run . build ..."""

    def _req(self, **kwargs) -> BuildRequest:
        return BuildRequest(**kwargs)

    def test_command_starts_with_go_run(self):
        cmd = build_mlc_cli_command(self._req())
        assert cmd[:3] == ["go", "run", "."]

    def test_subcommand_is_build(self):
        cmd = build_mlc_cli_command(self._req())
        assert cmd[3] == "build"

    def test_os_is_always_linux(self):
        cmd = build_mlc_cli_command(self._req())
        idx = cmd.index("--os")
        assert cmd[idx + 1] == "linux"

    def test_default_action_full(self):
        cmd = build_mlc_cli_command(self._req())
        idx = cmd.index("--action")
        assert cmd[idx + 1] == "full"

    def test_custom_action_passed_through(self):
        cmd = build_mlc_cli_command(self._req(action="install-wheels"))
        idx = cmd.index("--action")
        assert cmd[idx + 1] == "install-wheels"

    def test_cuda_arch_default(self):
        cmd = build_mlc_cli_command(self._req())
        idx = cmd.index("--cuda-arch")
        assert cmd[idx + 1] == "86"

    def test_cuda_arch_custom(self):
        cmd = build_mlc_cli_command(self._req(cuda_arch="89"))
        idx = cmd.index("--cuda-arch")
        assert cmd[idx + 1] == "89"

    def test_cutlass_disabled_by_default(self):
        cmd = build_mlc_cli_command(self._req())
        idx = cmd.index("--cutlass")
        assert cmd[idx + 1] == "n"

    def test_cutlass_can_be_enabled(self):
        cmd = build_mlc_cli_command(self._req(cutlass="y"))
        idx = cmd.index("--cutlass")
        assert cmd[idx + 1] == "y"

    def test_flash_infer_disabled_by_default(self):
        cmd = build_mlc_cli_command(self._req())
        idx = cmd.index("--flash-infer")
        assert cmd[idx + 1] == "n"

    def test_all_flags_present(self):
        """Every expected flag must appear in the command."""
        cmd = build_mlc_cli_command(self._req())
        expected_flags = [
            "--os", "--action", "--tvm-source", "--cuda", "--cuda-arch",
            "--cutlass", "--cublas", "--flash-infer", "--rocm",
            "--vulkan", "--opencl", "--build-wheels", "--force-clone",
        ]
        for flag in expected_flags:
            assert flag in cmd, f"Missing flag: {flag}"

    def test_returns_list_of_strings(self):
        cmd = build_mlc_cli_command(self._req())
        assert isinstance(cmd, list)
        assert all(isinstance(item, str) for item in cmd)


# ── try_restore_metadata ──────────────────────────────────────────────────────

class TestTryRestoreMetadata:
    """try_restore_metadata(metadata_path) → bool success."""

    @patch("app.helpers.subprocess.run")
    def test_missing_file_restores(self, mock_run, tmp_path):
        """If file is missing, calls git checkout and returns True if file appears."""
        meta = tmp_path / ".upstream-sha.json"
        
        def fake_git(cmd, **kwargs):
            meta.write_text(json.dumps({"pinned_sha": "restored"}))
            return MagicMock(returncode=0)
        mock_run.side_effect = fake_git
        
        result = try_restore_metadata(meta)
        assert result is True
        assert json.loads(meta.read_text())["pinned_sha"] == "restored"
        mock_run.assert_called_once()

    @patch("app.helpers.subprocess.run")
    def test_malformed_file_restores(self, mock_run, tmp_path):
        """If file is malformed JSON, calls git checkout and returns True if fixed."""
        meta = tmp_path / ".upstream-sha.json"
        meta.write_text("not json")
        
        def fake_git(cmd, **kwargs):
            meta.write_text(json.dumps({"pinned_sha": "restored"}))
            return MagicMock(returncode=0)
        mock_run.side_effect = fake_git
        
        result = try_restore_metadata(meta)
        assert result is True
        assert json.loads(meta.read_text())["pinned_sha"] == "restored"
        mock_run.assert_called_once()

    @patch("app.helpers.subprocess.run")
    def test_valid_file_not_restored(self, mock_run, tmp_path):
        """If file is valid, does nothing and returns True."""
        meta = tmp_path / ".upstream-sha.json"
        meta.write_text(json.dumps({"pinned_sha": "current"}))
        
        result = try_restore_metadata(meta)
        assert result is True
        assert json.loads(meta.read_text())["pinned_sha"] == "current"
        mock_run.assert_not_called()

    @patch("app.helpers.subprocess.run")
    def test_recovery_fails(self, mock_run, tmp_path):
        """If git checkout fails and file remains missing, returns False."""
        meta = tmp_path / ".upstream-sha.json"
        mock_run.return_value = MagicMock(returncode=1)
        
        result = try_restore_metadata(meta)
        assert result is False
        assert not meta.exists()
