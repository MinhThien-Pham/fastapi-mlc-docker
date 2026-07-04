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


# ── is_hf_model_id ────────────────────────────────────────────────────────────

from app.helpers import is_hf_model_id  # noqa: E402


class TestIsHfModelId:
    """is_hf_model_id(model) → bool."""

    # ── Accepted: real HF IDs ────────────────────────────────────────────────

    def test_tinyllama_hf_id_accepted(self):
        assert is_hf_model_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is True

    def test_meta_llama_hf_id_accepted(self):
        assert is_hf_model_id("meta-llama/Meta-Llama-3-8B") is True

    def test_org_model_hf_id_accepted(self):
        assert is_hf_model_id("myorg/mymodel") is True

    # ── Rejected: absolute paths ─────────────────────────────────────────────

    def test_unix_absolute_path_rejected(self):
        assert is_hf_model_id("/workspace/mlc-cli/models/Llama-3-8B") is False

    def test_windows_drive_path_rejected(self):
        assert is_hf_model_id("C:/models/Llama-3-8B") is False

    def test_windows_drive_path_backslash_rejected(self):
        assert is_hf_model_id("D:\\models\\Llama-3-8B") is False

    # ── Rejected: well-known local-path prefixes ─────────────────────────────

    def test_models_prefix_rejected(self):
        """models/SomeModel must not be treated as HF ID even if path does not exist."""
        assert is_hf_model_id("models/NonExistentModel") is False

    def test_dist_prefix_rejected(self):
        assert is_hf_model_id("dist/TinyLlama-1.1B-q4f16_1-MLC") is False

    def test_dotslash_prefix_rejected(self):
        assert is_hf_model_id("./local/model") is False

    def test_dotdotslash_prefix_rejected(self):
        assert is_hf_model_id("../sibling/model") is False

    def test_tilde_slash_prefix_rejected(self):
        assert is_hf_model_id("~/models/Llama") is False

    # ── Rejected: no slash (bare name, not an HF ID) ─────────────────────────

    def test_bare_model_name_rejected(self):
        assert is_hf_model_id("TinyLlama") is False

    def test_empty_string_rejected(self):
        assert is_hf_model_id("") is False

    # ── Rejected: existing local path ────────────────────────────────────────

    def test_existing_local_path_rejected(self, tmp_path):
        """A path that actually exists on disk is a local path, not an HF ID."""
        existing = tmp_path / "myorg" / "mymodel"
        existing.mkdir(parents=True)
        # Pass the string form — is_hf_model_id should detect Path.exists()
        result = is_hf_model_id(str(existing))
        assert result is False


# ── resolve_quantized_model_dir ───────────────────────────────────────────────

from app.helpers import resolve_quantized_model_dir  # noqa: E402


class TestResolveQuantizedModelDir:
    """resolve_quantized_model_dir(mlc_cli_path, model, quant) → Path | 'none' | 'multiple'."""

    def _make_artifact(self, base: "Path", name: str) -> "Path":
        artifact = base / "dist" / name
        artifact.mkdir(parents=True)
        (artifact / "mlc-chat-config.json").write_text("{}")
        return artifact

    def test_exact_absolute_path_returned(self, tmp_path):
        artifact = self._make_artifact(tmp_path, "TinyLlama-1.1B-q4f16_1-MLC")
        result = resolve_quantized_model_dir(tmp_path, str(artifact), "q4f16_1")
        assert result == artifact

    def test_exact_relative_path_resolved(self, tmp_path):
        self._make_artifact(tmp_path, "TinyLlama-1.1B-q4f16_1-MLC")
        result = resolve_quantized_model_dir(tmp_path, "dist/TinyLlama-1.1B-q4f16_1-MLC", "q4f16_1")
        assert isinstance(result, type(tmp_path))
        assert result.name == "TinyLlama-1.1B-q4f16_1-MLC"

    def test_short_model_name_resolved(self, tmp_path):
        self._make_artifact(tmp_path, "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC")
        result = resolve_quantized_model_dir(tmp_path, "TinyLlama-1.1B-Chat-v1.0", "q4f16_1")
        assert isinstance(result, type(tmp_path))

    def test_hf_id_uses_basename_for_search(self, tmp_path):
        self._make_artifact(tmp_path, "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC")
        result = resolve_quantized_model_dir(tmp_path, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "q4f16_1")
        assert isinstance(result, type(tmp_path))

    def test_no_match_returns_none_sentinel(self, tmp_path):
        (tmp_path / "dist").mkdir()
        result = resolve_quantized_model_dir(tmp_path, "NonExistentModel", "q4f16_1")
        assert result == "none"

    def test_no_dist_dir_returns_none_sentinel(self, tmp_path):
        result = resolve_quantized_model_dir(tmp_path, "NonExistentModel", "q4f16_1")
        assert result == "none"

    def test_multiple_matches_returns_multiple_sentinel(self, tmp_path):
        self._make_artifact(tmp_path, "TinyLlama-1.1B-Chat-q4f16_1-MLC")
        self._make_artifact(tmp_path, "TinyLlama-1.1B-Chat-py313-q4f16_1-MLC")
        result = resolve_quantized_model_dir(tmp_path, "TinyLlama", "q4f16_1")
        assert result == "multiple"

    def test_dir_without_config_not_matched(self, tmp_path):
        """Directories that look like artifact dirs but lack mlc-chat-config.json are ignored."""
        fake = tmp_path / "dist" / "TinyLlama-1.1B-q4f16_1-MLC"
        fake.mkdir(parents=True)
        # Do NOT write mlc-chat-config.json
        result = resolve_quantized_model_dir(tmp_path, "TinyLlama", "q4f16_1")
        assert result == "none"
