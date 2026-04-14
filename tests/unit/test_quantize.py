"""
tests/test_quantize.py
~~~~~~~~~~~~~~~~~~~~~
Tests for the ``build_quantize_command`` helper and the ``POST /quantize``
route.

Helper tests:  fast, no I/O, no FastAPI.
Route tests:   use TestClient + monkeypatching so no real GPU, Conda, or
               mlc-cli clone is required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.helpers import build_quantize_command
from app.main import QuantizeRequest


# ── build_quantize_command ─────────────────────────────────────────────────────

class TestBuildQuantizeCommand:
    """build_quantize_command(req) → list[str] for go run . quantize ..."""

    def _req(self, **kwargs) -> QuantizeRequest:
        model = kwargs.pop("model", "models/Llama-3-8B")
        return QuantizeRequest(model=model, **kwargs)

    def test_command_starts_with_go_run(self):
        cmd = build_quantize_command(self._req())
        assert cmd[:3] == ["go", "run", "."]

    def test_subcommand_is_quantize(self):
        cmd = build_quantize_command(self._req())
        assert cmd[3] == "quantize"

    def test_os_is_always_linux(self):
        cmd = build_quantize_command(self._req())
        idx = cmd.index("--os")
        assert cmd[idx + 1] == "linux"

    def test_model_is_passed_through(self):
        cmd = build_quantize_command(self._req(model="models/Mistral-7B"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "models/Mistral-7B"

    def test_default_quant_is_q4f16_1(self):
        cmd = build_quantize_command(self._req())
        idx = cmd.index("--quant")
        assert cmd[idx + 1] == "q4f16_1"

    def test_custom_quant_passed_through(self):
        cmd = build_quantize_command(self._req(quant="q0f32"))
        idx = cmd.index("--quant")
        assert cmd[idx + 1] == "q0f32"

    def test_default_device_is_cuda(self):
        cmd = build_quantize_command(self._req())
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "cuda"

    def test_custom_device_passed_through(self):
        cmd = build_quantize_command(self._req(device="vulkan"))
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "vulkan"

    def test_default_conv_template_is_llama3(self):
        cmd = build_quantize_command(self._req())
        idx = cmd.index("--template")
        assert cmd[idx + 1] == "llama-3"

    def test_custom_conv_template_passed_through(self):
        cmd = build_quantize_command(self._req(conv_template="chatml"))
        idx = cmd.index("--template")
        assert cmd[idx + 1] == "chatml"

    def test_output_omitted_when_empty(self):
        """mlc-cli derives a default output path when --output is absent."""
        cmd = build_quantize_command(self._req(output=""))
        assert "--output" not in cmd

    def test_output_included_when_provided(self):
        cmd = build_quantize_command(self._req(output="dist/my-model-MLC"))
        assert "--output" in cmd
        idx = cmd.index("--output")
        assert cmd[idx + 1] == "dist/my-model-MLC"

    def test_required_flags_present(self):
        cmd = build_quantize_command(self._req())
        for flag in ["--os", "--model", "--quant", "--device", "--template"]:
            assert flag in cmd, f"Missing expected flag: {flag}"

    def test_returns_list_of_strings(self):
        cmd = build_quantize_command(self._req())
        assert isinstance(cmd, list)
        assert all(isinstance(item, str) for item in cmd)


# ── POST /quantize route ───────────────────────────────────────────────────────

class TestQuantizeRouteRepoMissing:
    """When the mlc-cli repo does not exist /quantize must fail cleanly."""

    def test_returns_200_with_sse_content_type(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streams_error_message(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        body = resp.text
        assert "[ERROR]" in body
        assert "mlc-cli" in body.lower()

    def test_error_hints_at_ensure_repo_exists(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert "ensure-repo-exists" in resp.text


class TestQuantizeRouteRepoPresent:
    """When the repo exists, /quantize should build the right command and stream."""

    def _fake_stream(self, lines: list[str]):
        """Return an async generator that yields the given SSE lines."""
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def test_returns_200_sse_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: quantizing...\n\n", "data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_done_marker_present_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert "[DONE]" in resp.text

    def test_default_fields_accepted(self, client, monkeypatch, tmp_path):
        """Sending only 'model' (all others at default) must not 422."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert resp.status_code == 200

    def test_missing_model_field_returns_422(self, client, monkeypatch, tmp_path):
        """``model`` is required; omitting it should return a validation error."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={})
        assert resp.status_code == 422

    def test_invalid_quant_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B", "quant": "badquant"})
        assert resp.status_code == 422

    def test_invalid_device_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B", "device": "tpu"})
        assert resp.status_code == 422

    def test_cache_control_header_set(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={"model": "models/Llama-3-8B"})
        assert resp.headers.get("cache-control") == "no-cache"
