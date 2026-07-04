"""
tests/test_compile.py
~~~~~~~~~~~~~~~~~~~~~
Tests for the ``build_compile_command`` helper and the ``POST /compile``
route.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.helpers import build_compile_command
from app.main import CompileRequest


# -- build_compile_command -----------------------------------------------------

class TestBuildCompileCommand:
    """build_compile_command(req) -> list[str] for go run . compile ..."""

    def _req(self, **kwargs) -> CompileRequest:
        model = kwargs.pop("model", "models/Llama-3-8B")
        return CompileRequest(model=model, **kwargs)

    def test_command_starts_with_go_run(self):
        cmd = build_compile_command(self._req())
        assert cmd[:3] == ["go", "run", "."]

    def test_subcommand_is_compile(self):
        cmd = build_compile_command(self._req())
        assert cmd[3] == "compile"

    def test_os_is_always_linux(self):
        cmd = build_compile_command(self._req())
        idx = cmd.index("--os")
        assert cmd[idx + 1] == "linux"

    def test_model_is_passed_through(self):
        cmd = build_compile_command(self._req(model="models/Mistral-7B"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "models/Mistral-7B"

    def test_default_quant_is_q4f16_1(self):
        cmd = build_compile_command(self._req())
        idx = cmd.index("--quant")
        assert cmd[idx + 1] == "q4f16_1"

    def test_custom_quant_passed_through(self):
        cmd = build_compile_command(self._req(quant="q0f32"))
        idx = cmd.index("--quant")
        assert cmd[idx + 1] == "q0f32"

    def test_default_device_is_cuda(self):
        cmd = build_compile_command(self._req())
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "cuda"

    def test_custom_device_passed_through(self):
        cmd = build_compile_command(self._req(device="vulkan"))
        idx = cmd.index("--device")
        assert cmd[idx + 1] == "vulkan"

    def test_output_omitted_when_empty(self):
        cmd = build_compile_command(self._req(output=""))
        assert "--output" not in cmd

    def test_output_included_when_provided(self):
        cmd = build_compile_command(self._req(output="dist/my-model-MLC"))
        assert "--output" in cmd
        idx = cmd.index("--output")
        assert cmd[idx + 1] == "dist/my-model-MLC"

    def test_required_flags_present(self):
        cmd = build_compile_command(self._req())
        for flag in ["--os", "--model", "--quant", "--device"]:
            assert flag in cmd, f"Missing expected flag: {flag}"

    def test_returns_list_of_strings(self):
        cmd = build_compile_command(self._req())
        assert isinstance(cmd, list)
        assert all(isinstance(item, str) for item in cmd)


# -- POST /compile route -------------------------------------------------------

class TestCompileRouteRepoMissing:
    """When the mlc-cli repo does not exist /compile must fail cleanly."""

    def test_returns_200_with_sse_content_type(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/compile", json={"model": "models/Llama-3-8B"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streams_error_message(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/compile", json={"model": "models/Llama-3-8B"})
        body = resp.text
        assert "[ERROR]" in body
        assert "mlc-cli" in body.lower()

    def test_error_hints_at_repo_status(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/compile", json={"model": "models/Llama-3-8B"})
        assert "repo-status" in resp.text


class TestCompileRouteRepoPresent:
    """When the repo exists, /compile should build the right command and stream."""

    def _fake_stream(self, lines: list[str]):
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def _make_artifact(self, base: Path, name: str) -> Path:
        artifact = base / "dist" / name
        artifact.mkdir(parents=True)
        (artifact / "mlc-chat-config.json").write_text("{}")
        return artifact

    def test_returns_200_sse_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "Llama-3-8B-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        fake_stream = self._fake_stream(["data: compiling...\n\n", "data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)
        resp = client.post("/compile", json={"model": "Llama-3-8B"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_done_marker_present_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "Llama-3-8B-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)
        resp = client.post("/compile", json={"model": "Llama-3-8B"})
        assert "[DONE]" in resp.text

    def test_default_fields_accepted(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "Llama-3-8B-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)
        resp = client.post("/compile", json={"model": "Llama-3-8B"})
        assert resp.status_code == 200

    def test_missing_model_field_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/compile", json={})
        assert resp.status_code == 422

    def test_invalid_device_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/compile", json={"model": "Llama-3-8B", "device": "tpu"})
        assert resp.status_code == 422

    def test_cache_control_header_set(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "Llama-3-8B-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)
        resp = client.post("/compile", json={"model": "Llama-3-8B"})
        assert resp.headers.get("cache-control") == "no-cache"


# -- /compile model resolution -------------------------------------------------

class TestCompileWithModelResolution:
    """/compile accepts HF IDs, short names, artifact folder names, and exact paths."""

    def _fake_stream(self, lines: list[str]):
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def _make_artifact(self, base: Path, name: str) -> Path:
        artifact = base / "dist" / name
        artifact.mkdir(parents=True)
        (artifact / "mlc-chat-config.json").write_text("{}")
        return artifact

    def test_hf_id_resolves_to_matching_artifact(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))
        resp = client.post("/compile", json={"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
        assert resp.status_code == 200
        assert "[ERROR]" not in resp.text
        assert "[DONE]" in resp.text

    def test_short_name_resolves_to_matching_artifact(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))
        resp = client.post("/compile", json={"model": "TinyLlama-1.1B-Chat-v1.0"})
        assert resp.status_code == 200
        assert "[DONE]" in resp.text

    def test_exact_artifact_path_passes_through(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        artifact = self._make_artifact(fake_repo, "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))
        resp = client.post("/compile", json={"model": str(artifact)})
        assert resp.status_code == 200
        assert "[DONE]" in resp.text

    def test_no_artifact_streams_error_not_found(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "dist").mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/compile", json={"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
        assert resp.status_code == 200
        assert "[ERROR]" in resp.text
        assert "quantize" in resp.text.lower() or "artifact" in resp.text.lower()

    def test_multiple_artifacts_streams_error_not_silent_pick(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        self._make_artifact(fake_repo, "TinyLlama-1.1B-Chat-q4f16_1-MLC")
        self._make_artifact(fake_repo, "TinyLlama-1.1B-Chat-py313-q4f16_1-MLC")
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/compile", json={"model": "TinyLlama"})
        assert resp.status_code == 200
        assert "[ERROR]" in resp.text
        assert "/artifacts" in resp.text or "exact" in resp.text.lower()
