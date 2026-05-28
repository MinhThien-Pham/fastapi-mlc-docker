"""
tests/test_run.py
~~~~~~~~~~~~~~~~~
Tests for the ``build_run_command`` helper and the ``POST /run`` route.
"""
from __future__ import annotations

from pathlib import Path

from app.helpers import build_run_command, resolve_model_lib
from app.main import RunRequest


# ── build_run_command ─────────────────────────────────────────────────────────

class TestBuildRunCommand:
    """build_run_command(req) → list[str] for go run . run ..."""

    def _req(self, **kwargs) -> RunRequest:
        model_name = kwargs.pop("model_name", "Llama-3-8B")
        return RunRequest(model_name=model_name, **kwargs)

    def test_command_starts_with_go_run(self):
        cmd = build_run_command(self._req())
        assert cmd[:4] == ["go", "run", ".", "run"]

    def test_os_is_always_linux(self):
        cmd = build_run_command(self._req())
        assert "--os" in cmd
        assert cmd[cmd.index("--os") + 1] == "linux"

    def test_model_name_passed_through(self):
        cmd = build_run_command(self._req(model_name="my-model"))
        assert "--model-name" in cmd
        assert cmd[cmd.index("--model-name") + 1] == "my-model"

    def test_default_device_is_cuda(self):
        cmd = build_run_command(self._req())
        assert "--device" in cmd
        assert cmd[cmd.index("--device") + 1] == "cuda"

    def test_custom_device_passed_through(self):
        cmd = build_run_command(self._req(device="metal"))
        assert cmd[cmd.index("--device") + 1] == "metal"

    def test_default_profile_is_default(self):
        cmd = build_run_command(self._req())
        assert "--profile" in cmd
        assert cmd[cmd.index("--profile") + 1] == "default"

    def test_custom_profile_passed_through(self):
        cmd = build_run_command(self._req(profile="high"))
        assert cmd[cmd.index("--profile") + 1] == "high"

    def test_model_lib_omitted_when_empty(self):
        cmd = build_run_command(self._req(model_lib=""))
        assert "--model-lib" not in cmd

    def test_model_lib_included_when_provided(self):
        cmd = build_run_command(self._req(model_lib="dist/my-lib.so"))
        assert "--model-lib" in cmd
        assert cmd[cmd.index("--model-lib") + 1] == "dist/my-lib.so"

    def test_model_url_omitted_when_empty(self):
        cmd = build_run_command(self._req(model_url=""))
        assert "--model-url" not in cmd

    def test_model_url_included_when_provided(self):
        cmd = build_run_command(self._req(model_url="https://git.com/model"))
        assert "--model-url" in cmd
        assert cmd[cmd.index("--model-url") + 1] == "https://git.com/model"

    def test_quant_field_does_not_appear_in_command(self):
        """quant is used for auto-resolution only; it must NOT be forwarded to the CLI."""
        cmd = build_run_command(self._req(quant="q4f16_1"))
        assert "--quant" not in cmd


# ── resolve_model_lib ─────────────────────────────────────────────────────────

class TestResolveModelLib:
    """resolve_model_lib() → path | 'multiple' | None"""

    def test_returns_none_when_libs_dir_missing(self, tmp_path):
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "cuda")
        assert result is None

    def test_returns_none_when_no_match(self, tmp_path):
        (tmp_path / "dist" / "libs").mkdir(parents=True)
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "cuda")
        assert result is None

    def test_returns_path_for_so_match(self, tmp_path):
        libs = tmp_path / "dist" / "libs"
        libs.mkdir(parents=True)
        lib = libs / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so"
        lib.touch()
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "cuda")
        assert result == str(lib)

    def test_returns_path_for_dylib_match(self, tmp_path):
        libs = tmp_path / "dist" / "libs"
        libs.mkdir(parents=True)
        lib = libs / "MyModel-q4f16_1-MLC-q4f16_1-metal.dylib"
        lib.touch()
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "metal")
        assert result == str(lib)

    def test_returns_multiple_sentinel_when_both_extensions_exist(self, tmp_path):
        libs = tmp_path / "dist" / "libs"
        libs.mkdir(parents=True)
        (libs / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so").touch()
        (libs / "MyModel-q4f16_1-MLC-q4f16_1-cuda.dylib").touch()
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "cuda")
        assert result == "multiple"

    def test_does_not_match_different_device(self, tmp_path):
        libs = tmp_path / "dist" / "libs"
        libs.mkdir(parents=True)
        (libs / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so").touch()
        result = resolve_model_lib(tmp_path, "MyModel-q4f16_1-MLC", "q4f16_1", "metal")
        assert result is None


# ── POST /run route ───────────────────────────────────────────────────────────

class TestRunRouteRepoMissing:
    """When the mlc-cli repo does not exist /run must fail cleanly."""

    def test_returns_200_with_sse_content_type(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/run", json={"model_name": "mod"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streams_error_message(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/run", json={"model_name": "mod"})
        body = resp.text
        assert "[ERROR]" in body
        assert "mlc-cli" in body.lower()


class TestRunRouteRepoPresent:
    """When the repo exists, /run should stream."""

    def _fake_stream(self, lines: list[str]):
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def test_returns_200_sse_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: loading...\n\n", "data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/run", json={"model_name": "mod"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "[DONE]" in resp.text

    def test_missing_model_name_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/run", json={})
        assert resp.status_code == 422

    def test_quant_field_accepted(self, client, monkeypatch, tmp_path):
        """quant is an optional field; providing it must not cause 422."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/run", json={"model_name": "mod", "quant": "q4f16_1"})
        assert resp.status_code == 200

    def test_auto_resolved_lib_injected_into_command(self, client, monkeypatch, tmp_path):
        """When quant is provided and a matching .so exists, it is auto-resolved."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        libs_dir = fake_repo / "dist" / "libs"
        libs_dir.mkdir(parents=True)
        expected_lib = libs_dir / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so"
        expected_lib.touch()

        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        captured = {}

        async def _capture_stream(cmd, cwd=None):
            captured["cmd"] = cmd
            yield "data: [DONE]\n\n"

        monkeypatch.setattr("app.main.stream_subprocess", _capture_stream)

        client.post("/run", json={
            "model_name": "MyModel-q4f16_1-MLC",
            "quant": "q4f16_1",
            "device": "cuda",
        })

        assert "--model-lib" in captured["cmd"]
        idx = captured["cmd"].index("--model-lib")
        assert captured["cmd"][idx + 1] == str(expected_lib)

    def test_no_lib_found_falls_back_to_jit(self, client, monkeypatch, tmp_path):
        """When quant is provided but no .so exists, JIT fallback (no --model-lib)."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        captured = {}

        async def _capture_stream(cmd, cwd=None):
            captured["cmd"] = cmd
            yield "data: [DONE]\n\n"

        monkeypatch.setattr("app.main.stream_subprocess", _capture_stream)

        client.post("/run", json={
            "model_name": "MyModel-q4f16_1-MLC",
            "quant": "q4f16_1",
            "device": "cuda",
        })

        assert "--model-lib" not in captured.get("cmd", [])

    def test_multiple_libs_returns_error_sse(self, client, monkeypatch, tmp_path):
        """When multiple .so/.dylib match, the route returns an error SSE."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        libs_dir = fake_repo / "dist" / "libs"
        libs_dir.mkdir(parents=True)
        (libs_dir / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so").touch()
        (libs_dir / "MyModel-q4f16_1-MLC-q4f16_1-cuda.dylib").touch()

        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        resp = client.post("/run", json={
            "model_name": "MyModel-q4f16_1-MLC",
            "quant": "q4f16_1",
            "device": "cuda",
        })

        assert resp.status_code == 200
        assert "[ERROR]" in resp.text
        assert "multiple" in resp.text.lower()
