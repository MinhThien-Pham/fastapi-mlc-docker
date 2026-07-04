"""
tests/unit/test_quantize.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the ``build_quantize_command`` helper and the ``POST /quantize``
route, plus unit tests for the new conv_template helpers:

- get_supported_conv_templates
- infer_conv_template_for_model
- prepare_conv_template_for_quantize

Helper tests:  fast, no I/O, no FastAPI.
Route tests:   use TestClient + monkeypatching so no real GPU, Conda, or
               mlc-cli clone is required.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.helpers import (
    build_quantize_command,
    get_supported_conv_templates,
    infer_conv_template_for_model,
    prepare_conv_template_for_quantize,
    _FALLBACK_CONV_TEMPLATES,
)
from app.main import QuantizeRequest


# ── build_quantize_command ─────────────────────────────────────────────────────

class TestBuildQuantizeCommand:
    """build_quantize_command(req) → list[str] for go run . quantize ..."""

    def _req(self, **kwargs) -> QuantizeRequest:
        model = kwargs.pop("model", "models/Llama-3-8B")
        conv_template = kwargs.pop("conv_template", "llama-3")
        return QuantizeRequest(model=model, conv_template=conv_template, **kwargs)

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

    def test_conv_template_passed_through(self):
        cmd = build_quantize_command(self._req(conv_template="tinyllama_v1_0"))
        idx = cmd.index("--template")
        assert cmd[idx + 1] == "tinyllama_v1_0"

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


# ── get_supported_conv_templates ──────────────────────────────────────────────

class TestGetSupportedConvTemplates:
    """Parse CONV_TEMPLATES from gen_config.py, or fall back."""

    def test_falls_back_when_path_missing(self, tmp_path):
        result = get_supported_conv_templates(tmp_path / "nonexistent")
        assert result == _FALLBACK_CONV_TEMPLATES

    def test_parses_real_gen_config_py(self, tmp_path):
        """Write a fake gen_config.py with a small CONV_TEMPLATES set and parse it."""
        gen_cfg = tmp_path / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg.mkdir(parents=True)
        (gen_cfg / "gen_config.py").write_text(textwrap.dedent("""\
            CONV_TEMPLATES = {
                "llama-3",
                "chatml",
                "tinyllama_v1_0",
                "deepseek_v3",
            }
        """))
        result = get_supported_conv_templates(tmp_path)
        assert result == frozenset({"llama-3", "chatml", "tinyllama_v1_0", "deepseek_v3"})

    def test_falls_back_on_syntax_error(self, tmp_path):
        gen_cfg = tmp_path / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg.mkdir(parents=True)
        (gen_cfg / "gen_config.py").write_text("CONV_TEMPLATES = {invalid python!!")
        result = get_supported_conv_templates(tmp_path)
        assert result == _FALLBACK_CONV_TEMPLATES

    def test_falls_back_when_no_conv_templates_assignment(self, tmp_path):
        gen_cfg = tmp_path / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg.mkdir(parents=True)
        (gen_cfg / "gen_config.py").write_text("OTHER_SET = {'a', 'b'}\n")
        result = get_supported_conv_templates(tmp_path)
        assert result == _FALLBACK_CONV_TEMPLATES

    def test_fallback_contains_expected_templates(self):
        assert "tinyllama_v1_0" in _FALLBACK_CONV_TEMPLATES
        assert "deepseek_v3" in _FALLBACK_CONV_TEMPLATES
        assert "llama-3" in _FALLBACK_CONV_TEMPLATES
        assert "qwen3_5" in _FALLBACK_CONV_TEMPLATES

    def test_fallback_does_not_contain_harmony(self):
        assert "harmony" not in _FALLBACK_CONV_TEMPLATES

    def test_fallback_does_not_contain_gpt_oss(self):
        assert "gpt-oss" not in _FALLBACK_CONV_TEMPLATES


# ── infer_conv_template_for_model ─────────────────────────────────────────────

class TestInferConvTemplateForModel:
    """Heuristic mapping from model name to template name."""

    S = _FALLBACK_CONV_TEMPLATES  # use fallback as the supported set

    def test_tinyllama(self):
        assert infer_conv_template_for_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", self.S) == "tinyllama_v1_0"

    def test_tinyllama_basename(self):
        assert infer_conv_template_for_model("TinyLlama-1.1B-Chat-v1.0", self.S) == "tinyllama_v1_0"

    def test_deepseek_v3(self):
        assert infer_conv_template_for_model("deepseek-ai/DeepSeek-V3", self.S) == "deepseek_v3"

    def test_deepseek_v3_dot(self):
        # e.g. DeepSeek-V3.2
        assert infer_conv_template_for_model("deepseek-ai/DeepSeek-V3.2", self.S) == "deepseek_v3"

    def test_deepseek_v2(self):
        assert infer_conv_template_for_model("deepseek-ai/DeepSeek-V2-Chat", self.S) == "deepseek_v2"

    def test_deepseek_r1_qwen(self):
        result = infer_conv_template_for_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", self.S)
        assert result == "deepseek_r1_qwen"

    def test_deepseek_r1_llama(self):
        result = infer_conv_template_for_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", self.S)
        assert result == "deepseek_r1_llama"

    def test_llama3(self):
        assert infer_conv_template_for_model("meta-llama/Meta-Llama-3-8B", self.S) == "llama-3"

    def test_llama3_1(self):
        assert infer_conv_template_for_model("meta-llama/Llama-3.1-8B-Instruct", self.S) == "llama-3_1"

    def test_llama2(self):
        assert infer_conv_template_for_model("meta-llama/Llama-2-7b-chat-hf", self.S) == "llama-2"

    def test_qwen3_5(self):
        assert infer_conv_template_for_model("Qwen/Qwen3.5-7B-Instruct", self.S) == "qwen3_5"

    def test_qwen3(self):
        assert infer_conv_template_for_model("Qwen/Qwen3-7B-Instruct", self.S) == "qwen3"

    def test_qwen2(self):
        assert infer_conv_template_for_model("Qwen/Qwen2.5-7B-Instruct", self.S) == "qwen2"

    def test_phi4(self):
        assert infer_conv_template_for_model("microsoft/phi-4", self.S) == "phi-4"

    def test_phi3(self):
        # "phi-3-mini" — after normalisation: "phi_3_mini".
        # The phi-4 pattern requires a non-digit after the "4", so phi_3 won't match it.
        assert infer_conv_template_for_model("microsoft/Phi-3-mini-instruct", self.S) == "phi-3"

    def test_gemma3(self):
        assert infer_conv_template_for_model("google/gemma-3-4b-it", self.S) == "gemma3_instruction"

    def test_gemma(self):
        assert infer_conv_template_for_model("google/gemma-7b-it", self.S) == "gemma_instruction"

    def test_mistral(self):
        assert infer_conv_template_for_model("mistralai/Mistral-7B-Instruct-v0.3", self.S) == "mistral_default"

    def test_mixtral(self):
        assert infer_conv_template_for_model("mistralai/Mixtral-8x7B-Instruct-v0.1", self.S) == "mistral_default"

    def test_ministral3(self):
        assert infer_conv_template_for_model("mistralai/Ministral-3b-instruct", self.S) == "ministral3"

    def test_nemotron(self):
        assert infer_conv_template_for_model("nvidia/Llama-3.1-Nemotron-70B-Instruct", self.S) == "nemotron"

    def test_unknown_model_returns_none(self):
        assert infer_conv_template_for_model("some-obscure-custom-model", self.S) is None

    def test_gpt_oss_returns_none(self):
        # gpt-oss matches no rule (no entry in supported_templates)
        assert infer_conv_template_for_model("openai/gpt-oss-20b", self.S) is None

    def test_returns_none_if_inferred_not_in_supported(self):
        """If the inferred template is NOT in supported_templates, return None."""
        tiny_supported = frozenset({"llama-3", "chatml"})  # tinyllama_v1_0 absent
        assert infer_conv_template_for_model("TinyLlama/TinyLlama-1.1B", tiny_supported) is None


# ── prepare_conv_template_for_quantize ───────────────────────────────────────

class TestPrepareConvTemplateForQuantize:
    """prepare_conv_template_for_quantize resolution logic."""

    def _fake_mlc_path(self, tmp_path: Path, templates: set[str] | None = None) -> Path:
        """Write a minimal gen_config.py so get_supported_conv_templates can parse it."""
        gen_cfg_dir = tmp_path / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg_dir.mkdir(parents=True)
        t = templates if templates is not None else set(_FALLBACK_CONV_TEMPLATES)
        lines = "CONV_TEMPLATES = {\n" + "".join(f'    "{x}",\n' for x in sorted(t)) + "}\n"
        (gen_cfg_dir / "gen_config.py").write_text(lines)
        return tmp_path

    # ── auto mode ────────────────────────────────────────────────────────────

    def test_auto_tinyllama_resolves(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto", mlc
        )
        assert template == "tinyllama_v1_0"
        assert any("[INFO]" in m and "tinyllama_v1_0" in m for m in msgs)

    def test_auto_deepseek_v3_resolves(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "deepseek-ai/DeepSeek-V3.2", "auto", mlc
        )
        assert template == "deepseek_v3"
        assert any("deepseek_v3" in m for m in msgs)

    def test_auto_qwen3_5_resolves(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "Qwen/Qwen3.5-7B-Instruct", "auto", mlc
        )
        assert template == "qwen3_5"

    def test_auto_gpt_oss_raises(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        with pytest.raises(ValueError, match="gpt-oss"):
            prepare_conv_template_for_quantize("openai/gpt-oss-20b", "auto", mlc)

    def test_auto_harmony_raises(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        with pytest.raises(ValueError, match="[Hh]armony"):
            prepare_conv_template_for_quantize("some-org/HarmonyModel-7B", "auto", mlc)

    def test_auto_unknown_model_raises(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        with pytest.raises(ValueError, match="Could not infer conv_template"):
            prepare_conv_template_for_quantize("some-totally-obscure-model", "auto", mlc)

    # ── explicit mode ─────────────────────────────────────────────────────────

    def test_explicit_matching_template_no_warning(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_v1_0", mlc
        )
        assert template == "tinyllama_v1_0"
        assert not any("[WARNING]" in m for m in msgs)

    def test_explicit_mismatched_but_supported_emits_warning(self, tmp_path):
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "chatml", mlc
        )
        assert template == "chatml"
        assert any("[WARNING]" in m for m in msgs)

    def test_explicit_unsupported_template_emits_warning(self, tmp_path):
        """'harmony' is not in supported list — should warn but pass through."""
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "some-model", "harmony", mlc
        )
        assert template == "harmony"
        assert any("[WARNING]" in m and "harmony" in m for m in msgs)

    def test_explicit_unsupported_gpt_oss_template_warns_and_proceeds(self, tmp_path):
        """Explicit gpt-oss template name: warn but let mlc-cli handle it."""
        mlc = self._fake_mlc_path(tmp_path)
        template, msgs = prepare_conv_template_for_quantize(
            "openai/gpt-oss-20b", "gpt-oss", mlc
        )
        assert template == "gpt-oss"
        assert any("[WARNING]" in m for m in msgs)


# ── POST /quantize route ───────────────────────────────────────────────────────

class TestQuantizeRouteRepoMissing:
    """When the mlc-cli repo does not exist /quantize must fail cleanly."""

    def test_returns_200_with_sse_content_type(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B", "conv_template": "llama-3"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_streams_error_message(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B", "conv_template": "llama-3"})
        body = resp.text
        assert "[ERROR]" in body
        assert "mlc-cli" in body.lower()

    def test_error_hints_at_repo_status(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.post("/quantize", json={"model": "models/Llama-3-8B", "conv_template": "llama-3"})
        assert "repo-status" in resp.text


class TestQuantizeRouteConvTemplate:
    """conv_template auto-inference and warning behavior on /quantize."""

    def _fake_stream(self, lines: list[str]):
        """Return an async generator that yields the given SSE lines."""
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def _make_repo(self, tmp_path, templates: set[str] | None = None):
        """Create a fake mlc-cli workspace with a parseable gen_config.py."""
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        gen_cfg_dir = fake_repo / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg_dir.mkdir(parents=True)
        t = templates if templates is not None else set(_FALLBACK_CONV_TEMPLATES)
        lines = "CONV_TEMPLATES = {\n" + "".join(f'    "{x}",\n' for x in sorted(t)) + "}\n"
        (gen_cfg_dir / "gen_config.py").write_text(lines)
        return fake_repo

    def test_auto_tinyllama_resolves_before_building_command(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        # Make TinyLlama model dir exist so path validation passes
        (fake_repo / "models" / "TinyLlama-1.1B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        captured = {}
        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        original_build = __import__("app.helpers", fromlist=["build_quantize_command"]).build_quantize_command
        def capturing_build(req):
            captured["conv_template"] = req.conv_template
            return original_build(req)
        monkeypatch.setattr("app.main.build_quantize_command", capturing_build)

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "TinyLlama-1.1B"),
            "conv_template": "auto",
        })
        assert resp.status_code == 200
        assert captured.get("conv_template") == "tinyllama_v1_0"

    def test_auto_resolves_info_line_emitted(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        (fake_repo / "models" / "TinyLlama-1.1B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "TinyLlama-1.1B"),
            "conv_template": "auto",
        })
        assert "[INFO]" in resp.text
        assert "tinyllama_v1_0" in resp.text

    def test_auto_unknown_model_returns_400(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        resp = client.post("/quantize", json={
            "model": "some-totally-unknown-model-xyz",
            "conv_template": "auto",
        })
        assert resp.status_code == 400
        assert "conv_template" in resp.json()["detail"].lower()

    def test_auto_gpt_oss_returns_400(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        resp = client.post("/quantize", json={
            "model": "openai/gpt-oss-20b",
            "conv_template": "auto",
        })
        assert resp.status_code == 400
        assert "gpt-oss" in resp.json()["detail"]

    def test_explicit_mismatch_streams_warning_and_proceeds(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        (fake_repo / "models" / "TinyLlama-1.1B").mkdir(parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "TinyLlama-1.1B"),
            "conv_template": "chatml",  # mismatch — TinyLlama maps to tinyllama_v1_0
        })
        assert resp.status_code == 200
        assert "[WARNING]" in resp.text
        assert "[DONE]" in resp.text

    def test_explicit_unsupported_template_streams_warning_and_proceeds(self, client, monkeypatch, tmp_path):
        """harmony is not in supported list — warn but proceed, no 400."""
        fake_repo = self._make_repo(tmp_path)
        (fake_repo / "models" / "Llama-3-8B").mkdir(exist_ok=True, parents=True)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "conv_template": "harmony",
        })
        assert resp.status_code == 200
        assert "[WARNING]" in resp.text
        assert "[DONE]" in resp.text


class TestQuantizeRouteRepoPresent:
    """When the repo exists, /quantize should build the right command and stream."""

    def _fake_stream(self, lines: list[str]):
        """Return an async generator that yields the given SSE lines."""
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def _make_repo(self, tmp_path, templates: set[str] | None = None):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        (fake_repo / "models" / "Llama-3-8B").mkdir(parents=True)
        gen_cfg_dir = fake_repo / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg_dir.mkdir(parents=True)
        t = templates if templates is not None else set(_FALLBACK_CONV_TEMPLATES)
        lines = "CONV_TEMPLATES = {\n" + "".join(f'    "{x}",\n' for x in sorted(t)) + "}\n"
        (gen_cfg_dir / "gen_config.py").write_text(lines)
        return fake_repo

    def test_returns_200_sse_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: quantizing...\n\n", "data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "conv_template": "llama-3",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_done_marker_present_on_success(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "conv_template": "llama-3",
        })
        assert "[DONE]" in resp.text

    def test_default_auto_with_unknown_model_returns_400(self, client, monkeypatch, tmp_path):
        """The default conv_template is 'auto'; unknown model → 400."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        resp = client.post("/quantize", json={"model": "unknown-model-xyzxyz"})
        assert resp.status_code == 400

    def test_missing_model_field_returns_422(self, client, monkeypatch, tmp_path):
        """``model`` is required; omitting it should return a validation error."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={})
        assert resp.status_code == 422

    def test_invalid_quant_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "quant": "badquant",
            "conv_template": "llama-3",
        })
        assert resp.status_code == 422

    def test_invalid_device_returns_422(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "device": "tpu",
            "conv_template": "llama-3",
        })
        assert resp.status_code == 422

    def test_cache_control_header_set(self, client, monkeypatch, tmp_path):
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        fake_stream = self._fake_stream(["data: [DONE]\n\n"])
        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={
            "model": str(fake_repo / "models" / "Llama-3-8B"),
            "conv_template": "llama-3",
        })
        assert resp.headers.get("cache-control") == "no-cache"


# ── GET /conv-templates ───────────────────────────────────────────────────────

class TestConvTemplatesEndpoint:
    """GET /conv-templates lists supported templates."""

    def test_returns_200(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path)
        resp = client.get("/conv-templates")
        assert resp.status_code == 200

    def test_returns_sorted_list(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path)
        body = client.get("/conv-templates").json()
        templates = body["templates"]
        assert templates == sorted(templates)

    def test_default_is_auto(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path)
        body = client.get("/conv-templates").json()
        assert body["default"] == "auto"

    def test_contains_known_templates(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path)
        body = client.get("/conv-templates").json()
        templates = set(body["templates"])
        assert "tinyllama_v1_0" in templates
        assert "deepseek_v3" in templates
        assert "llama-3" in templates


# ── /quantize with Hugging Face model IDs ────────────────────────────────────

class TestQuantizeWithHfModelId:
    """HF model IDs should pass through /quantize unchanged and never trigger
    the local-path-not-found error.  Local-looking paths should NOT be treated
    as HF IDs even when the path does not yet exist on disk.
    """

    def _fake_stream(self, lines: list[str]):
        async def _gen(*_args, **_kwargs):
            for line in lines:
                yield line
        return _gen

    def _make_repo(self, tmp_path, templates: set[str] | None = None):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        gen_cfg_dir = fake_repo / "mlc-llm" / "python" / "mlc_llm" / "interface"
        gen_cfg_dir.mkdir(parents=True)
        t = templates if templates is not None else {"tinyllama_v1_0", "llama-3", "chatml"}
        lines = "CONV_TEMPLATES = {\n" + "".join(f'    "{x}",\n' for x in sorted(t)) + "}\n"
        (gen_cfg_dir / "gen_config.py").write_text(lines)
        return fake_repo

    def test_hf_model_id_does_not_stream_path_error(self, client, monkeypatch, tmp_path):
        """HF ID must not trigger the 'model path not found' SSE error."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "conv_template": "tinyllama_v1_0",  # explicit to skip auto-inference
        })
        assert resp.status_code == 200
        assert "[ERROR] model path not found" not in resp.text
        assert "[DONE]" in resp.text

    def test_hf_model_id_reaches_mlc_cli_unchanged(self, client, monkeypatch, tmp_path):
        """The HF ID must be forwarded to mlc-cli --model unchanged (no path mangling)."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        captured: list[list[str]] = []

        async def fake_stream(cmd, *args, **kwargs):
            captured.append(cmd)
            yield "data: [DONE]\n\n"

        monkeypatch.setattr("app.main.stream_subprocess", fake_stream)

        resp = client.post("/quantize", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "conv_template": "tinyllama_v1_0",
        })
        assert resp.status_code == 200
        assert captured, "stream_subprocess was never called"
        cmd = captured[0]
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_local_looking_path_not_treated_as_hf_id(self, client, monkeypatch, tmp_path):
        """models/SomeModel must NOT be treated as an HF ID — it must trigger path-not-found."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": "models/NonExistentModel",
            "conv_template": "llama-3",  # explicit to skip auto-inference
        })
        assert resp.status_code == 200
        assert "[ERROR] model path not found" in resp.text

    def test_dist_prefix_not_treated_as_hf_id(self, client, monkeypatch, tmp_path):
        """dist/SomeArtifact-MLC must NOT be treated as an HF ID."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": "dist/NonExistentArtifact-MLC",
            "conv_template": "llama-3",
        })
        assert resp.status_code == 200
        assert "[ERROR] model path not found" in resp.text

    def test_error_message_includes_hf_tip(self, client, monkeypatch, tmp_path):
        """When a local path is not found, the error should hint about HF IDs."""
        fake_repo = self._make_repo(tmp_path)
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)
        monkeypatch.setattr("app.main.stream_subprocess", self._fake_stream(["data: [DONE]\n\n"]))

        resp = client.post("/quantize", json={
            "model": "models/NonExistentModel",
            "conv_template": "llama-3",
        })
        assert resp.status_code == 200
        # The error stream must contain a helpful tip pointing to HF Hub
        assert "Hugging Face" in resp.text or "Owner/ModelName" in resp.text
