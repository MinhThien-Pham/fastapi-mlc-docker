"""
tests/test_artifacts.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests for the ``discover_artifacts`` helper and the ``GET /artifacts`` route.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.helpers import discover_artifacts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_mock_wheel(base_path: Path, name: str) -> Path:
    p = base_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("fake wheel content")
    return p

def _create_mock_model_dir(base_path: Path, name: str) -> Path:
    p = base_path / "models" / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "mlc-chat-config.json").write_text("{}")
    (p / "ndarray-cache.json").write_text("{}")
    return p

def _create_mock_compiled_lib(base_path: Path, name: str) -> Path:
    p = base_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("fake compiled lib content")
    return p


# ── discover_artifacts tests ──────────────────────────────────────────────────

class TestDiscoverArtifacts:

    def test_missing_base_path_returns_empty(self, tmp_path):
        artifacts = discover_artifacts(tmp_path / "nonexistent")
        assert artifacts == []

    def test_empty_dir_returns_empty(self, tmp_path):
        artifacts = discover_artifacts(tmp_path)
        assert artifacts == []

    def test_discovers_wheels(self, tmp_path):
        _create_mock_wheel(tmp_path, "build/mlc_llm-0.1.0-cp310-cp310-linux_x86_64.whl")
        _create_mock_wheel(tmp_path, "python/dist/mlc_llm-0.1.0-py3-none-any.whl")
        
        # Should ignore node_modules and .git
        _create_mock_wheel(tmp_path, "node_modules/some-module/fake.whl")
        _create_mock_wheel(tmp_path, ".git/objects/fake.whl")

        artifacts = discover_artifacts(tmp_path)
        wheels = [a for a in artifacts if a["type"] == "wheel"]
        
        assert len(wheels) == 2
        names = {w["name"] for w in wheels}
        assert "mlc_llm-0.1.0-cp310-cp310-linux_x86_64.whl" in names
        assert "mlc_llm-0.1.0-py3-none-any.whl" in names
        assert all(w["source_step"] == "build" for w in wheels)
        assert all("size_bytes" in w for w in wheels)

    def test_discovers_model_dirs(self, tmp_path):
        _create_mock_model_dir(tmp_path, "Llama-3-8B-q4f16_1-MLC")
        
        # Ignored paths
        (tmp_path / ".git" / "models" / "fake").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".git" / "models" / "fake" / "mlc-chat-config.json").write_text("{}")

        artifacts = discover_artifacts(tmp_path)
        models = [a for a in artifacts if a["type"] == "model_dir"]
        
        assert len(models) == 1
        assert models[0]["name"] == "Llama-3-8B-q4f16_1-MLC"
        assert models[0]["source_step"] == "quantize"
        assert models[0]["size_bytes"] == 4  # 2 chars per json file * 2

    def test_discovers_compiled_libs(self, tmp_path):
        _create_mock_compiled_lib(tmp_path, "dist/Llama-3-8B-q4f16_1-cuda.so")
        _create_mock_compiled_lib(tmp_path, "lib/some_lib.dylib")
        _create_mock_compiled_lib(tmp_path, "build/another.dll")
        
        # ignored paths
        (tmp_path / "node_modules" / "ignored.so").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "node_modules" / "ignored.so").write_text("ignored")

        artifacts = discover_artifacts(tmp_path)
        libs = [a for a in artifacts if a["type"] == "compiled_lib"]
        
        assert len(libs) == 3
        names = {l["name"] for l in libs}
        assert "Llama-3-8B-q4f16_1-cuda.so" in names
        assert "some_lib.dylib" in names
        assert "another.dll" in names
        assert all(l["source_step"] == "compile" for l in libs)


# ── GET /artifacts route tests ────────────────────────────────────────────────

class TestGetArtifactsRoute:

    def test_missing_repo(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr("app.main.MLC_CLI_PATH", tmp_path / "nonexistent")
        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["artifacts"] == []
        assert data["counts"]["total"] == 0

    def test_returns_discovered_artifacts(self, client, monkeypatch, tmp_path):
        fake_repo = tmp_path / "mlc-cli"
        fake_repo.mkdir()
        monkeypatch.setattr("app.main.MLC_CLI_PATH", fake_repo)

        _create_mock_wheel(fake_repo, "build/wheel.whl")
        _create_mock_model_dir(fake_repo, "model_dir")
        _create_mock_compiled_lib(fake_repo, "dist/lib.so")

        resp = client.get("/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["status"] == "ok"
        assert data["counts"]["build"] == 1
        assert data["counts"]["convert"] == 1
        assert data["counts"]["quantize"] == 1
        assert data["counts"]["compile"] == 1
        assert data["counts"]["total"] == 3
        
        assert len(data["artifacts"]) == 3
