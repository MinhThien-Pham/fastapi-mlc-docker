"""
tests/integration/test_chat_lifecycle.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end lifecycle proof for the direct-engine chat path.

Covers the exact flow the user asked to prove:

    load  →  status (loaded)  →  completion #1  →  completion #2
          →  unload  →  status (unloaded)

No GPU or live server is required.  The MLCEngine is replaced by a
deterministic mock so this test runs cleanly in CI.

Markers
-------
``integration`` — these are lifecycle tests, not isolated unit tests.
   Run with:  pytest tests/integration/test_chat_lifecycle.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app import chat_engine_manager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_engine_state():
    """Guarantee a completely clean engine state before and after each test."""
    chat_engine_manager.unload_engine()
    yield
    chat_engine_manager.unload_engine()


@pytest.fixture(autouse=True)
def mock_paths():
    """Treat all paths as valid so load_engine never rejects dummy paths."""
    with patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.is_file", return_value=True):
        yield


@pytest.fixture
def mock_engine_class():
    """
    Patch mlc_llm.MLCEngine with a mock whose chat.completions.create()
    returns a response bearing a non-null content string.
    """
    mock_module = MagicMock()
    engine_class = MagicMock()
    mock_module.MLCEngine = engine_class

    # Wire up a realistic (but fake) response object
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "I am a mocked model response."
    fake_response.choices = [fake_choice]
    engine_class.return_value.chat.completions.create.return_value = fake_response

    with patch.dict("sys.modules", {"mlc_llm": mock_module}):
        yield engine_class


@pytest.fixture
def client():
    return TestClient(app)


# ── Lifecycle proof ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestChatLifecycle:
    """
    Proves the full load → chat × N → unload cycle works correctly.

    These tests are intentionally written in execution order to read like
    a script — each one represents one step of the lifecycle.  They are
    independent (each gets a fresh engine state via autouse fixtures), but
    together they document the expected usage narrative.
    """

    def test_step1_status_before_load_shows_not_loaded(self, client):
        """Before anything is loaded, /chat/status reports loaded=false."""
        resp = client.get("/chat/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] is False
        assert "model" not in body

    def test_step2_load_engine(self, client, mock_engine_class):
        """POST /chat/load succeeds and MLCEngine is constructed exactly once."""
        resp = client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

        # MLCEngine was constructed exactly once with the right arguments
        mock_engine_class.assert_called_once_with(
            model="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            model_lib="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            device="cuda:0",
        )

    def test_step2b_load_engine_explicit_model_name_bare(self, client, mock_engine_class):
        """POST /chat/load explicit mode supports bare model_name."""
        resp = client.post("/chat/load", json={
            "model_name": "MyModel-q4f16_1-MLC",
            "model_lib": "dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        assert resp.status_code == 200
        
        mock_engine_class.assert_called_once_with(
            model="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            model_lib="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            device="cuda:0",
        )

    def test_step2c_load_engine_missing_model_with_lib(self, client, mock_engine_class):
        """POST /chat/load with model_lib but no model or model_name returns 400."""
        resp = client.post("/chat/load", json={
            "model_lib": "dist/MyModel-q4f16_1-MLC/MyModel.so",
        })
        assert resp.status_code == 400
        assert "model or model_name is required" in resp.json()["detail"]
        mock_engine_class.assert_not_called()

    def test_step2d_load_engine_shorthand(self, client, mock_engine_class):
        """POST /chat/load succeeds with shorthand model payload."""
        # Mock resolve_chat_artifacts to verify router integration without needing a real dist/ structure.
        with patch("app.main.resolve_chat_artifacts") as mock_resolve:
            mock_resolve.return_value = (
                "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
                "/workspace/mlc-cli/dist/libs/MyModel-q4f16_1-MLC-q4f16_1-cuda.so"
            )
            resp = client.post("/chat/load", json={
                "model": "MyModel",
                "device": "cuda:0",
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"

            mock_resolve.assert_called_once()
            
            # MLCEngine was constructed exactly once with the right arguments
            mock_engine_class.assert_called_once_with(
                model="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
                model_lib="/workspace/mlc-cli/dist/libs/MyModel-q4f16_1-MLC-q4f16_1-cuda.so",
                device="cuda:0",
            )

    def test_step2e_load_engine_device_normalization(self, client, mock_engine_class):
        """POST /chat/load normalizes 'cuda' to 'cuda:0' but leaves others unchanged."""
        client.post("/chat/unload")
        # Test 'cuda' becomes 'cuda:0'
        resp1 = client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda",
        })
        assert resp1.status_code == 200
        mock_engine_class.assert_called_with(
            model="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            model_lib="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            device="cuda:0",
        )
        
        client.post("/chat/unload")
        # Test 'vulkan' stays 'vulkan'
        resp2 = client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "vulkan",
        })
        assert resp2.status_code == 200
        mock_engine_class.assert_called_with(
            model="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            model_lib="/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            device="vulkan",
        )

    def test_step3_status_after_load_shows_loaded(self, client, mock_engine_class):
        """After load, /chat/status reports loaded=true with correct model path."""
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })

        resp = client.get("/chat/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] is True
        assert body["model"] == "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC"
        assert body["device"] == "cuda:0"

    def test_step4_first_completion_succeeds(self, client, mock_engine_class):
        """After load, the first /chat/completions call succeeds."""
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })

        resp = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello, model!"}],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "I am a mocked model response."
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_step5_second_completion_uses_same_engine_instance(
        self, client, mock_engine_class
    ):
        """
        The engine is NOT reloaded between requests.

        Two /chat/completions calls must both use the same engine instance
        (MLCEngine constructor called exactly once, chat.completions.create
        called exactly twice).
        """
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })

        # First call
        resp1 = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "First question."}],
        })
        assert resp1.status_code == 200

        # Second call — no reload
        resp2 = client.post("/chat/completions", json={
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "I am a mocked model response."},
                {"role": "user", "content": "Follow-up question."},
            ],
        })
        assert resp2.status_code == 200

        # Engine was only constructed once
        assert mock_engine_class.call_count == 1

        # But generate was called twice on the same instance
        engine_instance = mock_engine_class.return_value
        assert engine_instance.chat.completions.create.call_count == 2

    def test_step6_status_still_loaded_between_completions(
        self, client, mock_engine_class
    ):
        """
        /chat/status remains loaded=true between completion calls —
        the engine is not implicitly unloaded.
        """
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })

        client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "ping"}],
        })

        # Status still loaded after a completion — no auto-unload
        status = client.get("/chat/status").json()
        assert status["loaded"] is True

    def test_step7_unload_clears_engine(self, client, mock_engine_class):
        """POST /chat/unload calls engine.terminate() and clears state."""
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        engine_instance = mock_engine_class.return_value

        resp = client.post("/chat/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        # terminate() was called
        engine_instance.terminate.assert_called_once()

    def test_step8_status_after_unload_shows_not_loaded(
        self, client, mock_engine_class
    ):
        """After unload, /chat/status reports loaded=false again."""
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        client.post("/chat/unload")

        resp = client.get("/chat/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] is False
        assert "model" not in body

    def test_step9_completion_after_unload_returns_503(
        self, client, mock_engine_class
    ):
        """After unload, /chat/completions returns 503 (not 500 or 404)."""
        client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        client.post("/chat/unload")

        resp = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Are you there?"}],
        })
        assert resp.status_code == 503
        assert "No engine is loaded" in resp.json()["detail"]

    def test_full_lifecycle_in_one_shot(self, client, mock_engine_class):
        """
        Single narrative test covering the complete lifecycle in sequence.

        This is the canonical proof the user asked for:
          load → status(loaded) → completion #1 → completion #2 → unload → status(unloaded)
        """
        # ── 1. Load ───────────────────────────────────────────────────────────
        load_resp = client.post("/chat/load", json={
            "model": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC",
            "model_lib": "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC/MyModel.so",
            "device": "cuda:0",
        })
        assert load_resp.status_code == 200, f"load failed: {load_resp.json()}"

        # ── 2. Status shows loaded ────────────────────────────────────────────
        status = client.get("/chat/status").json()
        assert status["loaded"] is True
        assert status["model"] == "/workspace/mlc-cli/dist/MyModel-q4f16_1-MLC"

        # ── 3. First completion ───────────────────────────────────────────────
        c1 = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello!"}],
        })
        assert c1.status_code == 200, f"completion #1 failed: {c1.json()}"
        assert c1.json()["choices"][0]["message"]["content"] != ""

        # ── 4. Second completion (engine still loaded, not recreated) ─────────
        c2 = client.post("/chat/completions", json={
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": c1.json()["choices"][0]["message"]["content"]},
                {"role": "user", "content": "Tell me more."},
            ],
        })
        assert c2.status_code == 200, f"completion #2 failed: {c2.json()}"

        # Engine never recreated
        assert mock_engine_class.call_count == 1
        assert mock_engine_class.return_value.chat.completions.create.call_count == 2

        # ── 5. Unload ─────────────────────────────────────────────────────────
        unload_resp = client.post("/chat/unload")
        assert unload_resp.status_code == 200

        # ── 6. Status shows unloaded ──────────────────────────────────────────
        status_after = client.get("/chat/status").json()
        assert status_after["loaded"] is False
        assert "model" not in status_after

        # ── 7. Next completion returns 503 ────────────────────────────────────
        c3 = client.post("/chat/completions", json={
            "messages": [{"role": "user", "content": "Are you still there?"}],
        })
        assert c3.status_code == 503
