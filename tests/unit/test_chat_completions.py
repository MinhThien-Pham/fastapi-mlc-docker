"""
tests/unit/test_chat_completions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Focused tests for POST /chat/completions — both non-streaming and streaming.

All engine interaction is mocked; no GPU or real runtime required.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient

from app.main import app
from app import chat_engine_manager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_manager_state():
    """Guarantee a clean engine state before and after every test."""
    chat_engine_manager.unload_engine()
    yield
    chat_engine_manager.unload_engine()


@pytest.fixture(autouse=True)
def mock_paths():
    """Treat all paths as valid so load_engine never rejects dummy paths."""
    with patch("os.path.isdir", return_value=True), patch("os.path.isfile", return_value=True):
        yield


@pytest.fixture
def loaded_engine():
    """
    Load a mock engine into the manager and yield the mock engine instance.

    The mock is wired so that:
        engine_instance.chat.completions.create(...)
    returns a response whose first choice has a non-null content string.
    """
    mock_module = MagicMock()
    mock_engine_class = MagicMock()
    mock_module.MLCEngine = mock_engine_class

    # Build a fake response object that mirrors the mlc_llm response shape
    fake_response = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = "Hello, world!"
    fake_response.choices = [fake_choice]

    engine_instance = mock_engine_class.return_value
    engine_instance.chat.completions.create.return_value = fake_response

    with patch.dict("sys.modules", {"mlc_llm": mock_module}):
        chat_engine_manager.load_engine(
            model="/fake/model",
            model_lib="/fake/lib.so",
            device="cuda:0",
        )
        yield engine_instance


# ── Success path ───────────────────────────────────────────────────────────────

def test_completion_success(client, loaded_engine):
    """Happy path: engine loaded, valid messages → structured reply."""
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hello."}]},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "/fake/model"
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello, world!"
    assert choice["finish_reason"] == "stop"


def test_completion_passes_generation_params(client, loaded_engine):
    """Generation params (max_tokens, temperature, top_p) are forwarded to the engine."""
    client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    loaded_engine.chat.completions.create.assert_called_once_with(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )


def test_completion_uses_currently_loaded_engine(client, loaded_engine):
    """
    The endpoint must use the already-loaded engine instance —
    it must NOT create a new MLCEngine on each call.
    """
    for _ in range(3):
        client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}]},
        )

    # chat.completions.create was called 3 times on the *same* instance
    assert loaded_engine.chat.completions.create.call_count == 3


def test_completion_multi_turn_messages(client, loaded_engine):
    """Multi-turn conversation messages are forwarded as-is."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"},
    ]
    response = client.post("/chat/completions", json={"messages": messages})

    assert response.status_code == 200
    loaded_engine.chat.completions.create.assert_called_once()
    call_kwargs = loaded_engine.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == messages


# ── No engine loaded ───────────────────────────────────────────────────────────

def test_completion_no_engine_loaded(client):
    """503 when the engine has not been loaded yet."""
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 503
    assert "No engine is loaded" in response.json()["detail"]


# ── Invalid request payloads ───────────────────────────────────────────────────

def test_completion_empty_messages_list(client, loaded_engine):
    """422 when messages is an empty list."""
    response = client.post("/chat/completions", json={"messages": []})
    assert response.status_code == 422
    assert "non-empty" in response.json()["detail"]


def test_completion_blank_role(client, loaded_engine):
    """422 when a message has a blank role."""
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "   ", "content": "hello"}]},
    )
    assert response.status_code == 422
    assert "role" in response.json()["detail"]


def test_completion_blank_content(client, loaded_engine):
    """422 when a message has blank content."""
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "   "}]},
    )
    assert response.status_code == 422
    assert "content" in response.json()["detail"]


def test_completion_missing_messages_field(client, loaded_engine):
    """422 when the messages field is absent entirely (Pydantic validation)."""
    response = client.post("/chat/completions", json={})
    # Pydantic itself returns 422 for a missing required field
    assert response.status_code == 422


def test_completion_messages_not_a_list(client, loaded_engine):
    """422 when messages is not a list (type mismatch caught by Pydantic)."""
    response = client.post(
        "/chat/completions",
        json={"messages": "just a string"},
    )
    assert response.status_code == 422


# ── Engine generation failure ──────────────────────────────────────────────────

def test_completion_engine_raises_during_generation(client, loaded_engine):
    """500 when the engine itself raises during generation."""
    loaded_engine.chat.completions.create.side_effect = RuntimeError("GPU OOM")

    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    assert "GPU OOM" in response.json()["detail"]


def test_completion_engine_returns_empty_choices(client, loaded_engine):
    """500 when the engine returns a response with no choices."""
    fake_empty = MagicMock()
    fake_empty.choices = []
    loaded_engine.chat.completions.create.return_value = fake_empty

    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    assert "unexpected response structure" in response.json()["detail"].lower()


def test_completion_engine_returns_null_content(client, loaded_engine):
    """500 when the engine returns a choice with null content."""
    fake_null = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = None
    fake_null.choices = [fake_choice]
    loaded_engine.chat.completions.create.return_value = fake_null

    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    assert "null content" in response.json()["detail"].lower()


# ── Streaming helpers ──────────────────────────────────────────────────────────

def _parse_sse_lines(text: str) -> list[str]:
    """Return the data payloads from a raw SSE response body."""
    lines = []
    for line in text.splitlines():
        if line.startswith("data: "):
            lines.append(line[len("data: "):])
    return lines


def _make_async_gen(*items):
    """Return an async generator factory that yields the given items.

    The factory accepts and ignores any kwargs so it can be used as a
    ``side_effect`` on a mock that is called with keyword arguments.
    """
    async def _gen(**_kwargs):
        for item in items:
            yield item
    return _gen


# ── Streaming: success path ────────────────────────────────────────────────────

def test_stream_returns_sse_response(client, loaded_engine):
    """stream=true → 200 with text/event-stream content-type."""
    with patch.object(
        __import__("app.chat_engine_manager", fromlist=["stream_completion"]),
        "stream_completion",
        side_effect=_make_async_gen("Hello", ", ", "world", "!"),
    ):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_stream_chunks_contain_deltas(client, loaded_engine):
    """Each yielded delta appears as a data: {"delta": ...} SSE event."""
    import json

    with patch.object(
        __import__("app.chat_engine_manager", fromlist=["stream_completion"]),
        "stream_completion",
        side_effect=_make_async_gen("Hello", ", world"),
    ):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    payloads = _parse_sse_lines(response.text)

    # Last event must be [DONE]; all others must be JSON {"delta": ...}
    assert payloads[-1] == "[DONE]"
    delta_payloads = payloads[:-1]
    assert len(delta_payloads) == 2

    assert json.loads(delta_payloads[0]) == {"delta": "Hello"}
    assert json.loads(delta_payloads[1]) == {"delta": ", world"}


def test_stream_ends_with_done(client, loaded_engine):
    """The SSE stream always terminates with data: [DONE]."""
    with patch.object(
        __import__("app.chat_engine_manager", fromlist=["stream_completion"]),
        "stream_completion",
        side_effect=_make_async_gen("token"),
    ):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}], "stream": True},
        )

    payloads = _parse_sse_lines(response.text)
    assert payloads[-1] == "[DONE]"


def test_stream_passes_generation_params(client, loaded_engine):
    """Generation params are forwarded to stream_completion."""
    import app.chat_engine_manager as mgr

    with patch.object(mgr, "stream_completion", side_effect=_make_async_gen()) as mock_sc:
        client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "temperature": 0.5,
                "top_p": 0.8,
                "stream": True,
            },
        )

    mock_sc.assert_called_once_with(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        temperature=0.5,
        top_p=0.8,
    )


# ── Streaming: no engine loaded ────────────────────────────────────────────────

def test_stream_no_engine_loaded(client):
    """No engine → error event emitted in stream, then [DONE]."""
    import json

    # Engine is not loaded (reset_manager_state autouse fixture ensures this)
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert response.status_code == 200  # HTTP layer is 200; error is in the stream
    payloads = _parse_sse_lines(response.text)
    assert payloads[-1] == "[DONE]"

    error_payload = json.loads(payloads[0])
    assert "error" in error_payload
    assert "No engine is loaded" in error_payload["error"]


# ── Streaming: engine failure mid-stream ───────────────────────────────────────

def test_stream_engine_failure_emits_error_event(client, loaded_engine):
    """Engine raising during streaming → error event in stream, then [DONE]."""
    import json
    from app.chat_engine_manager import EngineStreamError

    async def _failing_gen(**_kwargs):
        yield "partial"
        raise EngineStreamError("GPU died mid-stream")

    with patch.object(
        __import__("app.chat_engine_manager", fromlist=["stream_completion"]),
        "stream_completion",
        side_effect=_failing_gen,
    ):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    payloads = _parse_sse_lines(response.text)
    assert payloads[-1] == "[DONE]"

    # First payload is the partial delta, second is the error
    assert json.loads(payloads[0]) == {"delta": "partial"}
    error_payload = json.loads(payloads[1])
    assert "error" in error_payload
    assert "GPU died mid-stream" in error_payload["error"]


# ── Streaming: stream=false default still works ────────────────────────────────

def test_stream_false_default_unchanged(client, loaded_engine):
    """Omitting stream (or stream=false) still returns the original JSON shape."""
    response = client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
