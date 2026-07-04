import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app import chat_engine_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_manager_state():
    chat_engine_manager.unload_engine()
    yield
    chat_engine_manager.unload_engine()


@pytest.fixture
def mock_mlc_llm():
    """Provides a safe sys.modules mock for endpoints."""
    mock_module = MagicMock()
    mock_engine_class = MagicMock()
    mock_module.MLCEngine = mock_engine_class
    
    with patch.dict("sys.modules", {"mlc_llm": mock_module}):
        yield mock_engine_class


@pytest.fixture(autouse=True)
def mock_paths():
    """Assume paths are valid by default unless overridden."""
    with patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.is_file", return_value=True):
        yield


def test_status_when_not_loaded(client):
    response = client.get("/chat/status")
    assert response.status_code == 200
    assert response.json()["loaded"] is False


def test_load_success(client, mock_mlc_llm):
    response = client.post("/chat/load", json={
        "model": "/valid/model/dir",
        "model_lib": "/valid/lib.so",
        "device": "cuda:0"
    })
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify status endpoint reflects loaded state
    status_resp = client.get("/chat/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["loaded"] is True

    from pathlib import Path
    expected_model = str(Path("/valid/model/dir"))
    assert status_resp.json()["model"] == expected_model

def test_load_relative_paths_resolved(client, mock_mlc_llm):
    response = client.post("/chat/load", json={
        "model": "relative/model",
        "model_lib": "relative/lib.so",
        "device": "cuda:0"
    })
    assert response.status_code == 200
    
    # Engine should be called with resolved absolute paths
    from app.main import MLC_CLI_PATH
    expected_model = str(MLC_CLI_PATH / "relative/model")
    expected_lib = str(MLC_CLI_PATH / "relative/lib.so")
    
    mock_mlc_llm.assert_called_once()
    args, kwargs = mock_mlc_llm.call_args
    assert kwargs.get("model") == expected_model
    assert kwargs.get("model_lib") == expected_lib

def test_load_invalid_paths(client):
    with patch("pathlib.Path.exists", return_value=False):
        response = client.post("/chat/load", json={
            "model": "/bad/model/dir",
            "model_lib": "/valid/lib.so"
        })
    
    assert response.status_code == 400
    assert "Model directory not found" in response.json()["detail"]

def test_load_relative_model_missing(client):
    with patch("pathlib.Path.is_dir", return_value=False):
        response = client.post("/chat/load", json={
            "model": "missing/model",
            "model_lib": "/valid/lib.so"
        })
    assert response.status_code == 400
    assert "Model directory not found" in response.json()["detail"]

def test_load_relative_model_lib_missing(client):
    # Model exists but lib does not
    with patch("pathlib.Path.is_file", return_value=False):
        response = client.post("/chat/load", json={
            "model": "/valid/model",
            "model_lib": "missing/lib.so"
        })
    assert response.status_code == 400
    assert "Model library file not found" in response.json()["detail"]


def test_load_conflict(client, mock_mlc_llm):
    # First load
    client.post("/chat/load", json={
        "model": "/model/A",
        "model_lib": "/lib/A.so"
    })
    
    # Conflicting load
    response = client.post("/chat/load", json={
        "model": "/model/B",
        "model_lib": "/lib/B.so"
    })
    
    assert response.status_code == 409
    assert "already loaded" in response.json()["detail"]


def test_load_import_error(client):
    with patch.dict("sys.modules", {"mlc_llm": None}):
        response = client.post("/chat/load", json={
            "model": "/valid/model/dir",
            "model_lib": "/valid/lib.so"
        })
    
    assert response.status_code == 503
    assert "not installed or importable" in response.json()["detail"]


def test_load_initialization_error(client, mock_mlc_llm):
    mock_mlc_llm.side_effect = Exception("CUDA out of memory")
    
    response = client.post("/chat/load", json={
        "model": "/valid/model/dir",
        "model_lib": "/valid/lib.so"
    })
    
    assert response.status_code == 500
    assert "CUDA out of memory" in response.json()["detail"]


def test_unload_success(client, mock_mlc_llm):
    # Load first
    client.post("/chat/load", json={
        "model": "/valid/model/dir",
        "model_lib": "/valid/lib.so"
    })
    
    # Unload
    response = client.post("/chat/unload")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Status should be false
    status_resp = client.get("/chat/status")
    assert status_resp.json()["loaded"] is False


def test_unload_when_not_loaded(client):
    # Unloading when nothing is loaded should be safe
    response = client.post("/chat/unload")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_unload_failure_still_clears_state(client, mock_mlc_llm):
    # Load first
    client.post("/chat/load", json={
        "model": "/valid/model/dir",
        "model_lib": "/valid/lib.so"
    })
    
    # Make terminate throw an exception
    engine_instance = mock_mlc_llm.return_value
    engine_instance.terminate.side_effect = Exception("Failed to free GPU memory")
    
    # We still return 500 because the operation had an error
    response = client.post("/chat/unload")
    assert response.status_code == 500
    assert "Failed to free GPU memory" in response.json()["detail"]
    
    # BUT the internal state must be cleared anyway!
    status_resp = client.get("/chat/status")
    assert status_resp.json()["loaded"] is False
