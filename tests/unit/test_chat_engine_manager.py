import sys
from unittest.mock import MagicMock, patch
import pytest
from app import chat_engine_manager


@pytest.fixture(autouse=True)
def reset_engine_state():
    """Ensure engine state is completely reset before and after each test."""
    chat_engine_manager.unload_engine()
    yield
    chat_engine_manager.unload_engine()


@pytest.fixture(autouse=True)
def mock_paths():
    """Mock os.path validation so we can pass dummy paths in tests."""
    with patch("os.path.isdir", return_value=True), patch("os.path.isfile", return_value=True):
        yield


@pytest.fixture
def mock_mlc_llm():
    """Mocks the mlc_llm module and MLCEngine safely."""
    mock_module = MagicMock()
    mock_engine_class = MagicMock()
    mock_module.MLCEngine = mock_engine_class

    with patch.dict("sys.modules", {"mlc_llm": mock_module}):
        yield mock_engine_class


def test_status_when_not_loaded():
    status = chat_engine_manager.get_status()
    assert status["loaded"] is False
    assert "model" not in status


def test_load_success(mock_mlc_llm):
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")

    mock_mlc_llm.assert_called_once_with(model="test_model", model_lib="test_lib", device="cuda:0")

    status = chat_engine_manager.get_status()
    assert status["loaded"] is True
    assert status["model"] == "test_model"


def test_load_already_loaded_same_config(mock_mlc_llm):
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")
    mock_mlc_llm.assert_called_once()  # Should not initialize twice


def test_load_already_loaded_different_config(mock_mlc_llm):
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")

    with pytest.raises(chat_engine_manager.EngineConflictError):
        chat_engine_manager.load_engine(model="other_model", model_lib="test_lib", device="cuda:0")


def test_load_invalid_paths():
    # Override the mock to simulate missing paths
    with patch("os.path.isdir", return_value=False):
        with pytest.raises(chat_engine_manager.InvalidArtifactPathError, match="not exist"):
            chat_engine_manager.load_engine(model="bad_model", model_lib="test_lib", device="cuda:0")

    with patch("os.path.isdir", return_value=True), patch("os.path.isfile", return_value=False):
        with pytest.raises(chat_engine_manager.InvalidArtifactPathError, match="not exist"):
            chat_engine_manager.load_engine(model="test_model", model_lib="bad_lib", device="cuda:0")


def test_load_import_error():
    with patch.dict("sys.modules", {"mlc_llm": None}):
        with pytest.raises(chat_engine_manager.EngineImportError):
            chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")


def test_load_initialization_error(mock_mlc_llm):
    mock_mlc_llm.side_effect = RuntimeError("Mocked GPU error")
    with pytest.raises(chat_engine_manager.EngineInitializationError, match="Mocked GPU error"):
        chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")


def test_unload_engine(mock_mlc_llm):
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")
    engine_instance = mock_mlc_llm.return_value

    chat_engine_manager.unload_engine()

    engine_instance.terminate.assert_called_once()
    assert chat_engine_manager.get_status()["loaded"] is False


def test_unload_engine_terminate_failure(mock_mlc_llm):
    chat_engine_manager.load_engine(model="test_model", model_lib="test_lib", device="cuda:0")
    engine_instance = mock_mlc_llm.return_value
    engine_instance.terminate.side_effect = RuntimeError("Failed to free resources")

    with pytest.raises(RuntimeError, match="Failed to free resources"):
        chat_engine_manager.unload_engine()

    # Crucially, the state MUST still be cleared!
    assert chat_engine_manager.get_status()["loaded"] is False
