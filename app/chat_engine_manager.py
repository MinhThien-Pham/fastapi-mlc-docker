import os
from typing import Any, Dict, Optional

# We use loose typing here because mlc_llm might not be installed in all environments (e.g., standard tests).
_engine_instance: Any = None
_loaded_model: Optional[str] = None
_loaded_model_lib: Optional[str] = None
_loaded_device: Optional[str] = None


class EngineConflictError(Exception):
    """Raised when trying to load an engine while another configuration is already active."""
    pass


class EngineImportError(Exception):
    """Raised when mlc_llm is not installed or importable."""
    pass


class EngineInitializationError(Exception):
    """Raised when the engine fails to initialize for hardware or model compatibility reasons."""
    pass


class InvalidArtifactPathError(Exception):
    """Raised when the provided model or model_lib paths do not exist locally."""
    pass


def load_engine(model: str, model_lib: str, device: str) -> None:
    """
    Load the MLCEngine.
    If the exact same configuration is already loaded, this is a no-op.
    """
    global _engine_instance, _loaded_model, _loaded_model_lib, _loaded_device

    if _engine_instance is not None:
        if _loaded_model == model and _loaded_model_lib == model_lib and _loaded_device == device:
            # Already loaded with the exact same configuration; idempotent success.
            return
        raise EngineConflictError("An engine is already loaded with a different configuration. Unload it first.")

    # Practical validation: Ensure local paths exist
    if not os.path.isdir(model):
        raise InvalidArtifactPathError(f"Model path does not exist or is not a directory: {model}")
    if not os.path.isfile(model_lib):
        raise InvalidArtifactPathError(f"Model library path does not exist or is not a file: {model_lib}")

    try:
        from mlc_llm import MLCEngine
    except ImportError:
        raise EngineImportError("mlc_llm is not installed or importable in this environment.")

    # Initialize the engine
    try:
        _engine_instance = MLCEngine(model=model, model_lib=model_lib, device=device)
    except Exception as e:
        raise EngineInitializationError(f"Failed to initialize MLCEngine: {str(e)}") from e
        
    _loaded_model = model
    _loaded_model_lib = model_lib
    _loaded_device = device


def get_status() -> Dict[str, Any]:
    """Return the current status of the loaded engine."""
    if _engine_instance is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "model": _loaded_model,
        "model_lib": _loaded_model_lib,
        "device": _loaded_device,
    }


def unload_engine() -> None:
    """Unload the engine if one is loaded, freeing resources."""
    global _engine_instance, _loaded_model, _loaded_model_lib, _loaded_device
    
    if _engine_instance is not None:
        if hasattr(_engine_instance, 'terminate'):
            _engine_instance.terminate()
        
        _engine_instance = None
        _loaded_model = None
        _loaded_model_lib = None
        _loaded_device = None
