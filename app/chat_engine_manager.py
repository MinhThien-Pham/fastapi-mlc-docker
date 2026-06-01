import os
import threading
from typing import Any, AsyncIterator, Dict, Optional

# We use loose typing here because mlc_llm might not be installed in all environments.
_engine_instance: Any = None
_loaded_model: Optional[str] = None
_loaded_model_lib: Optional[str] = None
_loaded_device: Optional[str] = None

# Protects state transitions (load/unload) for concurrency safety
_lifecycle_lock = threading.Lock()


class EngineConflictError(Exception):
    """Raised when trying to load an engine while another configuration is already active."""
    pass


class EngineImportError(Exception):
    """Raised when mlc_llm is not installed or importable."""
    pass


class EngineInitializationError(Exception):
    """Raised when the engine fails to initialize for hardware or model compatibility reasons."""
    pass


class EngineNotLoadedError(Exception):
    """Raised when a generation is requested but no engine is currently loaded."""
    pass


class EngineGenerationError(Exception):
    """Raised when the engine fails during token generation."""
    pass


class EngineStreamError(Exception):
    """Raised when the engine fails or produces unexpected output during streaming."""
    pass


class InvalidArtifactPathError(Exception):
    """Raised when the provided model or model_lib paths do not exist locally."""
    pass


def load_engine(model: str, model_lib: str, device: str) -> None:
    """
    Load the MLCEngine safely under a lock.
    If the exact same configuration is already loaded, this is a no-op.
    """
    global _engine_instance, _loaded_model, _loaded_model_lib, _loaded_device

    with _lifecycle_lock:
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
            import sys
            raise EngineImportError(
                f"mlc_llm is not installed or importable in this environment ({sys.executable}). "
                "Please run POST /build with action='full' or 'install-wheels' to install it."
            )

        # Initialize the engine
        try:
            instance = MLCEngine(model=model, model_lib=model_lib, device=device)
        except Exception as e:
            # State remains clean (None) because it failed partway
            raise EngineInitializationError(f"Failed to initialize MLCEngine: {str(e)}") from e
            
        _engine_instance = instance
        _loaded_model = model
        _loaded_model_lib = model_lib
        _loaded_device = device


def get_status() -> Dict[str, Any]:
    """Return the current status of the loaded engine (lock-free read)."""
    if _engine_instance is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "model": _loaded_model,
        "model_lib": _loaded_model_lib,
        "device": _loaded_device,
    }


def unload_engine() -> None:
    """Unload the engine safely under a lock, freeing resources."""
    global _engine_instance, _loaded_model, _loaded_model_lib, _loaded_device
    
    with _lifecycle_lock:
        if _engine_instance is not None:
            try:
                if hasattr(_engine_instance, 'terminate'):
                    _engine_instance.terminate()
            finally:
                # Always clear the state even if terminate throws an error
                _engine_instance = None
                _loaded_model = None
                _loaded_model_lib = None
                _loaded_device = None


def generate_completion(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """
    Run a single non-streaming chat completion against the loaded engine.

    Parameters
    ----------
    messages:
        List of dicts with ``role`` and ``content`` keys, e.g.
        [{"role": "user", "content": "Hello"}].
    max_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature (1.0 = default; lower = more deterministic).
    top_p:
        Nucleus sampling probability mass.

    Returns
    -------
    str
        The assistant's reply text from the first (and only) choice.

    Raises
    ------
    EngineNotLoadedError
        If no engine has been loaded yet.
    EngineGenerationError
        If the engine raises during generation or returns an empty response.
    """
    if _engine_instance is None:
        raise EngineNotLoadedError(
            "No engine is loaded. Call POST /chat/load before requesting completions."
        )

    try:
        response = _engine_instance.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
    except Exception as e:
        raise EngineGenerationError(f"Engine generation failed: {str(e)}") from e

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise EngineGenerationError(
            f"Engine returned an unexpected response structure: {str(e)}"
        ) from e

    if content is None:
        raise EngineGenerationError("Engine returned a null content field.")

    return content


async def stream_completion(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> AsyncIterator[str]:
    """
    Yield content-delta strings from a streaming chat completion.

    The engine is called with ``stream=True``.  For each chunk the engine
    yields, the content delta from the first choice is extracted and
    yielded to the caller.  Empty deltas (e.g. the final ``finish_reason``
    chunk) are skipped silently.

    Parameters
    ----------
    messages:
        List of dicts with ``role`` and ``content`` keys.
    max_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature.
    top_p:
        Nucleus sampling probability mass.

    Yields
    ------
    str
        Each non-empty content delta from the engine.

    Raises
    ------
    EngineNotLoadedError
        If no engine has been loaded yet (raised before any iteration).
    EngineStreamError
        If the engine raises during iteration or produces an unexpected
        chunk structure.
    """
    if _engine_instance is None:
        raise EngineNotLoadedError(
            "No engine is loaded. Call POST /chat/load before requesting completions."
        )

    try:
        chunks = _engine_instance.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
    except Exception as e:
        raise EngineStreamError(f"Engine failed to start streaming: {str(e)}") from e

    try:
        for chunk in chunks:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError) as e:
                raise EngineStreamError(
                    f"Engine returned an unexpected chunk structure: {str(e)}"
                ) from e
            if delta:  # skip empty / finish-reason-only chunks
                yield delta
    except EngineStreamError:
        raise
    except Exception as e:
        raise EngineStreamError(f"Engine stream interrupted: {str(e)}") from e
