import pytest
from pathlib import Path
from app.helpers import resolve_chat_artifacts

def test_resolve_exact_model_name(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    # Create candidate artifact
    model_dir = dist_dir / "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC"
    model_dir.mkdir()
    
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir()
    lib_file = libs_dir / "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC-q4f16_1-cuda.so"
    lib_file.touch()

    model, lib = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model="",
        model_name="TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
        quant="q4f16_1",
        device="cuda:0"
    )
    assert model == str(model_dir)
    assert lib == str(lib_file)

def test_resolve_shorthand_model(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    model_dir = dist_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC"
    model_dir.mkdir()
    
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir()
    lib_file = libs_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so"
    lib_file.touch()

    model, lib = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model="TinyLlama-1.1B-Chat-v1.0",
        model_name="",
        quant="q4f16_1",
        device="cuda:0"
    )
    assert model == str(model_dir)
    assert lib == str(lib_file)

def test_resolve_huggingface_id(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    model_dir = dist_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC"
    model_dir.mkdir()
    
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir()
    lib_file = libs_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so"
    lib_file.touch()

    model, lib = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_name="",
        quant="q4f16_1",
        device="cuda"
    )
    assert model == str(model_dir)
    assert lib == str(lib_file)

def test_resolve_quant_override(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    # Create two candidates
    dir1 = dist_dir / "MyModel-q4f16_1-MLC"
    dir1.mkdir()
    dir2 = dist_dir / "MyModel-q8f16_1-MLC"
    dir2.mkdir()
    
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir()
    (libs_dir / "MyModel-q4f16_1-MLC-q4f16_1-cuda.so").touch()
    (libs_dir / "MyModel-q8f16_1-MLC-q8f16_1-cuda.so").touch()

    model, lib = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model="MyModel",
        model_name="",
        quant="q8f16_1", # We specify q8
        device="cuda"
    )
    assert model == str(dir2)
    assert lib == str(libs_dir / "MyModel-q8f16_1-MLC-q8f16_1-cuda.so")

def test_resolve_no_artifact(tmp_path):
    with pytest.raises(ValueError, match="No compiled MLC artifact found for 'MissingModel'"):
        resolve_chat_artifacts(tmp_path, "MissingModel", "", "q4f16_1", "cuda:0")

def test_resolve_multiple_candidates(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    (dist_dir / "MyModel-A-q4f16_1-MLC").mkdir()
    (dist_dir / "MyModel-B-q4f16_1-MLC").mkdir()

    with pytest.raises(ValueError, match="Multiple artifact candidates found for 'MyModel': MyModel-A-q4f16_1-MLC, MyModel-B-q4f16_1-MLC"):
        resolve_chat_artifacts(tmp_path, "MyModel", "", "q4f16_1", "cuda:0")

def test_resolve_direct_path(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    model_dir = dist_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC"
    model_dir.mkdir()
    
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir()
    lib_file = libs_dir / "TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC-q4f16_1-cuda.so"
    lib_file.touch()

    # Pass the path relative to the workspace
    model, lib = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model="dist/TinyLlama-1.1B-Chat-v1.0-py313-q4f16_1-MLC",
        model_name="",
        quant="q4f16_1",
        device="cuda"
    )
    assert model == str(model_dir)
    assert lib == str(lib_file)

    # Pass an absolute path
    model2, lib2 = resolve_chat_artifacts(
        mlc_cli_path=tmp_path,
        model=str(model_dir),
        model_name="",
        quant="q4f16_1",
        device="cuda"
    )
    assert model2 == str(model_dir)
    assert lib2 == str(lib_file)

def test_resolve_missing_lib(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    
    (dist_dir / "MyModel-q4f16_1-MLC").mkdir()
    # No libs created

    with pytest.raises(ValueError, match="Found model artifact directory, but no compiled library was found"):
        resolve_chat_artifacts(tmp_path, "MyModel", "", "q4f16_1", "cuda:0")
