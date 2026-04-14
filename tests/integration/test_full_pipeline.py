import os
import sys
import httpx
import shutil
import subprocess

API_URL = "http://localhost:8000"

def fetch_json(client, url):
    """Helper to fetch and parse JSON with error raising."""
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()

def get_newest_artifact(client, artifact_type):
    """Helper to fetch artifacts and return the newest one of the given type."""
    data = fetch_json(client, "/artifacts")
    items = [a for a in data.get("artifacts", []) if a.get("type") == artifact_type]
    if not items:
        return None
    items.sort(key=lambda x: x.get("modified_time", 0), reverse=True)
    return items[0]

def stream_endpoint(client, method, url, json_payload=None):
    """Helper to stream SSE responses and print them live."""
    print(f"\n--- Starting {url} ---")
    try:
        with client.stream(method, url, json=json_payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    msg = line[6:]
                    if msg == "[DONE]":
                        print(f"--- {url} Completed Successfully ---")
                        return True
                    elif msg.startswith("[ERROR]"):
                        print(f"--- {url} Failed: {msg} ---")
                        sys.exit(1)
                    else:
                        print(msg)
    except Exception as e:
        print(f"\n[ERROR] Failed during stream of {url}: {e}")
        sys.exit(1)
    return False

def auto_detect_conv_template(raw_model_path: str) -> tuple:
    """Attempts to infer the conversation template from the config.json or raw model path."""
    template_map = {
        "llama-3.1": ["llama-3.1", "llama-3-1"],
        "llama-3": ["llama-3", "llama3"],
        "llama-2": ["llama-2", "llama2", "tinyllama"],
        "qwen2": ["qwen2"],
        "mistral_default": ["mistral", "mixtral"],
        "phi-3": ["phi-3", "phi3"],
        "phi-2": ["phi-2", "phi2"],
        "gemma": ["gemma"],
        "ministral": ["ministral"]
    }
    
    # 1. Prefer config.json if it exists
    config_path = os.path.join(raw_model_path, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r") as f:
                config_data = json.load(f)
                model_type = config_data.get("model_type", "").lower()
                for template, keywords in template_map.items():
                    if any(kw in model_type for kw in keywords):
                        return template, "config.json"
        except Exception:
            pass

    # 2. Fallback to path/name guessing
    lower_path = raw_model_path.lower()
    for template, keywords in template_map.items():
        if any(kw in lower_path for kw in keywords):
            return template, "path"

    return "", ""

def main():
    print("=== MLC-CLI Full Pipeline Integration Test ===")
    print("Ensure the FastAPI app is running at http://localhost:8000")
    
    # 1. Fetch Environment Variables
    full_raw_model = os.environ.get("FULL_RAW_MODEL")
    full_conv_template = os.environ.get("FULL_CONV_TEMPLATE")
    full_quant = os.environ.get("FULL_QUANT", "q4f16_1")
    full_device = os.environ.get("FULL_DEVICE", "cuda")
    full_build_action = os.environ.get("FULL_BUILD_ACTION", "install-wheels")
    full_output_dir = os.environ.get("FULL_OUTPUT_DIR", "")
    full_lib_output = os.environ.get("FULL_LIB_OUTPUT", "")
    full_model_name = os.environ.get("FULL_MODEL_NAME")
    full_model_url = os.environ.get("FULL_MODEL_URL", "")
    full_model_lib = os.environ.get("FULL_MODEL_LIB", "")

    download_cache_path = None
    was_downloaded_this_run = False

    if full_raw_model:
        print(f"   -> [EXPLICIT] Using provided FULL_RAW_MODEL: {full_raw_model}")
    else:
        # Auto-fallback: Download local deterministic raw model
        cache_dir = ".raw_model_cache"
        raw_model_name = "TinyLlama-1.1B-Chat-v1.0"
        model_path = os.path.join(cache_dir, raw_model_name)
        
        has_config = os.path.exists(os.path.join(model_path, "config.json"))
        has_weights = any(os.path.exists(os.path.join(model_path, w)) for w in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"])
        
        if os.path.exists(model_path) and has_config and has_weights:
            print(f"   -> [FALLBACK CACHE] Reusing previously downloaded raw model: {model_path}")
            full_raw_model = model_path
            download_cache_path = model_path
        else:
            print(f"   -> [FALLBACK DOWNLOAD] Auto-preparing raw model in: {model_path}")
            if os.path.exists(model_path):
                print("   -> Removing invalid/incomplete cache directory before cloning...")
                shutil.rmtree(model_path, ignore_errors=True)
                
            os.makedirs(cache_dir, exist_ok=True)
            print("   -> Cloning TinyLlama from HuggingFace (requires git and git-lfs)...")
            
            if not shutil.which("git"):
                print("\n[ERROR] `git` is not installed or not in PATH.")
                print("Please install git, or bypass auto-download by setting FULL_RAW_MODEL=/path/to/local/weights")
                sys.exit(1)
                
            if not shutil.which("git-lfs"):
                print("\n[ERROR] `git-lfs` is not installed or not in PATH.")
                print("Please install git-lfs (e.g. `git lfs install`), or bypass auto-download by setting FULL_RAW_MODEL=/path/to/local/weights")
                sys.exit(1)
                
            repo_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            try:
                subprocess.run(["git", "clone", repo_url, model_path], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] Failed to clone raw model from {repo_url}.")
                print(f"Subprocess stderr:\n{e.stderr}")
                print("Make sure you have `git` and `git-lfs` installed, or provide an explicit model.")
                sys.exit(1)
            
            full_raw_model = model_path
            download_cache_path = model_path
            was_downloaded_this_run = True

    try:
        if not full_conv_template:
            full_conv_template, source = auto_detect_conv_template(full_raw_model)
            if not full_conv_template:
                print(f"\n[ERROR] Failed to auto-detect conversation template for model: {full_raw_model}")
                print("Checked: model path name and config.json 'model_type'.")
                print("This script uses a maintained mapping of common templates. If your model uses a new or unsupported architecture, please manually override.")
                print("Example: export FULL_CONV_TEMPLATE=llama-3")
                sys.exit(1)
            print(f"   -> Auto-detected conv_template '{full_conv_template}' (from {source})")
        else:
            print(f"   -> Using explicit conv_template: {full_conv_template}")

        with httpx.Client(base_url=API_URL, timeout=None) as client:
            # Step 1-4: Basic checks
            print("\n1. Checking API Health...")
            client.get("/health").raise_for_status()
            
            print("\n2. Checking setup (/setup-check)...")
            setup_data = fetch_json(client, "/setup-check")
            
            print("\n3. Ensuring mlc-cli repo exists (/ensure-repo-exists)...")
            if not setup_data.get("repo_exists"):
                client.post("/ensure-repo-exists").raise_for_status()
                
            print("\n4. Checking repo status (/repo-status)...")
            client.get("/repo-status").raise_for_status()

            # Step 5: Build
            print(f"\n5. Testing /build stream (action: {full_build_action})...")
            build_payload = {
                "action": full_build_action,
                "cuda": "y",
                "cutlass": "n",
                "flash_infer": "n",
                "cublas": "y"
            }
            stream_endpoint(client, "POST", "/build", json_payload=build_payload)

            # Step 6: Quantize
            print("\n6. Testing /quantize stream...")
            quantize_payload = {
                "model": full_raw_model,
                "quant": full_quant,
                "device": full_device,
                "conv_template": full_conv_template
            }
            if full_output_dir:
                quantize_payload["output"] = full_output_dir
                
            stream_endpoint(client, "POST", "/quantize", json_payload=quantize_payload)

            # Verification: Artifacts
            best_model = get_newest_artifact(client, "model_dir")
            if not best_model:
                print("\n[ERROR] /quantize stream completed, but no model_dir artifacts found in /artifacts.")
                sys.exit(1)
                
            model_path = full_output_dir if full_output_dir else best_model.get("path")
            quantized_model_name = best_model.get("name")
            print(f"   -> Selected newest model artifact: {quantized_model_name} (path: {model_path})")

            # Step 7: Compile
            print("\n7. Testing /compile stream...")
            compile_model_target = model_path
            compile_payload = {
                "model": compile_model_target,
                "device": full_device,
                "quant": full_quant
            }
            if full_lib_output:
                compile_payload["output"] = full_lib_output
                
            stream_endpoint(client, "POST", "/compile", json_payload=compile_payload)
            
            # Verification: Compiled Lib
            best_lib = get_newest_artifact(client, "compiled_lib")
            if not best_lib:
                print("\n[ERROR] /compile stream completed, but no compiled_lib artifacts found in /artifacts.")
                sys.exit(1)
                
            lib_path = full_lib_output if full_lib_output else best_lib.get("path")
            print(f"   -> Selected newest compiled_lib artifact: {best_lib.get('name')} (path: {lib_path})")

            # Step 8: Run
            print("\n8. Testing /run stream...")
            run_target_name = full_model_name if full_model_name else quantized_model_name
            print(f"   -> Using model name '{run_target_name}' for load-test.")
            
            run_payload = {
                "model_name": run_target_name,
                "device": full_device,
                "profile": "low"
            }
            if full_model_url:
                run_payload["model_url"] = full_model_url
                
            run_model_lib = full_model_lib if full_model_lib else lib_path
            run_payload["model_lib"] = run_model_lib
            
            stream_endpoint(client, "POST", "/run", json_payload=run_payload)
            
            print("\n=== Full Integration Test Passed ===")
            
    finally:
        if download_cache_path and was_downloaded_this_run and os.environ.get("CLEANUP_FULL_MODEL") == "1":
            print(f"\n[CLEANUP] Removing cached raw model at: {download_cache_path}")
            shutil.rmtree(download_cache_path, ignore_errors=True)

if __name__ == "__main__":
    main()
