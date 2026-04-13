import os
import sys
import httpx

API_URL = "http://localhost:8000"

def stream_endpoint(client, method, url, json_payload=None):
    """Helper to stream SSE responses and print them live."""
    try:
        with client.stream(method, url, json=json_payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    msg = line[6:]
                    if msg == "[DONE]":
                        print(f"\n--- {url} Completed Successfully ---")
                        break
                    elif msg.startswith("[ERROR]"):
                        print(f"\n--- {url} Failed: {msg} ---")
                        sys.exit(1)
                    else:
                        print(msg)
    except Exception as e:
        print(f"\n[ERROR] Failed during stream of {url}: {e}")
        sys.exit(1)

def main():
    print("=== MLC-CLI Integration Smoke Test ===")
    print("Ensure the FastAPI app is running at http://localhost:8000")
    
    with httpx.Client(base_url=API_URL, timeout=None) as client:
        # 1. Health check
        print("\n1. Checking API Health...")
        try:
            resp = client.get("/health")
            resp.raise_for_status()
            print("   -> API is healthy.")
        except Exception as e:
            print(f"[ERROR] API is unreachable: {e}")
            sys.exit(1)

        # 2. Setup check
        print("\n2. Checking setup (/setup-check)...")
        resp = client.get("/setup-check")
        resp.raise_for_status()
        setup_data = resp.json()
        print(f"   -> Repo exists: {setup_data.get('repo_exists')}")
        
        # 3. Ensure repo exists
        print("\n3. Ensuring mlc-cli repo exists (/ensure-repo-exists)...")
        if not setup_data.get("repo_exists"):
            client.post("/ensure-repo-exists").raise_for_status()
            print("   -> Repo cloned successfully.")
        else:
            print("   -> Repo already exists, skipping clone.")

        # 4. Repo status
        print("\n4. Checking repo status (/repo-status)...")
        resp = client.get("/repo-status")
        resp.raise_for_status()
        status_data = resp.json()
        print(f"   -> Clean: {status_data.get('is_clean')}")

        # 5. Fast Build (install-wheels)
        print("\n5. Testing /build stream (fast 'install-wheels' action)...")
        build_payload = {
            "action": "install-wheels",
            "cuda": "y",
            "cutlass": "n",
            "flash_infer": "n"
        }
        stream_endpoint(client, "POST", "/build", json_payload=build_payload)

        # 6. Artifacts discovery
        print("\n6. Checking /artifacts...")
        resp = client.get("/artifacts")
        resp.raise_for_status()
        artifacts_data = resp.json()
        total_artifacts = artifacts_data.get("counts", {}).get("total", 0)
        print(f"   -> Discovered {total_artifacts} artifacts.")

        # 7. Run (load-test)
        print("\n7. Testing /run stream (lightweight load-test)...")
        
        # 1st Priority: Explicit Environment Variables
        env_model_name = os.environ.get("RUN_MODEL_NAME")
        env_model_url = os.environ.get("RUN_MODEL_URL")
        env_model_lib = os.environ.get("RUN_MODEL_LIB")
        env_device = os.environ.get("RUN_DEVICE", "cuda")
        
        run_payload = None

        if env_model_name:
            print(f"   -> [ENV] Using explicit RUN_MODEL_NAME='{env_model_name}'")
            run_payload = {
                "model_name": env_model_name,
                "device": env_device,
                "profile": "low"
            }
            if env_model_url:
                run_payload["model_url"] = env_model_url
            if env_model_lib:
                run_payload["model_lib"] = env_model_lib
        else:
            # 2nd Priority: Discovered Artifacts
            models = [a for a in artifacts_data.get("artifacts", []) if a.get("type") == "model_dir"]
            if models:
                model_name = models[0].get("name")
                print(f"   -> [ARTIFACTS] Found local model dir '{model_name}'. Initiating load-test...")
                run_payload = {
                    "model_name": model_name,
                    "device": "cuda",
                    "profile": "low"
                }

        # Execute or skip
        if run_payload:
            stream_endpoint(client, "POST", "/run", json_payload=run_payload)
        else:
            # 3rd Priority: Optional Download
            if os.environ.get("DOWNLOAD_RUN_MODEL_IF_MISSING") == "1":
                dl_name = "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC"
                dl_url = f"https://huggingface.co/mlc-ai/{dl_name}"
                print(f"   -> [DOWNLOAD] Auto-download enabled. Fetching {dl_name}...")
                run_payload = {
                    "model_name": dl_name,
                    "model_url": dl_url,
                    "device": "cuda",
                    "profile": "low"
                }
                stream_endpoint(client, "POST", "/run", json_payload=run_payload)
            else:
                print("   -> [SKIP] No converted model directories found in /artifacts.")
                print("   -> Set DOWNLOAD_RUN_MODEL_IF_MISSING=1 to test auto-download, or skipping /run.")
        
    print("\n=== Smoke Test Passed ===")

if __name__ == "__main__":
    main()
