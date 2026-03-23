import httpx
import sys
import argparse
import subprocess
import time

API_URL = "http://localhost:8000"

def main():
    print("=== MLC-CLI Pipeline Test Script ===")
    
    with httpx.Client(base_url=API_URL, timeout=None) as client:
        # 1. Check Setup and verify it's pulled
        print("\n1. Checking setup and if ML-CLI is pulled from GitHub...")
        try:
            resp = client.get("/setup-check")
            resp.raise_for_status()
            data = resp.json()
            if not data.get("repo_exists"):
                print("   -> Repo not found! Pulling from GitHub...")
                client.post("/ensure-repo-exists").raise_for_status()
                print("   -> Repository successfully cloned.")
            else:
                print("   -> Repository is already pulled and exists.")
        except Exception as e:
            print(f"[ERROR] Failed to check setup: {e}")
            sys.exit(1)

        # 2. Check if clean
        print("\n2. Checking if the repository is clean...")
        try:
            resp = client.get("/repo-status")
            resp.raise_for_status()
            data = resp.json()
            if data.get("is_clean"):
                print("   -> Repository is clean.")
            else:
                print("   -> Repository has uncommitted changes! Changes:")
                for change in data.get("changes", []):
                    print(f"      {change}")
                # The prompt asks to check if it's clean, but doesn't explicitly abort if it isn't.
                # We'll print a warning. Some users might still test local changes.
                print("   -> [WARNING] Proceeding with dirty repository.")
        except Exception as e:
            print(f"[ERROR] Failed to check repo status: {e}")
            sys.exit(1)

        # 3. Starting full build (cuda 86, cublas, tvm action full)
        print("\n3. Starting full build (CUDA 86, cuBLAS, TVM action: full)...")
        build_payload = {
            "action": "full",          # action full
            "cuda": "y",               # with cuda
            "cuda_arch": "86",         # cuda 86
            "cutlass": "n",            # disable cutlass to avoid libflash_attn build failures
            "cublas": "y",             # with cublas
            "tvm_source": "bundled",   # build tvm (bundled source)
            "build_wheels": "y",       # build wheels
            "force_clone": "n",        # do not force clone (use existing repo if available)
        }
        
        try:
            print("\n--- Build Output ---\n")
            with client.stream("POST", "/build", json=build_payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        msg = line[6:]
                        if msg == "[DONE]":
                            print("\n--- Build Completed Successfully ---")
                            break
                        elif msg.startswith("[ERROR]"):
                            print(f"\n--- Build Failed: {msg} ---")
                            sys.exit(1)
                        else:
                            print(msg)
        except Exception as e:
            print(f"\n[ERROR] Failed during build stream: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
