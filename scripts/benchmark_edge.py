#!/usr/bin/env python3
"""
benchmark_edge.py

A reproducible benchmark script for Edge/HPC backend LLM serving.
Measures FastAPI + MLC-CLI Docker backend latency and throughput.
Produces JSON, CSV, JSONL, and Markdown summaries.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def get_git_info() -> Dict[str, str]:
    info = {"commit": "", "branch": "", "dirty": False}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        info["commit"] = commit
        info["branch"] = branch
        info["dirty"] = len(status) > 0
    except Exception:
        pass
    return info


def get_gpu_info() -> Dict[str, Any]:
    info = {}
    try:
        # Check nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        lines = output.split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 4:
                info["name"] = parts[0]
                info["driver_version"] = parts[1]
                info["memory_total_mb"] = int(parts[2])
                info["memory_used_mb"] = int(parts[3])
    except Exception:
        pass
    return info


def get_env_metadata(args) -> Dict[str, Any]:
    is_wsl = False
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                is_wsl = True
    except Exception:
        pass

    is_docker = os.path.exists("/.dockerenv")
    
    return {
        "os": os.name,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "is_wsl": is_wsl,
        "is_docker": is_docker,
        "python_version": platform.python_version(),
        "cpu_info": platform.processor(),
        "machine": platform.machine(),
        "cwd": os.getcwd(),
        "server_url": args.server,
        "model": args.model,
        "backend": args.backend,
        "quant": args.quant,
        "compile_device": args.device if args.device else args.compile_device,
        "chat_device": args.device if args.device else args.chat_device,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "prompt_hash": hashlib.sha256(args.prompt.encode()).hexdigest()[:16],
        "temperature": args.temperature,
        "git": get_git_info(),
        "gpu_before": get_gpu_info()
    }


def call_api(url: str, payload: Optional[Dict] = None, timeout: int = 300) -> Tuple[int, str, float]:
    start = time.perf_counter()
    req = urllib.request.Request(url)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        req.data = data
    
    status_code = 500
    body_text = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            pass
    except urllib.error.URLError as e:
        status_code = 0
        body_text = str(e.reason)
    except Exception as e:
        status_code = 0
        body_text = str(e)
        
    latency = time.perf_counter() - start
    return status_code, body_text, latency


def check_sse_success(status: int, body: str) -> bool:
    return status == 200 and "[DONE]" in body and "[ERROR]" not in body


def check_windows(allow: bool):
    if platform.system() == "Windows":
        print("WARNING: Native Windows detected. The preferred benchmark environment for this repo is Ubuntu WSL or Linux.")
        if not allow:
            print("ERROR: Benchmark stopped.")
            print("Please rerun this script from WSL, or pass --allow-windows-client to override and run natively.")
            exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Edge AI Benchmark Tool for FastAPI MLC Docker")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model selector")
    parser.add_argument("--backend", default="FastAPI + MLC-CLI Docker", help="Backend name for reporting")
    parser.add_argument("--mode", choices=["serve", "prepare", "full"], default="serve", help="Benchmark mode")
    parser.add_argument("--runs", type=int, default=20, help="Number of measured inference runs")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup inference runs")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--prompt", default="Explain edge AI in one short paragraph.", help="Prompt text")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--device", default=None, help="Legacy: sets both compile and chat devices if specified")
    parser.add_argument("--compile-device", default="cuda", help="Target device for compile")
    parser.add_argument("--chat-device", default="cuda:0", help="Target device for chat/load")
    parser.add_argument("--quant", default="q4f16_1", help="Quantization format if preparing")
    parser.add_argument("--load-if-needed", action="store_true", help="Attempt to load the model if not loaded")
    parser.add_argument("--output-dir", default="benchmarks", help="Output directory for reports")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--allow-windows-client", action="store_true", help="Allow running natively on Windows")
    parser.add_argument("--include-prepare", action="store_true", help="Alias for --mode full or prepare")
    return parser.parse_args()


def calculate_stats(runs: List[Dict]) -> Dict:
    successes = [r for r in runs if r["success"]]
    failures = [r for r in runs if not r["success"]]
    
    count = len(successes)
    fail_count = len(failures)
    total = count + fail_count
    
    stats = {
        "total_runs": total,
        "success_count": count,
        "failure_count": fail_count,
        "success_rate": count / total if total > 0 else 0
    }
    
    if count == 0:
        return stats
        
    latencies = sorted(r["client_latency_seconds"] for r in successes)
    avg_latency = sum(latencies) / count
    stats["avg_latency"] = avg_latency
    stats["min_latency"] = latencies[0]
    stats["max_latency"] = latencies[-1]
    stats["median_latency"] = latencies[count // 2] if count % 2 == 1 else (latencies[count // 2 - 1] + latencies[count // 2]) / 2.0
    
    # Standard deviation
    if count > 1:
        variance = sum((x - avg_latency) ** 2 for x in latencies) / (count - 1)
        stats["std_dev"] = variance ** 0.5
        # Approx 95% CI for mean
        margin = 1.96 * (stats["std_dev"] / (count ** 0.5))
        stats["mean_ci_95"] = f"+/- {margin:.4f}"
    else:
        stats["std_dev"] = 0.0
        stats["mean_ci_95"] = "N/A"
        
    # Percentiles
    def percentile(p):
        if not latencies: return 0.0
        if len(latencies) == 1: return latencies[0]
        idx = (p / 100.0) * (count - 1)
        lower = int(idx)
        upper = lower + 1
        weight = idx - lower
        if upper >= count:
            return latencies[lower]
        return latencies[lower] * (1 - weight) + latencies[upper] * weight
        
    stats["p50"] = percentile(50)
    stats["p90"] = percentile(90)
    stats["p95"] = percentile(95)
    stats["p99"] = percentile(99)
    
    chars_sec = [r["chars_per_second"] for r in successes if r.get("chars_per_second")]
    stats["avg_chars_per_sec"] = sum(chars_sec) / len(chars_sec) if chars_sec else 0.0
    
    tokens_sec = [r["tokens_per_second"] for r in successes if r.get("tokens_per_second") is not None]
    if tokens_sec:
        stats["avg_tokens_per_sec"] = sum(tokens_sec) / len(tokens_sec)
    
    return stats


def save_markdown(out_path: str, args, meta: Dict, prep: List[Dict], serve_stats: Dict):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Edge Benchmark Report\n\n")
        f.write(f"Generated at: {datetime.datetime.now().isoformat()}\n\n")
        
        f.write("## Environment\n\n")
        f.write("| Key | Value |\n|---|---|\n")
        f.write(f"| OS | {meta['os']} / {meta['platform']} |\n")
        f.write(f"| WSL | {'Yes' if meta['is_wsl'] else 'No'} |\n")
        f.write(f"| Docker | {'Yes' if meta['is_docker'] else 'No'} |\n")
        f.write(f"| CPU | {meta['cpu_info']} |\n")
        f.write(f"| Python | {meta['python_version']} |\n")
        if meta["git"].get("commit"):
            f.write(f"| Git Commit | {meta['git']['commit']} (dirty: {meta['git']['dirty']}) |\n")
        if meta.get("gpu_before") and "name" in meta["gpu_before"]:
            f.write(f"| GPU | {meta['gpu_before']['name']} (Driver: {meta['gpu_before']['driver_version']}) |\n")
        f.write("\n")
        
        f.write("## Benchmark Settings\n\n")
        f.write("| Setting | Value |\n|---|---|\n")
        f.write(f"| Mode | {meta.get('mode', args.mode)} |\n")
        f.write(f"| Backend | {args.backend} |\n")
        f.write(f"| Model | {args.model} |\n")
        f.write(f"| Compile Device | {meta['compile_device']} |\n")
        f.write(f"| Chat Device | {meta['chat_device']} |\n")
        f.write(f"| Prompt Hash | {meta['prompt_hash']} |\n")
        f.write(f"| Max Tokens | {args.max_tokens} |\n")
        f.write(f"| Runs | {args.runs} (Warmup: {args.warmup}) |\n")
        f.write("\n")
        
        if prep:
            f.write("## Preparation Timings\n\n")
            f.write("| Step | Status Code | Latency (s) | Success |\n|---|---|---|---|\n")
            for p in prep:
                f.write(f"| {p['step']} | {p['status']} | {p['latency']:.2f} | {p['success']} |\n")
            f.write("\n")
            
        f.write("## Serving Results\n\n")
        if serve_stats.get("success_count", 0) > 0:
            f.write("| Backend | Runs | Avg Latency | Median | p90 | p95 | p99 | Min | Max | Success Rate | Avg chars/sec | Avg tokens/sec |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            s = serve_stats
            tps = f"{s.get('avg_tokens_per_sec', 0):.2f}" if "avg_tokens_per_sec" in s else "N/A"
            f.write(f"| {args.backend} | {s['success_count']}/{s['total_runs']} | {s['avg_latency']:.3f} | {s['median_latency']:.3f} | {s['p90']:.3f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['min_latency']:.3f} | {s['max_latency']:.3f} | {s['success_rate']*100:.1f}% | {s['avg_chars_per_sec']:.1f} | {tps} |\n")
        else:
            f.write("> **No successful runs to report.**\n")
            
        f.write("\n## Notes\n\n")
        f.write("* Results are hardware/runtime-specific.\n")
        f.write("* Do not compare directly with Apple M3/Metal or llama-cpp results unless hardware/runtime match.\n")
        f.write("* This benchmark measures FastAPI + Docker/Linux/MLC-CLI serving path.\n")


def generate_output_paths(output_dir: str, mode: str, ts: str = None) -> tuple:
    if not ts:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(output_dir, f"bench_{ts}_{mode}")
    return (
        f"{prefix}.json",
        f"{prefix}_runs.csv",
        f"{prefix}_summary.md",
        f"{prefix}_raw.jsonl"
    )


def main():
    args = parse_args()
    check_windows(args.allow_windows_client)
    
    os.makedirs(args.output_dir, exist_ok=True)
    mode = "full" if args.include_prepare else args.mode
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, csv_path, md_path, jsonl_path = generate_output_paths(args.output_dir, mode, ts)
    
    meta = get_env_metadata(args)
    meta["mode"] = mode
    compile_device = meta["compile_device"]
    chat_device = meta["chat_device"]
    prep_runs = []
    
    print(f"Starting {mode} benchmark for {args.model} on {args.backend}")
    print(f"Server: {args.server}")
    
    # Server Setup Check
    status, body, lat = call_api(f"{args.server}/setup-check", timeout=10)
    meta["setup_check"] = {"status": status, "latency": lat, "body": body[:500]}
    
    prepare_failed = False
    
    # Prepare Mode
    if mode in ["prepare", "full"]:
        print("Running preparation steps (/quantize -> /compile -> /chat/load)...")
        # Quantize
        q_payload = {"model": args.model, "quant": args.quant}
        status, body, lat = call_api(f"{args.server}/quantize", payload=q_payload, timeout=args.timeout)
        success = check_sse_success(status, body)
        prep_runs.append({"step": "/quantize", "status": status, "latency": lat, "success": success, "error": "" if success else body[:200]})
        print(f"  /quantize: {lat:.2f}s (Success: {success})")
        if not success:
            prepare_failed = True
        
        # Compile
        if not prepare_failed:
            c_payload = {"model": args.model, "quant": args.quant, "device": compile_device}
            status, body, lat = call_api(f"{args.server}/compile", payload=c_payload, timeout=args.timeout)
            success = check_sse_success(status, body)
            prep_runs.append({"step": "/compile", "status": status, "latency": lat, "success": success, "error": "" if success else body[:200]})
            print(f"  /compile: {lat:.2f}s (Success: {success})")
            if not success:
                prepare_failed = True
        
    # Check status and Load if needed
    status, body, lat = call_api(f"{args.server}/chat/status", timeout=10)
    loaded = False
    if status == 200:
        try:
            j = json.loads(body)
            loaded = j.get("loaded", False)
        except:
            pass
            
    if not prepare_failed and not loaded and (args.load_if_needed or mode in ["prepare", "full"]):
        print("Model not loaded, calling /chat/load...")
        l_payload = {"model": args.model, "device": chat_device}
        status, body, lat = call_api(f"{args.server}/chat/load", payload=l_payload, timeout=args.timeout)
        success = status == 200
        prep_runs.append({"step": "/chat/load", "status": status, "latency": lat, "success": success, "error": "" if success else body[:200]})
        print(f"  /chat/load: {lat:.2f}s (Success: {success})")
        if not success:
            prepare_failed = True
    elif not prepare_failed and not loaded:
        print("ERROR: Model is not loaded on the server. Please load the model first or pass --load-if-needed.")
        prepare_failed = True
        
    warmup_results = []
    measured_results = []

    if prepare_failed:
        print("Preparation failed. Skipping serving benchmark and saving partial report.")
    elif mode == "prepare":
        print("Prepare mode finished. Skipping serving benchmark.")
    else:
        # Serve Mode (Inference)
        completion_url = f"{args.server}/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": False
        }
        
        # Verify endpoint accepts optional fields, gracefully retry if not
        print("Verifying /chat/completions schema...")
        status, body, lat = call_api(completion_url, payload=payload, timeout=60)
        meta["schema_probe"] = {"status": status, "latency": lat, "stripped": False}
        if status == 422:
            print("  Temperature/optional field rejected by schema. Retrying stripped...")
            payload = {
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "stream": False
            }
            status, body, lat = call_api(completion_url, payload=payload, timeout=60)
            meta["schema_probe"]["stripped"] = True
            meta["schema_probe"]["retry_status"] = status
            meta["schema_probe"]["retry_latency"] = lat
            
        if status != 200:
            print(f"ERROR: Schema probe failed with status {status}. Body: {body[:200]}")
            prepare_failed = True
        else:
            def run_inference(index: int, is_warmup: bool) -> Dict:
                now = datetime.datetime.now().isoformat()
                s_code, text, run_lat = call_api(completion_url, payload=payload, timeout=args.timeout)
                
                result = {
                    "run_index": index,
                    "is_warmup": is_warmup,
                    "timestamp": now,
                    "client_latency_seconds": run_lat,
                    "status": s_code,
                    "success": s_code == 200
                }
                
                if s_code == 200:
                    try:
                        j_data = json.loads(text)
                        content = j_data["choices"][0]["message"]["content"]
                        result["response_length"] = len(content)
                        result["response_preview"] = content.replace("\n", " ")[:100]
                        result["chars_per_second"] = len(content) / run_lat if run_lat > 0 else 0
                        
                        usage = j_data.get("usage", {})
                        result["usage"] = usage
                        
                        out_tokens = usage.get("completion_tokens")
                        if out_tokens:
                            result["tokens_per_second"] = out_tokens / run_lat if run_lat > 0 else 0
                        else:
                            est = len(content) / 4.0
                            result["estimated_tokens"] = est
                            result["tokens_per_second"] = est / run_lat if run_lat > 0 else 0
                    except Exception as e:
                        result["success"] = False
                        result["error"] = f"Parse error: {str(e)}"
                else:
                    result["error"] = text[:200]
                    
                return result

            print(f"Running {args.warmup} warmup requests...")
            for i in range(args.warmup):
                res = run_inference(i, True)
                warmup_results.append(res)
                print(f"  Warmup {i+1}/{args.warmup}: {res['client_latency_seconds']:.3f}s (Success: {res['success']})")
                
            print(f"Running {args.runs} measured requests...")
            for i in range(args.runs):
                res = run_inference(i, False)
                measured_results.append(res)
                print(f"  Run {i+1}/{args.runs}: {res['client_latency_seconds']:.3f}s (Success: {res['success']})")
        
    meta["gpu_after"] = get_gpu_info()
    
    # Calculate stats
    stats = calculate_stats(measured_results)
    
    # Save outputs
    # Paths already generated at start of main
    
    full_data = {
        "metadata": meta,
        "preparation": prep_runs,
        "warmup": warmup_results,
        "measured": measured_results,
        "statistics": stats
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2)
        
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in warmup_results + measured_results:
            f.write(json.dumps(r) + "\n")
            
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        keys = ["run_index", "timestamp", "success", "status", "client_latency_seconds", "response_length", "tokens_per_second", "error"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        if measured_results:
            writer.writerows(measured_results)
            
    save_markdown(md_path, args, meta, prep_runs, stats)
    
    print("\nBenchmark Complete!")
    print(f"Mode       : {mode}")
    print(f"Backend    : {args.backend}")
    print(f"Model      : {args.model}")
    print(f"Successful : {stats.get('success_count', 0)}/{stats.get('total_runs', 0)}")
    
    if stats.get('success_count', 0) > 0:
        print(f"Avg Latency: {stats['avg_latency']:.3f}s")
        print(f"Med Latency: {stats['median_latency']:.3f}s")
        print(f"P95 Latency: {stats['p95']:.3f}s")
        print(f"Success Rt : {stats['success_rate']*100:.1f}%")
        
    print(f"\nOutputs saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {jsonl_path}")
    if prepare_failed:
        print("\nExiting with non-zero status due to preparation/probe failure.")
        exit(1)


if __name__ == "__main__":
    main()
