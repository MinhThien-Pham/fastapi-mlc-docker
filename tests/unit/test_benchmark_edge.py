import json
import pytest
import os
from scripts import benchmark_edge


def test_calculate_stats_empty():
    stats = benchmark_edge.calculate_stats([])
    assert stats["total_runs"] == 0
    assert stats["success_count"] == 0
    assert stats["success_rate"] == 0


def test_calculate_stats_basic():
    runs = [
        {"success": True, "client_latency_seconds": 0.5, "chars_per_second": 20, "tokens_per_second": 5},
        {"success": True, "client_latency_seconds": 1.5, "chars_per_second": 10, "tokens_per_second": 2},
        {"success": False, "client_latency_seconds": 10.0}
    ]
    
    stats = benchmark_edge.calculate_stats(runs)
    assert stats["total_runs"] == 3
    assert stats["success_count"] == 2
    assert stats["failure_count"] == 1
    
    # 2/3 success
    assert abs(stats["success_rate"] - 0.666) < 0.01
    
    assert stats["avg_latency"] == 1.0
    assert stats["min_latency"] == 0.5
    assert stats["max_latency"] == 1.5
    
    # percentiles with linear interpolation
    assert stats["p50"] == 1.0  # (0.5 + 1.5) / 2
    assert abs(stats["p99"] - 1.49) < 0.001  # 0.5 * 0.01 + 1.5 * 0.99
    
    assert stats["avg_chars_per_sec"] == 15.0
    assert stats["avg_tokens_per_sec"] == 3.5


def test_calculate_stats_single():
    runs = [
        {"success": True, "client_latency_seconds": 1.0, "chars_per_second": 10, "tokens_per_second": 5},
    ]
    
    stats = benchmark_edge.calculate_stats(runs)
    assert stats["std_dev"] == 0.0
    assert stats["mean_ci_95"] == "N/A"
    assert stats["p50"] == 1.0
    assert stats["p99"] == 1.0


def test_check_sse_success():
    # True positives
    assert benchmark_edge.check_sse_success(200, "data: [DONE]\n")
    assert benchmark_edge.check_sse_success(200, "some output... [DONE]")
    
    # False positives
    assert not benchmark_edge.check_sse_success(500, "data: [DONE]\n")
    assert not benchmark_edge.check_sse_success(200, "data: processing...")
    assert not benchmark_edge.check_sse_success(200, "data: [ERROR] failed\ndata: [DONE]\n")


def test_generate_output_paths():
    paths = benchmark_edge.generate_output_paths("benchmarks", "serve", "20260708_120000")

    expected_json = os.path.join("benchmarks", "bench_20260708_120000_serve.json")
    expected_csv = os.path.join("benchmarks", "bench_20260708_120000_serve_runs.csv")
    expected_md = os.path.join("benchmarks", "bench_20260708_120000_serve_summary.md")
    expected_jsonl = os.path.join("benchmarks", "bench_20260708_120000_serve_raw.jsonl")

    assert paths == (expected_json, expected_csv, expected_md, expected_jsonl)

    assert paths[0].endswith(".json") and not paths[0].endswith(".jsonl")
    assert paths[1].endswith(".csv")
    assert paths[2].endswith(".md")
    assert paths[3].endswith(".jsonl")
