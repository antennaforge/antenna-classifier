#!/usr/bin/env python3
"""Lightweight load test harness for the NEC solver API.

This script is intentionally stdlib-only so it can run from the host or inside
the repo venv without extra dependencies.

Examples:
    # Smoke test against local solver using 20 requests and 4 workers
    python scripts/load_test_solver.py \
        --solver-url http://localhost:8787 \
        --nec-dir ./nec_files \
        --endpoint run \
        --requests 20 \
        --workers 4

    # Sustained mixed endpoint load for 60s (run + pattern)
    python scripts/load_test_solver.py \
        --solver-url http://localhost:8787 \
        --nec-dir ./nec_files \
        --endpoint mixed \
        --duration 60 \
        --workers 8

    # Bias toward cache misses by varying the payload
    python scripts/load_test_solver.py \
        --solver-url http://localhost:8787 \
        --nec-dir ./nec_files \
        --endpoint run \
        --requests 50 \
        --workers 6 \
        --cache-mode cold
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any


ENDPOINT_CHOICES = ("run", "pattern", "currents", "mixed")


def _request_json(url: str, data: dict[str, Any], timeout: int = 300) -> tuple[int, dict[str, Any]]:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
        return resp.status, payload


def _get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _ensure_solver_health(solver_url: str) -> None:
    try:
        payload = _get_json(f"{solver_url}/healthz")
    except Exception as exc:
        raise RuntimeError(f"Cannot reach solver at {solver_url}: {exc}") from exc
    if not payload.get("ok"):
        raise RuntimeError(f"Solver health check failed at {solver_url}: {payload}")


def _load_nec_files(nec_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(nec_dir.rglob("*.nec"))
    if limit is not None:
        files = files[:limit]
    return files


def _mutate_nec_text(nec_text: str, request_index: int) -> str:
    """Slightly perturb the deck to avoid cache hits while preserving semantics.

    We prepend a unique CM line because sanitize_nec strips comment cards, so use
    harmless trailing whitespace on the first non-comment line instead.
    """
    lines = nec_text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped and not stripped.startswith("CM"):
            lines[idx] = f"{line} {' ' * ((request_index % 7) + 1)}"
            break
    return "\n".join(lines)


def _build_payload(endpoint: str, nec_text: str, request_index: int, cache_mode: str) -> dict[str, Any]:
    if cache_mode == "cold":
        nec_text = _mutate_nec_text(nec_text, request_index)

    if endpoint == "run":
        return {"nec_deck": nec_text}
    if endpoint == "pattern":
        return {"nec_text": nec_text}
    if endpoint == "currents":
        return {"nec_deck": nec_text}
    raise ValueError(f"Unsupported endpoint {endpoint}")


def _choose_endpoint(request_index: int, endpoint: str) -> str:
    if endpoint != "mixed":
        return endpoint
    cycle = ("run", "pattern", "currents")
    return cycle[request_index % len(cycle)]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _summarize_results(results: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    latencies = [result["latency_ms"] for result in results if result.get("latency_ms") is not None]
    endpoint_counts = Counter(result["endpoint"] for result in results)
    outcome_counts = Counter(result["outcome"] for result in results)
    error_counts = Counter(result.get("error") for result in results if result.get("error"))

    summary = {
        "elapsed_s": round(elapsed_s, 3),
        "requests": len(results),
        "requests_per_s": round(len(results) / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        "endpoint_counts": dict(endpoint_counts),
        "outcome_counts": dict(outcome_counts),
        "error_counts": dict(error_counts),
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "max": round(max(latencies), 3) if latencies else None,
        },
    }
    if summary["latency_ms"]["p50"] is not None:
        for key in ("p50", "p95", "p99"):
            summary["latency_ms"][key] = round(float(summary["latency_ms"][key]), 3)
    return summary


def _run_single_request(
    solver_url: str,
    endpoint: str,
    nec_path: Path,
    request_index: int,
    cache_mode: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        nec_text = nec_path.read_text(errors="replace")
        payload = _build_payload(endpoint, nec_text, request_index, cache_mode)
        status, resp = _request_json(f"{solver_url}/{endpoint}", payload, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if resp.get("ok"):
            return {
                "endpoint": endpoint,
                "file": nec_path.name,
                "status": status,
                "outcome": "success",
                "latency_ms": latency_ms,
            }
        return {
            "endpoint": endpoint,
            "file": nec_path.name,
            "status": status,
            "outcome": "error",
            "error": resp.get("error", "unknown"),
            "latency_ms": latency_ms,
        }
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "endpoint": endpoint,
            "file": nec_path.name,
            "status": exc.code,
            "outcome": "http_error",
            "error": str(exc),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "endpoint": endpoint,
            "file": nec_path.name,
            "status": None,
            "outcome": "exception",
            "error": type(exc).__name__,
            "detail": str(exc),
            "latency_ms": latency_ms,
        }


def _collect_solver_snapshot(solver_url: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        snapshot["healthz"] = _get_json(f"{solver_url}/healthz")
    except Exception as exc:
        snapshot["healthz_error"] = str(exc)
    try:
        snapshot["cache_stats"] = _get_json(f"{solver_url}/cache/stats")
    except Exception as exc:
        snapshot["cache_stats_error"] = str(exc)
    try:
        snapshot["telemetry_summary"] = _get_json(f"{solver_url}/telemetry/summary")
    except Exception as exc:
        snapshot["telemetry_summary_error"] = str(exc)
    try:
        snapshot["telemetry_recent"] = _get_json(f"{solver_url}/telemetry/recent")
    except Exception as exc:
        snapshot["telemetry_recent_error"] = str(exc)
    return snapshot


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nLoad test summary")
    print(f"  Requests:      {summary['requests']}")
    print(f"  Elapsed:       {summary['elapsed_s']:.2f}s")
    print(f"  Throughput:    {summary['requests_per_s']:.2f} req/s")
    print(f"  Endpoints:     {summary['endpoint_counts']}")
    print(f"  Outcomes:      {summary['outcome_counts']}")
    if summary["error_counts"]:
        print(f"  Errors:        {summary['error_counts']}")
    lat = summary["latency_ms"]
    print(f"  Latency avg:   {lat['avg']} ms")
    print(f"  Latency p50:   {lat['p50']} ms")
    print(f"  Latency p95:   {lat['p95']} ms")
    print(f"  Latency p99:   {lat['p99']} ms")
    print(f"  Latency max:   {lat['max']} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test the NEC solver endpoints safely")
    parser.add_argument("--solver-url", default=os.getenv("NEC_SOLVER_URL", "http://localhost:8787"),
                        help="NEC solver base URL")
    parser.add_argument("--nec-dir", default=os.getenv("NEC_DIR", "./nec_files"),
                        help="Directory containing .nec files to use as request payloads")
    parser.add_argument("--endpoint", choices=ENDPOINT_CHOICES, default="run",
                        help="Endpoint to exercise")
    parser.add_argument("--requests", type=int, default=0,
                        help="Fixed number of requests to send. Mutually exclusive with --duration")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run sustained load for N seconds. Mutually exclusive with --requests")
    parser.add_argument("--workers", type=int, default=4,
                        help="Maximum concurrent workers")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-request timeout in seconds")
    parser.add_argument("--file-limit", type=int, default=50,
                        help="Maximum number of NEC files to sample from the directory")
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm",
                        help="Warm reuses decks for cache-hit-biased testing; cold perturbs payloads")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for NEC file selection")
    parser.add_argument("--out", default="",
                        help="Optional path to write full JSON results")
    parser.add_argument("--print-solver-snapshots", action="store_true",
                        help="Capture solver cache/telemetry snapshots before and after the run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and print selected NEC files without sending load")
    args = parser.parse_args()

    if bool(args.requests) == bool(args.duration):
        parser.error("Specify exactly one of --requests or --duration")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    nec_dir = Path(args.nec_dir)
    files = _load_nec_files(nec_dir, limit=args.file_limit)
    if not files:
        parser.error(f"No .nec files found in {nec_dir}")

    random.seed(args.seed)
    _ensure_solver_health(args.solver_url)

    print(f"Selected {len(files)} NEC files from {nec_dir}")
    print(f"Target solver: {args.solver_url}")
    print(f"Endpoint mode: {args.endpoint}")
    print(f"Cache mode:    {args.cache_mode}")
    print(f"Workers:       {args.workers}")

    if args.dry_run:
        for path in files[: min(10, len(files))]:
            print(f"  {path.name}")
        return 0

    before_snapshot = _collect_solver_snapshot(args.solver_url) if args.print_solver_snapshots else None
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    started = time.perf_counter()

    def _record(result: dict[str, Any]) -> None:
        with results_lock:
            results.append(result)

    request_counter = 0
    deadline = started + args.duration if args.duration else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: set[Future] = set()

        def submit_one() -> bool:
            nonlocal request_counter
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            if args.requests and request_counter >= args.requests:
                return False
            endpoint = _choose_endpoint(request_counter, args.endpoint)
            nec_path = random.choice(files)
            future = pool.submit(
                _run_single_request,
                args.solver_url,
                endpoint,
                nec_path,
                request_counter,
                args.cache_mode,
                args.timeout,
            )
            futures.add(future)
            request_counter += 1
            return True

        for _ in range(args.workers):
            if not submit_one():
                break

        while futures:
            done, pending = wait(futures, return_when=FIRST_COMPLETED)
            futures = set(pending)
            for future in done:
                _record(future.result())
                submit_one()

    elapsed_s = time.perf_counter() - started
    summary = _summarize_results(results, elapsed_s)
    _print_summary(summary)

    payload = {
        "config": {
            "solver_url": args.solver_url,
            "endpoint": args.endpoint,
            "requests": args.requests,
            "duration": args.duration,
            "workers": args.workers,
            "cache_mode": args.cache_mode,
            "file_limit": args.file_limit,
        },
        "summary": summary,
        "before_snapshot": before_snapshot,
        "after_snapshot": _collect_solver_snapshot(args.solver_url) if args.print_solver_snapshots else None,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON summary to {out_path}")

    return 0 if summary["requests"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())