#!/usr/bin/env python3
"""Pre-compute simulation cache for all NEC catalog files.

Reads every .nec file from a directory and POSTs it to the nec-solver
/pattern and /run endpoints. The solver caches results transparently,
so subsequent user requests get instant responses.

Usage:
    # From host (solver must be running):
    python scripts/warmup_cache.py --nec-dir ./nec_files --solver-url http://localhost:8787

    # From inside the dashboard container:
    python /app/scripts/warmup_cache.py --nec-dir /data/nec_files

    # With parallelism (default: 4 workers):
    python scripts/warmup_cache.py --nec-dir ./nec_files --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _post_json(url: str, data: dict, timeout: int = 300) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def warmup_file(nec_path: Path, solver_url: str) -> dict:
    """Run pattern + run endpoints for a single NEC file. Returns status dict."""
    nec_text = nec_path.read_text(errors="replace")
    result = {"file": nec_path.name, "pattern": None, "run": None}

    # 1. POST /pattern (primary path — includes impedance + pattern)
    try:
        resp = _post_json(f"{solver_url}/pattern", {"nec_text": nec_text})
        result["pattern"] = "ok" if resp.get("ok") else resp.get("error", "fail")
    except Exception as e:
        result["pattern"] = f"error: {e}"

    # 2. POST /run (fallback path — impedance only, different cache key)
    try:
        resp = _post_json(f"{solver_url}/run", {"nec_deck": nec_text})
        result["run"] = "ok" if resp.get("ok") else resp.get("error", "fail")
    except Exception as e:
        result["run"] = f"error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Warm up NEC simulation cache")
    parser.add_argument("--nec-dir", default=os.getenv("NEC_DIR", "."),
                        help="Directory containing .nec files")
    parser.add_argument("--solver-url", default=os.getenv("NEC_SOLVER_URL", "http://localhost:8787"),
                        help="NEC solver base URL")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without running simulations")
    args = parser.parse_args()

    nec_dir = Path(args.nec_dir)
    files = sorted(nec_dir.rglob("*.nec"))
    if not files:
        print(f"No .nec files found in {nec_dir}")
        return

    print(f"Found {len(files)} NEC files in {nec_dir}")

    # Check solver health
    try:
        health_url = f"{args.solver_url}/healthz"
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            if resp.status != 200:
                print(f"Solver not healthy at {args.solver_url}")
                sys.exit(1)
    except Exception as e:
        print(f"Cannot reach solver at {args.solver_url}: {e}")
        sys.exit(1)

    # Check current cache stats
    try:
        stats = _post_json(f"{args.solver_url}/cache/stats", {})
        print(f"Cache before: {stats.get('cached_entries', '?')} entries")
    except Exception:
        # GET endpoint, use urllib directly
        try:
            with urllib.request.urlopen(f"{args.solver_url}/cache/stats", timeout=5) as resp:
                stats = json.loads(resp.read())
                print(f"Cache before: {stats.get('cached_entries', '?')} entries")
        except Exception:
            pass

    if args.dry_run:
        for f in files:
            print(f"  {f.name}")
        return

    t0 = time.time()
    ok_pattern = ok_run = fail_pattern = fail_run = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(warmup_file, f, args.solver_url): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                r = future.result()
                if r["pattern"] == "ok":
                    ok_pattern += 1
                else:
                    fail_pattern += 1
                if r["run"] == "ok":
                    ok_run += 1
                else:
                    fail_run += 1
                # Progress every 50 files
                if i % 50 == 0 or i == len(files):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(files) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(files)}] {rate:.1f} files/s, ETA {eta:.0f}s "
                          f"| pattern ok={ok_pattern} fail={fail_pattern} "
                          f"| run ok={ok_run} fail={fail_run}")
            except Exception as e:
                fail_pattern += 1
                fail_run += 1
                print(f"  ERROR {futures[future].name}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Pattern: {ok_pattern} ok, {fail_pattern} failed")
    print(f"  Run:     {ok_run} ok, {fail_run} failed")

    # Final cache stats
    try:
        with urllib.request.urlopen(f"{args.solver_url}/cache/stats", timeout=5) as resp:
            stats = json.loads(resp.read())
            print(f"  Cache after: {stats.get('cached_entries', '?')} entries")
    except Exception:
        pass


if __name__ == "__main__":
    main()
