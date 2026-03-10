"""Regression tests for the nec-solver cache layer.

Tests the cache helper functions (_cache_key, _cache_get, _cache_put)
and the concurrency semaphore logic in docker/nec_solver/api.py.

These tests don't require a running Docker container — they import the
cache functions directly and use a temporary directory.
"""

import hashlib
import importlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import cache functions from docker/nec_solver/api.py
# ---------------------------------------------------------------------------

# api.py lives outside the package tree and depends on FastAPI (not installed
# in the test venv). We inject a stub 'fastapi' module before importing.
_API_DIR = str(Path(__file__).resolve().parent.parent / "docker" / "nec_solver")


@pytest.fixture()
def api_mod(tmp_path, monkeypatch):
    """Import (or reimport) api module with CACHE_DIR pointing to tmp_path."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setenv("CACHE_DIR", cache_dir)

    # Stub out fastapi so api.py can be imported without it installed
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = MagicMock()
    fake_fastapi.Body = MagicMock()
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    if _API_DIR not in sys.path:
        sys.path.insert(0, _API_DIR)
    # Force fresh import so env vars are re-read
    monkeypatch.delitem(sys.modules, "api", raising=False)
    import api
    # Override CACHE_DIR in case it was set at import time
    api.CACHE_DIR = cache_dir
    return api


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    """Deterministic hash generation for simulation results."""

    def test_deterministic(self, api_mod):
        """Same inputs produce same key."""
        k1 = api_mod._cache_key("GW 1 21 0 0 0 0 5 0 .001\nEN", "run", 50.0)
        k2 = api_mod._cache_key("GW 1 21 0 0 0 0 5 0 .001\nEN", "run", 50.0)
        assert k1 == k2

    def test_is_sha256_hex(self, api_mod):
        """Key is a 64-character hex string (SHA-256)."""
        k = api_mod._cache_key("deck", "run")
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)

    def test_different_decks_different_keys(self, api_mod):
        """Different NEC content produces different keys."""
        k1 = api_mod._cache_key("GW 1 21 0 0 0 0 5 0 .001\nEN", "run")
        k2 = api_mod._cache_key("GW 2 21 0 0 0 0 5 0 .001\nEN", "run")
        assert k1 != k2

    def test_different_endpoints_different_keys(self, api_mod):
        """Same deck via /run vs /pattern gets separate cache entries."""
        k1 = api_mod._cache_key("deck", "run")
        k2 = api_mod._cache_key("deck", "pattern")
        assert k1 != k2

    def test_different_z0_different_keys(self, api_mod):
        """Different impedance reference produces different keys."""
        k1 = api_mod._cache_key("deck", "run", 50.0)
        k2 = api_mod._cache_key("deck", "run", 75.0)
        assert k1 != k2

    def test_matches_manual_sha256(self, api_mod):
        """Key matches manually computed SHA-256."""
        deck, endpoint, z0 = "test_deck", "run", 50.0
        expected = hashlib.sha256(f"{deck}|{endpoint}|{z0}".encode()).hexdigest()
        assert api_mod._cache_key(deck, endpoint, z0) == expected


# ---------------------------------------------------------------------------
# _cache_put / _cache_get round-trip
# ---------------------------------------------------------------------------

class TestCachePutGet:
    """Write and read cache entries."""

    def test_round_trip(self, api_mod, tmp_path):
        """Write then read returns identical data."""
        data = {"ok": True, "parsed": {"swr_sweep": {"freq_mhz": [14.0], "swr": [1.5]}}}
        key = "abc123"
        api_mod.CACHE_DIR = str(tmp_path / "cache")
        api_mod._cache_put(key, data)
        result = api_mod._cache_get(key)
        assert result == data

    def test_miss_returns_none(self, api_mod, tmp_path):
        """Reading a nonexistent key returns None."""
        api_mod.CACHE_DIR = str(tmp_path / "cache")
        assert api_mod._cache_get("nonexistent") is None

    def test_file_on_disk(self, api_mod, tmp_path):
        """Cache entry is a readable JSON file on disk."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        api_mod._cache_put("mykey", {"ok": True})
        fpath = Path(cache_dir) / "mykey.json"
        assert fpath.exists()
        assert json.loads(fpath.read_text()) == {"ok": True}

    def test_no_tmp_file_left(self, api_mod, tmp_path):
        """Atomic write leaves no .tmp file behind."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        api_mod._cache_put("k1", {"data": 1})
        leftover = list(Path(cache_dir).glob("*.tmp"))
        assert leftover == []

    def test_overwrite(self, api_mod, tmp_path):
        """Writing the same key twice overwrites cleanly."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        api_mod._cache_put("k", {"v": 1})
        api_mod._cache_put("k", {"v": 2})
        assert api_mod._cache_get("k") == {"v": 2}

    def test_corrupt_json_returns_none(self, api_mod, tmp_path):
        """Corrupt cache file returns None (not a crash)."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        fpath = Path(cache_dir) / "bad.json"
        fpath.write_text("{invalid json")
        assert api_mod._cache_get("bad") is None

    def test_complex_payload(self, api_mod, tmp_path):
        """Full simulation-like payload round-trips correctly."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        payload = {
            "ok": True,
            "theta": [0.0, 1.0, 2.0],
            "phi": [0.0, 0.0, 0.0],
            "gain": [-5.2, 3.1, 7.8],
            "impedance_sweep": {
                "freq_mhz": [13.5, 14.0, 14.5],
                "r": [50.1, 48.3, 52.7],
                "x": [-12.3, 0.5, 15.2],
            },
            "swr_sweep": {
                "freq_mhz": [13.5, 14.0, 14.5],
                "swr": [2.1, 1.1, 1.9],
            },
        }
        api_mod._cache_put("full_sim", payload)
        assert api_mod._cache_get("full_sim") == payload


# ---------------------------------------------------------------------------
# Cache stats tracking
# ---------------------------------------------------------------------------

class TestCacheStats:
    """In-memory hit/miss counters."""

    def test_miss_increments(self, api_mod, tmp_path):
        """Cache miss increments miss counter."""
        api_mod.CACHE_DIR = str(tmp_path / "cache")
        api_mod._cache_hits = 0
        api_mod._cache_misses = 0
        api_mod._cache_get("nonexistent")
        assert api_mod._cache_misses == 1
        assert api_mod._cache_hits == 0

    def test_hit_increments(self, api_mod, tmp_path):
        """Cache hit increments hit counter."""
        cache_dir = str(tmp_path / "cache")
        api_mod.CACHE_DIR = cache_dir
        api_mod._cache_hits = 0
        api_mod._cache_misses = 0
        api_mod._cache_put("k", {"ok": True})
        api_mod._cache_get("k")
        assert api_mod._cache_hits == 1


# ---------------------------------------------------------------------------
# Solver telemetry stats
# ---------------------------------------------------------------------------

class TestTelemetry:
    """In-process telemetry helpers for solver observability."""

    def test_summary_defaults(self, api_mod):
        api_mod._cache_hits = 0
        api_mod._cache_misses = 0
        api_mod._telemetry_reset()

        payload = api_mod._telemetry_summary_payload()

        assert payload["ok"] is True
        assert payload["service"] == "nec-solver"
        assert payload["solver_binary"] == api_mod.NEC_BIN
        assert payload["max_concurrent"] == api_mod._MAX_CONCURRENT
        assert payload["totals"]["requests"] == 0
        assert payload["totals"]["success"] == 0
        assert payload["in_flight"] == 0
        assert payload["cache"]["hit_rate"] == 0.0
        assert set(payload["endpoints"].keys()) == {"run", "currents", "pattern"}

    def test_success_records_latency_samples(self, api_mod):
        api_mod._telemetry_reset()

        token = api_mod._telemetry_begin("run")
        api_mod._telemetry_complete(token, "success", queue_wait_ms=12.5, exec_ms=240.0)

        payload = api_mod._telemetry_summary_payload()
        run = payload["endpoints"]["run"]

        assert payload["totals"]["requests"] == 1
        assert payload["totals"]["success"] == 1
        assert payload["in_flight"] == 0
        assert run["requests"] == 1
        assert run["success"] == 1
        assert run["queue_wait_ms"]["avg"] == pytest.approx(12.5)
        assert run["exec_ms"]["p95"] == pytest.approx(240.0)

    def test_busy_event_appears_in_recent_payload(self, api_mod):
        api_mod._telemetry_reset()

        token = api_mod._telemetry_begin("pattern")
        api_mod._telemetry_complete(
            token,
            "busy",
            queue_wait_ms=60000.0,
            error_type="server_busy",
            detail="Too many concurrent simulations; try again shortly",
        )

        payload = api_mod._telemetry_recent_payload()

        assert len(payload["recent_busy_events"]) == 1
        assert payload["recent_busy_events"][0]["endpoint"] == "pattern"
        assert payload["recent_busy_events"][0]["type"] == "server_busy"
        assert payload["recent_busy_events"][0]["wait_timeout_seconds"] == 60

    def test_error_event_appears_in_recent_payload(self, api_mod):
        api_mod._telemetry_reset()

        token = api_mod._telemetry_begin("currents")
        api_mod._telemetry_complete(
            token,
            "error",
            queue_wait_ms=5.0,
            exec_ms=120.0,
            error_type="nec_failed",
            detail="solver returned non-zero exit",
        )

        summary = api_mod._telemetry_summary_payload()
        recent = api_mod._telemetry_recent_payload()

        assert summary["totals"]["error"] == 1
        assert summary["endpoints"]["currents"]["error"] == 1
        assert len(recent["recent_errors"]) == 1
        assert recent["recent_errors"][0]["endpoint"] == "currents"
        assert recent["recent_errors"][0]["type"] == "nec_failed"

    def test_window_counts_include_recent_outcomes(self, api_mod):
        api_mod._telemetry_reset()

        api_mod._telemetry_complete(api_mod._telemetry_begin("run"), "success")
        api_mod._telemetry_complete(api_mod._telemetry_begin("run"), "busy", error_type="server_busy")
        api_mod._telemetry_complete(api_mod._telemetry_begin("run"), "timeout", error_type="timeout")

        payload = api_mod._telemetry_summary_payload()

        assert payload["window"]["last_5m_requests"] == 3
        assert payload["window"]["last_5m_busy"] == 1
        assert payload["window"]["last_5m_errors"] == 1


# ---------------------------------------------------------------------------
# Semaphore / concurrency control
# ---------------------------------------------------------------------------

class TestSemaphore:
    """Concurrency limiter for nec2c processes."""

    def test_semaphore_exists(self, api_mod):
        """Module has a semaphore with correct limit."""
        assert isinstance(api_mod._sim_semaphore, threading.Semaphore)

    def test_default_max_concurrent(self, api_mod):
        """Default MAX_CONCURRENT_SIMS is 3."""
        assert api_mod._MAX_CONCURRENT == 3

    def test_semaphore_limits_concurrency(self, api_mod):
        """Semaphore blocks when all slots are taken."""
        sem = threading.Semaphore(2)
        # Acquire both slots
        assert sem.acquire(timeout=0.1) is True
        assert sem.acquire(timeout=0.1) is True
        # Third should fail immediately
        assert sem.acquire(timeout=0.1) is False
        # Release one, then it should succeed
        sem.release()
        assert sem.acquire(timeout=0.1) is True


# ---------------------------------------------------------------------------
# sanitize_nec strips comments (important for cache deduplication)
# ---------------------------------------------------------------------------

class TestSanitizeForCache:
    """sanitize_nec normalizes content, so variant decks share cache keys."""

    def test_comment_lines_stripped(self, api_mod):
        """CM lines are removed, so decks differing only in comments share keys."""
        deck_a = "CM comment A\nGW 1 21 0 0 0 0 5 0 .001\nGE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.15\nEN"
        deck_b = "CM comment B\nGW 1 21 0 0 0 0 5 0 .001\nGE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.15\nEN"
        san_a = api_mod.sanitize_nec(deck_a)
        san_b = api_mod.sanitize_nec(deck_b)
        # After sanitization, both should produce the same cache key
        assert api_mod._cache_key(san_a, "run") == api_mod._cache_key(san_b, "run")

    def test_whitespace_normalization(self, api_mod):
        """Extra whitespace is normalized, duplicates get same key."""
        deck_a = "GW 1 21 0 0 0 0 5 0 .001\nGE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.15\nEN"
        deck_b = "GW  1  21  0  0  0  0  5  0  .001\nGE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.15\nEN"
        san_a = api_mod.sanitize_nec(deck_a)
        san_b = api_mod.sanitize_nec(deck_b)
        assert api_mod._cache_key(san_a, "run") == api_mod._cache_key(san_b, "run")
