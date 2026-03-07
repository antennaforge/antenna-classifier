"""NEC simulation runner — sends NEC files to the nec-solver Docker API.

The nec-solver container (see docker-compose.yml) exposes two endpoints:
  POST /run      — impedance + SWR sweep
  POST /pattern  — far-field radiation pattern + impedance

This module provides a thin client that reads .nec files, posts them to
the solver, and returns structured results.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_URL = "http://localhost:8787"


@dataclass
class SWRSweep:
    """SWR vs frequency."""
    freq_mhz: list[float] = field(default_factory=list)
    swr: list[float] = field(default_factory=list)
    z0: float = 50.0

    @property
    def min_swr(self) -> float:
        finite = [v for v in self.swr if math.isfinite(v)]
        return min(finite) if finite else float("inf")

    @property
    def resonant_freq(self) -> float | None:
        """Frequency with lowest SWR."""
        if not self.swr:
            return None
        finite = [(s, f) for s, f in zip(self.swr, self.freq_mhz) if math.isfinite(s)]
        if not finite:
            return None
        return min(finite)[1]

    @property
    def bandwidth_2to1(self) -> float | None:
        """2:1 SWR bandwidth in MHz (None if never below 2)."""
        in_band = [f for s, f in zip(self.swr, self.freq_mhz)
                    if math.isfinite(s) and s <= 2.0]
        if len(in_band) < 2:
            return None
        return max(in_band) - min(in_band)


@dataclass
class ImpedanceSweep:
    """Impedance (R + jX) vs frequency."""
    freq_mhz: list[float] = field(default_factory=list)
    r: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    z0: float = 50.0


@dataclass
class RadiationPattern:
    """Far-field gain pattern."""
    theta: list[float] = field(default_factory=list)
    phi: list[float] = field(default_factory=list)
    gain_db: list[float] = field(default_factory=list)

    @property
    def max_gain(self) -> float:
        return max(self.gain_db) if self.gain_db else float("-inf")

    @property
    def front_to_back(self) -> float | None:
        """Estimate F/B ratio (dB) for phi=0 vs phi=180 at peak theta."""
        if not self.gain_db:
            return None
        # Find peak direction
        peak_idx = self.gain_db.index(max(self.gain_db))
        peak_theta = self.theta[peak_idx]
        peak_phi = self.phi[peak_idx]
        # Find gain at 180 degrees opposite
        target_phi = (peak_phi + 180) % 360
        back_gains = []
        for t, p, g in zip(self.theta, self.phi, self.gain_db):
            if abs(t - peak_theta) < 1.0 and abs(p - target_phi) < 1.0:
                back_gains.append(g)
        if not back_gains:
            return None
        return max(self.gain_db) - max(back_gains)


@dataclass
class SimulationResult:
    """Complete simulation output for one NEC file."""
    filename: str
    ok: bool
    error: str | None = None
    swr: SWRSweep | None = None
    impedance: ImpedanceSweep | None = None
    pattern: RadiationPattern | None = None
    raw: dict[str, Any] | None = None

    @staticmethod
    def _json_safe(v: Any) -> Any:
        """Replace inf/nan with None for JSON serialization."""
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    def to_dict(self) -> dict:
        _s = self._json_safe
        d: dict[str, Any] = {"filename": self.filename, "ok": self.ok}
        if self.error:
            d["error"] = self.error
            # Pass through stderr from solver for nec_failed diagnostics
            if self.raw and "stderr" in self.raw:
                d["stderr"] = self.raw["stderr"]
        if self.swr:
            d["swr_sweep"] = {
                "freq_mhz": self.swr.freq_mhz,
                "swr": [_s(v) for v in self.swr.swr],
                "z0": self.swr.z0,
                "min_swr": _s(self.swr.min_swr),
                "resonant_freq_mhz": self.swr.resonant_freq,
                "bandwidth_2to1_mhz": self.swr.bandwidth_2to1,
            }
        if self.impedance:
            d["impedance_sweep"] = {
                "freq_mhz": self.impedance.freq_mhz,
                "r": self.impedance.r,
                "x": self.impedance.x,
                "z0": self.impedance.z0,
            }
        if self.pattern:
            d["radiation_pattern"] = {
                "theta": self.pattern.theta,
                "phi": self.pattern.phi,
                "gain_db": self.pattern.gain_db,
                "max_gain_dbi": _s(self.pattern.max_gain),
                "front_to_back_db": _s(self.pattern.front_to_back),
            }
        return d


def _post_json(url: str, data: dict, timeout: int = 120) -> dict:
    """POST JSON to url and return parsed response."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def simulate_impedance(
    nec_path: Path | str,
    *,
    base_url: str = DEFAULT_URL,
    z0: float = 50.0,
    timeout: int = 120,
) -> SimulationResult:
    """Run impedance/SWR simulation via /run endpoint."""
    nec_path = Path(nec_path)
    nec_text = nec_path.read_text(errors="replace")
    try:
        resp = _post_json(
            f"{base_url}/run",
            {"nec_deck": nec_text, "z0": z0},
            timeout=timeout,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return SimulationResult(filename=nec_path.name, ok=False, error=str(e))

    if not resp.get("ok"):
        return SimulationResult(
            filename=nec_path.name, ok=False,
            error=resp.get("error", "unknown"), raw=resp,
        )

    parsed = resp.get("parsed", resp)
    swr_data = parsed.get("swr_sweep", {})
    imp_data = parsed.get("impedance_sweep", {})

    swr = SWRSweep(
        freq_mhz=swr_data.get("freq_mhz", []),
        swr=swr_data.get("swr", []),
        z0=z0,
    ) if swr_data.get("freq_mhz") else None

    impedance = ImpedanceSweep(
        freq_mhz=imp_data.get("freq_mhz", []),
        r=imp_data.get("r", []),
        x=imp_data.get("x", []),
        z0=z0,
    ) if imp_data.get("freq_mhz") else None

    return SimulationResult(
        filename=nec_path.name, ok=True,
        swr=swr, impedance=impedance, raw=resp,
    )


# Standard RP cards for forced pattern types
_RP_CARDS = {
    # Elevation cut at phi=0: theta -90..90 in 1° steps
    "elevation": "RP 0 181 1 1000 -90 0 1 0",
    # Azimuth cut at theta near horizon: phi 0..360 in 1° steps
    "azimuth": "RP 0 1 361 1000 90 0 0 1",
    # Full 3D hemisphere: theta 0..90, phi 0..360 in 2° steps
    "full": "RP 0 46 181 1000 0 0 2 2",
}


def _inject_rp(nec_text: str, rp_card: str) -> str:
    """Replace all RP cards in a NEC deck with the given RP card."""
    lines = nec_text.splitlines()
    out = []
    rp_inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.split()[0].upper() == "RP":
            if not rp_inserted:
                out.append(rp_card)
                rp_inserted = True
            # Skip additional RP cards
            continue
        # If no RP card found yet and we hit EN, insert before EN
        if stripped.upper() == "EN" and not rp_inserted:
            out.append(rp_card)
            rp_inserted = True
        out.append(line)
    return "\n".join(out)


def simulate_pattern(
    nec_path: Path | str,
    *,
    base_url: str = DEFAULT_URL,
    z0: float = 50.0,
    timeout: int = 180,
    force_pattern: str | None = None,
) -> SimulationResult:
    """Run far-field pattern simulation via /pattern endpoint.

    The /pattern endpoint also returns impedance/SWR when available.
    If *force_pattern* is set ("elevation", "azimuth", or "full"), the
    file's RP card(s) are replaced with the requested scan.
    """
    nec_path = Path(nec_path)
    nec_text = nec_path.read_text(errors="replace")
    if force_pattern and force_pattern in _RP_CARDS:
        nec_text = _inject_rp(nec_text, _RP_CARDS[force_pattern])
    try:
        resp = _post_json(
            f"{base_url}/pattern",
            {"nec_text": nec_text},
            timeout=timeout,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return SimulationResult(filename=nec_path.name, ok=False, error=str(e))

    if not resp.get("ok"):
        return SimulationResult(
            filename=nec_path.name, ok=False,
            error=resp.get("error", "unknown"), raw=resp,
        )

    pattern = RadiationPattern(
        theta=resp.get("theta", []),
        phi=resp.get("phi", []),
        gain_db=resp.get("gain", []),
    ) if resp.get("theta") else None

    swr_data = resp.get("swr_sweep", {})
    imp_data = resp.get("impedance_sweep", {})

    swr = SWRSweep(
        freq_mhz=swr_data.get("freq_mhz", []),
        swr=swr_data.get("swr", []),
        z0=z0,
    ) if swr_data.get("freq_mhz") else None

    impedance = ImpedanceSweep(
        freq_mhz=imp_data.get("freq_mhz", []),
        r=imp_data.get("r", []),
        x=imp_data.get("x", []),
        z0=z0,
    ) if imp_data.get("freq_mhz") else None

    return SimulationResult(
        filename=nec_path.name, ok=True,
        swr=swr, impedance=impedance, pattern=pattern, raw=resp,
    )


def simulate(
    nec_path: Path | str,
    *,
    base_url: str = DEFAULT_URL,
    z0: float = 50.0,
    timeout: int = 180,
) -> SimulationResult:
    """Run full simulation — pattern first (includes impedance), fall back to impedance-only.

    Uses /pattern which returns both pattern + impedance data. If the NEC file
    lacks an RP card, falls back to /run for impedance-only results.
    """
    result = simulate_pattern(nec_path, base_url=base_url, z0=z0, timeout=timeout)
    if result.ok:
        return result
    # If pattern failed because no RP card, try impedance-only
    if result.error == "no_pattern_detected":
        return simulate_impedance(nec_path, base_url=base_url, z0=z0, timeout=timeout)
    return result


# ---------------------------------------------------------------------------
# Frequency sweep — generate multi-point SWR / impedance data
# ---------------------------------------------------------------------------

def _build_sweep_deck(
    nec_text: str,
    n_points: int = 21,
    bw_fraction: float = 0.15,
) -> tuple[str, float]:
    """Replace FR card with a frequency sweep, remove RP card for speed.

    Returns (modified_nec_deck, center_freq_mhz).
    """
    lines = nec_text.strip().splitlines()
    center_freq = 0.0
    new_lines: list[str] = []

    for line in lines:
        upper = line.strip().upper()

        if upper.startswith("FR"):
            parts = re.split(r"[,\s]+", line.strip())
            try:
                center_freq = float(parts[5])
            except (ValueError, IndexError):
                new_lines.append(line)
                continue

            f_low = center_freq * (1 - bw_fraction)
            step = (2 * bw_fraction * center_freq) / max(n_points - 1, 1)
            new_lines.append(f"FR 0,{n_points},0,0,{f_low:.6f},{step:.6f}")
            continue

        # Remove RP card — not needed for impedance/SWR sweep
        if upper.startswith("RP"):
            continue

        new_lines.append(line)

    return "\n".join(new_lines), center_freq


def simulate_sweep(
    nec_path: Path | str,
    *,
    base_url: str = DEFAULT_URL,
    z0: float = 50.0,
    n_points: int = 21,
    bw_fraction: float = 0.15,
    timeout: int = 180,
) -> SimulationResult:
    """Run frequency sweep (impedance + SWR across ±bw_fraction of design freq)."""
    nec_path = Path(nec_path)
    nec_text = nec_path.read_text(errors="replace")
    sweep_deck, center_freq = _build_sweep_deck(nec_text, n_points, bw_fraction)

    if center_freq <= 0:
        return SimulationResult(filename=nec_path.name, ok=False, error="no_freq_card")

    try:
        resp = _post_json(
            f"{base_url}/run",
            {"nec_deck": sweep_deck, "z0": z0},
            timeout=timeout,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return SimulationResult(filename=nec_path.name, ok=False, error=str(e))

    if not resp.get("ok"):
        return SimulationResult(
            filename=nec_path.name, ok=False,
            error=resp.get("error", "unknown"), raw=resp,
        )

    parsed = resp.get("parsed", resp)
    swr_data = parsed.get("swr_sweep", {})
    imp_data = parsed.get("impedance_sweep", {})

    swr = SWRSweep(
        freq_mhz=swr_data.get("freq_mhz", []),
        swr=swr_data.get("swr", []),
        z0=z0,
    ) if swr_data.get("freq_mhz") else None

    impedance = ImpedanceSweep(
        freq_mhz=imp_data.get("freq_mhz", []),
        r=imp_data.get("r", []),
        x=imp_data.get("x", []),
        z0=z0,
    ) if imp_data.get("freq_mhz") else None

    return SimulationResult(
        filename=nec_path.name, ok=True,
        swr=swr, impedance=impedance, raw=resp,
    )
