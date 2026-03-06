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

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"filename": self.filename, "ok": self.ok}
        if self.error:
            d["error"] = self.error
        if self.swr:
            d["swr_sweep"] = {
                "freq_mhz": self.swr.freq_mhz,
                "swr": self.swr.swr,
                "z0": self.swr.z0,
                "min_swr": self.swr.min_swr,
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
                "max_gain_dbi": self.pattern.max_gain,
                "front_to_back_db": self.pattern.front_to_back,
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
    nec_text = nec_path.read_text()
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


def simulate_pattern(
    nec_path: Path | str,
    *,
    base_url: str = DEFAULT_URL,
    z0: float = 50.0,
    timeout: int = 180,
) -> SimulationResult:
    """Run far-field pattern simulation via /pattern endpoint.

    The /pattern endpoint also returns impedance/SWR when available.
    """
    nec_path = Path(nec_path)
    nec_text = nec_path.read_text()
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
