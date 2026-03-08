"""Per-type NEC generation goals — simulation-verifiable constraints.

Each antenna type has characteristic electromagnetic behaviour that a nec2c
simulation can confirm: directivity (or lack of it), gain range, front-to-back
ratio, feedpoint impedance, bandwidth, and compactness relative to wavelength.

These goals serve as the oracle's acceptance criteria in the OODA refinement
loop.  After the classifier confirms the antenna *type*, the simulator confirms
the antenna *behaves* like that type.

Usage::

    from antenna_classifier.nec_goals import goals_for_type, evaluate_goals

    goals = goals_for_type("moxon")
    verdict = evaluate_goals(goals, sim_result, freq_mhz=14.175)
    # verdict.passed, verdict.feedback, verdict.score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Goal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AntennaGoals:
    """Simulation-verifiable goals for one antenna type."""

    antenna_type: str

    # --- Directivity ---
    directional: bool               # expects meaningful F/B ratio?
    min_gain_dbi: float             # minimum acceptable gain
    max_gain_dbi: float             # sanity ceiling (flags runaway models)
    min_fb_db: float | None = None  # minimum front-to-back (None = omni)

    # --- Impedance / SWR ---
    max_swr: float = 3.0            # at design frequency
    impedance_r_range: tuple[float, float] = (10.0, 600.0)  # feedpoint R (Ω)

    # --- Bandwidth ---
    min_bw_pct: float | None = None  # 2:1 SWR bandwidth as % of centre freq

    # --- Compactness (boom / turning radius vs λ) ---
    # max_boom_wl: max boom length in wavelengths (None = no constraint)
    # max_radius_wl: max turning radius in wavelengths (None = no constraint)
    max_boom_wl: float | None = None
    max_radius_wl: float | None = None

    # --- Polarisation ---
    polarisation: str = "any"  # horizontal, vertical, circular, any

    # --- Description of design tradeoffs ---
    tradeoffs: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "antenna_type": self.antenna_type,
            "directional": self.directional,
            "min_gain_dbi": self.min_gain_dbi,
            "max_gain_dbi": self.max_gain_dbi,
            "min_fb_db": self.min_fb_db,
            "max_swr": self.max_swr,
            "impedance_r_range": list(self.impedance_r_range),
            "min_bw_pct": self.min_bw_pct,
            "max_boom_wl": self.max_boom_wl,
            "max_radius_wl": self.max_radius_wl,
            "polarisation": self.polarisation,
            "tradeoffs": self.tradeoffs,
        }


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class GoalVerdict:
    """Result of evaluating simulation output against antenna goals."""

    passed: bool = True
    score: float = 1.0          # 0.0 – 1.0 (fraction of checks passed)
    checks_total: int = 0
    checks_passed: int = 0
    feedback: list[str] = field(default_factory=list)  # human-readable issues
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks_total": self.checks_total,
            "checks_passed": self.checks_passed,
            "feedback": self.feedback,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Per-type goal definitions
# ---------------------------------------------------------------------------

# Speed of light for λ calculations
_C_MPS = 299_792_458.0


_GOALS: dict[str, AntennaGoals] = {}


def _g(goals: AntennaGoals) -> AntennaGoals:
    """Register goals for a type."""
    _GOALS[goals.antenna_type] = goals
    return goals


# ── Simple resonant antennas (omnidirectional) ────────────────────────

_g(AntennaGoals(
    antenna_type="dipole",
    directional=False,
    min_gain_dbi=1.5,      # ~2.15 dBi theoretical
    max_gain_dbi=4.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(50.0, 100.0),  # ~73 Ω typical
    min_bw_pct=3.0,        # ~3-5% for half-wave dipole
    max_boom_wl=None,
    max_radius_wl=0.3,     # half-wave span ≈ 0.25λ per side
    polarisation="horizontal",
    tradeoffs="Simplest resonant antenna. No gain tradeoff — serves as "
              "the reference (0 dBd). Height above ground affects pattern.",
))

_g(AntennaGoals(
    antenna_type="inverted_v",
    directional=False,
    min_gain_dbi=1.0,      # slightly less than flat dipole due to droop
    max_gain_dbi=3.5,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(40.0, 90.0),  # lower R due to droop angle
    min_bw_pct=3.0,
    max_boom_wl=None,
    max_radius_wl=0.3,
    polarisation="any",     # mixed H/V due to droop
    tradeoffs="Trades ~0.5 dB gain vs flat dipole for single-support mounting. "
              "Droop angle affects impedance: steeper = lower R, wider BW.",
))

_g(AntennaGoals(
    antenna_type="vertical",
    directional=False,
    min_gain_dbi=-2.0,     # quarter-wave: ~0 dBi with good ground
    max_gain_dbi=3.0,
    min_fb_db=None,
    max_swr=3.0,
    impedance_r_range=(20.0, 80.0),  # ~36 Ω for λ/4 over perfect ground
    min_bw_pct=4.0,
    max_boom_wl=None,
    max_radius_wl=None,    # radials spread but not a turning radius issue
    polarisation="vertical",
    tradeoffs="Low-angle radiation ideal for DX. Gain depends heavily on "
              "ground system quality — more/longer radials = better efficiency. "
              "Compact height (λ/4) but needs ground radials.",
))

_g(AntennaGoals(
    antenna_type="end_fed",
    directional=False,
    min_gain_dbi=0.0,
    max_gain_dbi=4.0,
    min_fb_db=None,
    max_swr=3.0,
    impedance_r_range=(20.0, 5000.0),  # EFHW has very high Z at end
    min_bw_pct=2.0,
    max_boom_wl=None,
    max_radius_wl=0.6,
    polarisation="any",
    tradeoffs="Single feed point, no balun needed, but very high impedance "
              "at the end requires matching network. Multiband operation "
              "possible on harmonics.",
))

_g(AntennaGoals(
    antenna_type="j_pole",
    directional=False,
    min_gain_dbi=0.0,
    max_gain_dbi=4.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(30.0, 80.0),   # matching stub tunes to ~50 Ω
    min_bw_pct=4.0,
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="vertical",
    tradeoffs="Self-matching via λ/4 stub eliminates separate matching network. "
              "Slightly taller than a simple vertical (≈ 3/4λ total).",
))

_g(AntennaGoals(
    antenna_type="collinear",
    directional=False,
    min_gain_dbi=3.0,      # stacking gain over single dipole
    max_gain_dbi=9.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(30.0, 100.0),
    min_bw_pct=2.0,
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="any",
    tradeoffs="Higher gain through stacking, but narrower bandwidth and "
              "more complex phasing sections. Each added section gives ~3 dB "
              "but increases height/length.",
))

_g(AntennaGoals(
    antenna_type="discone",
    directional=False,
    min_gain_dbi=-2.0,     # broadband = lower peak gain
    max_gain_dbi=4.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(30.0, 80.0),
    min_bw_pct=50.0,       # extremely broadband (up to 10:1 freq ratio)
    max_boom_wl=None,
    max_radius_wl=0.35,
    polarisation="vertical",
    tradeoffs="Extreme bandwidth (10:1) at the cost of modest gain. "
              "No tuning needed across the entire operating range. "
              "Size is set by the lowest frequency.",
))

# ── Directional beams ─────────────────────────────────────────────────

_g(AntennaGoals(
    antenna_type="yagi",
    directional=True,
    min_gain_dbi=5.0,      # 3-el: ~7 dBi, bigger = more
    max_gain_dbi=20.0,
    min_fb_db=10.0,        # 3-el: ~12 dB typical
    max_swr=2.0,
    impedance_r_range=(15.0, 60.0),  # often ~25Ω, needs matching
    min_bw_pct=2.0,
    max_boom_wl=2.0,       # long-boom Yagis can reach ~2λ
    max_radius_wl=0.55,    # element half-span + boom/2
    polarisation="horizontal",
    tradeoffs="Gold standard gain-per-element. Longer boom = more gain but "
              "narrower bandwidth and larger turning radius. 3-element is "
              "the sweet spot (7 dBi, 12 dB F/B, manageable size). "
              "Impedance often requires a gamma/beta match for 50 Ω.",
))

_g(AntennaGoals(
    antenna_type="moxon",
    directional=True,
    min_gain_dbi=4.5,      # ~5-6 dBi (slightly less than 2-el Yagi)
    max_gain_dbi=8.0,
    min_fb_db=20.0,        # excellent F/B is the signature feature
    max_swr=2.0,
    impedance_r_range=(40.0, 70.0),  # naturally ~50 Ω — key advantage
    min_bw_pct=2.0,
    max_boom_wl=0.15,      # very compact boom (~0.13λ)
    max_radius_wl=0.35,    # small turning radius
    polarisation="horizontal",
    tradeoffs="Compact: ~70% the turning radius of a 2-el Yagi. Excellent "
              "F/B ratio (>25 dB) and natural 50 Ω feed. Trades ~1 dB gain "
              "vs 2-el Yagi for dramatic size reduction and better F/B. "
              "Narrow bandwidth — single-band use.",
))

_g(AntennaGoals(
    antenna_type="hexbeam",
    directional=True,
    min_gain_dbi=3.5,      # ~4-5 dBi per band
    max_gain_dbi=8.0,
    min_fb_db=8.0,         # ~10-15 dB typical
    max_swr=2.5,
    impedance_r_range=(30.0, 80.0),
    min_bw_pct=2.0,
    max_boom_wl=None,       # no boom — hex frame
    max_radius_wl=0.26,    # very compact: hex radius ≈ 0.236λ at lowest band
    polarisation="horizontal",
    tradeoffs="Extremely compact multiband beam — ~60% turning radius of "
              "equivalent Yagi. Trades 2-3 dB gain for covering 5-6 bands "
              "on one lightweight frame. Easy to rotate. Lower F/B than "
              "Moxon but multiband.",
))

_g(AntennaGoals(
    antenna_type="quad",
    directional=True,
    min_gain_dbi=6.0,      # 2-el: ~7 dBi, comparable to 3-el Yagi
    max_gain_dbi=16.0,
    min_fb_db=12.0,        # ~15-20 dB for 2-element
    max_swr=2.0,
    impedance_r_range=(80.0, 150.0),  # ~120 Ω typical, needs matching
    min_bw_pct=2.0,
    max_boom_wl=0.25,      # shorter boom than Yagi for same gain
    max_radius_wl=0.40,    # loop elements are larger than dipoles
    polarisation="horizontal",
    tradeoffs="~1 dB more gain than equivalent Yagi with shorter boom, "
              "but much larger element area (full-wave loops). Higher wind "
              "load. High impedance (~120 Ω) needs a matching section. "
              "Excellent performance but mechanically challenging.",
))

_g(AntennaGoals(
    antenna_type="quagi",
    directional=True,
    min_gain_dbi=6.0,
    max_gain_dbi=16.0,
    min_fb_db=12.0,
    max_swr=2.0,
    impedance_r_range=(30.0, 150.0),
    min_bw_pct=2.0,
    max_boom_wl=1.5,
    max_radius_wl=0.40,
    polarisation="horizontal",
    tradeoffs="Hybrid: quad loops as driven/reflector with Yagi directors. "
              "Gains the quad's low-angle advantage with simpler directors. "
              "Good compromise between quad gain and Yagi simplicity.",
))

# ── Log-periodic (broadband directional) ──────────────────────────────

_g(AntennaGoals(
    antenna_type="lpda",
    directional=True,
    min_gain_dbi=5.0,      # ~6-8 dBi typical
    max_gain_dbi=12.0,
    min_fb_db=10.0,        # ~14-18 dB
    max_swr=2.5,
    impedance_r_range=(30.0, 100.0),
    min_bw_pct=30.0,       # very broadband — that's the whole point
    max_boom_wl=2.5,       # long for the frequency range covered
    max_radius_wl=0.55,
    polarisation="horizontal",
    tradeoffs="Constant gain and pattern across a wide frequency range "
              "(often 2:1 or more). Trades peak gain (~2 dB less than "
              "equivalent mono-band Yagi) for extreme bandwidth. Requires "
              "phase-reversal feed line between elements. Boom length "
              "determined by lowest operating frequency.",
))

# ── Phased arrays ─────────────────────────────────────────────────────

_g(AntennaGoals(
    antenna_type="phased_array",
    directional=True,
    min_gain_dbi=2.0,      # 2-el: ~3-4 dBi improvement
    max_gain_dbi=12.0,
    min_fb_db=8.0,         # cardioid: ~10-15 dB
    max_swr=3.0,
    impedance_r_range=(15.0, 100.0),
    min_bw_pct=3.0,
    max_boom_wl=None,       # spacing, not boom
    max_radius_wl=None,
    polarisation="vertical",
    tradeoffs="Steerable directivity via phase control — no mechanical "
              "rotation needed. Gain proportional to number of elements. "
              "Requires precise phase/amplitude feed network. Mutual "
              "coupling between elements complicates impedance matching.",
))

# ── Full-wave loops ───────────────────────────────────────────────────

_g(AntennaGoals(
    antenna_type="loop",
    directional=True,       # full-wave loops have modest directivity
    min_gain_dbi=2.5,      # ~3.3 dBi for circular loop
    max_gain_dbi=6.0,
    min_fb_db=5.0,         # modest F/B
    max_swr=2.5,
    impedance_r_range=(80.0, 200.0),  # ~120 Ω typical
    min_bw_pct=2.0,
    max_boom_wl=None,
    max_radius_wl=0.20,    # circumference ≈ 1λ → radius ≈ 0.16λ
    polarisation="any",     # depends on feed point location
    tradeoffs="Full-wave loop: ~1 dB more gain than dipole, lower noise "
              "pickup. High impedance (~120 Ω). Orientation and feed point "
              "determine polarisation. Relatively large physical size but "
              "quiet reception.",
))

_g(AntennaGoals(
    antenna_type="delta_loop",
    directional=True,
    min_gain_dbi=2.5,
    max_gain_dbi=6.0,
    min_fb_db=5.0,
    max_swr=2.5,
    impedance_r_range=(80.0, 200.0),
    min_bw_pct=2.0,
    max_boom_wl=None,
    max_radius_wl=0.20,
    polarisation="any",
    tradeoffs="Triangular full-wave loop — easier to support than square "
              "loop (three attachment points). Same gain/impedance tradeoffs "
              "as circular loop. Bottom-fed = horizontal pol, corner-fed = "
              "vertical pol.",
))

_g(AntennaGoals(
    antenna_type="magnetic_loop",
    directional=True,       # figure-8 pattern with deep nulls
    min_gain_dbi=-15.0,    # very low efficiency for small loops
    max_gain_dbi=0.0,
    min_fb_db=None,         # bi-directional null, not F/B
    max_swr=2.5,
    impedance_r_range=(0.01, 5.0),  # extremely low R
    min_bw_pct=0.1,        # very narrow — 0.1-0.5% typical (high Q)
    max_boom_wl=None,
    max_radius_wl=0.05,    # circumference << λ, that's the definition
    polarisation="any",
    tradeoffs="Extreme compactness (circumference λ/10 to λ/4) at the cost "
              "of very low radiation resistance, narrow bandwidth (must retune "
              "every ~25 kHz on HF), and high circulating currents requiring "
              "thick conductor + high-voltage tuning capacitor. Despite low "
              "efficiency, useful in space-restricted situations.",
))

# ── Wire arrays ───────────────────────────────────────────────────────

_g(AntennaGoals(
    antenna_type="wire_array",
    directional=True,
    min_gain_dbi=4.0,
    max_gain_dbi=14.0,
    min_fb_db=5.0,
    max_swr=3.0,
    impedance_r_range=(20.0, 600.0),
    min_bw_pct=2.0,
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="any",
    tradeoffs="High gain from stacking/phasing multiple half-wave sections. "
              "Requires large support structure but uses only wire. "
              "Sterba curtain, lazy-H, bobtail — all trade mechanical "
              "complexity for gain with cheap materials.",
))

_g(AntennaGoals(
    antenna_type="bobtail_curtain",
    directional=True,       # bidirectional broadside
    min_gain_dbi=3.0,
    max_gain_dbi=7.0,
    min_fb_db=None,         # bidirectional, not unidirectional
    max_swr=3.0,
    impedance_r_range=(20.0, 300.0),
    min_bw_pct=3.0,
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="vertical",
    tradeoffs="Low-angle bidirectional vertical array. Three verticals + "
              "two horizontal wires give ~4 dBd broadside gain. Needs tall "
              "supports but no rotator. Excellent for fixed-path DX.",
))

_g(AntennaGoals(
    antenna_type="rhombic",
    directional=True,
    min_gain_dbi=8.0,      # very high gain when large
    max_gain_dbi=20.0,
    min_fb_db=15.0,
    max_swr=2.0,
    impedance_r_range=(400.0, 800.0),  # ~600 Ω typical
    min_bw_pct=40.0,       # very broadband
    max_boom_wl=None,
    max_radius_wl=None,    # huge: legs are many wavelengths
    polarisation="horizontal",
    tradeoffs="Highest gain of any wire antenna (up to 20 dBi) with "
              "enormous bandwidth. Requires vast real estate (legs 2-10λ "
              "long). Termination resistor wastes ~50% power for pattern "
              "cleanliness. Historical favourite for point-to-point HF.",
))

_g(AntennaGoals(
    antenna_type="v_beam",
    directional=True,
    min_gain_dbi=4.0,
    max_gain_dbi=15.0,
    min_fb_db=5.0,
    max_swr=3.0,
    impedance_r_range=(200.0, 800.0),
    min_bw_pct=20.0,
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="horizontal",
    tradeoffs="Open-ended rhombic — no termination resistor so all power "
              "is radiated. Less clean pattern (higher sidelobes) than "
              "terminated rhombic but higher efficiency. Still needs very "
              "long legs for useful gain.",
))

_g(AntennaGoals(
    antenna_type="beverage",
    directional=True,
    min_gain_dbi=-10.0,    # receive-only, very low gain
    max_gain_dbi=0.0,
    min_fb_db=10.0,        # good directivity is the point
    max_swr=3.0,
    impedance_r_range=(200.0, 600.0),
    min_bw_pct=50.0,       # very broadband
    max_boom_wl=None,
    max_radius_wl=None,
    polarisation="horizontal",
    tradeoffs="Receive-only directional antenna. Very low gain but excellent "
              "directivity and noise rejection. Requires 1-4λ length at very "
              "low height. Terminated with matching resistor.",
))

# ── Specialty types ───────────────────────────────────────────────────

_g(AntennaGoals(
    antenna_type="helix",
    directional=True,
    min_gain_dbi=8.0,      # axial-mode helix is high gain
    max_gain_dbi=20.0,
    min_fb_db=10.0,
    max_swr=2.5,
    impedance_r_range=(80.0, 200.0),  # ~140 Ω typical for axial mode
    min_bw_pct=20.0,       # broadband
    max_boom_wl=None,
    max_radius_wl=0.20,
    polarisation="circular",
    tradeoffs="Unique circular polarisation with high gain. Gain increases "
              "with number of turns (~3 dB per doubling). Broadband (~1.7:1). "
              "Long axially but narrow — good for VHF/UHF/microwave.",
))

_g(AntennaGoals(
    antenna_type="patch",
    directional=True,
    min_gain_dbi=5.0,      # ~6-9 dBi
    max_gain_dbi=12.0,
    min_fb_db=10.0,
    max_swr=2.5,
    impedance_r_range=(20.0, 300.0),
    min_bw_pct=2.0,        # narrow BW is a known limitation
    max_boom_wl=None,
    max_radius_wl=0.55,
    polarisation="any",
    tradeoffs="Flat, low-profile, hemispherical pattern. Narrow bandwidth "
              "(2-5%) unless stacked or using thick substrate. In wire-grid "
              "NEC model, ground plane is implicit via GN card.",
))

_g(AntennaGoals(
    antenna_type="fractal",
    directional=False,
    min_gain_dbi=0.0,
    max_gain_dbi=5.0,
    min_fb_db=None,
    max_swr=3.0,
    impedance_r_range=(20.0, 300.0),
    min_bw_pct=5.0,        # multiband / wideband is the goal
    max_boom_wl=None,
    max_radius_wl=0.25,    # compact — shorter span than dipole
    polarisation="any",
    tradeoffs="Compact multiband/wideband operation via self-similar geometry. "
              "Trades peak gain for size reduction. Higher-iteration fractals "
              "add bands but increase complexity and loss.",
))

_g(AntennaGoals(
    antenna_type="turnstile",
    directional=False,      # omnidirectional in azimuth
    min_gain_dbi=1.0,
    max_gain_dbi=5.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(20.0, 80.0),
    min_bw_pct=5.0,
    max_boom_wl=None,
    max_radius_wl=0.30,
    polarisation="circular",
    tradeoffs="Crossed dipoles fed in quadrature for circular polarisation "
              "with omnidirectional azimuth pattern. Useful for satellite "
              "work (any orientation received). ~3 dB loss vs linearly "
              "polarised dipole from power split.",
))

_g(AntennaGoals(
    antenna_type="batwing",
    directional=False,      # omnidirectional (FM broadcast)
    min_gain_dbi=1.0,
    max_gain_dbi=8.0,
    min_fb_db=None,
    max_swr=2.5,
    impedance_r_range=(20.0, 200.0),
    min_bw_pct=15.0,       # broadband for FM
    max_boom_wl=None,
    max_radius_wl=0.35,
    polarisation="horizontal",
    tradeoffs="Broadband omnidirectional horizontal polarisation — the "
              "standard FM broadcast antenna. Gain from stacking multiple "
              "bays. Moderate wind load.",
))

_g(AntennaGoals(
    antenna_type="zigzag",
    directional=False,
    min_gain_dbi=0.0,
    max_gain_dbi=4.0,
    min_fb_db=None,
    max_swr=3.0,
    impedance_r_range=(20.0, 200.0),
    min_bw_pct=3.0,
    max_boom_wl=None,
    max_radius_wl=0.25,   # compact
    polarisation="any",
    tradeoffs="Compact: zigzag bends shorten the physical span while "
              "maintaining electrical length. Trades slight efficiency "
              "loss and pattern distortion for smaller footprint.",
))


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

# Default goals for types without specific definitions
_DEFAULT_GOALS = AntennaGoals(
    antenna_type="unknown",
    directional=False,
    min_gain_dbi=-5.0,
    max_gain_dbi=25.0,
    max_swr=5.0,
    impedance_r_range=(1.0, 5000.0),
    tradeoffs="No specific type goals defined — using permissive defaults.",
)


def goals_for_type(antenna_type: str) -> AntennaGoals:
    """Return the simulation goals for *antenna_type* (case-insensitive)."""
    return _GOALS.get(antenna_type.lower().strip(), _DEFAULT_GOALS)


def all_goals() -> dict[str, AntennaGoals]:
    """Return a copy of all registered per-type goals."""
    return dict(_GOALS)


# ---------------------------------------------------------------------------
# Geometry-based compactness measurement
# ---------------------------------------------------------------------------

def _measure_geometry(nec: str, freq_mhz: float) -> dict[str, float]:
    """Measure bounding box and boom length from NEC text, normalised to λ.

    Returns dict with keys: span_x_wl, span_y_wl, span_z_wl,
    max_radius_wl, boom_length_wl, wavelength_m.
    """
    wavelength = _C_MPS / (freq_mhz * 1e6)
    xs, ys, zs = [], [], []

    for line in nec.splitlines():
        parts = line.split()
        if not parts or parts[0].upper() != "GW":
            continue
        try:
            # GW tag segs x1 y1 z1 x2 y2 z2 radius
            vals = [float(v.replace(",", "")) for v in parts[1:]]
            if len(vals) >= 8:
                xs.extend([vals[2], vals[5]])
                ys.extend([vals[3], vals[6]])
                zs.extend([vals[4], vals[7]])
        except (ValueError, IndexError):
            continue

    if not xs:
        return {"wavelength_m": wavelength}

    dx = (max(xs) - min(xs))
    dy = (max(ys) - min(ys))
    dz = (max(zs) - min(zs))
    # Turning radius = half the maximum horizontal span
    horiz_span = math.sqrt(dx ** 2 + dy ** 2)

    return {
        "span_x_wl": dx / wavelength,
        "span_y_wl": dy / wavelength,
        "span_z_wl": dz / wavelength,
        "max_radius_wl": horiz_span / (2.0 * wavelength),
        "boom_length_wl": dx / wavelength,  # convention: boom along X
        "wavelength_m": wavelength,
    }


# ---------------------------------------------------------------------------
# Evaluate simulation results against goals
# ---------------------------------------------------------------------------

def evaluate_goals(
    goals: AntennaGoals,
    sim_result: Any,          # SimulationResult from simulator.py
    *,
    nec_text: str = "",
    freq_mhz: float = 0.0,
) -> GoalVerdict:
    """Compare simulation output against the antenna-type goals.

    *sim_result* should be a ``SimulationResult`` (or anything with
    ``.pattern``, ``.swr``, ``.impedance`` attributes).

    Returns a ``GoalVerdict`` with pass/fail, score, and feedback.
    """
    v = GoalVerdict()
    checks: list[tuple[str, bool, str]] = []  # (name, passed, message)

    # ── Gain checks ───────────────────────────────────────────────
    if sim_result.pattern and sim_result.pattern.max_gain is not None:
        gain = sim_result.pattern.max_gain
        v.details["max_gain_dbi"] = gain

        ok = gain >= goals.min_gain_dbi
        checks.append((
            "min_gain",
            ok,
            f"Gain {gain:.1f} dBi {'≥' if ok else '<'} "
            f"minimum {goals.min_gain_dbi:.1f} dBi for {goals.antenna_type}",
        ))

        ok2 = gain <= goals.max_gain_dbi
        checks.append((
            "max_gain",
            ok2,
            f"Gain {gain:.1f} dBi {'≤' if ok2 else '>'} "
            f"ceiling {goals.max_gain_dbi:.1f} dBi (unrealistic if exceeded)",
        ))

    # ── Front-to-back ─────────────────────────────────────────────
    if goals.directional and goals.min_fb_db is not None:
        fb = (sim_result.pattern.front_to_back
              if sim_result.pattern else None)
        if fb is not None:
            v.details["front_to_back_db"] = fb
            ok = fb >= goals.min_fb_db
            checks.append((
                "front_to_back",
                ok,
                f"F/B {fb:.1f} dB {'≥' if ok else '<'} "
                f"minimum {goals.min_fb_db:.1f} dB for {goals.antenna_type}",
            ))
    elif not goals.directional and sim_result.pattern:
        # Omni antenna should NOT have large F/B
        fb = sim_result.pattern.front_to_back
        if fb is not None and fb > 10.0:
            v.details["front_to_back_db"] = fb
            checks.append((
                "omni_pattern",
                False,
                f"Omnidirectional {goals.antenna_type} has unexpected "
                f"F/B ratio of {fb:.1f} dB (expected < 10 dB)",
            ))

    # ── SWR at design frequency ───────────────────────────────────
    if sim_result.swr and sim_result.swr.min_swr is not None:
        swr = sim_result.swr.min_swr
        v.details["min_swr"] = swr
        ok = swr <= goals.max_swr
        checks.append((
            "swr",
            ok,
            f"SWR {swr:.2f} {'≤' if ok else '>'} "
            f"maximum {goals.max_swr:.1f} for {goals.antenna_type}",
        ))

    # ── Bandwidth ─────────────────────────────────────────────────
    if (goals.min_bw_pct is not None
            and sim_result.swr
            and sim_result.swr.bandwidth_2to1 is not None
            and sim_result.swr.resonant_freq is not None):
        bw_mhz = sim_result.swr.bandwidth_2to1
        center = sim_result.swr.resonant_freq
        bw_pct = 100.0 * bw_mhz / center if center else 0.0
        v.details["bandwidth_pct"] = bw_pct
        ok = bw_pct >= goals.min_bw_pct
        checks.append((
            "bandwidth",
            ok,
            f"Bandwidth {bw_pct:.1f}% {'≥' if ok else '<'} "
            f"minimum {goals.min_bw_pct:.1f}% for {goals.antenna_type}",
        ))

    # ── Feedpoint impedance ───────────────────────────────────────
    if sim_result.impedance and sim_result.impedance.r:
        # Use the R value closest to the design frequency
        if freq_mhz and sim_result.impedance.freq_mhz:
            idx = min(
                range(len(sim_result.impedance.freq_mhz)),
                key=lambda i: abs(sim_result.impedance.freq_mhz[i] - freq_mhz),
            )
            r_val = sim_result.impedance.r[idx]
        else:
            r_val = sim_result.impedance.r[len(sim_result.impedance.r) // 2]

        v.details["feedpoint_r"] = r_val
        lo, hi = goals.impedance_r_range
        ok = lo <= r_val <= hi
        checks.append((
            "impedance_r",
            ok,
            f"Feedpoint R = {r_val:.1f} Ω "
            f"{'within' if ok else 'OUTSIDE'} expected "
            f"range [{lo:.0f}–{hi:.0f}] Ω for {goals.antenna_type}",
        ))

    # ── Compactness (geometry-based) ──────────────────────────────
    if nec_text and freq_mhz:
        geo = _measure_geometry(nec_text, freq_mhz)
        v.details["geometry"] = geo

        if goals.max_boom_wl is not None and "boom_length_wl" in geo:
            boom = geo["boom_length_wl"]
            ok = boom <= goals.max_boom_wl
            checks.append((
                "boom_length",
                ok,
                f"Boom {boom:.2f}λ {'≤' if ok else '>'} "
                f"maximum {goals.max_boom_wl:.2f}λ for {goals.antenna_type}",
            ))

        if goals.max_radius_wl is not None and "max_radius_wl" in geo:
            radius = geo["max_radius_wl"]
            ok = radius <= goals.max_radius_wl
            checks.append((
                "turning_radius",
                ok,
                f"Turning radius {radius:.2f}λ {'≤' if ok else '>'} "
                f"maximum {goals.max_radius_wl:.2f}λ for {goals.antenna_type}",
            ))

    # ── Tally ─────────────────────────────────────────────────────
    v.checks_total = len(checks)
    v.checks_passed = sum(1 for _, ok, _ in checks if ok)
    v.score = v.checks_passed / v.checks_total if v.checks_total else 1.0
    v.passed = all(ok for _, ok, _ in checks)
    v.feedback = [msg for _, ok, msg in checks if not ok]
    v.details["checks"] = [
        {"name": name, "passed": ok, "message": msg}
        for name, ok, msg in checks
    ]

    return v
