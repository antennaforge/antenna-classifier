"""Buildability scoring for NEC antenna models.

Scores how practical a design is to actually construct — separate axis from
electromagnetic performance.  A Moxon with 25 dB F/B and a 2mm tip-gap
tolerance is NEC-perfect but garage-unbuildable.

Scoring rubric (0–100):

    30% — Sensitivity (perturbation stability)
    20% — Complexity (knob count + junction count)
    15% — Match practicality (feed Z + match complexity)
    15% — Mechanical risk (spans, clearances, deflection)
    10% — Materials realism (loads, Q, conductor)
    10% — Deployment sensitivity (height/ground sensitivity)

Usage::

    from antenna_classifier.nec_buildability import (
        assess_buildability, BuildabilityReport,
    )

    report = assess_buildability(nec_text, antenna_type="moxon", freq_mhz=14.175)
    # report.score, report.grade, report.top_risks, report.critical_dims
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# Speed of light
_C_MPS = 299_792_458.0


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class BuildabilityReport:
    """Complete buildability assessment for one antenna model."""

    score: float = 50.0             # 0–100
    grade: str = "Fair"             # Excellent / Good / Fair / Poor / Unbuildable

    # Sub-scores (each 0–100)
    sensitivity_score: float = 50.0
    complexity_score: float = 50.0
    match_score: float = 50.0
    mechanical_score: float = 50.0
    materials_score: float = 50.0
    deployment_score: float = 50.0

    top_risks: list[str] = field(default_factory=list)      # max 3
    critical_dims: list[str] = field(default_factory=list)   # key dimensions
    tuning_hints: list[str] = field(default_factory=list)    # practical tuning
    bom_hints: list[str] = field(default_factory=list)       # materials suggestions
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "sensitivity_score": round(self.sensitivity_score, 1),
            "complexity_score": round(self.complexity_score, 1),
            "match_score": round(self.match_score, 1),
            "mechanical_score": round(self.mechanical_score, 1),
            "materials_score": round(self.materials_score, 1),
            "deployment_score": round(self.deployment_score, 1),
            "top_risks": self.top_risks,
            "critical_dims": self.critical_dims,
            "tuning_hints": self.tuning_hints,
            "bom_hints": self.bom_hints,
            "details": self.details,
        }


def _grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score >= 30:
        return "Poor"
    return "Unbuildable"


# ---------------------------------------------------------------------------
# NEC geometry extraction helpers
# ---------------------------------------------------------------------------

@dataclass
class _Wire:
    """Wire extracted from a GW card."""
    tag: int
    segments: int
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    radius: float

    @property
    def length(self) -> float:
        return math.sqrt(
            (self.x2 - self.x1) ** 2 +
            (self.y2 - self.y1) ** 2 +
            (self.z2 - self.z1) ** 2
        )

    @property
    def midpoint(self) -> tuple[float, float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
            (self.z1 + self.z2) / 2,
        )


def _parse_wires(nec: str) -> list[_Wire]:
    """Extract wire geometry from NEC text."""
    wires: list[_Wire] = []
    for line in nec.splitlines():
        parts = line.replace(",", " ").split()
        if not parts or parts[0].upper() != "GW":
            continue
        try:
            vals = [float(v) for v in parts[1:]]
            if len(vals) >= 8:
                wires.append(_Wire(
                    tag=int(vals[0]),
                    segments=int(vals[1]),
                    x1=vals[2], y1=vals[3], z1=vals[4],
                    x2=vals[5], y2=vals[6], z2=vals[7],
                    radius=vals[8] if len(vals) >= 9 else 0.001,
                ))
        except (ValueError, IndexError):
            continue
    return wires


def _count_cards(nec: str, card_type: str) -> int:
    count = 0
    for line in nec.splitlines():
        parts = line.replace(",", " ").split()
        if parts and parts[0].upper() == card_type.upper():
            count += 1
    return count


def _min_clearance(wires: list[_Wire]) -> float:
    """Minimum distance between non-connected wire midpoints."""
    if len(wires) < 2:
        return float("inf")
    min_d = float("inf")
    for i, a in enumerate(wires):
        for b in wires[i + 1:]:
            # Skip wires that share endpoints (physically connected)
            if (abs(a.x2 - b.x1) < 1e-6 and abs(a.y2 - b.y1) < 1e-6
                    and abs(a.z2 - b.z1) < 1e-6):
                continue
            if (abs(a.x1 - b.x2) < 1e-6 and abs(a.y1 - b.y2) < 1e-6
                    and abs(a.z1 - b.z2) < 1e-6):
                continue
            # Approximate: distance between midpoints
            mx = a.midpoint[0] - b.midpoint[0]
            my = a.midpoint[1] - b.midpoint[1]
            mz = a.midpoint[2] - b.midpoint[2]
            d = math.sqrt(mx ** 2 + my ** 2 + mz ** 2)
            if d < min_d:
                min_d = d
    return min_d


def _longest_unsupported(wires: list[_Wire]) -> float:
    """Longest single wire span (metres) — sag risk indicator."""
    return max((w.length for w in wires), default=0.0)


# ---------------------------------------------------------------------------
# Per-type critical dimensions and buildability heuristics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypeBuildProfile:
    """Buildability profile for an antenna type."""
    antenna_type: str
    # How many critical "knobs" does this type have?
    typical_knob_count: int
    # Per-type risks and critical dimensions
    critical_dim_names: tuple[str, ...]
    # Known buildability issues
    type_risks: tuple[str, ...]
    # Sensitivity multiplier (1.0 = normal, >1 = more sensitive)
    sensitivity_factor: float = 1.0
    # Mechanical difficulty multiplier
    mechanical_factor: float = 1.0
    # Feed complexity (0 = direct 50Ω, 1 = needs matching)
    feed_complexity: float = 0.0
    # Tuning hints specific to this type
    tuning_tips: tuple[str, ...] = ()
    # BOM hints
    bom_tips: tuple[str, ...] = ()


_PROFILES: dict[str, TypeBuildProfile] = {}


def _p(profile: TypeBuildProfile) -> TypeBuildProfile:
    _PROFILES[profile.antenna_type] = profile
    return profile


# --- Dipole family ---

_p(TypeBuildProfile(
    antenna_type="dipole",
    typical_knob_count=2,  # length, height
    critical_dim_names=("total length", "feed height above ground"),
    type_risks=("wire sag changes resonance",),
    sensitivity_factor=0.5,   # very forgiving
    mechanical_factor=0.3,
    feed_complexity=0.1,      # direct 50Ω-ish feed
    tuning_tips=(
        "Trim element tips symmetrically to lower resonance",
        "VNA: look for R≈73Ω, X≈0 at design frequency",
    ),
    bom_tips=(
        "14 AWG stranded copper wire",
        "Centre insulator with SO-239 connector",
        "1:1 current balun at feedpoint",
    ),
))

_p(TypeBuildProfile(
    antenna_type="inverted_v",
    typical_knob_count=3,  # length, apex height, droop angle
    critical_dim_names=("total length", "apex height", "droop angle"),
    type_risks=("droop angle changes impedance", "wire tension varies with temperature"),
    sensitivity_factor=0.5,
    mechanical_factor=0.3,
    feed_complexity=0.1,
    tuning_tips=(
        "Steeper droop → lower impedance, wider bandwidth",
        "Start with elements 3% long and trim",
    ),
    bom_tips=(
        "14 AWG stranded copper wire",
        "Single centre support (mast/tree)",
        "End insulators with rope to tie-off points",
    ),
))

# --- Vertical family ---

_p(TypeBuildProfile(
    antenna_type="vertical",
    typical_knob_count=4,  # radiator length, radial count, radial length, radial slope
    critical_dim_names=(
        "radiator length", "radial count", "radial length", "radial slope angle",
    ),
    type_risks=(
        "ground system quality dominates real-world performance",
        "loading coil losses if shortened",
        "common-mode currents on coax shield",
    ),
    sensitivity_factor=0.6,
    mechanical_factor=0.5,
    feed_complexity=0.3,      # needs ground radials, maybe matching
    tuning_tips=(
        "Minimum 16 radials, more is always better",
        "VNA: R should be 30-40Ω over radials; use shunt match for 50Ω",
        "If loaded: measure coil Q independently before installing",
    ),
    bom_tips=(
        "Aluminium tubing (telescoping) for radiator",
        "16+ radials of 16 AWG wire, λ/4 each",
        "Choke balun at feedpoint",
    ),
))

_p(TypeBuildProfile(
    antenna_type="j_pole",
    typical_knob_count=3,  # radiator length, stub length, stub spacing
    critical_dim_names=("radiator length", "matching stub length", "stub spacing"),
    type_risks=("stub spacing is a critical match dimension",),
    sensitivity_factor=0.6,
    mechanical_factor=0.4,
    feed_complexity=0.2,
    tuning_tips=(
        "Feedpoint tap position on stub controls impedance",
        "VNA: sweep to find minimum SWR, adjust tap height",
    ),
    bom_tips=(
        "Copper pipe or aluminium rod",
        "Insulated spacers between the two parallel sections",
    ),
))

_p(TypeBuildProfile(
    antenna_type="end_fed",
    typical_knob_count=3,  # wire length, counterpoise, matching transformer
    critical_dim_names=("wire length", "counterpoise length", "matching transformer ratio"),
    type_risks=(
        "very high impedance requires good matching transformer",
        "RF on coax shield without choke",
        "multiband operation depends on harmonic relationship",
    ),
    sensitivity_factor=0.6,
    mechanical_factor=0.3,
    feed_complexity=0.7,   # needs matching transformer
    tuning_tips=(
        "49:1 or 64:1 transformer for EFHW",
        "Add a short counterpoise wire (0.05λ) at feedpoint",
        "VNA: check each harmonic band independently",
    ),
    bom_tips=(
        "FT240-43 toroid for matching transformer",
        "14 AWG stranded wire",
        "150 pF compensation capacitor across transformer",
    ),
))

# --- Directional beams ---

_p(TypeBuildProfile(
    antenna_type="yagi",
    typical_knob_count=7,  # R/DE/D lengths + spacings + match
    critical_dim_names=(
        "reflector length", "driven element length",
        "director length(s)", "reflector-DE spacing",
        "DE-director spacing(s)", "boom diameter correction",
    ),
    type_risks=(
        "element-to-boom mounting shifts resonance",
        "tight spacings increase sensitivity",
        "boom sag/twist at long boom lengths",
        "element taper changes electrical length",
    ),
    sensitivity_factor=1.0,   # normal — well-understood
    mechanical_factor=0.7,
    feed_complexity=0.5,      # usually needs gamma/hairpin match
    tuning_tips=(
        "Build driven element first, tune to resonance alone",
        "Add reflector: SWR should rise slightly, F/B appears",
        "Add directors one at a time from front",
        "VNA: R≈25Ω typical — use hairpin or gamma for 50Ω",
        "Check SWR across band — bandwidth narrows with more elements",
    ),
    bom_tips=(
        "6061-T6 aluminium tubing, telescoping for element taper",
        "Boom: square or round aluminium, insulated element mounts",
        "Stainless steel hardware at junctions",
    ),
))

_p(TypeBuildProfile(
    antenna_type="moxon",
    typical_knob_count=5,  # width, height, driven length, reflector length, tip gap
    critical_dim_names=(
        "element width (A)", "tip gap (B — CRITICAL)",
        "tail length driven (C)", "tail length reflector (D)",
        "overall width",
    ),
    type_risks=(
        "TIP GAP is the buildability killer — ±5mm shifts F/B by 10+ dB",
        "wind and thermal expansion change the gap",
        "spreader flex changes gap under load",
    ),
    sensitivity_factor=1.5,   # higher than Yagi due to tip gap
    mechanical_factor=0.8,
    feed_complexity=0.1,      # natural 50Ω is the big advantage
    tuning_tips=(
        "Get the tip gap right FIRST — everything else follows",
        "Use rigid fibreglass spreaders, not flexible PVC",
        "VNA: R should be 48-55Ω with X≈0 — if not, adjust gap",
        "F/B is the tuning indicator — peak F/B = correct gap",
    ),
    bom_tips=(
        "Fibreglass or HDPE spreaders (rigid!)",
        "14 AWG bare copper or 12 AWG for HF",
        "Precise spacers or jig for the tip gap",
        "Direct 50Ω coax feed — no matching network needed",
    ),
))

_p(TypeBuildProfile(
    antenna_type="hexbeam",
    typical_knob_count=8,  # wire lengths per band + compression + spacing
    critical_dim_names=(
        "frame radius", "wire lengths per band",
        "tip compression factor", "vertical spacing between planes",
        "wire-to-wire clearance at spreader tips",
    ),
    type_risks=(
        "spreader flex changes wire positions in wind",
        "close wire spacing between bands at spreader tips",
        "interaction between bands if wires too close",
        "UV degradation of wire insulation changes coupling",
    ),
    sensitivity_factor=1.3,
    mechanical_factor=1.0,    # complex frame
    feed_complexity=0.3,
    tuning_tips=(
        "Tune highest band first (least affected by others)",
        "Work downward to lowest band",
        "VNA: check each band independently, then all together",
        "Adjust wire lengths at the tips for SWR, not at feed",
    ),
    bom_tips=(
        "6 fibreglass spreaders (≈5m for 20m band)",
        "Insulated wire (THHN or similar) for each band",
        "Hub plate: aluminium or 3D-printed with wire guides",
        "Non-conductive cord to tension wires along spreaders",
    ),
))

_p(TypeBuildProfile(
    antenna_type="quad",
    typical_knob_count=6,  # loop perimeters + spacings + feed position
    critical_dim_names=(
        "reflector loop perimeter", "driven loop perimeter",
        "director loop perimeter(s)", "element spacing",
        "feed position on driven loop",
    ),
    type_risks=(
        "full-wave loops are large and catching wind",
        "high impedance (~120Ω) needs matching",
        "spreader breakage in ice/wind storms",
        "wire sag distorts loop shape",
    ),
    sensitivity_factor=0.8,   # loops are fairly tolerant
    mechanical_factor=1.3,    # big, heavy, wind-catching
    feed_complexity=0.5,      # 120Ω needs matching
    tuning_tips=(
        "Loop perimeter ≈ 1005/f(MHz) feet for driven",
        "Reflector: +5%, director: -5%",
        "VNA: expect ~120Ω — use λ/4 75Ω coax for match to 50Ω",
        "Side-fed = vertical polarisation, bottom-fed = horizontal",
    ),
    bom_tips=(
        "Fibreglass or bamboo spreaders",
        "14 AWG copper wire for loops",
        "75Ω coax (RG-11) for matching section",
        "Strong hub — must handle wind load on loops",
    ),
))

_p(TypeBuildProfile(
    antenna_type="lpda",
    typical_knob_count=4,  # τ, σ, N elements, feed line Z0
    critical_dim_names=(
        "design ratio tau (τ)", "spacing constant sigma (σ)",
        "number of elements", "feed line impedance",
        "phase-reversal connections",
    ),
    type_risks=(
        "per-element hand-tuning defeats the design — stick to τ/σ formula",
        "phase-reversal feedline wiring errors kill the pattern",
        "boom length set by lowest frequency",
    ),
    sensitivity_factor=0.6,   # parametric design is forgiving
    mechanical_factor=0.7,
    feed_complexity=0.4,      # built-in feed line
    tuning_tips=(
        "Do NOT tune individual elements — regenerate from τ/σ if wrong",
        "Verify phase reversal: crossed connections at EVERY element",
        "VNA: SWR should be flat across the entire band, not just at one freq",
        "Feed the shortest (front) element",
    ),
    bom_tips=(
        "Aluminium tubing, telescoping per element",
        "Twin-line feed or parallel tubes on boom",
        "Alternating-side element mounting for phase reversal",
    ),
))

# --- Phased arrays ---

_p(TypeBuildProfile(
    antenna_type="phased_array",
    typical_knob_count=6,  # element lengths, spacings, phase angles, feed impedances
    critical_dim_names=(
        "element spacing", "phase angle per element",
        "feed amplitude balance", "phasing line lengths",
        "element-to-element coupling",
    ),
    type_risks=(
        "phasing-line tolerance is critical — ±1° phase error shifts pattern",
        "common-mode currents on feed harness",
        "mutual coupling changes individual element impedance",
        "requires precise instrumentation to verify",
    ),
    sensitivity_factor=1.5,
    mechanical_factor=0.8,
    feed_complexity=0.9,   # complex phasing network
    tuning_tips=(
        "Build and tune each element individually first",
        "Measure mutual impedance between elements",
        "Use current probes to verify amplitude/phase balance",
        "VNA: check each port, then combined array pattern",
    ),
    bom_tips=(
        "Precision-cut phasing lines (measure velocity factor!)",
        "Current baluns on each element",
        "Weatherproof junction box for phasing harness",
    ),
))

# --- Loops ---

_p(TypeBuildProfile(
    antenna_type="loop",
    typical_knob_count=3,  # perimeter, height, feed position
    critical_dim_names=(
        "loop perimeter (≈1λ)", "height above ground", "feed position",
    ),
    type_risks=(
        "high impedance (~120Ω) needs matching",
        "shape distortion from wind/sag",
    ),
    sensitivity_factor=0.6,   # loops are tolerant
    mechanical_factor=0.6,
    feed_complexity=0.4,
    tuning_tips=(
        "Perimeter controls frequency — adjust total wire length",
        "VNA: expect ~120Ω, match with λ/4 75Ω section",
    ),
    bom_tips=(
        "14 AWG stranded copper wire",
        "Rope/cord for shape maintenance",
        "75Ω matching section",
    ),
))

_p(TypeBuildProfile(
    antenna_type="delta_loop",
    typical_knob_count=4,  # perimeter, apex height, base width, feed position
    critical_dim_names=(
        "total perimeter", "apex height", "base width", "feed position (side/corner)",
    ),
    type_risks=(
        "same as loop — shape distortion, high-Z feed",
        "apex support must handle full weight",
    ),
    sensitivity_factor=0.6,
    mechanical_factor=0.5,
    feed_complexity=0.4,
    tuning_tips=(
        "Feed bottom for horizontal pol, feed corner for vertical",
        "Total wire = 1005/f(MHz) feet",
    ),
    bom_tips=("14 AWG wire", "Three support points", "75Ω matching section"),
))

_p(TypeBuildProfile(
    antenna_type="magnetic_loop",
    typical_knob_count=3,  # circumference, conductor diameter, capacitor value
    critical_dim_names=(
        "loop circumference", "conductor diameter (CRITICAL for efficiency)",
        "tuning capacitor value and voltage rating",
    ),
    type_risks=(
        "EXTREMELY narrow bandwidth — must retune every 25 kHz on HF",
        "tuning capacitor sees very high RF voltage (kV at 100W)",
        "conductor loss dominates efficiency — must use thick pipe",
        "any joint resistance is a direct loss mechanism",
    ),
    sensitivity_factor=2.0,   # very sensitive — high Q
    mechanical_factor=0.6,
    feed_complexity=0.5,
    tuning_tips=(
        "Use a butterfly or vacuum variable capacitor rated for the voltage",
        "Solder ALL joints — mechanical joints add loss",
        "Coupling loop size controls feedpoint impedance",
        "VNA: very sharp dip, find it by sweeping slowly",
    ),
    bom_tips=(
        "1-inch or larger copper pipe (NOT wire!)",
        "High-voltage variable capacitor (vacuum or butterfly)",
        "Silver-solder all joints",
        "Small coupling loop or gamma match for feed",
    ),
))

# --- Specialty ---

_p(TypeBuildProfile(
    antenna_type="helix",
    typical_knob_count=4,  # turns, diameter, pitch, ground plane
    critical_dim_names=("number of turns", "helix diameter", "turn spacing/pitch", "ground plane size"),
    type_risks=("maintaining uniform pitch over many turns", "ground plane size affects pattern"),
    sensitivity_factor=0.5,
    mechanical_factor=0.8,
    feed_complexity=0.3,
    tuning_tips=("Diameter ≈ λ/π for axial mode", "Pitch ≈ λ/4 per turn"),
    bom_tips=("Rigid former (PVC pipe)", "Heavy copper wire or tube", "Metal ground plane"),
))

_p(TypeBuildProfile(
    antenna_type="collinear",
    typical_knob_count=4,
    critical_dim_names=("section lengths", "phasing stub dimensions", "overall height", "feed position"),
    type_risks=("phasing stubs must be correct length", "support structure for tall antenna"),
    sensitivity_factor=0.7,
    mechanical_factor=0.8,
    feed_complexity=0.4,
    tuning_tips=("Tune each section to resonance before assembling",),
    bom_tips=("Aluminium tubing", "Insulated phasing sections"),
))

_p(TypeBuildProfile(
    antenna_type="discone",
    typical_knob_count=3,
    critical_dim_names=("cone angle", "disc diameter", "element count/spacing"),
    type_risks=("many elements to fabricate", "cone symmetry matters"),
    sensitivity_factor=0.4,   # very broadband = tolerant
    mechanical_factor=0.7,
    feed_complexity=0.2,
    tuning_tips=("Should be flat SWR across entire range — if not, check cone angle",),
    bom_tips=("Brass or stainless steel rods", "Central feed hub"),
))

# Default for unknown types
_DEFAULT_PROFILE = TypeBuildProfile(
    antenna_type="unknown",
    typical_knob_count=5,
    critical_dim_names=("element dimensions", "feed position"),
    type_risks=("unknown type — verify all dimensions carefully",),
    sensitivity_factor=1.0,
    mechanical_factor=1.0,
    feed_complexity=0.5,
)


def _profile_for(antenna_type: str) -> TypeBuildProfile:
    return _PROFILES.get(antenna_type.lower().strip(), _DEFAULT_PROFILE)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_complexity(wires: list[_Wire], nec: str, profile: TypeBuildProfile) -> float:
    """Score based on number of independent "knobs" and junctions.

    More distinct tags, LD cards, TL cards → more things to get right → lower score.
    """
    n_wires = len(wires)
    n_tags = len({w.tag for w in wires})
    n_tl = _count_cards(nec, "TL")
    n_ld = _count_cards(nec, "LD")
    n_ex = _count_cards(nec, "EX")

    # Junction count: endpoints that touch (wire connections)
    junctions = 0
    endpoints = []
    for w in wires:
        endpoints.append((w.x1, w.y1, w.z1))
        endpoints.append((w.x2, w.y2, w.z2))
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if all(abs(a[k] - b[k]) < 1e-4 for k in range(3)):
                junctions += 1
                break

    # Fewer knobs = higher score
    knob_count = n_tags + n_tl + n_ld + (1 if n_ex > 1 else 0)
    knob_penalty = min(knob_count / 15.0, 1.0)  # 15+ knobs = 0 score

    # Junction count: more = more soldering/connections
    junction_penalty = min(junctions / 20.0, 1.0)

    raw = 100.0 * (1.0 - 0.6 * knob_penalty - 0.4 * junction_penalty)
    return max(0.0, min(100.0, raw))


def _score_match_practicality(nec: str, wires: list[_Wire],
                               profile: TypeBuildProfile) -> float:
    """Score based on feed complexity."""
    # Base from profile
    base = 100.0 * (1.0 - profile.feed_complexity)

    n_ex = _count_cards(nec, "EX")
    n_tl = _count_cards(nec, "TL")

    # Multiple feed points add complexity
    if n_ex > 2:
        base -= 15.0
    # TL matching adds complexity
    if n_tl > 0:
        base -= 10.0

    return max(0.0, min(100.0, base))


def _score_mechanical(wires: list[_Wire], wavelength: float,
                       profile: TypeBuildProfile) -> float:
    """Score based on physical construction difficulty."""
    if not wires:
        return 50.0

    longest = _longest_unsupported(wires)
    clearance = _min_clearance(wires)

    # Long unsupported spans → sag risk
    # > 2λ unsupported is very problematic
    span_penalty = min(longest / (2.0 * wavelength), 1.0)

    # Close wire clearance → alignment-critical
    # < 0.01λ is very tight, < 0.005λ is almost unbuildable
    if clearance < 0.005 * wavelength:
        clearance_penalty = 1.0
    elif clearance < 0.02 * wavelength:
        clearance_penalty = 0.7
    elif clearance < 0.05 * wavelength:
        clearance_penalty = 0.3
    else:
        clearance_penalty = 0.0

    # Wire count — more wires = more fabrication
    wire_penalty = min(len(wires) / 30.0, 1.0)

    raw = 100.0 * (1.0
                    - 0.35 * span_penalty
                    - 0.35 * clearance_penalty
                    - 0.15 * wire_penalty
                    - 0.15 * (profile.mechanical_factor - 0.3) / 0.7)  # type bias
    return max(0.0, min(100.0, raw))


def _score_materials(nec: str, profile: TypeBuildProfile) -> float:
    """Score based on materials realism.

    Penalise designs that rely on ideal components without loss modelling.
    """
    n_ld = _count_cards(nec, "LD")
    has_conductivity = False
    has_rll_load = False

    for line in nec.splitlines():
        parts = line.replace(",", " ").split()
        if not parts or parts[0].upper() != "LD":
            continue
        try:
            ld_type = int(float(parts[1]))
            if ld_type == 5:
                has_conductivity = True
            elif ld_type in (0, 1, 4):
                has_rll_load = True
        except (ValueError, IndexError):
            continue

    score = 80.0  # start optimistic

    # Penalise loads without conductivity model (idealized)
    if has_rll_load and not has_conductivity:
        score -= 20.0  # loading without loss model = fantasy

    # No conductivity at all — common in simple models, minor penalty
    if not has_conductivity and len(nec.splitlines()) > 10:
        score -= 10.0

    # Type-specific material concerns
    if profile.antenna_type == "magnetic_loop" and not has_conductivity:
        score -= 30.0  # loop efficiency is ALL about conductor loss

    return max(0.0, min(100.0, score))


def _score_sensitivity_heuristic(wires: list[_Wire], wavelength: float,
                                  profile: TypeBuildProfile) -> float:
    """Heuristic sensitivity score based on geometry characteristics.

    True sensitivity testing requires re-running NEC (done separately via
    perturbation analysis).  This heuristic estimates it from geometry:
    - Close wire spacings → high sensitivity
    - Many elements in a small area → coupling sensitivity
    - Type-specific sensitivity factor from profile
    """
    if not wires:
        return 50.0

    clearance = _min_clearance(wires)

    # Clearance-based sensitivity
    if clearance < 0.005 * wavelength:
        clearance_score = 20.0   # hair-trigger
    elif clearance < 0.01 * wavelength:
        clearance_score = 40.0
    elif clearance < 0.03 * wavelength:
        clearance_score = 60.0
    elif clearance < 0.1 * wavelength:
        clearance_score = 80.0
    else:
        clearance_score = 95.0

    # Scale by type-specific factor
    type_adj = 1.0 / profile.sensitivity_factor
    raw = clearance_score * type_adj
    return max(0.0, min(100.0, raw))


def _score_deployment(nec: str) -> float:
    """Score based on deployment sensitivity.

    Models that require perfect ground or free space are less deployable.
    """
    has_ground = False
    ground_type = None
    for line in nec.splitlines():
        parts = line.replace(",", " ").split()
        if parts and parts[0].upper() == "GN":
            has_ground = True
            try:
                ground_type = int(float(parts[1]))
            except (ValueError, IndexError):
                pass

    # Free-space only (GN -1) — deployment insensitive (no ground to worry about)
    if not has_ground or ground_type == -1:
        return 85.0
    # Perfect ground (GN 1) — won't match reality
    if ground_type == 1:
        return 60.0
    # Real ground model (GN 2) — most realistic
    if ground_type == 2:
        return 90.0
    return 75.0


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------

def assess_buildability(
    nec_text: str,
    antenna_type: str = "unknown",
    freq_mhz: float = 0.0,
) -> BuildabilityReport:
    """Assess the buildability of a NEC antenna model.

    Returns a ``BuildabilityReport`` with score 0–100, grade, risks,
    critical dimensions, and tuning/BOM hints.
    """
    profile = _profile_for(antenna_type)
    wires = _parse_wires(nec_text)
    wavelength = _C_MPS / (freq_mhz * 1e6) if freq_mhz > 0 else 20.0  # default 20m

    # Compute sub-scores
    sensitivity = _score_sensitivity_heuristic(wires, wavelength, profile)
    complexity = _score_complexity(wires, nec_text, profile)
    match = _score_match_practicality(nec_text, wires, profile)
    mechanical = _score_mechanical(wires, wavelength, profile)
    materials = _score_materials(nec_text, profile)
    deployment = _score_deployment(nec_text)

    # Weighted total
    total = (
        0.30 * sensitivity
        + 0.20 * complexity
        + 0.15 * match
        + 0.15 * mechanical
        + 0.10 * materials
        + 0.10 * deployment
    )

    # Build risks list (top 3 lowest-scoring areas)
    area_scores = [
        (sensitivity, "Geometry sensitivity — small dimension changes cause large performance shifts"),
        (complexity, "Design complexity — many independent parameters to set correctly"),
        (match, "Feed matching — non-trivial impedance matching required"),
        (mechanical, "Mechanical difficulty — long spans, close clearances, or many junctions"),
        (materials, "Materials realism — loading/conductor losses not fully modelled"),
        (deployment, "Deployment sensitivity — performance depends heavily on installation conditions"),
    ]
    area_scores.sort(key=lambda x: x[0])
    top_risks = [msg for score, msg in area_scores[:3] if score < 70.0]

    # Add type-specific risks
    for risk in profile.type_risks:
        if risk not in top_risks and len(top_risks) < 3:
            top_risks.append(risk)

    report = BuildabilityReport(
        score=round(total, 1),
        grade=_grade(total),
        sensitivity_score=round(sensitivity, 1),
        complexity_score=round(complexity, 1),
        match_score=round(match, 1),
        mechanical_score=round(mechanical, 1),
        materials_score=round(materials, 1),
        deployment_score=round(deployment, 1),
        top_risks=top_risks[:3],
        critical_dims=list(profile.critical_dim_names),
        tuning_hints=list(profile.tuning_tips),
        bom_hints=list(profile.bom_tips),
        details={
            "n_wires": len(wires),
            "n_tags": len({w.tag for w in wires}),
            "longest_span_m": round(_longest_unsupported(wires), 3) if wires else 0,
            "min_clearance_m": round(_min_clearance(wires), 4) if wires else 0,
            "wavelength_m": round(wavelength, 3),
            "knob_count": profile.typical_knob_count,
            "type_profile": profile.antenna_type,
        },
    )
    return report
