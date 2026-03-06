"""
Antenna type classifier.

Determines antenna type from NEC geometry, excitation, comments, and filename.
Uses heuristic rules based on wire topology, element counts, and structural patterns.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parser import NECCard, ParseResult


# Canonical antenna types
ANTENNA_TYPES = [
    "yagi",
    "dipole",
    "vertical",
    "loop",
    "quad",
    "hexbeam",
    "lpda",
    "phased_array",
    "helix",
    "collinear",
    "inverted_v",
    "end_fed",
    "j_pole",
    "moxon",
    "wire_array",
    "patch",
    "fractal",
    "magnetic_loop",
    "bobtail_curtain",
    "rhombic",
    "beverage",
    "discone",
    "turnstile",
    "unknown",
]


@dataclass
class ClassificationResult:
    """Result of antenna type classification."""
    antenna_type: str
    confidence: float  # 0.0 – 1.0
    subtypes: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    frequency_mhz: float | None = None
    band: str | None = None
    element_count: int = 0
    ground_type: str = "free_space"

    @property
    def label(self) -> str:
        parts = [self.antenna_type]
        if self.subtypes:
            parts.append(f"({', '.join(self.subtypes)})")
        if self.band:
            parts.append(f"[{self.band}]")
        return " ".join(parts)


def classify(parsed: ParseResult) -> ClassificationResult:
    """Classify the antenna type from a parsed NEC file."""
    ctx = _AnalysisContext(parsed)
    ctx.extract_features()

    result = ClassificationResult(
        antenna_type="unknown",
        confidence=0.0,
        frequency_mhz=ctx.frequency,
        band=_freq_to_band(ctx.frequency) if ctx.frequency else None,
        element_count=ctx.n_wire_groups,
        ground_type=ctx.ground_label,
    )

    # Run classifiers in priority order — first high-confidence match wins
    classifiers = [
        _classify_from_comments,
        _classify_from_path,
        _classify_helix,
        _classify_patch,
        _classify_loop_quad,
        _classify_hexbeam,
        _classify_lpda,
        _classify_vertical,
        _classify_moxon,
        _classify_yagi,
        _classify_phased_array,
        _classify_collinear,
        _classify_wire_array,
        _classify_dipole,
    ]

    for clf in classifiers:
        clf(ctx, result)
        if result.confidence >= 0.7:
            break

    # If still unknown, last-resort from directory hints
    if result.antenna_type == "unknown":
        _classify_from_directory(ctx, result)

    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class _WireGroup:
    """Group of GW wires sharing the same tag number."""
    tag: int
    wires: list[NECCard]

    @property
    def total_length(self) -> float:
        length = 0.0
        for w in self.wires:
            lp = w.labeled_params
            x1, y1, z1 = lp.get("x1", 0), lp.get("y1", 0), lp.get("z1", 0)
            x2, y2, z2 = lp.get("x2", 0), lp.get("y2", 0), lp.get("z2", 0)
            if all(isinstance(v, (int, float)) for v in (x1, y1, z1, x2, y2, z2)):
                length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return length

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Average of all wire midpoints."""
        xs, ys, zs, n = 0.0, 0.0, 0.0, 0
        for w in self.wires:
            lp = w.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if all(isinstance(v, (int, float)) for v in coords):
                x1, y1, z1, x2, y2, z2 = coords
                xs += (x1 + x2) / 2
                ys += (y1 + y2) / 2
                zs += (z1 + z2) / 2
                n += 1
        return (xs / n, ys / n, zs / n) if n else (0, 0, 0)

    @property
    def is_primarily_vertical(self) -> bool:
        """True if most wire length is along Z axis."""
        horiz = 0.0
        vert = 0.0
        for w in self.wires:
            lp = w.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if all(isinstance(v, (int, float)) for v in coords):
                x1, y1, z1, x2, y2, z2 = coords
                horiz += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                vert += abs(z2 - z1)
        return vert > horiz * 1.5 if (horiz + vert) > 0 else False

    @property
    def is_primarily_horizontal(self) -> bool:
        """True if most wire length is along X/Y plane."""
        horiz = 0.0
        vert = 0.0
        for w in self.wires:
            lp = w.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if all(isinstance(v, (int, float)) for v in coords):
                x1, y1, z1, x2, y2, z2 = coords
                horiz += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                vert += abs(z2 - z1)
        return horiz > vert * 1.5 if (horiz + vert) > 0 else False

    @property
    def span_x(self) -> float:
        xs = []
        for w in self.wires:
            lp = w.labeled_params
            for k in ("x1", "x2"):
                v = lp.get(k)
                if isinstance(v, (int, float)):
                    xs.append(v)
        return max(xs) - min(xs) if xs else 0.0

    @property
    def span_z(self) -> float:
        zs = []
        for w in self.wires:
            lp = w.labeled_params
            for k in ("z1", "z2"):
                v = lp.get(k)
                if isinstance(v, (int, float)):
                    zs.append(v)
        return max(zs) - min(zs) if zs else 0.0


class _AnalysisContext:
    """Extracted features from a parsed NEC file."""

    def __init__(self, parsed: ParseResult):
        self.parsed = parsed
        self.wire_groups: list[_WireGroup] = []
        self.n_wire_groups: int = 0
        self.n_total_wires: int = 0
        self.frequency: float | None = None
        self.wavelength: float | None = None
        self.has_ground: bool = False
        self.ground_type_code: int | None = None
        self.ground_label: str = "free_space"
        self.ex_tags: list[int] = []
        self.has_tl: bool = False
        self.has_helix: bool = False
        self.has_surface_patch: bool = False
        self.comment_text: str = ""
        self.source_path: str = parsed.source

    def extract_features(self) -> None:
        wires = self.parsed.wire_cards
        self.n_total_wires = len(wires)

        # Group wires by tag
        tag_map: dict[int, list[NECCard]] = {}
        for w in wires:
            tag = w.labeled_params.get("tag", 0)
            if isinstance(tag, int):
                tag_map.setdefault(tag, []).append(w)
        self.wire_groups = [_WireGroup(tag=t, wires=ws) for t, ws in sorted(tag_map.items())]
        self.n_wire_groups = len(self.wire_groups)

        # Frequency
        for card in self.parsed.cards:
            if card.card_type == "FR":
                freq = card.labeled_params.get("freq")
                if isinstance(freq, (int, float)) and freq > 0:
                    self.frequency = float(freq)
                    self.wavelength = 299.792458 / self.frequency  # meters
                    break

        # Ground
        for card in self.parsed.cards:
            if card.card_type == "GN":
                gt = card.labeled_params.get("groundType")
                if isinstance(gt, int):
                    self.ground_type_code = gt
                    if gt == -1:
                        self.ground_label = "free_space"
                    elif gt == 0:
                        self.ground_label = "finite_ground"
                        self.has_ground = True
                    elif gt == 1:
                        self.ground_label = "perfect_ground"
                        self.has_ground = True
                    elif gt == 2:
                        self.ground_label = "sommerfeld_norton"
                        self.has_ground = True
                    else:
                        self.ground_label = f"ground_{gt}"
                        self.has_ground = True

        # Excitation tags
        for card in self.parsed.cards:
            if card.card_type == "EX":
                tag = card.labeled_params.get("tag")
                if isinstance(tag, int):
                    self.ex_tags.append(tag)

        # Transmission lines
        self.has_tl = any(c.card_type == "TL" for c in self.parsed.cards)

        # Helix cards
        self.has_helix = any(c.card_type == "GH" for c in self.parsed.cards)

        # Surface patches
        self.has_surface_patch = any(c.card_type in ("SP", "SM") for c in self.parsed.cards)

        # Comment text
        self.comment_text = self.parsed.comment_text.lower()


# ---------------------------------------------------------------------------
# Individual classifiers
# ---------------------------------------------------------------------------

_KEYWORD_MAP: list[tuple[str, list[str], list[str]]] = [
    ("yagi", ["yagi", "yagi-uda"], []),
    ("dipole", ["dipole", "half-wave dipole", "folded dipole", "half wave"], ["inverted"]),
    ("inverted_v", ["inverted v", "inv-v", "inverted-v", "inv v"], []),
    ("vertical", ["vertical", "ground plane", "groundplane", "quarter-wave", "quarter wave", "1/4 wave"], []),
    ("loop", ["loop", "delta loop", "full wave loop"], ["log", "magnetic"]),
    ("quad", ["quad", "cubical quad"], []),
    ("hexbeam", ["hex beam", "hexbeam", "hex-beam"], []),
    ("lpda", ["lpda", "log periodic", "log-periodic", "log cell", "log-cell"], []),
    ("phased_array", ["phased", "endfire", "broadside", "curtain"], ["bobtail"]),
    ("helix", ["helix", "helical"], []),
    ("collinear", ["collinear", "coco", "coaxial collinear"], []),
    ("moxon", ["moxon"], []),
    ("wire_array", ["lazy h", "lazy-h", "sterba", "bruce", "8jk"], []),
    ("bobtail_curtain", ["bobtail", "bobtail curtain"], []),
    ("rhombic", ["rhombic"], []),
    ("beverage", ["beverage"], []),
    ("discone", ["discone", "disc-cone"], []),
    ("turnstile", ["turnstile"], []),
    ("fractal", ["fractal", "koch", "sierpinski"], []),
    ("magnetic_loop", ["magnetic loop", "mag loop", "small loop", "magloop"], []),
    ("end_fed", ["end fed", "end-fed", "efhw", "zepp"], []),
    ("j_pole", ["j-pole", "j pole", "jpole", "slim jim"], []),
    ("patch", ["patch", "microstrip"], []),
]


def _classify_from_comments(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Match antenna type from CM/CE comment text."""
    text = ctx.comment_text
    if not text:
        return
    for atype, keywords, excludes in _KEYWORD_MAP:
        if any(kw in text for kw in excludes):
            continue
        for kw in keywords:
            if kw in text:
                result.antenna_type = atype
                result.confidence = max(result.confidence, 0.8)
                result.evidence.append(f"comment contains '{kw}'")
                return


def _classify_from_path(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Match antenna type from the filename."""
    if result.confidence >= 0.7:
        return
    name = Path(ctx.source_path).stem.lower().replace("_", " ").replace("-", " ")
    for atype, keywords, excludes in _KEYWORD_MAP:
        if any(kw in name for kw in excludes):
            continue
        for kw in keywords:
            if kw in name:
                if result.antenna_type == "unknown":
                    result.antenna_type = atype
                    result.confidence = max(result.confidence, 0.5)
                elif result.antenna_type == atype:
                    result.confidence = min(result.confidence + 0.1, 1.0)
                result.evidence.append(f"filename contains '{kw}'")
                return


def _classify_from_directory(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Last-resort: infer from parent directory name."""
    parent = Path(ctx.source_path).parent.name.lower().replace("-", " ").replace("_", " ")
    dir_map = {
        "yagis hf": "yagi", "yagis": "yagi",
        "verticals": "vertical",
        "lpdas": "lpda",
        "quads": "quad",
        "phased arrays": "phased_array",
        "wire arrays": "wire_array",
        "vhf uhf": "yagi",
        "hfbeams": "yagi",
        "vhfbeams": "yagi",
        "hfsimple": "dipole",
        "vhfsimple": "dipole",
        "hfvertical": "vertical",
        "lfvertical": "vertical",
        "hfcollinear": "collinear",
        "hfmultiband": "dipole",
        "vhfmultiband": "dipole",
        "hfshort": "dipole",
        "hfactivefeed": "dipole",
        "aperiodic": "beverage",
        "fractals": "fractal",
        "spatch": "patch",
        "objects": "unknown",
    }
    if parent in dir_map:
        result.antenna_type = dir_map[parent]
        result.confidence = max(result.confidence, 0.3)
        result.evidence.append(f"directory '{parent}' suggests {dir_map[parent]}")


def _classify_helix(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Helix if GH cards are present."""
    if result.confidence >= 0.7:
        return
    if ctx.has_helix:
        result.antenna_type = "helix"
        result.confidence = max(result.confidence, 0.9)
        result.evidence.append("GH (helix) geometry card present")


def _classify_patch(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Patch antenna if surface patch cards."""
    if result.confidence >= 0.7:
        return
    if ctx.has_surface_patch:
        result.antenna_type = "patch"
        result.confidence = max(result.confidence, 0.85)
        result.evidence.append("SP/SM (surface patch) cards present")


def _classify_loop_quad(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Detect loop and quad antennas by closed-wire topology."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups < 1:
        return

    closed_groups = 0
    for grp in ctx.wire_groups:
        if len(grp.wires) < 3:
            continue
        # Check if wire endpoints form a closed path
        first = grp.wires[0]
        last = grp.wires[-1]
        start = (first.labeled_params.get("x1"), first.labeled_params.get("y1"), first.labeled_params.get("z1"))
        end = (last.labeled_params.get("x2"), last.labeled_params.get("y2"), last.labeled_params.get("z2"))
        if all(isinstance(v, (int, float)) for v in start + end):
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(start, end)))
            if dist < 0.01:  # effectively closed
                closed_groups += 1

    if closed_groups >= 2:
        result.antenna_type = "quad"
        result.confidence = max(result.confidence, 0.75)
        result.evidence.append(f"{closed_groups} closed wire loops detected (quad)")
    elif closed_groups == 1:
        result.antenna_type = "loop"
        result.confidence = max(result.confidence, 0.6)
        result.evidence.append("1 closed wire loop detected")


def _classify_hexbeam(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Hexbeam: 3+ wire groups, hexagonal symmetry (6-fold)."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups < 3:
        return
    # Count wires per group; hexbeam has ~6 wires per group (6 sides)
    groups_with_6 = sum(1 for g in ctx.wire_groups if len(g.wires) == 6)
    if groups_with_6 >= 2:
        result.antenna_type = "hexbeam"
        result.confidence = max(result.confidence, 0.75)
        result.evidence.append(f"{groups_with_6} groups with 6 wires each (hexagonal)")


def _classify_lpda(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """LPDA: many parallel horizontal elements with decreasing lengths and TL feed."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups < 4:
        return

    # Check for elements that are roughly parallel and horizontal with varying lengths
    horizontal_groups = [g for g in ctx.wire_groups if g.is_primarily_horizontal]
    if len(horizontal_groups) < 4:
        return

    # Check if lengths are monotonically changing (log periodic signature)
    lengths = [g.total_length for g in horizontal_groups]
    if len(lengths) >= 4:
        diffs = [lengths[i+1] - lengths[i] for i in range(len(lengths)-1)]
        # In an LPDA, elements progressively change size
        # Check if most diffs have the same sign (monotonic)
        positive = sum(1 for d in diffs if d > 0)
        negative = sum(1 for d in diffs if d < 0)
        monotonic_ratio = max(positive, negative) / len(diffs) if diffs else 0

        if monotonic_ratio > 0.7 and ctx.has_tl:
            result.antenna_type = "lpda"
            result.confidence = max(result.confidence, 0.7)
            result.evidence.append(f"{len(horizontal_groups)} elements with progressive lengths + TL feed")


def _classify_vertical(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Vertical if primary element is vertical, optionally with ground."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups < 1:
        return

    # Find the excited element
    excited = [g for g in ctx.wire_groups if g.tag in ctx.ex_tags]
    if not excited:
        excited = ctx.wire_groups[:1]  # fallback: first group

    if any(g.is_primarily_vertical for g in excited):
        # Check for ground plane: ground model or radial wires
        if ctx.has_ground or ctx.ground_type_code in (0, 1, 2):
            result.antenna_type = "vertical"
            result.confidence = max(result.confidence, 0.7)
            result.evidence.append("vertical excited element with ground model")
        else:
            result.antenna_type = "vertical"
            result.confidence = max(result.confidence, 0.5)
            result.evidence.append("vertical excited element (no ground model)")


def _classify_moxon(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Moxon: 2-element with bent ends (driver + reflector, 4-6 wires per element)."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups != 2:
        return
    # Moxon has 2 groups with ~3 wires each (center + 2 bent tails)
    wire_counts = [len(g.wires) for g in ctx.wire_groups]
    if all(2 <= c <= 5 for c in wire_counts):
        # Check for bent-end topology: wires going in different directions
        for g in ctx.wire_groups:
            if len(g.wires) >= 3:
                result.antenna_type = "moxon"
                result.confidence = max(result.confidence, 0.45)
                result.evidence.append("2 element groups with multi-wire bent geometry")
                return


def _classify_yagi(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Yagi: multiple parallel horizontal elements along a boom axis."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups < 2:
        return

    horizontal = [g for g in ctx.wire_groups if g.is_primarily_horizontal]
    if len(horizontal) < 2:
        return

    # Check elements are roughly parallel (similar X spans, stacked in Y or Z)
    spans = [g.span_x for g in horizontal]
    if all(s > 0 for s in spans):
        # Yagis have elements roughly symmetric about center, different lengths
        centroids = [g.centroid for g in horizontal]
        # Check if centroids are spread along one axis (boom)
        zs = [c[2] for c in centroids]
        ys = [c[1] for c in centroids]
        z_spread = max(zs) - min(zs) if zs else 0
        y_spread = max(ys) - min(ys) if ys else 0
        boom_spread = max(z_spread, y_spread)

        if boom_spread > 0 and len(horizontal) >= 2:
            if len(horizontal) >= 3:
                result.confidence = max(result.confidence, 0.7)
            else:
                result.confidence = max(result.confidence, 0.55)
            result.antenna_type = "yagi"
            result.evidence.append(f"{len(horizontal)} parallel horizontal elements spread {boom_spread:.2f}m")


def _classify_phased_array(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Phased array: multiple excitation sources or TL-fed identical elements."""
    if result.confidence >= 0.7:
        return
    if len(ctx.ex_tags) >= 2:
        result.antenna_type = "phased_array"
        result.confidence = max(result.confidence, 0.6)
        result.evidence.append(f"{len(ctx.ex_tags)} excitation sources")
    elif ctx.has_tl and ctx.n_wire_groups >= 3:
        tl_count = sum(1 for c in ctx.parsed.cards if c.card_type == "TL")
        if tl_count >= 2:
            result.antenna_type = "phased_array"
            result.confidence = max(result.confidence, 0.5)
            result.evidence.append(f"{tl_count} transmission lines feeding {ctx.n_wire_groups} elements")


def _classify_collinear(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Collinear: multiple co-axial vertical elements."""
    if result.confidence >= 0.7:
        return
    vertical_groups = [g for g in ctx.wire_groups if g.is_primarily_vertical]
    if len(vertical_groups) >= 3:
        # Check if they share the same X,Y position (co-linear)
        centers = [(g.centroid[0], g.centroid[1]) for g in vertical_groups]
        x_spread = max(c[0] for c in centers) - min(c[0] for c in centers)
        y_spread = max(c[1] for c in centers) - min(c[1] for c in centers)
        if x_spread < 0.1 and y_spread < 0.1:
            result.antenna_type = "collinear"
            result.confidence = max(result.confidence, 0.65)
            result.evidence.append(f"{len(vertical_groups)} co-axial vertical elements")


def _classify_wire_array(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Wire array: multiple parallel driven elements in a plane."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups >= 3 and ctx.has_tl:
        result.antenna_type = "wire_array"
        result.confidence = max(result.confidence, 0.4)
        result.evidence.append(f"{ctx.n_wire_groups} elements with TL interconnection")


def _classify_dipole(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Dipole: fallback for 1-2 horizontal elements without yagi-like boom."""
    if result.confidence >= 0.7:
        return
    if ctx.n_wire_groups == 0:
        return

    excited = [g for g in ctx.wire_groups if g.tag in ctx.ex_tags]
    if not excited:
        excited = ctx.wire_groups[:1]

    if any(g.is_primarily_horizontal for g in excited):
        if ctx.n_wire_groups <= 2 and not ctx.has_tl:
            # Check for inverted-V: horizontal wires with some Z slope
            for g in excited:
                z_span = g.span_z
                x_span = g.span_x
                if z_span > 0 and x_span > 0 and z_span / x_span > 0.15:
                    result.antenna_type = "inverted_v"
                    result.confidence = max(result.confidence, 0.55)
                    result.evidence.append("single element with significant Z droop (inverted-V)")
                    return
            result.antenna_type = "dipole"
            result.confidence = max(result.confidence, 0.5)
            result.evidence.append("1-2 horizontal elements, simple feed")


# ---------------------------------------------------------------------------
# Frequency-to-band mapping
# ---------------------------------------------------------------------------

_BAND_TABLE: list[tuple[str, float, float]] = [
    ("160m", 1.8, 2.0),
    ("80m", 3.5, 4.0),
    ("60m", 5.3, 5.4),
    ("40m", 7.0, 7.3),
    ("30m", 10.1, 10.15),
    ("20m", 14.0, 14.35),
    ("17m", 18.068, 18.168),
    ("15m", 21.0, 21.45),
    ("12m", 24.89, 24.99),
    ("10m", 28.0, 29.7),
    ("6m", 50.0, 54.0),
    ("2m", 144.0, 148.0),
    ("1.25m", 222.0, 225.0),
    ("70cm", 420.0, 450.0),
    ("33cm", 902.0, 928.0),
    ("23cm", 1240.0, 1300.0),
]


def _freq_to_band(freq: float) -> str | None:
    for name, lo, hi in _BAND_TABLE:
        if lo <= freq <= hi:
            return name
    # Approximate band by wavelength
    wl = 299.792458 / freq
    if wl > 100:
        return "LF"
    if wl > 10:
        return "HF"
    if wl > 1:
        return "VHF"
    if wl > 0.1:
        return "UHF"
    return "microwave"
