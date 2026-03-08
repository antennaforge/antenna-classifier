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
    "quagi",
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
    "delta_loop",
    "v_beam",
    "batwing",
    "zigzag",
    "rhombic",
    "beverage",
    "discone",
    "turnstile",
    "wire_object",
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
    bands: list[str] = field(default_factory=list)
    is_multiband: bool = False
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
    # Specific structural detectors first, then general topology (loops)
    classifiers = [
        _classify_from_comments,
        _classify_helix,
        _classify_patch,
        _classify_hexbeam,
        _classify_lpda,
        _classify_vertical,
        _classify_moxon,
        _classify_yagi,
        _classify_loop_quad,
        _classify_phased_array,
        _classify_collinear,
        _classify_wire_array,
        _classify_dipole,
    ]

    for clf in classifiers:
        clf(ctx, result)
        if result.confidence >= 0.7:
            break

    # Enrich moxon with construction subtype if comment detector got it
    # but _classify_moxon didn't run (stopped by confidence threshold).
    if result.antenna_type == "moxon" and not result.subtypes:
        if ctx.n_wire_groups == 2:
            result.subtypes.append("wire")
        elif 3 <= ctx.n_wire_groups <= 4:
            is_tube, _ = _detect_tube_moxon(ctx)
            if is_tube:
                result.subtypes.append("tube")

    # If still unknown, check for wire-grid objects (no EX/FR cards)
    if result.antenna_type == "unknown":
        _classify_wire_object(ctx, result)

    # Multiband detection — check if frequencies span multiple bands
    _detect_multiband(ctx, result)

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

    def _gather(self, k1: str, k2: str) -> list[float]:
        vals = []
        for w in self.wires:
            lp = w.labeled_params
            for k in (k1, k2):
                v = lp.get(k)
                if isinstance(v, (int, float)):
                    vals.append(v)
        return vals

    @property
    def span_x(self) -> float:
        xs = self._gather("x1", "x2")
        return max(xs) - min(xs) if xs else 0.0

    @property
    def span_y(self) -> float:
        ys = self._gather("y1", "y2")
        return max(ys) - min(ys) if ys else 0.0

    @property
    def span_z(self) -> float:
        zs = self._gather("z1", "z2")
        return max(zs) - min(zs) if zs else 0.0

    @property
    def dominant_span(self) -> float:
        """The largest span across any axis."""
        return max(self.span_x, self.span_y, self.span_z)

    @property
    def element_direction(self) -> tuple[float, float, float]:
        """Unit vector along the dominant axis of this wire group."""
        dx, dy, dz = 0.0, 0.0, 0.0
        for w in self.wires:
            lp = w.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if all(isinstance(v, (int, float)) for v in coords):
                x1, y1, z1, x2, y2, z2 = coords
                dx += x2 - x1
                dy += y2 - y1
                dz += z2 - z1
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        return (dx/mag, dy/mag, dz/mag) if mag > 0 else (0, 0, 0)


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
        self.all_frequencies: list[float] = []
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

        # Merge wire groups that share endpoints into connected components.
        # Handles stepped-diameter elements (collinear, different radii) and
        # multi-wire elements like half-square/quad (non-collinear U/loop shapes).
        self._merge_connected_wire_groups()
        self.n_wire_groups = len(self.wire_groups)

        # Frequency — collect all FR entries
        for card in self.parsed.cards:
            if card.card_type == "FR":
                freq = card.labeled_params.get("freq")
                n_freq = card.labeled_params.get("nFreq")
                freq_step = card.labeled_params.get("freqStep")
                freq_type = card.labeled_params.get("freqType")
                if isinstance(freq, (int, float)) and freq > 0:
                    f = float(freq)
                    if self.frequency is None:
                        self.frequency = f
                        self.wavelength = 299.792458 / f
                    self.all_frequencies.append(f)
                    # Expand frequency sweep steps
                    if isinstance(n_freq, int) and n_freq > 1 and isinstance(freq_step, (int, float)) and freq_step > 0:
                        for i in range(1, n_freq):
                            if isinstance(freq_type, int) and freq_type == 1:
                                self.all_frequencies.append(f * (freq_step ** i))
                            else:
                                self.all_frequencies.append(f + freq_step * i)

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

    def _merge_connected_wire_groups(self) -> None:
        """Merge wire groups that share endpoints into connected components.

        Handles stepped-diameter elements (collinear wires, different radii per
        section) and multi-wire element shapes like half-square or quad loops
        where wires meet at angles.  Uses union-find on endpoint proximity.
        """
        n = len(self.wire_groups)
        if n < 2:
            return

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def _wire_endpoints(group: _WireGroup) -> list[tuple[float, float, float]]:
            pts: list[tuple[float, float, float]] = []
            for w in group.wires:
                lp = w.labeled_params
                coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
                if all(isinstance(v, (int, float)) for v in coords):
                    pts.append((coords[0], coords[1], coords[2]))
                    pts.append((coords[3], coords[4], coords[5]))
            return pts

        eps = [_wire_endpoints(g) for g in self.wire_groups]

        tol = 1e-4
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                # Check connected (any shared endpoint within tolerance)
                connected = any(
                    all(abs(a - b) < tol for a, b in zip(pi, pj))
                    for pi in eps[i]
                    for pj in eps[j]
                )
                if connected:
                    union(i, j)

        # Rebuild merged groups
        merged: dict[int, _WireGroup] = {}
        for i in range(n):
            root = find(i)
            if root not in merged:
                merged[root] = _WireGroup(
                    tag=self.wire_groups[root].tag,
                    wires=list(self.wire_groups[root].wires),
                )
            elif i != root:
                merged[root].wires.extend(self.wire_groups[i].wires)

        self.wire_groups = list(merged.values())


# ---------------------------------------------------------------------------
# Individual classifiers
# ---------------------------------------------------------------------------

_KEYWORD_MAP: list[tuple[str, list[str], list[str]]] = [
    ("yagi", ["yagi", "yagi-uda"], []),
    ("quagi", ["quagi", "quad-yagi", "quad yagi"], []),
    ("dipole", ["dipole", "half-wave dipole", "folded dipole", "half wave"], ["inverted"]),
    ("inverted_v", ["inverted v", "inv-v", "inverted-v", "inv v"], []),
    ("vertical", ["vertical", "ground plane", "groundplane", "quarter-wave", "quarter wave", "1/4 wave"], []),
    ("delta_loop", ["delta", "delta loop"], []),
    ("loop", ["loop", "full wave loop"], ["log", "magnetic", "delta"]),
    ("quad", ["quad", "cubical quad"], ["quagi"]),
    ("hexbeam", ["hexbeam", "hex beam", "hex-beam", "hex ", " hex"], []),
    ("lpda", ["lpda", "log periodic", "log-periodic", "log cell", "log-cell"], []),
    ("phased_array", ["phased", "endfire", "broadside", "curtain"], ["bobtail"]),
    ("helix", ["helix", "helical"], []),
    ("collinear", ["collinear", "coco", "coaxial collinear"], []),
    ("moxon", ["moxon"], []),
    ("v_beam", ["v-beam", "vbeam", "v beam", "vee beam", "multi-vee", "multi vee", "multivee", " vee", "vee "], []),
    ("batwing", ["batwing", "bat wing", "bat-wing"], []),
    ("zigzag", ["zigzag", "zig-zag", "zig zag"], []),
    ("wire_array", ["lazy h", "lazy-h", "sterba", "bruce", "8jk"], []),
    ("bobtail_curtain", ["bobtail", "bobtail curtain"], []),
    ("rhombic", ["rhombic"], []),
    ("beverage", ["beverage"], []),
    ("discone", ["discone", "disc-cone", "biconical"], []),
    ("turnstile", ["turnstile"], []),
    ("fractal", ["fractal", "koch", "sierpinski"], []),
    ("magnetic_loop", ["magnetic loop", "mag loop", "small loop", "magloop"], []),
    ("end_fed", ["end fed", "end-fed", "efhw", "zepp"], []),
    ("j_pole", ["j-pole", "j pole", "jpole", "slim jim"], []),
    ("patch", ["patch", "microstrip"], []),
    ("wire_object", ["wire grid", "wiregrid", "wire-grid"], []),
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


def _classify_wire_object(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Wire-grid object (vehicle, structure) — has geometry but no EX or FR,
    or extremely high wire count indicating a wire-grid mesh model."""
    if result.confidence >= 0.7:
        return
    has_geometry = ctx.n_total_wires > 0 or ctx.has_surface_patch
    has_ex = bool(ctx.ex_tags)
    has_fr = ctx.frequency is not None
    if has_geometry and not has_ex and not has_fr:
        result.antenna_type = "wire_object"
        result.confidence = max(result.confidence, 0.6)
        result.evidence.append("geometry present but no EX/FR cards (wire-grid object)")
    elif has_geometry and not has_ex:
        result.antenna_type = "wire_object"
        result.confidence = max(result.confidence, 0.5)
        result.evidence.append("geometry present but no EX card (wire-grid object)")
    elif has_geometry and ctx.n_total_wires > 500:
        # Extremely high wire count with EX: mesh model (vehicle/ship/horn)
        result.antenna_type = "wire_object"
        result.confidence = max(result.confidence, 0.6)
        result.evidence.append(f"very high wire count ({ctx.n_total_wires}) suggests wire-grid model")


def _classify_loop_quad(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Detect loop and quad antennas by closed-wire topology."""
    if result.confidence >= 0.7:
        return
    # Don't override a Moxon — Moxon elements have bends that can look
    # loop-like to the cycle counter, but the defining tip gap means they
    # are NOT closed loops.
    if result.antenna_type == "moxon":
        return
    if ctx.n_wire_groups < 1:
        return
    # Wire-grid mesh models (vehicles, horns, reflectors) have many wires
    # and incidental topological cycles; real quads max out around 100 wires.
    if ctx.n_total_wires > 120:
        return

    n_loops = _count_wire_loops(ctx)

    if n_loops >= 2:
        result.antenna_type = "quad"
        result.confidence = max(result.confidence, 0.75)
        result.evidence.append(f"{n_loops} closed wire loops detected (quad)")
    elif n_loops == 1:
        result.antenna_type = "loop"
        result.confidence = max(result.confidence, 0.6)
        result.evidence.append("1 closed wire loop detected")


def _count_wire_loops(ctx: _AnalysisContext) -> int:
    """Count independent closed loops using cyclomatic number (E - V + C).

    Builds an endpoint graph from all wires, merges nearby points, and
    computes the cycle rank.  Works regardless of tag assignment or
    loading stubs hanging off the main loops.
    """
    tol = 0.05  # 5cm merge tolerance
    nodes: list[tuple[float, ...]] = []

    def find_node(pt: tuple[float, ...]) -> int:
        for i, n in enumerate(nodes):
            if math.sqrt(sum((a - b)**2 for a, b in zip(pt, n))) < tol:
                return i
        nodes.append(pt)
        return len(nodes) - 1

    edges: set[tuple[int, int]] = set()
    adj: dict[int, set[int]] = {}
    for grp in ctx.wire_groups:
        for w in grp.wires:
            lp = w.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if not all(isinstance(v, (int, float)) for v in coords):
                continue
            ni = find_node(tuple(coords[:3]))
            nj = find_node(tuple(coords[3:]))
            if ni == nj:
                continue
            edge = (min(ni, nj), max(ni, nj))
            if edge not in edges:
                edges.add(edge)
                adj.setdefault(ni, set()).add(nj)
                adj.setdefault(nj, set()).add(ni)

    if not edges:
        return 0

    # Count connected components
    visited: set[int] = set()
    n_components = 0
    for start in adj:
        if start in visited:
            continue
        n_components += 1
        queue = [start]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    queue.append(nb)

    # Cyclomatic number: independent cycles = E - V + C
    n_edges = len(edges)
    n_vertices = len(visited)
    cycles = n_edges - n_vertices + n_components
    return max(cycles, 0)


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


def _free_endpoints(grp: _WireGroup, tol: float = 1e-4) -> list[tuple[float, float, float]]:
    """Endpoints that appear exactly once in a group (dangling tips)."""
    all_pts: list[tuple[float, float, float]] = []
    for w in grp.wires:
        lp = w.labeled_params
        coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
        if not all(isinstance(v, (int, float)) for v in coords):
            continue
        all_pts.append((coords[0], coords[1], coords[2]))
        all_pts.append((coords[3], coords[4], coords[5]))
    free: list[tuple[float, float, float]] = []
    for i, pt in enumerate(all_pts):
        count = sum(
            1 for j, other in enumerate(all_pts)
            if i != j and all(abs(a - b) < tol for a, b in zip(pt, other))
        )
        if count == 0:
            free.append(pt)
    return free


def _find_tip_gaps(groups: list[_WireGroup], tol: float = 1e-4) -> list[float]:
    """Find the distances between free endpoints across wire groups.

    A "free endpoint" is a wire endpoint that is NOT shared with any
    other wire in the same group — i.e. it's a dangling tip, not an
    internal junction.  For a wire Moxon, there are 4 free tips (2 per
    element) and they approach the opposite element's tips across a
    small capacitive gap.

    Works for any number of groups >= 2.  Returns the list of
    inter-group free-endpoint distances, sorted ascending.
    """
    if len(groups) < 2:
        return []

    tips_per_group = [_free_endpoints(g, tol) for g in groups]

    dists: list[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for pa in tips_per_group[i]:
                for pb in tips_per_group[j]:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pa, pb)))
                    dists.append(d)
    dists.sort()
    return dists


def _group_has_bends(group: _WireGroup) -> bool:
    """True if at least one wire in the group is non-collinear with the
    group's dominant direction (i.e. the element bends)."""
    d = group.element_direction
    if d == (0, 0, 0) or len(group.wires) < 2:
        return False
    for w in group.wires:
        lp = w.labeled_params
        coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
        if not all(isinstance(v, (int, float)) for v in coords):
            continue
        x1, y1, z1, x2, y2, z2 = coords
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag > 0:
            dot = abs(d[0] * dx / mag + d[1] * dy / mag + d[2] * dz / mag)
            if dot < 0.9:
                return True
    return False


def _detect_tube_moxon(ctx: _AnalysisContext) -> tuple[bool, list[_WireGroup]]:
    """Detect tube Moxon: driven element split at feedpoint into two
    mirror-symmetric halves, each forming a closed path.

    Tube Moxons (e.g. Ceecom) model bent aluminium tubes with the driven
    element split into two halves at the feed gap.  This gives 3 groups:
    two mirror halves + one reflector.

    Returns (is_tube, logical_elements) where logical_elements merges the
    mirror halves into a synthetic group for downstream checks.
    """
    groups = ctx.wire_groups
    if not 3 <= len(groups) <= 4:
        return False, []

    # Find the boom axis by trying each axis.  The mirror-symmetric halves
    # of the split driven element can have larger centroid spread on the
    # element axis than on the boom axis, so we can't just pick max spread.
    # Instead, try each axis and accept the one that gives exactly 2 clusters
    # (one pair at the same boom position + one singleton for the reflector).
    centroids = [g.centroid for g in groups]

    best: tuple[int, list[list[int]], float] | None = None  # (axis, clusters, tol)
    for axis in (0, 1, 2):
        vals = [c[axis] for c in centroids]
        spread = max(vals) - min(vals)
        if spread < 0.01:
            continue
        tol = spread * 0.15
        cls: list[list[int]] = []
        used = [False] * len(groups)
        for i in range(len(groups)):
            if used[i]:
                continue
            cl = [i]
            used[i] = True
            for j in range(i + 1, len(groups)):
                if not used[j] and abs(vals[i] - vals[j]) < tol:
                    cl.append(j)
                    used[j] = True
            cls.append(cl)
        # Accept if exactly 2 clusters with one pair and one singleton
        if len(cls) == 2:
            sizes = sorted(len(c) for c in cls)
            if sizes == [1, 2]:
                best = (axis, cls, tol)
                break

    if best is None:
        return False, []

    boom_axis, clusters, _ = best

    split_cluster = None
    single_cluster = None
    for cl in clusters:
        if len(cl) == 2:
            split_cluster = cl
        elif len(cl) == 1:
            single_cluster = cl

    if split_cluster is None:
        return False, []

    # Verify the split pair are mirror-symmetric:
    # - similar wire counts and spans
    # - opposite on the non-boom, non-vertical axis (the element axis)
    g_a, g_b = groups[split_cluster[0]], groups[split_cluster[1]]
    # Wire counts within factor of 2
    if max(len(g_a.wires), len(g_b.wires)) > 2 * min(len(g_a.wires), len(g_b.wires)):
        return False, []
    # Spans similar (within 30%)
    span_a, span_b = g_a.dominant_span, g_b.dominant_span
    if span_a > 0 and span_b > 0:
        ratio = max(span_a, span_b) / min(span_a, span_b)
        if ratio > 1.3:
            return False, []

    # Check mirror symmetry on element axis
    elem_axes = [0, 1, 2]
    elem_axes.remove(boom_axis)
    ca, cb = g_a.centroid, g_b.centroid
    # On the element's main span axis, centroids should be on opposite sides
    mirror_found = False
    for ax in elem_axes:
        if abs(ca[ax] + cb[ax]) < 0.5 * max(abs(ca[ax]), abs(cb[ax]), 0.01):
            # Sum near zero → opposite signs → mirror pair
            mirror_found = True
            break
    if not mirror_found:
        return False, []

    # Both groups should have bends (tube bends at corners)
    if not _group_has_bends(g_a) or not _group_has_bends(g_b):
        return False, []

    # Build logical elements: merge the split pair into one synthetic group
    merged = _WireGroup(tag=g_a.tag, wires=list(g_a.wires) + list(g_b.wires))
    if single_cluster is not None:
        reflector = groups[single_cluster[0]]
    else:
        # Both clusters are pairs — merge each
        other = [c for c in clusters if c is not split_cluster][0]
        reflector = _WireGroup(
            tag=groups[other[0]].tag,
            wires=list(groups[other[0]].wires) + list(groups[other[1]].wires),
        )

    return True, [merged, reflector]


def _classify_moxon(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Moxon: 2-element with bent ends and a capacitive tip gap.

    Handles two construction types:
    - **Wire Moxon**: 2 groups, open tip gaps between elements.
    - **Tube Moxon**: 3-4 groups, driven element split at feedpoint
      into mirror-symmetric halves (closed paths modelling metal tubes).
    """
    if result.confidence >= 0.7:
        return
    if not 2 <= ctx.n_wire_groups <= 4:
        return

    is_tube = False
    check_groups = ctx.wire_groups

    # For 3-4 groups, check for tube Moxon (split driven element)
    if ctx.n_wire_groups >= 3:
        is_tube, logical = _detect_tube_moxon(ctx)
        if not is_tube:
            return
        check_groups = logical

    # From here, check_groups has 2 logical elements
    if len(check_groups) != 2:
        return
    wire_counts = [len(g.wires) for g in check_groups]
    if not all(2 <= c <= 20 for c in wire_counts):
        return

    # At least one element must have bends
    if not any(_group_has_bends(g) for g in check_groups):
        return

    # Must have at least one group with 3+ wires (center + 2 tails)
    if not any(len(g.wires) >= 3 for g in check_groups):
        return

    # --- Wire Moxon: tip-gap check (defining feature) ---
    if not is_tube:
        tip_dists = _find_tip_gaps(check_groups)
        dominant = max(g.dominant_span for g in check_groups)

        if tip_dists and dominant > 0:
            gap_threshold = dominant * 0.20
            close_pairs = [d for d in tip_dists if d < gap_threshold]

            if len(close_pairs) >= 2:
                avg_gap = (close_pairs[0] + close_pairs[1]) / 2
                if avg_gap > 0.001:  # > 1mm → open gap, not a loop
                    result.antenna_type = "moxon"
                    result.subtypes.append("wire")
                    result.confidence = max(result.confidence, 0.70)
                    result.evidence.append(
                        f"wire moxon: 2 bent elements with tip gap "
                        f"{avg_gap * 1000:.1f}mm (capacitive coupling)"
                    )
                    return

        # Fallback: bent geometry without measurable tip gap
        result.antenna_type = "moxon"
        result.subtypes.append("wire")
        result.confidence = max(result.confidence, 0.45)
        result.evidence.append("2 element groups with multi-wire bent geometry")
        return

    # --- Tube Moxon: split driven element (closed paths) ---
    # Verify 2-element spacing is consistent with Moxon proportions.
    # The boom spacing should be 0.04–0.20λ and element spans similar.
    c0, c1 = check_groups[0].centroid, check_groups[1].centroid
    boom_spacing = math.sqrt(sum((a - b) ** 2 for a, b in zip(c0, c1)))
    dominant = max(g.dominant_span for g in check_groups)

    if dominant > 0 and boom_spacing > 0:
        spacing_ratio = boom_spacing / dominant
        # Moxon boom spacing is typically 5–40% of element span
        if 0.03 <= spacing_ratio <= 0.50:
            result.antenna_type = "moxon"
            result.subtypes.append("tube")
            result.confidence = max(result.confidence, 0.70)
            result.evidence.append(
                f"tube moxon: split driven element "
                f"({ctx.n_wire_groups} groups), boom spacing "
                f"{boom_spacing:.3f}m ({spacing_ratio:.0%} of span)"
            )
            return

    # Relaxed fallback: structure matches tube pattern but spacing is unusual
    result.antenna_type = "moxon"
    result.subtypes.append("tube")
    result.confidence = max(result.confidence, 0.45)
    result.evidence.append(
        f"tube moxon: split driven element ({ctx.n_wire_groups} groups)"
    )


def _classify_yagi(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Yagi: multiple parallel elements along a boom axis (axis-agnostic)."""
    if result.confidence >= 0.7:
        return
    # A Moxon is a Yagi with bent elements — don't reclassify it
    if result.antenna_type == "moxon":
        return
    if ctx.n_wire_groups < 2:
        return

    # Gather linear elements: single-wire groups or groups that form a line
    linear = [g for g in ctx.wire_groups if g.dominant_span > 0]
    if len(linear) < 2:
        return

    # Compute centroid spread along each axis
    centroids = [g.centroid for g in linear]
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    zs = [c[2] for c in centroids]
    spreads = [
        (max(xs) - min(xs), "X"),
        (max(ys) - min(ys), "Y"),
        (max(zs) - min(zs), "Z"),
    ]
    boom_spread, boom_axis = max(spreads, key=lambda s: s[0])

    if boom_spread <= 0:
        return

    # Check elements are roughly parallel: direction vectors should be similar
    dirs = [g.element_direction for g in linear]
    ref = dirs[0]
    parallel_count = 0
    for d in dirs:
        dot = abs(ref[0]*d[0] + ref[1]*d[1] + ref[2]*d[2])
        if dot > 0.85:  # within ~30 degrees
            parallel_count += 1

    n_elements = len(linear)
    parallel_frac = parallel_count / n_elements

    # For stepped-diameter yagis: many tags share centroids (same element)
    # Group by proximity along boom
    unique_positions = _count_unique_boom_positions(centroids, boom_axis)

    if parallel_frac > 0.7 and unique_positions >= 2:
        # Single excitation + linear parallel elements = yagi
        if unique_positions >= 3:
            result.confidence = max(result.confidence, 0.75)
        else:
            result.confidence = max(result.confidence, 0.55)
        # Loading cards are very common on yagis (element tapering etc)
        has_ld = any(c.card_type == "LD" for c in ctx.parsed.cards)
        if has_ld and len(ctx.ex_tags) == 1:
            result.confidence = min(result.confidence + 0.1, 1.0)
        result.antenna_type = "yagi"
        result.evidence.append(
            f"{unique_positions} elements along {boom_axis} boom, "
            f"{parallel_frac:.0%} parallel ({n_elements} groups)"
        )


def _count_unique_boom_positions(centroids: list[tuple[float, float, float]], axis: str) -> int:
    """Count distinct element positions along the boom axis, merging nearby centroids."""
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    positions = sorted(c[idx] for c in centroids)
    if not positions:
        return 0
    unique = [positions[0]]
    for p in positions[1:]:
        if abs(p - unique[-1]) > 0.001:  # gap > 1mm = different element
            unique.append(p)
    return len(unique)


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
    """Dipole: fallback for 1-2 elements without yagi-like boom."""
    if result.confidence >= 0.7:
        return
    # Don't override a more specific type that already has decent confidence
    if result.antenna_type not in ("unknown",) and result.confidence >= 0.5:
        return
    if ctx.n_wire_groups == 0:
        return

    excited = [g for g in ctx.wire_groups if g.tag in ctx.ex_tags]
    if not excited:
        excited = ctx.wire_groups[:1]

    # Accept any linear element (horizontal or vertical with no ground)
    has_linear = any(g.dominant_span > 0 for g in excited)
    if not has_linear:
        return

    if ctx.n_wire_groups <= 2 and not ctx.has_tl:
        # Check for inverted-V: element with significant Z droop
        for g in excited:
            z_span = g.span_z
            horiz_span = max(g.span_x, g.span_y)
            if z_span > 0 and horiz_span > 0 and z_span / horiz_span > 0.15:
                result.antenna_type = "inverted_v"
                result.confidence = max(result.confidence, 0.55)
                result.evidence.append("single element with significant Z droop (inverted-V)")
                return
        if any(g.is_primarily_horizontal for g in excited):
            result.antenna_type = "dipole"
            result.confidence = max(result.confidence, 0.5)
            result.evidence.append("1-2 horizontal elements, simple feed")
        elif any(g.is_primarily_vertical for g in excited) and not ctx.has_ground:
            result.antenna_type = "dipole"
            result.confidence = max(result.confidence, 0.4)
            result.evidence.append("vertical element without ground (vertical dipole)")


# ---------------------------------------------------------------------------
# Multiband detection
# ---------------------------------------------------------------------------

def _detect_multiband(ctx: _AnalysisContext, result: ClassificationResult) -> None:
    """Detect if the model covers multiple amateur bands."""
    if not ctx.all_frequencies:
        return

    # Collect unique bands from all frequencies
    seen_bands: list[str] = []
    for f in ctx.all_frequencies:
        b = _freq_to_band(f)
        if b and b not in seen_bands:
            seen_bands.append(b)

    result.bands = seen_bands
    result.is_multiband = len(seen_bands) >= 2
    if result.is_multiband:
        result.subtypes.append("multiband")
        result.evidence.append(f"frequencies span {len(seen_bands)} bands: {', '.join(seen_bands)}")


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
