"""
Card-configuration fingerprint engine.

Generates a structural fingerprint for each NEC file based on the card
types present, their counts, and geometric / electrical feature ratios.
Two use cases:
  1. Independent classification signal (card-config alone can distinguish
     many antenna families).
  2. Similarity search — find files with the most similar card structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .parser import ParseResult, NECCard


@dataclass(frozen=True)
class LPDAFit:
    """Reverse-fit summary for LPDA-like element progressions."""

    element_count: int
    boom_axis: str
    order: str
    fitted_tau: float
    fitted_sigma: float
    monotonic_lengths: bool
    monotonic_spacings: bool
    mean_length_error_pct: float
    max_length_error_pct: float
    mean_spacing_error_pct: float
    max_spacing_error_pct: float
    conforms: bool
    element_lengths: list[float] = field(default_factory=list)
    element_spacings: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "boom_axis": self.boom_axis,
            "order": self.order,
            "fitted_tau": round(self.fitted_tau, 4),
            "fitted_sigma": round(self.fitted_sigma, 4),
            "monotonic_lengths": self.monotonic_lengths,
            "monotonic_spacings": self.monotonic_spacings,
            "mean_length_error_pct": round(self.mean_length_error_pct, 2),
            "max_length_error_pct": round(self.max_length_error_pct, 2),
            "mean_spacing_error_pct": round(self.mean_spacing_error_pct, 2),
            "max_spacing_error_pct": round(self.max_spacing_error_pct, 2),
            "conforms": self.conforms,
            "element_lengths": [round(v, 6) for v in self.element_lengths],
            "element_spacings": [round(v, 6) for v in self.element_spacings],
        }


@dataclass(frozen=True)
class Fingerprint:
    """Immutable structural fingerprint of a NEC file."""

    # --- card inventory ---
    card_types: frozenset[str]   # set of card types present
    n_gw: int                     # wire geometry cards
    n_ga: int                     # arc geometry cards
    n_gh: int                     # helix cards
    n_sp: int                     # surface patch cards
    n_tl: int                     # transmission line cards
    n_ld: int                     # loading cards
    n_ex: int                     # excitation cards
    n_nt: int                     # network cards
    n_gm: int                     # move/transform cards
    n_gx: int                     # reflection cards
    n_gr: int                     # cylindrical structure cards
    n_fr: int                     # frequency cards

    # --- tag structure ---
    n_tags: int                   # unique wire tag numbers
    tag_ex_ratio: float           # n_ex / n_tags (multi-feed indicator)
    wires_per_tag: float          # n_gw / n_tags (wires per element)

    # --- ground model ---
    ground_code: int | None       # GN type code (-1..2 or None)

    # --- frequency info ---
    freq_mhz: float | None       # primary frequency
    n_freq_steps: int             # number of frequency steps

    # --- geometry features ---
    has_symmetry: bool            # GM/GX/GR used
    has_loading: bool             # LD cards
    has_network: bool             # TL or NT cards
    has_patch: bool               # SP/SM cards
    has_helix: bool               # GH cards
    has_arc: bool                 # GA cards
    has_taper: bool               # GC cards

    # --- computed ratios ---
    complexity_score: float       # combined structural complexity 0.0-1.0
    feed_complexity: float        # 0 = no feed, 0.5 = single source, 1.0 = multi-source + TL

    @property
    def signature(self) -> str:
        """Compact human-readable signature string.

        Format: ``GW<n>:TAG<n>:EX<n>[:TL<n>][:LD][:GH][:SP][:GN<code>]``
        Only non-zero features are included.
        """
        parts = [
            f"GW{self.n_gw}",
            f"TAG{self.n_tags}",
            f"EX{self.n_ex}",
        ]
        if self.n_tl:
            parts.append(f"TL{self.n_tl}")
        if self.n_ld:
            parts.append("LD")
        if self.n_gh:
            parts.append("GH")
        if self.n_ga:
            parts.append("GA")
        if self.n_sp:
            parts.append("SP")
        if self.n_nt:
            parts.append("NT")
        if self.has_symmetry:
            parts.append("SYM")
        if self.ground_code is not None and self.ground_code >= 0:
            parts.append(f"GN{self.ground_code}")
        return ":".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "signature": self.signature,
            "card_types": sorted(self.card_types),
            "n_gw": self.n_gw,
            "n_ga": self.n_ga,
            "n_gh": self.n_gh,
            "n_sp": self.n_sp,
            "n_tl": self.n_tl,
            "n_ld": self.n_ld,
            "n_ex": self.n_ex,
            "n_nt": self.n_nt,
            "n_gm": self.n_gm,
            "n_gx": self.n_gx,
            "n_gr": self.n_gr,
            "n_fr": self.n_fr,
            "n_tags": self.n_tags,
            "tag_ex_ratio": round(self.tag_ex_ratio, 3),
            "wires_per_tag": round(self.wires_per_tag, 2),
            "ground_code": self.ground_code,
            "freq_mhz": self.freq_mhz,
            "n_freq_steps": self.n_freq_steps,
            "has_symmetry": self.has_symmetry,
            "has_loading": self.has_loading,
            "has_network": self.has_network,
            "has_patch": self.has_patch,
            "has_helix": self.has_helix,
            "has_arc": self.has_arc,
            "has_taper": self.has_taper,
            "complexity_score": round(self.complexity_score, 3),
            "feed_complexity": round(self.feed_complexity, 3),
        }

    def feature_vector(self) -> list[float]:
        """Numeric feature vector for similarity computations.

        Order is stable — see ``FEATURE_NAMES`` for labels.
        Values are log-scaled counts and binary flags to prevent
        large-count features from dominating distance calculations.
        """
        return [
            _log1p(self.n_gw),
            _log1p(self.n_ga),
            _log1p(self.n_gh),
            _log1p(self.n_sp),
            _log1p(self.n_tl),
            _log1p(self.n_ld),
            _log1p(self.n_ex),
            _log1p(self.n_nt),
            _log1p(self.n_tags),
            self.tag_ex_ratio,
            min(self.wires_per_tag, 10.0) / 10.0,  # clamp
            float(self.has_symmetry),
            float(self.has_loading),
            float(self.has_network),
            float(self.has_patch),
            float(self.has_helix),
            float(self.has_arc),
            float(self.has_taper),
            self.complexity_score,
            self.feed_complexity,
            float(self.ground_code is not None and self.ground_code >= 0),
        ]


FEATURE_NAMES: list[str] = [
    "log_gw", "log_ga", "log_gh", "log_sp", "log_tl", "log_ld",
    "log_ex", "log_nt", "log_tags",
    "tag_ex_ratio", "wires_per_tag_norm",
    "has_symmetry", "has_loading", "has_network",
    "has_patch", "has_helix", "has_arc", "has_taper",
    "complexity", "feed_complexity", "has_ground",
]


def _log1p(n: int) -> float:
    """log(1+n) scaling for count features."""
    return math.log1p(n)


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------

def fingerprint(parsed: ParseResult) -> Fingerprint:
    """Generate a structural fingerprint from a parsed NEC file."""
    cards = parsed.cards
    counts: dict[str, int] = {}
    for c in cards:
        if c.card_type not in ("CM", "CE", "SY"):
            counts[c.card_type] = counts.get(c.card_type, 0) + 1

    n_gw = counts.get("GW", 0)
    n_ga = counts.get("GA", 0)
    n_gh = counts.get("GH", 0)
    n_sp = counts.get("SP", 0) + counts.get("SM", 0)
    n_tl = counts.get("TL", 0)
    n_ld = counts.get("LD", 0)
    n_ex = counts.get("EX", 0)
    n_nt = counts.get("NT", 0)
    n_gm = counts.get("GM", 0)
    n_gx = counts.get("GX", 0)
    n_gr = counts.get("GR", 0)
    n_fr = counts.get("FR", 0)

    # Tags
    tags: set[int] = set()
    for c in cards:
        if c.card_type == "GW":
            tag = c.labeled_params.get("tag")
            if isinstance(tag, int):
                tags.add(tag)
    n_tags = len(tags)

    tag_ex_ratio = n_ex / n_tags if n_tags > 0 else 0.0
    wires_per_tag = n_gw / n_tags if n_tags > 0 else 0.0

    # Ground
    ground_code: int | None = None
    for c in cards:
        if c.card_type == "GN":
            gc = c.labeled_params.get("groundType")
            if isinstance(gc, int):
                ground_code = gc
                break

    # Frequency
    freq_mhz: float | None = None
    n_freq_steps = 0
    for c in cards:
        if c.card_type == "FR":
            f = c.labeled_params.get("freq")
            nf = c.labeled_params.get("nFreq")
            if isinstance(f, (int, float)) and f > 0:
                freq_mhz = float(f)
            if isinstance(nf, int):
                n_freq_steps = max(n_freq_steps, nf)

    # Boolean features
    has_symmetry = n_gm > 0 or n_gx > 0 or n_gr > 0
    has_loading = n_ld > 0
    has_network = n_tl > 0 or n_nt > 0
    has_patch = n_sp > 0
    has_helix = n_gh > 0
    has_arc = n_ga > 0
    has_taper = counts.get("GC", 0) > 0

    # Complexity score = normalize total distinct card types and counts
    distinct_types = len(counts)
    total_cards = sum(counts.values())
    complexity_score = min(1.0, (
        0.3 * min(distinct_types / 10, 1.0) +
        0.3 * min(total_cards / 100, 1.0) +
        0.15 * min(n_tags / 20, 1.0) +
        0.10 * float(has_symmetry) +
        0.10 * float(has_network) +
        0.05 * float(has_taper)
    ))

    # Feed complexity
    if n_ex == 0:
        feed_complexity = 0.0
    elif n_ex == 1 and not has_network:
        feed_complexity = 0.3
    elif n_ex == 1:
        feed_complexity = 0.5
    elif n_ex <= 3:
        feed_complexity = 0.7
    else:
        feed_complexity = 1.0

    return Fingerprint(
        card_types=frozenset(counts.keys()),
        n_gw=n_gw, n_ga=n_ga, n_gh=n_gh, n_sp=n_sp,
        n_tl=n_tl, n_ld=n_ld, n_ex=n_ex, n_nt=n_nt,
        n_gm=n_gm, n_gx=n_gx, n_gr=n_gr, n_fr=n_fr,
        n_tags=n_tags,
        tag_ex_ratio=tag_ex_ratio,
        wires_per_tag=wires_per_tag,
        ground_code=ground_code,
        freq_mhz=freq_mhz,
        n_freq_steps=n_freq_steps,
        has_symmetry=has_symmetry,
        has_loading=has_loading,
        has_network=has_network,
        has_patch=has_patch,
        has_helix=has_helix,
        has_arc=has_arc,
        has_taper=has_taper,
        complexity_score=complexity_score,
        feed_complexity=feed_complexity,
    )


def analyze_lpda_fit(parsed: ParseResult) -> LPDAFit | None:
    r"""Reverse-fit NEC geometry against the LPDA calculator progression.

    This checks whether the file's measured element lengths and boom spacings
    are consistent with a single $\tau$ / $\sigma$ LPDA progression.
    """
    groups = _collect_horizontal_wire_groups(parsed)
    if len(groups) < 4:
        return None

    boom_idx = max(
        range(3),
        key=lambda idx: max(g["centroid"][idx] for g in groups) - min(g["centroid"][idx] for g in groups),
    )
    boom_axis = "xyz"[boom_idx]

    fits = [
        _fit_lpda_direction(groups, boom_idx, boom_axis, reverse=False),
        _fit_lpda_direction(groups, boom_idx, boom_axis, reverse=True),
    ]
    fits = [fit for fit in fits if fit is not None]
    if not fits:
        return None
    return min(fits, key=_lpda_fit_score)


def _collect_horizontal_wire_groups(parsed: ParseResult) -> list[dict[str, Any]]:
    tag_map: dict[int, list[NECCard]] = {}
    for card in parsed.cards:
        if card.card_type != "GW":
            continue
        tag = card.labeled_params.get("tag")
        if isinstance(tag, int):
            tag_map.setdefault(tag, []).append(card)

    groups: list[dict[str, Any]] = []
    for tag, wires in sorted(tag_map.items()):
        total_length = 0.0
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        n_midpoints = 0
        horiz = 0.0
        vert = 0.0
        for wire in wires:
            lp = wire.labeled_params
            coords = [lp.get(k) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if not all(isinstance(v, (int, float)) for v in coords):
                continue
            x1, y1, z1, x2, y2, z2 = [float(v) for v in coords]
            xs.extend((x1, x2))
            ys.extend((y1, y2))
            zs.extend((z1, z2))
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            sum_x += (x1 + x2) / 2.0
            sum_y += (y1 + y2) / 2.0
            sum_z += (z1 + z2) / 2.0
            n_midpoints += 1
            horiz += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            vert += abs(z2 - z1)
        if total_length <= 0 or n_midpoints == 0:
            continue
        if horiz <= vert * 1.5:
            continue
        groups.append(
            {
                "tag": tag,
                "length": total_length,
                "centroid": (sum_x / n_midpoints, sum_y / n_midpoints, sum_z / n_midpoints),
                "span": (
                    (max(xs) - min(xs)) if xs else 0.0,
                    (max(ys) - min(ys)) if ys else 0.0,
                    (max(zs) - min(zs)) if zs else 0.0,
                ),
            }
        )
    return groups


def _fit_lpda_direction(
    groups: list[dict[str, Any]],
    boom_idx: int,
    boom_axis: str,
    *,
    reverse: bool,
) -> LPDAFit | None:
    ordered = sorted(groups, key=lambda group: group["centroid"][boom_idx], reverse=reverse)
    lengths = [float(group["length"]) for group in ordered]
    positions = [float(group["centroid"][boom_idx]) for group in ordered]
    spacings = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    if len(lengths) < 4 or any(length <= 0 for length in lengths) or any(spacing <= 0 for spacing in spacings):
        return None

    growth_terms = [
        math.log(lengths[index] / lengths[0]) / index
        for index in range(1, len(lengths))
        if lengths[index] > 0 and lengths[0] > 0
    ]
    if not growth_terms:
        return None
    gamma = sum(growth_terms) / len(growth_terms)
    tau = math.exp(-gamma)
    if not math.isfinite(tau) or tau <= 0:
        return None

    ideal_lengths = [lengths[0] / (tau ** index) for index in range(len(lengths))]
    sigma_terms = [
        spacings[index] / (2.0 * ideal_lengths[index + 1])
        for index in range(len(spacings))
        if ideal_lengths[index + 1] > 0
    ]
    if not sigma_terms:
        return None
    sigma = sum(sigma_terms) / len(sigma_terms)
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    ideal_spacings = [2.0 * sigma * ideal_lengths[index + 1] for index in range(len(spacings))]

    length_errors = [abs(actual - expected) / expected for actual, expected in zip(lengths, ideal_lengths) if expected > 0]
    spacing_errors = [abs(actual - expected) / expected for actual, expected in zip(spacings, ideal_spacings) if expected > 0]
    if not length_errors or not spacing_errors:
        return None

    monotonic_lengths = all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1))
    monotonic_spacings = all(spacings[i] <= spacings[i + 1] for i in range(len(spacings) - 1))
    conforms = (
        monotonic_lengths
        and monotonic_spacings
        and 0.80 <= tau <= 0.98
        and 0.02 <= sigma <= 0.12
        and (sum(length_errors) / len(length_errors)) <= 0.08
        and max(length_errors) <= 0.15
        and (sum(spacing_errors) / len(spacing_errors)) <= 0.12
        and max(spacing_errors) <= 0.20
    )

    return LPDAFit(
        element_count=len(lengths),
        boom_axis=boom_axis,
        order="descending" if reverse else "ascending",
        fitted_tau=tau,
        fitted_sigma=sigma,
        monotonic_lengths=monotonic_lengths,
        monotonic_spacings=monotonic_spacings,
        mean_length_error_pct=100.0 * (sum(length_errors) / len(length_errors)),
        max_length_error_pct=100.0 * max(length_errors),
        mean_spacing_error_pct=100.0 * (sum(spacing_errors) / len(spacing_errors)),
        max_spacing_error_pct=100.0 * max(spacing_errors),
        conforms=conforms,
        element_lengths=lengths,
        element_spacings=spacings,
    )


def _lpda_fit_score(fit: LPDAFit) -> float:
    penalty = 0.0
    if not fit.monotonic_lengths:
        penalty += 1000.0
    if not fit.monotonic_spacings:
        penalty += 500.0
    if not 0.80 <= fit.fitted_tau <= 0.98:
        penalty += 100.0
    if not 0.02 <= fit.fitted_sigma <= 0.12:
        penalty += 100.0
    return penalty + fit.mean_length_error_pct + fit.mean_spacing_error_pct


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def similarity(a: Fingerprint, b: Fingerprint) -> float:
    """Cosine similarity between two fingerprints (0.0 .. 1.0)."""
    va = a.feature_vector()
    vb = b.feature_vector()
    dot = sum(x * y for x, y in zip(va, vb))
    mag_a = math.sqrt(sum(x * x for x in va))
    mag_b = math.sqrt(sum(x * x for x in vb))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def find_similar(target: Fingerprint,
                 candidates: list[tuple[str, Fingerprint]],
                 top_n: int = 10,
                 min_similarity: float = 0.0) -> list[tuple[str, float, Fingerprint]]:
    """Find the most similar fingerprints to *target*.

    Parameters
    ----------
    target : Fingerprint
        The reference fingerprint.
    candidates : list of (label, Fingerprint)
        The pool to search.
    top_n : int
        Max results to return.
    min_similarity : float
        Minimum similarity threshold.

    Returns
    -------
    list of (label, similarity_score, fingerprint)
        Sorted by descending similarity.
    """
    scored = []
    for label, fp in candidates:
        sim = similarity(target, fp)
        if sim >= min_similarity:
            scored.append((label, sim, fp))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Archetype profiles — reference fingerprint signatures per antenna type
# ---------------------------------------------------------------------------

@dataclass
class ArchetypeProfile:
    """Statistical profile of a known antenna type's card configuration."""
    antenna_type: str
    sample_count: int
    avg_gw: float
    avg_tags: float
    avg_ex: float
    pct_tl: float           # 0.0 - 1.0
    pct_ld: float
    pct_symmetry: float
    pct_network: float      # TL or NT
    typical_ground: int | None
    avg_complexity: float
    avg_feed_complexity: float


def build_archetype(antenna_type: str,
                    fingerprints: list[Fingerprint]) -> ArchetypeProfile:
    """Build an archetype profile from a collection of same-type fingerprints."""
    n = len(fingerprints)
    if n == 0:
        return ArchetypeProfile(
            antenna_type=antenna_type, sample_count=0,
            avg_gw=0, avg_tags=0, avg_ex=0,
            pct_tl=0, pct_ld=0, pct_symmetry=0, pct_network=0,
            typical_ground=None, avg_complexity=0, avg_feed_complexity=0,
        )

    # Compute statistics
    ground_codes = [fp.ground_code for fp in fingerprints if fp.ground_code is not None]
    typical_ground = max(set(ground_codes), key=ground_codes.count) if ground_codes else None

    return ArchetypeProfile(
        antenna_type=antenna_type,
        sample_count=n,
        avg_gw=sum(fp.n_gw for fp in fingerprints) / n,
        avg_tags=sum(fp.n_tags for fp in fingerprints) / n,
        avg_ex=sum(fp.n_ex for fp in fingerprints) / n,
        pct_tl=sum(1 for fp in fingerprints if fp.n_tl > 0) / n,
        pct_ld=sum(1 for fp in fingerprints if fp.has_loading) / n,
        pct_symmetry=sum(1 for fp in fingerprints if fp.has_symmetry) / n,
        pct_network=sum(1 for fp in fingerprints if fp.has_network) / n,
        typical_ground=typical_ground,
        avg_complexity=sum(fp.complexity_score for fp in fingerprints) / n,
        avg_feed_complexity=sum(fp.feed_complexity for fp in fingerprints) / n,
    )


def classify_by_fingerprint(fp: Fingerprint,
                            archetypes: dict[str, ArchetypeProfile],
                            ) -> list[tuple[str, float]]:
    """Score a fingerprint against archetype profiles.

    Returns a ranked list of (antenna_type, score) pairs, highest first.
    The score is a heuristic match quality (0.0 .. 1.0).
    """
    scores: list[tuple[str, float]] = []

    for atype, arch in archetypes.items():
        if arch.sample_count == 0:
            continue
        score = _archetype_score(fp, arch)
        if score > 0:
            scores.append((atype, score))

    scores.sort(key=lambda x: -x[1])
    return scores


def _archetype_score(fp: Fingerprint, arch: ArchetypeProfile) -> float:
    """Heuristic scoring of fingerprint against an archetype."""
    score = 0.0
    max_score = 0.0

    # Wire count proximity (log-scale)
    max_score += 1.0
    if arch.avg_gw > 0:
        ratio = fp.n_gw / arch.avg_gw if arch.avg_gw > 0 else 0
        score += max(0.0, 1.0 - abs(math.log1p(ratio) - math.log1p(1.0)))

    # Tag count proximity
    max_score += 1.0
    if arch.avg_tags > 0:
        ratio = fp.n_tags / arch.avg_tags if arch.avg_tags > 0 else 0
        score += max(0.0, 1.0 - abs(math.log1p(ratio) - math.log1p(1.0)))

    # EX count proximity
    max_score += 1.0
    if arch.avg_ex > 0:
        diff = abs(fp.n_ex - arch.avg_ex) / max(arch.avg_ex, 1)
        score += max(0.0, 1.0 - diff)

    # TL presence match (important discriminator)
    max_score += 1.5
    has_tl = fp.n_tl > 0
    if arch.pct_tl > 0.8:
        score += 1.5 if has_tl else 0.0
    elif arch.pct_tl < 0.1:
        score += 1.5 if not has_tl else 0.3
    else:
        score += 0.75  # ambiguous — partial credit

    # Loading match
    max_score += 0.5
    if arch.pct_ld > 0.7:
        score += 0.5 if fp.has_loading else 0.1
    elif arch.pct_ld < 0.2:
        score += 0.5 if not fp.has_loading else 0.2
    else:
        score += 0.25

    # Symmetry match
    max_score += 0.5
    if arch.pct_symmetry > 0.3:
        score += 0.5 if fp.has_symmetry else 0.1
    else:
        score += 0.5 if not fp.has_symmetry else 0.2

    # Feed complexity proximity
    max_score += 1.0
    diff = abs(fp.feed_complexity - arch.avg_feed_complexity)
    score += max(0.0, 1.0 - diff * 2)

    # Complexity proximity
    max_score += 0.5
    diff = abs(fp.complexity_score - arch.avg_complexity)
    score += max(0.0, 0.5 - diff)

    return score / max_score if max_score > 0 else 0.0
