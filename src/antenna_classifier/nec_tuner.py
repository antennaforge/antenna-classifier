"""Deterministic element-length tuner for NEC antenna decks.

Operates on JSON decks (the ``{"cards": [...]}`` structure) and runs
fast nec2c evaluations via the Docker solver to converge SWR and
impedance toward goals — without any LLM involvement.

This module is the "Act" step of the OODA loop: once the LLM generates a
structurally valid deck that fails simulation goals, the tuner applies
known electromagnetic knobs (element lengths, spacing) iteratively until
the deck passes or the evaluation budget is exhausted.

Key insight from manual tuning of Cebik phased-driver Yagi:
  - Forward driver length is the strongest SWR lever (controls reactance X)
  - Director length controls feedpoint resistance R
  - Reflector length has moderate effect on resonant frequency
  - Each knob can be converged in ~5-7 nec2c evaluations via bisection

Usage::

    from antenna_classifier.nec_tuner import tune_deck
    tuned, report = tune_deck(deck, antenna_type="yagi", freq_mhz=28.5)
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Max total nec2c evaluations per tuning pass
_MAX_EVALS = 40
# Bisection tolerance: stop when scale factor change < this
_BISECT_TOL = 0.002
# Max bisection iterations per knob
_MAX_BISECT = 8
# SWR threshold — below this we declare success
_SWR_TARGET = 2.0
# Acceptable impedance R range for 50-ohm systems
_R_TARGET = (15.0, 80.0)


@dataclass
class TuneReport:
    """Result of a deterministic tuning pass."""
    success: bool = False
    evals_used: int = 0
    mode: str = "deterministic"
    initial_swr: float = float("inf")
    final_swr: float = float("inf")
    initial_r: float = 0.0
    final_r: float = 0.0
    initial_x: float = 0.0
    final_x: float = 0.0
    final_gain_dbi: float = 0.0
    final_front_to_back_db: float = 0.0
    adjustments: list[str] = field(default_factory=list)
    detail: str = ""


# ── Wire role classification ───────────────────────────────────────

@dataclass
class WireInfo:
    """Parsed wire element from a GW card."""
    card_index: int       # index into deck["cards"]
    tag: int
    segments: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float
    length: float         # computed
    boom_position: float  # position along boom axis


def _classify_wires(deck: dict[str, Any]) -> list[WireInfo]:
    """Extract GW cards and compute boom positions.

    Returns wires sorted by boom position (reflector first, directors last).
    """
    wires: list[WireInfo] = []
    for idx, card in enumerate(deck.get("cards", [])):
        if card.get("type") != "GW":
            continue
        p = card.get("params", [])
        if len(p) < 9:
            continue
        tag, segs = int(p[0]), int(p[1])
        x1, y1, z1 = float(p[2]), float(p[3]), float(p[4])
        x2, y2, z2 = float(p[5]), float(p[6]), float(p[7])
        radius = float(p[8])
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if length < 1e-6:
            continue
        wires.append(WireInfo(
            card_index=idx, tag=tag, segments=segs,
            x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2,
            radius=radius, length=length, boom_position=0.0,
        ))

    if not wires:
        return wires

    # Determine boom axis: the axis along which wire midpoints vary most.
    # For a horizontal Yagi, boom is along X or Y while elements extend
    # along the other axis at a fixed height Z.
    midpoints = [
        ((w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2, (w.z1 + w.z2) / 2)
        for w in wires
    ]
    if len(midpoints) < 2:
        return wires

    ranges = [
        max(m[i] for m in midpoints) - min(m[i] for m in midpoints)
        for i in range(3)
    ]
    boom_axis = ranges.index(max(ranges))

    for w, mid in zip(wires, midpoints):
        w.boom_position = mid[boom_axis]

    wires.sort(key=lambda w: w.boom_position)
    return wires


def _find_excitation_tag(deck: dict[str, Any]) -> int | None:
    """Find the wire tag that is excited (EX card)."""
    for card in deck.get("cards", []):
        if card.get("type") == "EX":
            p = card.get("params", [])
            if len(p) >= 2:
                return int(p[1])
    return None


def _identify_roles(
    wires: list[WireInfo],
    ex_tag: int | None,
) -> dict[str, list[WireInfo]]:
    """Classify wires into roles: reflector, driven, director.

    For multi-driven antennas (phased arrays), all driven elements are
    in the 'driven' list. Wires behind all driven elements are reflectors;
    wires ahead are directors.
    """
    roles: dict[str, list[WireInfo]] = {
        "reflector": [],
        "driven": [],
        "director": [],
    }

    if not wires:
        return roles

    # Find all driven wires — connected via EX or TL to the excitation
    driven_tags: set[int] = set()
    if ex_tag is not None:
        driven_tags.add(ex_tag)

    # Also check TL cards for coupled driven elements
    for card in []:  # placeholder — TL detection below
        pass

    # For now, use excitation tag. If multiple wires share a tag or are
    # connected via TL, they're all driven.
    tl_tags: set[int] = set()
    for card_item in wires:
        pass  # we'll build this from the deck
    # Scan TL cards to find wires coupled to the driven element
    for card in [c for c in [] if c.get("type") == "TL"]:
        pass

    # Better approach: scan deck for TL cards
    deck_cards = []  # We don't have the deck here, but _identify_roles
    # is called from tune_deck which has it. We'll pass it.

    # Simple heuristic: the excited tag + any tags connected via TL
    for w in wires:
        if w.tag in driven_tags:
            roles["driven"].append(w)

    if not roles["driven"]:
        # No excitation found — assume longest wire is driven
        longest = max(wires, key=lambda w: w.length)
        roles["driven"].append(longest)
        driven_tags.add(longest.tag)

    # Everything before the first driven element is reflector;
    # everything after the last driven is director.
    driven_positions = [w.boom_position for w in roles["driven"]]
    min_driven = min(driven_positions)
    max_driven = max(driven_positions)

    for w in wires:
        if w.tag in driven_tags:
            continue
        if w.boom_position < min_driven:
            roles["reflector"].append(w)
        else:
            roles["director"].append(w)

    return roles


def _identify_roles_with_deck(
    wires: list[WireInfo],
    deck: dict[str, Any],
) -> dict[str, list[WireInfo]]:
    """Classify wires using both geometry and TL card info."""
    ex_tag = _find_excitation_tag(deck)
    driven_tags: set[int] = set()
    if ex_tag is not None:
        driven_tags.add(ex_tag)

    # Find TL-coupled wires (phased drivers)
    for card in deck.get("cards", []):
        if card.get("type") == "TL":
            p = card.get("params", [])
            if len(p) >= 4:
                t1, t2 = int(p[0]), int(p[2])
                if t1 in driven_tags:
                    driven_tags.add(t2)
                elif t2 in driven_tags:
                    driven_tags.add(t1)
    # Second pass in case of chain coupling
    for card in deck.get("cards", []):
        if card.get("type") == "TL":
            p = card.get("params", [])
            if len(p) >= 4:
                t1, t2 = int(p[0]), int(p[2])
                if t1 in driven_tags:
                    driven_tags.add(t2)
                elif t2 in driven_tags:
                    driven_tags.add(t1)

    roles: dict[str, list[WireInfo]] = {
        "reflector": [],
        "driven": [],
        "director": [],
    }

    for w in wires:
        if w.tag in driven_tags:
            roles["driven"].append(w)

    if not roles["driven"]:
        longest = max(wires, key=lambda w: w.length)
        roles["driven"].append(longest)
        driven_tags.add(longest.tag)

    driven_positions = [w.boom_position for w in roles["driven"]]
    min_driven = min(driven_positions)
    max_driven = max(driven_positions)

    for w in wires:
        if w.tag in driven_tags:
            continue
        if w.boom_position < min_driven:
            roles["reflector"].append(w)
        else:
            roles["director"].append(w)

    return roles


# ── Deck manipulation ──────────────────────────────────────────────

def _scale_wire(deck: dict[str, Any], wire: WireInfo, factor: float) -> dict[str, Any]:
    """Return a new deck with one wire's length scaled by factor.

    Scaling preserves the wire midpoint and direction — endpoints move
    symmetrically outward (factor > 1) or inward (factor < 1).
    """
    deck = copy.deepcopy(deck)
    card = deck["cards"][wire.card_index]
    p = list(card["params"])

    mx = (p[2] + p[5]) / 2
    my = (p[3] + p[6]) / 2
    mz = (p[4] + p[7]) / 2

    dx = (p[5] - p[2]) / 2
    dy = (p[6] - p[3]) / 2
    dz = (p[7] - p[4]) / 2

    p[2] = mx - dx * factor
    p[3] = my - dy * factor
    p[4] = mz - dz * factor
    p[5] = mx + dx * factor
    p[6] = my + dy * factor
    p[7] = mz + dz * factor

    card["params"] = p
    return deck


def _scale_wire_group(
    deck: dict[str, Any],
    wires: list[WireInfo],
    factor: float,
) -> dict[str, Any]:
    """Scale all wires in a group by the same factor."""
    for w in wires:
        deck = _scale_wire(deck, w, factor)
        # Re-read updated WireInfo isn't needed since we use card_index
    return deck


# ── Simulation helper ──────────────────────────────────────────────

def _sim_deck(deck: dict[str, Any]) -> dict[str, Any] | None:
    """Convert JSON deck to NEC, simulate, return parsed result dict.

    Returns None if solver unavailable. Returns the SimulationResult.to_dict()
    otherwise.
    """
    from .nec_generator import _json_to_nec
    from .simulator import simulate
    nec_text = _json_to_nec(deck)

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".nec", mode="w", delete=False,
        ) as f:
            f.write(nec_text)
            tmp_path = f.name

        result = simulate(tmp_path)
        return result.to_dict() if result else None
    except Exception as exc:
        log.warning("Tuner simulation failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def _extract_swr_r_x(sim: dict[str, Any], freq_mhz: float) -> tuple[float, float, float]:
    """Extract (min_swr, R_at_center, X_at_center) from sim result dict."""
    swr = float("inf")
    r_val = 0.0
    x_val = 0.0

    swr_data = sim.get("swr_sweep", {})
    if swr_data:
        min_swr = swr_data.get("min_swr")
        if min_swr is not None:
            swr = min_swr

    imp_data = sim.get("impedance_sweep", {})
    if imp_data:
        freqs = imp_data.get("freq_mhz", [])
        rs = imp_data.get("r", [])
        xs = imp_data.get("x", [])
        if freqs and rs and xs:
            # Find index closest to target frequency
            idx = min(range(len(freqs)), key=lambda i: abs(freqs[i] - freq_mhz))
            r_val = rs[idx]
            x_val = xs[idx]

    return swr, r_val, x_val


def _extract_gain_fb(sim: dict[str, Any]) -> tuple[float, float]:
    """Extract peak gain and front-to-back ratio from simulation output."""
    pattern = sim.get("radiation_pattern") or {}
    gain = pattern.get("max_gain_dbi")
    front_to_back = pattern.get("front_to_back_db")
    return float(gain or 0.0), float(front_to_back or 0.0)


def _multi_objective_cost(sim: dict[str, Any], freq_mhz: float) -> tuple[float, float, float, float, float, float]:
    """Score a simulation on SWR, feed impedance, gain, and front-to-back."""
    swr, r_val, x_val = _extract_swr_r_x(sim, freq_mhz)
    gain_dbi, front_to_back_db = _extract_gain_fb(sim)
    clipped_swr = min(swr, 20.0) if math.isfinite(swr) else 20.0
    cost = (
        clipped_swr * 14.0
        + abs(r_val - 50.0) * 0.10
        + abs(x_val) * 0.05
        - gain_dbi * 0.9
        - front_to_back_db * 0.35
    )
    return cost, swr, r_val, x_val, gain_dbi, front_to_back_db


def _role_groups_for_scaling(deck: dict[str, Any]) -> list[tuple[str, list[WireInfo]]]:
    """Return tunable wire groups in reflector/driven/director order."""
    wires = _classify_wires(deck)
    if len(wires) < 2:
        return []
    roles = _identify_roles_with_deck(wires, deck)
    groups: list[tuple[str, list[WireInfo]]] = []
    for role in ("reflector", "driven", "director"):
        if roles[role]:
            groups.append((role, roles[role]))
    return groups


def _apply_group_scales(
    deck: dict[str, Any],
    groups: list[tuple[str, list[WireInfo]]],
    scales: tuple[float, ...],
) -> dict[str, Any]:
    """Scale each wire-role group by the corresponding factor."""
    tuned = copy.deepcopy(deck)
    for (_, wires), scale in zip(groups, scales, strict=False):
        if abs(scale - 1.0) < 1e-9:
            continue
        tuned = _scale_wire_group(tuned, wires, scale)
    return tuned


def _clamp_scale(scale: float) -> float:
    return max(0.90, min(1.10, scale))


def _mutate_scales(
    parent_a: tuple[float, ...],
    parent_b: tuple[float, ...],
    rng: random.Random,
) -> tuple[float, ...]:
    child: list[float] = []
    for scale_a, scale_b in zip(parent_a, parent_b, strict=False):
        midpoint = (scale_a + scale_b) / 2.0
        jitter = rng.uniform(-0.02, 0.02)
        child.append(_clamp_scale(midpoint + jitter))
    return tuple(child)


def _tune_deck_ga(
    deck: dict[str, Any],
    antenna_type: str,
    freq_mhz: float,
    *,
    max_evals: int,
    swr_target: float,
    r_range: tuple[float, float],
) -> tuple[dict[str, Any], TuneReport]:
    """Run a small seeded GA after the deterministic tuner establishes a baseline."""
    if max_evals <= 1:
        return tune_deck(
            deck,
            antenna_type,
            freq_mhz,
            max_evals=max_evals,
            swr_target=swr_target,
            r_range=r_range,
            mode="deterministic",
        )

    seed_budget = max(1, min(max_evals - 1, max(1, max_evals // 2)))
    seed_deck, seed_report = tune_deck(
        deck,
        antenna_type,
        freq_mhz,
        max_evals=seed_budget,
        swr_target=swr_target,
        r_range=r_range,
        mode="deterministic",
    )
    seed_report.mode = "ga"
    groups = _role_groups_for_scaling(seed_deck)
    if not groups or seed_report.evals_used >= max_evals:
        if seed_report.detail:
            seed_report.detail += " | GA skipped: insufficient remaining budget or tunable groups"
        else:
            seed_report.detail = "GA skipped: insufficient remaining budget or tunable groups"
        return seed_deck, seed_report

    remaining_budget = max_evals - seed_report.evals_used
    rng = random.Random(f"{antenna_type}:{freq_mhz:.6f}")
    population: list[tuple[float, ...]] = [tuple(1.0 for _ in groups)]
    population.extend(
        tuple(_clamp_scale(1.0 + delta if idx == role_idx else 1.0) for idx in range(len(groups)))
        for role_idx in range(len(groups))
        for delta in (-0.025, 0.025)
    )
    population = population[: max(4, min(remaining_budget, 8))]

    evals_used = seed_report.evals_used
    best_deck = seed_deck
    best_metrics = (
        seed_report.final_swr,
        seed_report.final_r,
        seed_report.final_x,
        seed_report.final_gain_dbi,
        seed_report.final_front_to_back_db,
    )
    best_cost = float("inf")
    visited: set[tuple[float, ...]] = set()

    while remaining_budget > 0 and population:
        scored: list[tuple[float, tuple[float, ...], dict[str, Any], tuple[float, float, float, float, float]]] = []
        next_population: list[tuple[float, ...]] = []
        for scales in population:
            if scales in visited:
                continue
            visited.add(scales)
            candidate = _apply_group_scales(seed_deck, groups, scales)
            sim = _sim_deck(candidate)
            evals_used += 1
            remaining_budget -= 1
            if sim is None or not sim.get("ok", False):
                continue
            cost, swr, r_val, x_val, gain_dbi, front_to_back_db = _multi_objective_cost(sim, freq_mhz)
            metrics = (swr, r_val, x_val, gain_dbi, front_to_back_db)
            scored.append((cost, scales, candidate, metrics))
            if cost < best_cost:
                best_cost = cost
                best_deck = candidate
                best_metrics = metrics
            if remaining_budget <= 0:
                break

        if not scored or remaining_budget <= 0:
            break

        scored.sort(key=lambda item: item[0])
        elites = scored[: min(3, len(scored))]
        next_population.extend(scale for _, scale, _, _ in elites)
        while remaining_budget > 0 and len(next_population) < max(4, len(population)):
            parent_a = elites[rng.randrange(len(elites))][1]
            parent_b = elites[rng.randrange(len(elites))][1]
            next_population.append(_mutate_scales(parent_a, parent_b, rng))
        population = next_population

    final_swr, final_r, final_x, final_gain, final_fb = best_metrics
    seed_report.evals_used = evals_used
    seed_report.final_swr = final_swr
    seed_report.final_r = final_r
    seed_report.final_x = final_x
    seed_report.final_gain_dbi = final_gain
    seed_report.final_front_to_back_db = final_fb
    seed_report.success = final_swr <= swr_target and r_range[0] <= final_r <= r_range[1]
    seed_report.adjustments.append(
        f"GA refinement: SWR={final_swr:.2f}, R={final_r:.1f}, X={final_x:.1f}, "
        f"gain={final_gain:.2f} dBi, F/B={final_fb:.2f} dB"
    )
    seed_report.detail = (
        f"Seeded GA tuning in {evals_used} evaluations: "
        f"SWR {seed_report.initial_swr:.2f} → {final_swr:.2f}, "
        f"gain {final_gain:.2f} dBi, F/B {final_fb:.2f} dB"
    )
    return best_deck, seed_report


# ── Main tuning loop ──────────────────────────────────────────────

def tune_deck(
    deck: dict[str, Any],
    antenna_type: str,
    freq_mhz: float,
    *,
    max_evals: int = _MAX_EVALS,
    swr_target: float = _SWR_TARGET,
    r_range: tuple[float, float] = _R_TARGET,
    mode: str = "deterministic",
) -> tuple[dict[str, Any], TuneReport]:
    """Deterministically tune element lengths to minimize SWR.

    Operates on the JSON deck in-place (returns a new copy).
    Returns (tuned_deck, report).

    Strategy:
    1. Simulate baseline
    2. If SWR already acceptable, return immediately
    3. Identify wire roles (reflector/driven/director)
    4. Apply knobs in priority order:
       a. Scale all driven elements to zero reactance (X → 0)
       b. Scale directors to bring R into range
       c. Fine-tune driven again for final SWR
    """
    if mode == "ga":
        return _tune_deck_ga(
            deck,
            antenna_type,
            freq_mhz,
            max_evals=max_evals,
            swr_target=swr_target,
            r_range=r_range,
        )

    report = TuneReport(mode=mode)
    deck = copy.deepcopy(deck)
    evals = 0

    # --- Baseline simulation ---
    sim = _sim_deck(deck)
    if sim is None:
        report.detail = "Solver unavailable"
        return deck, report
    evals += 1

    if not sim.get("ok", False):
        report.detail = f"Baseline simulation failed: {sim.get('error', 'unknown')}"
        return deck, report

    swr, r_val, x_val = _extract_swr_r_x(sim, freq_mhz)
    gain_dbi, front_to_back_db = _extract_gain_fb(sim)
    report.initial_swr = swr
    report.initial_r = r_val
    report.initial_x = x_val
    report.final_gain_dbi = gain_dbi
    report.final_front_to_back_db = front_to_back_db
    log.info("Tuner baseline: SWR=%.2f, R=%.1f, X=%.1f", swr, r_val, x_val)

    if swr <= swr_target and r_range[0] <= r_val <= r_range[1]:
        report.success = True
        report.final_swr = swr
        report.final_r = r_val
        report.final_x = x_val
        report.final_gain_dbi = gain_dbi
        report.final_front_to_back_db = front_to_back_db
        report.evals_used = evals
        report.detail = "Already meets goals"
        return deck, report

    # --- Classify wires ---
    wires = _classify_wires(deck)
    if len(wires) < 2:
        report.detail = "Too few wires for element tuning"
        report.evals_used = evals
        return deck, report

    roles = _identify_roles_with_deck(wires, deck)
    log.info("Tuner roles: %d reflector, %d driven, %d director",
             len(roles["reflector"]), len(roles["driven"]),
             len(roles["director"]))

    # --- Phase 1: Tune driven elements to minimize |X| (reactance → 0) ---
    if roles["driven"] and abs(x_val) > 5.0 and evals < max_evals:
        deck, phase_evals, new_swr, new_r, new_x = _bisect_group(
            deck, roles["driven"], freq_mhz,
            objective="min_x",
            budget=min(_MAX_BISECT, max_evals - evals),
        )
        evals += phase_evals
        swr, r_val, x_val = new_swr, new_r, new_x
        report.adjustments.append(
            f"Driven length tuning: SWR={swr:.2f}, R={r_val:.1f}, X={x_val:.1f} "
            f"({phase_evals} evals)"
        )
        log.info("Phase 1 (driven X→0): SWR=%.2f, R=%.1f, X=%.1f", swr, r_val, x_val)

    # --- Phase 2: Tune directors to adjust R ---
    if (roles["director"]
            and not (r_range[0] <= r_val <= r_range[1])
            and evals < max_evals):
        deck, phase_evals, new_swr, new_r, new_x = _bisect_group(
            deck, roles["director"], freq_mhz,
            objective="target_r",
            r_target=(r_range[0] + r_range[1]) / 2,
            budget=min(_MAX_BISECT, max_evals - evals),
        )
        evals += phase_evals
        swr, r_val, x_val = new_swr, new_r, new_x
        report.adjustments.append(
            f"Director R tuning: SWR={swr:.2f}, R={r_val:.1f}, X={x_val:.1f} "
            f"({phase_evals} evals)"
        )
        log.info("Phase 2 (director R): SWR=%.2f, R=%.1f, X=%.1f", swr, r_val, x_val)

    # --- Phase 3: Tune reflectors if they exist and SWR still high ---
    if (roles["reflector"]
            and swr > swr_target
            and evals < max_evals):
        deck, phase_evals, new_swr, new_r, new_x = _bisect_group(
            deck, roles["reflector"], freq_mhz,
            objective="min_swr",
            budget=min(_MAX_BISECT, max_evals - evals),
        )
        evals += phase_evals
        swr, r_val, x_val = new_swr, new_r, new_x
        report.adjustments.append(
            f"Reflector SWR tuning: SWR={swr:.2f}, R={r_val:.1f}, X={x_val:.1f} "
            f"({phase_evals} evals)"
        )
        log.info("Phase 3 (reflector SWR): SWR=%.2f, R=%.1f, X=%.1f", swr, r_val, x_val)

    # --- Phase 4: Final driven-element fine-tune for SWR ---
    if roles["driven"] and swr > swr_target and evals < max_evals:
        deck, phase_evals, new_swr, new_r, new_x = _bisect_group(
            deck, roles["driven"], freq_mhz,
            objective="min_swr",
            budget=min(_MAX_BISECT, max_evals - evals),
        )
        evals += phase_evals
        swr, r_val, x_val = new_swr, new_r, new_x
        report.adjustments.append(
            f"Driven SWR fine-tune: SWR={swr:.2f}, R={r_val:.1f}, X={x_val:.1f} "
            f"({phase_evals} evals)"
        )
        log.info("Phase 4 (driven SWR): SWR=%.2f, R=%.1f, X=%.1f", swr, r_val, x_val)

    report.final_swr = swr
    report.final_r = r_val
    report.final_x = x_val
    report.evals_used = evals
    report.success = swr <= swr_target
    report.detail = (
        f"Tuned in {evals} evaluations: "
        f"SWR {report.initial_swr:.2f} → {swr:.2f}, "
        f"R {report.initial_r:.1f} → {r_val:.1f} Ω"
    )
    log.info("Tuner result: %s", report.detail)
    return deck, report


# ── Bisection search on element groups ─────────────────────────────

def _bisect_group(
    deck: dict[str, Any],
    wire_group: list[WireInfo],
    freq_mhz: float,
    *,
    objective: str = "min_swr",
    r_target: float = 40.0,
    budget: int = _MAX_BISECT,
) -> tuple[dict[str, Any], int, float, float, float]:
    """Bisect-search a scaling factor for a wire group to optimize objective.

    Returns (best_deck, evals_used, best_swr, best_r, best_x).

    Objectives:
    - 'min_swr': minimize SWR
    - 'min_x': minimize |X| (reactance magnitude)
    - 'target_r': minimize |R - r_target|
    """
    # First, probe two directions to establish search bracket
    evals = 0
    best_deck = copy.deepcopy(deck)
    best_factor = 1.0

    # Get current metrics
    sim = _sim_deck(deck)
    if sim is None or not sim.get("ok"):
        return deck, 0, float("inf"), 0.0, 0.0
    evals += 1
    current_swr, current_r, current_x = _extract_swr_r_x(sim, freq_mhz)
    best_swr, best_r, best_x = current_swr, current_r, current_x
    best_cost = _cost(current_swr, current_r, current_x, objective, r_target)

    # Determine search direction: try +5% and -5%
    lo_factor, hi_factor = 0.90, 1.10

    # Probe +5%
    probe_up = _scale_wire_group(copy.deepcopy(deck), wire_group, 1.05)
    sim_up = _sim_deck(probe_up)
    evals += 1
    if sim_up and sim_up.get("ok"):
        swr_up, r_up, x_up = _extract_swr_r_x(sim_up, freq_mhz)
        cost_up = _cost(swr_up, r_up, x_up, objective, r_target)
        if cost_up < best_cost:
            best_cost = cost_up
            best_deck = probe_up
            best_swr, best_r, best_x = swr_up, r_up, x_up
            best_factor = 1.05

    if evals >= budget:
        return best_deck, evals, best_swr, best_r, best_x

    # Probe -5%
    probe_dn = _scale_wire_group(copy.deepcopy(deck), wire_group, 0.95)
    sim_dn = _sim_deck(probe_dn)
    evals += 1
    if sim_dn and sim_dn.get("ok"):
        swr_dn, r_dn, x_dn = _extract_swr_r_x(sim_dn, freq_mhz)
        cost_dn = _cost(swr_dn, r_dn, x_dn, objective, r_target)
        if cost_dn < best_cost:
            best_cost = cost_dn
            best_deck = probe_dn
            best_swr, best_r, best_x = swr_dn, r_dn, x_dn
            best_factor = 0.95

    if evals >= budget:
        return best_deck, evals, best_swr, best_r, best_x

    # Determine search bracket based on which direction improved
    if best_factor > 1.0:
        # Lengthening helped — search [1.0, 1.10]
        lo_factor, hi_factor = 1.0, 1.10
    elif best_factor < 1.0:
        # Shortening helped — search [0.90, 1.0]
        lo_factor, hi_factor = 0.90, 1.0
    else:
        # Neither direction helped — search narrow range
        lo_factor, hi_factor = 0.97, 1.03

    # Golden-section search within bracket
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi  # ~0.382

    a, b = lo_factor, hi_factor

    while evals < budget and (b - a) > _BISECT_TOL:
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

        # Evaluate x1
        d1 = _scale_wire_group(copy.deepcopy(deck), wire_group, x1)
        s1 = _sim_deck(d1)
        evals += 1
        if s1 and s1.get("ok"):
            swr1, r1, xr1 = _extract_swr_r_x(s1, freq_mhz)
            c1 = _cost(swr1, r1, xr1, objective, r_target)
            if c1 < best_cost:
                best_cost, best_deck = c1, d1
                best_swr, best_r, best_x = swr1, r1, xr1
                best_factor = x1
        else:
            c1 = float("inf")

        if evals >= budget:
            break

        # Evaluate x2
        d2 = _scale_wire_group(copy.deepcopy(deck), wire_group, x2)
        s2 = _sim_deck(d2)
        evals += 1
        if s2 and s2.get("ok"):
            swr2, r2, xr2 = _extract_swr_r_x(s2, freq_mhz)
            c2 = _cost(swr2, r2, xr2, objective, r_target)
            if c2 < best_cost:
                best_cost, best_deck = c2, d2
                best_swr, best_r, best_x = swr2, r2, xr2
                best_factor = x2
        else:
            c2 = float("inf")

        # Narrow bracket
        if c1 < c2:
            b = x2
        else:
            a = x1

    log.info("Bisect %s: factor=%.4f, cost=%.4f, SWR=%.2f, R=%.1f, X=%.1f (%d evals)",
             objective, best_factor, best_cost, best_swr, best_r, best_x, evals)

    return best_deck, evals, best_swr, best_r, best_x


def _cost(
    swr: float,
    r: float,
    x: float,
    objective: str,
    r_target: float = 40.0,
) -> float:
    """Compute scalar cost for a given objective."""
    if not math.isfinite(swr):
        return 1e6

    if objective == "min_swr":
        return swr
    elif objective == "min_x":
        # Minimize |X|, with small penalty for SWR
        return abs(x) + 0.1 * max(0, swr - 2.0)
    elif objective == "target_r":
        # Minimize distance from target R, with SWR penalty
        return abs(r - r_target) + 0.5 * max(0, swr - 2.0)
    else:
        return swr


# ── Applicability check ────────────────────────────────────────────

# Antenna types where element-length tuning is meaningful
_TUNABLE_TYPES = frozenset({
    "yagi", "moxon", "quad", "quagi", "hexbeam",
    "lpda", "phased_array", "dipole", "inverted_v",
})


def is_tunable(antenna_type: str) -> bool:
    """Check whether deterministic tuning applies to this antenna type."""
    return antenna_type in _TUNABLE_TYPES
