"""Preset tuning exercises for the dashboard Tuning Lab."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .nec_calculators import calc_dipole, calc_hairpin_match, calc_vertical, calc_yagi
from .parser import parse_text
from .simulator import DEFAULT_URL, ImpedanceSweep, RadiationPattern, SWRSweep, SimulationResult, simulate_pattern, simulate_sweep
from .visualizer import extract_geometry


@dataclass(frozen=True)
class TuningControl:
    id: str
    label: str
    min_value: float
    max_value: float
    step: float
    default_value: float
    unit: str
    description: str
    effect: str
    control_type: str = "range"
    reveal_reactance_abs_max: float | None = None
    reveal_control_id: str | None = None
    reveal_control_min_value: float | None = None
    reveal_analysis_flag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "min": self.min_value,
            "max": self.max_value,
            "step": self.step,
            "default": self.default_value,
            "unit": self.unit,
            "description": self.description,
            "effect": self.effect,
            "type": self.control_type,
        }
        if self.reveal_reactance_abs_max is not None:
            payload["reveal_reactance_abs_max"] = self.reveal_reactance_abs_max
        if self.reveal_control_id is not None:
            payload["reveal_control_id"] = self.reveal_control_id
        if self.reveal_control_min_value is not None:
            payload["reveal_control_min_value"] = self.reveal_control_min_value
        if self.reveal_analysis_flag is not None:
            payload["reveal_analysis_flag"] = self.reveal_analysis_flag
        return payload


@dataclass(frozen=True)
class TuningExercise:
    id: str
    title: str
    difficulty: str
    antenna_type: str
    default_pattern_type: str
    target_freq_mhz: float
    summary: str
    goal: str
    starter_values: dict[str, float]
    reference_values: dict[str, float]
    controls: tuple[TuningControl, ...]
    concepts: tuple[str, ...]
    progression_steps: tuple[dict[str, str], ...]
    equations: tuple[dict[str, str], ...]
    reading_notes: tuple[str, ...]
    success_criteria: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "difficulty": self.difficulty,
            "antenna_type": self.antenna_type,
            "default_pattern_type": self.default_pattern_type,
            "target_freq_mhz": self.target_freq_mhz,
            "summary": self.summary,
            "goal": self.goal,
        }

    def detail_dict(self) -> dict[str, Any]:
        return {
            **self.summary_dict(),
            "starter_values": dict(self.starter_values),
            "reference_values": dict(self.reference_values),
            "controls": [control.to_dict() for control in self.controls],
            "concepts": list(self.concepts),
            "progression_steps": list(self.progression_steps),
            "equations": list(self.equations),
            "reading_notes": list(self.reading_notes),
            "success_criteria": list(self.success_criteria),
        }


@dataclass(frozen=True)
class DirectivePatternSummary:
    gain_pattern: RadiationPattern | None = None
    azimuth_pattern: RadiationPattern | None = None


_EQUATIONS: tuple[dict[str, str], ...] = (
    {
        "label": "Feedpoint impedance",
        "expression": "Z = R + jX",
        "explanation": "R is the resistive part. X is the reactive part. When X is near zero, the antenna is near resonance at that frequency.",
    },
    {
        "label": "Impedance magnitude",
        "expression": "|Z| = sqrt(R^2 + X^2)",
        "explanation": "Large reactance can make the feedpoint look badly mismatched even if R is reasonable.",
    },
    {
        "label": "Reflection coefficient",
        "expression": "Gamma = (Z - Z0) / (Z + Z0)",
        "explanation": "This is the mismatch between the antenna and the line impedance Z0, usually 50 ohms.",
    },
    {
        "label": "Standing-wave ratio",
        "expression": "SWR = (1 + |Gamma|) / (1 - |Gamma|)",
        "explanation": "Use SWR as the quick scoreboard, but use R and X to understand what to adjust.",
    },
)


def _exercise_catalog() -> dict[str, TuningExercise]:
    calc_dipole(freq_mhz=14.2)
    calc_vertical(freq_mhz=7.15, n_radials=4, radial_slope_deg=35.0)
    calc_yagi(freq_mhz=28.4, n_elements=3)

    return {
        "dipole-basics": TuningExercise(
            id="dipole-basics",
            title="Dipole Resonance Basics",
            difficulty="Beginner",
            antenna_type="dipole",
            default_pattern_type="azimuth",
            target_freq_mhz=14.2,
            summary="Shift a simple half-wave dipole until the resonant dip lands on the design frequency.",
            goal="Bring X close to 0 ohms at 14.20 MHz and keep SWR comfortably below 2:1.",
            starter_values={"length_scale": 1.055, "height_scale": 0.9, "fine_trim_mm": 0.0},
            reference_values={"length_scale": 1.0, "height_scale": 1.0, "fine_trim_mm": 0.0},
            controls=(
                TuningControl(
                    id="length_scale",
                    label="Element length",
                    min_value=0.92,
                    max_value=1.08,
                    step=0.0025,
                    default_value=1.055,
                    unit="x",
                    description="Scale both dipole legs together.",
                    effect="Longer dipoles lower resonance. Shorter dipoles raise it.",
                ),
                TuningControl(
                    id="height_scale",
                    label="Feed height",
                    min_value=0.7,
                    max_value=1.3,
                    step=0.01,
                    default_value=0.9,
                    unit="x",
                    description="Scale the dipole height above ground.",
                    effect="Height changes feed resistance and SWR curve shape without moving resonance as much as length.",
                ),
                TuningControl(
                    id="fine_trim_mm",
                    label="Fine trim per leg",
                    min_value=-12.0,
                    max_value=12.0,
                    step=1.0,
                    default_value=0.0,
                    unit="mm",
                    description="Apply millimeter-level trimming to each dipole leg once the main tune is close.",
                    effect="Positive trim cuts each leg slightly shorter. Negative trim adds a little length back.",
                    reveal_reactance_abs_max=10.0,
                ),
            ),
            concepts=(
                "Resonance: use the sign of X to tell whether the dipole is electrically long or short.",
                "Feedpoint resistance: R moves with height and nearby ground interaction, not just element length.",
                "Fine trimming: once |X| is small, millimeter cuts can overshoot quickly.",
            ),
            progression_steps=(
                {
                    "title": "Coarse tune first",
                    "detail": "Use the main length control until the dip is close to the target frequency.",
                    "target": "Move X toward zero before trying fine trim.",
                },
                {
                    "title": "Use height as the match shaper",
                    "detail": "Once resonance is close, use feed height to nudge R and the SWR curve shape.",
                    "target": "Treat height as a resistance adjustment, not the first resonance control.",
                },
                {
                    "title": "Finish with millimeter cuts",
                    "detail": "Only trim in millimeter steps once reactance is already small.",
                    "target": "Fine trim unlocks when |X| is within 10 Ω.",
                },
            ),
            equations=_EQUATIONS,
            reading_notes=(
                "If X is positive at the target frequency, the dipole is electrically long there.",
                "If X is negative, the dipole is electrically short and usually needs more length.",
                "R near 50 to 75 ohms is workable for a simple dipole. Height shifts R more than trimming does.",
            ),
            success_criteria=(
                "Make the nearest dip sit close to 14.20 MHz.",
                "Reduce the absolute reactance at 14.20 MHz.",
                "Use R and X together instead of chasing SWR alone.",
            ),
        ),
        "vertical-match": TuningExercise(
            id="vertical-match",
            title="Ground-Plane Match",
            difficulty="Intermediate",
            antenna_type="vertical",
            default_pattern_type="elevation",
            target_freq_mhz=7.15,
            summary="Tune a quarter-wave vertical by balancing radiator length with radial geometry.",
            goal="Move the vertical toward 50 ohms resistive feed impedance while keeping reactance near zero.",
            starter_values={"radiator_scale": 0.92, "radial_scale": 1.0, "radial_slope_deg": 10.0},
            reference_values={"radiator_scale": 1.0, "radial_scale": 1.0, "radial_slope_deg": 35.0},
            controls=(
                TuningControl(
                    id="radiator_scale",
                    label="Radiator length",
                    min_value=0.88,
                    max_value=1.08,
                    step=0.0025,
                    default_value=0.92,
                    unit="x",
                    description="Scale the vertical radiator only.",
                    effect="Longer radiators lower the resonant frequency. Shorter radiators raise it.",
                ),
                TuningControl(
                    id="radial_scale",
                    label="Radial length",
                    min_value=0.85,
                    max_value=1.1,
                    step=0.005,
                    default_value=1.0,
                    unit="x",
                    description="Scale the radial wires together.",
                    effect="Radial length shifts current return paths and slightly changes feed resistance.",
                ),
                TuningControl(
                    id="radial_slope_deg",
                    label="Radial slope",
                    min_value=0.0,
                    max_value=50.0,
                    step=1.0,
                    default_value=10.0,
                    unit="deg",
                    description="Angle the radials down from the feedpoint.",
                    effect="Drooping radials generally raises feed resistance toward 50 ohms.",
                ),
            ),
            concepts=(
                "Ground return path: elevated radials are part of the radiator system, not just mechanical supports.",
                "Match transformation: drooping radials change feed resistance toward 50 ohms.",
                "Height above ground: elevated radial geometry and feedpoint height both shape current distribution.",
            ),
            progression_steps=(
                {
                    "title": "Set resonance with radiator length",
                    "detail": "Use the radiator first to move the dip onto the design frequency.",
                    "target": "Do not start by chasing radial geometry if X is still far from zero.",
                },
                {
                    "title": "Raise R with radial slope",
                    "detail": "Droop the radials after resonance is close to transform feed resistance upward.",
                    "target": "Radial slope is the cleanest lever for moving toward 50 Ω.",
                },
                {
                    "title": "Clean up with radial length",
                    "detail": "Use radial length last for smaller current-return adjustments.",
                    "target": "Expect modest R and SWR shaping, not a full resonance reset.",
                },
            ),
            equations=_EQUATIONS,
            reading_notes=(
                "Verticals often start with R below 50 ohms. Radial slope is one of the cleanest ways to raise it.",
                "Use radiator length to steer resonance first, then use radial slope to refine the match.",
                "A low SWR with large X away from the target frequency can still mean the antenna is tuned to the wrong place.",
            ),
            success_criteria=(
                "Push the dip toward 7.15 MHz.",
                "Raise R closer to 50 ohms by changing radial geometry.",
                "Use the reactance sign to decide whether the radiator is too long or too short.",
            ),
        ),
        "yagi-driven-element": TuningExercise(
            id="yagi-driven-element",
            title="3-Element Yagi Feedpoint",
            difficulty="Advanced",
            antenna_type="yagi",
            default_pattern_type="azimuth",
            target_freq_mhz=28.4,
            summary="Tune the driven element inside a 3-element Yagi while watching how the parasitic elements affect the feedpoint.",
            goal="Center the dip near 28.40 MHz and reduce mismatch without losing sight of the parasitic geometry.",
            starter_values={
                "driven_scale": 0.965,
                "height_scale": 1.0,
                "reflector_enabled": 0.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 0.88,
                "director_enabled": 0.0,
                "director_scale": 1.0,
                "director_spacing_scale": 0.88,
                "hairpin_match_scale": 0.0,
            },
            reference_values={
                "driven_scale": 1.0,
                "height_scale": 1.0,
                "reflector_enabled": 1.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 1.0,
                "director_enabled": 1.0,
                "director_scale": 1.0,
                "director_spacing_scale": 1.0,
                "hairpin_match_scale": 1.0,
            },
            controls=(
                TuningControl(
                    id="driven_scale",
                    label="Driven element",
                    min_value=0.92,
                    max_value=1.06,
                    step=0.0025,
                    default_value=0.965,
                    unit="x",
                    description="Scale the driven element length.",
                    effect="This is the fastest way to move the feedpoint reactance through resonance.",
                ),
                TuningControl(
                    id="height_scale",
                    label="Antenna height",
                    min_value=0.8,
                    max_value=2.2,
                    step=0.02,
                    default_value=1.0,
                    unit="x",
                    description="Scale the boom height above ground starting from a practical low Yagi installation.",
                    effect="Height changes ground coupling, feed resistance, and launch angle. Start above roughly 9 to 10 feet before adding parasitics.",
                ),
                TuningControl(
                    id="reflector_enabled",
                    label="Add reflector",
                    min_value=0.0,
                    max_value=1.0,
                    step=1.0,
                    default_value=0.0,
                    unit="",
                    description="Toggle the reflector into the array after the driven element is close.",
                    effect="This starts the first parasitic-coupling step and unlocks reflector length plus reflector spacing.",
                    control_type="checkbox",
                ),
                TuningControl(
                    id="reflector_scale",
                    label="Reflector length",
                    min_value=0.95,
                    max_value=1.05,
                    step=0.0025,
                    default_value=1.0,
                    unit="x",
                    description="Scale the reflector length.",
                    effect="Reflector changes feed coupling and front-to-back behavior more than raw resonance.",
                    reveal_control_id="reflector_enabled",
                    reveal_control_min_value=0.5,
                ),
                TuningControl(
                    id="reflector_spacing_scale",
                    label="Reflector spacing",
                    min_value=0.8,
                    max_value=1.15,
                    step=0.01,
                    default_value=0.88,
                    unit="x",
                    description="Scale the boom distance from the driven element back to the reflector.",
                    effect="Use reflector spacing after the reflector is present to shape the first coupling step.",
                    reveal_control_id="reflector_enabled",
                    reveal_control_min_value=0.5,
                ),
                TuningControl(
                    id="director_enabled",
                    label="Add director",
                    min_value=0.0,
                    max_value=1.0,
                    step=1.0,
                    default_value=0.0,
                    unit="",
                    description="Toggle the director into the array after the reflector step is under control.",
                    effect="This unlocks director length plus director spacing for the second parasitic stage.",
                    control_type="checkbox",
                    reveal_control_id="reflector_enabled",
                    reveal_control_min_value=0.5,
                ),
                TuningControl(
                    id="director_scale",
                    label="Director length",
                    min_value=0.88,
                    max_value=1.05,
                    step=0.0025,
                    default_value=1.0,
                    unit="x",
                    description="Scale the first director length.",
                    effect="Director trimming changes coupling and can sharpen or flatten the SWR dip.",
                    reveal_control_id="director_enabled",
                    reveal_control_min_value=0.5,
                ),
                TuningControl(
                    id="director_spacing_scale",
                    label="Director spacing",
                    min_value=0.8,
                    max_value=1.15,
                    step=0.01,
                    default_value=0.88,
                    unit="x",
                    description="Scale the boom distance from the driven element forward to the director.",
                    effect="Director spacing is the final parasitic-coupling control before any match network work.",
                    reveal_control_id="director_enabled",
                    reveal_control_min_value=0.5,
                ),
                TuningControl(
                    id="hairpin_match_scale",
                    label="Hairpin match",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.05,
                    default_value=0.0,
                    unit="x",
                    description="Scale the final hairpin-match inductive load after the reflector and director geometry are in range.",
                    effect="This approximates the last feed-match step separately from the element geometry so resonance and matching stay distinct.",
                    reveal_analysis_flag="match_stage_ready",
                ),
            ),
            concepts=(
                "Mutual coupling: reflector and director currents pull the driven element away from its isolated tune.",
                "Element spacing: boom distance changes coupling strength, feed impedance, and pattern shape together.",
                "Height above ground: even a small Yagi changes feedpoint and pattern behavior as it moves farther from ground.",
                "Matching: even when the Yagi is resonant, the coupled feedpoint may still need impedance transformation toward 50 ohms.",
                "Parasitic tuning order: get the driven element close first, then use reflector/director length and spacing to refine the coupled system.",
            ),
            progression_steps=(
                {
                    "title": "1. Tune the driven element first",
                    "detail": "Start with the driven element and a realistic low mounting height so the exercise behaves like a real Yagi over ground.",
                    "target": "Keep the antenna above roughly 9 to 10 ft, then move the driven element toward the reference length before parasitic cleanup.",
                },
                {
                    "title": "2. Keep the reflector longer",
                    "detail": "A practical 3-element Yagi reflector usually starts a little longer than the reference driven element.",
                    "target": "Aim for roughly +2% to +5% versus the reference driven element.",
                },
                {
                    "title": "3. Set reflector spacing before adding the director",
                    "detail": "Once the reflector is on, use its own spacing slider to shape the first coupling step instead of moving both parasitics together.",
                    "target": "Reflector spacing is its own boom distance, not a shared 3-element spacing shortcut.",
                },
                {
                    "title": "4. Keep the director shorter",
                    "detail": "The director typically starts slightly shorter than the reference driven element so it leads in phase.",
                    "target": "Aim for roughly −12% to −2% versus the reference driven element.",
                },
                {
                    "title": "5. Add director spacing as the second boom distance",
                    "detail": "After the director is present, its boom spacing becomes a separate control from the reflector spacing.",
                    "target": "Treat reflector spacing and director spacing as two different coupling levers.",
                },
                {
                    "title": "6. Finish with a hairpin match",
                    "detail": "Once the geometry is doing the right directional job, use the final hairpin control to add the last inductive feed match toward 50 Ω.",
                    "target": "Matching is the finish step after geometry, not the first cure for a mistuned Yagi.",
                },
            ),
            equations=_EQUATIONS,
            reading_notes=(
                "In a Yagi, R and X are influenced by coupling between all elements, not just the driven one.",
                "Tune the driven element first for reactance, then use spacing and parasitics to shape the final match.",
                "A low-reactance feedpoint is not automatically a 50-ohm feedpoint. Resonance and matching are related, but not the same task.",
                "Do not expect a bare split-less Yagi to land exactly on 50 ohms without some compromise or match network.",
                "This lab approximates the hairpin stage with an equivalent inductive feed load so the matching lesson stays stable without a full split-feed geometry model.",
            ),
            success_criteria=(
                "Get the nearest dip onto 28.40 MHz.",
                "Use the driven element to push X through zero.",
                "Observe how parasitic geometry changes R even when resonance stays close.",
            ),
        ),
    }


def _yagi_hairpin_reference(freq_mhz: float) -> dict[str, float]:
    match = calc_hairpin_match(z_antenna=25.0, z_target=50.0, freq_mhz=freq_mhz)
    reactance_ohm = float(match.components.get("reactance_ohms", 0.0))
    inductance_h = 0.0
    if reactance_ohm > 0.0:
        inductance_h = reactance_ohm / (2.0 * math.pi * freq_mhz * 1_000_000.0)
    return {
        "reactance_ohm": reactance_ohm,
        "inductance_h": inductance_h,
        "hairpin_length_m": float(match.components.get("hairpin_length_m", 0.0)),
    }


def list_exercises() -> list[dict[str, Any]]:
    return [exercise.summary_dict() for exercise in _exercise_catalog().values()]


def get_exercise(exercise_id: str) -> dict[str, Any]:
    exercise = _exercise_catalog().get(exercise_id)
    if exercise is None:
        raise KeyError(exercise_id)
    return exercise.detail_dict()


def simulate_exercise(
    exercise_id: str,
    values: dict[str, Any] | None = None,
    *,
    base_url: str = DEFAULT_URL,
) -> dict[str, Any]:
    exercise = _exercise_catalog().get(exercise_id)
    if exercise is None:
        raise KeyError(exercise_id)

    current_values = _normalized_values(exercise, values or {}, exercise.starter_values)
    reference_values = _normalized_values(exercise, exercise.reference_values, exercise.reference_values)
    current_case = _run_case(exercise, current_values, base_url=base_url)
    reference_case = _run_case(exercise, reference_values, base_url=base_url)

    return {
        "exercise": exercise.detail_dict(),
        "current_values": current_values,
        "reference_values": reference_values,
        "current": current_case,
        "reference": reference_case,
    }


def exercise_geometry(exercise_id: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
    exercise = _exercise_catalog().get(exercise_id)
    if exercise is None:
        raise KeyError(exercise_id)

    current_values = _normalized_values(exercise, values or {}, exercise.starter_values)
    geometry = _build_geometry(exercise, current_values)
    nec_text = _build_nec(exercise, geometry)
    parsed = parse_text(nec_text, source=f"{exercise_id}.nec")
    return extract_geometry(parsed)


def exercise_pattern(
    exercise_id: str,
    values: dict[str, Any] | None = None,
    *,
    pattern_type: str = "elevation",
    base_url: str = DEFAULT_URL,
) -> dict[str, Any]:
    exercise = _exercise_catalog().get(exercise_id)
    if exercise is None:
        raise KeyError(exercise_id)

    current_values = _normalized_values(exercise, values or {}, exercise.starter_values)
    geometry = _build_geometry(exercise, current_values)
    nec_text = _build_nec(exercise, geometry)
    result = _simulate_nec_pattern_text(
        f"{exercise_id}.nec",
        nec_text,
        base_url=base_url,
        force_pattern=pattern_type,
    )
    return result.to_dict()


def _normalized_values(
    exercise: TuningExercise,
    overrides: dict[str, Any],
    base_values: dict[str, float],
) -> dict[str, float]:
    values = dict(base_values)
    for control in exercise.controls:
        raw = overrides.get(control.id, values.get(control.id, control.default_value))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(values.get(control.id, control.default_value))
        values[control.id] = max(control.min_value, min(control.max_value, value))
    return values


def _run_case(exercise: TuningExercise, values: dict[str, float], *, base_url: str) -> dict[str, Any]:
    geometry = _build_geometry(exercise, values)
    nec_text = _build_nec(exercise, geometry)
    result = _simulate_nec_text(f"{exercise.id}.nec", nec_text, base_url=base_url)
    payload = result.to_dict()
    directive_pattern = _directive_pattern_summary(exercise, result, nec_text, base_url=base_url)
    payload["analysis"] = _analyse_result(exercise, values, geometry, result, directive_pattern=directive_pattern)
    payload["geometry_cards"] = _geometry_cards(exercise, geometry)
    return payload


def _directive_pattern_summary(
    exercise: TuningExercise,
    result: SimulationResult,
    nec_text: str,
    *,
    base_url: str,
) -> DirectivePatternSummary | None:
    if exercise.antenna_type != "yagi":
        return None
    azimuth_pattern = result.pattern if result.pattern is not None and result.pattern.gain_db else None
    if azimuth_pattern is None:
        directive_result = _simulate_nec_pattern_text(
            f"{exercise.id}.nec",
            nec_text,
            base_url=base_url,
            force_pattern="azimuth",
        )
        if directive_result.ok and directive_result.pattern is not None and directive_result.pattern.gain_db:
            azimuth_pattern = directive_result.pattern

    gain_pattern: RadiationPattern | None = None
    full_result = _simulate_nec_pattern_text(
        f"{exercise.id}.nec",
        nec_text,
        base_url=base_url,
        force_pattern="full",
    )
    if full_result.ok and full_result.pattern is not None and full_result.pattern.gain_db:
        gain_pattern = full_result.pattern

    if gain_pattern is None and azimuth_pattern is None:
        return None
    return DirectivePatternSummary(gain_pattern=gain_pattern or azimuth_pattern, azimuth_pattern=azimuth_pattern)


def _simulate_nec_text(filename: str, nec_text: str, *, base_url: str) -> SimulationResult:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False, mode="w") as handle:
            handle.write(nec_text)
            temp_path = Path(handle.name)
        return simulate_sweep(temp_path, base_url=base_url, n_points=31)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _simulate_nec_pattern_text(
    filename: str,
    nec_text: str,
    *,
    base_url: str,
    force_pattern: str,
) -> SimulationResult:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False, mode="w") as handle:
            handle.write(nec_text)
            temp_path = Path(handle.name)
        return simulate_pattern(temp_path, base_url=base_url, force_pattern=force_pattern)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _build_geometry(exercise: TuningExercise, values: dict[str, float]) -> dict[str, Any]:
    if exercise.id == "dipole-basics":
        calc = calc_dipole(exercise.target_freq_mhz)
        fine_trim_per_leg_m = values.get("fine_trim_mm", 0.0) / 1000.0
        half_length = max(0.05, (calc.dimensions["half_length"] * values["length_scale"]) - fine_trim_per_leg_m)
        height = calc.dimensions["feed_height_suggested"] * values["height_scale"]
        return {
            "half_length_m": half_length,
            "total_length_m": half_length * 2.0,
            "height_m": height,
            "fine_trim_mm": values.get("fine_trim_mm", 0.0),
            "wire_radius_m": 0.001,
        }
    if exercise.id == "vertical-match":
        calc = calc_vertical(exercise.target_freq_mhz, n_radials=4, radial_slope_deg=35.0)
        radiator = calc.dimensions["radiator_length"] * values["radiator_scale"]
        radial = calc.dimensions["radial_length"] * values["radial_scale"]
        return {
            "radiator_length_m": radiator,
            "radial_length_m": radial,
            "radial_slope_deg": values["radial_slope_deg"],
            "wire_radius_m": 0.001,
        }
    if exercise.id == "yagi-driven-element":
        calc = calc_yagi(exercise.target_freq_mhz, n_elements=3)
        hairpin_reference = _yagi_hairpin_reference(exercise.target_freq_mhz)
        base_director = calc.dimensions["director_lengths"][0]
        base_height_m = 3.2
        reflector_enabled = values.get("reflector_enabled", 0.0) >= 0.5
        director_enabled = values.get("director_enabled", 0.0) >= 0.5
        hairpin_match_scale = max(0.0, values.get("hairpin_match_scale", 0.0))
        reflector_spacing = calc.dimensions["re_de_spacing"] * values["reflector_spacing_scale"]
        director_spacing = calc.dimensions["de_director_spacings"][0] * values["director_spacing_scale"]
        return {
            "reflector_enabled": reflector_enabled,
            "director_enabled": director_enabled,
            "reflector_length_m": calc.dimensions["reflector_length"] * values["reflector_scale"],
            "driven_length_m": calc.dimensions["driven_length"] * values["driven_scale"],
            "director_length_m": base_director * values["director_scale"],
            "reflector_spacing_m": reflector_spacing,
            "director_spacing_m": director_spacing,
            "reflector_x_m": -reflector_spacing if reflector_enabled else None,
            "driven_x_m": 0.0,
            "director_x_m": director_spacing if director_enabled else None,
            "height_m": base_height_m * values["height_scale"],
            "hairpin_match_scale": hairpin_match_scale,
            "hairpin_inductance_h": hairpin_reference["inductance_h"] * hairpin_match_scale,
            "hairpin_reactance_ohm": hairpin_reference["reactance_ohm"] * hairpin_match_scale,
            "hairpin_length_m": hairpin_reference["hairpin_length_m"] * hairpin_match_scale,
            "wire_radius_m": 0.003,
        }
    raise KeyError(exercise.id)


def _build_nec(exercise: TuningExercise, geometry: dict[str, Any]) -> str:
    if exercise.id == "dipole-basics":
        return "\n".join([
            "CM Tuning Lab dipole exercise",
            f"GW 1 21 0 {-geometry['half_length_m']:.5f} {geometry['height_m']:.5f} 0 {geometry['half_length_m']:.5f} {geometry['height_m']:.5f} {geometry['wire_radius_m']:.5f}",
            "GE 1",
            "GN 1",
            "EX 0 1 11 0 1 0",
            f"FR 0 1 0 0 {exercise.target_freq_mhz:.5f}",
            "EN",
        ])

    if exercise.id == "vertical-match":
        radial_length = geometry["radial_length_m"]
        slope = math.radians(geometry["radial_slope_deg"])
        horizontal = radial_length * math.cos(slope)
        z_end = -radial_length * math.sin(slope)
        radials = [
            f"GW 2 9 0 0 0 {horizontal:.5f} 0 {z_end:.5f} {geometry['wire_radius_m']:.5f}",
            f"GW 3 9 0 0 0 {-horizontal:.5f} 0 {z_end:.5f} {geometry['wire_radius_m']:.5f}",
            f"GW 4 9 0 0 0 0 {horizontal:.5f} {z_end:.5f} {geometry['wire_radius_m']:.5f}",
            f"GW 5 9 0 0 0 0 {-horizontal:.5f} {z_end:.5f} {geometry['wire_radius_m']:.5f}",
        ]
        return "\n".join([
            "CM Tuning Lab vertical exercise",
            f"GW 1 21 0 0 0 0 0 {geometry['radiator_length_m']:.5f} {geometry['wire_radius_m']:.5f}",
            *radials,
            "GE 0",
            "EX 0 1 1 0 1 0",
            f"FR 0 1 0 0 {exercise.target_freq_mhz:.5f}",
            "EN",
        ])

    if exercise.id == "yagi-driven-element":
        lines = ["CM Tuning Lab 3-element yagi exercise"]
        if geometry["reflector_enabled"]:
            lines.append(
                f"GW 1 31 {geometry['reflector_x_m']:.5f} {-geometry['reflector_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['reflector_x_m']:.5f} {geometry['reflector_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['wire_radius_m']:.5f}"
            )
        lines.append(
            f"GW 2 31 {geometry['driven_x_m']:.5f} {-geometry['driven_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['driven_x_m']:.5f} {geometry['driven_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['wire_radius_m']:.5f}"
        )
        if geometry["director_enabled"]:
            lines.append(
                f"GW 3 31 {geometry['director_x_m']:.5f} {-geometry['director_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['director_x_m']:.5f} {geometry['director_length_m'] / 2.0:.5f} {geometry['height_m']:.5f} {geometry['wire_radius_m']:.5f}"
            )
        lines.extend([
            "GE 1",
            "GN 1",
        ])
        if geometry["hairpin_inductance_h"] > 0.0:
            lines.append(f"LD 0 2 16 16 0 {geometry['hairpin_inductance_h']:.12g} 0")
        lines.extend([
            "EX 0 2 16 0 1 0",
            f"FR 0 1 0 0 {exercise.target_freq_mhz:.5f}",
            "EN",
        ])
        return "\n".join(lines)

    raise KeyError(exercise.id)


def _nearest_impedance_point(impedance: ImpedanceSweep | None, swr: SWRSweep | None, target_freq: float) -> dict[str, float] | None:
    if impedance is None or not impedance.freq_mhz:
        return None
    nearest_index = min(range(len(impedance.freq_mhz)), key=lambda idx: abs(float(impedance.freq_mhz[idx]) - target_freq))
    swr_value = None
    if swr is not None and len(swr.swr) > nearest_index:
        swr_value = swr.swr[nearest_index]
    return {
        "freq_mhz": float(impedance.freq_mhz[nearest_index]),
        "r_ohm": float(impedance.r[nearest_index]),
        "x_ohm": float(impedance.x[nearest_index]),
        "swr": float(swr_value) if swr_value is not None else float("nan"),
    }


def _reactive_state(x_ohm: float) -> str:
    if abs(x_ohm) <= 5.0:
        return "near resonance"
    return "inductive" if x_ohm > 0 else "capacitive"


def _match_quality(swr_value: float) -> str:
    if not math.isfinite(swr_value):
        return "unknown"
    if swr_value <= 1.5:
        return "strong"
    if swr_value <= 2.0:
        return "workable"
    return "rough"


def _guidance_for(exercise: TuningExercise, r_ohm: float, x_ohm: float, resonance_error_mhz: float) -> list[str]:
    guidance: list[str] = []

    if exercise.id == "dipole-basics" and abs(x_ohm) <= 10.0:
        guidance.append("You are within 10 ohms of zero reactance. Use the fine-trim slider for millimeter-level cuts per leg.")

    if x_ohm > 5.0:
        if exercise.id == "dipole-basics":
            guidance.append("Positive X means the dipole is electrically long at the target. Shorten both legs slightly.")
        elif exercise.id == "vertical-match":
            guidance.append("Positive X means the radiator is effectively long. Shorten the radiator or shift resonance upward.")
        else:
            guidance.append("Positive X means the driven system is electrically long. Shorten the driven element before chasing the parasitics.")
    elif x_ohm < -5.0:
        if exercise.id == "dipole-basics":
            guidance.append("Negative X means the dipole is electrically short at the target. Add a little length.")
        elif exercise.id == "vertical-match":
            guidance.append("Negative X means the radiator is short at the target. Lengthen the radiator before matching the feedpoint.")
        else:
            guidance.append("Negative X means the driven element is short at the target. Add a bit of driven-element length.")
    else:
        guidance.append("X is already close to zero. Focus on feed resistance and the shape of the SWR curve.")

    if exercise.id == "vertical-match":
        if r_ohm < 42.0:
            guidance.append("Feed resistance is still low. Increase radial slope to bring R closer to 50 ohms.")
        elif r_ohm > 60.0:
            guidance.append("Feed resistance is high. Flatten the radials slightly or revisit radiator length.")
    elif exercise.id == "yagi-driven-element":
        if r_ohm < 35.0:
            guidance.append("The Yagi feedpoint is tightly coupled. Open the spacing slightly or retune the driven element.")
        elif r_ohm > 65.0:
            guidance.append("Resistance is high for a bare driven element. Tighten the geometry or expect to need a match network.")
    else:
        if r_ohm < 45.0:
            guidance.append("R is low for 50-ohm coax. Height or nearby coupling may need adjustment.")
        elif r_ohm > 75.0:
            guidance.append("R is high. Lowering the dipole slightly often reduces feed resistance.")

    if resonance_error_mhz > 0.15:
        guidance.append("The best dip is above the target frequency. Length usually needs to increase.")
    elif resonance_error_mhz < -0.15:
        guidance.append("The best dip is below the target frequency. Length usually needs to decrease.")
    else:
        guidance.append("The best dip is close to the target. You are in fine-tuning territory now.")

    return guidance


def _yagi_progression_snapshot(exercise: TuningExercise, geometry: dict[str, Any]) -> list[dict[str, str]]:
    reference_geometry = _build_geometry(exercise, dict(exercise.reference_values))
    reference_driven_length = reference_geometry["driven_length_m"]
    reflector_pct = ((geometry["reflector_length_m"] / reference_driven_length) - 1.0) * 100.0
    director_pct = ((geometry["director_length_m"] / reference_driven_length) - 1.0) * 100.0
    driven_pct = ((geometry["driven_length_m"] / reference_driven_length) - 1.0) * 100.0
    reflector_spacing_pct = ((geometry["reflector_spacing_m"] / reference_geometry["reflector_spacing_m"]) - 1.0) * 100.0
    director_spacing_pct = ((geometry["director_spacing_m"] / reference_geometry["director_spacing_m"]) - 1.0) * 100.0
    hairpin_pct = ((geometry["hairpin_match_scale"] / max(reference_geometry["hairpin_match_scale"], 0.01)) - 1.0) * 100.0

    def _fmt(percent: float) -> str:
        return f"{percent:+.1f}%"

    def _status(percent: float, low: float, high: float) -> str:
        if low <= percent <= high:
            return "in-range"
        return "low" if percent < low else "high"

    return [
        {
            "label": "Driven vs reference",
            "value": _fmt(driven_pct),
            "target": "Reference target 0.0%",
            "status": "in-range" if abs(driven_pct) <= 1.0 else ("low" if driven_pct < 0 else "high"),
        },
        {
            "label": "Reflector vs reference driven",
            "value": _fmt(reflector_pct) if geometry["reflector_enabled"] else "off",
            "target": "Target +2% to +5% after enabling",
            "status": _status(reflector_pct, 2.0, 5.0) if geometry["reflector_enabled"] else "pending",
        },
        {
            "label": "Reflector spacing vs reference",
            "value": _fmt(reflector_spacing_pct) if geometry["reflector_enabled"] else "off",
            "target": "Reference target 0.0% after enabling",
            "status": "in-range" if geometry["reflector_enabled"] and abs(reflector_spacing_pct) <= 5.0 else ("pending" if not geometry["reflector_enabled"] else ("low" if reflector_spacing_pct < 0 else "high")),
        },
        {
            "label": "Director vs reference driven",
            "value": _fmt(director_pct) if geometry["director_enabled"] else "off",
            "target": "Target -12% to -2% after enabling",
            "status": _status(director_pct, -12.0, -2.0) if geometry["director_enabled"] else "pending",
        },
        {
            "label": "Director spacing vs reference",
            "value": _fmt(director_spacing_pct) if geometry["director_enabled"] else "off",
            "target": "Reference target 0.0% after enabling",
            "status": "in-range" if geometry["director_enabled"] and abs(director_spacing_pct) <= 5.0 else ("pending" if not geometry["director_enabled"] else ("low" if director_spacing_pct < 0 else "high")),
        },
        {
            "label": "Hairpin match vs reference",
            "value": _fmt(hairpin_pct) if geometry["hairpin_match_scale"] > 0.0 else "off",
            "target": "Reference target 0.0% after geometry is ready",
            "status": "in-range" if geometry["hairpin_match_scale"] > 0.0 and abs(hairpin_pct) <= 15.0 else ("pending" if geometry["hairpin_match_scale"] <= 0.0 else ("low" if hairpin_pct < 0 else "high")),
        },
    ]


def _yagi_next_action(exercise: TuningExercise, geometry: dict[str, Any]) -> dict[str, str]:
    reference_geometry = _build_geometry(exercise, dict(exercise.reference_values))
    height_ft = geometry["height_m"] * 3.28084
    driven_delta_pct = ((geometry["driven_length_m"] / reference_geometry["driven_length_m"]) - 1.0) * 100.0
    reflector_delta_pct = ((geometry["reflector_length_m"] / reference_geometry["driven_length_m"]) - 1.0) * 100.0
    director_delta_pct = ((geometry["director_length_m"] / reference_geometry["driven_length_m"]) - 1.0) * 100.0
    reflector_spacing_delta_pct = ((geometry["reflector_spacing_m"] / reference_geometry["reflector_spacing_m"]) - 1.0) * 100.0
    director_spacing_delta_pct = ((geometry["director_spacing_m"] / reference_geometry["director_spacing_m"]) - 1.0) * 100.0
    hairpin_delta_pct = ((geometry["hairpin_match_scale"] / max(reference_geometry["hairpin_match_scale"], 0.01)) - 1.0) * 100.0

    if height_ft < 9.0:
        return {
            "control_id": "height_scale",
            "label": "Raise antenna height",
            "detail": "Get the boom above roughly 9 to 10 ft so ground interaction is realistic before tuning the parasitics.",
            "tone": "error",
        }
    if abs(driven_delta_pct) > 1.0:
        return {
            "control_id": "driven_scale",
            "label": "Tune driven element",
            "detail": "Move the driven element toward the reference first so reactance is under control before adding parasitics.",
            "tone": "pending",
        }
    if not geometry["reflector_enabled"]:
        return {
            "control_id": "reflector_enabled",
            "label": "Add reflector",
            "detail": "The driven element is close enough. Add the reflector to start the first coupling step.",
            "tone": "error",
        }
    if reflector_delta_pct < 2.0 or reflector_delta_pct > 5.0:
        return {
            "control_id": "reflector_scale",
            "label": "Set reflector length",
            "detail": "Keep the reflector a little longer than the reference driven element, usually about +2% to +5%.",
            "tone": "pending",
        }
    if abs(reflector_spacing_delta_pct) > 5.0:
        return {
            "control_id": "reflector_spacing_scale",
            "label": "Set reflector spacing",
            "detail": "Use the reflector spacing to shape the first parasitic coupling step before adding a director.",
            "tone": "pending",
        }
    if not geometry["director_enabled"]:
        return {
            "control_id": "director_enabled",
            "label": "Add director",
            "detail": "Reflector coupling is in range. Add the director to start the second boom-spacing step.",
            "tone": "error",
        }
    if director_delta_pct < -12.0 or director_delta_pct > -2.0:
        return {
            "control_id": "director_scale",
            "label": "Set director length",
            "detail": "Keep the director slightly shorter than the reference driven element, usually about −12% to −2%.",
            "tone": "pending",
        }
    if abs(director_spacing_delta_pct) > 5.0:
        return {
            "control_id": "director_spacing_scale",
            "label": "Set director spacing",
            "detail": "Treat director spacing as its own coupling lever after the director is present.",
            "tone": "pending",
        }
    if abs(hairpin_delta_pct) > 15.0:
        return {
            "control_id": "hairpin_match_scale",
            "label": "Dial hairpin match",
            "detail": "Geometry is in range. Use the final hairpin control to add the inductive feed transformation toward 50 ohms.",
            "tone": "error" if geometry["hairpin_match_scale"] <= 0.0 else "pending",
        }
    return {
        "control_id": "",
        "label": "Yagi lesson complete",
        "detail": "Geometry and the final hairpin match are both in range. Compare the feedpoint against the reference to see the completed transformation.",
        "tone": "ready",
    }


def _analyse_result(
    exercise: TuningExercise,
    values: dict[str, float],
    geometry: dict[str, Any],
    result: SimulationResult,
    *,
    directive_pattern: DirectivePatternSummary | None = None,
) -> dict[str, Any]:
    target_point = _nearest_impedance_point(result.impedance, result.swr, exercise.target_freq_mhz)
    if target_point is None:
        return {
            "target_freq_mhz": exercise.target_freq_mhz,
            "status": "no-data",
            "guidance": ["Simulation returned no sweep data."],
        }

    resonant_freq = result.swr.resonant_freq if result.swr is not None else None
    resonance_error = float(resonant_freq - exercise.target_freq_mhz) if resonant_freq is not None else math.nan
    r_ohm = target_point["r_ohm"]
    x_ohm = target_point["x_ohm"]
    swr_value = target_point["swr"]
    useful_match = math.isfinite(swr_value) and swr_value <= 1.5 and abs(r_ohm - 50.0) <= 10.0 and abs(x_ohm) <= 5.0
    fine_tuning_ready = exercise.id == "dipole-basics" and abs(x_ohm) <= 10.0
    stop_trimming = (
        exercise.id == "dipole-basics"
        and abs(x_ohm) <= 3.0
        and math.isfinite(swr_value)
        and swr_value <= 1.5
        and math.isfinite(resonance_error)
        and abs(resonance_error) <= 0.05
    )
    progression_snapshot = _yagi_progression_snapshot(exercise, geometry) if exercise.id == "yagi-driven-element" else []
    next_action = _yagi_next_action(exercise, geometry) if exercise.id == "yagi-driven-element" else None
    match_stage_ready = bool(next_action and next_action["control_id"] in {"hairpin_match_scale", ""}) if exercise.id == "yagi-driven-element" else False
    hairpin_useful_match = exercise.id == "yagi-driven-element" and match_stage_ready and useful_match
    gain_pattern = directive_pattern.gain_pattern if directive_pattern is not None else None
    azimuth_pattern = directive_pattern.azimuth_pattern if directive_pattern is not None else None
    directive_gain_dbi = gain_pattern.max_gain if gain_pattern is not None and gain_pattern.gain_db else None
    front_to_back_db = azimuth_pattern.front_to_back if azimuth_pattern is not None else None
    guidance = _guidance_for(exercise, r_ohm, x_ohm, resonance_error if math.isfinite(resonance_error) else 0.0)
    if hairpin_useful_match:
        guidance.insert(0, "The current hairpin setting is already useful: SWR is below 1.5, R is within 10 ohms of 50, and X is within ±5 ohms.")

    return {
        "target_freq_mhz": exercise.target_freq_mhz,
        "sample_freq_mhz": target_point["freq_mhz"],
        "resistance_ohm": r_ohm,
        "reactance_ohm": x_ohm,
        "swr_at_target": swr_value if math.isfinite(swr_value) else None,
        "reactive_state": _reactive_state(x_ohm),
        "match_quality": _match_quality(swr_value),
        "resonant_freq_mhz": resonant_freq,
        "resonance_error_mhz": resonance_error if math.isfinite(resonance_error) else None,
        "useful_match": useful_match,
        "fine_tuning_ready": fine_tuning_ready,
        "stop_trimming": stop_trimming,
        "progression_snapshot": progression_snapshot,
        "next_action": next_action,
        "match_stage_ready": match_stage_ready,
        "hairpin_useful_match": hairpin_useful_match,
        "directive_gain_dbi": directive_gain_dbi if directive_gain_dbi is not None and math.isfinite(directive_gain_dbi) else None,
        "front_to_back_db": front_to_back_db if front_to_back_db is not None and math.isfinite(front_to_back_db) else None,
        "guidance": guidance,
        "control_values": values,
        "geometry": geometry,
    }


def _geometry_cards(exercise: TuningExercise, geometry: dict[str, Any]) -> list[dict[str, str]]:
    def _length_card(label: str, meters: float) -> dict[str, str]:
        return {
            "label": label,
            "value": f"{meters:.2f} m",
            "secondary_value": f"{meters * 3.28084:.2f} ft",
        }

    if exercise.id == "dipole-basics":
        cards = [
            _length_card("Total length", geometry['total_length_m']),
            _length_card("Half length", geometry['half_length_m']),
            _length_card("Height", geometry['height_m']),
        ]
        if abs(float(geometry.get("fine_trim_mm", 0.0))) >= 0.5:
            cards.append({"label": "Fine trim", "value": f"{geometry['fine_trim_mm']:+.0f} mm/leg"})
        return cards
    if exercise.id == "vertical-match":
        return [
            _length_card("Radiator", geometry['radiator_length_m']),
            _length_card("Radials", geometry['radial_length_m']),
            {"label": "Radial slope", "value": f"{geometry['radial_slope_deg']:.0f} deg"},
        ]
    return [
        _length_card("Driven", geometry['driven_length_m']),
        _length_card("Height", geometry['height_m']),
        *([_length_card("Reflector", geometry['reflector_length_m'])] if geometry['reflector_enabled'] else []),
        *([_length_card("Reflector spacing", geometry['reflector_spacing_m'])] if geometry['reflector_enabled'] else []),
        *([_length_card("Director", geometry['director_length_m'])] if geometry['director_enabled'] else []),
        *([_length_card("Director spacing", geometry['director_spacing_m'])] if geometry['director_enabled'] else []),
        *([
            {
                "label": "Hairpin L",
                "value": f"{geometry['hairpin_inductance_h'] * 1_000_000_000.0:.1f} nH",
                "secondary_value": f"{geometry['hairpin_reactance_ohm']:.1f} Ω eq @ {exercise.target_freq_mhz:.2f} MHz",
            }
        ] if geometry['hairpin_inductance_h'] > 0.0 else []),
    ]
