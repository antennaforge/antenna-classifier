"""Type-specific improvement policies for the OODA NEC refinement loop.

Pattern: Type → Policy → Operators

Each antenna type gets a Policy that knows *what* to improve and *how*.
A generic "make it better" instruction to the LLM is replaced by
type-aware, parameterised improvement advice.

A policy exposes:

- ``knobs``    — the tunable parameters for this type
- ``operators`` — mutation strategies the LLM can apply
- ``objectives`` — scoring criteria in priority order
- ``gates``    — pass/fail diagnostics that must clear before accepting

Usage::

    from antenna_classifier.nec_policies import policy_for_type

    policy = policy_for_type("yagi")
    # policy.knobs          → list of KnobSpec
    # policy.operators      → list of Operator
    # policy.objectives     → list of Objective
    # policy.gates          → list of Gate
    # policy.improvement_prompt(feedback) → str  (for the LLM)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnobSpec:
    """A tunable parameter for an antenna type."""
    name: str
    description: str
    unit: str = "m"                 # m, mm, degrees, ohms, count, ratio
    typical_range: tuple[float, float] | None = None
    sensitivity: str = "medium"     # low / medium / high / critical


@dataclass(frozen=True)
class Operator:
    """A named mutation strategy — instruction for the LLM."""
    name: str
    instruction: str               # concise directive for the LLM
    affects: tuple[str, ...] = ()  # which knobs this touches
    risk: str = "low"              # low / medium / high


@dataclass(frozen=True)
class Objective:
    """A scoring objective in priority order."""
    name: str
    description: str
    weight: float = 1.0            # relative weight when scoring
    direction: str = "maximize"    # maximize / minimize / target


@dataclass(frozen=True)
class Gate:
    """A pass/fail diagnostic that must clear."""
    name: str
    description: str
    fatal: bool = True             # if True, failing -> reject entire design


@dataclass
class ImprovementPolicy:
    """Complete improvement policy for one antenna type."""

    antenna_type: str
    knobs: list[KnobSpec] = field(default_factory=list)
    operators: list[Operator] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    # Free-text strategy notes for the LLM
    strategy_notes: str = ""

    def improvement_prompt(
        self,
        feedback: list[str] | None = None,
        score_summary: dict[str, Any] | None = None,
    ) -> str:
        """Build a structured improvement prompt for the LLM.

        This tells the LLM exactly what knobs it can turn, which
        operators to try, and what the scoring objectives are.
        """
        lines: list[str] = []
        lines.append(f"## Improvement guidance for {self.antenna_type}\n")

        if self.strategy_notes:
            lines.append(self.strategy_notes)
            lines.append("")

        # Knobs
        if self.knobs:
            lines.append("### Tunable parameters (knobs)")
            for k in self.knobs:
                rng = f" [{k.typical_range[0]}–{k.typical_range[1]} {k.unit}]" if k.typical_range else ""
                lines.append(f"- **{k.name}** ({k.sensitivity} sensitivity): {k.description}{rng}")
            lines.append("")

        # Operators to try
        if self.operators:
            lines.append("### Improvement operators to try")
            for op in self.operators:
                affects = f" (affects: {', '.join(op.affects)})" if op.affects else ""
                lines.append(f"- **{op.name}** [{op.risk} risk]: {op.instruction}{affects}")
            lines.append("")

        # Objectives
        if self.objectives:
            lines.append("### Scoring objectives (priority order)")
            for obj in self.objectives:
                lines.append(f"- {obj.direction.upper()} {obj.name} (weight {obj.weight}): {obj.description}")
            lines.append("")

        # Gates
        if self.gates:
            lines.append("### Must-pass gates")
            for g in self.gates:
                fatal = " [FATAL]" if g.fatal else ""
                lines.append(f"- {g.name}{fatal}: {g.description}")
            lines.append("")

        # Current feedback
        if feedback:
            lines.append("### Current issues to fix")
            for f in feedback:
                lines.append(f"- {f}")
            lines.append("")

        if score_summary:
            lines.append("### Current scores")
            for k, v in score_summary.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("Apply the most promising operator(s) to address the issues. "
                      "Output ONLY the corrected NEC2 deck.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "antenna_type": self.antenna_type,
            "knobs": [{"name": k.name, "sensitivity": k.sensitivity} for k in self.knobs],
            "operators": [{"name": o.name, "risk": o.risk} for o in self.operators],
            "objectives": [{"name": o.name, "weight": o.weight} for o in self.objectives],
            "gates": [{"name": g.name, "fatal": g.fatal} for g in self.gates],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_POLICIES: dict[str, ImprovementPolicy] = {}


def _register(policy: ImprovementPolicy) -> ImprovementPolicy:
    _POLICIES[policy.antenna_type] = policy
    return policy


# ---------------------------------------------------------------------------
# Common building blocks — shared knobs, operators, objectives, gates
# ---------------------------------------------------------------------------

# Common gates every antenna must pass
_COMMON_GATES = [
    Gate("structural_validity", "NEC deck must parse without errors", fatal=True),
    Gate("type_classification", "Classifier must identify the correct antenna type", fatal=True),
    Gate("swr_at_design_freq", "SWR < 3:1 at design frequency", fatal=True),
    Gate("gain_sanity", "Gain within physically plausible range for this type", fatal=True),
    Gate("no_overlapping_wires", "No GW cards share identical coordinates", fatal=True),
]

# Common objectives
_OBJ_GAIN = Objective("gain", "Higher gain = better signal", weight=1.0, direction="maximize")
_OBJ_FB = Objective("front_to_back", "Higher F/B = better rear rejection", weight=0.8, direction="maximize")
_OBJ_SWR = Objective("swr", "Lower SWR = better match", weight=0.9, direction="minimize")
_OBJ_BW = Objective("bandwidth", "Wider 2:1 SWR bandwidth", weight=0.6, direction="maximize")
_OBJ_COMPACT = Objective("compactness", "Smaller physical size", weight=0.4, direction="minimize")
_OBJ_BUILD = Objective("buildability", "Easier to construct", weight=0.5, direction="maximize")


# ---------------------------------------------------------------------------
# YAGI
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="yagi",
    strategy_notes=(
        "Yagi optimisation is primarily about element length ratios and spacings.\n"
        "Tuning sequence: driven element resonance → reflector F/B → director(s) gain.\n"
        "Do NOT change more than one spacing AND one length per iteration."
    ),
    knobs=[
        KnobSpec("reflector_length", "Reflector element full length", "m",
                  sensitivity="medium"),
        KnobSpec("driven_element_length", "Driven element full length", "m",
                  sensitivity="high"),
        KnobSpec("director_lengths", "Director element length(s)", "m",
                  sensitivity="high"),
        KnobSpec("reflector_spacing", "Reflector-to-DE spacing", "m",
                  typical_range=(0.1, 0.25), sensitivity="medium"),
        KnobSpec("director_spacings", "DE-to-director and director-to-director spacing(s)", "m",
                  typical_range=(0.1, 0.4), sensitivity="high"),
        KnobSpec("element_diameter", "Element tube diameter (affects length correction)", "mm",
                  sensitivity="low"),
        KnobSpec("matching_network", "Gamma/hairpin match dimensions", "mm",
                  sensitivity="medium"),
    ],
    operators=[
        Operator("lengthen_reflector",
                 "Increase reflector length by 1–2% to improve F/B ratio",
                 affects=("reflector_length",), risk="low"),
        Operator("shorten_directors",
                 "Shorten director(s) by 1–2% to shift gain peak to design frequency",
                 affects=("director_lengths",), risk="low"),
        Operator("increase_de_re_spacing",
                 "Widen reflector-DE spacing to improve F/B at expense of some gain",
                 affects=("reflector_spacing",), risk="medium"),
        Operator("tighten_director_spacing",
                 "Reduce director spacing to increase gain (narrower bandwidth)",
                 affects=("director_spacings",), risk="medium"),
        Operator("add_director",
                 "Add another director element in front to increase gain ~1 dB",
                 affects=("director_spacings", "director_lengths"), risk="high"),
        Operator("adjust_driven_for_swr",
                 "Tweak driven element length to bring SWR dip to design frequency",
                 affects=("driven_element_length",), risk="low"),
    ],
    objectives=[
        _OBJ_GAIN,
        _OBJ_FB,
        _OBJ_SWR,
        _OBJ_BW,
        _OBJ_COMPACT,
    ],
    gates=_COMMON_GATES + [
        Gate("fb_minimum", "F/B ratio must be ≥ 10 dB for a useful Yagi"),
        Gate("boom_length", "Boom must not exceed specified max wavelengths"),
    ],
))


# ---------------------------------------------------------------------------
# MOXON
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="moxon",
    strategy_notes=(
        "The Moxon rectangle lives or dies by its TIP GAP.\n"
        "F/B ratio is the primary tuning indicator — peak F/B = correct gap.\n"
        "Do NOT chase gain — the Moxon trades gain for F/B and 50Ω natural match.\n"
        "Adjust tip gap FIRST, then tail lengths, then overall width."
    ),
    knobs=[
        KnobSpec("element_width", "Overall width of the rectangle (dimension A)", "m",
                  sensitivity="medium"),
        KnobSpec("tip_gap", "Gap between driven and reflector element tips (B)", "m",
                  sensitivity="critical"),
        KnobSpec("driven_tail_length", "Driven element tail/return length (C)", "m",
                  sensitivity="medium"),
        KnobSpec("reflector_tail_length", "Reflector tail/return length (D)", "m",
                  sensitivity="medium"),
        KnobSpec("wire_diameter", "Element wire/tube diameter", "mm",
                  sensitivity="low"),
    ],
    operators=[
        Operator("adjust_tip_gap",
                 "Increase or decrease tip gap by 2–5mm to peak F/B ratio. "
                 "Wider gap → lower F/B. Narrower → higher F/B until coupling saturates.",
                 affects=("tip_gap",), risk="high"),
        Operator("adjust_width",
                 "Change overall width to shift the resonant frequency",
                 affects=("element_width",), risk="medium"),
        Operator("balance_tails",
                 "Adjust driven vs reflector tail lengths to fine-tune impedance",
                 affects=("driven_tail_length", "reflector_tail_length"), risk="low"),
        Operator("scale_uniformly",
                 "Scale ALL dimensions by a factor to shift frequency without "
                 "changing the electromagnetic relationships",
                 affects=("element_width", "tip_gap", "driven_tail_length",
                          "reflector_tail_length"), risk="low"),
    ],
    objectives=[
        _OBJ_FB,            # F/B is THE Moxon objective
        _OBJ_SWR,
        Objective("natural_impedance", "Feedpoint R close to 50Ω without matching",
                  weight=0.8, direction="target"),
        _OBJ_COMPACT,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES + [
        Gate("fb_minimum", "F/B must be ≥ 15 dB — the whole point of a Moxon"),
        Gate("impedance_range", "Feedpoint R must be 40–70Ω (natural 50Ω match)"),
        Gate("tip_gap_positive", "Tip gap must be > 0 (tips must not touch!)"),
    ],
))


# ---------------------------------------------------------------------------
# VERTICAL (ground-mounted or elevated radials)
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="vertical",
    strategy_notes=(
        "Vertical performance is dominated by the ground system, not the radiator.\n"
        "More radials ALWAYS helps. Loading reduces efficiency — avoid if possible.\n"
        "The radiator length sets the resonant frequency; radials set the impedance."
    ),
    knobs=[
        KnobSpec("radiator_length", "Vertical radiator total length", "m",
                  sensitivity="medium"),
        KnobSpec("radial_count", "Number of ground radials", "count",
                  typical_range=(4, 120), sensitivity="medium"),
        KnobSpec("radial_length", "Individual radial length", "m",
                  sensitivity="low"),
        KnobSpec("radial_slope", "Radial slope angle from horizontal", "degrees",
                  typical_range=(0, 45), sensitivity="low"),
        KnobSpec("loading", "Loading coil or capacity hat (if shortened)", "ohms",
                  sensitivity="high"),
    ],
    operators=[
        Operator("add_radials",
                 "Increase radial count — diminishing returns past 32 but always positive",
                 affects=("radial_count",), risk="low"),
        Operator("lengthen_radials",
                 "Extend radial lengths toward λ/4 if shorter",
                 affects=("radial_length",), risk="low"),
        Operator("adjust_radiator",
                 "Tweak radiator length to move resonance to design frequency",
                 affects=("radiator_length",), risk="low"),
        Operator("add_top_loading",
                 "Add capacity hat / top-loading wires to shorten radiator while "
                 "maintaining efficiency better than base loading",
                 affects=("radiator_length", "loading"), risk="medium"),
        Operator("slope_radials",
                 "Slope radials downward to raise feedpoint impedance toward 50Ω",
                 affects=("radial_slope",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        Objective("low_angle_radiation", "Low take-off angle for DX", weight=0.8,
                  direction="minimize"),
        _OBJ_GAIN,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES + [
        Gate("ground_model", "Must include ground model (GN card) — verticals need it"),
        Gate("has_radials", "Must have at least 4 radial wires (or use GN 1 perfect ground)"),
    ],
))


# ---------------------------------------------------------------------------
# HEXBEAM
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="hexbeam",
    strategy_notes=(
        "Hexbeam tuning is per-band. Tune highest band first, work downward.\n"
        "Wire-to-wire clearance at spreader tips is the coupling-critical dimension.\n"
        "Do NOT try to optimise all bands simultaneously in one iteration."
    ),
    knobs=[
        KnobSpec("frame_radius", "Spreader length from hub to tip", "m",
                  sensitivity="medium"),
        KnobSpec("wire_lengths", "Wire lengths per band (driven + reflector)", "m",
                  sensitivity="high"),
        KnobSpec("tip_compression", "How much wire is folded back at spreader tips", "m",
                  sensitivity="high"),
        KnobSpec("vertical_spacing", "Height separation between driven and reflector plane", "m",
                  sensitivity="medium"),
    ],
    operators=[
        Operator("adjust_wire_length",
                 "Trim or extend wire length for one band to shift resonance",
                 affects=("wire_lengths",), risk="low"),
        Operator("adjust_compression",
                 "Change tip compression to tune coupling between driven and reflector",
                 affects=("tip_compression",), risk="medium"),
        Operator("adjust_vertical_spacing",
                 "Change the vertical gap between driven and reflector planes",
                 affects=("vertical_spacing",), risk="medium"),
    ],
    objectives=[
        _OBJ_FB,
        _OBJ_SWR,
        _OBJ_GAIN,
        _OBJ_COMPACT,
    ],
    gates=_COMMON_GATES + [
        Gate("wire_clearance", "Wire-to-wire clearance must be > 20mm at tips"),
    ],
))


# ---------------------------------------------------------------------------
# QUAD
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="quad",
    strategy_notes=(
        "Quad loops are relatively tolerant — loop perimeter is the main tuning variable.\n"
        "Reflector = +5% perimeter, director = -5% perimeter from driven.\n"
        "Full-wave loops have ~120Ω impedance — plan for matching."
    ),
    knobs=[
        KnobSpec("driven_perimeter", "Driven loop total perimeter", "m",
                  sensitivity="medium"),
        KnobSpec("reflector_perimeter", "Reflector loop perimeter", "m",
                  sensitivity="low"),
        KnobSpec("director_perimeter", "Director loop perimeter", "m",
                  sensitivity="medium"),
        KnobSpec("element_spacing", "Spacing between loop planes", "m",
                  typical_range=(0.15, 0.25), sensitivity="medium"),
        KnobSpec("feed_position", "Feed point position on driven loop", "degrees",
                  sensitivity="low"),
    ],
    operators=[
        Operator("adjust_perimeters",
                 "Scale loop perimeters to shift resonant frequency",
                 affects=("driven_perimeter", "reflector_perimeter", "director_perimeter"),
                 risk="low"),
        Operator("adjust_spacing",
                 "Change inter-element spacing to trade gain vs F/B",
                 affects=("element_spacing",), risk="medium"),
    ],
    objectives=[
        _OBJ_GAIN,
        _OBJ_FB,
        _OBJ_SWR,
        _OBJ_BW,
    ],
    gates=_COMMON_GATES + [
        Gate("impedance_range", "Feedpoint R should be 100–150Ω (full-wave loop)"),
    ],
))


# ---------------------------------------------------------------------------
# LPDA
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="lpda",
    strategy_notes=(
        "LPDA is a parametric design — τ and σ set everything.  Do NOT tune individual\n"
        "elements; regenerate from τ/σ if the frequency coverage is wrong.\n"
        "Phase reversal between elements is CRITICAL — verify the wiring."
    ),
    knobs=[
        KnobSpec("tau", "Design ratio τ (typical 0.88–0.95)", "ratio",
                  typical_range=(0.85, 0.96), sensitivity="high"),
        KnobSpec("sigma", "Spacing constant σ (typical 0.03–0.08)", "ratio",
                  typical_range=(0.02, 0.10), sensitivity="high"),
        KnobSpec("n_elements", "Number of elements", "count",
                  typical_range=(5, 20), sensitivity="medium"),
        KnobSpec("feed_line_z0", "Feed line impedance", "ohms",
                  typical_range=(50, 300), sensitivity="low"),
    ],
    operators=[
        Operator("increase_tau",
                 "Increase τ for higher gain (longer boom, more elements)",
                 affects=("tau", "n_elements"), risk="medium"),
        Operator("adjust_sigma",
                 "Adjust σ to optimise gain-bandwidth product",
                 affects=("sigma",), risk="medium"),
        Operator("regenerate_from_params",
                 "Recalculate ALL element lengths and spacings from τ, σ, and freq range",
                 affects=("tau", "sigma", "n_elements"), risk="high"),
    ],
    objectives=[
        _OBJ_BW,            # LPDA is a broadband antenna — bandwidth is king
        Objective("swr_flatness", "SWR should be flat across the band, not just at one freq",
                  weight=1.0, direction="minimize"),
        _OBJ_GAIN,
        _OBJ_FB,
    ],
    gates=_COMMON_GATES + [
        Gate("phase_reversal", "Elements must have alternating phase (crossed connections)"),
        Gate("frequency_coverage", "Design must cover the entire requested frequency range"),
    ],
))


# ---------------------------------------------------------------------------
# DIPOLE
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="dipole",
    strategy_notes=(
        "Half-wave dipole is simple — the main knob is total length.\n"
        "Height above ground affects impedance and pattern but is usually constrained."
    ),
    knobs=[
        KnobSpec("total_length", "Dipole total length (tip to tip)", "m",
                  sensitivity="medium"),
        KnobSpec("height", "Height above ground", "m",
                  sensitivity="low"),
        KnobSpec("wire_diameter", "Wire/tube diameter", "mm",
                  sensitivity="low"),
    ],
    operators=[
        Operator("adjust_length",
                 "Trim or extend total length to centre SWR dip on design frequency",
                 affects=("total_length",), risk="low"),
        Operator("adjust_height",
                 "Change height to optimise impedance (73Ω in free space, varies with height)",
                 affects=("height",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# INVERTED V
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="inverted_v",
    strategy_notes=(
        "Inverted-V is a bent dipole — apex height and droop angle are the two main knobs.\n"
        "Steeper droop = lower impedance, wider bandwidth, lower gain."
    ),
    knobs=[
        KnobSpec("total_length", "Overall wire length tip to tip", "m",
                  sensitivity="medium"),
        KnobSpec("apex_height", "Height at centre feed point", "m",
                  sensitivity="low"),
        KnobSpec("droop_angle", "Angle of wire from horizontal", "degrees",
                  typical_range=(30, 60), sensitivity="medium"),
    ],
    operators=[
        Operator("adjust_length",
                 "Change total length to shift resonance",
                 affects=("total_length",), risk="low"),
        Operator("adjust_droop",
                 "Steeper droop → lower Z, wider BW; shallower → more dipole-like",
                 affects=("droop_angle",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# MAGNETIC LOOP
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="magnetic_loop",
    strategy_notes=(
        "Magnetic loop is a HIGH-Q resonator.  Conductor diameter is everything.\n"
        "Bandwidth is inherently narrow (<0.5%) — do not try to widen it.\n"
        "Focus on efficiency (conductor losses) and correct resonance."
    ),
    knobs=[
        KnobSpec("loop_circumference", "Main loop circumference", "m",
                  sensitivity="medium"),
        KnobSpec("conductor_diameter", "Pipe/tube diameter — dominates efficiency", "mm",
                  typical_range=(10, 50), sensitivity="critical"),
        KnobSpec("capacitor_value", "Tuning capacitor value", "pF",
                  sensitivity="critical"),
        KnobSpec("coupling_loop_size", "Feed coupling loop circumference", "m",
                  sensitivity="high"),
    ],
    operators=[
        Operator("increase_conductor",
                 "Use larger diameter conductor to reduce loss resistance",
                 affects=("conductor_diameter",), risk="low"),
        Operator("adjust_circumference",
                 "Change loop size — larger = more efficient but physically bigger",
                 affects=("loop_circumference",), risk="medium"),
        Operator("adjust_coupling",
                 "Change coupling loop size to match feedpoint to 50Ω",
                 affects=("coupling_loop_size",), risk="medium"),
    ],
    objectives=[
        Objective("efficiency", "Radiation efficiency — must overcome conductor loss",
                  weight=1.0, direction="maximize"),
        _OBJ_SWR,
        _OBJ_COMPACT,
    ],
    gates=_COMMON_GATES + [
        Gate("conductor_model", "Must include conductivity (LD 5) for loss modelling"),
        Gate("loop_size", "Circumference must be ≤ 0.25λ (small loop regime)"),
    ],
))


# ---------------------------------------------------------------------------
# PHASED ARRAY
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="phased_array",
    strategy_notes=(
        "Phased arrays require precise amplitude AND phase control.\n"
        "Mutual coupling between elements changes individual impedances — must account for it.\n"
        "Phasing network is the hardest part to build correctly."
    ),
    knobs=[
        KnobSpec("element_spacing", "Inter-element spacing", "m",
                  typical_range=(0.15, 0.5), sensitivity="high"),
        KnobSpec("phase_angles", "Phase angle per element relative to reference", "degrees",
                  sensitivity="critical"),
        KnobSpec("phasing_line_lengths", "Physical lengths of phasing transmission lines", "m",
                  sensitivity="critical"),
        KnobSpec("element_lengths", "Individual element lengths", "m",
                  sensitivity="medium"),
    ],
    operators=[
        Operator("adjust_phasing",
                 "Fine-tune phase angles to steer main beam or deepen nulls",
                 affects=("phase_angles", "phasing_line_lengths"), risk="high"),
        Operator("adjust_spacing",
                 "Change element spacing to trade beamwidth vs grating lobes",
                 affects=("element_spacing",), risk="medium"),
        Operator("adjust_elements",
                 "Tune element lengths to account for mutual impedance",
                 affects=("element_lengths",), risk="medium"),
    ],
    objectives=[
        Objective("pattern_shape", "Achieve target pattern (beam steering or null placement)",
                  weight=1.0, direction="target"),
        _OBJ_GAIN,
        _OBJ_FB,
        _OBJ_SWR,
    ],
    gates=_COMMON_GATES + [
        Gate("phase_coherence", "Phase relationships between elements must be correct"),
        Gate("mutual_coupling", "Model must account for mutual coupling (elements at correct spacing)"),
    ],
))


# ---------------------------------------------------------------------------
# END-FED
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="end_fed",
    strategy_notes=(
        "End-fed half-wave requires a matching transformer (49:1 or 64:1).\n"
        "Multiband operation depends on harmonic relationship.\n"
        "A short counterpoise is essential to prevent RF on the coax."
    ),
    knobs=[
        KnobSpec("wire_length", "Total wire length", "m", sensitivity="medium"),
        KnobSpec("transformer_ratio", "Matching transformer turns ratio", "ratio",
                  sensitivity="medium"),
        KnobSpec("counterpoise_length", "Counterpoise wire length", "m",
                  sensitivity="low"),
    ],
    operators=[
        Operator("adjust_length",
                 "Change wire length to shift resonance or optimise harmonic relationships",
                 affects=("wire_length",), risk="low"),
        Operator("add_counterpoise",
                 "Add or extend counterpoise wire at feedpoint",
                 affects=("counterpoise_length",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# HELIX (axial mode)
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="helix",
    strategy_notes=(
        "Axial-mode helix: diameter ≈ λ/π, pitch ≈ λ/4 per turn.\n"
        "Gain increases with number of turns (~3 dB per doubling).\n"
        "Ground plane must be ≥ 0.6λ diameter."
    ),
    knobs=[
        KnobSpec("helix_diameter", "Helix winding diameter", "m", sensitivity="medium"),
        KnobSpec("turn_pitch", "Spacing between turns", "m", sensitivity="medium"),
        KnobSpec("n_turns", "Number of turns", "count",
                  typical_range=(3, 20), sensitivity="medium"),
        KnobSpec("ground_plane_size", "Ground plane diameter", "m", sensitivity="low"),
    ],
    operators=[
        Operator("add_turns",
                 "Add more turns for higher gain (longer antenna)",
                 affects=("n_turns",), risk="low"),
        Operator("adjust_pitch",
                 "Fine-tune turn spacing for optimal axial-mode operation",
                 affects=("turn_pitch",), risk="low"),
    ],
    objectives=[
        _OBJ_GAIN,
        Objective("axial_ratio", "Circular polarisation purity", weight=0.8,
                  direction="minimize"),
        _OBJ_SWR,
    ],
    gates=_COMMON_GATES + [
        Gate("axial_mode", "Helix diameter must be ≈ λ/π for axial mode"),
    ],
))


# ---------------------------------------------------------------------------
# COLLINEAR
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="collinear",
    strategy_notes=(
        "Collinear gain comes from stacking λ/2 sections in phase.\n"
        "Phasing stubs between sections must be correct length.\n"
        "More sections = more gain but taller and harder to support."
    ),
    knobs=[
        KnobSpec("section_length", "Length of each radiating section", "m", sensitivity="medium"),
        KnobSpec("phasing_stub", "Phasing stub dimensions", "m", sensitivity="high"),
        KnobSpec("n_sections", "Number of stacked sections", "count", sensitivity="low"),
    ],
    operators=[
        Operator("adjust_sections",
                 "Tweak section length for resonance at design frequency",
                 affects=("section_length",), risk="low"),
        Operator("adjust_phasing",
                 "Tune phasing stub length for correct in-phase operation",
                 affects=("phasing_stub",), risk="medium"),
    ],
    objectives=[
        _OBJ_GAIN,
        Objective("low_angle", "Low elevation angle", weight=0.7, direction="minimize"),
        _OBJ_SWR,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# DISCONE
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="discone",
    strategy_notes=(
        "Discone is inherently broadband — typical 10:1 frequency ratio.\n"
        "Cone angle and disc size set the low-frequency cutoff.\n"
        "SWR should be flat across the entire range."
    ),
    knobs=[
        KnobSpec("cone_angle", "Half-angle of the cone", "degrees",
                  typical_range=(25, 45), sensitivity="medium"),
        KnobSpec("disc_diameter", "Top disc diameter", "m", sensitivity="low"),
        KnobSpec("element_count", "Number of cone/disc rods", "count", sensitivity="low"),
    ],
    operators=[
        Operator("adjust_cone_angle",
                 "Change cone angle to shift the low-frequency cutoff",
                 affects=("cone_angle",), risk="low"),
    ],
    objectives=[
        _OBJ_BW,
        _OBJ_SWR,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# DELTA LOOP
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="delta_loop",
    strategy_notes=(
        "Delta loop perimeter ≈ 1λ. Feed position controls polarisation.\n"
        "Tolerant design — main knob is total wire length."
    ),
    knobs=[
        KnobSpec("perimeter", "Total loop perimeter", "m", sensitivity="medium"),
        KnobSpec("apex_height", "Height of apex above base", "m", sensitivity="low"),
        KnobSpec("feed_position", "Feed position (bottom/side/corner)", "degrees", sensitivity="low"),
    ],
    operators=[
        Operator("adjust_perimeter",
                 "Scale perimeter to shift resonant frequency",
                 affects=("perimeter",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_GAIN,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# LOOP (generic full-wave)
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="loop",
    strategy_notes=(
        "Full-wave loop: perimeter ≈ 1λ.  ~120Ω feedpoint impedance.\n"
        "Shape (square, circle, triangle) affects gain slightly."
    ),
    knobs=[
        KnobSpec("perimeter", "Total loop perimeter", "m", sensitivity="medium"),
        KnobSpec("height", "Height above ground", "m", sensitivity="low"),
        KnobSpec("feed_position", "Feed point location on loop", "degrees", sensitivity="low"),
    ],
    operators=[
        Operator("adjust_perimeter",
                 "Change wire length to shift resonance",
                 affects=("perimeter",), risk="low"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# J-POLE
# ---------------------------------------------------------------------------

_register(ImprovementPolicy(
    antenna_type="j_pole",
    strategy_notes=(
        "J-pole is a half-wave vertical with a quarter-wave matching stub.\n"
        "Feedpoint tap position on the stub controls impedance."
    ),
    knobs=[
        KnobSpec("radiator_length", "Main radiator length", "m", sensitivity="medium"),
        KnobSpec("stub_length", "Matching stub length", "m", sensitivity="medium"),
        KnobSpec("stub_spacing", "Gap between stub conductors", "m", sensitivity="high"),
        KnobSpec("tap_position", "Feedpoint tap height on stub", "m", sensitivity="high"),
    ],
    operators=[
        Operator("adjust_radiator",
                 "Change radiator length to shift resonance",
                 affects=("radiator_length",), risk="low"),
        Operator("adjust_tap",
                 "Move feedpoint tap to optimise 50Ω match",
                 affects=("tap_position",), risk="medium"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_GAIN,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
))


# ---------------------------------------------------------------------------
# Generic fallback for types without a specific policy
# ---------------------------------------------------------------------------

_GENERIC_POLICY = ImprovementPolicy(
    antenna_type="generic",
    strategy_notes=(
        "No type-specific policy available. Apply general antenna optimisation:\n"
        "1. Verify structural validity first.\n"
        "2. Check SWR at design frequency — adjust element lengths.\n"
        "3. Check gain — verify geometry matches the intended type.\n"
        "4. Check build practicality — avoid tight tolerances."
    ),
    knobs=[
        KnobSpec("element_dimensions", "Wire/element lengths and positions", "m",
                  sensitivity="medium"),
        KnobSpec("feed_position", "Excitation position", "segment", sensitivity="medium"),
    ],
    operators=[
        Operator("scale_dimensions",
                 "Scale all dimensions to shift resonant frequency",
                 affects=("element_dimensions",), risk="low"),
        Operator("adjust_feed",
                 "Move feed point to improve impedance match",
                 affects=("feed_position",), risk="medium"),
    ],
    objectives=[
        _OBJ_SWR,
        _OBJ_GAIN,
        _OBJ_BW,
        _OBJ_BUILD,
    ],
    gates=_COMMON_GATES,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def policy_for_type(antenna_type: str) -> ImprovementPolicy:
    """Look up the improvement policy for an antenna type.

    Returns a type-specific policy if registered, otherwise a generic
    fallback policy.
    """
    return _POLICIES.get(antenna_type.lower().strip(), _GENERIC_POLICY)


def all_policy_types() -> list[str]:
    """Return all antenna types that have specific policies registered."""
    return sorted(_POLICIES.keys())
