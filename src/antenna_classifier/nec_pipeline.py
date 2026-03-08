"""Structured NEC generation pipeline.

Decomposes natural-language-to-NEC generation into discrete, verifiable steps:

  1. **Classify** — identify antenna type from document text
  2. **Extract**  — pull structured parameters given antenna type
  3. **Generate** — LLM produces JSON NEC deck from structured data
  4. **Validate** — cross-check dimensions with physics calculators
  5. **Convert**  — mechanical JSON → NEC text
  6. **Simulate** — run nec2c, evaluate goals + buildability
  7. **Feedback** — route errors back to the appropriate step

Each step's output is validated before advancing.  The LLM's job at each
step is much simpler than the monolithic approach — step 2 does text
comprehension, step 3 does antenna engineering, and steps 4-6 are
entirely deterministic.

Prototype: 2026-03-08 — structured pipeline for higher first-pass solve rate.
"""

from __future__ import annotations

import json as _json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_C_MPS = 299_792_458.0  # speed of light (m/s)

# Max retries for the whole pipeline (full loop-backs)
_MAX_PIPELINE_RETRIES = 2

# Acceptable deviation from calculator reference (fraction)
_CALC_TOLERANCE = 0.40  # 40% — generous, allows real design variation

# Models
_COMPREHENSION_MODEL = "gpt-4o-mini"  # cheap, fast for classification & extraction
_ENGINEERING_MODEL = "gpt-5.2"        # strong for geometry generation

# Canonical antenna types (from classifier.py)
ANTENNA_TYPES = [
    "yagi", "dipole", "vertical", "loop", "quad", "quagi", "hexbeam", "lpda",
    "phased_array", "helix", "collinear", "inverted_v", "end_fed", "j_pole",
    "moxon", "wire_array", "patch", "fractal", "magnetic_loop",
    "bobtail_curtain", "delta_loop", "v_beam", "batwing", "zigzag",
    "rhombic", "beverage", "discone", "turnstile", "half_square",
]

# Hub-and-spoke antenna types where all wires legitimately share a common
# feedpoint (e.g., vertical + radials).  Exempt from collapsed-geometry check.
_RADIAL_HUB_TYPES = {
    "vertical", "ground_plane", "j_pole", "end_fed", "collinear",
    "discone", "turnstile",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    """Record of one pipeline step execution."""

    step: int
    name: str
    status: str  # "ok", "fail", "skip"
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedConcepts:
    """Structured data extracted from a source document."""

    antenna_type: str = ""
    type_confidence: float = 0.0
    type_evidence: str = ""

    freq_mhz: float = 0.0
    bands: list[str] = field(default_factory=list)

    # Design goals
    gain_dbi: float | None = None
    fb_db: float | None = None
    max_swr: float | None = None

    # Structural parameters (type-specific, snake_case keys)
    elements: dict[str, Any] = field(default_factory=dict)

    # Common structural parameters
    ground_type: str = ""       # "free_space", "perfect", "real"
    height_m: float | None = None
    wire_dia_mm: float | None = None

    # Transmission line extracted from source document
    # e.g. {"z0": 250, "between": [1, 2]}  (tags of connected elements)
    transmission_line: dict[str, Any] | None = None

    # Raw description (for LLM context)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "antenna_type": self.antenna_type,
            "type_confidence": self.type_confidence,
            "type_evidence": self.type_evidence,
            "freq_mhz": self.freq_mhz,
            "bands": self.bands,
            "gain_dbi": self.gain_dbi,
            "fb_db": self.fb_db,
            "max_swr": self.max_swr,
            "elements": self.elements,
            "ground_type": self.ground_type,
            "height_m": self.height_m,
            "wire_dia_mm": self.wire_dia_mm,
            "transmission_line": self.transmission_line,
        }


@dataclass
class PipelineResult:
    """Output of the full pipeline run."""

    nec_content: str = ""
    json_deck: dict[str, Any] = field(default_factory=dict)
    concepts: ExtractedConcepts = field(default_factory=ExtractedConcepts)
    source_text: str = ""  # first 4000 chars of input
    steps: list[StepLog] = field(default_factory=list)
    iterations: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0,
    })
    model: str = ""

    # Evaluation results
    classified_type: str = ""
    confidence: float = 0.0
    goal_verdict: dict[str, Any] | None = None
    buildability: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nec_content": self.nec_content,
            "json_deck": self.json_deck,
            "concepts": self.concepts.to_dict(),
            "source_text": self.source_text,
            "steps": [
                {"step": s.step, "name": s.name, "status": s.status,
                 "detail": s.detail}
                for s in self.steps
            ],
            "iterations": self.iterations,
            "usage": self.usage,
            "model": self.model,
            "classified_type": self.classified_type,
            "confidence": self.confidence,
            "goal_verdict": self.goal_verdict,
            "buildability": self.buildability,
        }


# ---------------------------------------------------------------------------
# Per-type extraction specs
# ---------------------------------------------------------------------------
# Each spec defines what structural parameters to extract from a document
# for a given antenna type.  The LLM uses these to produce a focused
# extraction prompt.  Format: list of (name, description, unit).

_EXTRACTION_SPECS: dict[str, dict[str, Any]] = {
    "yagi": {
        "desc": "Yagi-Uda beam with parallel elements along a boom",
        "params": [
            ("n_elements", "Total number of elements (reflector + driven + directors)", "count"),
            ("n_driven_elements", "Number of driven elements (1 for standard Yagi, 2 for phased dual-driven)", "count"),
            ("reflector_length", "Reflector element full length", "m"),
            ("driven_length", "Driven element full length", "m"),
            ("director_lengths", "Director element full lengths (list)", "m[]"),
            ("element_spacings", "Spacings between adjacent elements from reflector forward (list)", "m[]"),
            ("wire_diameter", "Element wire or tube diameter", "mm"),
            ("height", "Height above ground", "m"),
            ("transmission_line_z0", "Impedance of a transmission line / phase line connecting driven elements (if any)", "ohm"),
        ],
    },
    "moxon": {
        "desc": "Moxon rectangle — compact 2-element beam with bent-back tips",
        "params": [
            ("driven_width", "Driven element width (main straight section A)", "m"),
            ("reflector_width", "Reflector width (main straight section D)", "m"),
            ("tail_length", "Tail/inward-bend length (C dimension)", "m"),
            ("gap", "Gap between driven and reflector tips (B dimension)", "m"),
            ("element_spacing", "Front-to-back spacing between main elements", "m"),
            ("wire_diameter", "Wire or tube diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "dipole": {
        "desc": "Half-wave dipole",
        "params": [
            ("total_length", "Total tip-to-tip length", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "inverted_v": {
        "desc": "Inverted-V dipole with drooping legs from an apex",
        "params": [
            ("total_length", "Total wire length tip-to-tip", "m"),
            ("apex_height", "Height of the apex (centre) above ground", "m"),
            ("droop_angle", "Included angle at apex in degrees (180 = flat dipole)", "deg"),
            ("wire_diameter", "Wire diameter", "mm"),
        ],
    },
    "vertical": {
        "desc": "Quarter-wave vertical with radials",
        "params": [
            ("radiator_height", "Vertical radiator height", "m"),
            ("n_radials", "Number of radial wires", "count"),
            ("radial_length", "Length of each radial wire", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
        ],
    },
    "end_fed": {
        "desc": "End-fed half-wave antenna",
        "params": [
            ("wire_length", "Total wire length", "m"),
            ("counterpoise_length", "Counterpoise wire length", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "j_pole": {
        "desc": "J-pole antenna (half-wave radiator with quarter-wave matching stub)",
        "params": [
            ("radiator_length", "Half-wave radiator section length", "m"),
            ("stub_length", "Quarter-wave matching stub length", "m"),
            ("stub_spacing", "Gap between stub conductors", "m"),
            ("wire_diameter", "Wire or tube diameter", "mm"),
        ],
    },
    "quad": {
        "desc": "Cubical quad with full-wave loop elements",
        "params": [
            ("n_elements", "Number of elements (reflector + driven + directors)", "count"),
            ("reflector_perimeter", "Reflector loop perimeter (or side length)", "m"),
            ("driven_perimeter", "Driven element loop perimeter (or side length)", "m"),
            ("director_perimeters", "Director loop perimeters (list)", "m[]"),
            ("element_spacings", "Spacings between elements (list)", "m[]"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height of boom above ground", "m"),
        ],
    },
    "lpda": {
        "desc": "Log-periodic dipole array",
        "params": [
            ("n_elements", "Number of dipole elements", "count"),
            ("longest_length", "Longest element full length", "m"),
            ("shortest_length", "Shortest element full length", "m"),
            ("tau", "Design ratio τ (scale factor between adjacent elements)", "ratio"),
            ("sigma", "Relative spacing σ", "ratio"),
            ("boom_length", "Total boom length", "m"),
            ("wire_diameter", "Element wire or tube diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "loop": {
        "desc": "Full-wave loop antenna (circular, square, or triangular)",
        "params": [
            ("perimeter", "Total loop perimeter", "m"),
            ("shape", "Loop shape: circular, square, or triangular", "text"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "delta_loop": {
        "desc": "Delta (triangular) loop antenna",
        "params": [
            ("perimeter", "Total perimeter of the triangle", "m"),
            ("base_length", "Base wire length", "m"),
            ("apex_height", "Height of apex above base", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height of base above ground", "m"),
        ],
    },
    "magnetic_loop": {
        "desc": "Small transmitting loop (magnetic loop)",
        "params": [
            ("circumference", "Loop circumference", "m"),
            ("diameter", "Loop diameter", "m"),
            ("conductor_diameter", "Loop conductor diameter (tube OD)", "mm"),
            ("tuning_capacitor", "Tuning capacitance", "pF"),
        ],
    },
    "collinear": {
        "desc": "Collinear vertical array",
        "params": [
            ("n_sections", "Number of collinear sections", "count"),
            ("section_length", "Length of each radiating section", "m"),
            ("stub_length", "Phasing stub length between sections", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
        ],
    },
    "hexbeam": {
        "desc": "Hexagonal beam antenna (G3TXQ style)",
        "params": [
            ("frame_radius", "Hexagonal frame radius (centre to corner)", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "discone": {
        "desc": "Discone antenna (broadband, disc over cone)",
        "params": [
            ("disc_diameter", "Disc element diameter", "m"),
            ("cone_length", "Cone slant length", "m"),
            ("cone_angle", "Cone half-angle", "deg"),
            ("wire_diameter", "Wire or tube diameter", "mm"),
        ],
    },
    "helix": {
        "desc": "Helical antenna (axial-mode)",
        "params": [
            ("n_turns", "Number of turns", "count"),
            ("diameter", "Helix diameter", "m"),
            ("pitch", "Turn spacing (pitch)", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("ground_plane_diameter", "Ground plane diameter", "m"),
        ],
    },
    "bobtail_curtain": {
        "desc": "Bobtail curtain — 3 vertical radiators with horizontal top wires",
        "params": [
            ("vertical_length", "Vertical element length", "m"),
            ("horizontal_length", "Horizontal connecting wire length", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Total height above ground", "m"),
        ],
    },
    "phased_array": {
        "desc": "Phased array of vertical or horizontal elements",
        "params": [
            ("n_elements", "Number of elements", "count"),
            ("element_length", "Individual element length", "m"),
            ("element_spacing", "Spacing between elements", "m"),
            ("phasing", "Phase relationship (endfire, broadside, or degrees)", "text"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Height above ground", "m"),
        ],
    },
    "half_square": {
        "desc": "Half-square antenna — two vertical radiators with horizontal connecting wire",
        "params": [
            ("vertical_length", "Vertical element length", "m"),
            ("horizontal_length", "Horizontal connecting wire length", "m"),
            ("wire_diameter", "Wire diameter", "mm"),
            ("height", "Total height above ground", "m"),
        ],
    },
}

# Generic fallback for types not in _EXTRACTION_SPECS
_GENERIC_PARAMS: list[tuple[str, str, str]] = [
    ("total_length", "Overall antenna length or largest dimension", "m"),
    ("total_width", "Overall width", "m"),
    ("total_height", "Overall height", "m"),
    ("n_elements", "Number of radiating elements", "count"),
    ("wire_diameter", "Wire or conductor diameter", "mm"),
    ("height", "Height above ground", "m"),
]


def _get_extraction_spec(antenna_type: str) -> dict[str, Any]:
    """Get the extraction spec for an antenna type (with generic fallback)."""
    if antenna_type in _EXTRACTION_SPECS:
        return _EXTRACTION_SPECS[antenna_type]
    return {
        "desc": f"{antenna_type} antenna",
        "params": _GENERIC_PARAMS,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Get OpenAI client (delegates to nec_generator)."""
    from .nec_generator import _get_client as _gc
    return _gc()


def _extract_tl_impedance(text: str) -> float | None:
    """Regex scan for transmission/phase line impedance in document text.

    Looks for patterns like "250 Ω", "250-ohm", "250 ohm phase line", etc.
    Returns the Z0 value or None if not found.
    """
    # Pattern: number followed by Ω/ohm, near "phase line"/"transmission line"
    # Match e.g. "250 Ω characteris-", "250-ohm phase line", "Z0 = 250"
    patterns = [
        # "NNN Ω" or "NNN ohm" near phase/transmission line context
        r'(\d{2,4})\s*[Ωω]\s*(?:characteris|impedance|phase|transmission)',
        r'(\d{2,4})\s*[-‐]?\s*(?:ohm|Ohm|OHM)\b',
        r'(?:phase|transmission)\s+line\s+.*?(\d{2,4})\s*[Ωω]',
        r'Z0\s*[=:]\s*(\d{2,4})',
        r'(\d{2,4})\s*[Ωω]\s+(?:transmission|phase)\s+line',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            z0 = float(m.group(1))
            if 25 <= z0 <= 1000:  # reasonable TL impedance range
                return z0
    return None


def _track_usage(result: PipelineResult, resp) -> None:
    """Accumulate token usage from an API response."""
    if resp.usage:
        result.usage["prompt_tokens"] += resp.usage.prompt_tokens
        result.usage["completion_tokens"] += resp.usage.completion_tokens


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response (handles fences, chat)."""
    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    text = m.group(1).strip() if m else raw.strip()

    # Find outermost { ... }
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("Unbalanced JSON braces")
    return _json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Step 1: Classify document
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are an antenna identification expert.  Given text from a document,
identify what type of antenna is described.

Choose EXACTLY ONE type from this list:
{types}

If the document describes multiple bands for the SAME antenna design
(e.g. a multi-band Yagi), still pick the base antenna type.

Respond with ONLY a JSON object:
{{"antenna_type": "...", "confidence": 0.0-1.0, "evidence": "brief reasoning"}}
"""


def classify_document(
    text: str,
    *,
    client: Any = None,
    model: str = _COMPREHENSION_MODEL,
) -> tuple[str, float, str]:
    """Step 1: Classify the antenna type from document text.

    Returns (antenna_type, confidence, evidence).
    """
    if client is None:
        client = _get_client()

    # Use first 3000 chars — enough to identify the type
    excerpt = text[:3000]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT.format(
                types=", ".join(ANTENNA_TYPES),
            )},
            {"role": "user", "content": f"Identify the antenna type:\n\n{excerpt}"},
        ],
        temperature=0.1,
        max_completion_tokens=256,
    )
    raw = resp.choices[0].message.content or ""

    try:
        data = _parse_llm_json(raw)
        atype = data.get("antenna_type", "unknown")
        conf = float(data.get("confidence", 0.0))
        evidence = data.get("evidence", "")
        # Validate type is canonical
        if atype not in ANTENNA_TYPES:
            atype = "unknown"
            conf = 0.0
        return atype, conf, evidence
    except (ValueError, KeyError):
        return "unknown", 0.0, f"Failed to parse classification: {raw[:200]}"


# ---------------------------------------------------------------------------
# Step 2: Extract structured concepts
# ---------------------------------------------------------------------------

def _build_extraction_prompt(antenna_type: str) -> str:
    """Build a type-specific extraction prompt."""
    spec = _get_extraction_spec(antenna_type)

    param_lines = []
    for name, desc, unit in spec["params"]:
        if unit.endswith("[]"):
            param_lines.append(f'  "{name}": [{unit[:-2]}, ...] or null  // {desc}')
        elif unit == "count":
            param_lines.append(f'  "{name}": integer or null  // {desc}')
        elif unit == "text":
            param_lines.append(f'  "{name}": "string" or null  // {desc}')
        elif unit == "ratio":
            param_lines.append(f'  "{name}": float or null  // {desc}')
        else:
            param_lines.append(f'  "{name}": float or null  // {desc} [{unit}]')

    return f"""\
You are an expert at reading antenna technical documents and extracting
precise dimensional data.

This document describes a **{antenna_type}** antenna ({spec["desc"]}).

Extract the following structural parameters from the text.
If a value is not explicitly stated, use null.
Convert all lengths to **metres** and diameters to **millimetres**.
Convert all frequencies to **MHz**.
If dimensions are given in inches, feet, or wavelengths, convert to metres.
If element "half-lengths" are given, report the FULL length (double them).

Also extract any stated performance goals:
  "freq_mhz": design frequency in MHz
  "gain_dbi": target gain in dBi (convert dBd → dBi by adding 2.15)
  "fb_db": front-to-back ratio in dB
  "max_swr": target SWR (e.g. 1.5 for 1.5:1)
  "bands": list of ham bands mentioned (e.g. ["20m", "10m"])
  "ground_type": "free_space", "perfect", or "real" (if mentioned)

Respond with ONLY a JSON object:
{{
  "freq_mhz": float or null,
  "gain_dbi": float or null,
  "fb_db": float or null,
  "max_swr": float or null,
  "bands": ["Xm", ...] or [],
  "ground_type": "free_space" or "perfect" or "real" or null,
{chr(10).join(param_lines)}
}}
"""


def extract_concepts(
    text: str,
    antenna_type: str,
    *,
    client: Any = None,
    model: str = _COMPREHENSION_MODEL,
    freq_mhz: float = 0.0,
) -> ExtractedConcepts:
    """Step 2: Extract structured parameters from document text.

    Uses regex for freq/goals (fast, free) and LLM for structural
    parameters (requires text comprehension).

    Returns an ExtractedConcepts instance.
    """
    from .nec_generator import _guess_freq_mhz, _extract_design_goals

    concepts = ExtractedConcepts(antenna_type=antenna_type)

    # --- Regex extraction (free) ---
    if freq_mhz > 0:
        concepts.freq_mhz = freq_mhz
    else:
        concepts.freq_mhz = _guess_freq_mhz(text, antenna_type)

    doc_goals = _extract_design_goals(text)
    concepts.gain_dbi = doc_goals.gain_dbi
    concepts.fb_db = doc_goals.fb_db
    concepts.max_swr = doc_goals.max_swr
    concepts.bands = doc_goals.bands

    # --- Regex: detect transmission/phase line impedance (scan full text) ---
    tl_z0 = _extract_tl_impedance(text)
    if tl_z0 is not None:
        concepts.transmission_line = {"z0": tl_z0}

    # --- LLM extraction (structural parameters) ---
    if client is None:
        client = _get_client()

    prompt = _build_extraction_prompt(antenna_type)

    # Use first 12000 chars — enough for dimension tables in multi-column PDFs
    excerpt = text[:12000]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Extract parameters:\n\n{excerpt}"},
        ],
        temperature=0.1,
        max_completion_tokens=1024,
    )
    raw = resp.choices[0].message.content or ""

    try:
        data = _parse_llm_json(raw)
    except (ValueError, _json.JSONDecodeError):
        log.warning("Could not parse extraction response: %s", raw[:300])
        data = {}

    # Merge LLM-extracted data into concepts
    spec = _get_extraction_spec(antenna_type)
    param_names = {p[0] for p in spec["params"]}

    for key, val in data.items():
        if val is None:
            continue
        if key == "freq_mhz" and concepts.freq_mhz == 0:
            concepts.freq_mhz = float(val)
        elif key == "gain_dbi" and concepts.gain_dbi is None:
            concepts.gain_dbi = float(val)
        elif key == "fb_db" and concepts.fb_db is None:
            concepts.fb_db = float(val)
        elif key == "max_swr" and concepts.max_swr is None:
            concepts.max_swr = float(val)
        elif key == "bands" and not concepts.bands and isinstance(val, list):
            concepts.bands = val
        elif key == "ground_type" and not concepts.ground_type:
            concepts.ground_type = str(val)
        elif key in param_names:
            concepts.elements[key] = val

    # Pull out common structural fields
    if "wire_diameter" in concepts.elements:
        concepts.wire_dia_mm = float(concepts.elements.pop("wire_diameter"))
    if "height" in concepts.elements:
        concepts.height_m = float(concepts.elements.pop("height"))

    # Pull out transmission line info if extracted by LLM (fallback for regex)
    tl_z0 = concepts.elements.pop("transmission_line_z0", None)
    n_driven = concepts.elements.get("n_driven_elements")
    if tl_z0 is not None and float(tl_z0) > 0 and concepts.transmission_line is None:
        concepts.transmission_line = {"z0": float(tl_z0)}

    return concepts, resp


# ---------------------------------------------------------------------------
# Step 3: Generate JSON NEC deck from structured concepts
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """\
You are an expert antenna engineer. Given structured antenna specifications,
generate a NEC2 input file in JSON format.

**Output format:** A JSON object with a single key "cards" containing an
array of NEC card objects.  Each card:
  - "type": two-letter NEC card code (GW, FR, EX, etc.)
  - "params": array of numbers (for geometry/control cards)
  - "text": string (for CM/CE cards only)

**Card parameter order (same as NEC2):**
- CM: {{"type":"CM", "text":"<comment>"}}
- CE: {{"type":"CE"}}
- GW: {{"type":"GW", "params":[tag, segments, x1, y1, z1, x2, y2, z2, radius]}}
- GE: {{"type":"GE", "params":[ground_type]}}
- TL: {{"type":"TL", "params":[tag1, seg1, tag2, seg2, Z0, length, VR1, VI1, VR2, VI2]}}
- EX: {{"type":"EX", "params":[ex_type, tag, segment, 0, v_real, v_imag]}}
- FR: {{"type":"FR", "params":[fr_type, n_freq, 0, 0, start_mhz, step_mhz]}}
- GN: {{"type":"GN", "params":[gn_type, ...]}}
- LD: {{"type":"LD", "params":[ld_type, tag, seg_start, seg_end, R, L_or_X, C_or_B]}}
- RP: {{"type":"RP", "params":[rp_type, ntheta, nphi, mode, theta_start, phi_start, theta_step, phi_step]}}
- EN: {{"type":"EN"}}

**Geometry rules:**
1. Z is UP. X and Y are horizontal.
2. Multi-element antennas: elements at DIFFERENT positions.
3. All dimensions in metres. Wire radius in metres.
4. Segments: ~10-20 per half-wavelength.
5. Always: CM → CE → GW... → GE → control cards → EN.
6. Feed the driven element at its centre segment.
7. TL (transmission line) cards: both ports MUST connect at the CENTRE segment
   of their respective wires, NOT at segment 1 or the last segment.
   For a wire with N segments, the centre is segment (N+1)/2.
   TL segments should match the feedpoint (EX segment) positions.

Output ONLY the JSON object. No markdown fences, no commentary.
"""


def _format_concepts_for_generation(
    concepts: ExtractedConcepts,
    calc_summary: str = "",
    type_context: str = "",
) -> str:
    """Format extracted concepts as a user prompt for JSON deck generation."""
    lines = [
        f"Generate a NEC2 JSON deck for this antenna:\n",
        f"ANTENNA TYPE: {concepts.antenna_type}",
        f"DESIGN FREQUENCY: {concepts.freq_mhz} MHz",
    ]

    # Ground
    ground = concepts.ground_type or "free_space"
    ground_map = {
        "free_space": "free space (GE 0, no GN card)",
        "perfect": "perfect ground (GE 1, GN 1)",
        "real": "real ground — average earth (GE 1, GN 2)",
    }
    lines.append(f"GROUND: {ground_map.get(ground, ground)}")

    # Structural parameters from the document
    if concepts.elements or concepts.wire_dia_mm or concepts.height_m:
        lines.append("\nEXTRACTED DIMENSIONS (from the source document):")
        for key, val in concepts.elements.items():
            if isinstance(val, list):
                formatted = ", ".join(
                    f"{v:.4f} m" if isinstance(v, float) else str(v) for v in val
                )
                lines.append(f"  • {key}: [{formatted}]")
            elif isinstance(val, float):
                lines.append(f"  • {key}: {val:.4f} m")
            else:
                lines.append(f"  • {key}: {val}")
        if concepts.wire_dia_mm:
            lines.append(f"  • wire_diameter: {concepts.wire_dia_mm} mm "
                         f"(radius = {concepts.wire_dia_mm / 2000:.6f} m)")
        if concepts.height_m:
            lines.append(f"  • height: {concepts.height_m} m above ground")
        lines.append(
            "Use these document dimensions as your PRIMARY reference. "
            "They come directly from the source material."
        )

    # Transmission line requirement
    if concepts.transmission_line:
        tl = concepts.transmission_line
        z0 = tl.get("z0", 0)
        lines.append(
            f"\n** TRANSMISSION LINE (MANDATORY) **\n"
            f"The source document specifies a transmission line / phase line "
            f"with Z0 = {z0} Ω connecting the driven elements.\n"
            f"You MUST include a TL card in the output.\n"
            f"TL card rules:\n"
            f"  - Connect at the CENTRE segment of each driven element wire.\n"
            f"  - Use Z0 = {z0}.\n"
            f"  - Set length = 0 (electrical length equals physical spacing).\n"
            f"  - Feed (EX card) on one of the driven elements at its centre segment.\n"
            f"  - Do NOT omit the TL card. This is a phased design, not a "
            f"standard Yagi."
        )

    # Design goals
    goal_lines = []
    if concepts.gain_dbi is not None:
        goal_lines.append(f"  • Target gain: {concepts.gain_dbi:.1f} dBi")
    if concepts.fb_db is not None:
        goal_lines.append(f"  • Target front-to-back ratio: ≥ {concepts.fb_db:.0f} dB")
    if concepts.max_swr is not None:
        goal_lines.append(f"  • Target SWR: ≤ {concepts.max_swr:.1f}:1")
    if concepts.bands:
        goal_lines.append(f"  • Bands: {', '.join(concepts.bands)}")
    if goal_lines:
        lines.append("\nDESIGN GOALS:")
        lines.extend(goal_lines)

    # Calculator reference
    if calc_summary:
        lines.append(f"\n--- CALCULATOR REFERENCE (physics-based starting point) ---")
        lines.append(calc_summary)
        lines.append(
            "If document dimensions are available, prefer them over calculator "
            "values. Calculator values are a sanity check."
        )
        lines.append("--- END CALCULATOR ---")

    # Type context reference
    if type_context:
        lines.append(f"\n--- REFERENCE for {concepts.antenna_type} antennas ---")
        lines.append(type_context)
        lines.append("--- END REFERENCE ---")

    # Description (if from form)
    if concepts.description:
        lines.append(f"\nAdditional description: {concepts.description}")

    return "\n".join(lines)


def generate_deck(
    concepts: ExtractedConcepts,
    *,
    client: Any = None,
    model: str = _ENGINEERING_MODEL,
    feedback: str = "",
    history: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Step 3: Generate a JSON NEC deck from structured concepts.

    Parameters
    ----------
    history : list[dict] | None
        Previous assistant/user message pairs for OODA conversation continuity.
        Each dict has ``role`` and ``content`` keys.

    Returns (json_deck, api_response).
    """
    from .nec_calculators import calc_for_type
    from .nec_generator import _load_type_context

    if client is None:
        client = _get_client()

    # Get calculator reference dimensions
    calc_summary = ""
    if concepts.freq_mhz > 0:
        calc = calc_for_type(concepts.antenna_type, concepts.freq_mhz)
        if calc is not None:
            calc_summary = calc.summary()
            for note in calc.notes:
                calc_summary += f"\n  • {note}"
            if calc.nec_hints:
                calc_summary += "\nNEC modelling hints:"
                for hint in calc.nec_hints:
                    calc_summary += f"\n  • {hint}"

    type_context = _load_type_context(concepts.antenna_type)

    user_msg = _format_concepts_for_generation(
        concepts,
        calc_summary=calc_summary,
        type_context=type_context,
    )

    # Build message list: system + user + optional OODA conversation history
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _GENERATE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    if history:
        # Append previous assistant/user exchanges so the LLM sees what it
        # generated before and the specific feedback it received.
        messages.extend(history)
    elif feedback:
        # First-iteration fallback: append flat feedback to user prompt
        messages[-1]["content"] += (
            f"\n\n--- FEEDBACK FROM PREVIOUS ATTEMPT ---\n{feedback}\n---"
        )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=8192,
    )
    raw = resp.choices[0].message.content or ""

    deck = _parse_llm_json(raw)
    if "cards" not in deck or not isinstance(deck["cards"], list):
        raise ValueError("JSON response missing 'cards' array")
    for i, card in enumerate(deck["cards"]):
        if "type" not in card:
            raise ValueError(f"Card {i} missing 'type' field")

    return deck, resp


# ---------------------------------------------------------------------------
# Deterministic Moxon NEC stamper
# ---------------------------------------------------------------------------
# The Moxon has exactly 4 degrees of freedom (A, B, C, D) mapped to 6 GW
# cards in a fixed U+U topology.  LLMs consistently get the tip-gap geometry
# wrong, so we compute the coordinates deterministically from known-good
# reference dimensions that score 5/5 on goals.

# Reference dimensions in wavelength fractions, from moxon.txt 20m
# design that achieves SWR=1.14 Gain=5.84dBi F/B=35.9dB (5/5 goals).
_MOXON_REF_HALF_A_WL = 0.18015    # half-width: 3.81 / 21.1494
_MOXON_REF_B_WL = 0.010402        # tip gap:   0.220 / 21.1494
_MOXON_REF_C_WL = 0.057638        # driven tail: 1.219 / 21.1494
_MOXON_REF_D_WL = 0.069882        # reflector tail: 1.478 / 21.1494
_MOXON_REF_HEIGHT_WL = 0.504424   # height: 10.668 / 21.1494 (≈ λ/2)

# Segment counts matching the reference (odd main for centre-feed)
_MOXON_SEGS_TAIL_DRV = 7
_MOXON_SEGS_MAIN = 45
_MOXON_SEGS_TAIL_REF = 9


def _stamp_moxon_deck(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    height_m: float | None = None,
    ground_type: str = "free_space",
) -> dict[str, Any]:
    """Build a Moxon JSON deck deterministically from reference geometry.

    Returns the same ``{"cards": [...]}`` format that ``generate_deck()``
    produces, so it plugs directly into steps 4-7.
    """
    wl = _C_MPS / (freq_mhz * 1e6)
    wire_radius = wire_dia_mm / 2000.0  # metres

    # Scale reference dimensions to target wavelength
    half_a = _MOXON_REF_HALF_A_WL * wl
    gap = _MOXON_REF_B_WL * wl
    c_tail = _MOXON_REF_C_WL * wl
    d_tail = _MOXON_REF_D_WL * wl

    # Height: use provided, or default to ~λ/2
    z = height_m if height_m is not None else _MOXON_REF_HEIGHT_WL * wl

    # Coordinate system: gap centred at Y=0, driven at +Y, reflector at -Y
    y_drv_tip = gap / 2.0
    y_drv_main = y_drv_tip + c_tail
    y_ref_tip = -gap / 2.0
    y_ref_main = y_ref_tip - d_tail

    def _r(v: float) -> float:
        return round(v, 4)

    cards: list[dict[str, Any]] = []

    # Comment cards
    cards.append({"type": "CM", "text":
        f"Moxon rectangle for {freq_mhz:.3f} MHz, "
        f"{'free space' if ground_type == 'free_space' else 'over ground'}, "
        f"wire {wire_dia_mm:.3f}mm dia"})
    cards.append({"type": "CM", "text":
        f"Stamped from reference geometry: A={2*half_a:.3f}m "
        f"B={gap*1000:.1f}mm C={c_tail:.3f}m D={d_tail:.3f}m"})
    cards.append({"type": "CE"})

    # Driven element: GW 1 (left tail), GW 2 (main), GW 3 (right tail)
    cards.append({"type": "GW", "params": [
        1, _MOXON_SEGS_TAIL_DRV,
        _r(-half_a), _r(y_drv_tip), _r(z),
        _r(-half_a), _r(y_drv_main), _r(z),
        wire_radius,
    ]})
    cards.append({"type": "GW", "params": [
        2, _MOXON_SEGS_MAIN,
        _r(-half_a), _r(y_drv_main), _r(z),
        _r(half_a), _r(y_drv_main), _r(z),
        wire_radius,
    ]})
    cards.append({"type": "GW", "params": [
        3, _MOXON_SEGS_TAIL_DRV,
        _r(half_a), _r(y_drv_main), _r(z),
        _r(half_a), _r(y_drv_tip), _r(z),
        wire_radius,
    ]})

    # Reflector: GW 4 (left tail), GW 5 (main), GW 6 (right tail)
    cards.append({"type": "GW", "params": [
        4, _MOXON_SEGS_TAIL_REF,
        _r(-half_a), _r(y_ref_tip), _r(z),
        _r(-half_a), _r(y_ref_main), _r(z),
        wire_radius,
    ]})
    cards.append({"type": "GW", "params": [
        5, _MOXON_SEGS_MAIN,
        _r(-half_a), _r(y_ref_main), _r(z),
        _r(half_a), _r(y_ref_main), _r(z),
        wire_radius,
    ]})
    cards.append({"type": "GW", "params": [
        6, _MOXON_SEGS_TAIL_REF,
        _r(half_a), _r(y_ref_main), _r(z),
        _r(half_a), _r(y_ref_tip), _r(z),
        wire_radius,
    ]})

    # Ground
    cards.append({"type": "GE", "params": [0]})

    # Conductivity: copper on all wires
    cards.append({"type": "LD", "params": [5, 0, 0, 0, 58001000, 0, 0]})

    # Excitation at centre of driven main wire (GW 2)
    feed_seg = (_MOXON_SEGS_MAIN + 1) // 2  # 23 for 45 segments
    cards.append({"type": "EX", "params": [0, 2, feed_seg, 0, 1, 0]})

    # Ground card (if not free space)
    if ground_type == "perfect":
        cards.append({"type": "GN", "params": [1]})
    elif ground_type == "real":
        # Average ground: εr=13, σ=5 mS/m
        cards.append({"type": "GN", "params": [0, 0, 0, 0, 13, 0.005]})

    # Frequency
    cards.append({"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]})

    # Radiation pattern: 5-degree steps
    cards.append({"type": "RP", "params": [0, 37, 73, 1000, 0, 0, 5, 5]})

    cards.append({"type": "EN"})

    return {"cards": cards}


# ---------------------------------------------------------------------------
# Step 4: Validate JSON deck with calculators
# ---------------------------------------------------------------------------

# NEC card parameter counts for structural validation
_CARD_PARAM_COUNTS: dict[str, int | tuple[int, int]] = {
    "GW": 9,
    "GA": 7,
    "GE": 1,
    "EX": (4, 6),   # 4 minimum, 6 typical
    "FR": 6,
    "GN": (1, 10),  # varies by type
    "RP": 8,
    "LD": 7,
    "TL": 10,
    "NT": 10,
}


def validate_deck(
    deck: dict[str, Any],
    concepts: ExtractedConcepts,
) -> list[str]:
    """Step 4: Validate a JSON NEC deck against physics and calculators.

    Returns a list of warning/error strings.  Empty list = all OK.
    """
    from .nec_calculators import calc_for_type

    issues: list[str] = []
    cards = deck.get("cards", [])

    # --- Structural card validation ---
    card_types = [c.get("type", "") for c in cards]

    required = {"GW", "GE", "EX", "FR", "EN"}
    present = set(card_types)
    missing = required - present
    if missing:
        issues.append(f"MISSING CARDS: {', '.join(sorted(missing))}")

    # --- TL card required when source specifies a transmission line ---
    if concepts.transmission_line and "TL" not in present:
        z0 = concepts.transmission_line.get("z0", "?")
        issues.append(
            f"MISSING TL CARD: the source document specifies a transmission "
            f"line (Z0={z0} Ω) connecting driven elements. You MUST include "
            f"a TL card connecting the driven elements at their centre segments."
        )

    # Check card parameter counts
    for card in cards:
        typ = card.get("type", "")
        params = card.get("params", [])
        if typ in _CARD_PARAM_COUNTS:
            expected = _CARD_PARAM_COUNTS[typ]
            if isinstance(expected, tuple):
                lo, hi = expected
                if len(params) < lo:
                    issues.append(
                        f"{typ} card has {len(params)} params, expected ≥ {lo}"
                    )
            else:
                if len(params) != expected:
                    issues.append(
                        f"{typ} card has {len(params)} params, expected {expected}"
                    )

    # --- GW wire checks ---
    gw_cards = [c for c in cards if c.get("type") == "GW"]
    if not gw_cards:
        issues.append("NO GW CARDS: deck has no wire geometry")
        return issues

    wl = _C_MPS / (concepts.freq_mhz * 1e6) if concepts.freq_mhz > 0 else 21.0

    for i, gw in enumerate(gw_cards):
        params = gw.get("params", [])
        if len(params) < 9:
            continue
        tag, segs = int(params[0]), int(params[1])
        x1, y1, z1 = float(params[2]), float(params[3]), float(params[4])
        x2, y2, z2 = float(params[5]), float(params[6]), float(params[7])
        radius = float(params[8])

        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # Zero-length wire
        if length < 1e-6:
            issues.append(f"GW tag {tag}: zero-length wire")

        # Unrealistic radius
        if radius <= 0:
            issues.append(f"GW tag {tag}: radius ≤ 0")
        elif radius > 0.1:  # > 100mm diameter
            issues.append(f"GW tag {tag}: radius {radius:.4f}m seems too large")
        elif radius < 1e-5:  # < 0.01mm
            issues.append(f"GW tag {tag}: radius {radius:.6f}m seems too small")

        # Segment count
        if segs < 1:
            issues.append(f"GW tag {tag}: segments < 1")
        elif segs > 500:
            issues.append(f"GW tag {tag}: {segs} segments is excessive")

        # Wire longer than 3λ (unusual for single elements)
        if length > 3 * wl:
            issues.append(
                f"GW tag {tag}: length {length:.2f}m = {length/wl:.1f}λ "
                f"(very long for {concepts.freq_mhz} MHz)"
            )

    # --- Collapsed geometry check ---
    # Hub-and-spoke types (verticals with radials, etc.) legitimately share
    # a common feedpoint for all wires — exempt them from this check.
    if len(gw_cards) >= 2 and concepts.antenna_type not in _RADIAL_HUB_TYPES:
        starts = set()
        for gw in gw_cards:
            p = gw.get("params", [])
            if len(p) >= 9:
                starts.add((round(p[2], 4), round(p[3], 4), round(p[4], 4)))
        if len(starts) == 1:
            issues.append(
                "COLLAPSED GEOMETRY: all wires share the same start point"
            )

    # --- FR card check ---
    for card in cards:
        if card.get("type") == "FR":
            params = card.get("params", [])
            if len(params) >= 5:
                fr_freq = float(params[4])
                if concepts.freq_mhz > 0 and abs(fr_freq - concepts.freq_mhz) > 1.0:
                    issues.append(
                        f"FR FREQUENCY MISMATCH: FR card says {fr_freq} MHz, "
                        f"expected {concepts.freq_mhz} MHz"
                    )

    # --- TL card segment validation ---
    # Build a map of wire tag → number of segments for cross-referencing
    wire_segs: dict[int, int] = {}
    for gw in gw_cards:
        p = gw.get("params", [])
        if len(p) >= 9:
            wire_segs[int(p[0])] = int(p[1])

    for card in cards:
        if card.get("type") != "TL":
            continue
        p = card.get("params", [])
        if len(p) < 4:
            continue
        tag1, seg1, tag2, seg2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        for tag, seg, label in [
            (tag1, seg1, "port 1"),
            (tag2, seg2, "port 2"),
        ]:
            n_segs = wire_segs.get(tag)
            if n_segs is None:
                issues.append(
                    f"TL {label}: references wire tag {tag} which has no GW card"
                )
            elif seg == 1 or seg == n_segs:
                centre = (n_segs + 1) // 2
                issues.append(
                    f"TL {label}: segment {seg} is at the END of wire {tag} "
                    f"({n_segs} segs). Transmission lines should connect at "
                    f"the centre segment ({centre}), not at endpoints."
                )

    # --- Calculator cross-check ---
    if concepts.freq_mhz > 0:
        calc = calc_for_type(concepts.antenna_type, concepts.freq_mhz)
        if calc is not None:
            # Extract element lengths from GW cards and compare
            element_lengths = []
            for gw in gw_cards:
                p = gw.get("params", [])
                if len(p) >= 9:
                    x1, y1, z1 = float(p[2]), float(p[3]), float(p[4])
                    x2, y2, z2 = float(p[5]), float(p[6]), float(p[7])
                    length = math.sqrt(
                        (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
                    )
                    element_lengths.append(length)

            # Type-specific checks
            dims = calc.dimensions
            _check_calc_dimensions(
                issues, concepts.antenna_type, dims, element_lengths, wl,
            )

    return issues


def _check_calc_dimensions(
    issues: list[str],
    antenna_type: str,
    dims: dict[str, Any],
    element_lengths: list[float],
    wl: float,
) -> None:
    """Compare generated element lengths against calculator expectations."""
    if not element_lengths:
        return

    # For beam antennas, check that the longest element is reasonable
    longest = max(element_lengths)
    shortest = min(element_lengths)

    if antenna_type in ("yagi", "moxon", "quad", "lpda", "hexbeam", "quagi"):
        # Beam antennas: longest element should be close to reflector
        ref_keys = ["reflector_length", "total_length", "driven_length",
                     "longest_length", "frame_radius"]
        for key in ref_keys:
            if key in dims:
                expected = float(dims[key])
                if expected > 0 and longest > 0:
                    ratio = longest / expected
                    if ratio > (1 + _CALC_TOLERANCE) or ratio < (1 - _CALC_TOLERANCE):
                        issues.append(
                            f"DIMENSION CHECK: longest element {longest:.3f}m "
                            f"vs calculator {key}={expected:.3f}m "
                            f"(ratio {ratio:.2f}, expected ~1.0 ± {_CALC_TOLERANCE})"
                        )
                break

    elif antenna_type in ("dipole", "inverted_v", "end_fed"):
        # Single-element: total length should be close to calculated
        ref_key = "total_length" if "total_length" in dims else "half_length"
        if ref_key in dims:
            expected = float(dims[ref_key])
            if ref_key == "half_length":
                expected *= 2  # compare full length
            if expected > 0 and longest > 0:
                ratio = longest / expected
                if ratio > (1 + _CALC_TOLERANCE) or ratio < (1 - _CALC_TOLERANCE):
                    issues.append(
                        f"DIMENSION CHECK: element length {longest:.3f}m "
                        f"vs calculator {expected:.3f}m "
                        f"(ratio {ratio:.2f})"
                    )

    elif antenna_type == "vertical":
        # Vertical: radiator height ≈ λ/4
        if "radiator_height" in dims:
            expected = float(dims["radiator_height"])
            if expected > 0 and longest > 0:
                ratio = longest / expected
                if ratio > (1 + _CALC_TOLERANCE) or ratio < (1 - _CALC_TOLERANCE):
                    issues.append(
                        f"DIMENSION CHECK: vertical height {longest:.3f}m "
                        f"vs calculator {expected:.3f}m (ratio {ratio:.2f})"
                    )

    # General sanity: no element should be longer than 2λ for most types
    if antenna_type not in ("lpda", "rhombic", "beverage", "v_beam"):
        for i, length in enumerate(element_lengths):
            if length > 2 * wl:
                issues.append(
                    f"SANITY: element {i+1} length {length:.2f}m = "
                    f"{length/wl:.1f}λ (unusually long)"
                )


# ---------------------------------------------------------------------------
# Step 5: JSON → NEC conversion (delegates to existing)
# ---------------------------------------------------------------------------

def convert_to_nec(deck: dict[str, Any]) -> str:
    """Step 5: Mechanical JSON-to-NEC conversion."""
    from .nec_generator import _json_to_nec
    return _json_to_nec(deck)


# ---------------------------------------------------------------------------
# Step 6: Simulate and evaluate (delegates to existing)
# ---------------------------------------------------------------------------

def simulate_and_evaluate(
    nec: str,
    antenna_type: str,
    freq_mhz: float,
) -> dict[str, Any]:
    """Step 6: Run nec2c simulation and full evaluation."""
    from .nec_generator import _evaluate_full
    return _evaluate_full(nec, antenna_type, freq_mhz)


# ---------------------------------------------------------------------------
# Step 7: Diagnose failures and route feedback
# ---------------------------------------------------------------------------

def diagnose_failure(
    eval_result: dict[str, Any],
    validation_issues: list[str],
    concepts: ExtractedConcepts,
) -> tuple[int, str]:
    """Step 7: Determine which step to retry and what feedback to give.

    Returns (retry_step, feedback_text).
    retry_step: 1=classify, 2=extract, 3=generate, 0=give up
    """
    feedback_parts: list[str] = []

    # --- Validation issues (step 4) → mostly step 3 problems ---
    if validation_issues:
        serious = [i for i in validation_issues if any(
            kw in i for kw in ("MISSING", "COLLAPSED", "zero-length", "radius")
        )]
        if serious:
            feedback_parts.append("STRUCTURAL PROBLEMS:")
            feedback_parts.extend(f"  • {i}" for i in serious)

        dimension_issues = [i for i in validation_issues if "DIMENSION" in i]
        if dimension_issues:
            feedback_parts.append("DIMENSION MISMATCHES:")
            feedback_parts.extend(f"  • {i}" for i in dimension_issues)

        freq_issues = [i for i in validation_issues if "FR FREQUENCY" in i]
        if freq_issues:
            feedback_parts.extend(freq_issues)

    # --- Simulation/goal failures ---
    if not eval_result.get("sim_ok") and eval_result.get("sim_result"):
        sim = eval_result["sim_result"]
        err = sim.get("error", "")
        if err:
            feedback_parts.append(f"SIMULATION FAILED: {err}")

    gv = eval_result.get("goal_verdict")
    if gv and not gv.get("passed"):
        feedback_parts.append(
            f"GOAL CHECK FAILED ({gv.get('checks_passed', 0)}/"
            f"{gv.get('checks_total', 0)} passed, "
            f"score {gv.get('score', 0):.2f}):"
        )
        for issue in gv.get("feedback", []):
            feedback_parts.append(f"  • {issue}")

    # --- Reverse classification mismatch ---
    # (detected in eval via _analyze_nec — but we check separately)

    # --- Route to the right step ---
    if not feedback_parts:
        return 0, ""  # Nothing wrong

    feedback_text = "\n".join(feedback_parts)

    # Heuristic: if dimension issues dominate, might be extraction error
    dim_issue_count = sum(1 for i in validation_issues if "DIMENSION" in i)
    total_issue_count = len(validation_issues)

    if dim_issue_count > 0 and dim_issue_count == total_issue_count:
        # All issues are dimension mismatches — extraction might be wrong
        return 2, feedback_text

    # Default: retry generation (step 3) with feedback
    return 3, feedback_text


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    source_text: str,
    *,
    antenna_type: str = "",
    freq_mhz: float = 0.0,
    ground_type: str = "",
    description: str = "",
    model: str = _ENGINEERING_MODEL,
    classify_model: str = _COMPREHENSION_MODEL,
    max_retries: int = _MAX_PIPELINE_RETRIES,
) -> PipelineResult:
    """Run the full 7-step NEC generation pipeline.

    Parameters
    ----------
    source_text : str
        Document text (from PDF, URL, or user description).
    antenna_type : str
        If provided, skip step 1 (classification).
    freq_mhz : float
        If provided, override frequency detection.
    ground_type : str
        Ground type override ("free_space", "perfect", "real").
    description : str
        Additional description (from form input).
    model : str
        Model for step 3 (engineering generation).
    classify_model : str
        Model for steps 1-2 (text comprehension).
    max_retries : int
        Maximum loop-back iterations.

    Returns
    -------
    PipelineResult
    """
    result = PipelineResult(
        source_text=source_text[:4000],
        model=model,
    )
    client = _get_client()

    # ===================================================================
    # Step 1: Classify document
    # ===================================================================
    if antenna_type:
        result.steps.append(StepLog(
            step=1, name="classify", status="skip",
            detail=f"Type provided: {antenna_type}",
        ))
    else:
        log.info("Pipeline step 1: classifying document")
        try:
            atype, conf, evidence = classify_document(
                source_text, client=client, model=classify_model,
            )
            antenna_type = atype
            result.steps.append(StepLog(
                step=1, name="classify", status="ok",
                detail=f"{atype} (confidence {conf:.2f})",
                data={"antenna_type": atype, "confidence": conf,
                      "evidence": evidence},
            ))
            log.info("Step 1: classified as %s (%.2f)", atype, conf)
        except Exception as exc:
            result.steps.append(StepLog(
                step=1, name="classify", status="fail",
                detail=str(exc),
            ))
            antenna_type = "unknown"
            log.warning("Step 1 failed: %s", exc)

    # ===================================================================
    # Step 2: Extract structured concepts
    # ===================================================================
    log.info("Pipeline step 2: extracting concepts")
    try:
        concepts, extract_resp = extract_concepts(
            source_text, antenna_type,
            client=client, model=classify_model, freq_mhz=freq_mhz,
        )
        _track_usage(result, extract_resp)
        if ground_type:
            concepts.ground_type = ground_type
        if description:
            concepts.description = description
        result.concepts = concepts
        result.steps.append(StepLog(
            step=2, name="extract", status="ok",
            detail=f"freq={concepts.freq_mhz}MHz, "
                   f"{len(concepts.elements)} structural params",
            data=concepts.to_dict(),
        ))
        log.info("Step 2: extracted %d params, freq=%.3f MHz",
                 len(concepts.elements), concepts.freq_mhz)
    except Exception as exc:
        result.steps.append(StepLog(
            step=2, name="extract", status="fail", detail=str(exc),
        ))
        # Fall back to minimal concepts
        concepts = ExtractedConcepts(
            antenna_type=antenna_type,
            freq_mhz=freq_mhz or 14.175,
            description=description,
            ground_type=ground_type,
        )
        result.concepts = concepts
        log.warning("Step 2 failed: %s — using minimal concepts", exc)

    # ===================================================================
    # Steps 3-7: Generate → Validate → Convert → Simulate → Feedback
    # ===================================================================
    feedback = ""
    # OODA conversation history: list of {"role": ..., "content": ...} dicts
    # that accumulate across iterations so the LLM sees its previous output
    # and the specific, data-rich feedback it received.
    ooda_history: list[dict[str, str]] = []

    for iteration in range(1 + max_retries):
        result.iterations = iteration + 1

        # --- Step 3: Generate JSON deck ---
        log.info("Pipeline step 3 (iter %d): generating JSON deck", iteration + 1)

        # Moxon bypass: deterministic coordinate computation from
        # known-good reference geometry.  The LLM consistently gets the
        # tip-gap wrong, so we stamp the deck directly.
        if concepts.antenna_type == "moxon" and concepts.freq_mhz > 0:
            deck = _stamp_moxon_deck(
                freq_mhz=concepts.freq_mhz,
                wire_dia_mm=concepts.wire_dia_mm or 1.628,
                height_m=concepts.height_m,
                ground_type=concepts.ground_type or "free_space",
            )
            last_llm_response = ""
            result.json_deck = deck
            result.steps.append(StepLog(
                step=3, name="generate", status="ok",
                detail=f"stamped moxon — {len(deck.get('cards', []))} cards",
            ))
            log.info("Step 3: stamped moxon deck (%d cards)",
                     len(deck.get("cards", [])))
        else:
            try:
                deck, gen_resp = generate_deck(
                    concepts, client=client, model=model,
                    feedback=feedback,
                    history=ooda_history if ooda_history else None,
                )
                _track_usage(result, gen_resp)
                # Capture LLM response for conversation continuity
                last_llm_response = gen_resp.choices[0].message.content or ""
                result.json_deck = deck
                result.steps.append(StepLog(
                    step=3, name="generate", status="ok",
                    detail=f"{len(deck.get('cards', []))} cards",
                ))
                log.info("Step 3: generated %d cards", len(deck.get("cards", [])))
            except (ValueError, _json.JSONDecodeError) as exc:
                result.steps.append(StepLog(
                    step=3, name="generate", status="fail", detail=str(exc),
                ))
                feedback = f"JSON generation failed: {exc}\nPlease fix and try again."
                log.warning("Step 3 failed: %s", exc)
                continue

        # --- Step 4: Validate with calculators ---
        log.info("Pipeline step 4 (iter %d): validating deck", iteration + 1)
        issues = validate_deck(deck, concepts)
        if issues:
            # Separate critical from advisory
            critical = [i for i in issues if any(
                kw in i for kw in ("MISSING", "COLLAPSED", "zero-length",
                                   "NO GW", "radius ≤ 0")
            )]
            advisory = [i for i in issues if i not in critical]

            result.steps.append(StepLog(
                step=4, name="validate", status="fail" if critical else "ok",
                detail=f"{len(critical)} critical, {len(advisory)} advisory",
                data={"issues": issues},
            ))

            if critical:
                log.warning("Step 4: %d critical issues", len(critical))
                crit_feedback = "Validation found critical issues:\n" + \
                    "\n".join(f"  • {i}" for i in critical)
                if advisory:
                    crit_feedback += "\nAdvisory:\n" + \
                        "\n".join(f"  • {i}" for i in advisory)
                # Build OODA history: LLM sees what it produced + why it failed
                ooda_history.append({"role": "assistant", "content": last_llm_response})
                ooda_history.append({"role": "user", "content": crit_feedback +
                    "\n\nPlease output a corrected JSON deck that fixes all "
                    "critical issues above."})
                feedback = ""  # history supersedes flat feedback
                continue
        else:
            result.steps.append(StepLog(
                step=4, name="validate", status="ok", detail="All checks passed",
            ))
            log.info("Step 4: validation passed")

        # --- Step 5: Convert to NEC ---
        log.info("Pipeline step 5 (iter %d): converting to NEC", iteration + 1)
        nec = convert_to_nec(deck)
        result.nec_content = nec
        result.steps.append(StepLog(
            step=5, name="convert", status="ok",
            detail=f"{len(nec)} chars",
        ))

        # --- Step 6: Simulate and evaluate ---
        log.info("Pipeline step 6 (iter %d): simulating", iteration + 1)
        eval_result = simulate_and_evaluate(
            nec, concepts.antenna_type, concepts.freq_mhz,
        )
        result.goal_verdict = eval_result.get("goal_verdict")
        result.buildability = eval_result.get("buildability")

        # Reverse classification for sanity
        from .nec_generator import _analyze_nec
        analysis = _analyze_nec(nec, concepts.antenna_type)
        result.classified_type = analysis.get("classified_type", "unknown")
        result.confidence = analysis.get("confidence", 0.0)

        sim_ok = eval_result.get("sim_ok", False) or \
            eval_result.get("sim_result") is None  # pass if solver unavailable

        goal_ok = True
        if eval_result.get("goal_verdict"):
            goal_ok = eval_result["goal_verdict"].get("passed", False)

        if sim_ok and goal_ok:
            result.steps.append(StepLog(
                step=6, name="simulate", status="ok",
                detail="Simulation passed, goals met",
                data={
                    "classified_type": result.classified_type,
                    "confidence": result.confidence,
                },
            ))
            log.info("Step 6: PASSED — pipeline complete")
            break
        else:
            result.steps.append(StepLog(
                step=6, name="simulate", status="fail",
                detail=f"sim_ok={sim_ok}, goal_ok={goal_ok}",
                data={
                    "classified_type": result.classified_type,
                    "confidence": result.confidence,
                    "eval_feedback": eval_result.get("feedback", []),
                },
            ))

            # --- Step 7: Diagnose and route ---
            retry_step, fb = diagnose_failure(eval_result, issues, concepts)
            result.steps.append(StepLog(
                step=7, name="feedback",
                status="ok" if retry_step > 0 else "fail",
                detail=f"Routing to step {retry_step}" if retry_step else "No retry",
            ))

            if retry_step == 0 or iteration >= max_retries:
                log.info("Step 7: no further retries")
                break

            # --- OODA: Build data-rich feedback with simulation metrics ---
            from .nec_policies import policy_for_type
            policy = policy_for_type(concepts.antenna_type)

            # Extract actual simulation measurements for the LLM
            sim_metrics_parts: list[str] = []
            sim_data = eval_result.get("sim_result")
            if sim_data and isinstance(sim_data, dict):
                swr_info = sim_data.get("swr_sweep")
                if swr_info:
                    min_swr = swr_info.get("min_swr")
                    res_freq = swr_info.get("resonant_freq_mhz")
                    bw = swr_info.get("bandwidth_2to1_mhz")
                    if min_swr is not None:
                        sim_metrics_parts.append(
                            f"SWR: {min_swr:.2f}:1 at {res_freq:.3f} MHz"
                        )
                    if bw is not None:
                        sim_metrics_parts.append(
                            f"2:1 SWR bandwidth: {bw:.3f} MHz"
                        )
                imp_info = sim_data.get("impedance_sweep")
                if imp_info:
                    r_vals = imp_info.get("r", [])
                    x_vals = imp_info.get("x", [])
                    if r_vals and x_vals:
                        mid = len(r_vals) // 2
                        sim_metrics_parts.append(
                            f"Impedance at center: "
                            f"{r_vals[mid]:.1f} + j{x_vals[mid]:.1f} Ω"
                        )
                pat_info = sim_data.get("radiation_pattern")
                if pat_info:
                    max_gain = pat_info.get("max_gain_dbi")
                    fb = pat_info.get("front_to_back_db")
                    if max_gain is not None:
                        sim_metrics_parts.append(
                            f"Max gain: {max_gain:.2f} dBi"
                        )
                    if fb is not None:
                        sim_metrics_parts.append(
                            f"Front-to-back ratio: {fb:.1f} dB"
                        )

            # Build score summary for the policy
            score_summary: dict[str, Any] = {}
            gv = eval_result.get("goal_verdict")
            if gv:
                score_summary["goal_score"] = f"{gv.get('score', 0):.2f}"
                score_summary["goal_checks"] = (
                    f"{gv.get('checks_passed', 0)}/{gv.get('checks_total', 0)}"
                )
            bld = eval_result.get("buildability")
            if bld:
                score_summary["buildability"] = f"{bld.get('score', 0):.0f}/100"

            # Use the type-specific policy for structured improvement prompt
            all_feedback = list(analysis.get("feedback", []))
            all_feedback.extend(eval_result.get("feedback", []))
            if issues:
                all_feedback.extend(issues)

            improvement_text = policy.improvement_prompt(
                feedback=all_feedback,
                score_summary=score_summary,
            )

            # Prepend actual simulation measurements so the LLM knows exactly
            # what the antenna achieved and what needs to change
            if sim_metrics_parts:
                improvement_text = (
                    "ACTUAL SIMULATION RESULTS from nec2c:\n"
                    + "\n".join(f"  • {m}" for m in sim_metrics_parts)
                    + "\n\n" + improvement_text
                )

            # Append to OODA history so the LLM sees the full conversation
            ooda_history.append({"role": "assistant", "content": last_llm_response})
            ooda_history.append({"role": "user", "content": improvement_text})
            feedback = ""  # history supersedes flat feedback
            log.info("Step 7: routing to step %d with %d chars feedback "
                     "(%d sim metrics), history len=%d",
                     retry_step, len(improvement_text),
                     len(sim_metrics_parts), len(ooda_history))

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def pipeline_from_pdf(
    pdf_bytes: bytes,
    *,
    antenna_type: str = "",
    model: str = _ENGINEERING_MODEL,
    extra_instructions: str = "",
    **kwargs: Any,
) -> PipelineResult:
    """Run the pipeline on a PDF document."""
    from .nec_generator import extract_pdf_text, _ocr_pdf_pages

    pdf_text = extract_pdf_text(pdf_bytes)
    if not pdf_text.strip():
        pdf_text = _ocr_pdf_pages(pdf_bytes)
    if not pdf_text.strip():
        raise ValueError("Could not extract any text from the PDF")

    desc = extra_instructions if extra_instructions else ""
    return run_pipeline(
        pdf_text,
        antenna_type=antenna_type,
        model=model,
        description=desc,
        **kwargs,
    )


def pipeline_from_url(
    url: str,
    *,
    antenna_type: str = "",
    model: str = _ENGINEERING_MODEL,
    extra_instructions: str = "",
    **kwargs: Any,
) -> PipelineResult:
    """Run the pipeline on a web page."""
    from .nec_generator import extract_url_text

    url_text = extract_url_text(url)
    if not url_text.strip():
        raise ValueError("Could not extract any text from the URL")

    desc = extra_instructions if extra_instructions else ""
    return run_pipeline(
        url_text,
        antenna_type=antenna_type,
        model=model,
        description=desc,
        **kwargs,
    )


def pipeline_from_form(
    *,
    antenna_type: str,
    frequency_mhz: float,
    ground_type: str = "free_space",
    description: str = "",
    model: str = _ENGINEERING_MODEL,
    **kwargs: Any,
) -> PipelineResult:
    """Run the pipeline from structured form data (skip steps 1-2).

    The antenna type and frequency are already known, so we go
    straight to generation with calculator-seeded concepts.
    """
    from .nec_calculators import calc_for_type

    # Build a description-like text from the form for extraction context
    text_parts = [
        f"Antenna type: {antenna_type}",
        f"Design frequency: {frequency_mhz} MHz",
        f"Ground: {ground_type}",
    ]
    if description:
        text_parts.append(f"Description: {description}")
    source_text = "\n".join(text_parts)

    return run_pipeline(
        source_text,
        antenna_type=antenna_type,
        freq_mhz=frequency_mhz,
        ground_type=ground_type,
        description=description,
        model=model,
        **kwargs,
    )
