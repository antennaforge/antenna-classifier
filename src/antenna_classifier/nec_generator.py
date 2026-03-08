"""AI-powered NEC file generator using OpenAI.

Given an antenna description (from a form or extracted from a PDF),
calls the OpenAI chat completions API to produce a valid NEC2 input
deck.  The output is validated with the project parser + validator
before being returned.

The OODA refinement loop layers:
  1. Structural validation (parse + validate)
  2. Reverse classification (classify + fingerprint → type match)
  3. Simulation verification (nec2c → goals check)
  4. Buildability assessment (geometry → practical score)
  5. Type-specific policy (structured improvement prompts)
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# System prompt — teaches the model NEC2 format
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert antenna engineer who generates NEC2 input files.

**Rules:**
1. Output ONLY the NEC2 input deck — no markdown fences, no commentary.
2. Start with CM (comment) cards describing the antenna.
3. Include CE to end comments.
4. Use GW cards for wire elements (tag, segments, x1 y1 z1 x2 y2 z2, radius).
5. Use GE to end geometry (GE 0 for free-space, GE 1 for ground present).
6. Use EX card for excitation (voltage source on driven element).
7. Use FR card for the design frequency.
8. If ground is requested, add GN card (GN 1 for perfect, GN 2 for real ground).
9. Add RP card for radiation pattern: RP 0,91,1,1000,0,0,1,0 for a simple elevation cut.
10. End with EN card.
11. Use realistic dimensions in metres for the requested frequency.
12. Wire radius should be realistic (e.g. 0.001 m for #14 AWG).
13. Segment count: ~10-20 segments per half-wavelength.
14. Coordinate system: Z axis is UP (height above ground).  X and Y are the
    horizontal plane.  Every GW card MUST have physically correct 3-D
    coordinates:
    - Horizontal elements: extend along X or Y at a fixed Z height.
      Example dipole at 10 m height along Y:
        GW 1 21  0.0  -2.5  10.0   0.0  2.5  10.0  0.001
    - Vertical elements: extend along Z.
      Example vertical from ground to 5 m:
        GW 1 11  0.0  0.0  0.0   0.0  0.0  5.0  0.001
    - Multi-element antennas: elements MUST be at different positions.
      Space them apart along the appropriate axis.  Do NOT place all
      wires at the origin.
      Example 3-element beam along X at 10 m height:
        GW 1 21  0.000  -2.6  10.0   0.000  2.6  10.0  0.001
        GW 2 21  0.700  -2.5  10.0   0.700  2.5  10.0  0.001
        GW 3 21  1.600  -2.4  10.0   1.600  2.4  10.0  0.001
15. When element lengths and spacings are given in inches or feet, convert
    them to metres (1 inch = 0.0254 m, 1 foot = 0.3048 m).  If a "total
    length" is given, the half-length extends ±Y (or ±X) from the boom.
16. If the source document describes a transmission line, phase line, or
    phasing stub connecting two elements, you MUST model it with a TL card.
    TL cards go AFTER GE and BEFORE EX.  Format:
      TL tag1 seg1 tag2 seg2 Z0 length VR1 VI1 VR2 VI2
    where tag1/seg1 and tag2/seg2 are the wire endpoints, Z0 is the line
    impedance in ohms, and length is the physical length in metres (0 = use
    NEC-computed distance).  Example — 250 Ω phase line between wires 1 and 2:
      TL 1 11 2 11 250.0 0.0 0.0 0.0 0.0 0.0

**NEC2 card format reference:**
- CM <text>
- CE
- GW tag segs x1 y1 z1 x2 y2 z2 radius
- GE ground_type
- TL tag1 seg1 tag2 seg2 Z0 length VR1 VI1 VR2 VI2
- EX ex_type tag seg v_real v_imag ...
- FR fr_type n_freq 0 0 start_mhz step_mhz
- GN gn_type ...
- RP rp_type ntheta nphi mode theta_start phi_start theta_step phi_step
- EN

Output a complete, runnable NEC2 deck now.
"""


# ---------------------------------------------------------------------------
# JSON-intermediate system prompt
# ---------------------------------------------------------------------------

_JSON_SYSTEM_PROMPT = """\
You are an expert antenna engineer who generates NEC2 input files in a
structured JSON format.  The JSON is then mechanically translated to NEC2
cards, so your output MUST be a valid JSON object — nothing else.

**Output format:**
Return a JSON object with a single key "cards" containing an array of card
objects.  Each card object has:
  - "type": two-letter NEC card code (e.g. "GW", "FR", "EX")
  - "params": array of numbers (integers or floats) for the card fields
  - "text": string (only for CM, CE, SY cards — the comment or variable text)

**Card parameter order (same as NEC2):**
- CM: {"type":"CM", "text":"<comment>"}
- CE: {"type":"CE"}
- GW: {"type":"GW", "params":[tag, segments, x1, y1, z1, x2, y2, z2, radius]}
- GE: {"type":"GE", "params":[ground_type]}
- TL: {"type":"TL", "params":[tag1, seg1, tag2, seg2, Z0, length, VR1, VI1, VR2, VI2]}
- EX: {"type":"EX", "params":[ex_type, tag, segment, 0, v_real, v_imag]}
- FR: {"type":"FR", "params":[fr_type, n_freq, 0, 0, start_mhz, step_mhz]}
- GN: {"type":"GN", "params":[gn_type, ...]}
- LD: {"type":"LD", "params":[ld_type, tag, seg_start, seg_end, R, L_or_X, C_or_B]}
- RP: {"type":"RP", "params":[rp_type, ntheta, nphi, mode, theta_start, phi_start, theta_step, phi_step]}
- EN: {"type":"EN"}

**Geometry rules:**
1. Coordinate system: Z is UP.  X and Y are horizontal.
2. Horizontal elements extend along X or Y at a fixed Z height.
3. Vertical elements extend along Z.
4. Multi-element antennas: elements at DIFFERENT positions.  Do NOT place
   all wires at the origin.
5. Use realistic dimensions in metres.
6. Wire radius: realistic (e.g. 0.001 for #14 AWG, 8.14e-4 for #12 AWG).
7. Segments: ~10-20 per half-wavelength.
8. When element lengths are given in inches or feet, convert to metres
   (1 inch = 0.0254 m, 1 foot = 0.3048 m).
9. If the source describes transmission lines or phasing stubs, model
   them with TL cards (placed after GE, before EX).
10. Always start with CM comments, then CE, then geometry (GW), then GE,
    then control cards (TL, EX, LD, FR, GN, RP), then EN.

**Example — 20m dipole at 10 m height:**
```json
{
  "cards": [
    {"type":"CM", "text":"20m dipole at 10m height"},
    {"type":"CE"},
    {"type":"GW", "params":[1, 21, 0.0, -5.05, 10.0, 0.0, 5.05, 10.0, 0.001]},
    {"type":"GE", "params":[0]},
    {"type":"EX", "params":[0, 1, 11, 0, 1.0, 0.0]},
    {"type":"FR", "params":[0, 1, 0, 0, 14.175, 0]},
    {"type":"RP", "params":[0, 91, 1, 1000, 0.0, 0.0, 1.0, 0.0]},
    {"type":"EN"}
  ]
}
```

Output ONLY the JSON object.  No markdown fences, no commentary.
"""


def _read_secret(name: str, env_fallback: str = "") -> str:
    """Read a Docker secret file, falling back to an env var."""
    secret_path = f"/run/secrets/{name}"
    try:
        with open(secret_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    # Fallback: env var pointing to a file, or plain value
    file_env = os.getenv(f"{env_fallback}_FILE", "")
    if file_env:
        try:
            with open(file_env) as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
    return os.getenv(env_fallback, "")


def _get_client():
    """Lazy-import and build an OpenAI client."""
    from openai import OpenAI

    api_key = _read_secret("openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Provide it via Docker secret "
            "(/run/secrets/openai_api_key) or OPENAI_API_KEY env var."
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Per-type NEC context documents
# ---------------------------------------------------------------------------

_NEC_CONTEXT_DIR = pathlib.Path(__file__).parent / "nec_context"

# Band-centre frequencies (MHz) keyed by ham-band name fragments
_BAND_CENTRES: dict[str, float] = {
    "160": 1.85, "80": 3.6, "60": 5.35, "40": 7.1, "30": 10.125,
    "20": 14.175, "17": 18.118, "15": 21.2, "12": 24.94,
    "10": 28.4, "6": 50.15, "2": 146.0,
}


def _guess_freq_mhz(text: str, antenna_type: str = "") -> float:
    """Try to extract a design frequency from free text.

    Priority order:
    1. Ham-band reference in the title/first 200 chars ("10 Meters")
    2. Frequency range ("28 to 29 MHz" → centre 28.5)
    3. Explicit MHz value ("28.5 MHz")
    4. Ham-band reference anywhere in text
    5. Default 14.175 (20 m)
    """
    # 1. Band reference in the title / first 200 chars — strongest signal
    title = text[:200]
    m = re.search(r"\b(\d{1,3})\s*[-]?\s*[Mm](?:eter|etre)?s?\b", title)
    if m and m.group(1) in _BAND_CENTRES:
        return _BAND_CENTRES[m.group(1)]

    # 2. Frequency range (e.g. "28 to 29 MHz", "28-29 MHz", "28 – 29 MHz")
    m = re.search(
        r"(\d{1,4}(?:\.\d+)?)\s*(?:to|[-\u2013\u2014])\s*(\d{1,4}(?:\.\d+)?)\s*[Mm][Hh][Zz]",
        text,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0.5 < lo < 1500 and 0.5 < hi < 1500:
            return round((lo + hi) / 2, 3)

    # 3. Explicit MHz value (e.g. "14.175 MHz", "28.5MHz", "146 mhz")
    m = re.search(r"(\d{1,4}(?:\.\d+)?)\s*[Mm][Hh][Zz]", text)
    if m:
        val = float(m.group(1))
        if 0.5 < val < 1500:
            return val

    # 4. Ham-band reference anywhere (e.g. "20m band", "10-meter", "15 m")
    m = re.search(r"\b(\d{1,3})\s*[-]?\s*[Mm](?:eter|etre)?s?\b", text)
    if m and m.group(1) in _BAND_CENTRES:
        return _BAND_CENTRES[m.group(1)]

    return 14.175  # 20 m default


# ---------------------------------------------------------------------------
# Design-goal extraction from document text
# ---------------------------------------------------------------------------

@dataclass
class DocumentGoals:
    """Design goals extracted from a PDF or web page."""

    freq_mhz: float = 0.0          # primary design frequency
    bands: list[str] = field(default_factory=list)  # band labels found
    gain_dbi: float | None = None   # stated gain in dBi (converted from dBd if needed)
    fb_db: float | None = None      # front-to-back ratio in dB
    max_swr: float | None = None    # target SWR (e.g. 1.5)

    def prompt_block(self) -> str:
        """Format goals as an LLM prompt block.  Empty string if nothing found."""
        lines: list[str] = []
        if self.gain_dbi is not None:
            lines.append(f"  • Target gain: {self.gain_dbi:.1f} dBi")
        if self.fb_db is not None:
            lines.append(f"  • Target front-to-back ratio: ≥ {self.fb_db:.0f} dB")
        if self.max_swr is not None:
            lines.append(f"  • Target SWR: ≤ {self.max_swr:.1f}:1")
        if self.bands:
            lines.append(f"  • Bands: {', '.join(self.bands)}")
        if not lines:
            return ""
        header = "DESIGN GOALS (from the source document):\n"
        return header + "\n".join(lines) + "\n"


def _extract_design_goals(text: str) -> DocumentGoals:
    """Extract design goals (gain, F/B, SWR, bands) from document text."""
    goals = DocumentGoals()

    # --- Gain (dBi or dBd → dBi) ---
    gains: list[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*dBi", text):
        gains.append(float(m.group(1)))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*dBd", text):
        gains.append(float(m.group(1)) + 2.15)  # convert dBd → dBi
    if gains:
        # Pick the most commonly mentioned value, or the first one
        goals.gain_dbi = round(max(set(gains), key=gains.count), 1)

    # --- Front-to-back ratio ---
    fbs: list[float] = []
    # "12 dB front-to-back", "21 dB F/B"
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*dB\s*(?:front.to.back|F/?B|f/?b)", text
    ):
        fbs.append(float(m.group(1)))
    # "front-to-back ratio of 25 dB", "F/B of 20 dB"
    for m in re.finditer(
        r"(?:front.to.back|F/?B|f/?b)\s*(?:ratio)?\s*(?:of|:|\s)\s*"
        r"(\d+(?:\.\d+)?)\s*dB",
        text,
    ):
        fbs.append(float(m.group(1)))
    if fbs:
        goals.fb_db = round(max(fbs), 1)  # take highest stated F/B as target

    # --- SWR target ---
    swrs: list[float] = []
    # "1.5:1 SWR", "1.5:1 VSWR"
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*:\s*1\s*(?:SWR|VSWR)", text, re.IGNORECASE
    ):
        swrs.append(float(m.group(1)))
    # "SWR below 1.5:1", "VSWR of 2:1"
    for m in re.finditer(
        r"(?:SWR|VSWR)\s*(?:of|below|under|<|less\s+than)?\s*"
        r"(\d+(?:\.\d+)?)\s*:\s*1",
        text,
        re.IGNORECASE,
    ):
        swrs.append(float(m.group(1)))
    if swrs:
        # Take the tightest (lowest) SWR spec that's realistic (≥ 1.1)
        valid = [s for s in swrs if s >= 1.1]
        if valid:
            goals.max_swr = round(min(valid), 1)

    # --- Band references ---
    band_pattern = re.compile(r"\b(\d{1,3})\s*[-]?\s*[Mm](?:eter|etre)?s?\b")
    seen: set[str] = set()
    for m in band_pattern.finditer(text):
        label = m.group(1)
        if label in _BAND_CENTRES and label not in seen:
            seen.add(label)
            goals.bands.append(f"{label}m")

    return goals


def _load_type_context(antenna_type: str) -> str:
    """Load the NEC modelling reference for *antenna_type*, if available.

    Returns the file contents as a string, or an empty string if no
    context document exists for the requested type.
    """
    path = _NEC_CONTEXT_DIR / f"{antenna_type}.txt"
    if path.is_file():
        return path.read_text()
    return ""


log = logging.getLogger(__name__)

# Maximum refinement iterations (initial generation + correction rounds)
_MAX_REFINE = 3
# Minimum classifier confidence to accept a type match
_CONFIDENCE_THRESHOLD = 0.6

# Minimum buildability score to accept without flagging
_BUILDABILITY_WARN = 40.0

# Minimum goal score to accept (0–1)
_GOAL_SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Simulation from NEC text (tempfile bridge)
# ---------------------------------------------------------------------------

def _extract_freq_from_nec(nec: str) -> float:
    """Extract design frequency (MHz) from the FR card."""
    for line in nec.splitlines():
        parts = re.split(r"[,\s]+", line.strip())
        if parts and parts[0].upper() == "FR":
            try:
                return float(parts[5])
            except (ValueError, IndexError):
                pass
    return 0.0


def _simulate_nec_text(nec: str) -> "SimulationResult | None":
    """Run nec2c simulation on NEC text via the solver API.

    Writes to a temp file, calls simulate(), returns result.
    Returns None if the solver is unavailable.
    """
    from .simulator import simulate

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".nec", mode="w", delete=False,
        ) as f:
            f.write(nec)
            tmp_path = f.name
        result = simulate(tmp_path)
        return result
    except Exception as exc:
        log.warning("Simulation unavailable: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def _evaluate_full(
    nec: str,
    antenna_type: str,
    freq_mhz: float = 0.0,
) -> dict[str, Any]:
    """Run simulation + goals + buildability on a NEC string.

    Returns a dict with:
      - sim_ok (bool)
      - sim_result (dict | None)
      - goal_verdict (dict | None)
      - buildability (dict | None)
      - feedback (list[str]) — human-readable issues for the LLM
      - passed (bool) — all layers acceptable
    """
    from .nec_goals import goals_for_type, evaluate_goals
    from .nec_buildability import assess_buildability

    if not freq_mhz:
        freq_mhz = _extract_freq_from_nec(nec)

    feedback: list[str] = []
    result: dict[str, Any] = {
        "sim_ok": False,
        "sim_result": None,
        "goal_verdict": None,
        "buildability": None,
        "feedback": feedback,
        "passed": False,
    }

    # --- Simulation ---
    sim = _simulate_nec_text(nec)
    if sim is None:
        # Solver not running — skip simulation layer, still assess buildability
        feedback.append("NOTE: NEC solver unavailable — skipping simulation verification.")
    elif not sim.ok:
        feedback.append(f"SIMULATION FAILED: {sim.error}")
        result["sim_result"] = sim.to_dict() if sim else None
        # Buildability can still be assessed from geometry alone
    else:
        result["sim_ok"] = True
        result["sim_result"] = sim.to_dict()

        # --- Goal evaluation ---
        goals = goals_for_type(antenna_type)
        verdict = evaluate_goals(goals, sim, nec_text=nec, freq_mhz=freq_mhz)
        result["goal_verdict"] = verdict.to_dict()

        if not verdict.passed:
            feedback.append(
                f"GOAL CHECK FAILED ({verdict.checks_passed}/{verdict.checks_total} "
                f"passed, score {verdict.score:.2f}):"
            )
            for issue in verdict.feedback:
                feedback.append(f"  • {issue}")

    # --- Buildability ---
    build = assess_buildability(nec, antenna_type, freq_mhz)
    result["buildability"] = build.to_dict()

    if build.score < _BUILDABILITY_WARN:
        feedback.append(
            f"BUILDABILITY WARNING: Score {build.score}/100 ({build.grade}). "
            f"Top risks: {'; '.join(build.top_risks[:2])}"
        )

    # --- Overall pass ---
    goal_ok = True
    if result.get("goal_verdict"):
        goal_ok = result["goal_verdict"].get("passed", False)
    sim_ok = result["sim_ok"] or sim is None  # pass if solver unavailable

    result["passed"] = sim_ok and goal_ok
    return result


# ---------------------------------------------------------------------------
# Reverse-classification analysis
# ---------------------------------------------------------------------------

def _analyze_nec(nec: str, target_type: str | None) -> dict[str, Any]:
    """Parse, validate, classify, and fingerprint a generated NEC string.

    Returns a dict with analysis results:
      - valid (bool): passes structural validation
      - type_match (bool): classified type matches target
      - classified_type (str)
      - confidence (float)
      - evidence (list[str])
      - fingerprint (dict)
      - feedback (list[str]): human-readable issues for the LLM
    """
    from . import parser, validator, classifier
    from .fingerprint import fingerprint as make_fp

    feedback: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "type_match": False,
        "classified_type": "unknown",
        "confidence": 0.0,
        "evidence": [],
        "fingerprint": {},
        "feedback": feedback,
    }

    # --- Parse ---
    try:
        parsed = parser.parse_text(nec)
    except Exception as exc:
        feedback.append(f"NEC PARSE ERROR: {exc}")
        return result

    # --- Validate ---
    vr = validator.validate(parsed)
    if not vr.valid:
        for issue in vr.errors:
            loc = f" (line {issue.line})" if issue.line else ""
            feedback.append(f"VALIDATION ERROR{loc}: {issue.message}")
    for issue in vr.warnings:
        feedback.append(f"WARNING: {issue.message}")
    result["valid"] = vr.valid

    # --- Classify ---
    cls = classifier.classify(parsed)
    result["classified_type"] = cls.antenna_type
    result["confidence"] = cls.confidence
    result["evidence"] = list(cls.evidence)

    # --- Fingerprint ---
    fp = make_fp(parsed)
    result["fingerprint"] = {
        "n_gw": fp.n_gw, "n_tl": fp.n_tl, "n_ex": fp.n_ex,
        "n_ld": fp.n_ld, "n_tags": fp.n_tags,
        "signature": fp.signature,
    }

    # --- Type-match check ---
    if target_type:
        if cls.antenna_type == target_type and cls.confidence >= _CONFIDENCE_THRESHOLD:
            result["type_match"] = True
        else:
            feedback.append(
                f"CLASSIFICATION MISMATCH: Target antenna type is '{target_type}', "
                f"but your output classifies as '{cls.antenna_type}' "
                f"(confidence {cls.confidence:.2f}).\n"
                f"  Classifier evidence: {cls.evidence}\n"
                f"  Fingerprint: {fp.n_gw} GW wires, {fp.n_tl} TL cards, "
                f"{fp.n_ex} EX sources, {fp.n_tags} unique tags.\n"
                f"  Please restructure the geometry so it matches a "
                f"'{target_type}' antenna."
            )
    else:
        # No explicit target — accept any confident, non-trivial classification
        result["type_match"] = (
            cls.confidence >= 0.5
            and cls.antenna_type not in ("unknown", "wire_object")
        )

    return result


# ---------------------------------------------------------------------------
# JSON → NEC conversion (from antenna-agent-model/fidelity_checker.py)
# ---------------------------------------------------------------------------

import json as _json


def _json_to_nec(parsed_json: dict[str, Any]) -> str:
    """Convert a parsed-NEC JSON structure to a NEC2 deck string.

    Expects ``{"cards": [{"type": "GW", "params": [...], "text": "..."}, ...]}``
    — the same schema used by the antenna-agent-model translator pipeline.
    """
    lines: list[str] = []
    for card in parsed_json.get("cards", []):
        typ = card.get("type", "")
        params = card.get("params", [])

        if typ in ("CM", "CE"):
            text = card.get("text", "")
            line = f"{typ} {text}" if text else typ
        elif typ == "SY":
            text = card.get("text", "")
            line = f"{typ} {text}" if text else typ
        else:
            param_strs: list[str] = []
            for p in params:
                if isinstance(p, float):
                    if abs(p) < 0.01 and p != 0:
                        param_strs.append(f"{p:.7g}")
                    else:
                        param_strs.append(str(p))
                else:
                    param_strs.append(str(p))
            line = f"{typ} {','.join(param_strs)}" if param_strs else typ

        lines.append(line)
    return "\n".join(lines) + "\n"


def _extract_json_deck(raw: str) -> dict[str, Any]:
    """Extract a JSON antenna deck from the LLM response.

    Handles markdown fences (```json ... ```) and strips trailing chat.
    Returns the parsed dict.  Raises ``ValueError`` on failure.
    """
    # Try to find a JSON block in markdown fences
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    text = m.group(1).strip() if m else raw.strip()

    # Find the outermost { ... } in case there's trailing chat
    start = text.find("{")
    if start == -1:
        raise ValueError("LLM response does not contain a JSON object")
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
        raise ValueError("LLM response has unbalanced JSON braces")

    try:
        deck = _json.loads(text[start:end])
    except _json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    if "cards" not in deck or not isinstance(deck["cards"], list):
        raise ValueError("JSON response missing 'cards' array")

    # Basic sanity: every card needs a "type"
    for i, card in enumerate(deck["cards"]):
        if "type" not in card:
            raise ValueError(f"Card {i} missing 'type' field")

    return deck


def _generate_and_refine(
    messages: list[dict[str, str]],
    target_type: str | None,
    model: str = "gpt-5.2",
    max_iterations: int = _MAX_REFINE,
    freq_mhz: float = 0.0,
) -> dict[str, Any]:
    """Generate a NEC file and iteratively refine it via the 5-layer OODA loop.

    Layers (applied in order, each iteration):
      1. Structural validation (parse errors → immediate retry)
      2. Reverse classification (type match + fingerprint)
      3. Simulation verification (nec2c → goals check)
      4. Buildability assessment (practical construction score)
      5. Type-specific policy (structured improvement prompt)

    *messages* is the initial ``[system, user]`` message list.
    *target_type* is the expected antenna type (or ``None`` for best-effort).

    Returns a dict with keys: nec_content, usage, iterations, refinement_log,
    classified_type, confidence, goal_verdict, buildability.
    """
    from .nec_policies import policy_for_type

    client = _get_client()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    refinement_log: list[dict[str, Any]] = []
    nec = ""
    analysis: dict[str, Any] = {}
    eval_result: dict[str, Any] = {}
    finish_reason = ""

    # Work on a mutable copy so callers' list isn't modified
    msgs = [dict(m) for m in messages]

    # Get the improvement policy for this type (used for structured feedback)
    policy = policy_for_type(target_type or "generic")

    for iteration in range(max_iterations):
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.3,
            max_completion_tokens=8192,
        )
        content = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason or ""
        if resp.usage:
            total_usage["prompt_tokens"] += resp.usage.prompt_tokens
            total_usage["completion_tokens"] += resp.usage.completion_tokens

        nec = _extract_nec(content)

        # --- Layer 1: Lightweight structural check ---
        try:
            _validate_nec(nec)
        except ValueError as exc:
            log.info("Refinement iter %d: structural error — %s", iteration + 1, exc)
            refinement_log.append({
                "iteration": iteration + 1,
                "layer": "structural",
                "issue": str(exc),
                "passed": False,
            })
            if iteration < max_iterations - 1:
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": (
                    f"Your output has a structural error:\n  {exc}\n"
                    "Please output a corrected, complete NEC2 deck."
                )})
            continue

        # --- Layer 2: Reverse-classification analysis ---
        analysis = _analyze_nec(nec, target_type)
        log.info(
            "Refinement iter %d: classified=%s conf=%.2f valid=%s type_match=%s",
            iteration + 1, analysis["classified_type"],
            analysis["confidence"], analysis["valid"], analysis["type_match"],
        )

        classification_passed = analysis["valid"] and analysis["type_match"]

        if not classification_passed:
            refinement_log.append({
                "iteration": iteration + 1,
                "layer": "classification",
                "classified_type": analysis["classified_type"],
                "confidence": analysis["confidence"],
                "evidence": analysis["evidence"],
                "fingerprint": analysis["fingerprint"],
                "passed": False,
            })
            # Feed classification feedback and retry
            if analysis["feedback"] and iteration < max_iterations - 1:
                feedback_text = (
                    "I analysed your NEC output with our antenna classifier and "
                    "found these issues:\n\n"
                    + "\n".join(f"• {f}" for f in analysis["feedback"])
                    + "\n\nPlease output a corrected NEC2 deck that addresses "
                    "all issues above. Output ONLY the NEC deck, no commentary."
                )
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": feedback_text})
            continue

        # --- Layers 3+4: Simulation + Goals + Buildability ---
        detected_type = target_type or analysis["classified_type"]
        eval_result = _evaluate_full(nec, detected_type, freq_mhz)

        log.info(
            "Refinement iter %d: sim_ok=%s goal_passed=%s buildability=%.0f",
            iteration + 1,
            eval_result.get("sim_ok"),
            eval_result.get("passed"),
            (eval_result.get("buildability") or {}).get("score", 0),
        )

        iter_log: dict[str, Any] = {
            "iteration": iteration + 1,
            "layer": "full_evaluation",
            "classified_type": analysis["classified_type"],
            "confidence": analysis["confidence"],
            "evidence": analysis["evidence"],
            "fingerprint": analysis["fingerprint"],
            "sim_ok": eval_result.get("sim_ok"),
            "goal_verdict": eval_result.get("goal_verdict"),
            "buildability_score": (eval_result.get("buildability") or {}).get("score"),
            "passed": eval_result.get("passed", False),
        }
        refinement_log.append(iter_log)

        if eval_result.get("passed", False):
            break

        # --- Layer 5: Policy-driven structured feedback ---
        if eval_result.get("feedback") and iteration < max_iterations - 1:
            all_feedback = list(analysis.get("feedback", []))
            all_feedback.extend(eval_result.get("feedback", []))

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
            improvement_text = policy.improvement_prompt(
                feedback=all_feedback,
                score_summary=score_summary,
            )
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": improvement_text})

    return {
        "nec_content": nec,
        "usage": {**total_usage, "finish_reason": finish_reason},
        "iterations": len(refinement_log),
        "refinement_log": refinement_log,
        "classified_type": analysis.get("classified_type", "unknown"),
        "confidence": analysis.get("confidence", 0.0),
        "goal_verdict": eval_result.get("goal_verdict"),
        "buildability": eval_result.get("buildability"),
    }


# ---------------------------------------------------------------------------
# JSON-intermediate generation and refinement
# ---------------------------------------------------------------------------

def _generate_and_refine_json(
    messages: list[dict[str, str]],
    target_type: str | None,
    model: str = "gpt-5.2",
    max_iterations: int = _MAX_REFINE,
    freq_mhz: float = 0.0,
) -> dict[str, Any]:
    """Generate a NEC file via JSON intermediate representation.

    Same 5-layer OODA loop as ``_generate_and_refine``, but the LLM
    produces a structured JSON deck which is mechanically translated to
    NEC text.  This gives cleaner NEC output because:
      - The LLM focuses on geometry values, not card formatting
      - ``json_to_nec()`` produces correctly-formatted NEC every time
      - No SY resolution needed (LLM outputs resolved numeric values)
    """
    from .nec_policies import policy_for_type

    client = _get_client()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    refinement_log: list[dict[str, Any]] = []
    nec = ""
    json_deck: dict[str, Any] = {}
    analysis: dict[str, Any] = {}
    eval_result: dict[str, Any] = {}
    finish_reason = ""

    msgs = [dict(m) for m in messages]
    policy = policy_for_type(target_type or "generic")

    for iteration in range(max_iterations):
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.3,
            max_completion_tokens=8192,
        )
        content = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason or ""
        if resp.usage:
            total_usage["prompt_tokens"] += resp.usage.prompt_tokens
            total_usage["completion_tokens"] += resp.usage.completion_tokens

        # --- Layer 0: JSON extraction + NEC conversion ---
        try:
            json_deck = _extract_json_deck(content)
            nec = _json_to_nec(json_deck)
        except ValueError as exc:
            log.info("Refinement iter %d: JSON parse error — %s", iteration + 1, exc)
            refinement_log.append({
                "iteration": iteration + 1,
                "layer": "json_parse",
                "issue": str(exc),
                "passed": False,
            })
            if iteration < max_iterations - 1:
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": (
                    f"Your output could not be parsed as valid JSON:\n  {exc}\n"
                    "Please output a corrected JSON object with the same schema. "
                    "Output ONLY the JSON — no markdown, no commentary."
                )})
            continue

        # --- Layer 1: Structural validation ---
        try:
            _validate_nec(nec)
        except ValueError as exc:
            log.info("Refinement iter %d: structural error — %s", iteration + 1, exc)
            refinement_log.append({
                "iteration": iteration + 1,
                "layer": "structural",
                "issue": str(exc),
                "passed": False,
            })
            if iteration < max_iterations - 1:
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": (
                    f"The NEC deck generated from your JSON has a structural error:\n"
                    f"  {exc}\n"
                    "Please fix the JSON cards and output the corrected JSON object."
                )})
            continue

        # --- Layer 2: Reverse-classification analysis ---
        analysis = _analyze_nec(nec, target_type)
        log.info(
            "Refinement iter %d (JSON): classified=%s conf=%.2f valid=%s type_match=%s",
            iteration + 1, analysis["classified_type"],
            analysis["confidence"], analysis["valid"], analysis["type_match"],
        )

        classification_passed = analysis["valid"] and analysis["type_match"]

        if not classification_passed:
            refinement_log.append({
                "iteration": iteration + 1,
                "layer": "classification",
                "classified_type": analysis["classified_type"],
                "confidence": analysis["confidence"],
                "evidence": analysis["evidence"],
                "fingerprint": analysis["fingerprint"],
                "passed": False,
            })
            if analysis["feedback"] and iteration < max_iterations - 1:
                feedback_text = (
                    "I analysed the NEC output generated from your JSON with "
                    "our antenna classifier and found these issues:\n\n"
                    + "\n".join(f"• {f}" for f in analysis["feedback"])
                    + "\n\nPlease output a corrected JSON deck that addresses "
                    "all issues above. Output ONLY the JSON object."
                )
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": "user", "content": feedback_text})
            continue

        # --- Layers 3+4: Simulation + Goals + Buildability ---
        detected_type = target_type or analysis["classified_type"]
        eval_result = _evaluate_full(nec, detected_type, freq_mhz)

        log.info(
            "Refinement iter %d (JSON): sim_ok=%s goal_passed=%s buildability=%.0f",
            iteration + 1,
            eval_result.get("sim_ok"),
            eval_result.get("passed"),
            (eval_result.get("buildability") or {}).get("score", 0),
        )

        iter_log: dict[str, Any] = {
            "iteration": iteration + 1,
            "layer": "full_evaluation",
            "classified_type": analysis["classified_type"],
            "confidence": analysis["confidence"],
            "evidence": analysis["evidence"],
            "fingerprint": analysis["fingerprint"],
            "sim_ok": eval_result.get("sim_ok"),
            "goal_verdict": eval_result.get("goal_verdict"),
            "buildability_score": (eval_result.get("buildability") or {}).get("score"),
            "passed": eval_result.get("passed", False),
        }
        refinement_log.append(iter_log)

        if eval_result.get("passed", False):
            break

        # --- Layer 5: Policy-driven structured feedback ---
        if eval_result.get("feedback") and iteration < max_iterations - 1:
            all_feedback = list(analysis.get("feedback", []))
            all_feedback.extend(eval_result.get("feedback", []))

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

            improvement_text = policy.improvement_prompt(
                feedback=all_feedback,
                score_summary=score_summary,
            )
            # Remind LLM to stay in JSON mode
            improvement_text += (
                "\n\nRemember: output ONLY a JSON object with 'cards' array. "
                "No NEC text, no markdown, no commentary."
            )
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "user", "content": improvement_text})

    return {
        "nec_content": nec,
        "json_deck": json_deck,
        "usage": {**total_usage, "finish_reason": finish_reason},
        "iterations": len(refinement_log),
        "refinement_log": refinement_log,
        "classified_type": analysis.get("classified_type", "unknown"),
        "confidence": analysis.get("confidence", 0.0),
        "goal_verdict": eval_result.get("goal_verdict"),
        "buildability": eval_result.get("buildability"),
    }


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 20000) -> str:
    """Extract plain text and tables from a PDF.

    Uses pdfplumber to pull both free text and tabular data.  Tables
    are formatted as pipe-delimited rows so the LLM can parse
    dimensional data that pure ``extract_text()`` often mangles in
    multi-column magazine layouts.
    """
    import pdfplumber

    text_parts: list[str] = []
    total = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # --- free text ---
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            total += len(page_text)

            # --- structured tables ---
            for table in (page.extract_tables() or []):
                formatted_rows: list[str] = []
                for row in table:
                    cells = [c.strip() if c else "" for c in row]
                    if any(cells):
                        formatted_rows.append(" | ".join(cells))
                if formatted_rows:
                    block = "\n[TABLE]\n" + "\n".join(formatted_rows) + "\n[/TABLE]\n"
                    text_parts.append(block)
                    total += len(block)

            if total >= max_chars:
                break

    full = "\n".join(text_parts)
    # Clean common PDF artefacts
    full = re.sub(r"\(cid:\d+\)", "•", full)  # replace CID placeholders
    full = re.sub(r"[ \t]{3,}", "  ", full)     # collapse excessive spaces
    return full[:max_chars]


def _ocr_pdf_pages(
    pdf_bytes: bytes,
    *,
    max_pages: int = 6,
    max_chars: int = 20000,
) -> str:
    """OCR a scanned PDF by sending page images to GPT-4o-mini vision.

    Falls back gracefully: returns empty string if the API call fails
    or no readable text is found.
    """
    import base64
    import pdfplumber

    log = logging.getLogger(__name__)
    log.info("PDF has no extractable text — attempting vision OCR")

    # Render pages to JPEG and collect as base64 data URIs
    image_parts: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            try:
                pimg = page.to_image(resolution=200)
                buf = io.BytesIO()
                pimg.original.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                })
            except Exception:
                log.warning("Could not render page %d to image", i)

    if not image_parts:
        return ""

    client = _get_client()
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "These are pages from a technical document about an antenna "
                "design.  I need to extract the antenna dimensions and "
                "specifications to create a computer simulation model "
                "(NEC2 format).\n\n"
                "Please extract: antenna type, frequency/band, element "
                "dimensions (lengths, spacings, wire sizes), height above "
                "ground, and any construction details with measurements.  "
                "Reproduce any dimensional tables as pipe-delimited rows.  "
                "Focus on numerical data, coordinates, and specifications."
            ),
        },
        *image_parts,
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
        )
        text = resp.choices[0].message.content or ""
        log.info("Vision OCR returned %d chars from %d page(s)",
                 len(text), len(image_parts))
        return text[:max_chars]
    except Exception as exc:
        log.warning("Vision OCR failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# NEC generation from form data
# ---------------------------------------------------------------------------

def generate_nec_from_form(
    *,
    antenna_type: str,
    frequency_mhz: float,
    ground_type: str = "free_space",
    description: str = "",
    model: str = "gpt-5.2",
    json_mode: bool = False,
    pipeline: bool = False,
) -> dict[str, Any]:
    """Generate a NEC file from structured form data.

    When *pipeline* is True the structured 7-step pipeline is used
    (classify → extract → generate → validate → convert → simulate →
    feedback).  When *json_mode* is True the single-shot JSON
    intermediate mode is used instead.

    Returns ``{"nec_content": str, "model": str, "usage": dict}``.
    """
    if pipeline:
        from .nec_pipeline import pipeline_from_form

        pr = pipeline_from_form(
            antenna_type=antenna_type,
            frequency_mhz=frequency_mhz,
            ground_type=ground_type,
            description=description,
            model=model,
        )
        result = pr.to_dict()
        result["json_mode"] = True
        result["pipeline"] = True
        return result
    from .nec_calculators import calc_for_type

    ground_map = {
        "free_space": "free space (no ground plane, GE 0)",
        "perfect": "perfect ground (GN 1, GE 1)",
        "real": "real ground — average earth (GN 2, GE 1)",
    }
    ground_desc = ground_map.get(ground_type, ground_type)

    user_msg = (
        f"Generate a NEC2 input file for:\n"
        f"  Antenna type: {antenna_type}\n"
        f"  Design frequency: {frequency_mhz} MHz\n"
        f"  Ground: {ground_desc}\n"
    )
    if description.strip():
        user_msg += f"\nAdditional details:\n{description.strip()}\n"

    # Inject computed starting dimensions from the calculator
    calc = calc_for_type(antenna_type, frequency_mhz)
    if calc is not None:
        user_msg += (
            f"\n--- COMPUTED STARTING DIMENSIONS ---\n"
            f"{calc.summary()}\n"
        )
        for note in calc.notes:
            user_msg += f"  • {note}\n"
        if calc.nec_hints:
            user_msg += "NEC modelling hints:\n"
            for hint in calc.nec_hints:
                user_msg += f"  • {hint}\n"
        user_msg += (
            "Use these dimensions as your starting point. "
            "They are physics-based and should be close to optimal.\n"
            "--- END DIMENSIONS ---\n"
        )

    type_ctx = _load_type_context(antenna_type)
    if type_ctx:
        user_msg += (
            f"\n--- REFERENCE for {antenna_type} antennas ---\n"
            f"{type_ctx}\n--- END REFERENCE ---\n"
        )

    sys_prompt = _JSON_SYSTEM_PROMPT if json_mode else _SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    refine_fn = _generate_and_refine_json if json_mode else _generate_and_refine
    result = refine_fn(
        messages, target_type=antenna_type, model=model,
        freq_mhz=frequency_mhz,
    )
    result["model"] = model
    result["json_mode"] = json_mode
    return result


# ---------------------------------------------------------------------------
# NEC generation from PDF
# ---------------------------------------------------------------------------

def generate_nec_from_pdf(
    pdf_bytes: bytes,
    *,
    model: str = "gpt-5.2",
    extra_instructions: str = "",
    antenna_type: str = "",
    json_mode: bool = False,
    pipeline: bool = False,
) -> dict[str, Any]:
    """Extract text from a PDF and ask the model to produce a NEC file.

    When *pipeline* is True the structured 7-step pipeline is used.
    When *json_mode* is True the single-shot JSON intermediate mode
    is used.

    Returns ``{"nec_content": str, "pdf_text": str, "model": str, "usage": dict}``.
    """
    if pipeline:
        from .nec_pipeline import pipeline_from_pdf

        pr = pipeline_from_pdf(
            pdf_bytes,
            antenna_type=antenna_type,
            model=model,
            extra_instructions=extra_instructions,
        )
        result = pr.to_dict()
        result["pdf_text"] = pr.source_text
        result["json_mode"] = True
        result["pipeline"] = True
        return result
    pdf_text = extract_pdf_text(pdf_bytes)
    if not pdf_text.strip():
        # Scanned / image-only PDF — try vision OCR
        pdf_text = _ocr_pdf_pages(pdf_bytes)
    if not pdf_text.strip():
        raise ValueError("Could not extract any text from the PDF")

    # Always detect the design frequency and goals from the document
    detected_freq = _guess_freq_mhz(pdf_text, antenna_type)
    doc_goals = _extract_design_goals(pdf_text)
    doc_goals.freq_mhz = detected_freq

    user_msg = (
        "Below is text extracted from a PDF document describing an antenna.\n"
        "Based on the description, generate a complete NEC2 input file that "
        "models this antenna as accurately as possible.\n\n"
        f"DESIGN FREQUENCY: The document describes an antenna for "
        f"{detected_freq} MHz.  Use {detected_freq} MHz in the FR card.\n\n"
    )
    goals_block = doc_goals.prompt_block()
    if goals_block:
        user_msg += goals_block + "\n"
    user_msg += (
        "IMPORTANT: Look for dimensional tables or text giving element lengths "
        "and spacings.  Convert all dimensions to metres.  Place elements at "
        "physically correct 3-D coordinates — do NOT stack all wires at the "
        "origin.\n\n"
        f"--- PDF TEXT ---\n{pdf_text}\n--- END ---\n"
    )
    if extra_instructions.strip():
        user_msg += f"\nAdditional instructions: {extra_instructions.strip()}\n"

    if antenna_type:
        # Inject computed starting dimensions from the calculator
        from .nec_calculators import calc_for_type

        calc = calc_for_type(antenna_type, detected_freq)
        if calc is not None:
            user_msg += (
                f"\n--- COMPUTED STARTING DIMENSIONS ---\n"
                f"{calc.summary()}\n"
            )
            for note in calc.notes:
                user_msg += f"  \u2022 {note}\n"
            if calc.nec_hints:
                user_msg += "NEC modelling hints:\n"
                for hint in calc.nec_hints:
                    user_msg += f"  \u2022 {hint}\n"
            user_msg += (
                "Use these dimensions as your starting point. "
                "They are physics-based and should be close to optimal.\n"
                "--- END DIMENSIONS ---\n"
            )

        type_ctx = _load_type_context(antenna_type)
        if type_ctx:
            user_msg += (
                f"\n--- REFERENCE for {antenna_type} antennas ---\n"
                f"{type_ctx}\n--- END REFERENCE ---\n"
            )

    sys_prompt = _JSON_SYSTEM_PROMPT if json_mode else _SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]

    refine_fn = _generate_and_refine_json if json_mode else _generate_and_refine
    result = refine_fn(
        messages,
        target_type=antenna_type or None,
        model=model,
    )
    result["pdf_text"] = pdf_text[:4000]
    result["model"] = model
    result["json_mode"] = json_mode
    return result


# ---------------------------------------------------------------------------
# URL text extraction
# ---------------------------------------------------------------------------

def extract_url_text(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and extract plain text + tables via BeautifulSoup.

    Strips scripts/styles, preserves table structure as pipe-delimited
    rows, and returns clean text suitable for the LLM prompt.
    """
    import urllib.request
    from html import unescape

    # Validate URL scheme
    if not url.startswith(("http://", "https://")):
        raise ValueError("Only http:// and https:// URLs are supported")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AntennaClassifier/1.0 (NEC research tool)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — validated scheme
        raw = resp.read()

    # Detect encoding
    charset = resp.headers.get_content_charset() or "utf-8"
    try:
        html = raw.decode(charset)
    except (UnicodeDecodeError, LookupError):
        html = raw.decode("utf-8", errors="replace")

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    parts: list[str] = []
    total = 0

    # Extract tables with structure preserved
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            block = "\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"
            parts.append(block)
            total += len(block)
        table.decompose()  # remove so it doesn't duplicate in body text

    # Remaining body text
    body_text = soup.get_text(separator="\n", strip=True)
    body_text = unescape(body_text)
    # Collapse blank lines
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)
    parts.insert(0, body_text)
    total += len(body_text)

    full = "\n".join(parts)
    return full[:max_chars]


# ---------------------------------------------------------------------------
# NEC generation from URL
# ---------------------------------------------------------------------------

def generate_nec_from_url(
    url: str,
    *,
    model: str = "gpt-5.2",
    extra_instructions: str = "",
    antenna_type: str = "",
    json_mode: bool = False,
    pipeline: bool = False,
) -> dict[str, Any]:
    """Fetch a web page, extract text, and produce a NEC file via AI.

    When *pipeline* is True the structured 7-step pipeline is used.
    When *json_mode* is True the single-shot JSON intermediate mode
    is used.

    Returns ``{"nec_content": str, "url_text": str, "model": str, "usage": dict}``.
    """
    if pipeline:
        from .nec_pipeline import pipeline_from_url

        pr = pipeline_from_url(
            url,
            antenna_type=antenna_type,
            model=model,
            extra_instructions=extra_instructions,
        )
        result = pr.to_dict()
        result["url"] = url
        result["url_text"] = pr.source_text
        result["json_mode"] = True
        result["pipeline"] = True
        return result
    url_text = extract_url_text(url)
    if not url_text.strip():
        raise ValueError("Could not extract any text from the URL")

    # Always detect the design frequency and goals from the document
    detected_freq = _guess_freq_mhz(url_text, antenna_type)
    doc_goals = _extract_design_goals(url_text)
    doc_goals.freq_mhz = detected_freq

    user_msg = (
        "Below is text extracted from a web page describing an antenna.\n"
        "Based on the description, generate a complete NEC2 input file that "
        "models this antenna as accurately as possible.\n\n"
        f"DESIGN FREQUENCY: The document describes an antenna for "
        f"{detected_freq} MHz.  Use {detected_freq} MHz in the FR card.\n\n"
    )
    goals_block = doc_goals.prompt_block()
    if goals_block:
        user_msg += goals_block + "\n"
    user_msg += (
        "IMPORTANT: Look for dimensional tables or text giving element lengths "
        "and spacings. Convert all dimensions to metres. Place elements at "
        "physically correct 3-D coordinates — do NOT stack all wires at the "
        "origin.\n\n"
        f"Source URL: {url}\n\n"
        f"--- WEB PAGE TEXT ---\n{url_text}\n--- END ---\n"
    )
    if extra_instructions.strip():
        user_msg += f"\nAdditional instructions: {extra_instructions.strip()}\n"

    if antenna_type:
        from .nec_calculators import calc_for_type

        calc = calc_for_type(antenna_type, detected_freq)
        if calc is not None:
            user_msg += (
                f"\n--- COMPUTED STARTING DIMENSIONS ---\n"
                f"{calc.summary()}\n"
            )
            for note in calc.notes:
                user_msg += f"  \u2022 {note}\n"
            if calc.nec_hints:
                user_msg += "NEC modelling hints:\n"
                for hint in calc.nec_hints:
                    user_msg += f"  \u2022 {hint}\n"
            user_msg += (
                "Use these dimensions as your starting point. "
                "They are physics-based and should be close to optimal.\n"
                "--- END DIMENSIONS ---\n"
            )

        type_ctx = _load_type_context(antenna_type)
        if type_ctx:
            user_msg += (
                f"\n--- REFERENCE for {antenna_type} antennas ---\n"
                f"{type_ctx}\n--- END REFERENCE ---\n"
            )

    messages = [
        {"role": "system", "content": _JSON_SYSTEM_PROMPT if json_mode else _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    refine_fn = _generate_and_refine_json if json_mode else _generate_and_refine
    result = refine_fn(
        messages,
        target_type=antenna_type or None,
        model=model,
    )
    result["url"] = url
    result["url_text"] = url_text[:4000]
    result["model"] = model
    result["json_mode"] = json_mode
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_nec(raw: str) -> str:
    """Strip markdown fences and trailing chat from model output."""
    # Remove ```nec ... ``` or ``` ... ``` wrappers
    m = re.search(r"```(?:nec)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    # Trim trailing chat after EN card
    lines: list[str] = []
    for line in raw.splitlines():
        lines.append(line)
        if line.strip().upper() == "EN":
            break
    return "\n".join(lines) + "\n"


_REQUIRED_CARDS = {"GW", "GE", "EX", "FR", "EN"}


def _validate_nec(nec: str) -> None:
    """Sanity-check a generated NEC deck.

    Raises ``ValueError`` with a human-readable message if the deck is
    clearly broken so the caller can surface a useful error to the user.
    """
    cards_present: set[str] = set()
    gw_lines: list[str] = []
    for line in nec.splitlines():
        token = line.split()[0].upper() if line.split() else ""
        if token in ("CM", "CE", "GW", "GE", "GN", "TL", "EX", "FR", "RP", "EN"):
            cards_present.add(token)
        if token == "GW":
            gw_lines.append(line.strip())

    missing = _REQUIRED_CARDS - cards_present
    if missing:
        raise ValueError(
            f"Generated NEC deck is incomplete — missing required cards: "
            f"{', '.join(sorted(missing))}. "
            f"The AI model may have been cut off. Try again or simplify the description."
        )

    # Detect degenerate repetition (all GW lines identical after tag number)
    if len(gw_lines) > 10:
        normalised = [re.sub(r"^GW\s+\d+", "GW _", l) for l in gw_lines]
        if len(set(normalised)) == 1:
            raise ValueError(
                f"Generated NEC deck is degenerate — all {len(gw_lines)} wires "
                f"are identical. The AI model fell into a repetition loop. "
                f"Try again or provide more specific dimensions in the description."
            )

    # Detect collapsed geometry: multiple wires that all share the same start
    # point and differ only in length along one axis (classic "stacked vertical"
    # failure mode).
    if len(gw_lines) >= 2:
        starts: set[str] = set()
        for gw in gw_lines:
            parts = gw.split()
            if len(parts) >= 9:
                start = (parts[3], parts[4], parts[5])
                starts.add(start)
        if len(starts) == 1:
            raise ValueError(
                "Generated NEC deck has collapsed geometry — all wires share "
                "the same start point. Multi-element antennas need elements "
                "at different spatial positions. Try again."
            )
