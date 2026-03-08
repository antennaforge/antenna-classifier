"""AI-powered NEC file generator using OpenAI.

Given an antenna description (from a form or extracted from a PDF),
calls the OpenAI chat completions API to produce a valid NEC2 input
deck.  The output is validated with the project parser + validator
before being returned.
"""

from __future__ import annotations

import io
import os
import pathlib
import re
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


def _load_type_context(antenna_type: str) -> str:
    """Load the NEC modelling reference for *antenna_type*, if available.

    Returns the file contents as a string, or an empty string if no
    context document exists for the requested type.
    """
    path = _NEC_CONTEXT_DIR / f"{antenna_type}.txt"
    if path.is_file():
        return path.read_text()
    return ""


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
) -> dict[str, Any]:
    """Generate a NEC file from structured form data.

    Returns ``{"nec_content": str, "model": str, "usage": dict}``.
    """
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

    type_ctx = _load_type_context(antenna_type)
    if type_ctx:
        user_msg += (
            f"\n--- REFERENCE for {antenna_type} antennas ---\n"
            f"{type_ctx}\n--- END REFERENCE ---\n"
        )

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_completion_tokens=8192,
    )
    content = resp.choices[0].message.content or ""
    nec = _extract_nec(content)
    _validate_nec(nec)
    return {
        "nec_content": nec,
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "finish_reason": resp.choices[0].finish_reason,
        },
    }


# ---------------------------------------------------------------------------
# NEC generation from PDF
# ---------------------------------------------------------------------------

def generate_nec_from_pdf(
    pdf_bytes: bytes,
    *,
    model: str = "gpt-5.2",
    extra_instructions: str = "",
    antenna_type: str = "",
) -> dict[str, Any]:
    """Extract text from a PDF and ask the model to produce a NEC file.

    Returns ``{"nec_content": str, "pdf_text": str, "model": str, "usage": dict}``.
    """
    pdf_text = extract_pdf_text(pdf_bytes)
    if not pdf_text.strip():
        raise ValueError("Could not extract any text from the PDF")

    user_msg = (
        "Below is text extracted from a PDF document describing an antenna.\n"
        "Based on the description, generate a complete NEC2 input file that "
        "models this antenna as accurately as possible.\n\n"
        "IMPORTANT: Look for dimensional tables or text giving element lengths "
        "and spacings.  Convert all dimensions to metres.  Place elements at "
        "physically correct 3-D coordinates — do NOT stack all wires at the "
        "origin.\n\n"
        f"--- PDF TEXT ---\n{pdf_text}\n--- END ---\n"
    )
    if extra_instructions.strip():
        user_msg += f"\nAdditional instructions: {extra_instructions.strip()}\n"

    if antenna_type:
        type_ctx = _load_type_context(antenna_type)
        if type_ctx:
            user_msg += (
                f"\n--- REFERENCE for {antenna_type} antennas ---\n"
                f"{type_ctx}\n--- END REFERENCE ---\n"
            )

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_completion_tokens=8192,
    )
    content = resp.choices[0].message.content or ""
    nec = _extract_nec(content)
    _validate_nec(nec)
    return {
        "nec_content": nec,
        "pdf_text": pdf_text[:4000],
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "finish_reason": resp.choices[0].finish_reason,
        },
    }


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
