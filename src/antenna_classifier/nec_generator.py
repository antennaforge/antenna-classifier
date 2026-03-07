"""AI-powered NEC file generator using OpenAI.

Given an antenna description (from a form or extracted from a PDF),
calls the OpenAI chat completions API to produce a valid NEC2 input
deck.  The output is validated with the project parser + validator
before being returned.
"""

from __future__ import annotations

import io
import os
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
14. Coordinates: Z axis is vertical (height above ground).

**NEC2 card format reference:**
- CM <text>
- CE
- GW tag segs x1 y1 z1 x2 y2 z2 radius
- GE ground_type
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
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 12000) -> str:
    """Extract plain text from a PDF (first *max_chars* characters)."""
    import pdfplumber

    text_parts: list[str] = []
    total = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
    full = "\n".join(text_parts)
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
    model: str = "gpt-4o",
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

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    content = resp.choices[0].message.content or ""
    nec = _extract_nec(content)
    return {
        "nec_content": nec,
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        },
    }


# ---------------------------------------------------------------------------
# NEC generation from PDF
# ---------------------------------------------------------------------------

def generate_nec_from_pdf(
    pdf_bytes: bytes,
    *,
    model: str = "gpt-4o",
    extra_instructions: str = "",
) -> dict[str, Any]:
    """Extract text from a PDF and ask the model to produce a NEC file.

    Returns ``{"nec_content": str, "pdf_text": str, "model": str, "usage": dict}``.
    """
    pdf_text = extract_pdf_text(pdf_bytes)
    if not pdf_text.strip():
        raise ValueError("Could not extract any text from the PDF")

    user_msg = (
        "Below is text extracted from a PDF document describing an antenna. "
        "Based on the description, generate a complete NEC2 input file that "
        "models this antenna as accurately as possible.\n\n"
        f"--- PDF TEXT ---\n{pdf_text}\n--- END ---\n"
    )
    if extra_instructions.strip():
        user_msg += f"\nAdditional instructions: {extra_instructions.strip()}\n"

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    content = resp.choices[0].message.content or ""
    nec = _extract_nec(content)
    return {
        "nec_content": nec,
        "pdf_text": pdf_text[:2000],  # truncated preview
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
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
