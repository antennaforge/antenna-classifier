"""Antenna dimension calculators — physics-based starting geometry.

Pure-math functions that compute initial element lengths, spacings, and
critical dimensions from design frequency (and optionally wire diameter).
No simulation needed — these give the LLM a solid starting point that the
OODA loop can then refine via nec2c.

All dimensions are returned in **metres**.

Usage::

    from antenna_classifier.nec_calculators import calc_for_type

    dims = calc_for_type("yagi", freq_mhz=14.175, n_elements=3)
    # dims["reflector_length"], dims["driven_length"], dims["spacings"], ...

    dims = calc_for_type("moxon", freq_mhz=14.175, wire_dia_mm=1.63)
    # dims["A"], dims["B"], dims["C"], dims["D"], dims["total_width"], ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_C_MPS = 299_792_458.0     # speed of light (m/s)


def _wavelength(freq_mhz: float) -> float:
    """Free-space wavelength in metres."""
    return _C_MPS / (freq_mhz * 1e6)


def _wire_radius_m(wire_dia_mm: float) -> float:
    """Wire radius in metres from diameter in mm."""
    return wire_dia_mm / 2000.0


# Standard wire gauges → diameter in mm
WIRE_GAUGES: dict[str, float] = {
    "10awg": 2.588,
    "12awg": 2.053,
    "14awg": 1.628,
    "16awg": 1.291,
    "18awg": 1.024,
    "6mm_tube": 6.0,
    "10mm_tube": 10.0,
    "12mm_tube": 12.0,
    "19mm_tube": 19.05,   # 3/4"
    "25mm_tube": 25.4,    # 1"
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CalcResult:
    """Dimension calculator output."""
    antenna_type: str
    freq_mhz: float
    wavelength_m: float
    wire_dia_mm: float
    dimensions: dict[str, Any]       # all computed dimensions (metres)
    notes: list[str] = field(default_factory=list)
    nec_hints: list[str] = field(default_factory=list)  # NEC modelling tips

    def to_dict(self) -> dict[str, Any]:
        return {
            "antenna_type": self.antenna_type,
            "freq_mhz": self.freq_mhz,
            "wavelength_m": round(self.wavelength_m, 4),
            "wire_dia_mm": self.wire_dia_mm,
            "dimensions": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.dimensions.items()
            },
            "notes": self.notes,
            "nec_hints": self.nec_hints,
        }

    def summary(self) -> str:
        """One-line summary suitable for an LLM prompt."""
        parts = []
        for k, v in self.dimensions.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}m")
            elif isinstance(v, list):
                parts.append(f"{k}=[{', '.join(f'{x:.4f}m' for x in v)}]")
            else:
                parts.append(f"{k}={v}")
        return f"{self.antenna_type} @ {self.freq_mhz} MHz: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# DIPOLE
# ---------------------------------------------------------------------------

def calc_dipole(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Half-wave dipole dimensions.

    Uses the K-factor correction for wire diameter:
        L = K * (λ/2)
    where K ≈ 0.95 for typical wire diameters relative to wavelength.
    """
    wl = _wavelength(freq_mhz)
    # K factor: accounts for wire thickness and end effects
    # K = 0.95 is a good starting point; more precise formulas exist
    # but simulation will refine anyway
    dia_wl = (wire_dia_mm / 1000.0) / wl
    k = 0.98 - 0.05 * math.log10(max(dia_wl, 1e-6) * 1000)
    k = min(max(k, 0.93), 0.98)
    total_length = k * wl / 2.0
    half_length = total_length / 2.0

    return CalcResult(
        antenna_type="dipole",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "total_length": total_length,
            "half_length": half_length,
            "feed_height_suggested": max(wl / 4, 5.0),
        },
        notes=[
            f"K-factor = {k:.3f} (correction for end effects)",
            "Impedance ≈ 73Ω in free space, varies with height",
            "Trim tips symmetrically to lower resonance",
        ],
        nec_hints=[
            f"GW 1 21 0.0 -{half_length:.4f} H 0.0 {half_length:.4f} H {_wire_radius_m(wire_dia_mm):.6f}",
            "Feed: EX 0 1 11 0 1 0  (centre segment)",
        ],
    )


# ---------------------------------------------------------------------------
# INVERTED V
# ---------------------------------------------------------------------------

def calc_inverted_v(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    droop_angle_deg: float = 45.0,
    apex_height_m: float = 0.0,
) -> CalcResult:
    """Inverted-V dimensions.

    Element length is slightly longer than a flat dipole due to droop.
    The correction factor increases with droop angle.
    """
    dipole = calc_dipole(freq_mhz, wire_dia_mm)
    flat_length = dipole.dimensions["total_length"]
    # Droop correction: ~2-5% longer
    droop_rad = math.radians(droop_angle_deg)
    correction = 1.0 + 0.02 * (droop_angle_deg / 45.0)
    total_length = flat_length * correction
    leg_length = total_length / 2.0
    wl = _wavelength(freq_mhz)
    if apex_height_m <= 0:
        apex_height_m = max(wl / 4, 10.0)

    # End height
    horizontal_reach = leg_length * math.cos(droop_rad)
    vertical_drop = leg_length * math.sin(droop_rad)
    end_height = apex_height_m - vertical_drop

    return CalcResult(
        antenna_type="inverted_v",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "total_length": total_length,
            "leg_length": leg_length,
            "apex_height": apex_height_m,
            "droop_angle_deg": droop_angle_deg,
            "end_height": max(end_height, 1.0),
            "horizontal_spread": horizontal_reach * 2,
        },
        notes=[
            f"Droop correction factor: {correction:.3f}",
            f"Impedance ≈ {50 + 23 * (1 - droop_angle_deg / 90):.0f}Ω "
            f"(lower with steeper droop)",
            "Steeper droop → wider bandwidth, lower gain",
        ],
    )


# ---------------------------------------------------------------------------
# VERTICAL (quarter-wave)
# ---------------------------------------------------------------------------

def calc_vertical(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    n_radials: int = 16,
    radial_slope_deg: float = 0.0,
) -> CalcResult:
    """Quarter-wave vertical with radials."""
    wl = _wavelength(freq_mhz)
    radiator_length = wl / 4.0 * 0.95  # ~5% shortening for practical antennas
    radial_length = wl / 4.0

    return CalcResult(
        antenna_type="vertical",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "radiator_length": radiator_length,
            "radial_length": radial_length,
            "radial_count": n_radials,
            "radial_slope_deg": radial_slope_deg,
        },
        notes=[
            f"Radiator: {radiator_length:.3f}m (λ/4 × 0.95)",
            f"Radials: {n_radials}× {radial_length:.3f}m each",
            "Feed impedance ≈ 36Ω over perfect ground, ~50Ω with sloped radials",
            "More radials always better — diminishing returns past 32",
        ],
    )


# ---------------------------------------------------------------------------
# J-POLE
# ---------------------------------------------------------------------------

def calc_j_pole(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """J-pole antenna dimensions."""
    wl = _wavelength(freq_mhz)
    # Total height: 3/4λ (λ/2 radiator + λ/4 stub)
    radiator_length = wl / 2.0 * 0.95
    stub_length = wl / 4.0 * 0.95
    stub_spacing = wl * 0.01  # ~1% of wavelength between conductors
    total_height = radiator_length + stub_length
    # Feedpoint is typically ~3-5% up from the bottom of the stub
    tap_height = stub_length * 0.04

    return CalcResult(
        antenna_type="j_pole",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "radiator_length": radiator_length,
            "stub_length": stub_length,
            "stub_spacing": stub_spacing,
            "total_height": total_height,
            "feed_tap_height": tap_height,
        },
        notes=[
            "Total height ≈ 3/4λ",
            f"Stub spacing: {stub_spacing * 1000:.0f}mm (adjust for 50Ω match)",
            "Feedpoint tap position controls impedance — move up for higher Z",
        ],
    )


# ---------------------------------------------------------------------------
# END-FED HALF-WAVE (EFHW)
# ---------------------------------------------------------------------------

def calc_end_fed(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """End-fed half-wave antenna dimensions."""
    wl = _wavelength(freq_mhz)
    wire_length = wl / 2.0 * 0.97  # slightly less shortening than centre-fed
    counterpoise = wl * 0.05  # ~5% of wavelength

    return CalcResult(
        antenna_type="end_fed",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "wire_length": wire_length,
            "counterpoise_length": counterpoise,
            "transformer_ratio": "49:1 or 64:1",
        },
        notes=[
            "Feed impedance ≈ 2500–5000Ω — needs matching transformer",
            "49:1 on FT240-43 toroid is standard",
            "Counterpoise helps prevent RF on coax",
            "Multiband operation on harmonics (40/20/15/10m from 7 MHz)",
        ],
    )


# ---------------------------------------------------------------------------
# YAGI
# ---------------------------------------------------------------------------

def calc_yagi(
    freq_mhz: float,
    wire_dia_mm: float = 12.0,
    n_elements: int = 3,
) -> CalcResult:
    """N-element Yagi-Uda dimensions using classic design formulas.

    Based on NBS Technical Note 688 (Viezbicke) and DL6WU long-Yagi data.
    Returns reflector, driven element, and director lengths + all spacings.
    """
    wl = _wavelength(freq_mhz)
    # dia/lambda ratio affects element lengths
    dia_wl = (wire_dia_mm / 1000.0) / wl

    # Reflector: ~0.495λ (5% longer than resonant λ/2)
    reflector = wl * 0.495

    # Driven element: ~0.473λ (resonant at design freq with typical dia/λ)
    driven = wl * 0.473

    # Reflector–DE spacing: ~0.15–0.20λ (closer = higher F/B, less gain)
    re_de_spacing = wl * 0.18

    # Directors: progressively shorter
    # First director: ~0.440λ
    # Subsequent: each 0.5–1% shorter
    # Spacings: DL6WU formula — starts at ~0.10λ, increases toward 0.30λ
    director_lengths: list[float] = []
    director_spacings: list[float] = []

    n_directors = max(0, n_elements - 2)
    if n_directors > 0:
        # First director
        d1_length = wl * 0.440
        # DL6WU spacing formula (approximation)
        d1_spacing = wl * 0.10
        director_lengths.append(d1_length)
        director_spacings.append(d1_spacing)

        for i in range(1, n_directors):
            # Each subsequent director: 0.5–1% shorter
            prev_len = director_lengths[-1]
            director_lengths.append(prev_len * 0.995)
            # Spacing increases: 0.10λ → up to 0.30λ for long Yagis
            spacing = wl * min(0.10 + 0.02 * (i + 1), 0.30)
            director_spacings.append(spacing)

    # Boom length
    boom_length = re_de_spacing + sum(director_spacings)

    # Expected gain (rough estimate)
    expected_gain_dbi = 5.0 + 1.5 * math.log2(max(n_elements - 1, 1))

    dims: dict[str, Any] = {
        "reflector_length": reflector,
        "driven_length": driven,
        "re_de_spacing": re_de_spacing,
        "director_lengths": director_lengths,
        "de_director_spacings": director_spacings,
        "boom_length": boom_length,
        "boom_length_wl": boom_length / wl,
    }

    return CalcResult(
        antenna_type="yagi",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions=dims,
        notes=[
            f"{n_elements}-element Yagi, boom = {boom_length:.3f}m "
            f"({boom_length / wl:.2f}λ)",
            f"Expected gain ≈ {expected_gain_dbi:.1f} dBi",
            f"Element dia/λ = {dia_wl:.5f}",
            "Hairpin or gamma match needed for 50Ω feed",
            "Build driven element first, tune to resonance, then add parasitic elements",
        ],
        nec_hints=[
            "Place elements along X axis, extending along Y",
            "All elements at same Z height",
            f"Wire radius: {wire_dia_mm / 2000:.6f}m",
        ],
    )


# ---------------------------------------------------------------------------
# MOXON RECTANGLE
# ---------------------------------------------------------------------------

def calc_moxon(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Moxon rectangle dimensions — A, B, C, D formulas.

    Based on the AC6LA Moxon Rectangle Generator polynomials and
    G4ZU's original Moxon dimensions.  The four critical dimensions:

        A = element width (the "front" of the rectangle)
        B = tip gap (CRITICAL — controls F/B)
        C = driven element tail length (the bent-back portion)
        D = reflector tail length (the bent-back portion)

    The element-to-element spacing = B + C + D.
    """
    wl = _wavelength(freq_mhz)
    wire_radius_wl = (wire_dia_mm / 1000.0) / wl / 2.0
    # Natural log of wire diameter in wavelengths
    # The polynomial fits use d1 = ln(wire_dia_in_wavelengths)
    d_wl = (wire_dia_mm / 1000.0) / wl
    d1 = math.log(d_wl)

    # Polynomial coefficients from AC6LA Moxon Calculator
    # A (half-width of the main element, expressed as fraction of λ)
    a1 = -0.0008571428571
    a2 = -0.009571428571
    a3 = 0.3398571429
    A_wl = a1 * d1 * d1 + a2 * d1 + a3

    # B (tip gap, fraction of λ)
    b1 = -0.002142857143
    b2 = -0.02035714286
    b3 = -0.008
    B_wl = b1 * d1 * d1 + b2 * d1 + b3

    # C (driven tail, fraction of λ)
    c1 = 0.001809523810
    c2 = 0.01780952381
    c3 = 0.05164285714
    C_wl = c1 * d1 * d1 + c2 * d1 + c3

    # D (reflector tail, fraction of λ)
    d_c1 = 0.001
    d_c2 = 0.07178571429
    D_wl = d_c1 * d1 + d_c2

    # Convert to metres
    A = A_wl * wl  # total width (full element)
    B = B_wl * wl  # tip gap
    C = C_wl * wl  # driven tail
    D = D_wl * wl  # reflector tail

    # Ensure gap is positive and reasonable
    B = max(B, wl * 0.005)

    total_width = A
    total_depth = B + C + D  # element-to-element spacing (boom length)

    return CalcResult(
        antenna_type="moxon",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "A_width": A,
            "B_tip_gap": B,
            "C_driven_tail": C,
            "D_reflector_tail": D,
            "total_width": total_width,
            "total_depth": total_depth,
            "turning_radius": math.sqrt((A / 2) ** 2 + total_depth ** 2),
        },
        notes=[
            f"A (width) = {A:.4f}m ({A / wl:.4f}λ)",
            f"B (tip gap) = {B * 1000:.1f}mm — CRITICAL dimension",
            f"C (driven tail) = {C:.4f}m",
            f"D (reflector tail) = {D:.4f}m",
            f"Element spacing = {total_depth:.4f}m",
            "Natural 50Ω impedance — no matching network needed!",
            "F/B peaks at the correct gap — ±5mm shifts it dramatically",
            "Use rigid spreaders to maintain the tip gap",
        ],
        nec_hints=[
            "6 GW cards: 3 for driven (left tail, main, right tail), "
            "3 for reflector",
            "All wires at same Z height",
            f"Tip gap B = {B:.4f}m between tail ends",
        ],
    )


# ---------------------------------------------------------------------------
# QUAD (cubical quad beam)
# ---------------------------------------------------------------------------

def calc_quad(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    n_elements: int = 2,
) -> CalcResult:
    """Cubical quad beam dimensions.

    Each element is a full-wave loop (circumference ≈ 1λ).
    Reflector: +5%, director: -5%.
    """
    wl = _wavelength(freq_mhz)

    driven_circ = wl * 1.02   # full-wave loop, slight correction
    driven_side = driven_circ / 4.0

    reflector_circ = driven_circ * 1.05
    reflector_side = reflector_circ / 4.0

    element_spacing = wl * 0.20  # 0.15–0.25λ typical

    dims: dict[str, Any] = {
        "driven_circumference": driven_circ,
        "driven_side_length": driven_side,
        "reflector_circumference": reflector_circ,
        "reflector_side_length": reflector_side,
        "element_spacing": element_spacing,
    }

    n_directors = max(0, n_elements - 2)
    if n_directors > 0:
        director_circs = []
        for i in range(n_directors):
            d_circ = driven_circ * (0.95 - 0.01 * i)  # each 1% shorter
            director_circs.append(d_circ)
        dims["director_circumferences"] = director_circs
        dims["director_side_lengths"] = [c / 4 for c in director_circs]
        dims["director_spacings"] = [wl * 0.15] * n_directors

    boom = element_spacing
    if n_directors > 0:
        boom += sum(dims.get("director_spacings", []))
    dims["boom_length"] = boom

    return CalcResult(
        antenna_type="quad",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions=dims,
        notes=[
            f"Driven loop: {driven_circ:.3f}m circumference, "
            f"{driven_side:.3f}m per side",
            f"Reflector loop: {reflector_circ:.3f}m circumference, "
            f"{reflector_side:.3f}m per side (+5%)",
            f"Element spacing (boom): {element_spacing:.3f}m ({element_spacing / wl:.3f}λ)",
            "Impedance ≈ 120Ω — use λ/4 75Ω coax transformer for 50Ω match",
            "Bottom-fed = horizontal polarisation, side-fed = vertical",
        ],
        nec_hints=[
            "Each quad element is 4 GW cards forming a closed square loop",
            "Loops in Y-Z plane, spaced along X (boom axis)",
            f"Driven element: 4 wires, each side {driven_side:.4f}m, "
            f"square from Y=-{driven_side / 2:.4f} to Y=+{driven_side / 2:.4f}, "
            f"Z=0 to Z={driven_side:.4f}",
            f"Reflector: 4 wires, each side {reflector_side:.4f}m, "
            f"at X=-{element_spacing:.4f}",
            "Use UNIQUE tag numbers per wire (e.g. reflector 1-4, driven 5-8)",
            "Feed at bottom centre of driven element — split bottom wire "
            "into two halves with excitation at the junction",
            "Wire endpoints MUST connect exactly to close each loop",
        ],
    )


# ---------------------------------------------------------------------------
# LPDA (Log-Periodic Dipole Array)
# ---------------------------------------------------------------------------

def calc_lpda(
    freq_mhz_low: float,
    freq_mhz_high: float,
    tau: float = 0.92,
    sigma: float = 0.05,
    wire_dia_mm: float = 12.0,
) -> CalcResult:
    """Log-periodic dipole array element dimensions from τ and σ.

    τ (tau) = design ratio (0.85–0.96, higher = more gain, longer boom)
    σ (sigma) = spacing constant (0.03–0.08)
    """
    wl_low = _wavelength(freq_mhz_low)
    wl_high = _wavelength(freq_mhz_high)
    wl_centre = _wavelength((freq_mhz_low + freq_mhz_high) / 2)

    # Generate elements from highest to lowest frequency
    # Shortest element first (front of antenna)
    half_lengths: list[float] = []
    spacings: list[float] = []

    # Start with the shortest element at the high-frequency end
    L1 = wl_high / 2.0 * 0.95  # half-wave at highest freq, with correction
    half_lengths.append(L1 / 2.0)

    # Generate until we cover the low-frequency end
    while True:
        L_next = half_lengths[-1] / tau
        if L_next * 2 > wl_low * 0.55:  # covered the low end
            half_lengths.append(L_next)
            break
        half_lengths.append(L_next)
        if len(half_lengths) > 30:  # safety limit
            break

    # Spacings from σ formula: d_n = 2 * σ * L_n
    for i in range(len(half_lengths) - 1):
        s = 2 * sigma * half_lengths[i + 1]  # spacing to next (larger) element
        spacings.append(s)

    # Full element lengths
    element_lengths = [h * 2 for h in half_lengths]
    boom_length = sum(spacings)

    n_elements = len(half_lengths)

    return CalcResult(
        antenna_type="lpda",
        freq_mhz=freq_mhz_low,  # use low freq as reference
        wavelength_m=wl_centre,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "tau": tau,
            "sigma": sigma,
            "n_elements": n_elements,
            "element_lengths": element_lengths,
            "half_lengths": [h for h in half_lengths],
            "spacings": spacings,
            "boom_length": boom_length,
            "freq_range_mhz": [freq_mhz_low, freq_mhz_high],
        },
        notes=[
            f"τ={tau}, σ={sigma}, {n_elements} elements",
            f"Frequency range: {freq_mhz_low}–{freq_mhz_high} MHz",
            f"Boom length: {boom_length:.3f}m",
            "Feed at the shortest (front) element",
            "CRITICAL: Phase reversal (crossed connection) between ALL adjacent elements",
            "Use TL cards with negative Z0 for phase reversal in NEC",
        ],
    )


# ---------------------------------------------------------------------------
# MAGNETIC LOOP
# ---------------------------------------------------------------------------

def calc_magnetic_loop(
    freq_mhz: float,
    conductor_dia_mm: float = 22.0,
    circumference_fraction: float = 0.20,
) -> CalcResult:
    """Small transmitting loop (magnetic loop) dimensions.

    circumference_fraction: loop circumference as fraction of λ (0.10–0.25).
    The loop must be < λ/4 (0.25) to maintain small-loop behaviour.
    """
    wl = _wavelength(freq_mhz)
    circumference = wl * circumference_fraction
    radius = circumference / (2 * math.pi)
    area = math.pi * radius ** 2

    # Radiation resistance (approximate for small loop)
    # Rr = 31171 * (A/λ²)² for area A in m²
    Rr = 31171.0 * (area / (wl ** 2)) ** 2

    # Loss resistance (approximate, depends heavily on conductor)
    # R_loss ≈ circumference / (π * d * σ * δ) where δ = skin depth
    # Simplified: assume copper at HF
    sigma = 5.8e7  # copper conductivity S/m
    skin_depth = 1.0 / math.sqrt(math.pi * freq_mhz * 1e6 * 4e-7 * math.pi * sigma)
    R_loss = circumference / (math.pi * (conductor_dia_mm / 1000.0) * sigma * skin_depth)

    efficiency = Rr / (Rr + R_loss) * 100 if (Rr + R_loss) > 0 else 0

    # Tuning capacitor: C = 1 / (ω² L) where L ≈ μ₀ * radius * (ln(8R/a) - 2)
    a = conductor_dia_mm / 2000.0  # wire radius in metres
    L_henry = 4e-7 * math.pi * radius * (math.log(8 * radius / a) - 2)
    omega = 2 * math.pi * freq_mhz * 1e6
    C_farads = 1.0 / (omega ** 2 * L_henry)
    C_pf = C_farads * 1e12

    # Voltage across capacitor at 100W (approximate)
    Q = omega * L_henry / (Rr + R_loss) if (Rr + R_loss) > 0 else 100
    # V = sqrt(P * Q * Xl) — can be several kV
    Xl = omega * L_henry
    V_cap = math.sqrt(100 * Q * Xl) if Q > 0 else 0

    # Bandwidth (3 dB)
    bw_hz = freq_mhz * 1e6 / Q if Q > 0 else 0
    bw_khz = bw_hz / 1000

    # Coupling loop size (typically 1/5 of main loop)
    coupling_circumference = circumference / 5.0
    coupling_radius = coupling_circumference / (2 * math.pi)

    return CalcResult(
        antenna_type="magnetic_loop",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=conductor_dia_mm,
        dimensions={
            "circumference": circumference,
            "radius": radius,
            "diameter": radius * 2,
            "conductor_dia_mm": conductor_dia_mm,
            "tuning_capacitor_pf": C_pf,
            "cap_voltage_at_100w": V_cap,
            "coupling_loop_circumference": coupling_circumference,
            "coupling_loop_radius": coupling_radius,
        },
        notes=[
            f"Loop circumference = {circumference:.3f}m "
            f"({circumference_fraction:.2f}λ)",
            f"Radiation resistance = {Rr:.4f}Ω",
            f"Loss resistance = {R_loss:.4f}Ω (copper, {conductor_dia_mm}mm)",
            f"Efficiency ≈ {efficiency:.1f}%",
            f"Q ≈ {Q:.0f} — bandwidth ≈ {bw_khz:.1f} kHz",
            f"Tuning capacitor: {C_pf:.1f} pF",
            f"DANGER: Capacitor voltage ≈ {V_cap:.0f}V RMS at 100W!",
            "Use vacuum or butterfly capacitor rated for the voltage",
            "ALL joints must be soldered — mechanical joints add loss",
        ],
    )


# ---------------------------------------------------------------------------
# DELTA LOOP
# ---------------------------------------------------------------------------

def calc_delta_loop(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Full-wave delta (triangular) loop."""
    wl = _wavelength(freq_mhz)
    perimeter = wl * 1.05  # full-wave with correction
    side_length = perimeter / 3.0  # equilateral triangle
    # For right-angle delta: base = 0.431λ, sides = 0.373λ each
    base = wl * 0.431
    leg = wl * 0.373
    apex_height = math.sqrt(leg ** 2 - (base / 2) ** 2)

    return CalcResult(
        antenna_type="delta_loop",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "perimeter": perimeter,
            "equilateral_side": side_length,
            "right_angle_base": base,
            "right_angle_leg": leg,
            "right_angle_height": apex_height,
        },
        notes=[
            f"Perimeter ≈ {perimeter:.3f}m (1.05λ)",
            "Impedance ≈ 120Ω — use 75Ω λ/4 matching section",
            "Bottom-fed = horizontal pol, corner-fed = vertical pol",
        ],
    )


# ---------------------------------------------------------------------------
# COLLINEAR
# ---------------------------------------------------------------------------

def calc_collinear(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    n_sections: int = 2,
) -> CalcResult:
    """Stacked collinear antenna dimensions."""
    wl = _wavelength(freq_mhz)
    section_length = wl / 2.0 * 0.95
    phasing_stub_length = wl / 4.0 * 0.95
    # Stub spacing (parallel conductors)
    stub_spacing = wl * 0.01

    total_height = (n_sections * section_length +
                    (n_sections - 1) * phasing_stub_length)
    expected_gain = 2.15 + 3.0 * math.log2(n_sections)

    return CalcResult(
        antenna_type="collinear",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "section_length": section_length,
            "phasing_stub_length": phasing_stub_length,
            "stub_spacing": stub_spacing,
            "n_sections": n_sections,
            "total_height": total_height,
        },
        notes=[
            f"{n_sections}-section collinear, total height {total_height:.3f}m",
            f"Expected gain ≈ {expected_gain:.1f} dBi",
            "Phasing stubs connect sections in-phase",
        ],
    )


# ---------------------------------------------------------------------------
# DISCONE
# ---------------------------------------------------------------------------

def calc_discone(
    freq_mhz_low: float,
    wire_dia_mm: float = 3.0,
    n_elements: int = 8,
) -> CalcResult:
    """Discone dimensions — broadband from low freq to ~10× higher."""
    wl_low = _wavelength(freq_mhz_low)

    cone_length = wl_low / 4.0
    disc_diameter = wl_low * 0.7 * 0.25  # ~70% of cone length
    cone_angle_deg = 60.0  # half-angle from axis (30° from horizontal)

    return CalcResult(
        antenna_type="discone",
        freq_mhz=freq_mhz_low,
        wavelength_m=wl_low,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "cone_slant_length": cone_length,
            "disc_diameter": disc_diameter,
            "cone_half_angle_deg": cone_angle_deg,
            "n_cone_elements": n_elements,
            "n_disc_elements": n_elements,
            "low_freq_mhz": freq_mhz_low,
            "high_freq_mhz": freq_mhz_low * 10,
        },
        notes=[
            f"Coverage: {freq_mhz_low}–{freq_mhz_low * 10:.0f} MHz (10:1 ratio)",
            f"Cone element length: {cone_length:.3f}m",
            f"Disc diameter: {disc_diameter:.3f}m",
            "SWR should be flat across entire range",
            "Good for wideband receive; modest gain for transmit",
        ],
    )


# ---------------------------------------------------------------------------
# HEXBEAM
# ---------------------------------------------------------------------------

def calc_hexbeam(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Hexbeam (broadband hex) dimensions.

    Based on the classic G3TXQ broadband hex design.
    """
    wl = _wavelength(freq_mhz)

    # Frame radius ≈ 0.236λ
    frame_radius = wl * 0.236

    # Driven element wire length ≈ slightly less than λ/2
    driven_wire = wl * 0.455

    # Reflector wire length ≈ slightly more than λ/2
    reflector_wire = wl * 0.481

    # Vertical spacing between driven and reflector planes
    vertical_spacing = wl * 0.07

    # Compression (how much wire folds back at spreader tips)
    tip_compression = wl * 0.04

    return CalcResult(
        antenna_type="hexbeam",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "frame_radius": frame_radius,
            "frame_diameter": frame_radius * 2,
            "driven_wire_length": driven_wire,
            "reflector_wire_length": reflector_wire,
            "vertical_spacing": vertical_spacing,
            "tip_compression": tip_compression,
            "n_spreaders": 6,
        },
        notes=[
            f"Frame radius: {frame_radius:.3f}m ({frame_radius / wl:.3f}λ)",
            f"Driven wire: {driven_wire:.3f}m per side",
            f"Reflector wire: {reflector_wire:.3f}m per side",
            "Tune highest band first, work downward",
            "Wire-to-wire clearance at tips is critical",
        ],
        nec_hints=[
            "6 GW cards per band: 3 driven wire sections + 3 reflector sections",
            "Driven wire is a W-shape in X-Y plane; reflector below it by vertical_spacing",
            "Each element forms an inverted-V between adjacent spreader tips",
            f"Driven: 6 wire sections around hexagonal frame (radius {frame_radius:.4f}m)",
            f"Reflector: same pattern, offset Z=-{vertical_spacing:.4f}m",
            "Feed at centre of driven element",
        ],
    )


# ---------------------------------------------------------------------------
# PHASED ARRAY (2-element)
# ---------------------------------------------------------------------------

def calc_phased_array(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    n_elements: int = 2,
    element_type: str = "vertical",
    spacing_wl: float = 0.25,
) -> CalcResult:
    """Phased array starting dimensions."""
    wl = _wavelength(freq_mhz)

    if element_type == "vertical":
        element_length = wl / 4.0 * 0.95
    else:
        element_length = wl / 2.0 * 0.95

    spacing = wl * spacing_wl

    # For λ/4 spacing with 90° phase: cardoid pattern
    # For λ/2 spacing with 180° phase: broadside pattern
    phase_per_element = 360 * spacing_wl  # degrees for end-fire

    # Phasing line (for passive phasing with coax)
    if spacing_wl <= 0.25:
        phasing_line_z0 = 75.0
        phasing_line_length = wl * spacing_wl * 0.66  # velocity factor ≈ 0.66
    else:
        phasing_line_z0 = 50.0
        phasing_line_length = wl * spacing_wl * 0.66

    return CalcResult(
        antenna_type="phased_array",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "n_elements": n_elements,
            "element_type": element_type,
            "element_length": element_length,
            "element_spacing": spacing,
            "spacing_wl": spacing_wl,
            "phase_shift_deg": phase_per_element,
            "phasing_line_z0": phasing_line_z0,
            "phasing_line_length": phasing_line_length,
        },
        notes=[
            f"{n_elements}-element phased {element_type}s, "
            f"spacing {spacing:.3f}m ({spacing_wl}λ)",
            f"Phase shift for end-fire: {phase_per_element:.0f}° per element",
            "For cardioid: λ/4 spacing + 90° phase offset",
            "Mutual coupling changes individual element impedance!",
        ],
    )


# ---------------------------------------------------------------------------
# LOOP (full-wave)
# ---------------------------------------------------------------------------

def calc_loop(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
    shape: str = "square",
) -> CalcResult:
    """Full-wave loop dimensions (square or circular)."""
    wl = _wavelength(freq_mhz)
    perimeter = wl * 1.02  # correction factor

    if shape == "circular":
        radius = perimeter / (2 * math.pi)
        dims = {
            "perimeter": perimeter,
            "radius": radius,
            "diameter": radius * 2,
            "shape": "circular",
        }
    else:
        side = perimeter / 4.0
        dims = {
            "perimeter": perimeter,
            "side_length": side,
            "shape": "square",
        }

    return CalcResult(
        antenna_type="loop",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions=dims,
        notes=[
            f"Perimeter: {perimeter:.3f}m (1.02λ)",
            "Impedance ≈ 120Ω — match with λ/4 75Ω coax",
            "~1 dB gain over dipole at same height",
        ],
    )


# ---------------------------------------------------------------------------
# HALF SQUARE
# ---------------------------------------------------------------------------

def calc_half_square(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Half-square antenna dimensions."""
    wl = _wavelength(freq_mhz)
    vertical_length = wl / 4.0 * 0.95
    horizontal_length = wl / 2.0 * 0.95
    total_width = horizontal_length
    total_height = vertical_length

    return CalcResult(
        antenna_type="half_square",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "vertical_length": vertical_length,
            "horizontal_length": horizontal_length,
            "total_width": total_width,
            "total_height": total_height,
        },
        notes=[
            f"Two λ/4 verticals ({vertical_length:.3f}m) + λ/2 horizontal top",
            "Bidirectional, broadside to the horizontal wire",
            "Low-angle radiation — good DX performance",
            "Feed at the base of one vertical — ~50Ω impedance",
        ],
    )


# ---------------------------------------------------------------------------
# BOBTAIL CURTAIN
# ---------------------------------------------------------------------------

def calc_bobtail_curtain(
    freq_mhz: float,
    wire_dia_mm: float = 1.628,
) -> CalcResult:
    """Bobtail curtain dimensions — 3 verticals + 2 horizontal wires."""
    wl = _wavelength(freq_mhz)
    vertical_length = wl / 4.0 * 0.95
    horizontal_length = wl / 2.0 * 0.95
    total_width = horizontal_length * 2  # two horizontal spans

    return CalcResult(
        antenna_type="bobtail_curtain",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "vertical_length": vertical_length,
            "horizontal_length": horizontal_length,
            "total_width": total_width,
            "vertical_count": 3,
        },
        notes=[
            f"Three λ/4 verticals ({vertical_length:.3f}m each)",
            f"Two λ/2 horizontal top wires ({horizontal_length:.3f}m each)",
            "~5 dBd gain, bidirectional",
            "Feed at base of centre vertical",
        ],
    )


# ---------------------------------------------------------------------------
# HELIX (axial mode)
# ---------------------------------------------------------------------------

def calc_helix(
    freq_mhz: float,
    wire_dia_mm: float = 2.0,
    n_turns: int = 8,
) -> CalcResult:
    """Axial-mode helical antenna dimensions."""
    wl = _wavelength(freq_mhz)

    # Axial mode: circumference ≈ 1λ → diameter ≈ λ/π
    diameter = wl / math.pi
    radius = diameter / 2.0
    circumference = math.pi * diameter

    # Pitch ≈ λ/4 per turn (12.5° pitch angle for C=λ)
    pitch = wl * 0.25
    pitch_angle_deg = math.degrees(math.atan(pitch / circumference))

    total_length = n_turns * pitch
    # Ground plane diameter ≈ 0.75λ
    ground_plane_dia = wl * 0.75

    # Approximate gain: G ≈ 15 * C²nS/λ³ where C = circumference, n = turns, S = pitch
    gain_linear = 15 * circumference ** 2 * n_turns * pitch / wl ** 3
    gain_dbi = 10 * math.log10(max(gain_linear, 0.1))

    return CalcResult(
        antenna_type="helix",
        freq_mhz=freq_mhz,
        wavelength_m=wl,
        wire_dia_mm=wire_dia_mm,
        dimensions={
            "diameter": diameter,
            "radius": radius,
            "circumference": circumference,
            "pitch": pitch,
            "pitch_angle_deg": pitch_angle_deg,
            "n_turns": n_turns,
            "total_length": total_length,
            "ground_plane_diameter": ground_plane_dia,
        },
        notes=[
            f"{n_turns}-turn helix, diameter {diameter:.3f}m, pitch {pitch:.3f}m",
            f"Total axial length: {total_length:.3f}m",
            f"Expected gain ≈ {gain_dbi:.1f} dBi",
            "Circular polarisation (RHCP if wound clockwise from feed)",
            f"Ground plane: ≥ {ground_plane_dia:.3f}m diameter",
            "Feed impedance ≈ 140Ω — use λ/4 matching section",
        ],
    )


# ---------------------------------------------------------------------------
# MATCHING NETWORKS
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result from a matching network calculator."""
    match_type: str
    components: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "components": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.components.items()
            },
            "notes": self.notes,
        }


def calc_quarter_wave_match(
    z_load: float,
    z_source: float = 50.0,
    freq_mhz: float = 14.175,
    velocity_factor: float = 0.66,
) -> MatchResult:
    """Quarter-wave impedance transformer.

    Z0_transformer = sqrt(Z_load × Z_source)
    Physical length = (λ/4) × velocity_factor
    """
    wl = _wavelength(freq_mhz)
    z_transformer = math.sqrt(z_load * z_source)
    electrical_length = wl / 4.0
    physical_length = electrical_length * velocity_factor

    return MatchResult(
        match_type="quarter_wave_transformer",
        components={
            "z_transformer": z_transformer,
            "electrical_length_m": electrical_length,
            "physical_length_m": physical_length,
            "velocity_factor": velocity_factor,
        },
        notes=[
            f"Z0 = √({z_load} × {z_source}) = {z_transformer:.1f}Ω",
            f"Physical length = {physical_length:.3f}m "
            f"(λ/4 × VF={velocity_factor})",
            f"Use {z_transformer:.0f}Ω transmission line",
            "Common: 75Ω coax matches 112Ω to 50Ω",
        ],
    )


def calc_gamma_match(
    z_antenna: float = 25.0,
    z_target: float = 50.0,
    freq_mhz: float = 14.175,
    element_dia_mm: float = 12.0,
) -> MatchResult:
    """Gamma match starting dimensions.

    Commonly used on Yagi driven elements where Z ≈ 25Ω.
    """
    wl = _wavelength(freq_mhz)
    # Gamma rod length ≈ 0.04–0.05λ
    rod_length = wl * 0.045
    # Rod spacing from element ≈ 2–3× element diameter
    rod_spacing = element_dia_mm / 1000.0 * 3.0
    # Gamma rod diameter ≈ 30–50% of element diameter
    rod_dia = element_dia_mm * 0.4
    # Series capacitor (approximate)
    # C ≈ 1/(2πf × Xc) where Xc ≈ sqrt(Z_ant × Z_target)
    Xc = math.sqrt(z_antenna * z_target)
    C_pf = 1e12 / (2 * math.pi * freq_mhz * 1e6 * Xc)

    return MatchResult(
        match_type="gamma_match",
        components={
            "rod_length_m": rod_length,
            "rod_spacing_m": rod_spacing,
            "rod_diameter_mm": rod_dia,
            "series_capacitor_pf": C_pf,
            "rod_to_element_ratio": rod_dia / element_dia_mm,
        },
        notes=[
            f"Gamma rod: {rod_length * 1000:.0f}mm long, "
            f"{rod_dia:.1f}mm diameter",
            f"Spacing from element: {rod_spacing * 1000:.0f}mm",
            f"Series capacitor: ≈{C_pf:.0f} pF",
            "Adjust rod length and spacing on the antenna for best SWR",
            "Rod diameter = 30–50% of element diameter",
        ],
    )


def calc_hairpin_match(
    z_antenna: float = 25.0,
    z_target: float = 50.0,
    freq_mhz: float = 14.175,
) -> MatchResult:
    """Hairpin (beta) match dimensions.

    A shorted stub across the driven element feedpoint provides
    the inductive reactance needed to match low Z to 50Ω.
    """
    wl = _wavelength(freq_mhz)
    # Required shunt inductance
    # For step-up from Z_ant to Z_target:
    #   X_L = sqrt(Z_ant × Z_target - Z_ant²) × Z_target / (Z_target - Z_ant)
    if z_target <= z_antenna:
        return MatchResult(
            match_type="hairpin_match",
            components={},
            notes=["Hairpin not needed — Z_ant ≥ Z_target"],
        )

    X_L = math.sqrt(z_antenna * z_target * (z_target - z_antenna)) / (z_target - z_antenna)
    # Hairpin stub length for shorted transmission line
    # X_L = Z0 × tan(β × L) → L = arctan(X_L / Z0) / β
    Z0_stub = 300.0  # open-wire hairpin ~300Ω
    beta = 2 * math.pi / wl
    stub_length = math.atan(X_L / Z0_stub) / beta

    # Wire spacing for ~300Ω twin-lead
    # Z0 ≈ 276 × log10(2D/d) for D = spacing, d = wire diameter
    wire_dia = 3.0  # mm
    wire_spacing = (wire_dia / 2.0) * 10 ** (Z0_stub / 276.0)

    return MatchResult(
        match_type="hairpin_match",
        components={
            "hairpin_length_m": stub_length,
            "hairpin_z0": Z0_stub,
            "reactance_ohms": X_L,
            "wire_spacing_mm": wire_spacing,
            "wire_dia_mm": wire_dia,
        },
        notes=[
            f"Hairpin length: {stub_length * 1000:.0f}mm",
            f"Required reactance: {X_L:.1f}Ω inductive",
            f"Wire spacing: {wire_spacing:.0f}mm for Z0≈{Z0_stub:.0f}Ω",
            "Solder hairpin across the split driven element feedpoint",
        ],
    )


def calc_balun(
    z_balanced: float = 73.0,
    z_unbalanced: float = 50.0,
    freq_mhz: float = 14.175,
) -> MatchResult:
    """Balun selection guide."""
    ratio = z_balanced / z_unbalanced
    nearest_ratio = min([1, 2, 4, 6, 9], key=lambda r: abs(r - ratio))

    return MatchResult(
        match_type="balun",
        components={
            "impedance_ratio": f"{nearest_ratio}:1",
            "z_balanced": z_balanced,
            "z_unbalanced": z_unbalanced,
            "actual_ratio": ratio,
        },
        notes=[
            f"Nearest standard ratio: {nearest_ratio}:1",
            "1:1 current balun for dipoles (choke common-mode)",
            "4:1 for quads (200Ω→50Ω) or folded dipoles (300Ω→75Ω)",
            "Use ferrite cores appropriate for the frequency range",
            "HF: FT240-43, VHF: FT240-61",
        ],
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

_CALC_MAP: dict[str, Any] = {
    "dipole": calc_dipole,
    "inverted_v": calc_inverted_v,
    "vertical": calc_vertical,
    "j_pole": calc_j_pole,
    "end_fed": calc_end_fed,
    "yagi": calc_yagi,
    "moxon": calc_moxon,
    "quad": calc_quad,
    "delta_loop": calc_delta_loop,
    "collinear": calc_collinear,
    "discone": calc_discone,
    "hexbeam": calc_hexbeam,
    "magnetic_loop": calc_magnetic_loop,
    "phased_array": calc_phased_array,
    "loop": calc_loop,
    "half_square": calc_half_square,
    "bobtail_curtain": calc_bobtail_curtain,
    "helix": calc_helix,
}


def calc_for_type(
    antenna_type: str,
    freq_mhz: float,
    **kwargs: Any,
) -> CalcResult | None:
    """Compute starting dimensions for any supported antenna type.

    Extra keyword arguments are passed through to the type-specific
    calculator (e.g. ``n_elements=5`` for Yagi, ``wire_dia_mm=2.0``).

    Returns ``None`` if the antenna type has no calculator.
    """
    fn = _CALC_MAP.get(antenna_type.lower().strip())
    if fn is None:
        return None
    # LPDA needs different args
    if antenna_type == "lpda":
        return fn(**kwargs)
    return fn(freq_mhz=freq_mhz, **kwargs)


def supported_types() -> list[str]:
    """Return all antenna types that have calculators."""
    return sorted(_CALC_MAP.keys())


def calc_match(
    match_type: str,
    **kwargs: Any,
) -> MatchResult | None:
    """Compute matching network dimensions.

    match_type: "quarter_wave", "gamma", "hairpin", or "balun"
    """
    match_map = {
        "quarter_wave": calc_quarter_wave_match,
        "gamma": calc_gamma_match,
        "hairpin": calc_hairpin_match,
        "balun": calc_balun,
    }
    fn = match_map.get(match_type.lower().strip())
    if fn is None:
        return None
    return fn(**kwargs)
