"""3D geometry extraction from parsed NEC files for visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .parser import ParseResult


@dataclass
class _Wire:
    """Internal wire representation during geometry extraction."""
    tag: int
    points: list[list[float]]
    radius: float
    segments: int
    is_excited: bool = False
    is_loaded: bool = False
    excited_segments: list[int] = field(default_factory=list)
    loaded_segments: list[int] = field(default_factory=list)


def _to_float(val: Any, default: float = 0.0) -> float | None:
    """Convert a value to float. Returns None if it's an unresolved SY expression."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return default


def extract_geometry(parsed: ParseResult) -> dict[str, Any]:
    """Extract 3D geometry from a parsed NEC file for visualization.

    Returns a dict with keys: wires, ground_type, bounds.
    Coordinate system follows NEC convention (Z-up).
    """
    wires: list[_Wire] = []

    for card in parsed.cards:
        ct = card.card_type
        p = card.labeled_params

        if ct == "GW":
            coords = [_to_float(p.get(k, 0.0)) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
            if any(c is None for c in coords):
                continue  # skip wires with unresolved SY expressions
            wires.append(_Wire(
                tag=p.get("tag", 0),
                points=[
                    [coords[0], coords[1], coords[2]],
                    [coords[3], coords[4], coords[5]],
                ],
                radius=_to_float(p.get("radius", 0.001), 0.001) or 0.001,
                segments=p.get("segments", 1),
            ))

        elif ct == "GA":
            tag = p.get("tag", 0)
            segs = p.get("segments", 10) or 10
            arc_r = p.get("arcRadius", 1.0) or 1.0
            start_deg = p.get("startAngle", 0.0) or 0.0
            end_deg = p.get("endAngle", 360.0) or 360.0
            wire_r = p.get("wireRadius") or 0.001
            n_pts = segs + 1
            points = []
            for i in range(n_pts):
                theta = math.radians(
                    start_deg + (end_deg - start_deg) * i / max(n_pts - 1, 1)
                )
                points.append([arc_r * math.cos(theta), 0.0, arc_r * math.sin(theta)])
            wires.append(_Wire(tag=tag, points=points, radius=wire_r, segments=segs))

        elif ct == "GH":
            tag = p.get("tag", 0)
            segs = p.get("segments", 10) or 10
            spacing = p.get("spacing", 0.0) or 0.0
            hl = p.get("hl", 1.0) or 1.0
            r1 = p.get("r1", 0.1) or 0.1
            r2 = p.get("r2", 0.1) or 0.1
            r3 = p.get("r3", 0.1) or 0.1
            r4 = p.get("r4", 0.1) or 0.1
            wire_r = p.get("wireRadius") or 0.001
            n_turns = abs(hl / spacing) if spacing else 1
            n_pts = segs + 1
            points = []
            for i in range(n_pts):
                t = i / max(n_pts - 1, 1)
                z = hl * t
                theta = 2 * math.pi * n_turns * t
                rx = r1 + (r3 - r1) * t
                ry = r2 + (r4 - r2) * t
                points.append([rx * math.cos(theta), ry * math.sin(theta), z])
            wires.append(_Wire(tag=tag, points=points, radius=wire_r, segments=segs))

        elif ct == "GS":
            factor = _to_float(p.get("factor", 1.0), 1.0) or 1.0
            for w in wires:
                for pt in w.points:
                    pt[0] *= factor
                    pt[1] *= factor
                    pt[2] *= factor
                w.radius *= factor

        elif ct == "GX":
            i1 = p.get("i1", 0)
            i2 = p.get("i2", 0)
            reflect_y = (i2 % 10) >= 1
            reflect_x = (i2 // 10 % 10) >= 1
            reflect_z = (i2 // 100 % 10) >= 1
            new_wires = []
            for w in list(wires):
                new_points = []
                for pt in w.points:
                    np = [pt[0], pt[1], pt[2]]
                    if reflect_x:
                        np[0] = -np[0]
                    if reflect_y:
                        np[1] = -np[1]
                    if reflect_z:
                        np[2] = -np[2]
                    new_points.append(np)
                new_wires.append(_Wire(
                    tag=w.tag + i1,
                    points=new_points,
                    radius=w.radius,
                    segments=w.segments,
                ))
            wires.extend(new_wires)

        elif ct == "GM":
            i1_tag = p.get("i1", 0)
            i2_tag = p.get("i2", 0)
            ro_x = math.radians(_to_float(p.get("roX", 0.0), 0.0) or 0.0)
            ro_y = math.radians(_to_float(p.get("roY", 0.0), 0.0) or 0.0)
            ro_z = math.radians(_to_float(p.get("roZ", 0.0), 0.0) or 0.0)
            tr_x = _to_float(p.get("trX", 0.0), 0.0) or 0.0
            tr_y = _to_float(p.get("trY", 0.0), 0.0) or 0.0
            tr_z = _to_float(p.get("trZ", 0.0), 0.0) or 0.0
            its = int(_to_float(p.get("its", 0), 0) or 0)

            def match_tag(tag: int) -> bool:
                if i1_tag == 0 and i2_tag == 0:
                    return True
                return i1_tag <= tag <= i2_tag

            if its == 0:
                for w in wires:
                    if match_tag(w.tag):
                        for pt in w.points:
                            _rotate_translate(pt, ro_x, ro_y, ro_z, tr_x, tr_y, tr_z)
            else:
                new_wires = []
                for w in wires:
                    if match_tag(w.tag):
                        new_points = []
                        for pt in w.points:
                            np = [pt[0], pt[1], pt[2]]
                            _rotate_translate(np, ro_x, ro_y, ro_z, tr_x, tr_y, tr_z)
                            new_points.append(np)
                        new_wires.append(_Wire(
                            tag=w.tag + its,
                            points=new_points,
                            radius=w.radius,
                            segments=w.segments,
                        ))
                wires.extend(new_wires)

    # Collect excitations, loads, transmission lines, frequency, ground
    excitations: list[dict] = []
    loads: list[dict] = []
    transmission_lines: list[dict] = []
    frequency: float | None = None
    ground_type = "none"
    ground_params: dict = {}

    _LD_TYPE_NAMES = {
        0: "series RLC",
        1: "parallel RLC",
        2: "series RLC (per length)",
        3: "parallel RLC (per length)",
        4: "impedance",
        5: "wire conductivity",
    }

    for card in parsed.cards:
        ct = card.card_type
        p = card.labeled_params

        if ct == "EX":
            ex_type = p.get("exType", 0)
            if ex_type in (0, 5):
                tag = p.get("tag", 0)
                seg = p.get("segment", 0)
                v_real = _to_float(p.get("vReal", 0.0), 0.0) or 0.0
                v_imag = _to_float(p.get("vImag", 0.0), 0.0) or 0.0
                for w in wires:
                    if w.tag == tag:
                        w.is_excited = True
                        w.excited_segments.append(seg)
                excitations.append({
                    "tag": tag, "segment": seg,
                    "v_real": round(v_real, 6), "v_imag": round(v_imag, 6),
                })

        elif ct == "LD":
            tag = p.get("tag", 0)
            ld_type = p.get("ldType", 0)
            s_start = p.get("segStart", 0)
            s_end = p.get("segEnd", 0)
            zlr = _to_float(p.get("zlr", 0.0), 0.0) or 0.0
            zli = _to_float(p.get("zli", 0.0), 0.0) or 0.0
            zlc = _to_float(p.get("zlc", 0.0), 0.0) or 0.0
            for w in wires:
                if w.tag == tag or tag == 0:
                    w.is_loaded = True
                    w.loaded_segments.extend(range(s_start, max(s_end, s_start) + 1))
            loads.append({
                "ld_type": ld_type,
                "type_name": _LD_TYPE_NAMES.get(ld_type, f"type {ld_type}"),
                "tag": tag,
                "seg_start": s_start, "seg_end": s_end,
                "zlr": zlr, "zli": zli, "zlc": zlc,
            })

        elif ct == "TL":
            tag1 = p.get("tag1", 0)
            seg1 = p.get("seg1", 0)
            tag2 = p.get("tag2", 0)
            seg2 = p.get("seg2", 0)
            z0 = _to_float(p.get("z0", 0.0), 0.0) or 0.0
            tl_len = _to_float(p.get("length", 0.0), 0.0) or 0.0
            transmission_lines.append({
                "tag1": tag1, "seg1": seg1,
                "tag2": tag2, "seg2": seg2,
                "z0": round(z0, 2), "length": round(tl_len, 6),
            })

        elif ct == "FR":
            freq_val = _to_float(p.get("freq", 0.0), 0.0)
            if freq_val:
                frequency = round(freq_val, 6)

        elif ct == "GN":
            gt = p.get("groundType", -1)
            if gt == -1:
                ground_type = "free_space"
            elif gt == 1:
                ground_type = "perfect"
            else:
                ground_type = "real"
            ground_params = {
                "type_code": gt,
                "epsr": _to_float(p.get("epsr", 0.0), 0.0) or 0.0,
                "sigma": _to_float(p.get("sigma", 0.0), 0.0) or 0.0,
            }

    # Compute bounds
    all_pts = [pt for w in wires for pt in w.points]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        bounds_min = [min(xs), min(ys), min(zs)]
        bounds_max = [max(xs), max(ys), max(zs)]
    else:
        bounds_min = [0, 0, 0]
        bounds_max = [1, 1, 1]

    # Compute wire lengths and inter-wire spacing
    wire_info: list[dict] = []
    for w in wires:
        length = 0.0
        for i in range(len(w.points) - 1):
            dx = w.points[i + 1][0] - w.points[i][0]
            dy = w.points[i + 1][1] - w.points[i][1]
            dz = w.points[i + 1][2] - w.points[i][2]
            length += math.sqrt(dx * dx + dy * dy + dz * dz)
        midpt = _wire_midpoint(w.points)
        wire_info.append({"tag": w.tag, "length": round(length, 4), "midpoint": midpt})

    spacings: list[dict] = []
    for i in range(len(wire_info)):
        for j in range(i + 1, len(wire_info)):
            m1 = wire_info[i]["midpoint"]
            m2 = wire_info[j]["midpoint"]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(m1, m2)))
            spacings.append({
                "wire_a": wire_info[i]["tag"],
                "wire_b": wire_info[j]["tag"],
                "distance": round(d, 4),
            })

    return {
        "wires": [
            {
                "tag": w.tag,
                "points": [[round(c, 6) for c in pt] for pt in w.points],
                "radius": round(w.radius, 6),
                "segments": w.segments,
                "is_excited": w.is_excited,
                "is_loaded": w.is_loaded,
                "excited_segments": w.excited_segments,
                "loaded_segments": w.loaded_segments,
            }
            for w in wires
        ],
        "ground_type": ground_type,
        "ground_params": ground_params,
        "frequency": frequency,
        "excitations": excitations,
        "transmission_lines": transmission_lines,
        "loads": loads,
        "wire_dimensions": wire_info,
        "spacings": spacings,
        "bounds": {
            "min": [round(v, 6) for v in bounds_min],
            "max": [round(v, 6) for v in bounds_max],
        },
    }


def _wire_midpoint(points: list[list[float]]) -> list[float]:
    """Return the midpoint of a wire defined by a list of points."""
    if not points:
        return [0.0, 0.0, 0.0]
    n = len(points)
    return [
        round(sum(p[0] for p in points) / n, 6),
        round(sum(p[1] for p in points) / n, 6),
        round(sum(p[2] for p in points) / n, 6),
    ]


def _rotate_translate(
    pt: list[float],
    rx: float, ry: float, rz: float,
    tx: float, ty: float, tz: float,
) -> None:
    """Apply Euler rotations (X, Y, Z order) then translate. Modifies in-place."""
    x, y, z = pt
    if rx:
        c, s = math.cos(rx), math.sin(rx)
        y, z = c * y - s * z, s * y + c * z
    if ry:
        c, s = math.cos(ry), math.sin(ry)
        x, z = c * x + s * z, -s * x + c * z
    if rz:
        c, s = math.cos(rz), math.sin(rz)
        x, y = c * x - s * y, s * x + c * y
    pt[0] = x + tx
    pt[1] = y + ty
    pt[2] = z + tz
