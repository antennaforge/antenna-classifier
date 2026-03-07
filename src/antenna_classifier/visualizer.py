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

    # Mark excitations and loads
    for card in parsed.cards:
        if card.card_type == "EX":
            ex_type = card.labeled_params.get("exType", 0)
            if ex_type in (0, 5):
                tag = card.labeled_params.get("tag", 0)
                seg = card.labeled_params.get("segment", 0)
                for w in wires:
                    if w.tag == tag:
                        w.is_excited = True
                        w.excited_segments.append(seg)
        elif card.card_type == "LD":
            tag = card.labeled_params.get("tag", 0)
            s_start = card.labeled_params.get("segStart", 0)
            s_end = card.labeled_params.get("segEnd", 0)
            for w in wires:
                if w.tag == tag or tag == 0:
                    w.is_loaded = True
                    w.loaded_segments.extend(range(s_start, max(s_end, s_start) + 1))

    # Ground type
    ground_type = "none"
    for card in parsed.cards:
        if card.card_type == "GN":
            gt = card.labeled_params.get("groundType", -1)
            if gt == -1:
                ground_type = "free_space"
            elif gt == 1:
                ground_type = "perfect"
            else:
                ground_type = "real"

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
        "bounds": {
            "min": [round(v, 6) for v in bounds_min],
            "max": [round(v, 6) for v in bounds_max],
        },
    }


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
