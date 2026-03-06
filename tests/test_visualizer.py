"""Tests for the visualizer geometry extraction module."""

import math
import pytest
from antenna_classifier.parser import parse_text
from antenna_classifier.visualizer import extract_geometry


def _dipole_nec(z_height=10.0):
    return f"""\
CM Test dipole
CE
GW 1 11 0 -0.5 {z_height} 0 0.5 {z_height} 0.001
GE 0
EX 0 1 6 0 1.0
FR 0 1 0 0 14.0
GN 1
EN
"""


class TestBasicGeometry:
    def test_wire_extraction(self):
        geo = extract_geometry(parse_text(_dipole_nec(), "d.nec"))
        assert len(geo["wires"]) == 1
        w = geo["wires"][0]
        assert w["tag"] == 1
        assert w["segments"] == 11
        assert len(w["points"]) == 2
        assert w["points"][0] == [0.0, -0.5, 10.0]
        assert w["points"][1] == [0.0, 0.5, 10.0]
        assert w["radius"] == 0.001

    def test_excitation_marked(self):
        geo = extract_geometry(parse_text(_dipole_nec(), "d.nec"))
        w = geo["wires"][0]
        assert w["is_excited"] is True
        assert 6 in w["excited_segments"]

    def test_ground_perfect(self):
        geo = extract_geometry(parse_text(_dipole_nec(), "d.nec"))
        assert geo["ground_type"] == "perfect"

    def test_ground_none(self):
        nec = "CM test\nCE\nGW 1 11 0 -0.5 0 0 0.5 0 0.001\nGE\nEX 0 1 6 0 1\nFR 0 1 0 0 14\nEN\n"
        geo = extract_geometry(parse_text(nec, "t.nec"))
        assert geo["ground_type"] == "none"

    def test_bounds(self):
        geo = extract_geometry(parse_text(_dipole_nec(10.0), "d.nec"))
        assert geo["bounds"]["min"] == [0.0, -0.5, 10.0]
        assert geo["bounds"]["max"] == [0.0, 0.5, 10.0]

    def test_empty_geometry(self):
        nec = "CM empty\nCE\nEN\n"
        geo = extract_geometry(parse_text(nec, "e.nec"))
        assert geo["wires"] == []
        assert geo["bounds"]["min"] == [0, 0, 0]


class TestLoading:
    def test_loaded_wire_marked(self):
        nec = """\
CM loaded
CE
GW 1 11 0 -0.5 10 0 0.5 10 0.001
GE 0
EX 0 1 6 0 1
LD 5 1 1 11 5.8e7
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "l.nec"))
        w = geo["wires"][0]
        assert w["is_loaded"] is True
        assert len(w["loaded_segments"]) == 11


class TestGXReflection:
    def test_reflection_doubles_wires(self):
        nec = """\
CM reflect
CE
GW 1 11 0 0 0 0 0 5 0.001
GX 1 100
GE 0
EX 0 1 6 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "r.nec"))
        assert len(geo["wires"]) == 2
        # Original wire
        assert geo["wires"][0]["points"][1][2] == 5.0
        # Reflected wire has Z negated
        assert geo["wires"][1]["points"][1][2] == -5.0
        assert geo["wires"][1]["tag"] == 2

    def test_reflection_y_axis(self):
        nec = """\
CE
GW 1 5 0 1 0 0 2 0 0.001
GX 1 1
GE
EX 0 1 3 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "ry.nec"))
        assert len(geo["wires"]) == 2
        # Original: y1=1, y2=2
        assert geo["wires"][0]["points"][0][1] == 1.0
        # Reflected: y1=-1, y2=-2
        assert geo["wires"][1]["points"][0][1] == -1.0


class TestGSScaling:
    def test_scale_factor(self):
        nec = """\
CE
GW 1 11 0 -6 120 0 6 120 0.5
GS 0 0 0.0254
GE 0
EX 0 1 6 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "s.nec"))
        w = geo["wires"][0]
        # 120 inches * 0.0254 = 3.048 meters
        assert abs(w["points"][0][2] - 3.048) < 0.001
        # 6 inches * 0.0254 = 0.1524
        assert abs(w["points"][1][1] - 0.1524) < 0.001


class TestGMTransform:
    def test_translate_in_place(self):
        nec = """\
CE
GW 1 5 0 0 0 1 0 0 0.001
GM 0 0 0 0 0 0 0 5
GE
EX 0 1 3 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "gm.nec"))
        # Wire should be translated 5 meters up in Z
        assert abs(geo["wires"][0]["points"][0][2] - 5.0) < 1e-6
        assert abs(geo["wires"][0]["points"][1][2] - 5.0) < 1e-6

    def test_copy_with_tag_increment(self):
        nec = """\
CE
GW 1 5 0 0 0 1 0 0 0.001
GM 0 0 0 0 0 0 0 2 1
GE
EX 0 1 3 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "gmc.nec"))
        assert len(geo["wires"]) == 2
        # Original stays at z=0
        assert abs(geo["wires"][0]["points"][0][2]) < 1e-6
        # Copy at z=2 with tag=2
        assert geo["wires"][1]["tag"] == 2
        assert abs(geo["wires"][1]["points"][0][2] - 2.0) < 1e-6


class TestHelixGeometry:
    def test_helix_points(self):
        nec = """\
CE
GH 1 36 0.1 1.0 0.15 0.15 0.15 0.15 0.001
GE
EX 0 1 1 0 1
FR 0 1 0 0 300
EN
"""
        geo = extract_geometry(parse_text(nec, "h.nec"))
        assert len(geo["wires"]) == 1
        pts = geo["wires"][0]["points"]
        assert len(pts) == 37  # 36 segments + 1
        # First point at Z=0, last at Z=1.0
        assert abs(pts[0][2]) < 1e-6
        assert abs(pts[-1][2] - 1.0) < 1e-6
        # Should trace a circle of radius 0.15
        assert abs(pts[0][0] - 0.15) < 1e-4


class TestArcGeometry:
    def test_arc_points(self):
        nec = """\
CE
GA 1 10 1.0 0.0 180.0 0.001
GE
EX 0 1 5 0 1
FR 0 1 0 0 14
EN
"""
        geo = extract_geometry(parse_text(nec, "a.nec"))
        assert len(geo["wires"]) == 1
        pts = geo["wires"][0]["points"]
        assert len(pts) == 11  # 10 segments + 1
        # First point: cos(0)=1, sin(0)=0
        assert abs(pts[0][0] - 1.0) < 1e-6
        assert abs(pts[0][2]) < 1e-6
        # Last point: cos(180°)=-1, sin(180°)≈0
        assert abs(pts[-1][0] - (-1.0)) < 1e-6
        assert abs(pts[-1][2]) < 1e-6


class TestOutputFormat:
    def test_json_serializable(self):
        """Geometry output must be JSON-serializable (no numpy, no custom objects)."""
        import json
        geo = extract_geometry(parse_text(_dipole_nec(), "d.nec"))
        serialized = json.dumps(geo)
        assert isinstance(serialized, str)

    def test_coordinates_rounded(self):
        geo = extract_geometry(parse_text(_dipole_nec(), "d.nec"))
        for w in geo["wires"]:
            for pt in w["points"]:
                for coord in pt:
                    # Should have at most 6 decimal places
                    assert coord == round(coord, 6)
