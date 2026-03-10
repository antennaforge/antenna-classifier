"""Unit tests for antenna_classifier.classifier."""

import pytest

from antenna_classifier.parser import parse_text
from antenna_classifier.classifier import ANTENNA_TYPES, classify


def _dipole_nec(comment: str = "Half-wave dipole") -> str:
    return (
        f"CM {comment}\n"
        "GW 1 21 0 -5.0 0 0 5.0 0 0.001\n"
        "GE 0\n"
        "EX 0 1 11 0 1 0\n"
        "FR 0 1 0 0 14.15\n"
        "EN\n"
    )


def _yagi_nec(n_elements: int = 5) -> str:
    """Build a multi-element yagi along the X boom axis."""
    lines = ["CM yagi-uda antenna"]
    spacing = 3.0  # metres between elements along X
    for i in range(1, n_elements + 1):
        x = (i - 1) * spacing
        half_len = 5.0 - 0.3 * i  # progressively shorter directors
        lines.append(f"GW {i} 21 {x} -{half_len} 0 {x} {half_len} 0 0.001")
    lines.append("GE 0")
    lines.append("EX 0 1 11 0 1 0")
    lines.append("FR 0 1 0 0 14.15")
    lines.append("EN")
    return "\n".join(lines)


def _vertical_nec() -> str:
    return (
        "CM Vertical antenna\n"
        "GW 1 21 0 0 0 0 0 10.0 0.001\n"
        "GE 0\n"
        "GN 1\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 7.1\n"
        "EN\n"
    )


def _loop_nec() -> str:
    """Single quad loop (4 sides)."""
    return (
        "CM Full wave loop\n"
        "GW 1 11 0 0 0 5 0 0 0.001\n"
        "GW 1 11 5 0 0 5 0 5 0.001\n"
        "GW 1 11 5 0 5 0 0 5 0.001\n"
        "GW 1 11 0 0 5 0 0 0 0.001\n"
        "GE 0\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 14.15\n"
        "EN\n"
    )


def _helix_nec() -> str:
    return (
        "CM Helical antenna\n"
        "GH 1 36 0.3 3.0 0.15 0.15 0.15 0.15 0.001\n"
        "GE 0\n"
        "GN 1\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 435.0\n"
        "EN\n"
    )


def _lpda_nec() -> str:
    """LPDA with TL feed and progressive element lengths."""
    lines = ["CM log-periodic dipole array"]
    for i in range(1, 9):
        x = i * 1.5
        half_len = 3.0 + i * 0.4  # monotonically increasing
        lines.append(f"GW {i} 21 {x} -{half_len} 0 {x} {half_len} 0 0.001")
    lines.append("GE 0")
    for i in range(1, 8):
        lines.append(f"TL {i} 11 {i+1} 11 300.0")
    lines.append("EX 0 1 11 0 1 0")
    lines.append("FR 0 1 0 0 14.0")
    lines.append("EN")
    return "\n".join(lines)


def _hybrid_log_orr_single_tl_nec() -> str:
    """Progressive-length array with only one TL link: not strict enough for LPDA."""
    return (
        "CM log-Orr 10m\n"
        "CE\n"
        "GW 1,21,-2.809013,0.,0.,2.711731,0.,0.,.00635\n"
        "GW 2,21,-2.359084,.7539341,0.,2.359084,.7539341,0.,.00635\n"
        "GW 3,21,-2.1402,1.775393,0.,2.1402,1.775393,0.,.00635\n"
        "GW 4,21,-2.371244,3.575107,0.,2.371244,3.575107,0.,.00635\n"
        "GW 5,21,-2.334764,6.080114,0.,2.334764,6.080114,0.,.00635\n"
        "GE 0\n"
        "LD 5,1,0,0,2.5E+07,1.\n"
        "LD 5,2,0,0,2.5E+07,1.\n"
        "LD 5,3,0,0,2.5E+07,1.\n"
        "LD 5,4,0,0,2.5E+07,1.\n"
        "LD 5,5,0,0,2.5E+07,1.\n"
        "FR 0,1,0,0,28.5\n"
        "GN -1\n"
        "EX 0,3,11,0,1.414214,0.\n"
        "TL 2,11,3,11,-150.,0.,0.,0.,0.,0.\n"
        "RP 0,1,361,1000,90.,0.,0.,1.,0.\n"
        "EN\n"
    )


def _patch_nec() -> str:
    return (
        "CM Microstrip patch\n"
        "SP 0 0.05 0.05 0 0 0 0.01\n"
        "GE 0\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 2400.0\n"
        "EN\n"
    )


def _wire_object_nec() -> str:
    """Wire-grid model with no EX or FR."""
    lines = ["CM wire grid vehicle model"]
    for i in range(1, 51):
        lines.append(f"GW {i} 5 0 0 {i*0.1} 1 0 {i*0.1} 0.001")
    lines.append("GE 0")
    lines.append("EN")
    return "\n".join(lines)


def _phased_nec() -> str:
    """Phased array with multiple excitation sources."""
    return (
        "CM Phased array\n"
        "GW 1 21 0 -5 0 0 5 0 0.001\n"
        "GW 2 21 3 -5 0 3 5 0 0.001\n"
        "GW 3 21 6 -5 0 6 5 0 0.001\n"
        "GE 0\n"
        "EX 0 1 11 0 1 0\n"
        "EX 0 2 11 0 1 0\n"
        "FR 0 1 0 0 14.0\n"
        "EN\n"
    )


# ---------------------------------------------------------------------------
# Comment-based classification
# ---------------------------------------------------------------------------

class TestCommentClassification:
    @pytest.mark.parametrize("keyword,expected_type", [
        ("yagi", "yagi"),
        ("dipole", "dipole"),
        ("vertical", "vertical"),
        ("loop", "loop"),
        ("quad", "quad"),
        ("hexbeam", "hexbeam"),
        ("log periodic", "lpda"),
        ("moxon", "moxon"),
        ("helix", "helix"),
        ("inverted v", "inverted_v"),
        ("end fed", "end_fed"),
        ("j-pole", "j_pole"),
        ("rhombic", "rhombic"),
        ("beverage", "beverage"),
        ("discone", "discone"),
        ("turnstile", "turnstile"),
        ("fractal", "fractal"),
        ("bobtail", "bobtail_curtain"),
        ("delta loop", "delta_loop"),
        ("v-beam", "v_beam"),
        ("batwing", "batwing"),
        ("zigzag", "zigzag"),
        ("magnetic loop", "magnetic_loop"),
        ("collinear", "collinear"),
    ])
    def test_keyword_detection(self, keyword, expected_type):
        nec = (
            f"CM {keyword} antenna model\n"
            "GW 1 21 0 -5 0 0 5 0 0.001\n"
            "GE\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.0\nEN\n"
        )
        result = classify(parse_text(nec))
        assert result.antenna_type == expected_type
        assert result.confidence >= 0.7


# ---------------------------------------------------------------------------
# Structural classification
# ---------------------------------------------------------------------------

class TestStructuralClassification:
    def test_helix_from_gh_card(self):
        result = classify(parse_text(_helix_nec()))
        assert result.antenna_type == "helix"
        assert result.confidence >= 0.8

    def test_patch_from_sp_card(self):
        result = classify(parse_text(_patch_nec()))
        assert result.antenna_type == "patch"
        assert result.confidence >= 0.8

    def test_wire_object_no_ex_fr(self):
        result = classify(parse_text(_wire_object_nec()))
        assert result.antenna_type == "wire_object"

    def test_vertical_with_ground(self):
        result = classify(parse_text(_vertical_nec()))
        assert result.antenna_type == "vertical"
        assert result.confidence >= 0.5

    def test_yagi_multi_element(self):
        result = classify(parse_text(_yagi_nec(5)))
        assert result.antenna_type == "yagi"
        assert result.confidence >= 0.5

    def test_yagi_stepped_diameter(self):
        """Stepped-diameter Yagi: 10 GW wires (5 per element) should merge to 2 elements."""
        nec = (
            "CM 2el stepped-dia Yagi\nCE\n"
            "GW 1,5,-2.603,0,0,-1.524,0,0,.008\n"
            "GW 2,3,-1.524,0,0,-1.041,0,0,.010\n"
            "GW 3,9,-1.041,0,0,1.041,0,0,.010\n"
            "GW 4,3,1.041,0,0,1.524,0,0,.010\n"
            "GW 5,5,1.524,0,0,2.603,0,0,.008\n"
            "GW 6,5,-2.411,.856,0,-1.524,.856,0,.008\n"
            "GW 7,3,-1.524,.856,0,-1.041,.856,0,.010\n"
            "GW 8,9,-1.041,.856,0,1.041,.856,0,.010\n"
            "GW 9,3,1.041,.856,0,1.524,.856,0,.010\n"
            "GW 10,5,1.524,.856,0,2.411,.856,0,.008\n"
            "GE 0\nEX 0,3,5,0,1,0\nFR 0,1,0,0,29.\nGN -1\nEN"
        )
        result = classify(parse_text(nec))
        assert result.antenna_type == "yagi"
        assert result.element_count == 2

    def test_half_square_element_count(self):
        """Half-square Yagi: 6 GW wires (3 per U-shaped element) should merge to 2 elements."""
        nec = (
            "CM half square, Yagi 2M\nCE\n"
            "GW 1,11,-.508,0.,9.659619,-.508,0.,9.144,.0015875\n"
            "GW 2,22,-.508,0.,9.144,.508,0.,9.144,.009525\n"
            "GW 3,11,.508,0.,9.144,.508,0.,9.659619,.0015875\n"
            "GW 4,11,-.508,-.3048,9.69772,-.508,-.3048,9.144,.0015875\n"
            "GW 5,22,-.508,-.3048,9.144,.508,-.3048,9.144,.009525\n"
            "GW 6,11,.508,-.3048,9.144,.508,-.3048,9.69772,.0015875\n"
            "GE 1\nFR 0,1,0,0,146.\nGN 2,0,0,0,13.,.005\n"
            "EX 0,1,11,0,.707107,0.\nEX 0,2,1,0,.707107,0.\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 2

    def test_phased_array_multi_ex(self):
        result = classify(parse_text(_phased_nec()))
        assert result.antenna_type == "phased_array"
        assert result.confidence >= 0.5

    def test_lpda_requires_multi_link_feed_network(self):
        result = classify(parse_text(_hybrid_log_orr_single_tl_nec()))
        assert result.antenna_type != "lpda"
        assert not any("progressive lengths" in evidence for evidence in result.evidence)


# ---------------------------------------------------------------------------
# Classification result fields
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_frequency_extracted(self):
        result = classify(parse_text(_dipole_nec()))
        assert result.frequency_mhz == pytest.approx(14.15)

    def test_band_assigned(self):
        result = classify(parse_text(_dipole_nec()))
        assert result.band == "20m"

    def test_element_count(self):
        result = classify(parse_text(_yagi_nec(5)))
        assert result.element_count == 5

    def test_ground_type_free_space(self):
        result = classify(parse_text(_dipole_nec()))
        assert result.ground_type == "free_space"

    def test_ground_type_perfect(self):
        result = classify(parse_text(_vertical_nec()))
        assert result.ground_type == "perfect_ground"

    def test_evidence_populated(self):
        result = classify(parse_text(_helix_nec()))
        assert len(result.evidence) > 0

    def test_label_format(self):
        result = classify(parse_text(_dipole_nec()))
        assert isinstance(result.label, str)
        assert len(result.label) > 0

    def test_all_types_in_constant(self):
        assert "unknown" in ANTENNA_TYPES
        assert "yagi" in ANTENNA_TYPES
        assert "dipole" in ANTENNA_TYPES


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_file(self):
        result = classify(parse_text(""))
        assert result.antenna_type == "unknown"
        assert result.confidence == 0.0

    def test_comments_only(self):
        result = classify(parse_text("CM Just a comment\nEN"))
        assert result.antenna_type == "unknown"

    def test_exclude_keywords(self):
        """'loop' keyword should not match when 'log' is also present (LPDA exclude)."""
        nec = (
            "CM log periodic antenna\n"
            "GW 1 21 0 -5 0 0 5 0 0.001\n"
            "GE\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.0\nEN\n"
        )
        result = classify(parse_text(nec))
        assert result.antenna_type == "lpda"

    def test_inverted_v_exclude(self):
        """'inverted v' should match inverted_v, not dipole."""
        nec = (
            "CM inverted v\n"
            "GW 1 21 0 -5 0 0 5 0 0.001\n"
            "GE\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.0\nEN\n"
        )
        result = classify(parse_text(nec))
        assert result.antenna_type == "inverted_v"


# ---------------------------------------------------------------------------
# Frequency-to-band mapping
# ---------------------------------------------------------------------------

class TestFreqToBand:
    @pytest.mark.parametrize("freq,band", [
        (1.9, "160m"),
        (3.6, "80m"),
        (7.1, "40m"),
        (14.15, "20m"),
        (21.2, "15m"),
        (28.5, "10m"),
        (50.1, "6m"),
        (145.0, "2m"),
        (432.0, "70cm"),
    ])
    def test_ham_bands(self, freq, band):
        nec = (
            "GW 1 21 0 -5 0 0 5 0 0.001\n"
            "GE\nEX 0 1 11 0 1 0\n"
            f"FR 0 1 0 0 {freq}\n"
            "EN\n"
        )
        result = classify(parse_text(nec))
        assert result.band == band


# ---------------------------------------------------------------------------
# Wire-group merging regression tests
# ---------------------------------------------------------------------------

class TestWireGroupMerging:
    """Regression: _merge_connected_wire_groups must merge wires sharing
    endpoints into single elements, regardless of collinearity.
    Fixed in commit 9f2eac1."""

    def test_stepped_diameter_merges(self):
        """5 collinear GW wires sharing endpoints → 1 element."""
        nec = (
            "CM stepped-dia element\nCE\n"
            "GW 1,5,-2.603,0,0,-1.524,0,0,.008\n"
            "GW 2,3,-1.524,0,0,-1.041,0,0,.010\n"
            "GW 3,9,-1.041,0,0,1.041,0,0,.010\n"
            "GW 4,3,1.041,0,0,1.524,0,0,.010\n"
            "GW 5,5,1.524,0,0,2.603,0,0,.008\n"
            "GE 0\nEX 0,3,5,0,1,0\nFR 0,1,0,0,29.\nGN -1\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 1

    def test_u_shaped_half_square_merges(self):
        """3 GW wires forming a U shape (2 vertical + 1 horizontal) → 1 element."""
        nec = (
            "CM U-shape element\nCE\n"
            "GW 1,11,0,0,10,0,0,9,.001\n"
            "GW 2,22,0,0,9,1,0,9,.009\n"
            "GW 3,11,1,0,9,1,0,10,.001\n"
            "GE 1\nEX 0,1,11,0,1,0\nFR 0,1,0,0,146.\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 1

    def test_disconnected_elements_stay_separate(self):
        """Two wires with no shared endpoints → 2 elements."""
        nec = (
            "CM two disconnected wires\n"
            "GW 1 21 0 -5 0 0 5 0 0.001\n"
            "GW 2 21 3 -5 0 3 5 0 0.001\n"
            "GE 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.15\nEN\n"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 2

    def test_right_angle_wires_merge(self):
        """Two wires meeting at a right angle → 1 element."""
        nec = (
            "CM L-bend\nCE\n"
            "GW 1,11,0,0,0,0,0,5,.001\n"
            "GW 2,11,0,0,5,5,0,5,.001\n"
            "GE 0\nEX 0,1,1,0,1,0\nFR 0,1,0,0,14.\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 1

    def test_chain_of_three_wires_merges(self):
        """Three wires in a chain (A-B, B-C, C-D) → 1 element."""
        nec = (
            "CM chain\nCE\n"
            "GW 1,11,0,0,0,1,0,0,.001\n"
            "GW 2,11,1,0,0,2,0,0,.001\n"
            "GW 3,11,2,0,0,3,0,0,.001\n"
            "GE 0\nEX 0,1,6,0,1,0\nFR 0,1,0,0,14.\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 1

    def test_two_separate_chains_stay_separate(self):
        """Two disconnected chains → 2 elements."""
        nec = (
            "CM two chains\nCE\n"
            "GW 1,11,0,0,0,1,0,0,.001\n"
            "GW 2,11,1,0,0,2,0,0,.001\n"
            "GW 3,11,10,0,0,11,0,0,.001\n"
            "GW 4,11,11,0,0,12,0,0,.001\n"
            "GE 0\nEX 0,1,6,0,1,0\nFR 0,1,0,0,14.\nEN"
        )
        result = classify(parse_text(nec))
        assert result.element_count == 2

    def test_single_wire_no_merge_needed(self):
        """Single wire element needs no merging."""
        result = classify(parse_text(_dipole_nec()))
        assert result.element_count == 1
