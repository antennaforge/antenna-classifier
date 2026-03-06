"""Unit tests for antenna_classifier.validator."""

import pytest

from antenna_classifier.parser import parse_text
from antenna_classifier.validator import Severity, validate


def _make_nec(*extra_lines: str, comment: str = "", wires: str = "", ground: str = "") -> str:
    """Build a minimal NEC deck with optional overrides."""
    lines = []
    if comment:
        lines.append(f"CM {comment}")
    if wires:
        lines.append(wires)
    else:
        lines.append("GW 1 21 0 -5.0 0 0 5.0 0 0.001")
    lines.append("GE 0")
    if ground:
        lines.append(ground)
    for ln in extra_lines:
        lines.append(ln)
    lines.append("EX 0 1 11 0 1 0")
    lines.append("FR 0 1 0 0 14.15")
    lines.append("EN")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Valid deck baseline
# ---------------------------------------------------------------------------

class TestValidDeck:
    def test_minimal_dipole_is_valid(self):
        result = validate(parse_text(_make_nec()))
        assert result.valid is True

    def test_no_errors(self):
        result = validate(parse_text(_make_nec()))
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------

class TestGeometryValidation:
    def test_no_geometry_is_error(self):
        nec = "GE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("geometry" in m.lower() for m in msgs)

    def test_wire_count_info(self):
        result = validate(parse_text(_make_nec()))
        info = [i for i in result.issues if i.severity == Severity.INFO]
        assert any("wire" in i.message.lower() for i in info)


# ---------------------------------------------------------------------------
# Excitation checks
# ---------------------------------------------------------------------------

class TestExcitationValidation:
    def test_no_excitation_is_error(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("excitation" in m.lower() or "EX" in m for m in msgs)


# ---------------------------------------------------------------------------
# Frequency checks
# ---------------------------------------------------------------------------

class TestFrequencyValidation:
    def test_no_frequency_is_error(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nEX 0 1 6 0 1 0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("frequency" in m.lower() or "FR" in m for m in msgs)

    def test_zero_frequency_is_error(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid


# ---------------------------------------------------------------------------
# Ground checks
# ---------------------------------------------------------------------------

class TestGroundValidation:
    def test_no_ground_info_message(self):
        result = validate(parse_text(_make_nec()))
        info = [i for i in result.issues if i.severity == Severity.INFO]
        assert any("GN" in i.message or "free-space" in i.message for i in info)

    def test_with_ground_no_free_space_info(self):
        result = validate(parse_text(_make_nec(ground="GN 1")))
        info_msgs = [i.message for i in result.issues if i.severity == Severity.INFO]
        # Should not have the "no GN" info message
        assert not any("No GN" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# EN/GE card checks
# ---------------------------------------------------------------------------

class TestENGECards:
    def test_no_en_card_warning(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0"
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert any("EN" in m for m in warnings)

    def test_no_ge_card_warning(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert any("GE" in m for m in warnings)


# ---------------------------------------------------------------------------
# Card order checks
# ---------------------------------------------------------------------------

class TestCardOrder:
    def test_geometry_after_control_warns(self):
        nec = (
            "GW 1 11 0 0 0 0 0 10 0.001\n"
            "GE\n"
            "EX 0 1 6 0 1 0\n"
            "GW 2 11 5 0 0 5 0 10 0.001\n"  # Geometry after control
            "FR 0 1 0 0 14.0\n"
            "EN\n"
        )
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert any("after control" in m.lower() for m in warnings)


# ---------------------------------------------------------------------------
# Wire parameter checks
# ---------------------------------------------------------------------------

class TestWireParams:
    def test_zero_segment_error(self):
        nec = "GW 1 0 0 0 0 0 0 10 0.001\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("segment" in m.lower() for m in msgs)

    def test_negative_radius_error(self):
        nec = "GW 1 11 0 0 0 0 0 10 -0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("radius" in m.lower() for m in msgs)

    def test_zero_length_wire_error(self):
        nec = "GW 1 11 0 0 0 0 0 0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        assert not result.valid
        msgs = [i.message for i in result.errors]
        assert any("zero-length" in m.lower() for m in msgs)


# ---------------------------------------------------------------------------
# Tag reference checks
# ---------------------------------------------------------------------------

class TestTagReferences:
    def test_ex_undefined_tag_warning(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nEX 0 99 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert any("99" in m and "undefined" in m.lower() for m in warnings)

    def test_ld_undefined_tag_warning(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nLD 5 99 1 1 58000000\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert any("99" in m and "undefined" in m.lower() for m in warnings)

    def test_tag_0_no_warning(self):
        """Tag 0 means 'all wires' — should not warn."""
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE\nLD 5 0 1 1 58000000\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = validate(parse_text(nec))
        warnings = [i.message for i in result.warnings]
        assert not any("undefined" in m.lower() for m in warnings)

    def test_valid_tag_no_warning(self):
        result = validate(parse_text(_make_nec()))
        warnings = [i.message for i in result.warnings]
        assert not any("undefined" in m.lower() for m in warnings)
