"""Unit tests for antenna_classifier.parser."""

import math
import pytest

from antenna_classifier.parser import (
    AWG_LOOKUP,
    CARD_SPECS,
    NECCard,
    ParseResult,
    _expand_scientific,
    _preprocess_awg,
    _safe_eval,
    _split_params,
    parse_text,
)


# ---------------------------------------------------------------------------
# _preprocess_awg
# ---------------------------------------------------------------------------

class TestPreprocessAWG:
    def test_bare_gauge(self):
        assert _preprocess_awg("#12") == str(AWG_LOOKUP["#12"])

    def test_gauge_per_inch(self):
        result = _preprocess_awg("#14/in")
        assert result == str(AWG_LOOKUP["#14"])

    def test_no_match(self):
        assert _preprocess_awg("0.001") == "0.001"

    def test_multiple_gauges(self):
        result = _preprocess_awg("#12+#14")
        assert str(AWG_LOOKUP["#12"]) in result
        assert str(AWG_LOOKUP["#14"]) in result


# ---------------------------------------------------------------------------
# _expand_scientific
# ---------------------------------------------------------------------------

class TestExpandScientific:
    def test_positive_exponent(self):
        result = _expand_scientific("1.5e3")
        assert float(result) == 1500.0

    def test_negative_exponent(self):
        result = _expand_scientific("3.0e-2")
        assert float(result) == pytest.approx(0.03)

    def test_integer_result(self):
        result = _expand_scientific("1e3")
        assert result == "1000"

    def test_no_match(self):
        assert _expand_scientific("hello") == "hello"

    def test_plain_number(self):
        assert _expand_scientific("42") == "42"


# ---------------------------------------------------------------------------
# _safe_eval
# ---------------------------------------------------------------------------

class TestSafeEval:
    def test_integer(self):
        assert _safe_eval("42", {}) == 42

    def test_float(self):
        assert _safe_eval("3.14", {}) == pytest.approx(3.14)

    def test_scientific_notation(self):
        assert _safe_eval("1.5e3", {}) == pytest.approx(1500.0)

    def test_negative_scientific(self):
        assert _safe_eval("2.5e-3", {}) == pytest.approx(0.0025)

    def test_addition(self):
        assert _safe_eval("1+2", {}) == 3

    def test_multiplication(self):
        assert _safe_eval("3*4", {}) == 12

    def test_division(self):
        assert _safe_eval("10/3", {}) == pytest.approx(10 / 3)

    def test_power(self):
        assert _safe_eval("2**3", {}) == 8

    def test_variable_substitution(self):
        assert _safe_eval("x", {"x": 5.0}) == 5.0

    def test_variable_case_insensitive(self):
        assert _safe_eval("Freq", {"freq": 14.0}) == 14.0

    def test_expression_with_vars(self):
        sym = {"a": 10.0, "b": 3.0}
        assert _safe_eval("a+b", sym) == pytest.approx(13.0)

    def test_pi_constant(self):
        assert _safe_eval("pi", {}) == pytest.approx(math.pi)

    def test_sqrt_function(self):
        assert _safe_eval("sqrt(4)", {}) == pytest.approx(2.0)

    def test_sin_degrees(self):
        assert _safe_eval("sin(90)", {}) == pytest.approx(1.0)

    def test_cos_degrees(self):
        assert _safe_eval("cos(0)", {}) == pytest.approx(1.0)

    def test_log_is_log10(self):
        assert _safe_eval("log(100)", {}) == pytest.approx(2.0)

    def test_ln_is_natural(self):
        assert _safe_eval("ln(1)", {}) == pytest.approx(0.0)

    def test_int_function(self):
        assert _safe_eval("int(3.7)", {}) == 3

    def test_abs_function(self):
        assert _safe_eval("abs(-5)", {}) == 5

    def test_exp_function(self):
        assert _safe_eval("exp(0)", {}) == pytest.approx(1.0)

    def test_percent_notation(self):
        assert _safe_eval("50%", {}) == 50

    def test_awg_in_expression(self):
        result = _safe_eval("#12", {})
        assert isinstance(result, float)
        assert result == pytest.approx(AWG_LOOKUP["#12"])

    def test_unresolvable_returns_string(self):
        result = _safe_eval("undefined_var", {})
        assert isinstance(result, str)

    def test_nested_expression(self):
        sym = {"h": 2.0}
        assert _safe_eval("sqrt(h*2)", sym) == pytest.approx(2.0)

    def test_unary_minus(self):
        assert _safe_eval("-5", {}) == -5


# ---------------------------------------------------------------------------
# _split_params
# ---------------------------------------------------------------------------

class TestSplitParams:
    def test_comma_separated(self):
        assert _split_params("1, 2, 3") == ["1", "2", "3"]

    def test_space_separated(self):
        assert _split_params("1 2 3") == ["1", "2", "3"]

    def test_mixed_separators(self):
        result = _split_params("1, 2 3")
        assert result == ["1", "2", "3"]

    def test_inline_comment_stripped(self):
        result = _split_params("1 2 3 'this is a comment")
        assert result == ["1", "2", "3"]

    def test_exclamation_comment(self):
        result = _split_params("1 2 3 !comment")
        assert result == ["1", "2", "3"]

    def test_empty(self):
        assert _split_params("") == []


# ---------------------------------------------------------------------------
# SY symbol resolution
# ---------------------------------------------------------------------------

class TestSYResolution:
    def test_single_assignment(self):
        nec = "SY N=12\nGW N 1 0 0 0 0 0 1 0.001\nGE\nEX 0 N 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert "n" in result.symbol_table
        assert result.symbol_table["n"] == 12

    def test_multiple_comma_assignments(self):
        nec = "SY A=1, B=2, C=3\nGW 1 1 0 0 0 0 0 1 0.001\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert result.symbol_table["a"] == 1
        assert result.symbol_table["b"] == 2
        assert result.symbol_table["c"] == 3

    def test_dependent_variables(self):
        nec = "SY L=10\nSY H=L/2\nGW 1 1 0 0 0 0 0 1 0.001\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert result.symbol_table["l"] == 10
        assert result.symbol_table["h"] == pytest.approx(5.0)

    def test_expression_with_math(self):
        nec = "SY R=sqrt(4)\nGW 1 1 0 0 0 0 0 1 0.001\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert result.symbol_table["r"] == pytest.approx(2.0)

    def test_inline_comment_stripped(self):
        nec = "SY X=5 'this is five\nGW 1 1 0 0 0 0 0 1 0.001\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert result.symbol_table["x"] == 5


# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------

class TestCardParsing:
    def test_gw_card(self):
        nec = "GW 1 21 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        gw = [c for c in result.cards if c.card_type == "GW"]
        assert len(gw) == 1
        lp = gw[0].labeled_params
        assert lp["tag"] == 1
        assert lp["segments"] == 21
        assert lp["z2"] == pytest.approx(10.0)
        assert lp["radius"] == pytest.approx(0.001)

    def test_ex_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        ex = [c for c in result.cards if c.card_type == "EX"]
        assert len(ex) == 1
        lp = ex[0].labeled_params
        assert lp["exType"] == 0
        assert lp["tag"] == 1
        assert lp["segment"] == 6
        assert lp["vReal"] == pytest.approx(1.0)

    def test_fr_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.15\nEN"
        result = parse_text(nec)
        fr = [c for c in result.cards if c.card_type == "FR"]
        assert len(fr) == 1
        assert fr[0].labeled_params["freq"] == pytest.approx(14.15)

    def test_gn_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE\nGN 1\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        gn = [c for c in result.cards if c.card_type == "GN"]
        assert len(gn) == 1
        assert gn[0].labeled_params["groundType"] == 1

    def test_ld_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE\nLD 5 1 1 1 58000000\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        ld = [c for c in result.cards if c.card_type == "LD"]
        assert len(ld) == 1
        assert ld[0].labeled_params["ldType"] == 5
        assert ld[0].labeled_params["tag"] == 1

    def test_tl_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGW 2 11 0 0 0 0 0 10.0 0.001\nGE\nTL 1 6 2 6 50.0\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        tl = [c for c in result.cards if c.card_type == "TL"]
        assert len(tl) == 1
        assert tl[0].labeled_params["z0"] == pytest.approx(50.0)

    def test_cm_comment(self):
        nec = "CM Simple dipole model\nGW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert "Simple dipole model" in result.comment_text

    def test_ce_comment(self):
        nec = "CM Line 1\nCE Line 2\nGW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        assert "Line 1" in result.comment_text
        assert "Line 2" in result.comment_text

    def test_en_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        en = [c for c in result.cards if c.card_type == "EN"]
        assert len(en) == 1

    def test_ge_card(self):
        nec = "GW 1 11 0 0 0 0 0 10.0 0.001\nGE 0\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        ge = [c for c in result.cards if c.card_type == "GE"]
        assert len(ge) == 1
        assert ge[0].labeled_params["ground"] == 0


# ---------------------------------------------------------------------------
# parse_text full integration
# ---------------------------------------------------------------------------

class TestParseText:
    def test_minimal_dipole(self):
        nec = (
            "CM Half-wave dipole\n"
            "GW 1 21 0 -5.0 0 0 5.0 0 0.001\n"
            "GE 0\n"
            "EX 0 1 11 0 1 0\n"
            "FR 0 1 0 0 14.15\n"
            "EN\n"
        )
        result = parse_text(nec)
        assert len(result.errors) == 0
        assert len(result.wire_cards) == 1
        assert result.comment_text == "Half-wave dipole"
        assert result.cards[-1].card_type == "EN"

    def test_sy_substitution_in_gw(self):
        nec = (
            "SY L=10.0, R=0.001\n"
            "GW 1 21 0 0 0 0 0 L R\n"
            "GE\n"
            "EX 0 1 11 0 1 0\n"
            "FR 0 1 0 0 14.0\n"
            "EN\n"
        )
        result = parse_text(nec)
        gw = result.wire_cards[0]
        assert gw.labeled_params["z2"] == pytest.approx(10.0)
        assert gw.labeled_params["radius"] == pytest.approx(0.001)

    def test_tab_separated_fields(self):
        nec = "GW\t1\t11\t0\t0\t0\t0\t0\t10.0\t0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        gw = result.wire_cards[0]
        assert gw.labeled_params["tag"] == 1

    def test_apostrophe_comment_lines(self):
        nec = "'This is a comment line\nGW 1 11 0 0 0 0 0 10.0 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        cm = [c for c in result.cards if c.card_type == "CM"]
        assert len(cm) == 1
        assert cm[0].text == "This is a comment line"

    def test_empty_lines_ignored(self):
        nec = "\n\nGW 1 11 0 0 0 0 0 10.0 0.001\n\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN\n\n"
        result = parse_text(nec)
        assert len(result.wire_cards) == 1

    def test_multiple_wires(self):
        nec = (
            "GW 1 11 0 0 0 0 0 10 0.001\n"
            "GW 2 11 5 0 0 5 0 10 0.001\n"
            "GW 3 11 -5 0 0 -5 0 10 0.001\n"
            "GE\n"
            "EX 0 1 6 0 1 0\n"
            "FR 0 1 0 0 14.0\n"
            "EN\n"
        )
        result = parse_text(nec)
        assert len(result.wire_cards) == 3
        tags = {c.labeled_params["tag"] for c in result.wire_cards}
        assert tags == {1, 2, 3}

    def test_geometry_cards_property(self):
        nec = (
            "GW 1 11 0 0 0 0 0 10 0.001\n"
            "GA 2 10 5.0 0 360 0.001\n"
            "GE\n"
            "EX 0 1 6 0 1 0\n"
            "FR 0 1 0 0 14.0\n"
            "EN\n"
        )
        result = parse_text(nec)
        geo = result.geometry_cards
        assert len(geo) == 2
        assert {c.card_type for c in geo} == {"GW", "GA"}

    def test_unknown_card_preserved(self):
        nec = "XX 1 2 3\nGW 1 11 0 0 0 0 0 10 0.001\nGE\nEX 0 1 6 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        xx = [c for c in result.cards if c.card_type == "XX"]
        assert len(xx) == 1
        assert xx[0].params == ["1", "2", "3"]

    def test_missing_required_params_error(self):
        nec = "GW 1\nGE\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14.0\nEN"
        result = parse_text(nec)
        gw = result.wire_cards[0]
        assert len(gw.errors) > 0
