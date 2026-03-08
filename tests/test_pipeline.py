"""Tests for the structured NEC generation pipeline.

Tests each step independently with mocked LLM responses,
plus integration tests for the validation and conversion steps
(which are deterministic and don't need mocks).
"""

import json
import math
import pytest
from unittest.mock import MagicMock, patch

from antenna_classifier.nec_pipeline import (
    ANTENNA_TYPES,
    ExtractedConcepts,
    PipelineResult,
    StepLog,
    classify_document,
    extract_concepts,
    generate_deck,
    validate_deck,
    convert_to_nec,
    diagnose_failure,
    _get_extraction_spec,
    _parse_llm_json,
    _build_extraction_prompt,
    _format_concepts_for_generation,
    _stamp_moxon_deck,
    _extract_tl_impedance,
    _EXTRACTION_SPECS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(content: str, prompt_tokens: int = 100,
                   completion_tokens: int = 50):
    """Create a mock OpenAI API response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _mock_client(content: str):
    """Create a mock OpenAI client that returns the given content."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(content)
    return client


def _make_deck(cards: list[dict]) -> dict:
    """Build a minimal JSON deck."""
    return {"cards": cards}


def _simple_dipole_deck(freq_mhz: float = 14.175,
                        half_len: float = 5.05,
                        height: float = 10.0,
                        radius: float = 0.001) -> dict:
    """A valid dipole JSON deck."""
    return _make_deck([
        {"type": "CM", "text": f"Dipole at {freq_mhz} MHz"},
        {"type": "CE"},
        {"type": "GW", "params": [1, 21, 0.0, -half_len, height,
                                   0.0, half_len, height, radius]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 91, 1, 1000, 0.0, 0.0, 1.0, 0.0]},
        {"type": "EN"},
    ])


def _simple_yagi_deck(freq_mhz: float = 14.175) -> dict:
    """A valid 3-element Yagi JSON deck."""
    return _make_deck([
        {"type": "CM", "text": f"3-el Yagi at {freq_mhz} MHz"},
        {"type": "CE"},
        {"type": "GW", "params": [1, 21, 0.0, -5.41, 10.0,
                                   0.0, 5.41, 10.0, 0.0127]},
        {"type": "GW", "params": [2, 21, 1.84, -4.95, 10.0,
                                   1.84, 4.95, 10.0, 0.0127]},
        {"type": "GW", "params": [3, 21, 4.60, -4.72, 10.0,
                                   4.60, 4.72, 10.0, 0.0127]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 91, 1, 1000, 0.0, 0.0, 1.0, 0.0]},
        {"type": "EN"},
    ])


# ===================================================================
# Test _parse_llm_json
# ===================================================================

class TestParseLLMJson:
    def test_plain_json(self):
        result = _parse_llm_json('{"antenna_type": "yagi", "confidence": 0.9}')
        assert result["antenna_type"] == "yagi"
        assert result["confidence"] == 0.9

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"antenna_type": "dipole"}\n```'
        result = _parse_llm_json(raw)
        assert result["antenna_type"] == "dipole"

    def test_json_with_trailing_chat(self):
        raw = '{"type": "yagi"} Here is the explanation...'
        result = _parse_llm_json(raw)
        assert result["type"] == "yagi"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_llm_json("No JSON here at all")

    def test_unbalanced_braces_raises(self):
        with pytest.raises(ValueError, match="[Uu]nbalanced"):
            _parse_llm_json('{"broken": ')


# ===================================================================
# Test extraction specs
# ===================================================================

class TestExtractionSpecs:
    def test_all_common_types_have_specs(self):
        common = ["yagi", "moxon", "dipole", "vertical", "quad", "lpda",
                   "inverted_v", "end_fed", "j_pole", "loop"]
        for t in common:
            spec = _get_extraction_spec(t)
            assert "params" in spec
            assert len(spec["params"]) >= 2
            assert "desc" in spec

    def test_unknown_type_gets_generic(self):
        spec = _get_extraction_spec("unknown_type_xyz")
        assert "params" in spec
        assert len(spec["params"]) >= 3  # generic has several params

    def test_param_tuples_have_three_elements(self):
        for atype, spec in _EXTRACTION_SPECS.items():
            for param in spec["params"]:
                assert len(param) == 3, \
                    f"{atype}.{param[0]}: expected (name, desc, unit)"

    def test_yagi_has_key_params(self):
        spec = _get_extraction_spec("yagi")
        names = {p[0] for p in spec["params"]}
        assert "reflector_length" in names
        assert "driven_length" in names
        assert "director_lengths" in names
        assert "element_spacings" in names

    def test_moxon_has_key_params(self):
        spec = _get_extraction_spec("moxon")
        names = {p[0] for p in spec["params"]}
        assert "driven_width" in names
        assert "gap" in names


# ===================================================================
# Test _build_extraction_prompt
# ===================================================================

class TestBuildExtractionPrompt:
    def test_yagi_prompt_contains_params(self):
        prompt = _build_extraction_prompt("yagi")
        assert "reflector_length" in prompt
        assert "driven_length" in prompt
        assert "director_lengths" in prompt
        assert "Yagi-Uda" in prompt

    def test_prompt_contains_goals(self):
        prompt = _build_extraction_prompt("dipole")
        assert "gain_dbi" in prompt
        assert "fb_db" in prompt
        assert "max_swr" in prompt

    def test_prompt_contains_unit_instructions(self):
        prompt = _build_extraction_prompt("vertical")
        assert "metres" in prompt
        assert "millimetres" in prompt


# ===================================================================
# Test _format_concepts_for_generation
# ===================================================================

class TestFormatConcepts:
    def test_basic_formatting(self):
        concepts = ExtractedConcepts(
            antenna_type="yagi",
            freq_mhz=14.175,
            elements={"reflector_length": 10.82, "driven_length": 9.90},
            wire_dia_mm=25.4,
            height_m=10.0,
        )
        text = _format_concepts_for_generation(concepts)
        assert "ANTENNA TYPE: yagi" in text
        assert "14.175 MHz" in text
        assert "reflector_length" in text
        assert "10.82" in text
        assert "25.4 mm" in text

    def test_goals_included(self):
        concepts = ExtractedConcepts(
            antenna_type="yagi",
            freq_mhz=14.175,
            gain_dbi=7.1,
            fb_db=21.0,
            max_swr=1.5,
        )
        text = _format_concepts_for_generation(concepts)
        assert "7.1 dBi" in text
        assert "21 dB" in text
        assert "1.5:1" in text

    def test_calc_summary_included(self):
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        text = _format_concepts_for_generation(
            concepts, calc_summary="half_length=5.05m"
        )
        assert "CALCULATOR REFERENCE" in text
        assert "half_length=5.05m" in text

    def test_description_included(self):
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
            description="Use copper wire #14",
        )
        text = _format_concepts_for_generation(concepts)
        assert "copper wire #14" in text


# ===================================================================
# Test classify_document (mocked LLM)
# ===================================================================

class TestClassifyDocument:
    def test_successful_classification(self):
        client = _mock_client(
            '{"antenna_type": "yagi", "confidence": 0.95, '
            '"evidence": "mentions reflector and directors"}'
        )
        atype, conf, evidence = classify_document(
            "This is a 3-element Yagi beam antenna for 20 meters",
            client=client,
        )
        assert atype == "yagi"
        assert conf == 0.95
        assert "reflector" in evidence

    def test_unknown_type_normalised(self):
        client = _mock_client(
            '{"antenna_type": "super_antenna", "confidence": 0.8, "evidence": ""}'
        )
        atype, conf, _ = classify_document("test", client=client)
        assert atype == "unknown"
        assert conf == 0.0

    def test_parse_failure_returns_unknown(self):
        client = _mock_client("I think it's a dipole antenna")
        atype, conf, _ = classify_document("test", client=client)
        assert atype == "unknown"


# ===================================================================
# Test extract_concepts (mocked LLM)
# ===================================================================

class TestExtractConcepts:
    def test_extracts_yagi_params(self):
        llm_response = json.dumps({
            "freq_mhz": 14.175,
            "reflector_length": 10.82,
            "driven_length": 9.90,
            "director_lengths": [9.44],
            "element_spacings": [1.84, 2.76],
            "n_elements": 3,
            "wire_diameter": 25.4,
            "height": 10.0,
            "gain_dbi": 7.1,
            "fb_db": 21.0,
            "max_swr": 1.5,
            "bands": ["20m"],
            "ground_type": "free_space",
        })
        client = _mock_client(llm_response)

        concepts, resp = extract_concepts(
            "3-element Yagi for 20 meters, reflector 10.82m...",
            "yagi",
            client=client,
        )
        assert concepts.antenna_type == "yagi"
        assert concepts.freq_mhz == 14.175
        assert concepts.elements["reflector_length"] == 10.82
        assert concepts.elements["driven_length"] == 9.90
        assert concepts.wire_dia_mm == 25.4
        assert concepts.height_m == 10.0

    def test_null_values_handled(self):
        llm_response = json.dumps({
            "freq_mhz": 28.4,
            "reflector_length": 5.41,
            "driven_length": None,
            "director_lengths": None,
            "wire_diameter": None,
            "height": None,
        })
        client = _mock_client(llm_response)

        concepts, _ = extract_concepts(
            "10m Yagi", "yagi", client=client,
        )
        assert "reflector_length" in concepts.elements
        assert "driven_length" not in concepts.elements
        assert concepts.wire_dia_mm is None

    def test_freq_override(self):
        client = _mock_client('{"freq_mhz": 14.175}')
        concepts, _ = extract_concepts(
            "some text", "dipole", client=client, freq_mhz=28.4,
        )
        assert concepts.freq_mhz == 28.4  # override wins


# ===================================================================
# Test validate_deck (deterministic, no mocks needed)
# ===================================================================

class TestValidateDeck:
    def test_valid_dipole_passes(self):
        deck = _simple_dipole_deck()
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        issues = validate_deck(deck, concepts)
        # Should have no critical issues
        critical = [i for i in issues if "MISSING" in i or "COLLAPSED" in i]
        assert len(critical) == 0

    def test_valid_yagi_passes(self):
        deck = _simple_yagi_deck()
        concepts = ExtractedConcepts(
            antenna_type="yagi", freq_mhz=14.175,
        )
        issues = validate_deck(deck, concepts)
        critical = [i for i in issues if "MISSING" in i or "COLLAPSED" in i]
        assert len(critical) == 0

    def test_missing_cards_detected(self):
        deck = _make_deck([
            {"type": "CM", "text": "test"},
            {"type": "CE"},
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            # Missing GE, EX, FR, EN
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("MISSING" in i for i in issues)

    def test_zero_length_wire_detected(self):
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, 0, 10, 0, 0, 10, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("zero-length" in i for i in issues)

    def test_bad_radius_detected(self):
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, -0.001]},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("radius" in i for i in issues)

    def test_collapsed_geometry_detected(self):
        # All wires start at same point
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, 0, 0, 0, 5, 0, 0.001]},
            {"type": "GW", "params": [2, 21, 0, 0, 0, 0, 4, 0, 0.001]},
            {"type": "GW", "params": [3, 21, 0, 0, 0, 0, 3, 0, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="yagi", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("COLLAPSED" in i for i in issues)

    def test_collapsed_geometry_exempt_for_vertical(self):
        # Same geometry (all wires share start point) but as a vertical
        # with radials — this is legitimate and should NOT be flagged.
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, 0, 0, 0, 0, 5.0, 0.001]},
            {"type": "GW", "params": [2, 11, 0, 0, 0, 2.5, 0, 0, 0.001]},
            {"type": "GW", "params": [3, 11, 0, 0, 0, -2.5, 0, 0, 0.001]},
            {"type": "GW", "params": [4, 11, 0, 0, 0, 0, 2.5, 0, 0.001]},
            {"type": "GE", "params": [1]},
            {"type": "EX", "params": [0, 1, 1, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="vertical", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert not any("COLLAPSED" in i for i in issues)

    def test_collapsed_geometry_exempt_for_ground_plane(self):
        # ground_plane type is also in the exemption set
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, 0, 0, 0, 0, 5.0, 0.001]},
            {"type": "GW", "params": [2, 11, 0, 0, 0, 2.5, 0, 0, 0.001]},
            {"type": "GE", "params": [1]},
            {"type": "EX", "params": [0, 1, 1, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="ground_plane", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert not any("COLLAPSED" in i for i in issues)

    def test_frequency_mismatch_detected(self):
        deck = _simple_dipole_deck(freq_mhz=7.1)  # FR says 7.1
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,  # but we expect 14.175
        )
        issues = validate_deck(deck, concepts)
        assert any("FR FREQUENCY" in i for i in issues)

    def test_wrong_param_count_detected(self):
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10]},  # only 5, need 9
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("GW card has 5 params" in i for i in issues)

    def test_very_long_wire_flagged(self):
        # 100m wire at 14.175 MHz (λ ≈ 21m) = ~4.7λ
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -50, 10, 0, 50, 10, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("very long" in i.lower() or "λ" in i for i in issues)

    def test_calculator_crosscheck_flags_oversized_dipole(self):
        # Dipole with half-length 15m at 14.175 MHz — total 30m vs expected ~10m
        deck = _simple_dipole_deck(freq_mhz=14.175, half_len=15.0)
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        issues = validate_deck(deck, concepts)
        assert any("DIMENSION" in i for i in issues)

    def test_calculator_crosscheck_accepts_reasonable_dipole(self):
        # Dipole with half-length 5.05m at 14.175 MHz — close to λ/2
        deck = _simple_dipole_deck(freq_mhz=14.175, half_len=5.05)
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        issues = validate_deck(deck, concepts)
        # Should not have dimension check failures
        dim_issues = [i for i in issues if "DIMENSION" in i]
        assert len(dim_issues) == 0

    def test_no_gw_cards(self):
        deck = _make_deck([
            {"type": "CM", "text": "empty"},
            {"type": "CE"},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 14.175, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="dipole", freq_mhz=14.175)
        issues = validate_deck(deck, concepts)
        assert any("NO GW" in i for i in issues)

    def test_tl_at_wire_endpoint_flagged(self):
        """TL card connecting at segment 21 (end) of a 21-segment wire."""
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "GW", "params": [2, 21, 0, -5, 12, 0, 5, 12, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "TL", "params": [1, 21, 2, 21, 250, 0, 0, 0, 0, 0]},
            {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 28.4, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="yagi", freq_mhz=28.4)
        issues = validate_deck(deck, concepts)
        tl_issues = [i for i in issues if "TL" in i and "END" in i]
        assert len(tl_issues) == 2  # both ports at endpoints

    def test_tl_at_wire_centre_accepted(self):
        """TL card connecting at centre segments — no TL issues."""
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "GW", "params": [2, 21, 0, -5, 12, 0, 5, 12, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "TL", "params": [1, 11, 2, 11, 250, 0, 0, 0, 0, 0]},
            {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 28.4, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="yagi", freq_mhz=28.4)
        issues = validate_deck(deck, concepts)
        tl_issues = [i for i in issues if "TL" in i and "END" in i]
        assert len(tl_issues) == 0

    def test_tl_references_nonexistent_wire(self):
        """TL referencing a wire tag that has no GW card."""
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "TL", "params": [1, 11, 99, 11, 250, 0, 0, 0, 0, 0]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 28.4, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(antenna_type="yagi", freq_mhz=28.4)
        issues = validate_deck(deck, concepts)
        assert any("no GW card" in i for i in issues)

    def test_missing_tl_card_flagged_when_concepts_require_it(self):
        """Deck without TL card when concepts.transmission_line is set."""
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "GW", "params": [2, 21, 0, -5, 12, 0, 5, 12, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 28.4, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(
            antenna_type="yagi", freq_mhz=28.4,
            transmission_line={"z0": 250.0},
        )
        issues = validate_deck(deck, concepts)
        assert any("MISSING TL CARD" in i for i in issues)

    def test_tl_card_present_no_missing_flag(self):
        """Deck with TL card when concepts.transmission_line is set — no flag."""
        deck = _make_deck([
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "GW", "params": [2, 21, 0, -5, 12, 0, 5, 12, 0.001]},
            {"type": "GE", "params": [0]},
            {"type": "TL", "params": [1, 11, 2, 11, 250, 0, 0, 0, 0, 0]},
            {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
            {"type": "FR", "params": [0, 1, 0, 0, 28.4, 0]},
            {"type": "EN"},
        ])
        concepts = ExtractedConcepts(
            antenna_type="yagi", freq_mhz=28.4,
            transmission_line={"z0": 250.0},
        )
        issues = validate_deck(deck, concepts)
        assert not any("MISSING TL CARD" in i for i in issues)


# ===================================================================
# Test _extract_tl_impedance
# ===================================================================

class TestExtractTlImpedance:
    def test_ohm_symbol_pattern(self):
        text = "The antenna uses a 250 Ω characteristic impedance phase line."
        assert _extract_tl_impedance(text) == 250.0

    def test_ohm_word_pattern(self):
        text = "We connect the two driven elements with a 300-ohm transmission line."
        assert _extract_tl_impedance(text) == 300.0

    def test_z0_equals_pattern(self):
        text = "The phase line has Z0 = 450 connecting the feeds."
        assert _extract_tl_impedance(text) == 450.0

    def test_no_tl_in_text(self):
        text = "This is a standard 3-element Yagi with no phase line at 28 MHz."
        assert _extract_tl_impedance(text) is None

    def test_impedance_outside_range_rejected(self):
        text = "The 10 Ω characteristic wire runs along the boom."
        assert _extract_tl_impedance(text) is None

    def test_cebik_style_hyphenated_text(self):
        text = "a 250 Ω characteris- tic impedance phase line"
        assert _extract_tl_impedance(text) == 250.0


# ===================================================================
# Test convert_to_nec
# ===================================================================

class TestConvertToNec:
    def test_dipole_conversion(self):
        deck = _simple_dipole_deck()
        nec = convert_to_nec(deck)
        assert "CM" in nec
        assert "GW" in nec
        assert "FR" in nec
        assert "EN" in nec
        assert "14.175" in nec

    def test_yagi_conversion(self):
        deck = _simple_yagi_deck()
        nec = convert_to_nec(deck)
        lines = nec.strip().splitlines()
        gw_lines = [l for l in lines if l.startswith("GW")]
        assert len(gw_lines) == 3

    def test_roundtrip_cards(self):
        deck = _simple_dipole_deck()
        nec = convert_to_nec(deck)
        # Count non-empty lines
        lines = [l for l in nec.strip().splitlines() if l.strip()]
        # CM, CE, GW, GE, EX, FR, RP, EN = 8 lines
        assert len(lines) == 8


# ===================================================================
# Test diagnose_failure
# ===================================================================

class TestDiagnoseFailure:
    def test_no_issues_returns_zero(self):
        eval_result = {"sim_ok": True, "goal_verdict": {"passed": True}}
        step, fb = diagnose_failure(eval_result, [], ExtractedConcepts())
        assert step == 0

    def test_dimension_only_routes_to_step_2(self):
        eval_result = {"sim_ok": True, "goal_verdict": {"passed": True}}
        issues = ["DIMENSION CHECK: longest element 20m vs calculator 10m"]
        step, fb = diagnose_failure(eval_result, issues, ExtractedConcepts())
        assert step == 2
        assert "DIMENSION" in fb

    def test_structural_routes_to_step_3(self):
        eval_result = {
            "sim_ok": False,
            "sim_result": {"error": "Geometry error"},
            "goal_verdict": None,
        }
        issues = ["COLLAPSED GEOMETRY: all wires share same start"]
        step, fb = diagnose_failure(eval_result, issues, ExtractedConcepts())
        assert step == 3

    def test_goal_failure_routes_to_step_3(self):
        eval_result = {
            "sim_ok": True,
            "sim_result": {},
            "goal_verdict": {
                "passed": False,
                "checks_passed": 2,
                "checks_total": 5,
                "score": 0.4,
                "feedback": ["SWR too high: 3.5 vs target 2.0"],
            },
        }
        step, fb = diagnose_failure(eval_result, [], ExtractedConcepts())
        assert step == 3
        assert "SWR" in fb


# ===================================================================
# Test generate_deck (mocked LLM)
# ===================================================================

class TestGenerateDeck:
    def test_generates_deck(self):
        deck_json = json.dumps(_simple_dipole_deck())
        client = _mock_client(deck_json)

        concepts = ExtractedConcepts(
            antenna_type="dipole",
            freq_mhz=14.175,
        )
        deck, resp = generate_deck(concepts, client=client)
        assert "cards" in deck
        assert len(deck["cards"]) == 8

    def test_feedback_included_in_prompt(self):
        deck_json = json.dumps(_simple_dipole_deck())
        client = _mock_client(deck_json)

        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        deck, resp = generate_deck(
            concepts, client=client,
            feedback="SWR too high, shorten driven element",
        )
        # Verify feedback was passed to the LLM
        call_args = client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][-1]["content"]
        assert "FEEDBACK" in user_msg
        assert "SWR too high" in user_msg

    def test_bad_json_raises(self):
        client = _mock_client("This is not JSON at all")
        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        with pytest.raises(ValueError):
            generate_deck(concepts, client=client)

    def test_history_passed_to_llm(self):
        """When history is provided, messages include prior assistant/user turns."""
        deck_json = json.dumps(_simple_dipole_deck())
        client = _mock_client(deck_json)

        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        history = [
            {"role": "assistant", "content": '{"cards": []}'},
            {"role": "user", "content": "SWR was 5.2:1 — shorten driven element"},
        ]
        deck, resp = generate_deck(
            concepts, client=client, history=history,
        )
        call_args = client.chat.completions.create.call_args
        msgs = call_args.kwargs["messages"]
        # Should be system + user + 2 history entries = 4 messages
        assert len(msgs) == 4
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert "SWR was 5.2" in msgs[3]["content"]

    def test_history_supersedes_feedback(self):
        """When both history and feedback are provided, history is used."""
        deck_json = json.dumps(_simple_dipole_deck())
        client = _mock_client(deck_json)

        concepts = ExtractedConcepts(
            antenna_type="dipole", freq_mhz=14.175,
        )
        history = [
            {"role": "assistant", "content": '{"cards": []}'},
            {"role": "user", "content": "gain too low"},
        ]
        deck, resp = generate_deck(
            concepts, client=client,
            feedback="this should not appear",
            history=history,
        )
        call_args = client.chat.completions.create.call_args
        msgs = call_args.kwargs["messages"]
        # history takes precedence — feedback string should NOT be appended
        user_content = msgs[1]["content"]
        assert "this should not appear" not in user_content
        assert msgs[2]["role"] == "assistant"


# ===================================================================
# Test data structures
# ===================================================================

class TestDataStructures:
    def test_extracted_concepts_to_dict(self):
        c = ExtractedConcepts(
            antenna_type="yagi", freq_mhz=14.175,
            gain_dbi=7.1, fb_db=21.0, max_swr=1.5,
            elements={"reflector_length": 10.82},
            wire_dia_mm=25.4, height_m=10.0,
        )
        d = c.to_dict()
        assert d["antenna_type"] == "yagi"
        assert d["freq_mhz"] == 14.175
        assert d["gain_dbi"] == 7.1
        assert d["elements"]["reflector_length"] == 10.82

    def test_pipeline_result_to_dict(self):
        r = PipelineResult(
            nec_content="CM test\nEN\n",
            model="gpt-5.2",
            iterations=1,
        )
        r.steps.append(StepLog(step=1, name="classify", status="ok"))
        d = r.to_dict()
        assert d["nec_content"] == "CM test\nEN\n"
        assert d["model"] == "gpt-5.2"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["name"] == "classify"

    def test_step_log_fields(self):
        s = StepLog(step=3, name="generate", status="ok",
                    detail="8 cards", data={"cards": 8})
        assert s.step == 3
        assert s.name == "generate"
        assert s.data["cards"] == 8


# ===================================================================
# Test extraction spec coverage
# ===================================================================

class TestExtractionSpecCoverage:
    """Verify extraction specs exist for common antenna types."""

    def test_specs_cover_calculators(self):
        """Every type with a calculator should have an extraction spec."""
        # Types known to have calculators (from exploration)
        calculator_types = [
            "dipole", "inverted_v", "vertical", "j_pole", "end_fed",
            "yagi", "moxon", "quad", "lpda", "magnetic_loop",
            "delta_loop", "collinear", "discone", "hexbeam",
            "phased_array", "loop", "bobtail_curtain", "helix",
        ]
        for t in calculator_types:
            assert t in _EXTRACTION_SPECS, \
                f"Type '{t}' has a calculator but no extraction spec"

    def test_all_spec_types_are_canonical(self):
        """All types in extraction specs should be valid antenna types."""
        for t in _EXTRACTION_SPECS:
            assert t in ANTENNA_TYPES, \
                f"Extraction spec '{t}' is not a canonical antenna type"


# ===================================================================
# Moxon stamper tests
# ===================================================================

class TestMoxonStamper:
    """Tests for the deterministic Moxon NEC deck stamper."""

    def test_stamp_produces_valid_deck(self):
        """Stamped deck has required NEC cards."""
        deck = _stamp_moxon_deck(14.175)
        card_types = [c["type"] for c in deck["cards"]]
        for req in ("GW", "GE", "EX", "FR", "RP", "EN", "LD"):
            assert req in card_types, f"Missing {req} card"

    def test_stamp_has_six_gw_cards(self):
        """Moxon needs exactly 6 GW cards (3 driven + 3 reflector)."""
        deck = _stamp_moxon_deck(14.175)
        gw_cards = [c for c in deck["cards"] if c["type"] == "GW"]
        assert len(gw_cards) == 6

    def test_stamp_tags_sequential(self):
        """GW tags should be 1-6."""
        deck = _stamp_moxon_deck(14.175)
        gw_cards = [c for c in deck["cards"] if c["type"] == "GW"]
        tags = [int(c["params"][0]) for c in gw_cards]
        assert tags == [1, 2, 3, 4, 5, 6]

    def test_stamp_frequency_in_fr_card(self):
        """FR card should carry the requested frequency."""
        for freq in [7.15, 14.175, 28.4]:
            deck = _stamp_moxon_deck(freq)
            fr = [c for c in deck["cards"] if c["type"] == "FR"][0]
            assert fr["params"][4] == freq

    def test_stamp_feed_at_driven_main(self):
        """EX card should excite GW 2 at the centre segment."""
        deck = _stamp_moxon_deck(14.175)
        ex = [c for c in deck["cards"] if c["type"] == "EX"][0]
        assert ex["params"][1] == 2  # tag 2 = driven main
        assert ex["params"][2] == 23  # centre of 45 segments

    def test_stamp_scales_with_frequency(self):
        """Higher frequency → shorter wires."""
        deck_20m = _stamp_moxon_deck(14.175)
        deck_10m = _stamp_moxon_deck(28.4)
        gw20 = [c for c in deck_20m["cards"] if c["type"] == "GW"]
        gw10 = [c for c in deck_10m["cards"] if c["type"] == "GW"]
        # Driven main wire (GW 2) X-span should scale ~2:1
        width_20 = abs(gw20[1]["params"][2] - gw20[1]["params"][5])
        width_10 = abs(gw10[1]["params"][2] - gw10[1]["params"][5])
        ratio = width_20 / width_10
        assert 1.8 < ratio < 2.2, f"Width ratio {ratio:.2f} should be ~2.0"

    def test_stamp_wire_radius(self):
        """Wire radius should match the requested diameter."""
        deck = _stamp_moxon_deck(14.175, wire_dia_mm=3.0)
        gw = [c for c in deck["cards"] if c["type"] == "GW"][0]
        assert abs(gw["params"][8] - 0.0015) < 1e-6  # 3mm dia → 1.5mm radius

    def test_stamp_passes_validation(self):
        """Stamped deck should pass validate_deck with no critical issues."""
        deck = _stamp_moxon_deck(14.175)
        concepts = ExtractedConcepts(
            antenna_type="moxon", freq_mhz=14.175,
        )
        issues = validate_deck(deck, concepts)
        critical = [i for i in issues if any(
            kw in i for kw in ("MISSING", "COLLAPSED", "zero-length",
                               "NO GW", "radius ≤ 0")
        )]
        assert not critical, f"Critical issues: {critical}"

    def test_stamp_converts_to_nec(self):
        """Stamped deck converts to NEC text with GW cards."""
        deck = _stamp_moxon_deck(14.175)
        nec = convert_to_nec(deck)
        assert "GW" in nec
        assert "EX" in nec
        assert "EN" in nec

    def test_stamp_ground_types(self):
        """Ground type parameter should produce correct cards."""
        # Free space: GE 0, no GN
        deck_fs = _stamp_moxon_deck(14.175, ground_type="free_space")
        types_fs = [c["type"] for c in deck_fs["cards"]]
        assert "GN" not in types_fs

        # Real ground: GE 0 + GN card present
        deck_rg = _stamp_moxon_deck(14.175, ground_type="real")
        types_rg = [c["type"] for c in deck_rg["cards"]]
        assert "GN" in types_rg

    def test_stamp_custom_height(self):
        """Custom height should be used in GW Z coordinates."""
        deck = _stamp_moxon_deck(14.175, height_m=5.0)
        gw = [c for c in deck["cards"] if c["type"] == "GW"][0]
        assert gw["params"][4] == 5.0  # z1
        assert gw["params"][7] == 5.0  # z2

    def test_stamp_tip_gap_geometry(self):
        """Driven and reflector tips should have the correct gap."""
        wl = 299_792_458.0 / (14.175e6)
        expected_gap = 0.010402 * wl  # ~0.220 m

        deck = _stamp_moxon_deck(14.175)
        gw_cards = [c for c in deck["cards"] if c["type"] == "GW"]

        # GW 1 tip Y (driven tip) = params[3] (y1)
        drv_tip_y = gw_cards[0]["params"][3]
        # GW 4 tip Y (reflector tip) = params[3] (y1)
        ref_tip_y = gw_cards[3]["params"][3]

        actual_gap = drv_tip_y - ref_tip_y
        assert abs(actual_gap - expected_gap) < 0.01, \
            f"Gap {actual_gap:.4f}m vs expected {expected_gap:.4f}m"

    def test_stamp_driven_tail_neq_reflector_tail(self):
        """Driven tail (C) and reflector tail (D) must be different."""
        deck = _stamp_moxon_deck(14.175)
        gw_cards = [c for c in deck["cards"] if c["type"] == "GW"]
        # GW 1 = driven tail: length = abs(y2 - y1)
        c_len = abs(gw_cards[0]["params"][6] - gw_cards[0]["params"][3])
        # GW 4 = reflector tail: length = abs(y2 - y1)
        d_len = abs(gw_cards[3]["params"][6] - gw_cards[3]["params"][3])
        assert abs(c_len - d_len) > 0.1, \
            f"C={c_len:.3f} ≈ D={d_len:.3f} — tails should differ"


# ===================================================================
# LPDA calc_for_type regression (GH bug: missing freq args)
# ===================================================================

class TestLPDACalcForType:
    """calc_for_type('lpda', freq) must not crash when only a single
    design frequency is supplied — it should derive a broadband range."""

    def test_lpda_single_freq_does_not_crash(self):
        """Regression: calc_lpda() was called with no args via calc_for_type."""
        from antenna_classifier.nec_calculators import calc_for_type

        result = calc_for_type("lpda", 14.0)
        assert result is not None
        assert result.antenna_type == "lpda"
        assert result.dimensions["freq_range_mhz"][0] < 14.0
        assert result.dimensions["freq_range_mhz"][1] > 14.0

    def test_lpda_explicit_range_respected(self):
        """Explicit freq_mhz_low/high kwargs must override the default."""
        from antenna_classifier.nec_calculators import calc_for_type

        result = calc_for_type("lpda", 14.0,
                               freq_mhz_low=10.0, freq_mhz_high=30.0)
        assert result.dimensions["freq_range_mhz"] == [10.0, 30.0]

    def test_lpda_elements_generated(self):
        """LPDA calc must produce multiple elements and a boom."""
        from antenna_classifier.nec_calculators import calc_for_type

        result = calc_for_type("lpda", 21.0)
        assert result.dimensions["n_elements"] >= 3
        assert result.dimensions["boom_length"] > 0
        assert len(result.dimensions["element_lengths"]) == result.dimensions["n_elements"]
