"""Unit tests for antenna_classifier.fingerprint."""

import math
import pytest

from antenna_classifier.parser import parse_text
from antenna_classifier.fingerprint import (
    FEATURE_NAMES,
    ArchetypeProfile,
    Fingerprint,
    analyze_lpda_fit,
    build_archetype,
    classify_by_fingerprint,
    find_similar,
    fingerprint,
    similarity,
)


def _simple_dipole() -> str:
    return (
        "CM dipole\n"
        "GW 1 21 0 -5 0 0 5 0 0.001\n"
        "GE 0\n"
        "EX 0 1 11 0 1 0\n"
        "FR 0 1 0 0 14.15\n"
        "EN\n"
    )


def _yagi_3el() -> str:
    return (
        "CM 3-element yagi\n"
        "GW 1 21 0 -5.5 0 0 5.5 0 0.001\n"
        "GW 2 21 3 -5.0 0 3 5.0 0 0.001\n"
        "GW 3 21 6 -4.5 0 6 4.5 0 0.001\n"
        "GE 0\n"
        "EX 0 1 11 0 1 0\n"
        "FR 0 1 0 0 14.15\n"
        "EN\n"
    )


def _loaded_vertical() -> str:
    return (
        "CM loaded vertical\n"
        "GW 1 21 0 0 0 0 0 10 0.001\n"
        "GE 0\n"
        "GN 1\n"
        "LD 5 1 1 1 58000000\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 7.1\n"
        "EN\n"
    )


def _helix() -> str:
    return (
        "GH 1 36 0.3 3 0.15 0.15 0.15 0.15 0.001\n"
        "GE 0\n"
        "GN 1\n"
        "EX 0 1 1 0 1 0\n"
        "FR 0 1 0 0 435.0\n"
        "EN\n"
    )


def _tl_fed() -> str:
    return (
        "GW 1 21 0 -5 0 0 5 0 0.001\n"
        "GW 2 21 3 -5 0 3 5 0 0.001\n"
        "GE 0\n"
        "TL 1 11 2 11 50.0\n"
        "EX 0 1 11 0 1 0\n"
        "FR 0 1 0 0 14.0\n"
        "EN\n"
    )


def _calculator_lpda() -> str:
    lengths = [2.0]
    tau = 0.92
    sigma = 0.05
    for _ in range(4):
        lengths.append(lengths[-1] / tau)
    positions = [0.0]
    for index in range(len(lengths) - 1):
        positions.append(positions[-1] + 2 * sigma * lengths[index + 1])
    lines = ["CM calculator-like LPDA"]
    for tag, (y_pos, length) in enumerate(zip(positions, lengths), start=1):
        half = length / 2.0
        lines.append(f"GW {tag} 21 {-half:.6f} {y_pos:.6f} 0 {half:.6f} {y_pos:.6f} 0 0.001")
    lines.extend([
        "GE 0",
        "TL 1 11 2 11 300.0",
        "TL 2 11 3 11 300.0",
        "TL 3 11 4 11 300.0",
        "TL 4 11 5 11 300.0",
        "EX 0 1 11 0 1 0",
        "FR 0 1 0 0 100.0",
        "EN",
    ])
    return "\n".join(lines)


def _hybrid_lpda_candidate() -> str:
    return (
        "CM hybrid log periodic candidate\n"
        "GW 1 21 -2.809013 0 0 2.711731 0 0 0.00635\n"
        "GW 2 21 -2.359084 0.7539341 0 2.359084 0.7539341 0 0.00635\n"
        "GW 3 21 -2.1402 1.775393 0 2.1402 1.775393 0 0.00635\n"
        "GW 4 21 -2.371244 3.575107 0 2.371244 3.575107 0 0.00635\n"
        "GW 5 21 -2.334764 6.080114 0 2.334764 6.080114 0 0.00635\n"
        "GE 0\n"
        "TL 2 11 3 11 -150.0\n"
        "EX 0 3 11 0 1 0\n"
        "FR 0 1 0 0 28.5\n"
        "EN\n"
    )


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_basic_counts(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.n_gw == 1
        assert fp.n_ex == 1
        assert fp.n_fr == 1
        assert fp.n_tags == 1

    def test_yagi_counts(self):
        fp = fingerprint(parse_text(_yagi_3el()))
        assert fp.n_gw == 3
        assert fp.n_tags == 3
        assert fp.n_ex == 1

    def test_card_types(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert "GW" in fp.card_types
        assert "EX" in fp.card_types
        assert "FR" in fp.card_types
        assert "GE" in fp.card_types
        assert "EN" in fp.card_types

    def test_ground_code(self):
        fp = fingerprint(parse_text(_loaded_vertical()))
        assert fp.ground_code == 1

    def test_no_ground(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.ground_code is None

    def test_frequency(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.freq_mhz == pytest.approx(14.15)

    def test_helix_detected(self):
        fp = fingerprint(parse_text(_helix()))
        assert fp.has_helix is True
        assert fp.n_gh == 1

    def test_lpda_reverse_fit_accepts_calculator_geometry(self):
        fit = analyze_lpda_fit(parse_text(_calculator_lpda()))
        assert fit is not None
        assert fit.conforms is True
        assert fit.fitted_tau == pytest.approx(0.92, abs=0.01)
        assert fit.fitted_sigma == pytest.approx(0.05, abs=0.01)

    def test_lpda_reverse_fit_rejects_hybrid_geometry(self):
        fit = analyze_lpda_fit(parse_text(_hybrid_lpda_candidate()))
        assert fit is not None
        assert fit.conforms is False
        assert fit.monotonic_lengths is False
        assert fit.max_spacing_error_pct > 20.0

    def test_loading_detected(self):
        fp = fingerprint(parse_text(_loaded_vertical()))
        assert fp.has_loading is True
        assert fp.n_ld == 1

    def test_network_detected(self):
        fp = fingerprint(parse_text(_tl_fed()))
        assert fp.has_network is True
        assert fp.n_tl == 1

    def test_tag_ex_ratio(self):
        fp = fingerprint(parse_text(_yagi_3el()))
        assert fp.tag_ex_ratio == pytest.approx(1 / 3)

    def test_wires_per_tag(self):
        fp = fingerprint(parse_text(_yagi_3el()))
        assert fp.wires_per_tag == pytest.approx(1.0)

    def test_symmetry_not_present(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.has_symmetry is False


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

class TestSignature:
    def test_basic_signature(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        sig = fp.signature
        assert sig.startswith("GW1:")
        assert "TAG1" in sig
        assert "EX1" in sig

    def test_tl_in_signature(self):
        fp = fingerprint(parse_text(_tl_fed()))
        assert "TL1" in fp.signature

    def test_ld_in_signature(self):
        fp = fingerprint(parse_text(_loaded_vertical()))
        assert "LD" in fp.signature

    def test_ground_in_signature(self):
        fp = fingerprint(parse_text(_loaded_vertical()))
        assert "GN1" in fp.signature


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_correct_length(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        vec = fp.feature_vector()
        assert len(vec) == len(FEATURE_NAMES)
        assert len(vec) == 21

    def test_values_are_finite(self):
        fp = fingerprint(parse_text(_yagi_3el()))
        vec = fp.feature_vector()
        assert all(math.isfinite(v) for v in vec)

    def test_binary_flags_are_0_or_1(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        vec = fp.feature_vector()
        # Indices 11-17 are boolean flags; 20 is has_ground
        for idx in [11, 12, 13, 14, 15, 16, 17, 20]:
            assert vec[idx] in (0.0, 1.0), f"Feature {FEATURE_NAMES[idx]} = {vec[idx]}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_keys(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        d = fp.to_dict()
        assert "signature" in d
        assert "n_gw" in d
        assert "complexity_score" in d
        assert "feed_complexity" in d
        assert "card_types" in d

    def test_to_dict_card_types_sorted(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        d = fp.to_dict()
        assert d["card_types"] == sorted(d["card_types"])


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_fingerprints(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert similarity(fp, fp) == pytest.approx(1.0)

    def test_similar_antennas(self):
        fp1 = fingerprint(parse_text(_simple_dipole()))
        fp2 = fingerprint(parse_text(_yagi_3el()))
        sim = similarity(fp1, fp2)
        assert 0.0 < sim < 1.0

    def test_different_types_lower_similarity(self):
        fp_dipole = fingerprint(parse_text(_simple_dipole()))
        fp_helix = fingerprint(parse_text(_helix()))
        fp_yagi = fingerprint(parse_text(_yagi_3el()))
        # Dipole should be more similar to yagi than to helix
        sim_dy = similarity(fp_dipole, fp_yagi)
        sim_dh = similarity(fp_dipole, fp_helix)
        assert sim_dy > sim_dh


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_basic_ranking(self):
        target = fingerprint(parse_text(_simple_dipole()))
        candidates = [
            ("yagi", fingerprint(parse_text(_yagi_3el()))),
            ("helix", fingerprint(parse_text(_helix()))),
            ("vertical", fingerprint(parse_text(_loaded_vertical()))),
        ]
        results = find_similar(target, candidates, top_n=3)
        assert len(results) > 0
        # Results sorted by descending similarity
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_min_similarity_filter(self):
        target = fingerprint(parse_text(_simple_dipole()))
        candidates = [
            ("helix", fingerprint(parse_text(_helix()))),
        ]
        results = find_similar(target, candidates, min_similarity=0.99)
        # Helix vs dipole shouldn't reach 0.99
        assert len(results) == 0

    def test_top_n_limit(self):
        target = fingerprint(parse_text(_simple_dipole()))
        candidates = [
            ("yagi", fingerprint(parse_text(_yagi_3el()))),
            ("helix", fingerprint(parse_text(_helix()))),
            ("vertical", fingerprint(parse_text(_loaded_vertical()))),
        ]
        results = find_similar(target, candidates, top_n=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Archetype building and classification
# ---------------------------------------------------------------------------

class TestArchetype:
    def test_build_archetype(self):
        fps = [
            fingerprint(parse_text(_simple_dipole())),
            fingerprint(parse_text(_simple_dipole())),
        ]
        arch = build_archetype("dipole", fps)
        assert arch.antenna_type == "dipole"
        assert arch.sample_count == 2
        assert arch.avg_gw == 1.0
        assert arch.avg_ex == 1.0

    def test_build_archetype_empty(self):
        arch = build_archetype("unknown", [])
        assert arch.sample_count == 0
        assert arch.avg_gw == 0

    def test_classify_by_fingerprint(self):
        dipole_fps = [fingerprint(parse_text(_simple_dipole()))]
        yagi_fps = [fingerprint(parse_text(_yagi_3el()))]
        archetypes = {
            "dipole": build_archetype("dipole", dipole_fps),
            "yagi": build_archetype("yagi", yagi_fps),
        }
        target = fingerprint(parse_text(_simple_dipole()))
        scores = classify_by_fingerprint(target, archetypes)
        assert len(scores) > 0
        # Dipole should rank first
        assert scores[0][0] == "dipole"


# ---------------------------------------------------------------------------
# Complexity and feed complexity
# ---------------------------------------------------------------------------

class TestComplexity:
    def test_simple_dipole_low_complexity(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.complexity_score < 0.5

    def test_loaded_vertical_higher_complexity(self):
        fp = fingerprint(parse_text(_loaded_vertical()))
        fp_dipole = fingerprint(parse_text(_simple_dipole()))
        assert fp.complexity_score >= fp_dipole.complexity_score

    def test_no_ex_zero_feed(self):
        nec = "GW 1 11 0 0 0 0 0 10 0.001\nGE 0\nEN"
        fp = fingerprint(parse_text(nec))
        assert fp.feed_complexity == 0.0

    def test_single_ex_feed(self):
        fp = fingerprint(parse_text(_simple_dipole()))
        assert fp.feed_complexity == pytest.approx(0.3)

    def test_single_ex_with_tl_feed(self):
        fp = fingerprint(parse_text(_tl_fed()))
        assert fp.feed_complexity == pytest.approx(0.5)
