"""Tests for the deterministic element-length tuner (nec_tuner.py).

Unit tests use mock simulation results. Integration tests (marked
with pytest.mark.solver) require the Docker nec2c solver at localhost:8787.
"""

import copy
import math
import pytest
from unittest.mock import patch, MagicMock

from antenna_classifier.nec_tuner import (
    TuneReport,
    WireInfo,
    _classify_wires,
    _cost,
    _extract_gain_fb,
    _extract_swr_r_x,
    _find_excitation_tag,
    _identify_roles_with_deck,
    _scale_wire,
    _scale_wire_group,
    is_tunable,
    tune_deck,
)


# ── Fixtures ───────────────────────────────────────────────────────

def _yagi_deck(freq_mhz: float = 28.5) -> dict:
    """A 3-element Yagi deck (reflector, driven, director)."""
    return {"cards": [
        {"type": "CM", "text": f"3-el Yagi at {freq_mhz} MHz"},
        {"type": "CE"},
        # Reflector (tag 1) at x=0
        {"type": "GW", "params": [1, 21, 0.0, -2.7, 10.0, 0.0, 2.7, 10.0, 0.001]},
        # Driven (tag 2) at x=1.5
        {"type": "GW", "params": [2, 21, 1.5, -2.5, 10.0, 1.5, 2.5, 10.0, 0.001]},
        # Director (tag 3) at x=3.5
        {"type": "GW", "params": [3, 21, 3.5, -2.3, 10.0, 3.5, 2.3, 10.0, 0.001]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 37, 73, 1000, 0, 0, 5, 5]},
        {"type": "EN"},
    ]}


def _phased_yagi_deck(freq_mhz: float = 28.5) -> dict:
    """Phased-driver Yagi with TL coupling (like Cebik design)."""
    return {"cards": [
        {"type": "CM", "text": "Phased-driver Yagi"},
        {"type": "CE"},
        {"type": "GW", "params": [1, 21, 0.0, -2.6, 10.0, 0.0, 2.6, 10.0, 0.001]},
        {"type": "GW", "params": [2, 21, 0.6, -2.45, 10.0, 0.6, 2.45, 10.0, 0.001]},
        {"type": "GW", "params": [3, 21, 2.2, -2.4, 10.0, 2.2, 2.4, 10.0, 0.001]},
        {"type": "GE", "params": [0]},
        {"type": "TL", "params": [1, 11, 2, 11, 50.0, 0.6, 0, 0, 0, 0]},
        {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "EN"},
    ]}


# ── Wire classification tests ─────────────────────────────────────

class TestClassifyWires:
    def test_finds_all_gw_cards(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        assert len(wires) == 3

    def test_boom_position_order(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        # Sorted by boom position (X axis)
        positions = [w.boom_position for w in wires]
        assert positions == sorted(positions)

    def test_wire_lengths(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        # Reflector is longest
        assert wires[0].length > wires[1].length > wires[2].length

    def test_skips_zero_length(self):
        deck = {"cards": [
            {"type": "GW", "params": [1, 21, 0, 0, 0, 0, 0, 0, 0.001]},
            {"type": "GW", "params": [2, 21, 0, -5, 10, 0, 5, 10, 0.001]},
        ]}
        wires = _classify_wires(deck)
        assert len(wires) == 1
        assert wires[0].tag == 2

    def test_empty_deck(self):
        wires = _classify_wires({"cards": []})
        assert wires == []

    def test_single_wire(self):
        deck = {"cards": [
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
        ]}
        wires = _classify_wires(deck)
        assert len(wires) == 1


# ── Role identification tests ─────────────────────────────────────

class TestIdentifyRoles:
    def test_standard_yagi_roles(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        roles = _identify_roles_with_deck(wires, deck)
        assert len(roles["reflector"]) == 1
        assert roles["reflector"][0].tag == 1
        assert len(roles["driven"]) == 1
        assert roles["driven"][0].tag == 2
        assert len(roles["director"]) == 1
        assert roles["director"][0].tag == 3

    def test_phased_driver_both_driven(self):
        deck = _phased_yagi_deck()
        wires = _classify_wires(deck)
        roles = _identify_roles_with_deck(wires, deck)
        driven_tags = {w.tag for w in roles["driven"]}
        assert driven_tags == {1, 2}
        assert len(roles["director"]) == 1
        assert roles["director"][0].tag == 3

    def test_excitation_tag_found(self):
        deck = _yagi_deck()
        assert _find_excitation_tag(deck) == 2


# ── Wire scaling tests ────────────────────────────────────────────

class TestScaleWire:
    def test_scale_up(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        driven = wires[1]
        original_len = driven.length

        new_deck = _scale_wire(deck, driven, 1.10)
        new_wires = _classify_wires(new_deck)
        assert new_wires[1].length == pytest.approx(original_len * 1.10, rel=1e-6)

    def test_scale_down(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        director = wires[2]
        original_len = director.length

        new_deck = _scale_wire(deck, director, 0.95)
        new_wires = _classify_wires(new_deck)
        assert new_wires[2].length == pytest.approx(original_len * 0.95, rel=1e-6)

    def test_midpoint_preserved(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        w = wires[0]
        mid_before = ((w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2, (w.z1 + w.z2) / 2)

        new_deck = _scale_wire(deck, w, 1.05)
        p = new_deck["cards"][w.card_index]["params"]
        mid_after = ((p[2] + p[5]) / 2, (p[3] + p[6]) / 2, (p[4] + p[7]) / 2)

        assert mid_after[0] == pytest.approx(mid_before[0], abs=1e-10)
        assert mid_after[1] == pytest.approx(mid_before[1], abs=1e-10)
        assert mid_after[2] == pytest.approx(mid_before[2], abs=1e-10)

    def test_scale_group(self):
        deck = _yagi_deck()
        wires = _classify_wires(deck)
        original_lengths = [w.length for w in wires]

        new_deck = _scale_wire_group(deck, wires, 1.05)
        new_wires = _classify_wires(new_deck)
        for orig, new in zip(original_lengths, new_wires):
            assert new.length == pytest.approx(orig * 1.05, rel=1e-6)

    def test_does_not_mutate_original(self):
        deck = _yagi_deck()
        deck_copy = copy.deepcopy(deck)
        wires = _classify_wires(deck)
        _scale_wire(deck, wires[0], 1.10)
        assert deck == deck_copy


# ── Cost function tests ───────────────────────────────────────────

class TestCostFunction:
    def test_min_swr(self):
        assert _cost(1.5, 50, 0, "min_swr") < _cost(3.0, 50, 0, "min_swr")

    def test_min_x(self):
        assert _cost(1.5, 50, 5, "min_x") < _cost(1.5, 50, 50, "min_x")

    def test_target_r(self):
        assert _cost(1.5, 50, 0, "target_r", 50) < _cost(1.5, 20, 0, "target_r", 50)

    def test_infinite_swr(self):
        assert _cost(float("inf"), 50, 0, "min_swr") == 1e6


# ── SWR extraction tests ──────────────────────────────────────────

class TestExtractSWR:
    def test_extracts_values(self):
        sim = {
            "swr_sweep": {"min_swr": 1.5},
            "impedance_sweep": {
                "freq_mhz": [28.0, 28.5, 29.0],
                "r": [30.0, 50.0, 45.0],
                "x": [-10.0, 0.0, 15.0],
            },
        }
        swr, r, x = _extract_swr_r_x(sim, 28.5)
        assert swr == 1.5
        assert r == 50.0
        assert x == 0.0

    def test_finds_closest_freq(self):
        sim = {
            "swr_sweep": {"min_swr": 2.0},
            "impedance_sweep": {
                "freq_mhz": [14.0, 14.175, 14.35],
                "r": [40.0, 55.0, 60.0],
                "x": [-20.0, 5.0, 30.0],
            },
        }
        swr, r, x = _extract_swr_r_x(sim, 14.2)
        assert r == 55.0  # closest to 14.175

    def test_empty_sim(self):
        swr, r, x = _extract_swr_r_x({}, 28.5)
        assert swr == float("inf")
        assert r == 0.0
        assert x == 0.0


class TestExtractGainFb:
    def test_extracts_pattern_metrics(self):
        gain, front_to_back = _extract_gain_fb({
            "radiation_pattern": {
                "max_gain_dbi": 7.2,
                "front_to_back_db": 18.5,
            },
        })
        assert gain == 7.2
        assert front_to_back == 18.5

    def test_missing_pattern_metrics_default_zero(self):
        gain, front_to_back = _extract_gain_fb({})
        assert gain == 0.0
        assert front_to_back == 0.0


# ── Tunability check ──────────────────────────────────────────────

class TestIsTunable:
    def test_yagi_tunable(self):
        assert is_tunable("yagi")

    def test_moxon_tunable(self):
        assert is_tunable("moxon")

    def test_dipole_tunable(self):
        assert is_tunable("dipole")

    def test_vertical_not_tunable(self):
        assert not is_tunable("vertical")

    def test_unknown_not_tunable(self):
        assert not is_tunable("unknown_type")


# ── tune_deck with mocked solver ──────────────────────────────────

class TestTuneDeckMocked:
    def _mock_sim_result(self, swr=1.5, r=50.0, x=0.0, ok=True, gain=6.0, front_to_back=12.0):
        return {
            "ok": ok,
            "swr_sweep": {
                "freq_mhz": [28.0, 28.5, 29.0],
                "swr": [swr * 1.1, swr, swr * 1.05],
                "min_swr": swr,
                "resonant_freq_mhz": 28.5,
            },
            "impedance_sweep": {
                "freq_mhz": [28.0, 28.5, 29.0],
                "r": [r * 0.9, r, r * 1.1],
                "x": [x - 5, x, x + 5],
                "z0": 50.0,
            },
            "radiation_pattern": {
                "max_gain_dbi": gain,
                "front_to_back_db": front_to_back,
            },
        }

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_already_passing_returns_immediately(self, mock_sim):
        mock_sim.return_value = self._mock_sim_result(swr=1.3, r=50.0, x=2.0)
        deck = _yagi_deck()
        tuned, report = tune_deck(deck, "yagi", 28.5)
        assert report.success
        assert report.evals_used == 1
        assert report.detail == "Already meets goals"

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_solver_unavailable(self, mock_sim):
        mock_sim.return_value = None
        deck = _yagi_deck()
        tuned, report = tune_deck(deck, "yagi", 28.5)
        assert not report.success
        assert "unavailable" in report.detail.lower()

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_sim_failure(self, mock_sim):
        mock_sim.return_value = {"ok": False, "error": "parse error"}
        deck = _yagi_deck()
        tuned, report = tune_deck(deck, "yagi", 28.5)
        assert not report.success
        assert "failed" in report.detail.lower()

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_too_few_wires(self, mock_sim):
        mock_sim.return_value = self._mock_sim_result(swr=5.0, r=20.0, x=-50.0)
        deck = {"cards": [
            {"type": "GW", "params": [1, 21, 0, -5, 10, 0, 5, 10, 0.001]},
            {"type": "EX", "params": [0, 1, 11, 0, 1.0, 0.0]},
        ]}
        tuned, report = tune_deck(deck, "dipole", 14.175)
        assert not report.success
        assert "few wires" in report.detail.lower()

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_convergence_sequence(self, mock_sim):
        """Simulate progressive improvement over multiple evals."""
        # First call = baseline (bad), then subsequent calls improve
        call_count = [0]
        def improving_sim(deck):
            call_count[0] += 1
            n = call_count[0]
            # Start bad, converge toward good
            swr = max(1.2, 5.0 - n * 0.5)
            r = min(50.0, 20.0 + n * 4)
            x = max(0.0, -60.0 + n * 10)
            return self._mock_sim_result(swr=swr, r=r, x=x)

        mock_sim.side_effect = improving_sim
        deck = _yagi_deck()
        tuned, report = tune_deck(deck, "yagi", 28.5)
        # Should have made progress
        assert report.evals_used > 1
        assert report.final_swr <= report.initial_swr

    @patch("antenna_classifier.nec_tuner._sim_deck")
    def test_ga_mode_uses_seeded_refinement(self, mock_sim):
        results = [
            self._mock_sim_result(swr=4.2, r=22.0, x=-45.0, gain=5.0, front_to_back=9.0),
            self._mock_sim_result(swr=2.1, r=46.0, x=-8.0, gain=6.0, front_to_back=13.0),
            self._mock_sim_result(swr=1.7, r=49.0, x=-2.0, gain=7.8, front_to_back=18.0),
            self._mock_sim_result(swr=1.9, r=48.0, x=-4.0, gain=7.2, front_to_back=16.0),
            self._mock_sim_result(swr=1.8, r=47.5, x=-3.0, gain=7.4, front_to_back=17.0),
        ]
        mock_sim.side_effect = lambda _deck: results.pop(0) if results else self._mock_sim_result()

        deck = _yagi_deck()
        tuned, report = tune_deck(deck, "yagi", 28.5, mode="ga", max_evals=5)

        assert report.mode == "ga"
        assert report.evals_used == 5
        assert report.final_swr <= 1.9
        assert report.final_gain_dbi >= 7.2
        assert report.final_front_to_back_db >= 16.0
        assert any("GA refinement" in step for step in report.adjustments)
