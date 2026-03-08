"""Integration tests for the deterministic tuner with real NEC files.

These tests construct intentionally off-tune antenna decks and verify
that the tuner fires, improves performance metrics, or correctly
declines to act for non-tunable types.

Tests marked @pytest.mark.solver require the nec2c Docker solver at
localhost:8787.  Run with: pytest -m solver tests/test_tuner_integration.py
"""

import copy
import pytest

from antenna_classifier.nec_tuner import (
    TuneReport,
    _classify_wires,
    _identify_roles_with_deck,
    is_tunable,
    tune_deck,
)


def _solver_available() -> bool:
    try:
        import urllib.request
        r = urllib.request.urlopen("http://localhost:8787/health", timeout=2)
        return r.status == 200
    except Exception:
        return False


_skip_no_solver = pytest.mark.skipif(
    not _solver_available(), reason="nec2c solver not running on localhost:8787"
)
# Composite marker: both selectable via -m solver AND auto-skipped
solver = pytest.mark.solver(_skip_no_solver)


# ── Off-tune decks ────────────────────────────────────────────────

def _offtune_yagi_3el(freq_mhz: float = 28.5) -> dict:
    """3-element Yagi with deliberately wrong element lengths.

    Driven element is ~15% too long (should resonate much lower),
    director ~10% too long, reflector OK — forces high SWR / bad X.
    """
    return {"cards": [
        {"type": "CM", "text": f"Off-tune 3-el Yagi at {freq_mhz} MHz"},
        {"type": "CE"},
        # Reflector (tag 1) — roughly correct
        {"type": "GW", "params": [1, 21, 0.0, -2.70, 10.0, 0.0, 2.70, 10.0, 0.001]},
        # Driven (tag 2) — 15% too long → high |X|, bad SWR
        {"type": "GW", "params": [2, 21, 1.5, -2.88, 10.0, 1.5, 2.88, 10.0, 0.001]},
        # Director (tag 3) — 10% too long
        {"type": "GW", "params": [3, 21, 3.5, -2.53, 10.0, 3.5, 2.53, 10.0, 0.001]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 2, 11, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 37, 73, 1000, 0, 0, 5, 5]},
        {"type": "EN"},
    ]}


def _offtune_dipole(freq_mhz: float = 14.175) -> dict:
    """Half-wave dipole 12% too long — resonates below target frequency.

    A correct half-wave at 14.175 MHz has total length ~10.13m.
    We make it ~11.35m → high SWR at design frequency.
    """
    half_len = 5.675  # total ~11.35m, about 12% too long
    return {"cards": [
        {"type": "CM", "text": f"Off-tune dipole at {freq_mhz} MHz"},
        {"type": "CE"},
        # Left arm
        {"type": "GW", "params": [1, 21, 0.0, -half_len, 10.0,
                                   0.0, 0.0, 10.0, 0.001]},
        # Right arm
        {"type": "GW", "params": [2, 21, 0.0, 0.0, 10.0,
                                   0.0, half_len, 10.0, 0.001]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 1, 21, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 37, 73, 1000, 0, 0, 5, 5]},
        {"type": "EN"},
    ]}


def _vertical_groundplane(freq_mhz: float = 21.2) -> dict:
    """Quarter-wave vertical with radials.

    Vertical is NOT a tunable type — tests that the tuner correctly
    declines to act.
    """
    quarter_wl = 299.792 / freq_mhz / 4  # ~3.53m
    return {"cards": [
        {"type": "CM", "text": f"Vertical at {freq_mhz} MHz"},
        {"type": "CE"},
        # Vertical element (tag 1) — 10% too long
        {"type": "GW", "params": [1, 21, 0.0, 0.0, 0.0,
                                   0.0, 0.0, quarter_wl * 1.1, 0.001]},
        # 4 radials at ground level
        {"type": "GW", "params": [2, 11, 0.0, 0.0, 0.0,
                                   quarter_wl, 0.0, 0.0, 0.001]},
        {"type": "GW", "params": [3, 11, 0.0, 0.0, 0.0,
                                   -quarter_wl, 0.0, 0.0, 0.001]},
        {"type": "GW", "params": [4, 11, 0.0, 0.0, 0.0,
                                   0.0, quarter_wl, 0.0, 0.001]},
        {"type": "GW", "params": [5, 11, 0.0, 0.0, 0.0,
                                   0.0, -quarter_wl, 0.0, 0.001]},
        {"type": "GE", "params": [0]},
        {"type": "EX", "params": [0, 1, 1, 0, 1.0, 0.0]},
        {"type": "FR", "params": [0, 1, 0, 0, freq_mhz, 0]},
        {"type": "RP", "params": [0, 37, 73, 1000, 0, 0, 5, 5]},
        {"type": "EN"},
    ]}


# ── Unit tests (no solver needed) ─────────────────────────────────

class TestOffTuneDeckStructure:
    """Verify test decks are valid for the tuner's wire classifier."""

    def test_yagi_has_3_wires(self):
        wires = _classify_wires(_offtune_yagi_3el())
        assert len(wires) == 3

    def test_yagi_roles(self):
        deck = _offtune_yagi_3el()
        wires = _classify_wires(deck)
        roles = _identify_roles_with_deck(wires, deck)
        assert len(roles["driven"]) >= 1
        assert len(roles["reflector"]) >= 1
        assert len(roles["director"]) >= 1

    def test_dipole_has_2_wires(self):
        wires = _classify_wires(_offtune_dipole())
        assert len(wires) == 2

    def test_dipole_roles(self):
        deck = _offtune_dipole()
        wires = _classify_wires(deck)
        roles = _identify_roles_with_deck(wires, deck)
        assert len(roles["driven"]) >= 1

    def test_vertical_not_tunable(self):
        assert not is_tunable("vertical")

    def test_yagi_tunable(self):
        assert is_tunable("yagi")

    def test_dipole_tunable(self):
        assert is_tunable("dipole")


# ── Solver-backed integration tests ───────────────────────────────

@solver
class TestTunerFiresYagi:
    """Verify the tuner fires and improves SWR for an off-tune 3-el Yagi."""

    def test_yagi_improves_swr(self):
        deck = _offtune_yagi_3el(freq_mhz=28.5)
        tuned, report = tune_deck(deck, "yagi", 28.5)
        assert report.evals_used > 1, "Tuner should have used multiple evaluations"
        assert report.initial_swr > 2.0, (
            f"Off-tune Yagi should start with SWR > 2.0, got {report.initial_swr:.2f}"
        )
        assert report.final_swr < report.initial_swr, (
            f"Tuner should improve SWR: {report.initial_swr:.2f} → {report.final_swr:.2f}"
        )

    def test_yagi_deck_modified(self):
        deck = _offtune_yagi_3el(freq_mhz=28.5)
        original = copy.deepcopy(deck)
        tuned, report = tune_deck(deck, "yagi", 28.5)
        if report.evals_used > 1:
            assert tuned != original, "Tuned deck should differ from original"

    def test_yagi_report_has_adjustments(self):
        deck = _offtune_yagi_3el(freq_mhz=28.5)
        _, report = tune_deck(deck, "yagi", 28.5)
        if report.evals_used > 1:
            assert len(report.adjustments) > 0, "Should have recorded adjustments"


@solver
class TestTunerFiresDipole:
    """Verify the tuner fires on an off-tune dipole."""

    def test_dipole_processes(self):
        deck = _offtune_dipole(freq_mhz=14.175)
        tuned, report = tune_deck(deck, "dipole", 14.175)
        # Dipole has only 2 wires, both driven — tuner should at least try
        assert report.evals_used >= 1

    def test_dipole_swr_direction(self):
        """If tuner acts, SWR should improve or stay same."""
        deck = _offtune_dipole(freq_mhz=14.175)
        _, report = tune_deck(deck, "dipole", 14.175)
        if report.evals_used > 1:
            assert report.final_swr <= report.initial_swr + 0.5, (
                "Tuner should not make SWR significantly worse"
            )


@solver
class TestTunerSkipsVertical:
    """Verify the tuner correctly declines for non-tunable types."""

    def test_vertical_rejected_by_is_tunable(self):
        assert not is_tunable("vertical")

    def test_vertical_tune_deck_with_wrong_type(self):
        """If someone accidentally calls tune_deck with vertical type,
        the tuner should still handle it (few wires / no roles)."""
        deck = _vertical_groundplane(freq_mhz=21.2)
        tuned, report = tune_deck(deck, "vertical", 21.2)
        # Vertical has 5 wires (1 vert + 4 radials), tuner may try
        # but it's not in _TUNABLE_TYPES so pipeline won't call it.
        # Direct call should still be safe (no crash).
        assert isinstance(report, TuneReport)


@solver
class TestTunerCallbackFires:
    """Verify the on_step callback works during tuner execution."""

    def test_callback_fires_during_tune(self):
        """Run tune_deck and check that we can observe internal behaviour."""
        from unittest.mock import patch

        calls = []
        original_sim = None

        # Wrap _sim_deck to count calls — acts as a poor man's callback
        import antenna_classifier.nec_tuner as tuner_mod
        original_sim = tuner_mod._sim_deck

        def counting_sim(deck):
            result = original_sim(deck)
            calls.append(result is not None)
            return result

        deck = _offtune_yagi_3el(freq_mhz=28.5)
        with patch.object(tuner_mod, "_sim_deck", counting_sim):
            tuned, report = tune_deck(deck, "yagi", 28.5)

        assert len(calls) > 1, "Should have called solver multiple times"
        assert report.evals_used == len(calls), "evals_used should match sim calls"
