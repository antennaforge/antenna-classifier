"""Regression tests for antenna_classifier.simulator.

Tests the helper functions that don't require a running NEC solver:
- _inject_rp: RP card replacement / insertion
- _build_sweep_deck: FR card rewriting for frequency sweeps
- SWRSweep dataclass properties (min_swr, resonant_freq, bandwidth)
- RadiationPattern dataclass properties (max_gain, front_to_back)
- SimulationResult.to_dict JSON serialization (inf/nan safety)
"""

import math
import textwrap

import pytest

from antenna_classifier.simulator import (
    ImpedanceSweep,
    RadiationPattern,
    SWRSweep,
    SimulationResult,
    _RP_CARDS,
    _build_sweep_deck,
    _inject_rp,
)


# ---------------------------------------------------------------------------
# _inject_rp
# ---------------------------------------------------------------------------

class TestInjectRP:
    """RP card injection/replacement — used by forced pattern feature."""

    DECK_WITH_RP = textwrap.dedent("""\
        CM Test
        GW 1 21 0 -5 0 0 5 0 0.001
        GE 0
        EX 0 1 11 0 1 0
        FR 0 1 0 0 14.15
        RP 0 1 361 1000 89 0 0 1
        EN
    """).strip()

    DECK_WITHOUT_RP = textwrap.dedent("""\
        CM Test
        GW 1 21 0 -5 0 0 5 0 0.001
        GE 0
        EX 0 1 11 0 1 0
        FR 0 1 0 0 14.15
        EN
    """).strip()

    DECK_MULTI_RP = textwrap.dedent("""\
        CM Test
        GW 1 21 0 -5 0 0 5 0 0.001
        GE 0
        EX 0 1 11 0 1 0
        FR 0 1 0 0 14.15
        RP 0 1 361 1000 89 0 0 1
        RP 0 181 1 1000 -90 0 1 0
        EN
    """).strip()

    def test_replaces_existing_rp(self):
        result = _inject_rp(self.DECK_WITH_RP, "RP 0 46 181 1000 0 0 2 2")
        lines = result.splitlines()
        rp_lines = [l for l in lines if l.strip().startswith("RP")]
        assert len(rp_lines) == 1
        assert "46 181" in rp_lines[0]

    def test_inserts_before_en_when_no_rp(self):
        result = _inject_rp(self.DECK_WITHOUT_RP, "RP 0 46 181 1000 0 0 2 2")
        lines = result.splitlines()
        rp_lines = [l for l in lines if l.strip().startswith("RP")]
        assert len(rp_lines) == 1
        # RP should appear before EN
        en_idx = next(i for i, l in enumerate(lines) if l.strip() == "EN")
        rp_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("RP"))
        assert rp_idx < en_idx

    def test_replaces_multiple_rp_with_one(self):
        result = _inject_rp(self.DECK_MULTI_RP, "RP 0 46 181 1000 0 0 2 2")
        lines = result.splitlines()
        rp_lines = [l for l in lines if l.strip().startswith("RP")]
        assert len(rp_lines) == 1

    def test_en_card_preserved(self):
        result = _inject_rp(self.DECK_WITH_RP, "RP 0 46 181 1000 0 0 2 2")
        assert "EN" in result.splitlines()[-1].upper()

    def test_all_standard_rp_cards_valid(self):
        """Every predefined RP card should start with 'RP'."""
        for name, card in _RP_CARDS.items():
            assert card.startswith("RP"), f"{name} card doesn't start with RP"


# ---------------------------------------------------------------------------
# _build_sweep_deck
# ---------------------------------------------------------------------------

class TestBuildSweepDeck:
    """Frequency-sweep deck builder — replaces FR, removes RP."""

    DECK = textwrap.dedent("""\
        CM Test
        GW 1 21 0 -5 0 0 5 0 0.001
        GE 0
        EX 0 1 11 0 1 0
        FR 0 1 0 0 14.15
        RP 0 1 361 1000 89 0 0 1
        EN
    """).strip()

    def test_returns_center_freq(self):
        _, center = _build_sweep_deck(self.DECK, n_points=11)
        assert center == pytest.approx(14.15)

    def test_rp_card_removed(self):
        deck, _ = _build_sweep_deck(self.DECK, n_points=11)
        assert "RP" not in deck.upper().split("\n")[-1]
        assert not any(l.strip().upper().startswith("RP") for l in deck.splitlines())

    def test_fr_card_rewritten(self):
        deck, _ = _build_sweep_deck(self.DECK, n_points=11)
        fr_lines = [l for l in deck.splitlines() if l.strip().upper().startswith("FR")]
        assert len(fr_lines) == 1
        # Should contain n_points
        assert "11" in fr_lines[0]

    def test_sweep_range_correct(self):
        """15% bandwidth around 14.15 MHz."""
        deck, center = _build_sweep_deck(self.DECK, n_points=21, bw_fraction=0.15)
        fr_line = [l for l in deck.splitlines() if l.strip().upper().startswith("FR")][0]
        parts = fr_line.replace(",", " ").split()
        # FR type, nfreq, 0, 0, fstart, fstep
        f_low = float(parts[5])
        step = float(parts[6])
        f_high = f_low + step * 20
        assert f_low == pytest.approx(14.15 * 0.85, rel=0.01)
        assert f_high == pytest.approx(14.15 * 1.15, rel=0.01)
        deck_no_fr = "GW 1 21 0 -5 0 0 5 0 0.001\nGE\nEX 0 1 11 0 1 0\nEN"
        _, center = _build_sweep_deck(deck_no_fr)
        assert center == 0.0


# ---------------------------------------------------------------------------
# SWRSweep dataclass
# ---------------------------------------------------------------------------

class TestSWRSweep:
    def test_min_swr(self):
        sweep = SWRSweep(freq_mhz=[14.0, 14.15, 14.3], swr=[2.5, 1.2, 3.1])
        assert sweep.min_swr == pytest.approx(1.2)

    def test_resonant_freq(self):
        sweep = SWRSweep(freq_mhz=[14.0, 14.15, 14.3], swr=[2.5, 1.2, 3.1])
        assert sweep.resonant_freq == pytest.approx(14.15)

    def test_bandwidth_2to1(self):
        sweep = SWRSweep(
            freq_mhz=[13.5, 13.8, 14.0, 14.15, 14.3, 14.5, 14.8],
            swr=[3.5, 1.9, 1.5, 1.1, 1.6, 1.8, 3.2],
        )
        bw = sweep.bandwidth_2to1
        assert bw is not None
        assert bw == pytest.approx(14.5 - 13.8)

    def test_no_bandwidth_when_all_above_2(self):
        """Regression: SWR all above 2 → bandwidth should be None."""
        sweep = SWRSweep(
            freq_mhz=[14.0, 14.15, 14.3],
            swr=[70.0, 65.0, 80.0],
        )
        assert sweep.bandwidth_2to1 is None

    def test_min_swr_with_inf(self):
        sweep = SWRSweep(freq_mhz=[14.0, 14.15], swr=[float("inf"), 1.5])
        assert sweep.min_swr == pytest.approx(1.5)

    def test_empty_swr(self):
        sweep = SWRSweep()
        assert sweep.min_swr == float("inf")
        assert sweep.resonant_freq is None
        assert sweep.bandwidth_2to1 is None


# ---------------------------------------------------------------------------
# RadiationPattern dataclass
# ---------------------------------------------------------------------------

class TestRadiationPattern:
    def test_max_gain(self):
        pat = RadiationPattern(theta=[0, 90], phi=[0, 0], gain_db=[5.0, 2.0])
        assert pat.max_gain == pytest.approx(5.0)

    def test_empty_pattern(self):
        pat = RadiationPattern()
        assert pat.max_gain == float("-inf")
        assert pat.front_to_back is None


# ---------------------------------------------------------------------------
# SimulationResult JSON serialization
# ---------------------------------------------------------------------------

class TestSimulationResultSerialization:
    """Regression: inf/nan values must not break JSON serialization."""

    def test_inf_replaced_with_none(self):
        result = SimulationResult(
            filename="test.nec", ok=True,
            swr=SWRSweep(freq_mhz=[14.0], swr=[float("inf")]),
        )
        d = result.to_dict()
        assert d["swr_sweep"]["swr"] == [None]
        assert d["swr_sweep"]["min_swr"] is None

    def test_nan_replaced_with_none(self):
        result = SimulationResult(
            filename="test.nec", ok=True,
            swr=SWRSweep(freq_mhz=[14.0], swr=[float("nan")]),
        )
        d = result.to_dict()
        assert d["swr_sweep"]["swr"] == [None]

    def test_to_dict_is_json_serializable(self):
        import json
        result = SimulationResult(
            filename="test.nec", ok=True,
            swr=SWRSweep(freq_mhz=[14.0, 14.15], swr=[2.5, float("inf")]),
            impedance=ImpedanceSweep(freq_mhz=[14.0], r=[50.0], x=[10.0]),
            pattern=RadiationPattern(theta=[0, 90], phi=[0, 0], gain_db=[5.0, 2.0]),
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_error_result_dict(self):
        result = SimulationResult(filename="bad.nec", ok=False, error="timeout")
        d = result.to_dict()
        assert d["ok"] is False
        assert d["error"] == "timeout"
