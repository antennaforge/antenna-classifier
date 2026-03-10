"""Regression tests for the dashboard Tuning Lab backend."""

import pytest

from antenna_classifier import tuning_lab
from antenna_classifier.simulator import ImpedanceSweep, RadiationPattern, SWRSweep, SimulationResult


def _fake_result() -> SimulationResult:
    return SimulationResult(
        filename="exercise.nec",
        ok=True,
        swr=SWRSweep(
            freq_mhz=[13.6, 14.2, 14.8],
            swr=[2.6, 1.8, 1.4],
            z0=50.0,
            design_freq_mhz=14.2,
        ),
        impedance=ImpedanceSweep(
            freq_mhz=[13.6, 14.2, 14.8],
            r=[42.0, 47.5, 55.0],
            x=[18.0, 12.0, -6.0],
            z0=50.0,
        ),
        pattern=RadiationPattern(
            theta=[89.0, 89.0, 89.0],
            phi=[0.0, 180.0, 360.0],
            gain_db=[7.4, -12.8, 7.4],
        ),
    )


def _fake_full_pattern_result() -> SimulationResult:
    return SimulationResult(
        filename="exercise-full.nec",
        ok=True,
        pattern=RadiationPattern(
            theta=[15.0, 25.0, 25.0],
            phi=[0.0, 0.0, 180.0],
            gain_db=[6.2, 8.9, -3.4],
        ),
    )


class TestTuningLabCatalog:
    def test_lists_expected_exercises(self):
        exercises = tuning_lab.list_exercises()
        ids = {item["id"] for item in exercises}
        assert ids == {"dipole-basics", "vertical-match", "yagi-driven-element"}

    def test_detail_exposes_equations_and_controls(self):
        detail = tuning_lab.get_exercise("dipole-basics")
        assert detail["title"] == "Dipole Resonance Basics"
        assert detail["default_pattern_type"] == "azimuth"
        assert len(detail["controls"]) == 3
        assert len(detail["concepts"]) >= 3
        assert detail["equations"][0]["expression"] == "Z = R + jX"
        fine_trim = next(control for control in detail["controls"] if control["id"] == "fine_trim_mm")
        assert fine_trim["reveal_reactance_abs_max"] == 10.0

    def test_exercises_expose_expected_default_pattern_types(self):
        dipole = tuning_lab.get_exercise("dipole-basics")
        vertical = tuning_lab.get_exercise("vertical-match")
        yagi = tuning_lab.get_exercise("yagi-driven-element")

        assert dipole["default_pattern_type"] == "azimuth"
        assert vertical["default_pattern_type"] == "elevation"
        assert yagi["default_pattern_type"] == "azimuth"

    def test_yagi_detail_calls_out_coupling_concepts(self):
        detail = tuning_lab.get_exercise("yagi-driven-element")
        control_ids = {control["id"] for control in detail["controls"]}
        assert {"height_scale", "reflector_enabled", "reflector_spacing_scale", "director_enabled", "director_spacing_scale", "hairpin_match_scale"}.issubset(control_ids)
        hairpin_control = next(control for control in detail["controls"] if control["id"] == "hairpin_match_scale")
        assert hairpin_control["reveal_analysis_flag"] == "match_stage_ready"
        assert hairpin_control["max"] == 2.0
        concepts = " ".join(detail["concepts"])
        assert "Mutual coupling" in concepts
        assert "Element spacing" in concepts
        assert "Matching" in concepts
        progression = detail["progression_steps"]
        assert any("+2% to +5%" in step["target"] for step in progression)
        assert any("−12% to −2%" in step["target"] for step in progression)
        assert any("hairpin" in step["detail"].lower() for step in progression)
        reading_notes = " ".join(detail["reading_notes"])
        assert "Resonance and matching are related, but not the same task" in reading_notes
        director_control = next(control for control in detail["controls"] if control["id"] == "director_scale")
        assert director_control["min"] == 0.88

    def test_yagi_analysis_reports_live_reference_deltas(self, monkeypatch):
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: _fake_result())

        payload = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {"reflector_enabled": 1.0, "director_enabled": 1.0},
        )

        snapshot = payload["current"]["analysis"]["progression_snapshot"]
        assert len(snapshot) == 6
        assert snapshot[0]["label"] == "Driven vs reference"
        assert snapshot[1]["label"] == "Reflector vs reference driven"
        assert snapshot[2]["label"] == "Reflector spacing vs reference"
        assert snapshot[3]["label"] == "Director vs reference driven"
        assert snapshot[4]["label"] == "Director spacing vs reference"
        assert snapshot[5]["label"] == "Hairpin match vs reference"
        assert snapshot[1]["target"] == "Target +2% to +5% after enabling"
        assert snapshot[3]["target"] == "Target -12% to -2% after enabling"

    def test_yagi_starts_as_driven_only_progression(self, monkeypatch):
        decks: list[str] = []

        def _capture_deck(filename, nec_text, **kwargs):
            decks.append(nec_text)
            return _fake_result()

        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", _capture_deck)

        payload = tuning_lab.simulate_exercise("yagi-driven-element")

        deck = decks[0]
        assert "GE 1" in deck
        assert "GN 1" in deck
        assert "GW 1" not in deck
        assert "GW 3" not in deck
        snapshot = payload["current"]["analysis"]["progression_snapshot"]
        assert snapshot[1]["value"] == "off"
        assert snapshot[3]["value"] == "off"
        next_action = payload["current"]["analysis"]["next_action"]
        assert next_action["control_id"] == "driven_scale"
        assert next_action["label"] == "Tune driven element"

        payload_with_driven_tuned = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {"driven_scale": 1.0, "height_scale": 1.0},
        )
        reflector_action = payload_with_driven_tuned["current"]["analysis"]["next_action"]
        assert reflector_action["control_id"] == "reflector_enabled"
        assert reflector_action["label"] == "Add reflector"

    def test_yagi_height_changes_deck_and_geometry_cards(self, monkeypatch):
        decks: list[str] = []

        def _capture_deck(filename, nec_text, **kwargs):
            decks.append(nec_text)
            return _fake_result()

        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", _capture_deck)

        low_payload = tuning_lab.simulate_exercise("yagi-driven-element", {"height_scale": 0.85})
        high_payload = tuning_lab.simulate_exercise("yagi-driven-element", {"height_scale": 1.4})

        assert len(decks) == 4
        current_decks = [decks[0], decks[2]]
        assert current_decks[0] != current_decks[1]
        height_card = low_payload["current"]["geometry_cards"][1]
        assert height_card["label"] == "Height"
        assert height_card["secondary_value"].endswith("ft")
        next_action = low_payload["current"]["analysis"]["next_action"]
        assert next_action["control_id"] == "height_scale"
        assert next_action["label"] == "Raise antenna height"

    def test_yagi_hairpin_match_is_final_unlock_and_load(self, monkeypatch):
        decks: list[str] = []

        def _capture_deck(filename, nec_text, **kwargs):
            decks.append(nec_text)
            return _fake_result()

        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", _capture_deck)

        geometry_ready = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {
                "driven_scale": 1.0,
                "height_scale": 1.0,
                "reflector_enabled": 1.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 1.0,
                "director_enabled": 1.0,
                "director_scale": 1.0,
                "director_spacing_scale": 1.0,
                "hairpin_match_scale": 0.0,
            },
        )
        next_action = geometry_ready["current"]["analysis"]["next_action"]
        assert geometry_ready["current"]["analysis"]["match_stage_ready"] is True
        assert next_action["control_id"] == "hairpin_match_scale"
        assert next_action["label"] == "Dial hairpin match"
        assert geometry_ready["current"]["analysis"]["progression_snapshot"][5]["value"] == "off"

        matched = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {
                "driven_scale": 1.0,
                "height_scale": 1.0,
                "reflector_enabled": 1.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 1.0,
                "director_enabled": 1.0,
                "director_scale": 1.0,
                "director_spacing_scale": 1.0,
                "hairpin_match_scale": 1.0,
            },
        )

        assert any("LD 0 2 16 16 0" in deck for deck in decks)
        ready_action = matched["current"]["analysis"]["next_action"]
        assert ready_action["control_id"] == ""
        assert ready_action["label"] == "Yagi lesson complete"
        hairpin_card = matched["current"]["geometry_cards"][-1]
        assert hairpin_card["label"] == "Hairpin L"
        assert hairpin_card["value"].endswith("nH")

    def test_yagi_hairpin_useful_match_exposes_directive_metrics(self, monkeypatch):
        useful_yagi = SimulationResult(
            filename="yagi.nec",
            ok=True,
            swr=SWRSweep(
                freq_mhz=[28.2, 28.4, 28.6],
                swr=[1.62, 1.32, 1.58],
                z0=50.0,
                design_freq_mhz=28.4,
            ),
            impedance=ImpedanceSweep(
                freq_mhz=[28.2, 28.4, 28.6],
                r=[44.0, 54.5, 57.0],
                x=[8.0, 3.2, -7.0],
                z0=50.0,
            ),
            pattern=RadiationPattern(
                theta=[89.0, 89.0, 89.0],
                phi=[0.0, 180.0, 360.0],
                gain_db=[8.1, -11.7, 8.0],
            ),
        )
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: useful_yagi)
        monkeypatch.setattr(tuning_lab, "_simulate_nec_pattern_text", lambda *args, **kwargs: _fake_full_pattern_result())

        payload = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {
                "driven_scale": 1.0,
                "height_scale": 1.0,
                "reflector_enabled": 1.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 1.0,
                "director_enabled": 1.0,
                "director_scale": 1.0,
                "director_spacing_scale": 1.0,
                "hairpin_match_scale": 1.0,
            },
        )

        analysis = payload["current"]["analysis"]
        assert analysis["useful_match"] is True
        assert analysis["hairpin_useful_match"] is True
        assert analysis["directive_gain_dbi"] == pytest.approx(8.9)
        assert analysis["front_to_back_db"] == pytest.approx(19.8)
        assert analysis["guidance"][0].startswith("The current hairpin setting is already useful")

    def test_yagi_directive_gain_uses_full_pattern_peak(self, monkeypatch):
        calls: list[str] = []

        useful_yagi = SimulationResult(
            filename="yagi.nec",
            ok=True,
            swr=SWRSweep(
                freq_mhz=[28.2, 28.4, 28.6],
                swr=[1.62, 1.32, 1.58],
                z0=50.0,
                design_freq_mhz=28.4,
            ),
            impedance=ImpedanceSweep(
                freq_mhz=[28.2, 28.4, 28.6],
                r=[44.0, 54.5, 57.0],
                x=[8.0, 3.2, -7.0],
                z0=50.0,
            ),
            pattern=RadiationPattern(
                theta=[89.0, 89.0, 89.0],
                phi=[0.0, 180.0, 360.0],
                gain_db=[-4.0, -23.8, -4.0],
            ),
        )

        def _capture_pattern(filename, nec_text, *, base_url, force_pattern):
            calls.append(force_pattern)
            if force_pattern == "full":
                return _fake_full_pattern_result()
            raise AssertionError(f"unexpected force_pattern {force_pattern}")

        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: useful_yagi)
        monkeypatch.setattr(tuning_lab, "_simulate_nec_pattern_text", _capture_pattern)

        payload = tuning_lab.simulate_exercise(
            "yagi-driven-element",
            {
                "driven_scale": 1.0,
                "height_scale": 1.0,
                "reflector_enabled": 1.0,
                "reflector_scale": 1.0,
                "reflector_spacing_scale": 1.0,
                "director_enabled": 1.0,
                "director_scale": 1.0,
                "director_spacing_scale": 1.0,
                "hairpin_match_scale": 1.0,
            },
        )

        analysis = payload["current"]["analysis"]
        assert calls == ["full", "full"]
        assert analysis["directive_gain_dbi"] == pytest.approx(8.9)
        assert analysis["front_to_back_db"] == pytest.approx(19.8)


class TestTuningLabSimulation:
    def test_analysis_translates_reactance_into_guidance(self, monkeypatch):
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: _fake_result())

        payload = tuning_lab.simulate_exercise("dipole-basics", {"length_scale": 1.04, "height_scale": 1.0})

        analysis = payload["current"]["analysis"]
        assert analysis["reactive_state"] == "inductive"
        assert analysis["match_quality"] == "workable"
        assert any("Shorten both legs" in item for item in analysis["guidance"])
        assert payload["current"]["geometry_cards"][0]["label"] == "Total length"

    def test_dipole_basics_uses_ground_model_so_height_can_change_match(self, monkeypatch):
        decks: list[str] = []

        def _capture_deck(filename, nec_text, **kwargs):
            decks.append(nec_text)
            return _fake_result()

        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", _capture_deck)

        tuning_lab.simulate_exercise("dipole-basics", {"length_scale": 1.0, "height_scale": 0.7})
        tuning_lab.simulate_exercise("dipole-basics", {"length_scale": 1.0, "height_scale": 1.3})

        assert len(decks) == 4
        current_decks = [decks[0], decks[2]]
        assert all("GE 1" in deck for deck in current_decks)
        assert all("GN 1" in deck for deck in current_decks)
        assert current_decks[0] != current_decks[1]

    def test_dipole_basics_fine_trim_adjusts_each_leg_in_millimeters(self, monkeypatch):
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: _fake_result())

        baseline = tuning_lab.simulate_exercise(
            "dipole-basics",
            {"length_scale": 1.0, "height_scale": 1.0, "fine_trim_mm": 0.0},
        )
        trimmed = tuning_lab.simulate_exercise(
            "dipole-basics",
            {"length_scale": 1.0, "height_scale": 1.0, "fine_trim_mm": 6.0},
        )

        baseline_geometry = baseline["current"]["analysis"]["geometry"]
        trimmed_geometry = trimmed["current"]["analysis"]["geometry"]

        assert baseline_geometry["half_length_m"] - trimmed_geometry["half_length_m"] == pytest.approx(0.006)
        fine_trim_card = trimmed["current"]["geometry_cards"][-1]
        assert fine_trim_card == {"label": "Fine trim", "value": "+6 mm/leg"}

    def test_geometry_cards_include_feet_for_length_values(self, monkeypatch):
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: _fake_result())

        payload = tuning_lab.simulate_exercise("vertical-match", {"radiator_scale": 1.0, "radial_scale": 1.0, "radial_slope_deg": 35.0})

        radiator_card = payload["current"]["geometry_cards"][0]
        radial_card = payload["current"]["geometry_cards"][1]
        slope_card = payload["current"]["geometry_cards"][2]

        assert radiator_card["secondary_value"].endswith("ft")
        assert radial_card["secondary_value"].endswith("ft")
        assert "secondary_value" not in slope_card

    def test_dipole_basics_flags_fine_tune_and_stop_when_match_is_good_enough(self, monkeypatch):
        good_enough = SimulationResult(
            filename="exercise.nec",
            ok=True,
            swr=SWRSweep(
                freq_mhz=[14.0, 14.2, 14.4],
                swr=[1.7, 1.34, 1.8],
                z0=50.0,
                design_freq_mhz=14.2,
            ),
            impedance=ImpedanceSweep(
                freq_mhz=[14.0, 14.2, 14.4],
                r=[47.0, 49.2, 52.0],
                x=[6.0, 2.4, -7.0],
                z0=50.0,
            ),
        )
        monkeypatch.setattr(tuning_lab, "_simulate_nec_text", lambda *args, **kwargs: good_enough)

        payload = tuning_lab.simulate_exercise("dipole-basics", {"length_scale": 1.0, "height_scale": 1.0, "fine_trim_mm": 0.0})

        analysis = payload["current"]["analysis"]
        assert analysis["fine_tuning_ready"] is True
        assert analysis["stop_trimming"] is True