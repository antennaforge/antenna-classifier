"""Tests for frequency and design-goal extraction helpers."""

from antenna_classifier.nec_generator import (
    _extract_design_goals,
    _guess_freq_mhz,
    DocumentGoals,
)


class TestGuessFreqMhz:
    """Frequency extraction from free text."""

    # --- Band reference in title (priority 1) ---

    def test_10_meters_in_title(self):
        text = "A Short Boom Three Element Yagi for 10 Meters\nSWR from 28 to 29 MHz."
        assert _guess_freq_mhz(text) == 28.4

    def test_20m_in_title(self):
        assert _guess_freq_mhz("20m Elevated Vertical design notes") == 14.175

    def test_40_meter_in_title(self):
        assert _guess_freq_mhz("A 40 meter dipole for field use") == 7.1

    def test_80m_in_title(self):
        assert _guess_freq_mhz("80m inverted V for NVIS") == 3.6

    # --- Frequency range (priority 2) ---

    def test_range_28_to_29(self):
        text = "X" * 300 + "Designed to cover 28 to 29 MHz with low SWR"
        assert _guess_freq_mhz(text) == 28.5

    def test_range_with_dash(self):
        text = "X" * 300 + "Operates across 144-148 MHz"
        assert _guess_freq_mhz(text) == 146.0

    def test_range_with_decimals(self):
        text = "X" * 300 + "Covers 14.0 to 14.35 MHz"
        assert _guess_freq_mhz(text) == 14.175

    # --- Explicit MHz (priority 3) ---

    def test_explicit_28_5(self):
        text = "X" * 300 + "Tuned at 28.5 MHz center frequency"
        assert _guess_freq_mhz(text) == 28.5

    def test_explicit_146(self):
        text = "X" * 300 + "A 146 MHz J-pole antenna"
        assert _guess_freq_mhz(text) == 146.0

    # --- Band reference in body (priority 4) ---

    def test_band_ref_in_body(self):
        text = "X" * 300 + "This is a great 15 meter yagi design"
        assert _guess_freq_mhz(text) == 21.2

    # --- Default fallback ---

    def test_no_frequency_info(self):
        assert _guess_freq_mhz("A nice antenna design with good gain") == 14.175

    # --- Title band reference beats body MHz ---

    def test_title_band_beats_body_mhz(self):
        """Title '10 Meters' should win over '29 MHz' deeper in the text."""
        text = "Yagi for 10 Meters\n" + "X" * 300 + "SWR from 28 to 29 MHz."
        assert _guess_freq_mhz(text) == 28.4


class TestExtractDesignGoals:
    """Design-goal extraction from document text."""

    # --- Gain ---

    def test_gain_dbi(self):
        goals = _extract_design_goals("This antenna achieves 7.1 dBi free-space gain.")
        assert goals.gain_dbi == 7.1

    def test_gain_dbd_converted(self):
        """dBd values should be converted to dBi (+2.15)."""
        goals = _extract_design_goals("Measured gain is 5.0 dBd over dipole.")
        assert goals.gain_dbi == 7.2  # 5.0 + 2.15, rounded to .1

    def test_gain_most_common(self):
        """When multiple values appear, pick the most frequent."""
        goals = _extract_design_goals("6.0 dBi at low end, 6.0 dBi mid, 7.1 dBi at top.")
        assert goals.gain_dbi == 6.0

    def test_no_gain(self):
        goals = _extract_design_goals("A simple dipole antenna design.")
        assert goals.gain_dbi is None

    # --- Front-to-back ---

    def test_fb_ratio_after_value(self):
        goals = _extract_design_goals("only 10 to 12 dB front-to-back ratio")
        assert goals.fb_db == 12.0

    def test_fb_ratio_before_value(self):
        goals = _extract_design_goals("front-to-back ratio of 25 dB")
        assert goals.fb_db == 25.0

    def test_fb_abbreviation(self):
        goals = _extract_design_goals("averages about 20 to 21 dB F/B")
        assert goals.fb_db == 21.0

    def test_no_fb(self):
        goals = _extract_design_goals("An omnidirectional vertical antenna.")
        assert goals.fb_db is None

    # --- SWR ---

    def test_swr_before_label(self):
        goals = _extract_design_goals("less than 1.5:1 SWR from 28 to 29 MHz")
        assert goals.max_swr == 1.5

    def test_swr_after_label(self):
        goals = _extract_design_goals("VSWR of 2.0:1 across the band")
        assert goals.max_swr == 2.0

    def test_swr_picks_tightest(self):
        goals = _extract_design_goals("1.5:1 SWR on CW, 2.0:1 SWR on SSB")
        assert goals.max_swr == 1.5

    def test_swr_ignores_1_0(self):
        """1.0:1 SWR is a theoretical point, not a realistic target."""
        goals = _extract_design_goals("SWR of 1.0:1 at resonance, 1.5:1 SWR at band edges")
        assert goals.max_swr == 1.5

    def test_swr_all_below_threshold(self):
        """If only 1.0:1 is mentioned, no SWR target is set."""
        goals = _extract_design_goals("SWR of 1.0:1 at center frequency")
        assert goals.max_swr is None

    # --- Bands ---

    def test_single_band(self):
        goals = _extract_design_goals("A 10 meter Yagi design")
        assert "10m" in goals.bands

    def test_multi_band(self):
        goals = _extract_design_goals("Covers 20m, 15m and 10m bands")
        assert goals.bands == ["20m", "15m", "10m"]

    def test_no_bands(self):
        goals = _extract_design_goals("Generic antenna design notes")
        assert goals.bands == []

    # --- Prompt block ---

    def test_prompt_block_full(self):
        goals = _extract_design_goals(
            "This 10 meter Yagi has 7.1 dBi gain, 21 dB F/B, and 1.5:1 SWR"
        )
        block = goals.prompt_block()
        assert "7.1 dBi" in block
        assert "21 dB" in block
        assert "1.5:1" in block
        assert "10m" in block

    def test_prompt_block_empty(self):
        goals = _extract_design_goals("Nothing useful here")
        assert goals.prompt_block() == ""

    # --- Cebik PDF-like text ---

    def test_cebik_like_text(self):
        text = (
            "A Short Boom, Wideband, Three Element Yagi for 10 Meters\n"
            "The beam must show less than 1.5:1 SWR from 28 to 29 MHz.\n"
            "manages about 6.0 dBi free-space gain across the band\n"
            "only 10 to 12 dB front-to-back ratio\n"
            "provides about 7.1 dBi free-space gain\n"
            "averages about 20 to 21 dB F/B\n"
        )
        goals = _extract_design_goals(text)
        assert goals.gain_dbi is not None
        assert goals.fb_db == 21.0
        assert goals.max_swr == 1.5
        assert "10m" in goals.bands
