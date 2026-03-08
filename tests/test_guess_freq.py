"""Tests for _guess_freq_mhz frequency extraction helper."""

from antenna_classifier.nec_generator import _guess_freq_mhz


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
