import unittest
import numpy as np
from voicelab import dsp


class DSPTests(unittest.TestCase):
    def test_profile_can_never_inject_reference_speech(self):
        sr = 48000
        t = np.arange(sr * 3) / sr
        x = np.sin(2 * np.pi * 240 * t) * 0.1 + np.sin(2 * np.pi * 2400 * t) * 0.02
        y = np.sin(2 * np.pi * 240 * t) * 0.1 + np.sin(2 * np.pi * 2400 * t) * 0.05
        profile = dsp.fit_tone(x, y, x, interval=(0, 3))
        out = dsp.apply_tone(np.zeros(sr * 2), profile)
        self.assertLess(np.max(abs(out)), 1e-12)
        self.assertNotIn("samples", profile)

    def test_fir_preserves_impulse_timing_and_duration(self):
        x = np.zeros(48000)
        x[24000] = 0.1
        p = {
            "frequencies_hz": [60, 300, 1000, 5000, 18000],
            "gains_db": [0, 1, -2, 2, 0],
        }
        y = dsp.apply_tone(x, p)
        self.assertEqual(len(x), len(y))
        self.assertEqual(np.argmax(abs(y)), 24000)

    def test_quiet_input_does_not_turn_into_loud_noise(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1e-8, 48000)
        y, _ = dsp.master(x)
        self.assertLess(dsp.rms(y), 1e-5)

    def test_peak_ceiling_and_no_compression_when_disabled(self):
        x = np.sin(np.arange(48000) * 0.04) * 0.8
        y, m = dsp.master(x, ratio=1, cutoff=0, level_strength=0)
        self.assertLessEqual(float(dsp.db(np.max(abs(y)))), -3.999)
        self.assertEqual(m["compression_max_db"], 0.0)


if __name__ == "__main__":
    unittest.main()


class LimiterTests(unittest.TestCase):
    def test_isolated_transient_control_has_no_shift_or_clipping(self):
        x = np.sin(np.arange(48000) * 0.03) * 0.02
        x[24000] = 1.4
        y, m = dsp.limit_peaks(x, peak_db=-4)
        self.assertEqual(len(x), len(y))
        self.assertEqual(np.argmax(abs(y)), 24000)
        self.assertLessEqual(np.max(abs(y)), 10 ** (-4 / 20) + 1e-10)
        self.assertGreater(m["limiter_max_reduction_db"], 0)
        self.assertGreater(dsp.rms(y[:10000]), dsp.rms(x[:10000]) * 0.999)

    def test_normal_speech_level_is_unmodified_below_ceiling(self):
        x = np.random.default_rng(7).normal(0, 0.01, 100003)
        y, m = dsp.limit_peaks(x)
        np.testing.assert_allclose(y, x, atol=1e-12)
        self.assertEqual(m["limiter_max_reduction_db"], 0)
