import unittest

import numpy as np

from src.config import FILTER_PARAMS, FS
from src.preprocessing.filters import SignalProcessor


class ButterworthFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = SignalProcessor(fs=FS.CHEST)
        cls.time = np.arange(int(400 * FS.CHEST)) / FS.CHEST
        cls.center = slice(int(100 * FS.CHEST), int(300 * FS.CHEST))

    def amplitude(self, values, frequency):
        values = values[self.center]
        phase = np.exp(-2j * np.pi * frequency * self.time[self.center])
        return 2 * abs(np.mean((values - values.mean()) * phase))

    def test_low_and_high_pass_keep_the_requested_order_at_small_cutoffs(self):
        for cutoff in (0.1, 3.5):
            frequencies = (cutoff / 2, 2 * cutoff)
            data = sum(np.sin(2 * np.pi * f * self.time) for f in frequencies)
            for order in (3, 4, 6):
                for kind in ("low", "high"):
                    with self.subTest(cutoff=cutoff, order=order, kind=kind):
                        filtered = self.processor.butterworth_filter(
                            data, cutoff, order=order, btype=kind
                        )
                        self.assertTrue(np.isfinite(filtered).all())
                        for frequency in frequencies:
                            ratio = np.tan(np.pi * frequency / FS.CHEST) / np.tan(
                                np.pi * cutoff / FS.CHEST
                            )
                            if kind == "high":
                                ratio = 1 / ratio
                            expected = 1 / (1 + ratio ** (2 * order))
                            self.assertAlmostEqual(
                                self.amplitude(filtered, frequency),
                                expected,
                                delta=2e-5,
                            )

    def test_low_cutoff_band_filters_keep_the_requested_design_order(self):
        frequencies = (0.2, 0.8)
        data = sum(np.sin(2 * np.pi * f * self.time) for f in frequencies)
        low, high = np.tan(np.pi * np.array([0.1, 0.5]) / FS.CHEST)
        for order in (3, 4, 6):
            for kind in ("band", "bandstop"):
                with self.subTest(order=order, kind=kind):
                    filtered = self.processor.butterworth_filter(
                        data, (0.1, 0.5), order=order, btype=kind
                    )
                    self.assertEqual(filtered.shape, data.shape)
                    self.assertTrue(np.isfinite(filtered).all())
                    for frequency in frequencies:
                        warped = np.tan(np.pi * frequency / FS.CHEST)
                        ratio = abs((warped**2 - low * high) / ((high - low) * warped))
                        if kind == "bandstop":
                            ratio = 1 / ratio
                        expected = 1 / (1 + ratio ** (2 * order))
                        self.assertAlmostEqual(
                            self.amplitude(filtered, frequency), expected, delta=2e-5
                        )

    def test_respiration_uses_its_configured_fourth_order_design(self):
        data = np.sin(2 * np.pi * 0.8 * self.time)
        filtered = self.processor.process_respiration(data)
        self.assertEqual(FILTER_PARAMS.RESP_FILTER_ORDER, 4)
        self.assertLess(self.amplitude(filtered, 0.8), 0.01)
        lower_order = self.processor.butterworth_filter(
            data, (0.1, 0.5), order=2, btype="band"
        )
        self.assertGreater(self.amplitude(lower_order, 0.8), 0.07)

    def test_real_window_lengths_stay_finite_without_mutating_input(self):
        for duration in (60, 300):
            time = np.arange(int(duration * FS.CHEST)) / FS.CHEST
            respiration = np.sin(2 * np.pi * 0.25 * time)
            temperature = np.full(len(time), 35.0)
            original = respiration.copy()
            filtered = self.processor.process_respiration(respiration)
            self.assertEqual(filtered.shape, respiration.shape)
            self.assertTrue(np.isfinite(filtered).all())
            np.testing.assert_array_equal(respiration, original)
            np.testing.assert_allclose(
                self.processor.process_temperature(temperature), 35.0, atol=1e-6
            )

    def test_short_signals_raise_instead_of_lowering_the_requested_order(self):
        for length in (10, 20):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "length.*padlen"):
                    self.processor.butterworth_filter(np.ones(length), 0.1, order=6)

    def test_empty_and_invalid_inputs_keep_explicit_validation(self):
        np.testing.assert_array_equal(
            self.processor.butterworth_filter([], 0.1, order=4), []
        )
        for order in (0, -1, 2.5, True):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "order"):
                self.processor.butterworth_filter(np.ones(100), 0.1, order=order)
        for cutoff in (0, -0.1, FS.CHEST / 2, np.inf, np.nan, (0.5, 0.1)):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(
                ValueError, "Cutoff"
            ):
                self.processor.butterworth_filter(np.ones(100), cutoff)
        with self.assertRaisesRegex(ValueError, "finite samples"):
            self.processor.butterworth_filter([0, np.nan, 1], 0.1)


if __name__ == "__main__":
    unittest.main()
