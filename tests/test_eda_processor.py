import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.config import FS
from src.dataset_wrist import WristDataset
from src.features.eda_features import EDAFeatureExtractor
from src.preprocessing.eda_processor import EDAProcessor


def component_amplitude(values, time, frequency):
    centered = values - np.mean(values)
    return 2 * abs(np.mean(centered * np.exp(-2j * np.pi * frequency * time)))


class EDAFilterTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(480) / FS.WRIST_EDA
        self.eda = (
            2
            + 0.5 * np.sin(2 * np.pi * 0.2 * self.time)
            + 0.25 * np.sin(2 * np.pi * 1.5 * self.time)
        )

    def test_wrist_default_preserves_slow_variation_and_attenuates_high_frequencies(
        self,
    ):
        original = self.eda.copy()
        filtered = EDAProcessor().lowpass_filter(self.eda)
        time, filtered_center = self.time[40:-40], filtered[40:-40]
        self.assertEqual(filtered.shape, self.eda.shape)
        self.assertTrue(np.isfinite(filtered).all())
        self.assertFalse(np.array_equal(filtered, self.eda))
        self.assertAlmostEqual(
            component_amplitude(filtered_center, time, 0.2), 0.5, delta=0.005
        )
        self.assertLess(component_amplitude(filtered_center, time, 1.5), 0.0025)
        np.testing.assert_array_equal(self.eda, original)

    def test_chest_default_keeps_the_existing_five_hertz_cutoff(self):
        time = np.arange(7000) / FS.CHEST
        eda = 2 + np.sin(2 * np.pi * time) + np.sin(2 * np.pi * 20 * time)
        processor = EDAProcessor(sampling_rate=FS.CHEST)
        np.testing.assert_array_equal(
            processor.lowpass_filter(eda), processor.lowpass_filter(eda, cutoff=5)
        )

    def test_explicit_cutoff_overrides_the_wrist_default(self):
        eda = 2 + np.sin(2 * np.pi * 0.75 * self.time)
        processor = EDAProcessor()
        default = processor.lowpass_filter(eda)
        lower_cutoff = processor.lowpass_filter(eda, cutoff=0.5)
        default_amplitude = component_amplitude(
            default[40:-40], self.time[40:-40], 0.75
        )
        lower_amplitude = component_amplitude(
            lower_cutoff[40:-40], self.time[40:-40], 0.75
        )
        self.assertGreater(default_amplitude, 0.9)
        self.assertLess(lower_amplitude, default_amplitude / 10)

    def test_invalid_cutoffs_raise_instead_of_returning_unfiltered_data(self):
        for eda in (self.eda, []):
            for cutoff in (2, 5):
                with self.subTest(cutoff=cutoff, samples=len(eda)):
                    with self.assertRaisesRegex(ValueError, "Nyquist"):
                        EDAProcessor().lowpass_filter(eda, cutoff=cutoff)
            for cutoff in (0, -1, np.nan, np.inf):
                with self.subTest(cutoff=cutoff, samples=len(eda)):
                    with self.assertRaisesRegex(ValueError, "cutoff"):
                        EDAProcessor().lowpass_filter(eda, cutoff=cutoff)

    def test_empty_and_constant_inputs_keep_their_shape_and_level(self):
        processor = EDAProcessor()
        np.testing.assert_array_equal(processor.lowpass_filter([]), [])
        np.testing.assert_allclose(processor.lowpass_filter(np.full(80, 3.0)), 3.0)

    def test_wrist_windows_receive_filtered_eda_before_decomposition(self):
        recording = {
            "label": np.ones(120 * int(FS.CHEST), dtype=int),
            "wrist": {
                "BVP": np.zeros(120 * int(FS.WRIST_BVP)),
                "EDA": self.eda[:, None],
                "TEMP": np.full((480, 1), 35.0),
                "ACC": np.zeros((120 * int(FS.WRIST_ACC), 3)),
            },
        }
        loader = SimpleNamespace(load_subject=lambda subject: recording)
        with patch("src.dataset_wrist.WESADLoader", return_value=loader):
            builder = WristDataset()
        peaks = np.arange(32, 120 * int(FS.WRIST_BVP), int(FS.WRIST_BVP))
        with patch.object(builder, "_bvp_peaks", return_value=peaks):
            windows, labels = builder._process_subject("S2")
        self.assertEqual(labels, [1, 1, 1])
        middle = windows[1]
        filtered = middle["eda_raw"]
        time = self.time[120:360]
        self.assertEqual(filtered.shape, (240,))
        self.assertAlmostEqual(
            component_amplitude(filtered, time, 0.2), 0.5, delta=0.005
        )
        self.assertLess(component_amplitude(filtered, time, 1.5), 0.0025)
        np.testing.assert_allclose(
            middle["eda_tonic"] + middle["eda_phasic"], filtered, rtol=1e-12
        )


class SCRDetectionTests(unittest.TestCase):
    def setUp(self):
        self.processor = EDAProcessor(sampling_rate=4)
        self.response = np.array(
            [0.02, 0.01, 0, 0.005, 0.01, 0.015, 0.02, 0.015, 0.01, 0.005, 0, 0.01, 0.02]
        )

    def test_baseline_shifts_preserve_the_known_response_and_its_timing(self):
        original = self.response.copy()
        for sampling_rate in (4, 700):
            samples_per_knot = sampling_rate // 4
            phasic = np.interp(
                np.arange((len(self.response) - 1) * samples_per_knot + 1)
                / samples_per_knot,
                np.arange(len(self.response)),
                self.response,
            )
            processor = EDAProcessor(sampling_rate=sampling_rate)
            for offset in (-1, -0.04, -0.015, 0, 0.04, 1):
                with self.subTest(sampling_rate=sampling_rate, offset=offset):
                    shifted = phasic + offset
                    peaks, responses = processor.detect_scr_peaks(shifted)
                    np.testing.assert_array_equal(peaks, [6 * samples_per_knot])
                    self.assertEqual(len(responses), 1)
                    self.assertEqual(responses[0]["peak_idx"], 6 * samples_per_knot)
                    self.assertEqual(responses[0]["onset_idx"], 2 * samples_per_knot)
                    self.assertAlmostEqual(responses[0]["amplitude"], 0.02)
                    self.assertEqual(responses[0]["rise_time"], 1)
                    self.assertEqual(responses[0]["recovery_time"], 0.5)
                    self.assertEqual(responses[0]["peak_time"], 1.5)
                    np.testing.assert_array_equal(shifted, phasic + offset)
        np.testing.assert_array_equal(self.response, original)

    def test_scr_feature_values_are_invariant_to_a_constant_baseline_shift(self):
        phasic = np.array(
            [
                0.02,
                0.01,
                0,
                0.01,
                0.03,
                0.06,
                0.03,
                0.01,
                0,
                0.01,
                0.02,
                0.04,
                0.02,
                0.01,
                0,
            ]
        )
        extractor = EDAFeatureExtractor(sampling_rate=4)
        _, responses = self.processor.detect_scr_peaks(phasic)
        expected = extractor.extract_phasic_features(responses, len(phasic) / 4)
        self.assertEqual(expected["SCR_count"], 2)
        for offset in (-2, -0.1, 0.1, 2):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                np.testing.assert_array_equal(peaks, [5, 11])
                actual = extractor.extract_phasic_features(responses, len(phasic) / 4)
                self.assertEqual(list(actual), list(expected))
                np.testing.assert_allclose(
                    list(actual.values()), list(expected.values())
                )

    def test_minimum_amplitude_boundary_tolerates_subtraction_roundoff(self):
        for offset in (-1, -0.1, 0, 0.1, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(
                    self.response / 2 + offset
                )
                np.testing.assert_array_equal(peaks, [6])
                self.assertAlmostEqual(responses[0]["amplitude"], 0.01)

    def test_small_responses_are_rejected_even_on_a_high_positive_baseline(self):
        for amplitude in (0.0075, 0.00999999):
            for offset in (-1, 0, 1):
                with self.subTest(amplitude=amplitude, offset=offset):
                    peaks, responses = self.processor.detect_scr_peaks(
                        self.response * (amplitude / 0.02) + offset
                    )
                    self.assertEqual(peaks.size, 0)
                    self.assertEqual(responses, [])

    def test_prominence_boundary_tolerates_subtraction_roundoff(self):
        phasic = np.array(
            [0.02, 0.01, 0, 0.005, 0.01, 0.015, 0.02, 0.018, 0.017, 0.016, 0.015]
        )
        for offset in (-1, -0.04, 0, 0.04, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                np.testing.assert_array_equal(peaks, [6])
                self.assertAlmostEqual(responses[0]["amplitude"], 0.02)
                self.assertIsNone(responses[0]["recovery_time"])

    def test_fast_rises_remain_rejected_after_a_baseline_shift(self):
        phasic = np.array([0.02, 0.01, 0, 0.02, 0.01, 0])
        for offset in (-1, 0, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                self.assertEqual(peaks.size, 0)
                self.assertEqual(responses, [])

    def test_missing_recovery_remains_missing_after_a_baseline_shift(self):
        phasic = np.array([0.02, 0.01, 0, 0.01, 0.03, 0.02, 0.02, 0.02])
        for offset in (-1, 0, 1):
            with self.subTest(offset=offset):
                peaks, responses = self.processor.detect_scr_peaks(phasic + offset)
                np.testing.assert_array_equal(peaks, [4])
                self.assertIsNone(responses[0]["recovery_time"])
