import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.config import FS
from src.dataset_wrist import WristDataset
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
