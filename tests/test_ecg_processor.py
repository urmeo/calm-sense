import unittest

import numpy as np

from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor
from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.preprocessing.ecg_processor import ECGProcessor


class RRCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.processor = ECGProcessor(sampling_rate=700)

    def test_ectopic_correction_holds_endpoints_instead_of_extrapolating(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([100, 600, 800, 750, 850], [600, 600, 800, 750, 850]),
                ([850, 750, 800, 600, 100], [850, 750, 800, 600, 600]),
            ):
                with self.subTest(method=method, rr=rr):
                    _, mask = self.processor.remove_ectopic_beats(rr)
                    corrected = self.processor.interpolate_artifacts(rr, mask, method)
                    np.testing.assert_array_equal(corrected, expected)

    def test_linear_interpolation_preserves_anchors_and_fills_interior_gaps(self):
        rr = np.array([0, 600, -1, np.nan, 900, np.inf])
        original = rr.copy()
        corrected = self.processor.interpolate_artifacts(rr)
        np.testing.assert_array_equal(corrected, [600, 600, 700, 800, 900, 900])
        np.testing.assert_array_equal(rr, original)

    def test_internal_gaps_cannot_overshoot_neighboring_valid_intervals(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([2500, 200, 700, 200, 2500], 200),
                ([200, 2500, 700, 2500, 200], 2500),
            ):
                with self.subTest(method=method, rr=rr):
                    corrected = self.processor.interpolate_artifacts(
                        rr, [True, True, False, True, True], method
                    )
                    self.assertEqual(corrected[2], expected)
                    np.testing.assert_array_equal(
                        corrected[[0, 1, 3, 4]], np.array(rr)[[0, 1, 3, 4]]
                    )

    def test_interpolation_stays_within_each_gap_and_preserves_valid_anchors(self):
        rng = np.random.default_rng(112)
        for method in ("linear", "cubic"):
            for _ in range(30):
                rr = rng.uniform(200, 2500, 40)
                mask = rng.random(40) > 0.5
                mask[[3, 36]] = True
                mask[:3] = mask[37:] = False
                anchors = np.flatnonzero(mask)
                corrected = self.processor.interpolate_artifacts(rr, mask, method)
                with self.subTest(method=method, anchors=anchors.tolist()):
                    self.assertEqual(corrected.shape, rr.shape)
                    self.assertTrue(np.isfinite(corrected).all())
                    np.testing.assert_array_equal(corrected[mask], rr[mask])
                    np.testing.assert_array_equal(corrected[:3], np.full(3, rr[3]))
                    np.testing.assert_array_equal(corrected[37:], np.full(3, rr[36]))
                    for left, right in zip(anchors[:-1], anchors[1:]):
                        gap = corrected[left : right + 1]
                        self.assertTrue(np.all(gap >= min(rr[left], rr[right])))
                        self.assertTrue(np.all(gap <= max(rr[left], rr[right])))

    def test_cubic_roundoff_at_a_physiological_bound_does_not_reject_valid_input(self):
        edge = np.nextafter(200.0, np.inf)
        rr = np.array([edge, 0, 0, 0, 0, 0, 0, 200, edge])
        corrected = self.processor.interpolate_artifacts(rr, method="cubic")
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue(np.all(corrected >= 200))
        self.assertTrue(np.all(corrected <= edge))
        np.testing.assert_array_equal(corrected[[0, 7, 8]], rr[[0, 7, 8]])

    def test_physiological_bounds_apply_without_a_mask_or_with_an_all_true_mask(self):
        rr = [100, 199.999, 200, 800, 2500, 2500.001, 3000, np.inf, np.nan]
        for mask in (None, np.ones(len(rr), dtype=bool)):
            with self.subTest(mask=mask):
                corrected = self.processor.interpolate_artifacts(rr, mask)
                np.testing.assert_array_equal(
                    corrected, [200, 200, 200, 800, 2500, 2500, 2500, 2500, 2500]
                )
        retained, mask = self.processor.remove_ectopic_beats(rr, threshold=100)
        np.testing.assert_array_equal(retained, [200, 800, 2500])
        np.testing.assert_array_equal(
            mask, [False, False, True, True, True, False, False, False, False]
        )

    def test_invalid_intervals_do_not_distort_the_local_median(self):
        for rr in ([100, 100, 800], [3000, 3000, 800], [np.nan, np.inf, 800]):
            with self.subTest(rr=rr):
                retained, mask = self.processor.remove_ectopic_beats(rr)
                np.testing.assert_array_equal(retained, [800])
                np.testing.assert_array_equal(mask, [False, False, True])

    def test_no_valid_anchor_returns_nan_without_inventing_intervals(self):
        for method in ("linear", "cubic"):
            for rr, mask in (
                ([0, -1, 100, 3000, np.nan, np.inf], None),
                ([800, 810], [False, False]),
            ):
                with self.subTest(method=method, rr=rr):
                    corrected = self.processor.interpolate_artifacts(rr, mask, method)
                    self.assertEqual(len(corrected), len(rr))
                    self.assertTrue(np.isnan(corrected).all())

    def test_one_or_two_anchors_work_for_both_methods(self):
        for method in ("linear", "cubic"):
            for rr, expected in (
                ([0, 800, 0], [800, 800, 800]),
                ([0, 600, 0, 800, 0], [600, 600, 700, 800, 800]),
            ):
                with self.subTest(method=method, rr=rr):
                    np.testing.assert_array_equal(
                        self.processor.interpolate_artifacts(rr, method=method),
                        expected,
                    )

    def test_valid_input_and_empty_input_keep_their_values(self):
        for method in ("linear", "cubic"):
            for rr in ([], [800], [200, 2500], [600, 800, 750, 850]):
                with self.subTest(method=method, rr=rr):
                    np.testing.assert_array_equal(
                        self.processor.interpolate_artifacts(rr, method=method), rr
                    )

    def test_units_match_rr_extraction_and_preserve_the_same_bounds(self):
        peaks = np.array([0, 70, 630, 1190, 1260])
        for unit, scale in (("ms", 1), ("s", 0.001), ("samples", 0.7)):
            with self.subTest(unit=unit):
                rr = self.processor.extract_rr_intervals(peaks, unit=unit)
                retained, mask = self.processor.remove_ectopic_beats(rr, unit=unit)
                np.testing.assert_allclose(retained, np.array([800, 800]) * scale)
                np.testing.assert_array_equal(mask, [False, True, True, False])
                np.testing.assert_allclose(
                    self.processor.interpolate_artifacts(rr, mask, unit=unit),
                    np.full(4, 800) * scale,
                )

    def test_local_outlier_threshold_is_consistent_across_units(self):
        for unit, scale in (("ms", 1), ("s", 0.001), ("samples", 0.7)):
            with self.subTest(unit=unit):
                rr = np.array([100, 600, 800, 750, 850]) * scale
                _, mask = self.processor.remove_ectopic_beats(rr, unit=unit)
                np.testing.assert_array_equal(mask, [False, True, True, True, True])
                np.testing.assert_allclose(
                    self.processor.interpolate_artifacts(rr, mask, unit=unit),
                    np.array([600, 600, 800, 750, 850]) * scale,
                )

    def test_invalid_options_are_rejected_even_when_no_interpolation_is_needed(self):
        for rr in ([], [800], [0]):
            with self.subTest(rr=rr):
                with self.assertRaisesRegex(ValueError, "method"):
                    self.processor.interpolate_artifacts(rr, method="quadratic")
                with self.assertRaisesRegex(ValueError, "unit"):
                    self.processor.interpolate_artifacts(rr, unit="minutes")
                with self.assertRaisesRegex(ValueError, "unit"):
                    self.processor.remove_ectopic_beats(rr, unit="minutes")
        with self.assertRaisesRegex(ValueError, "Mask length"):
            self.processor.interpolate_artifacts([800, 810], [True])
        for mask in ([True, 2], [True, np.nan], [True, "false"]):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    self.processor.interpolate_artifacts([800, 810], mask)

    def test_corrected_intervals_reach_hrv_extractors_without_losing_beats(self):
        rr = np.full(60, 800.0)
        rr[[0, 1, 15, 16, 40, 59]] = [100, 100, -200, np.inf, 3000, 3000]
        _, mask = self.processor.remove_ectopic_beats(rr)
        corrected = self.processor.interpolate_artifacts(rr, mask)
        np.testing.assert_array_equal(corrected, np.full(60, 800.0))
        time_features = HRVTimeDomainExtractor().extract_all(corrected)
        self.assertEqual(time_features["MeanNN"], 800)
        self.assertEqual(time_features["SDNN"], 0)
        self.assertEqual(time_features["RMSSD"], 0)
        frequency_features = HRVFrequencyDomainExtractor().extract_all(corrected)
        self.assertAlmostEqual(frequency_features["Total_power"], 0, delta=1e-12)
