import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import product
import pickle
import numpy as np
import pandas as pd
from src import dataset, features, synthetic
from src.cache import cache_fingerprint
from src.config import VALID_SUBJECTS
from src.data.loader import WESADLoader
from src.dataset import WindowedDataset
from src.dataset_wrist import WristDataset
from src.features.accelerometer_features import AccelerometerFeatureExtractor
from src.features.eda_features import EDAFeatureExtractor
from src.features.feature_pipeline import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractionPipeline,
)
from src.features.hrv_frequency_domain import HRVFrequencyDomainExtractor
from src.features.hrv_nonlinear import HRVNonlinearExtractor
from src.features.hrv_time_domain import HRVTimeDomainExtractor
from src.features.respiration_features import RespirationFeatureExtractor
from src.features.temperature_features import TemperatureFeatureExtractor
from src.preprocessing.ecg_processor import ECGProcessor
from src.portable import wesad_portable
from tests.cache_fixtures import source_loader


class FeaturesTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_missing_eda_preserves_feature_order_and_zero_event_counts(self):
        extractor = EDAFeatureExtractor()
        features = extractor.extract_all({})
        self.assertEqual(
            list(features),
            [
                "SCL_mean",
                "SCL_std",
                "SCL_slope",
                "SCL_min",
                "SCL_max",
                "SCR_count",
                "SCR_rate",
                "SCR_amplitude_mean",
                "SCR_amplitude_max",
                "SCR_rise_time_mean",
                "SCR_recovery_time_mean",
                "SCR_AUC",
                "EDA_mean",
                "EDA_range",
                "EDA_kurtosis",
            ],
        )
        self.assertTrue(
            all(
                (
                    value == 0.0 if key.startswith("SCR_") else np.isnan(value)
                    for key, value in features.items()
                )
            )
        )
        features["SCL_mean"] = 3.0
        self.assertTrue(np.isnan(extractor.extract_all({})["SCL_mean"]))

    def test_compute_failure_preserves_earlier_values_and_remaining_nans(self):
        extractor = HRVTimeDomainExtractor()

        def fail(_):
            raise RuntimeError("synthetic compute failure")

        self.stack.enter_context(patch.object(extractor, "compute_sdnn", fail))
        features = extractor.extract_all(np.full(60, 800.0))
        self.assertEqual(features["MeanNN"], 800.0)
        self.assertTrue(
            all((np.isnan(value) for key, value in features.items() if key != "MeanNN"))
        )

    def test_rmssd_constant_rr_is_zero(self):
        rr = np.full(60, 800.0)
        features = HRVTimeDomainExtractor().extract_all(rr)
        self.assertEqual(features["RMSSD"], 0.0)
        self.assertLess(abs(features["MeanNN"] - 800.0), 1e-06)

    def test_triangular_index_uses_fixed_rr_bins_and_handles_constant_intervals(self):
        extractor = HRVTimeDomainExtractor()
        self.assertEqual(extractor.compute_hrvti(np.full(60, 800.0)), 1.0)
        self.assertEqual(extractor.compute_hrvti(np.tile([800.0, 807.0], 30)), 2.0)

    def test_frequency_features_nan_below_min_rr(self):
        feats = HRVFrequencyDomainExtractor().extract_all(np.full(5, 800.0))
        self.assertTrue(all((np.isnan(v) for v in feats.values())))

    def test_nonlinear_features_finite_with_enough_rr(self):
        rng = np.random.RandomState(0)
        rr = 800 + 30 * rng.randn(120)
        feats = HRVNonlinearExtractor().extract_all(rr)
        for key in ("SD1", "SD2", "SampEn"):
            self.assertTrue(
                np.isfinite(feats[key]),
                f"{key} should be finite on a well-formed RR series",
            )

    def test_poincare_geometry_and_cvi_use_the_paired_ellipse_axes(self):
        extractor = HRVNonlinearExtractor()
        trending = extractor.compute_poincare(np.array([800.0, 820.0, 840.0]))
        self.assertEqual(trending["SD1"], 0.0)
        np.testing.assert_allclose(trending["SD2"], 20.0, rtol=1e-06, atol=1e-12)
        features = extractor.compute_poincare(np.array([800.0, 810.0, 790.0, 820.0]))
        np.testing.assert_allclose(
            features["SD1"], np.sqrt(950 / 3), rtol=1e-06, atol=1e-12
        )
        np.testing.assert_allclose(
            features["SD2"], np.sqrt(50 / 3), rtol=1e-06, atol=1e-12
        )
        np.testing.assert_allclose(
            features["CVI"],
            np.log10(16 * np.sqrt(950 / 3) * np.sqrt(50 / 3)),
            rtol=1e-06,
            atol=1e-12,
        )

    def test_hrv_matches_known_sequence(self):
        rr = np.tile([800.0, 820.0], 30)
        f = HRVTimeDomainExtractor().extract_all(rr)
        self.assertLess(abs(f["MeanNN"] - 810.0), 1e-06)
        self.assertLess(abs(f["RMSSD"] - 20.0), 1e-06)
        self.assertLess(abs(f["SDNN"] - np.std(rr, ddof=1)), 1e-06)

    def test_too_few_rr_returns_nan(self):
        features = HRVTimeDomainExtractor().extract_all(np.array([800.0, 810.0]))
        self.assertTrue(np.isnan(features["SDNN"]))

    def test_rpeaks_recover_known_rate(self):
        fs = 700
        t = np.arange(0, 30, 1 / fs)
        ecg = np.sin(2 * np.pi * 1.0 * t) ** 21
        peaks = ECGProcessor(sampling_rate=fs).detect_r_peaks(ecg)
        rate = len(peaks) / 30
        self.assertTrue(0.8 < rate < 1.2)

    def test_tonic_and_temperature_slopes_preserve_missing_sample_times(self):
        time = np.arange(40) / 4.0
        tonic, temperature = (3 + 0.25 * time, 35 - 0.01 * time)
        tonic[[1, 4, 7, 15]] = np.nan
        temperature[[2, 6, 14, 29]] = np.inf
        np.testing.assert_allclose(
            EDAFeatureExtractor(4).extract_tonic_features(tonic)["SCL_slope"],
            0.25,
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            TemperatureFeatureExtractor(4).extract_all(temperature)["TEMP_slope"],
            -0.01,
            rtol=1e-06,
            atol=1e-12,
        )

    def test_rr_interpolation_uses_its_declared_sampling_rate(self):
        time, interpolated = HRVFrequencyDomainExtractor(
            interpolation_rate=4
        )._interpolate_rr(np.full(40, 810.0))
        np.testing.assert_allclose(np.diff(time), 0.25)
        np.testing.assert_allclose(interpolated, 810.0)

    def test_band_power_includes_unsampled_endpoints(self):
        extractor = HRVFrequencyDomainExtractor()
        freqs = np.arange(0, 2.01, 0.05)
        density = np.full(len(freqs), 2.0)
        np.testing.assert_allclose(
            extractor.compute_vlf_power(freqs, density),
            2 * (0.04 - 0.0033),
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            extractor.compute_lf_power(freqs, density),
            2 * (0.15 - 0.04),
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            extractor.compute_hf_power(freqs, density),
            2 * (0.4 - 0.15),
            rtol=1e-06,
            atol=1e-12,
        )

    def test_lomb_density_integrates_to_rr_variance_and_handles_constant_rr(self):
        extractor = HRVFrequencyDomainExtractor()
        rr = 800 + 30 * np.sin(np.arange(100) * 0.6)
        freqs, density = extractor.compute_psd(rr, method="lomb")
        np.testing.assert_allclose(
            np.trapezoid(density, freqs), np.var(rr), rtol=1e-06, atol=1e-12
        )
        features = extractor.extract_all(np.full(60, 800.0), method="lomb")
        self.assertEqual(features["Total_power"], 0.0)
        self.assertTrue(np.isnan(features["LF_peak_freq"]))
        with self.assertRaisesRegex(ValueError, "PSD method"):
            extractor.compute_psd(rr, method="typo")

    def test_lomb_default_grid_and_density_keep_the_existing_computation(self):
        from scipy.signal import lombscargle

        rr = 800 + 30 * np.sin(np.arange(100) * 0.6)
        time = np.r_[0.0, np.cumsum(rr[:-1]) / 1000.0]
        centered = rr - rr.mean()
        expected_freqs = np.linspace(0.001, 0.5, 512)
        expected = np.maximum(
            lombscargle(time, centered, 2 * np.pi * expected_freqs, normalize=False),
            0.0,
        )
        expected *= np.var(centered) / np.trapezoid(expected, expected_freqs)
        freqs, density = HRVFrequencyDomainExtractor().compute_psd(rr, method="lomb")
        np.testing.assert_array_equal(freqs, expected_freqs)
        np.testing.assert_array_equal(density, expected)

    def test_lomb_grid_covers_custom_bands_above_half_a_hertz(self):
        rr = 500 + 20 * np.sin(2 * np.pi * 0.8 * np.arange(200) * 0.5)
        extractor = HRVFrequencyDomainExtractor(hf_band=(0.7, 0.9))
        freqs, density = extractor.compute_psd(rr, method="lomb")
        self.assertEqual(freqs[-1], 0.9)
        self.assertGreater(extractor.compute_hf_power(freqs, density), 100.0)
        self.assertAlmostEqual(freqs[np.argmax(density)], 0.8, delta=0.005)
        np.testing.assert_allclose(np.trapezoid(density, freqs), np.var(rr))

    def test_portable_integer_acceleration_preserves_magnitude(self):
        duration = 60
        recording = {
            "label": np.ones(duration * 700, dtype=int),
            "signal": {
                "chest": {"ECG": np.zeros(duration * 700)},
                "wrist": {
                    "BVP": np.zeros(duration * 64),
                    "EDA": np.ones(duration * 4),
                    "TEMP": np.full(duration * 4, 35.0),
                    "ACC": np.full((duration * 32, 3), 200, dtype=np.int16),
                },
            },
        }
        subject = self.tmp_path / "S2"
        subject.mkdir()
        (subject / "S2.pkl").write_bytes(pickle.dumps(recording))
        with (
            patch("neurokit2.ppg_clean", side_effect=lambda values, **kwargs: values),
            patch(
                "neurokit2.ppg_findpeaks",
                return_value={"PPG_Peaks": np.arange(0, duration * 64, 64)},
            ),
        ):
            portable = wesad_portable(data_path=self.tmp_path)

        expected = np.sqrt(3 * 200.0**2)
        self.assertAlmostEqual(portable.iloc[0]["ACC_mean"], expected)
        self.assertAlmostEqual(portable.iloc[0]["ACC_energy"], expected**2)
        acceleration = recording["signal"]["wrist"]["ACC"]
        self.assertEqual(acceleration.dtype, np.int16)
        np.testing.assert_array_equal(acceleration, 200)

    def test_sample_entropy_is_zero_when_every_matching_pattern_extends(self):
        for rr in (np.full(20, 800.0), np.tile([800.0, 820.0], 10)):
            with self.subTest(rr=rr.tolist()):
                self.assertEqual(
                    HRVNonlinearExtractor().compute_sample_entropy(rr), 0.0
                )

    def test_sample_entropy_reports_zero_continuation_probability(self):
        rr = np.array([800, 800, 900, 800, 800, 1000])
        self.assertTrue(
            np.isinf(HRVNonlinearExtractor().compute_sample_entropy(rr, m=2, r=0.01))
        )

    def test_recurrence_determinism_uses_the_requested_embedding(self):
        extractor = HRVNonlinearExtractor()
        rr = np.tile([800.0, 820.0], 3)
        np.testing.assert_allclose(
            extractor.compute_rqa_determinism(rr, embedding_dim=2),
            6 / 8,
            rtol=1e-06,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            extractor.compute_rqa_determinism(rr, embedding_dim=2, time_delay=2),
            1.0,
            rtol=1e-06,
            atol=1e-12,
        )
        self.assertTrue(
            np.isnan(
                extractor.compute_rqa_determinism(rr, embedding_dim=4, time_delay=2)
            )
        )
        self.assertTrue(np.isnan(extractor.compute_dfa(np.full(100, 800.0))).all())

    def test_flat_signals_have_no_detected_respiration_or_motion_frequency(self):
        respiration = RespirationFeatureExtractor(10).extract_all(np.ones(200))
        self.assertTrue(np.isnan(respiration["RESP_rate"]))
        self.assertEqual(respiration["RESP_amplitude"], 0.0)
        self.assertEqual(respiration["RESP_apnea_index"], 100.0)
        movement = AccelerometerFeatureExtractor(32).extract_from_magnitude(
            np.ones(320)
        )
        self.assertTrue(np.isnan(movement["ACC_peak_freq"]))
        self.assertEqual(movement["ACC_zero_crossings"], 0.0)

    def test_missing_accelerometer_samples_do_not_compress_event_time(self):
        magnitude = np.tile([0.0, 2.0], 50)
        magnitude[20:30] = np.nan
        features = AccelerometerFeatureExtractor(10).extract_from_magnitude(magnitude)
        np.testing.assert_allclose(
            features["ACC_zero_crossings"], 88 / 10, rtol=1e-06, atol=1e-12
        )
        self.assertTrue(np.isnan(features["ACC_peak_freq"]))

    def test_ectopic_correction_excludes_nonfinite_and_nonpositive_intervals(self):
        processor = ECGProcessor()
        _, valid = processor.remove_ectopic_beats(
            np.array([800.0, np.nan, -10.0, 810.0, 800.0])
        )
        np.testing.assert_array_equal(valid, [True, False, False, True, True])
        corrected = processor.interpolate_artifacts(
            [800.0, np.nan, -10.0, 810.0, 800.0], valid
        )
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue(
            np.isnan(
                processor.interpolate_artifacts([4000, 5000], [False, False])
            ).all()
        )

    def test_rr_extraction_rejects_invalid_peak_order_or_values(self):
        for peaks in ([700, 350], [350, 350], [0, np.nan], [-1, 350]):
            with self.subTest(peaks=peaks):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    ECGProcessor().extract_rr_intervals(np.asarray(peaks))

    def test_ecg_empty_or_constant_input_has_no_peaks(self):
        processor = ECGProcessor()
        self.assertEqual(processor.detect_r_peaks(np.array([])).size, 0)
        self.assertEqual(processor.detect_r_peaks(np.ones(100)).size, 0)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            processor.bandpass_filter(np.ones(100), low=10, high=5)

    def test_feature_extractors_reject_invalid_sampling_rates(self):
        extractors = (
            "EDAFeatureExtractor",
            "TemperatureFeatureExtractor",
            "RespirationFeatureExtractor",
            "AccelerometerFeatureExtractor",
        )
        for name, rate in product(extractors, (0, -1, np.inf, np.nan, True)):
            with self.subTest(extractor=name, sampling_rate=rate):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    getattr(features, name)(sampling_rate=rate)

    def test_window_configuration_is_rejected_before_loading_data(self):
        configurations = (
            {"window_sec": 0},
            {"overlap": 1},
            {"overlap": -0.1},
            {"purity": 0},
            {"purity": np.nan},
            {"window_sec": 0.0001},
        )
        for parameters, builder in product(
            configurations, (WindowedDataset, WristDataset)
        ):
            with self.subTest(builder=builder.__name__, parameters=parameters):
                with self.assertRaises(ValueError):
                    builder(**parameters)

    def test_feature_schema_does_not_compute_dummy_signals(self):
        pipeline = FeatureExtractionPipeline()
        self.stack.enter_context(
            patch.object(
                pipeline,
                "extract_window_features",
                lambda _: self.fail("Schema discovery must not compute signals"),
            )
        )
        names = pipeline.get_feature_names()
        self.assertEqual(names, list(pipeline.get_feature_descriptions()))
        names.clear()
        self.assertEqual(pipeline.get_feature_count(), 60)
        self.assertEqual(
            list(pipeline.extract_all_features([])),
            ["subject_id", "window_id", "label", *pipeline.get_feature_names()],
        )

    def test_feature_groups_preserve_schema_order_and_disabled_groups(self):
        for enabled in (
            {},
            {"hrv_time": False, "eda": False},
            dict.fromkeys(FeatureExtractionPipeline.DEFAULT_CONFIG, False),
        ):
            with self.subTest(configuration=enabled):
                pipeline = FeatureExtractionPipeline(enabled)
                grouped = pipeline.get_feature_groups()
                self.assertEqual(
                    [name for names in grouped.values() for name in names],
                    pipeline.get_feature_names(),
                )
                for group, (_, label) in pipeline._GROUPS.items():
                    if not pipeline.feature_config[group]:
                        self.assertEqual(grouped[label], [])

    def test_synthetic_subjects_skip_unavailable_wesad_ids(self):
        tmp_path = self.tmp_path
        self.stack.enter_context(
            patch.object(synthetic, "_subject", lambda seed, block_sec: {"seed": seed})
        )
        data_path = synthetic.write_dataset(tmp_path, n_subjects=15, block_sec=1)
        self.assertEqual(WESADLoader(data_path).subjects, VALID_SUBJECTS)
        self.assertFalse((data_path / "S12").exists())
        with self.assertRaisesRegex(ValueError, "real-data caches"):
            synthetic.features(cache=True)

    def test_wesad_loader_rejects_misaligned_and_malformed_channels(self):
        tmp_path = self.tmp_path
        subject = tmp_path / "S2"
        subject.mkdir()
        record = {
            "signal": {
                "chest": {"ECG": np.zeros((700, 1))},
                "wrist": {"EDA": np.zeros((4, 1))},
            },
            "label": np.ones(700),
        }
        path = subject / "S2.pkl"
        path.write_bytes(pickle.dumps(record))
        loader = WESADLoader(tmp_path)
        self.assertEqual(len(loader.load_subject("S2")["label"]), 700)
        record["signal"]["chest"]["ECG"] = np.zeros((650, 1))
        path.write_bytes(pickle.dumps(record))
        with self.assertRaisesRegex(ValueError, "duration"):
            loader.load_subject("S2")
        record["signal"]["chest"]["ECG"] = np.zeros((700, 3))
        path.write_bytes(pickle.dumps(record))
        with self.assertRaisesRegex(ValueError, "signal shape"):
            loader.load_subject("S2")

    def test_cached_chest_features_require_current_protocol_and_aligned_rows(self):
        tmp_path = self.tmp_path
        self.stack.enter_context(patch.object(dataset, "PROCESSED_DATA_DIR", tmp_path))
        source_path = self.stack.enter_context(TemporaryDirectory())
        loader = source_loader(source_path)
        self.stack.enter_context(patch.object(dataset, "WESAD_DIR", loader.data_path))
        columns = dataset.FeatureExtractionPipeline().get_feature_names()
        frame = pd.DataFrame(
            {
                "subject_id": ["S2"],
                "window_id": [0],
                "label": [1],
                **{name: [0.0] for name in columns},
            }
        )
        frame.to_parquet(tmp_path / "features.parquet", index=False)
        np.savez(tmp_path / "raw_windows.npz", x=np.zeros((1, 5, 1024)))
        self.assertIs(dataset.load_cached(), None)
        frame.attrs.update(
            {
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "rr_correction_version": dataset.RR_CORRECTION_VERSION,
                "scr_detection_version": dataset.SCR_DETECTION_VERSION,
                "extraction_fingerprint": cache_fingerprint(
                    "chest", loader.data_path, VALID_SUBJECTS
                ),
                "dataset_parameters": {
                    "fs": 700.0,
                    "window_samples": 42000,
                    "step": 21000,
                    "purity": 0.9,
                    "cnn_length": 1024,
                    "subjects": sorted(VALID_SUBJECTS),
                },
            }
        )
        builder = dataset.WindowedDataset.__new__(dataset.WindowedDataset)
        builder.loader = loader
        builder._save(frame, np.zeros((1, 5, 1024)))
        self.assertEqual(dataset.load_cached()[1].shape, (1, 5, 1024))
        frame = pd.read_parquet(tmp_path / "features.parquet")
        frame["window_id"] = 700
        frame.to_parquet(tmp_path / "features.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            dataset.load_cached()
        frame.attrs["dataset_parameters"]["purity"] = 0.7
        frame.to_parquet(tmp_path / "features.parquet", index=False)
        self.assertIs(dataset.load_cached(), None)
