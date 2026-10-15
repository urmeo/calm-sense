import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.datasets.records import read_annotations, read_record, record_files


def word(kind, value=0):
    return struct.pack("<H", (kind << 10) | value)


def skip(interval):
    return word(59) + struct.pack("<hH", interval >> 16, interval & 65535)


class RecordTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.folder = self.directory / "records"
        self.folder.mkdir()

    def signal(self, values=((1,), (2,)), scalings=None, zero=0):
        digital = np.asarray(values, dtype="<i2")
        samples, channels = digital.shape
        scalings = scalings or ["1/mV"] * channels
        record = self.folder / "sample"
        lines = [f"sample {channels} 8 {samples}"]
        for channel, scaling in enumerate(scalings):
            checksum = int(digital[:, channel].sum(dtype=np.int64))
            checksum = (checksum + 32768) % 65536 - 32768
            lines.append(
                f"signal.dat 16 {scaling} 16 {zero} {digital[0, channel]} "
                f"{checksum} 0 lead {channel}"
            )
        record.with_suffix(".hea").write_text("\n".join(lines) + "\n")
        (self.folder / "signal.dat").write_bytes(digital.tobytes())
        return record

    def change_header(self, record, line, field, value):
        path = record.with_suffix(".hea")
        lines = path.read_text().splitlines()
        fields = lines[line].split()
        fields[field] = str(value)
        lines[line] = " ".join(fields)
        path.write_text("\n".join(lines) + "\n")

    def test_interleaving_scaling_metadata_and_invalid_sample(self):
        record = self.signal([[102, 204], [106, -32768]], ["2(100)/mV", "4(200)/C"])
        result = read_record(record)
        np.testing.assert_allclose(
            result.p_signal, [[1, 1], [3, np.nan]], equal_nan=True
        )
        self.assertEqual(result.fs, 8)
        self.assertEqual(result.units, ["mV", "C"])
        self.assertEqual(result.sig_name, ["lead 0", "lead 1"])

    def test_record_files_uses_header_reference_without_loading_samples(self):
        record = self.signal()
        data = self.folder / "signal.dat"
        data.unlink()
        self.assertEqual(record_files(record), (record.with_suffix(".hea"), data))
        self.change_header(record, 1, 0, "../outside.dat")
        with self.assertRaisesRegex(ValueError, "record directory"):
            record_files(record)

    def test_little_endian_signed_samples_and_negative_checksum(self):
        record = self.signal([[0x1234], [-0x2345]])
        self.assertEqual((self.folder / "signal.dat").read_bytes(), b"\x34\x12\xbb\xdc")
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [4660, -9029])

    def test_baselines_outside_adc_range_do_not_overflow(self):
        record = self.signal([[0, 0], [10, -10]], ["2(-100000)/C", "2(100000)/C"])
        np.testing.assert_array_equal(
            read_record(record).p_signal, [[50000, -50000], [50005, -50005]]
        )

    def test_missing_baseline_uses_adc_zero(self):
        record = self.signal([[104], [108]], ["2/mV"], zero=100)
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [2, 4])

    def test_header_comments_and_blank_lines_preserve_data(self):
        record = self.signal()
        header = record.with_suffix(".hea")
        header.write_text("\n# acquisition\n" + header.read_text() + "  # metadata\n\n")
        np.testing.assert_array_equal(read_record(record).p_signal[:, 0], [1, 2])

    def test_uncalibrated_nonpositive_and_nonfinite_gains_are_rejected(self):
        for gain in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(gain=gain):
                record = self.signal(scalings=[f"{gain}/mV"])
                with self.assertRaisesRegex(
                    ValueError, "gain must be finite and positive"
                ):
                    read_record(record)

    def test_unsupported_encoding_modifiers_resolution_and_blocks_are_rejected(self):
        cases = [(1, value) for value in ("212", "16x2", "16:1", "16+2")]
        cases += [(3, "12"), (7, "512")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                record = self.signal()
                self.change_header(record, 1, field, value)
                with self.assertRaisesRegex(
                    ValueError, "format-16|resolution|block size"
                ):
                    read_record(record)

    def test_invalid_record_dimensions_and_rates_are_rejected(self):
        for field, value in ((1, 0), (1, -1), (2, 0), (2, "nan"), (2, "inf"), (3, 0)):
            with self.subTest(field=field, value=value):
                record = self.signal()
                self.change_header(record, 0, field, value)
                with self.assertRaisesRegex(
                    ValueError, "channel count|sample count|sampling rate"
                ):
                    read_record(record)

    def test_multisegment_and_inconsistent_signal_headers_are_rejected(self):
        for fault in ("segment", "count", "files", "scaling", "empty"):
            with self.subTest(fault=fault):
                record = self.signal([[1, 2], [3, 4]])
                if fault == "segment":
                    self.change_header(record, 0, 0, "sample/2")
                elif fault == "count":
                    self.change_header(record, 0, 1, 1)
                elif fault == "files":
                    self.change_header(record, 2, 0, "other.dat")
                elif fault == "scaling":
                    self.change_header(record, 1, 2, "1(bad)/mV")
                else:
                    record.with_suffix(".hea").write_text("")
                with self.assertRaisesRegex(ValueError, "Invalid Non-EEG header"):
                    read_record(record)

    def test_corrupt_samples_fail_the_header_checksum(self):
        record = self.signal()
        (self.folder / "signal.dat").write_bytes(struct.pack("<hh", 1, 3))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            read_record(record)

    def test_truncated_or_extra_signal_bytes_are_rejected(self):
        for data in (b"\x01\x00\x02", b"\x01\x00\x02\x00\x00\x00"):
            with self.subTest(data=data):
                record = self.signal()
                (self.folder / "signal.dat").write_bytes(data)
                with self.assertRaisesRegex(ValueError, "byte count"):
                    read_record(record)

    def test_signal_paths_cannot_escape_the_record_directory(self):
        outside = self.directory / "outside.dat"
        outside.write_bytes(struct.pack("<hh", 1, 2))
        (self.folder / "linked.dat").symlink_to(outside)
        for filename in ("../outside.dat", str(outside), "linked.dat"):
            with self.subTest(filename=filename):
                record = self.signal()
                self.change_header(record, 1, 0, filename)
                with self.assertRaisesRegex(ValueError, "record directory"):
                    read_record(record)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.record = Path(directory.name) / "sample"

    def annotations(self, data, extension="atr"):
        self.record.with_suffix(f".{extension}").write_bytes(data)
        return read_annotations(self.record, extension)

    def test_long_and_signed_skips_use_pdp11_word_order(self):
        result = self.annotations(
            skip(70000) + word(22, 10) + skip(-65539) + word(22, 6) + word(0)
        )
        np.testing.assert_array_equal(result.sample, [70010, 4477])
        self.assertEqual(result.aux_note, ["", ""])

    def test_aux_odd_padding_latin1_and_modifiers_do_not_advance_time(self):
        data = word(22, 7) + word(63, 5) + b"Relax\x00"
        data += word(60, 3) + word(61, 2) + word(62, 1)
        data += word(22, 9) + word(63, 2) + b"\xe9C" + word(0)
        result = self.annotations(data)
        np.testing.assert_array_equal(result.sample, [7, 16])
        self.assertEqual(result.aux_note, ["Relax", "\xe9C"])

    def test_empty_notes_and_custom_extension(self):
        result = self.annotations(
            word(22, 0) + word(63, 0) + word(22, 8) + word(0), "labels"
        )
        np.testing.assert_array_equal(result.sample, [0, 8])
        self.assertEqual(result.aux_note, ["", ""])

    def test_empty_annotation_file_requires_a_terminator(self):
        with self.assertRaisesRegex(ValueError, "Missing annotation terminator"):
            self.annotations(b"")
        empty = self.annotations(word(0))
        self.assertEqual(empty.sample.dtype, np.dtype("int64"))
        self.assertEqual(empty.sample.size, 0)
        self.assertEqual(empty.aux_note, [])

    def test_truncated_words_skips_and_notes_are_rejected(self):
        cases = (
            (b"\x01", "Truncated annotation word"),
            (word(59) + b"\x00\x00\x01", "annotation skip"),
            (word(59, 1) + struct.pack("<i", 1), "annotation skip"),
            (word(22) + word(63, 4) + b"ab", "annotation note"),
            (word(22) + word(63, 3) + b"abc", "annotation note"),
        )
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, message):
                    self.annotations(data)

    def test_unattached_aux_and_modifiers_are_rejected(self):
        for kind in (60, 61, 62, 63):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "[Uu]nattached"):
                    self.annotations(word(kind) + word(0))

    def test_missing_terminator_unsupported_code_and_trailing_data_are_rejected(self):
        for data, message in (
            (word(22, 4), "Missing annotation terminator"),
            (word(0, 1) + word(0), "Unsupported annotation code"),
            (word(22) + word(0) + word(22), "after annotation terminator"),
        ):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, message):
                    self.annotations(data)

    def test_nonzero_odd_note_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            self.annotations(word(22) + word(63, 3) + b"abc!" + word(0))
