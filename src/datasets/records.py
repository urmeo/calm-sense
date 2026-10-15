"""Read Non-EEG format-16 signals and MIT annotations."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Record:
    fs: float
    p_signal: np.ndarray
    units: list[str]
    sig_name: list[str]


@dataclass(frozen=True)
class Annotations:
    sample: np.ndarray
    aux_note: list[str]


def _read_header(record: str | Path):
    header = Path(f"{record}.hea")
    lines = [
        line.split()
        for line in header.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    try:
        name, channels, rate, samples = lines[0][:4]
        if "/" in name:
            raise ValueError("Multi-segment records are unsupported")
        channels, samples, rate = int(channels), int(samples), float(rate)
        if channels < 1 or samples < 1 or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid channel count, sample count, or sampling rate")
        if len(lines) != channels + 1:
            raise ValueError("Signal count does not match the header")
        specs = lines[1:]
        if any(len(spec) < 8 or spec[1] != "16" for spec in specs):
            raise ValueError("Only unskewed format-16 signals are supported")
        if len({spec[0] for spec in specs}) != 1:
            raise ValueError("Signals must share one interleaved data file")
        data_path = (header.parent / specs[0][0]).resolve()
        if data_path.parent != header.parent.resolve():
            raise ValueError("Signal file must remain in the record directory")
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Non-EEG header {header}: {error}") from error
    return header, data_path, specs, rate, samples, channels


def record_files(record: str | Path) -> tuple[Path, Path]:
    """Return the header and its referenced signal file without loading samples."""
    header, data_path, *_ = _read_header(record)
    return header, data_path


def read_record(record: str | Path) -> Record:
    header, data_path, specs, rate, samples, channels = _read_header(record)
    try:
        gains, baselines, units, names, checksums = [], [], [], [], []
        for spec in specs:
            scaling = re.fullmatch(r"([^(/]+)(?:\((-?\d+)\))?/([^\s]+)", spec[2])
            if scaling is None:
                raise ValueError("Signal gain and units are required")
            gain = float(scaling[1])
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError("Signal gain must be finite and positive")
            if int(spec[3]) != 16 or int(spec[7]) != 0:
                raise ValueError("Unsupported ADC resolution or block size")
            gains.append(gain)
            baselines.append(
                int(scaling[2]) if scaling[2] is not None else int(spec[4])
            )
            units.append(scaling[3])
            names.append(" ".join(spec[8:]))
            checksums.append(int(spec[6]) % 65536)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Non-EEG header {header}: {error}") from error
    if data_path.stat().st_size != samples * channels * 2:
        raise ValueError(f"Signal byte count does not match the header: {data_path}")
    digital = np.fromfile(data_path, dtype="<i2").reshape(samples, channels)
    if not np.array_equal(digital.sum(axis=0, dtype=np.int64) % 65536, checksums):
        raise ValueError(f"Signal checksum mismatch: {data_path}")
    physical = (digital.astype(np.float64) - baselines) / gains
    physical[digital == -32768] = np.nan
    return Record(rate, physical, units, names)


def read_annotations(record: str | Path, extension: str = "atr") -> Annotations:
    data = Path(f"{record}.{extension}").read_bytes()
    samples: list[int] = []
    notes: list[str] = []
    offset = time = 0
    terminated = False
    while offset < len(data):
        if offset + 2 > len(data):
            raise ValueError("Truncated annotation word")
        word = int.from_bytes(data[offset : offset + 2], "little")
        offset += 2
        kind, value = word >> 10, word & 1023
        if word == 0:
            terminated = True
            if any(data[offset:]):
                raise ValueError("Unexpected data after annotation terminator")
            break
        if kind == 59:
            if value != 0 or offset + 4 > len(data):
                raise ValueError("Truncated or invalid annotation skip")
            high = int.from_bytes(data[offset : offset + 2], "little", signed=True)
            low = int.from_bytes(data[offset + 2 : offset + 4], "little")
            time += (high << 16) + low
            offset += 4
        elif kind == 63:
            length = value + value % 2
            if not notes or offset + length > len(data):
                raise ValueError("Truncated or unattached annotation note")
            if value % 2 and data[offset + value] != 0:
                raise ValueError("Annotation note padding must be NUL")
            notes[-1] = data[offset : offset + value].decode("latin-1")
            offset += length
        elif kind in (60, 61, 62):
            if not samples:
                raise ValueError("Unattached annotation modifier")
        elif 0 < kind < 59:
            time += value
            samples.append(time)
            notes.append("")
        else:
            raise ValueError("Unsupported annotation code")
    if not terminated:
        raise ValueError("Missing annotation terminator")
    return Annotations(np.asarray(samples, dtype=np.int64), notes)
