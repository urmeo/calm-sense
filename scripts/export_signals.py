import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.signal import resample

from src.config import FS, OUTPUT_DIR
from src.data.loader import WESADLoader
from src.utils import atomic_write_text

SUBJECTS = ["S2", "S3", "S4"]
CONDITIONS = {1: "Baseline", 2: "Stress", 3: "Amusement"}
SECONDS = 40
OUT_FS = 30
DASHBOARD_SIGNALS = OUTPUT_DIR / "dashboard" / "signals.ts"


def _slice(signal, labels, label, want):
    if len(signal) != len(labels):
        raise ValueError("Signal and label lengths differ")
    idx = np.flatnonzero(labels == label)
    if len(idx) == 0:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    if len(run) < want:
        return None
    start = int(run[0]) + (len(run) - want) // 2
    return signal[start : start + want]


def run():
    loader = WESADLoader()
    fs = int(FS.CHEST)
    want = SECONDS * fs
    out_n = SECONDS * OUT_FS
    data = {}

    for sid in SUBJECTS:
        d = loader.load_subject(sid)
        chest = d["chest"]
        labels = np.asarray(d["label"]).flatten()
        acc = np.asarray(chest["ACC"])
        if acc.shape != (len(labels), 3):
            raise ValueError(
                f"{sid}: chest ACC must align with labels and contain three axes"
            )

        signals = {
            "ecg": chest["ECG"].flatten(),
            "eda": chest["EDA"].flatten(),
            "temp": chest["Temp"].flatten(),
            "accX": acc[:, 0],
            "accY": acc[:, 1],
            "accZ": acc[:, 2],
        }
        channels = {name: [] for name in signals}
        conditions = []
        for label, name in CONDITIONS.items():
            clips = {
                channel: _slice(signal, labels, label, want)
                for channel, signal in signals.items()
            }
            if any(clip is None for clip in clips.values()):
                continue
            for channel, clip in clips.items():
                channels[channel].extend(np.round(resample(clip, out_n), 3).tolist())
            conditions.extend([name] * out_n)

        n = len(conditions)
        if n == 0:
            raise ValueError(f"{sid}: no condition contains a complete {SECONDS}s clip")
        data[sid] = {
            "time": np.round(np.arange(n) / OUT_FS, 3).tolist(),
            **channels,
            "conditions": conditions,
        }
        print(f"{sid}: {n} points ({n / OUT_FS:.0f}s real signal)")

    payload = json.dumps(data, allow_nan=False)
    out = DASHBOARD_SIGNALS
    atomic_write_text(out, f"const data = {payload};\n\nexport default data;\n")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    run()
